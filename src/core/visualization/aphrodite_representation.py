import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix
from typing import Mapping, List, Tuple, Any, Union, Callable, Dict
from bertopic.representation._base import BaseRepresentation
from bertopic.representation._utils import truncate_document, validate_truncate_document_parameters
import logging
import sys
import os
from pathlib import Path
from transformers import AutoTokenizer
import traceback # For better error logging

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))

# Import Aphrodite service utilities
try:
    from src.utils.aphrodite_service import get_service, AphroditeService

    APHRODITE_SERVICE_AVAILABLE = True
except ImportError:
    logging.warning("Aphrodite service not available. Will attempt to use direct model loading.")
    APHRODITE_SERVICE_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aphrodite_representation")

# Default prompts - same as in LlamaCPP implementation
DEFAULT_PROMPT = """
This is a list of texts where each collection of texts describe a topic. After each collection of texts, the name of the topic they represent is mentioned as a short-highly-descriptive title
---
Topic:
Sample texts from this topic:
- Traditional diets in most cultures were primarily plant-based with a little meat on top, but with the rise of industrial style meat production and factory farming, meat has become a staple food.
- Meat, but especially beef, is the word food in terms of emissions.
- Eating meat doesn't make you a bad person, not eating meat doesn't make you a good one.

Keywords: meat beef eat eating emissions steak food health processed chicken
Topic name: Environmental impacts of eating meat
---
Topic:
Sample texts from this topic:
- I have ordered the product weeks ago but it still has not arrived!
- The website mentions that it only takes a couple of days to deliver but I still have not received mine.
- I got a message stating that I received the monitor but that is not true!
- It took a month longer to deliver than was advised...

Keywords: deliver weeks product shipping long delivery received arrived arrive week
Topic name: Shipping and delivery issues
---
Topic:
Sample texts from this topic:
[DOCUMENTS]
Keywords: [KEYWORDS]
Topic name:"""

DEFAULT_SYSTEM_PROMPT = "You are an assistant that extracts high-level topics from texts."


class AphroditeRepresentation(BaseRepresentation):
    """An Aphrodite implementation to use as a representation model for BERTopic.

    This class connects to the Aphrodite service to generate topic labels.
    If the service is running and has a model loaded, it uses that model.
    If the service is running but no model is loaded, it tries to load the specified model.
    If the service is not running or loading a model fails, it falls back to direct loading.

    Arguments:
        model_name: Name of the model to use (if service doesn't have one loaded or for direct loading)
        prompt: The prompt to be used in the model. If no prompt is given,
                `self.default_prompt_` is used instead.
                NOTE: Use `"[KEYWORDS]"` and `"[DOCUMENTS]"` in the prompt
                to decide where the keywords and documents need to be
                inserted.
        system_prompt: The system prompt to be used in the model. If no system prompt is given,
                       `self.default_system_prompt_` is used instead.
        generation_kwargs: Kwargs for text generation such as `max_tokens`
        nr_docs: The number of documents to pass to the LLM if a prompt
                 with the `["DOCUMENTS"]` tag is used.
        diversity: The diversity of documents to pass to the LLM.
                   Accepts values between 0 and 1. A higher
                   values results in passing more diverse documents
                   whereas lower values passes more similar documents.
        doc_length: The maximum length of each document. If a document is longer,
                    it will be truncated. If None, the entire document is passed.
        tokenizer: The tokenizer used to calculate to split the document into segments
                   used to count the length of a document.
                       * If tokenizer is 'char', then the document is split up
                         into characters which are counted to adhere to `doc_length`
                       * If tokenizer is 'whitespace', the the document is split up
                         into words separated by whitespaces. These words are counted
                         and truncated depending on `doc_length`
                       * If tokenizer is 'vectorizer', then the internal CountVectorizer
                         is used to tokenize the document. These tokens are counted
                         and truncated depending on `doc_length`
                       * If tokenizer is a callable, then that callable is used to tokenize
                         the document. These tokens are counted and truncated depending
                         on `doc_length`
        always_direct_load: If True, always uses direct loading regardless of service state

    Usage:

    ```python
    from bertopic import BERTopic
    from aphrodite_representation import AphroditeRepresentation

    # Use Aphrodite service if available, otherwise load directly
    representation_model = AphroditeRepresentation(model_name="Qwen/Qwen2.5-7B-Instruct")

    # Create our BERTopic model
    topic_model = BERTopic(representation_model=representation_model, verbose=True)
    ```
    """

    def __init__(
            self,
            model_name: str = "Qwen/Qwen2.5-7B-Instruct",  # Default model if needed
            prompt: str = None,
            system_prompt: str = None,
            generation_kwargs: Mapping[str, Any] = {},
            nr_docs: int = 4,
            diversity: float = None,
            doc_length: int = None,
            tokenizer: Union[str, Callable] = None,
            # Keep tokenizer param for doc length, but load HF tokenizer separately
            direct_load: bool = False,
            always_direct_load: bool = False,
    ):
        self.model_name = model_name
        self.prompt_template = prompt if prompt is not None else DEFAULT_PROMPT  # Store the template
        self.system_prompt = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT
        self.default_prompt_ = DEFAULT_PROMPT
        self.default_system_prompt_ = DEFAULT_SYSTEM_PROMPT
        self.generation_kwargs = generation_kwargs
        self.nr_docs = nr_docs
        self.diversity = diversity
        self.doc_length = doc_length
        self.doc_length_tokenizer = tokenizer  # Keep the original tokenizer param specifically for doc length calculation
        self.always_direct_load = direct_load or always_direct_load
        validate_truncate_document_parameters(self.doc_length_tokenizer, self.doc_length)

        self.prompts_ = []
        self.service = None
        self.direct_model = None
        self.using_service = False

        # --- NEW: Load HF Tokenizer for Chat Templating ---
        self.hf_tokenizer = None
        try:
            logger.info(f"[AphroditeRepresentation] Loading HF tokenizer for: {self.model_name}")
            self.hf_tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            if self.hf_tokenizer.chat_template is None:
                logger.warning(
                    f"[AphroditeRepresentation] Tokenizer for {self.model_name} has no chat_template. Manual formatting may be needed or fail.")
            else:
                logger.info(f"[AphroditeRepresentation] HF tokenizer loaded successfully with chat template.")
        except Exception as e:
            logger.error(f"[AphroditeRepresentation] Failed to load HF tokenizer for {self.model_name}: {e}",
                         exc_info=True)
            # Proceeding without a tokenizer will cause errors later

        self._initialize()  # Keep initialization logic

    def _initialize(self):
        """Initialize the representation model by checking service status."""
        # Skip service check if direct_load is True
        if self.always_direct_load:
            logger.info("direct_load=True, will load model directly")
            self._load_direct_model()
            return

        # Check if Aphrodite service is available
        if APHRODITE_SERVICE_AVAILABLE:
            try:
                self.service = get_service()

                # Check if service is running
                if self.service.is_running():
                    logger.info("Found running Aphrodite service")

                    # Check if a model is loaded
                    status = self.service.get_status()
                    if status.get("model_loaded", False):
                        logger.info(f"Using existing loaded model: {status.get('current_model')}")
                        self.using_service = True
                    else:
                        logger.info("Service running but no model loaded")
                        # Will load model when needed
                else:
                    logger.info("Aphrodite service not running, will load model directly")
                    self._load_direct_model()
            except Exception as e:
                logger.error(f"Error connecting to Aphrodite service: {e}")
                logger.info("Falling back to direct model loading")
                self._load_direct_model()
        else:
            # No service available, use direct loading
            logger.warning("Aphrodite service not available, using direct model loading")
            self._load_direct_model()

    def _ensure_model_available(self):
        """Ensure that a model is available for generation, either through service or direct loading."""
        if self.always_direct_load and self.direct_model is not None:
            # Already using direct model
            return True

        if APHRODITE_SERVICE_AVAILABLE and self.service and self.service.is_running():
            # Check if model is loaded in service
            status = self.service.get_status()
            if status.get("model_loaded", False):
                logger.info(f"Using Aphrodite service with model: {status.get('current_model')}")
                self.using_service = True
                return True

            # Try to load model in service
            logger.info(f"Attempting to load model {self.model_name} in service")
            if self.service.load_model(self.model_name):
                logger.info(f"Successfully loaded model {self.model_name} in service")
                self.using_service = True
                return True

            # Service model loading failed, try direct loading
            logger.warning(f"Failed to load model {self.model_name} in service, falling back to direct loading")
        else:
            # Service not running/available, use direct loading
            logger.info("Service not available/running, using direct loading")

        # Try direct loading if we don't have a direct model yet
        if self.direct_model is None:
            self._load_direct_model()

        return self.direct_model is not None

    def _load_direct_model(self):
        """Attempt to load an Aphrodite model directly (not via service)."""
        try:
            from aphrodite import LLM

            logger.info(f"Loading Aphrodite model directly: {self.model_name}")

            # Get quantization setting from config if possible
            try:
                import yaml
                config_path = ROOT_DIR / "config.yaml"
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                quantization = config.get("aphrodite", {}).get("quantization", "fp8")
                max_model_len = config.get("aphrodite", {}).get("max_model_len", 4096)
            except Exception as e:
                logger.warning(f"Error loading config, using default values: {e}")
                max_model_len = 8196

            # Load the model
            self.direct_model = LLM(
                model=self.model_name,
                max_model_len=8196,
                quantization="fp5",
                dtype="bfloat16",  # Reasonable default
                enforce_eager=True,  # Ensure deterministic results
                trust_remote_code=True
            )
            logger.info(f"Successfully loaded model: {self.model_name}")
            self.using_service = False
            return True
        except ImportError:
            logger.error("Failed to import Aphrodite. Make sure it's installed.")
            self.direct_model = None
            return False
        except Exception as e:
            logger.error(f"Error loading Aphrodite model directly: {e}")
            self.direct_model = None
            return False

    def extract_topics(
            self,
            topic_model,
            documents: pd.DataFrame,
            c_tf_idf: csr_matrix,
            topics: Mapping[str, List[Tuple[str, float]]],
    ) -> Mapping[str, List[Tuple[str, float]]]:
        """Extract topic representations and return a single label for each topic."""
        repr_docs_mappings, _, _, _ = topic_model._extract_representative_docs(
            c_tf_idf, documents, topics, 500, self.nr_docs, self.diversity
        )

        if not self._ensure_model_available():
            logger.error("[AphroditeRepresentation] Cannot proceed - no Aphrodite model available")
            updated_topics = {topic: [("", 0)] * 10 for topic in topics}
            return updated_topics

        if not self.hf_tokenizer or self.hf_tokenizer.chat_template is None:
             logger.error("[AphroditeRepresentation] HF Tokenizer or chat template not available. Cannot format prompts.")
             updated_topics = {topic: [("", 0)] * 10 for topic in topics}
             return updated_topics

        all_topics = []
        all_formatted_prompts = [] # Store the final strings for the service/model
        self.prompts_ = [] # Store the structured messages for potential debugging

        logger.info(f"[AphroditeRepresentation] Preparing and formatting prompts for {len(repr_docs_mappings)} topics")

        for topic, docs in tqdm(repr_docs_mappings.items(), disable=not topic_model.verbose):
            # 1. Truncate documents (using the BERTopic tokenizer setting)
            truncated_docs = [truncate_document(topic_model, self.doc_length, self.doc_length_tokenizer, doc) for doc in docs]

            # 2. Create structured messages
            messages = self._create_prompt(truncated_docs, topic, topics)
            self.prompts_.append(messages) # Store the structured version

            # 3. Apply chat template using the HF tokenizer
            try:
                final_prompt_string = self.hf_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True # Add assistant prompt marker
                )
                all_topics.append(topic)
                all_formatted_prompts.append(final_prompt_string)
            except Exception as template_err:
                logger.error(f"[AphroditeRepresentation] Error applying chat template for topic {topic}: {template_err}", exc_info=True)
                # Optionally add a placeholder error prompt or skip this topic
                # Skipping for now:
                # all_topics.append(topic)
                # all_formatted_prompts.append("Error: Could not format prompt")


        if not all_formatted_prompts:
             logger.warning("[AphroditeRepresentation] No prompts were successfully formatted.")
             updated_topics = {topic: [("", 0)] * 10 for topic in topics}
             return updated_topics

        logger.info(f"[AphroditeRepresentation] Generating labels for {len(all_formatted_prompts)} topics at once")
        # Pass the already formatted strings to the generation method
        all_labels = self._generate_topic_labels_batch(all_formatted_prompts)

        updated_topics = {}
        label_idx = 0
        # Iterate through the topics we *attempted* to format
        for topic in repr_docs_mappings.keys():
             # Check if this topic's prompt was successfully formatted and sent
             if topic in all_topics:
                 if label_idx < len(all_labels):
                      label = all_labels[label_idx]
                      updated_topics[topic] = [(label, 1)] + [("", 0)] * 9
                      label_idx += 1
                 else:
                      # Mismatch, should not happen if logic is correct
                      logger.error(f"Label index out of bounds for topic {topic}")
                      updated_topics[topic] = [("Error", 1)] + [("", 0)] * 9
             else:
                 # Prompt formatting failed for this topic
                 logger.warning(f"Skipping topic {topic} due to earlier formatting error.")
                 updated_topics[topic] = [("Formatting Error", 1)] + [("", 0)] * 9


        return updated_topics

    def _generate_topic_labels_batch(self, formatted_prompts: List[str]):
        """Generate topic labels for all pre-formatted prompts at once.

        Args:
            formatted_prompts: List of fully formatted prompt strings ready for the LLM.

        Returns:
            List of generated labels
        """
        try:
            # Use service if available
            if self.using_service and APHRODITE_SERVICE_AVAILABLE and self.service and self.service.is_running():
                logger.info(f"[AphroditeRepresentation] Sending batch of {len(formatted_prompts)} prompts to Aphrodite service")
                try:
                    # Use extract_entities as it seems to handle batching in the service
                    response = self.service.extract_entities(prompts=formatted_prompts)
                    if response.get("status") == "success":
                        results = response.get("results", [])
                        logger.info(f"[AphroditeRepresentation] Received {len(results)} results from service")
                        if len(results) != len(formatted_prompts):
                            logger.warning(f"Expected {len(formatted_prompts)} results but got {len(results)}")
                            while len(results) < len(formatted_prompts): results.append("Topic Extraction Error")
                        # Clean up results (remove potential extra quotes, newlines)
                        cleaned_results = [str(res).strip().strip('"').strip("'").split('\n')[0] for res in results]
                        return cleaned_results
                    else:
                        error_msg = response.get("error", "Unknown error")
                        logger.error(f"Error from Aphrodite service batch processing: {error_msg}")
                        if self.direct_model is not None:
                            logger.info("Falling back to direct model after service error")
                            self.using_service = False
                            return self._generate_with_direct_model_batch(formatted_prompts)
                        return ["Topic Extraction Error" for _ in formatted_prompts]
                except Exception as batch_error:
                    logger.error(f"Error using extract_entities for batch processing: {batch_error}")
                    logger.info("Falling back to sequential processing via generate_chat")
                    results = []
                    for i, prompt_str in enumerate(formatted_prompts):
                        logger.info(f"Processing prompt {i + 1}/{len(formatted_prompts)} sequentially")
                        response = self.service.generate_chat(prompt=prompt_str) # generate_chat takes the formatted string
                        if response.get("status") == "success":
                            label = response.get("result", "").strip().strip('"').strip("'").split('\n')[0]
                            results.append(label)
                        else:
                            results.append("Topic Extraction Error")
                    return results

            # Use direct model
            elif self.direct_model is not None:
                return self._generate_with_direct_model_batch(formatted_prompts)
            else:
                logger.error("[AphroditeRepresentation] No model available for topic label generation")
                return ["Topic Extraction Error" for _ in formatted_prompts]

        except Exception as e:
            logger.error(f"[AphroditeRepresentation] Error generating topic labels: {e}", exc_info=True)
            return ["Topic Extraction Error" for _ in formatted_prompts]

    def _generate_topic_label(self, prompt: str) -> str:
        """
        Generate a single topic label using the correct chat templating.
        (Updated for consistency with dynamic templating).

        Args:
            prompt: The user content part of the prompt (e.g., the part containing keywords/docs).

        Returns:
            The generated topic label string or an error message.
        """
        if not self.hf_tokenizer or self.hf_tokenizer.chat_template is None:
            logger.error("[AphroditeRepresentation] HF Tokenizer or chat template not available in _generate_topic_label.")
            return "Topic Extraction Error: Tokenizer Missing"

        # 1. Create structured messages
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt} # Assume input 'prompt' is the user content
        ]

        # 2. Apply chat template
        try:
            final_prompt_string = self.hf_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True # Add assistant prompt marker
            )
        except Exception as template_err:
            logger.error(f"[AphroditeRepresentation] Error applying chat template in _generate_topic_label: {template_err}", exc_info=True)
            return "Topic Extraction Error: Template Failed"

        # 3. Call the batch method with the single *correctly formatted* prompt
        results = self._generate_topic_labels_batch([final_prompt_string])

        # 4. Return the result
        return results[0] if results else "Topic Extraction Error: Generation Failed"

        # Call the batch method with a single prompt
        results = self._generate_topic_labels_batch([chat_prompt])

        # Return the first (and only) result
        return results[0] if results else "Topic Extraction Error"

    def _generate_with_direct_model_batch(self, chat_prompts):
        """Generate text for multiple prompts using the direct model."""
        try:
            from aphrodite import SamplingParams

            # Create sampling parameters
            params = SamplingParams(
                temperature=self.generation_kwargs.get("temperature", 0.7),
                max_tokens=self.generation_kwargs.get("max_tokens", 64),
                top_p=self.generation_kwargs.get("top_p", 0.9),
                **{k: v for k, v in self.generation_kwargs.items()
                   if k not in ["temperature", "max_tokens", "top_p"]}
            )

            logger.info(f"Generating {len(chat_prompts)} responses directly with Aphrodite model")

            # Generate using the model (Aphrodite's generate method already supports batching)
            outputs = self.direct_model.generate(
                prompts=chat_prompts,
                sampling_params=params
            )

            # Extract results
            results = []
            for i, output in enumerate(outputs):
                if output.outputs:
                    results.append(output.outputs[0].text.strip())
                else:
                    logger.error(f"No output generated for prompt {i}")
                    results.append("Topic Extraction Error")

            return results
        except Exception as e:
            logger.error(f"Error generating with direct model: {e}")
            return ["Topic Extraction Error" for _ in chat_prompts]

    def _generate_with_direct_model(self, chat_prompt):
        """Generate text for a single prompt (for backward compatibility)."""
        results = self._generate_with_direct_model_batch([chat_prompt])
        return results[0] if results else "Topic Extraction Error"

    def _create_prompt(self, docs, topic, topics) -> List[Dict[str, str]]:
        """
        Creates the structured message list for the topic extraction task.

        Args:
            docs: List of representative documents for the topic.
            topic: The topic ID.
            topics: Dictionary mapping topic IDs to lists of (keyword, score) tuples.

        Returns:
            A list of dictionaries representing the conversation structure.
        """
        keywords = list(zip(*topics[topic]))[0]
        keyword_str = ", ".join(keywords)

        # Prepare document string
        doc_str = ""
        for doc in docs:
            doc_str += f"- {doc}\n"

        # Populate the user prompt using the stored template
        user_prompt = self.prompt_template # Use the template stored in __init__
        if "[KEYWORDS]" in user_prompt:
            user_prompt = user_prompt.replace("[KEYWORDS]", keyword_str)
        if "[DOCUMENTS]" in user_prompt:
            user_prompt = user_prompt.replace("[DOCUMENTS]", doc_str)

        # Return structured messages
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return messages

    @staticmethod
    def _replace_documents(prompt, docs):
        """Replace the [DOCUMENTS] tag with actual document text."""
        to_replace = ""
        for doc in docs:
            to_replace += f"- {doc}\n"
        prompt = prompt.replace("[DOCUMENTS]", to_replace)
        return prompt