import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix
from typing import Mapping, List, Tuple, Any, Union, Callable
from bertopic.representation._base import BaseRepresentation
from bertopic.representation._utils import truncate_document, validate_truncate_document_parameters
import logging
import sys
import os
from pathlib import Path

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
            direct_load: bool = False,  # Keep original parameter name for compatibility
            always_direct_load: bool = False,  # Add new parameter but don't use it directly
    ):
        self.model_name = model_name
        self.prompt = prompt if prompt is not None else DEFAULT_PROMPT
        self.system_prompt = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT
        self.default_prompt_ = DEFAULT_PROMPT
        self.default_system_prompt_ = DEFAULT_SYSTEM_PROMPT
        self.generation_kwargs = generation_kwargs
        self.nr_docs = nr_docs
        self.diversity = diversity
        self.doc_length = doc_length
        self.tokenizer = tokenizer
        # Use direct_load for backward compatibility
        self.always_direct_load = direct_load or always_direct_load
        validate_truncate_document_parameters(self.tokenizer, self.doc_length)

        # Track generated prompts for analysis
        self.prompts_ = []

        # Service and model state
        self.service = None
        self.direct_model = None
        self.using_service = False

        # Initialize (but don't load models yet - defer until needed)
        self._initialize()

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
            quantization = "fp8"  # Default
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
        """Extract topic representations and return a single label for each topic.

        Arguments:
            topic_model: A BERTopic model
            documents: Not used
            c_tf_idf: Not used
            topics: The candidate topics as calculated with c-TF-IDF

        Returns:
            updated_topics: Updated topic representations
        """
        # Extract the top representative documents per topic
        repr_docs_mappings, _, _, _ = topic_model._extract_representative_docs(
            c_tf_idf, documents, topics, 500, self.nr_docs, self.diversity
        )

        # Ensure a model is available for generation
        if not self._ensure_model_available():
            logger.error("Cannot proceed with topic extraction - no Aphrodite model available")
            # Return empty topics as fallback
            updated_topics = {}
            for topic in topics:
                updated_topics[topic] = [("", 0) for _ in range(10)]
            return updated_topics

        # Process all topics at once
        all_topics = []
        all_prompts = []
        all_chat_prompts = []

        logger.info(f"Preparing prompts for {len(repr_docs_mappings)} topics")

        # Create all prompts first
        for topic, docs in tqdm(repr_docs_mappings.items(), disable=not topic_model.verbose):
            # Prepare prompt
            truncated_docs = [truncate_document(topic_model, self.doc_length, self.tokenizer, doc) for doc in docs]
            prompt = self._create_prompt(truncated_docs, topic, topics)
            self.prompts_.append(prompt)

            # Format as ChatML
            chat_prompt = f"""<|im_start|>system
{self.system_prompt}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""

            # Store the topic, original prompt, and chat prompt
            all_topics.append(topic)
            all_prompts.append(prompt)
            all_chat_prompts.append(chat_prompt)

        # Generate all labels at once
        logger.info(f"Generating labels for {len(all_chat_prompts)} topics at once")
        all_labels = self._generate_topic_labels_batch(all_chat_prompts)

        # Create the results dictionary
        updated_topics = {}
        for topic, label in zip(all_topics, all_labels):
            updated_topics[topic] = [(label, 1)] + [("", 0) for _ in range(9)]

        return updated_topics

    def _generate_topic_labels_batch(self, chat_prompts):
        """Generate topic labels for all prompts at once.

        Args:
            chat_prompts: List of ChatML-formatted prompts

        Returns:
            List of generated labels
        """
        try:
            # Use service if available and we've confirmed it has a model loaded
            if self.using_service and APHRODITE_SERVICE_AVAILABLE and self.service and self.service.is_running():
                logger.info(f"Sending batch of {len(chat_prompts)} prompts to Aphrodite service")

                # Check if extract_entities can be used (it's designed for batch processing)
                try:
                    # The extract_entities method in the service can handle batches
                    response = self.service.extract_entities(prompts=chat_prompts)

                    if response.get("status") == "success":
                        # Extract the results
                        results = response.get("results", [])
                        logger.info(f"Received {len(results)} results from service")

                        # Validate results count
                        if len(results) != len(chat_prompts):
                            logger.warning(f"Expected {len(chat_prompts)} results but got {len(results)}")
                            # Pad with error messages if needed
                            while len(results) < len(chat_prompts):
                                results.append("Topic Extraction Error")

                        return results
                    else:
                        error_msg = response.get("error", "Unknown error")
                        logger.error(f"Error from Aphrodite service batch processing: {error_msg}")

                        # Try with direct model as fallback if available
                        if self.direct_model is not None:
                            logger.info("Falling back to direct model after service error")
                            self.using_service = False
                            return self._generate_with_direct_model_batch(chat_prompts)

                        # Return error messages for all prompts
                        return ["Topic Extraction Error" for _ in chat_prompts]

                except Exception as batch_error:
                    logger.error(f"Error using extract_entities for batch processing: {batch_error}")
                    logger.info("Falling back to sequential processing")

                    # Fall back to processing one by one
                    results = []
                    for i, chat_prompt in enumerate(chat_prompts):
                        logger.info(f"Processing prompt {i + 1}/{len(chat_prompts)}")
                        response = self.service.generate_chat(prompt=chat_prompt)

                        if response.get("status") == "success":
                            results.append(response.get("result", "").strip())
                        else:
                            results.append("Topic Extraction Error")

                    return results

            # Use direct model
            elif self.direct_model is not None:
                return self._generate_with_direct_model_batch(chat_prompts)

            else:
                logger.error("No model available for topic label generation")
                return ["Topic Extraction Error" for _ in chat_prompts]

        except Exception as e:
            logger.error(f"Error generating topic labels: {e}")
            return ["Topic Extraction Error" for _ in chat_prompts]

    def _generate_topic_label(self, prompt):
        """Generate a single topic label (for backward compatibility).

        This method is kept for backward compatibility but internally
        calls the batch method with a single prompt.
        """
        # Format the prompt in ChatML format
        chat_prompt = f"""<|im_start|>system
{self.system_prompt}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""

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

    def _create_prompt(self, docs, topic, topics):
        """Create a prompt for the topic using keywords and documents."""
        keywords = list(zip(*topics[topic]))[0]

        # Use the Default Chat Prompt
        if self.prompt == DEFAULT_PROMPT:
            prompt = self.prompt.replace("[KEYWORDS]", ", ".join(keywords))
            prompt = self._replace_documents(prompt, docs)

        # Use a custom prompt that leverages keywords, documents or both using
        # custom tags, namely [KEYWORDS] and [DOCUMENTS] respectively
        else:
            prompt = self.prompt
            if "[KEYWORDS]" in prompt:
                prompt = prompt.replace("[KEYWORDS]", ", ".join(keywords))
            if "[DOCUMENTS]" in prompt:
                prompt = self._replace_documents(prompt, docs)

        return prompt

    @staticmethod
    def _replace_documents(prompt, docs):
        """Replace the [DOCUMENTS] tag with actual document text."""
        to_replace = ""
        for doc in docs:
            to_replace += f"- {doc}\n"
        prompt = prompt.replace("[DOCUMENTS]", to_replace)
        return prompt