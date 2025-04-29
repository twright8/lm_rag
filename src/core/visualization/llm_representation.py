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
import traceback # For better error logging

# Add project root to path
# Assuming this file is in src/core/visualization/
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))

# Import backend managers conditionally based on config
from src.ui.app_setup import (
    get_active_llm_manager, IS_OPENROUTER_ACTIVE, IS_GEMINI_ACTIVE, LLM_BACKEND, logger, CONFIG
)

# Import specific manager classes conditionally
if IS_OPENROUTER_ACTIVE:
    from src.utils.openrouter_manager import OpenRouterManager
    AphroditeService = None
    GeminiManager = None
    logger.info("BERTopic: OpenRouter backend active for topic labeling.")
elif IS_GEMINI_ACTIVE:
    from src.utils.gemini_manager import GeminiManager
    AphroditeService = None
    OpenRouterManager = None
    logger.info("BERTopic: Gemini backend active for topic labeling.")
else: # Aphrodite
    try:
        from src.utils.aphrodite_service import AphroditeService
        APHRODITE_SERVICE_AVAILABLE = True
        logger.info("BERTopic: Aphrodite backend active for topic labeling.")
    except ImportError:
        APHRODITE_SERVICE_AVAILABLE = False
        AphroditeService = None
        logger.warning("BERTopic: Aphrodite backend selected but service not available.")
    OpenRouterManager = None
    GeminiManager = None

# Import tokenizer (needed for Aphrodite prompt templating)
if LLM_BACKEND == "aphrodite":
    try:
        from transformers import AutoTokenizer
    except ImportError:
        AutoTokenizer = None
        logger.error("BERTopic: Transformers AutoTokenizer not found. Cannot format prompts for Aphrodite.")
else:
    AutoTokenizer = None # Not needed for API backends

# Set up logging
logging.basicConfig(level=logging.INFO)
# Use logger from app_setup
# logger = logging.getLogger("bertopic_representation") # Replaced with logger from app_setup

# Default prompts - same as before
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


class LLMRepresentation(BaseRepresentation): # Renamed class
    """
    Uses the active LLM backend (Aphrodite, OpenRouter, or Gemini) as a representation model for BERTopic.

    Arguments:
        model_name: Name of the model to use (passed to the active backend).
        prompt: The prompt template. Use `"[KEYWORDS]"` and `"[DOCUMENTS]"`.
        system_prompt: The system prompt.
        generation_kwargs: Kwargs for text generation (temperature, max_tokens).
        nr_docs: The number of documents to include in the prompt.
        diversity: Diversity of documents to include (0 to 1).
        doc_length: Max length of each document (truncated if longer).
        tokenizer: Tokenizer for doc_length calculation ('char', 'whitespace', 'vectorizer', callable).
    """

    def __init__(
            self,
            model_name: str = None, # Model name determined by backend config if None
            prompt: str = None,
            system_prompt: str = None,
            generation_kwargs: Mapping[str, Any] = {},
            nr_docs: int = 4,
            diversity: float = None,
            doc_length: int = None,
            tokenizer: Union[str, Callable] = None,
    ):
        self.llm_manager = get_active_llm_manager()
        self.is_openrouter = isinstance(self.llm_manager, OpenRouterManager) if OpenRouterManager else False
        self.is_aphrodite = isinstance(self.llm_manager, AphroditeService) if AphroditeService else False
        self.is_gemini = isinstance(self.llm_manager, GeminiManager) if GeminiManager else False # Add Gemini flag

        # Determine model name based on backend and override
        if model_name:
            self.model_name = model_name
        elif self.is_openrouter:
            self.model_name = self.llm_manager.models.get("topic_labeling", "mistralai/mistral-7b-instruct:free")
        elif self.is_gemini: # Add Gemini case
            self.model_name = self.llm_manager.models.get("topic_labeling", "gemini-1.5-flash-latest")
        elif self.is_aphrodite:
            # Use a standard model for Aphrodite topic labeling
            self.model_name = CONFIG["models"]["extraction_models"]["text_standard"]
        else:
            self.model_name = "default/unknown"
            logger.error("BERTopic Representation: Initialized without a valid LLM manager!")

        self.prompt_template = prompt if prompt is not None else DEFAULT_PROMPT
        self.system_prompt = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT
        self.generation_kwargs = generation_kwargs
        self.nr_docs = nr_docs
        self.diversity = diversity
        self.doc_length = doc_length
        self.doc_length_tokenizer = tokenizer
        validate_truncate_document_parameters(self.doc_length_tokenizer, self.doc_length)

        self.prompts_ = [] # Stores structured messages for debugging

        # --- Load HF Tokenizer (Only needed for Aphrodite prompt formatting) ---
        self.hf_tokenizer = None
        if self.is_aphrodite and AutoTokenizer:
            try:
                # Use the determined model name for the tokenizer
                logger.info(f"[BERTopic Representation] Loading HF tokenizer for Aphrodite: {self.model_name}")
                self.hf_tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
                if self.hf_tokenizer.chat_template is None:
                    logger.warning(f"[BERTopic Representation] Tokenizer for {self.model_name} has no chat_template. Manual formatting may fail.")
                else:
                    logger.info(f"[BERTopic Representation] HF tokenizer loaded successfully.")
            except Exception as e:
                logger.error(f"[BERTopic Representation] Failed to load HF tokenizer for {self.model_name}: {e}", exc_info=True)
        elif self.is_aphrodite:
            logger.error("[BERTopic Representation] AutoTokenizer not available. Cannot format prompts for Aphrodite.")

        logger.info(f"BERTopic Representation initialized with backend: {LLM_BACKEND.upper()}, model: {self.model_name}")


    def extract_topics(
            self,
            topic_model,
            documents: pd.DataFrame,
            c_tf_idf: csr_matrix,
            topics: Mapping[str, List[Tuple[str, float]]],
    ) -> Mapping[str, List[Tuple[str, float]]]:
        """Extract topic representations using the active LLM backend."""

        # Check if manager is ready
        if not self.llm_manager:
            logger.error("[BERTopic Representation] LLM Manager not available. Cannot extract topics.")
            return {topic: [("Error: No LLM", 1)] + [("", 0)] * 9 for topic in topics}

        # Backend specific checks
        if self.is_openrouter and not self.llm_manager.client:
             logger.error("[BERTopic Representation] OpenRouter client not initialized.")
             return {topic: [("Error: OpenRouter Client", 1)] + [("", 0)] * 9 for topic in topics}
        elif self.is_gemini and not self.llm_manager.client: # Add Gemini check
             logger.error("[BERTopic Representation] Gemini client not initialized.")
             return {topic: [("Error: Gemini Client", 1)] + [("", 0)] * 9 for topic in topics}
        elif self.is_aphrodite:
             if not self.hf_tokenizer or self.hf_tokenizer.chat_template is None:
                 logger.error("[BERTopic Representation] Aphrodite: HF Tokenizer or chat template not available. Cannot format prompts.")
                 return {topic: [("Error: Tokenizer", 1)] + [("", 0)] * 9 for topic in topics}
             if not self.llm_manager.is_running():
                 logger.warning("[BERTopic Representation] Aphrodite service not running. Attempting start...")
                 if not self.llm_manager.start():
                     logger.error("[BERTopic Representation] Failed to start Aphrodite service.")
                     return {topic: [("Error: Aphrodite Start", 1)] + [("", 0)] * 9 for topic in topics}
                 # Ensure model is loaded (important for Aphrodite)
                 status = self.llm_manager.get_status()
                 if not status.get("model_loaded") or status.get("current_model") != self.model_name:
                     logger.info(f"[BERTopic Representation] Loading Aphrodite model {self.model_name} for topics...")
                     if not self.llm_manager.load_model(self.model_name):
                         logger.error(f"[BERTopic Representation] Failed to load Aphrodite model {self.model_name}.")
                         return {topic: [("Error: Aphrodite Load", 1)] + [("", 0)] * 9 for topic in topics}

        # --- Proceed with topic extraction ---
        repr_docs_mappings, _, _, _ = topic_model._extract_representative_docs(
            c_tf_idf, documents, topics, 500, self.nr_docs, self.diversity
        )

        all_topics = []
        all_prompts_for_backend = [] # Store the final strings/messages for the backend
        self.prompts_ = [] # Store structured messages (mainly for Aphrodite/OpenRouter debugging)

        logger.info(f"[BERTopic Representation] Preparing and formatting prompts for {len(repr_docs_mappings)} topics")
        for topic, docs in tqdm(repr_docs_mappings.items(), disable=not topic_model.verbose):
            truncated_docs = [truncate_document(topic_model, self.doc_length, self.doc_length_tokenizer, doc) for doc in docs]
            # Create the prompt content string (used by all backends)
            prompt_content = self._create_prompt_content(truncated_docs, topic, topics)

            if self.is_aphrodite:
                # Aphrodite needs structured messages formatted by tokenizer
                messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt_content}]
                self.prompts_.append(messages) # Store structured for debug
                try:
                    final_prompt_string = self.hf_tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    all_topics.append(topic)
                    all_prompts_for_backend.append(final_prompt_string)
                except Exception as template_err:
                    logger.error(f"[BERTopic Representation] Error applying chat template for topic {topic}: {template_err}", exc_info=True)
                    # Skip this topic if formatting fails
            elif self.is_openrouter:
                 # OpenRouter needs structured messages list
                 messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt_content}]
                 self.prompts_.append(messages) # Store structured for debug
                 all_topics.append(topic)
                 all_prompts_for_backend.append(messages) # Append the list
            elif self.is_gemini:
                 # Gemini needs a single prompt string
                 # Combine system prompt and user content simply for chat generation
                 # (More complex history formatting isn't needed for topic labeling)
                 final_prompt_string = f"System: {self.system_prompt}\n\nUser: {prompt_content}\n\nAssistant:"
                 self.prompts_.append([{"role": "user", "content": final_prompt_string}]) # Store wrapped for debug consistency
                 all_topics.append(topic)
                 all_prompts_for_backend.append(final_prompt_string) # Append the string

        if not all_prompts_for_backend:
             logger.warning("[BERTopic Representation] No prompts were successfully formatted.")
             return {topic: [("Error: Formatting", 1)] + [("", 0)] * 9 for topic in topics}

        logger.info(f"[BERTopic Representation] Generating labels for {len(all_prompts_for_backend)} topics using {LLM_BACKEND.upper()}")
        all_labels = self._generate_topic_labels_batch(all_prompts_for_backend)

        # Map labels back to topics
        updated_topics = {}
        label_idx = 0
        for topic in repr_docs_mappings.keys():
             if topic in all_topics: # Check if formatting succeeded
                 if label_idx < len(all_labels):
                      label = all_labels[label_idx]
                      # Basic cleaning of the label
                      cleaned_label = label.strip().strip('"').strip("'").split('\n')[0]
                      if not cleaned_label or "error" in cleaned_label.lower():
                          cleaned_label = f"Topic {topic} (Generation Error)"
                      updated_topics[topic] = [(cleaned_label, 1)] + [("", 0)] * 9
                      label_idx += 1
                 else:
                      logger.error(f"Label index out of bounds for topic {topic}")
                      updated_topics[topic] = [("Error: Index", 1)] + [("", 0)] * 9
             else:
                 logger.warning(f"Skipping topic {topic} due to earlier formatting error.")
                 updated_topics[topic] = [("Error: Formatting", 1)] + [("", 0)] * 9

        return updated_topics

    def _generate_topic_labels_batch(self, prompts_for_backend: List[Union[str, List[Dict[str, str]]]]) -> List[str]:
        """Generate topic labels for all pre-formatted prompts using the active backend."""
        all_results = ["Topic Extraction Error" for _ in prompts_for_backend] # Default error

        try:
            if self.is_aphrodite:
                # --- Aphrodite Batch Processing ---
                logger.info(f"[BERTopic Representation] Sending batch of {len(prompts_for_backend)} prompts to Aphrodite service")
                aphrodite_results = []
                # Aphrodite generate_chat expects single strings
                valid_prompts = [p for p in prompts_for_backend if isinstance(p, str)]
                if len(valid_prompts) != len(prompts_for_backend):
                     logger.error("Type mismatch in prompts for Aphrodite batch.")
                     # Handle error or proceed with valid ones
                for i, prompt_str in enumerate(valid_prompts):
                    response = self.llm_manager.generate_chat(prompt=prompt_str)
                    if response.get("status") == "success":
                        aphrodite_results.append(response.get("result", "Error: Empty Result"))
                    else:
                        logger.warning(f"Aphrodite generate_chat error for prompt {i}: {response.get('error')}")
                        aphrodite_results.append("Topic Extraction Error")
                # Pad results if some prompts were invalid
                while len(aphrodite_results) < len(prompts_for_backend): aphrodite_results.append("Topic Extraction Error: Invalid Input")
                all_results = aphrodite_results

            elif self.is_openrouter:
                # --- OpenRouter Sequential Processing ---
                logger.info(f"[BERTopic Representation] Sending {len(prompts_for_backend)} prompts sequentially to OpenRouter")
                openrouter_results = []
                # OpenRouter generate_chat expects list of message dicts
                valid_prompts = [p for p in prompts_for_backend if isinstance(p, list)]
                if len(valid_prompts) != len(prompts_for_backend):
                     logger.error("Type mismatch in prompts for OpenRouter batch.")
                     # Handle error or proceed with valid ones
                for i, messages in enumerate(valid_prompts):
                    response = self.llm_manager.generate_chat(
                        messages=messages,
                        model_name=self.model_name, # Pass model name
                        temperature=self.generation_kwargs.get("temperature", 0.3),
                        max_tokens=self.generation_kwargs.get("max_tokens", 64)
                    )
                    if response.get("status") == "success":
                        openrouter_results.append(response.get("result", "Error: Empty Result"))
                    else:
                        logger.warning(f"OpenRouter generate_chat error for prompt {i}: {response.get('error')}")
                        openrouter_results.append("Topic Extraction Error")
                # Pad results if some prompts were invalid
                while len(openrouter_results) < len(prompts_for_backend): openrouter_results.append("Topic Extraction Error: Invalid Input")
                all_results = openrouter_results

            elif self.is_gemini:
                 # --- Gemini Sequential Processing ---
                 logger.info(f"[BERTopic Representation] Sending {len(prompts_for_backend)} prompts sequentially to Gemini")
                 gemini_results = []
                 # Gemini generate_chat expects single strings
                 valid_prompts = [p for p in prompts_for_backend if isinstance(p, str)]
                 if len(valid_prompts) != len(prompts_for_backend):
                      logger.error("Type mismatch in prompts for Gemini batch.")
                      # Handle error or proceed with valid ones
                 for i, prompt_str in enumerate(valid_prompts):
                     response = self.llm_manager.generate_chat(
                         prompt=prompt_str,
                         model_name=self.model_name, # Pass model name
                         temperature=self.generation_kwargs.get("temperature", 0.3),
                         max_tokens=self.generation_kwargs.get("max_tokens", 64)
                     )
                     if response.get("status") == "success":
                         gemini_results.append(response.get("result", "Error: Empty Result"))
                     else:
                         logger.warning(f"Gemini generate_chat error for prompt {i}: {response.get('error')}")
                         gemini_results.append("Topic Extraction Error")
                 # Pad results if some prompts were invalid
                 while len(gemini_results) < len(prompts_for_backend): gemini_results.append("Topic Extraction Error: Invalid Input")
                 all_results = gemini_results
            else:
                logger.error("[BERTopic Representation] No valid LLM backend available.")

        except Exception as e:
            logger.error(f"[BERTopic Representation] Error generating topic labels batch: {e}", exc_info=True)
            # all_results remains default error messages

        # Final check for length consistency
        if len(all_results) != len(prompts_for_backend):
            logger.error(f"Result length mismatch after generation! Expected {len(prompts_for_backend)}, got {len(all_results)}")
            # Pad with errors if necessary
            while len(all_results) < len(prompts_for_backend): all_results.append("Topic Extraction Error: Length Mismatch")
            all_results = all_results[:len(prompts_for_backend)]

        return all_results


    def _create_prompt_content(self, docs, topic, topics) -> str:
        """ Creates the user prompt content string for the topic extraction task. """
        keywords = list(zip(*topics[topic]))[0]
        keyword_str = ", ".join(keywords)
        doc_str = "".join([f"- {doc}\n" for doc in docs])
        user_prompt = self.prompt_template
        if "[KEYWORDS]" in user_prompt: user_prompt = user_prompt.replace("[KEYWORDS]", keyword_str)
        if "[DOCUMENTS]" in user_prompt: user_prompt = user_prompt.replace("[DOCUMENTS]", doc_str)
        # Return only the content string, not the full message structure
        return user_prompt
