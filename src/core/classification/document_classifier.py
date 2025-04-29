"""
Document classification module for Anti-Corruption RAG System.
Uses the active LLM backend (Aphrodite, OpenRouter, or Gemini) to classify document chunks
according to user-defined schemas.
"""
import sys
import os
from pathlib import Path
import uuid
import yaml
import json
import time
import csv
import random
from typing import List, Dict, Any, Optional, Union, Type, Set, Literal, get_type_hints
import pandas as pd
import traceback # For better error logging

# Add project root to path
# Assuming this file is in src/core/classification/
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))

# Import utils
from src.utils.logger import setup_logger
from src.utils.resource_monitor import log_memory_usage
# Import backend managers conditionally based on config
from src.ui.app_setup import (
    get_active_llm_manager, IS_OPENROUTER_ACTIVE, IS_GEMINI_ACTIVE, LLM_BACKEND
)

# Import Pydantic for dynamic model creation (needed for Gemini)
try:
    from pydantic import BaseModel, Field, create_model
    from pydantic_core import PydanticUndefined
    PYDANTIC_AVAILABLE = True
except ImportError:
    BaseModel = None
    Field = None
    create_model = None
    PydanticUndefined = None
    PYDANTIC_AVAILABLE = False
    # Logger might not be initialized yet, use print
    print("WARNING: Pydantic not found. Dynamic schema creation for Gemini will fail.")

# Import specific manager classes conditionally
if IS_OPENROUTER_ACTIVE:
    from src.utils.openrouter_manager import OpenRouterManager
    AphroditeService = None
    GeminiManager = None
    try: from transformers import AutoTokenizer
    except ImportError: AutoTokenizer = None
elif IS_GEMINI_ACTIVE:
    from src.utils.gemini_manager import GeminiManager
    AphroditeService = None
    OpenRouterManager = None
    AutoTokenizer = None # Not needed for Gemini prompt formatting
else: # Aphrodite
    from src.utils.aphrodite_service import AphroditeService
    from transformers import AutoTokenizer # Needed for Aphrodite templating
    OpenRouterManager = None
    GeminiManager = None

# Initialize logger
logger = setup_logger(__name__)

# Load configuration
CONFIG_PATH = ROOT_DIR / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

# Helper function to map schema types to Python types for Pydantic
def _get_python_type(type_str: str) -> Type:
    """Convert string type name to Python type for Pydantic."""
    type_map = {
        "string": str,
        "number": float,
        "integer": int,
        "boolean": bool,
        "date": str,  # Store dates as strings
        "array": list, # Basic list, might need refinement for typed lists
        "object": dict # Basic dict
    }
    return type_map.get(type_str.lower(), str) # Default to string

class DocumentClassifier:
    """
    LLM-based document classifier using the active backend.
    """

    def __init__(self, model_name=None, debug=False):
        """
        Initialize document classifier.

        Args:
            model_name (str, optional): Name of the model to use for classification.
            debug (bool, optional): Enable debugging output.
        """
        self.debug = debug
        self.llm_manager = get_active_llm_manager()
        self.is_openrouter = isinstance(self.llm_manager, OpenRouterManager) if OpenRouterManager else False
        self.is_aphrodite = isinstance(self.llm_manager, AphroditeService) if AphroditeService else False
        self.is_gemini = isinstance(self.llm_manager, GeminiManager) if GeminiManager else False # Add Gemini flag

        # Determine model name based on backend
        if model_name:
            self.model_name = model_name
        elif self.is_openrouter:
            self.model_name = self.llm_manager.models.get("classification", CONFIG["openrouter"]["classification_model"])
        elif self.is_gemini: # Add Gemini case
            # Use the specific model from the example
            self.model_name = self.llm_manager.models.get("classification", "gemini-2.5-pro-exp-03-25")
        elif self.is_aphrodite:
            # Use the standard extraction model for Aphrodite classification as well
            self.model_name = CONFIG["models"]["extraction_models"]["text_standard"]
        else:
            self.model_name = "default/unknown"
            logger.error("DocumentClassifier initialized without a valid LLM manager!")

        self.classification_data_path = ROOT_DIR / "data" / "classification"
        self.classification_data_path.mkdir(parents=True, exist_ok=True)

        # --- Load Tokenizer (Only needed for Aphrodite path) ---
        self.tokenizer = None
        if self.is_aphrodite and AutoTokenizer:
            try:
                # Use the specific model designated for classification
                aphrodite_model_for_class = CONFIG["models"]["extraction_models"]["text_standard"]
                logger.info(f"Loading tokenizer for Aphrodite classification model: {aphrodite_model_for_class}")
                self.tokenizer = AutoTokenizer.from_pretrained(aphrodite_model_for_class, trust_remote_code=True)
                if self.tokenizer.chat_template is None:
                    logger.warning(f"Tokenizer for {aphrodite_model_for_class} lacks chat_template. Manual formatting fallback.")
                else:
                    logger.info(f"Tokenizer for {aphrodite_model_for_class} loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load tokenizer for {aphrodite_model_for_class}: {e}", exc_info=True)
        elif self.is_aphrodite:
             logger.error("AutoTokenizer could not be imported. Aphrodite classification may fail.")

        logger.info(f"Initialized DocumentClassifier with backend: {LLM_BACKEND.upper()}, model={self.model_name}, debug={debug}")
        log_memory_usage(logger)

    def _create_dynamic_classification_model(self, schema: Dict[str, Any], multi_label_fields: Set[str]) -> Optional[Type[BaseModel]]:
        """
        Dynamically creates a Pydantic model class for classification.

        Args:
            schema: Dictionary defining the classification schema {'field_name': {'values': [], 'description': ''}}
            multi_label_fields: Set of field names that accept multiple values

        Returns:
            A Pydantic BaseModel class or None on failure.
        """
        if not PYDANTIC_AVAILABLE:
            logger.error("Pydantic is not available. Cannot create dynamic model for Gemini classification.")
            return None
        if not schema:
            logger.error("Cannot create dynamic classification model from empty schema.")
            return None

        try:
            fields = {}
            for field_name, field_info in schema.items():
                allowed_values = field_info.get("values", [])
                description = field_info.get("description", "")

                if not allowed_values:
                    logger.warning(f"Field '{field_name}' has no allowed values. Skipping.")
                    continue # Skip fields without values

                is_multi_label = field_name in multi_label_fields

                # Create the appropriate field type using Literal and List
                if is_multi_label:
                    # Ensure all values are strings for Literal
                    str_values = tuple(str(v) for v in allowed_values)
                    if not str_values: continue # Skip if no valid string values
                    field_type = Optional[List[Literal[str_values]]]
                else:
                    # Ensure all values are strings for Literal
                    str_values = tuple(str(v) for v in allowed_values)
                    if not str_values: continue # Skip if no valid string values
                    field_type = Optional[Literal[str_values]]

                # Add field with proper type and metadata, default to None
                fields[field_name] = (field_type, Field(description=description, default=None))

            if not fields:
                 logger.error("No valid fields could be defined for the dynamic classification model.")
                 return None

            # Create the dynamic model class
            DynamicClassificationModel = create_model("DynamicClassificationModel", **fields)

            logger.info(f"Successfully created dynamic Pydantic classification model with {len(fields)} fields.")
            return DynamicClassificationModel

        except Exception as e:
            logger.error(f"Error creating dynamic Pydantic classification model: {e}", exc_info=True)
            return None

    def ensure_model_loaded(self):
        """Ensure the designated classification model is ready in the active backend."""
        if self.is_openrouter:
            if self.llm_manager and self.llm_manager.client:
                logger.info(f"OpenRouter backend ready for classification (using model: {self.model_name}).")
                return True
            else:
                logger.error("OpenRouter manager or client not ready.")
                return False
        elif self.is_gemini: # Add Gemini check
             if self.llm_manager and self.llm_manager.client:
                 logger.info(f"Gemini backend ready for classification (using model: {self.model_name}).")
                 return True
             else:
                 logger.error("Gemini manager or client not ready.")
                 return False
        elif self.is_aphrodite:
            if not self.llm_manager.is_running():
                logger.info("Aphrodite service not running, starting it")
                if not self.llm_manager.start(): logger.error("Failed to start Aphrodite service"); return False
            status = self.llm_manager.get_status()
            # Use the specific model designated for classification
            aphrodite_model_for_class = CONFIG["models"]["extraction_models"]["text_standard"]
            if not status.get("model_loaded", False) or status.get("current_model") != aphrodite_model_for_class:
                logger.info(f"Loading Aphrodite classification model: {aphrodite_model_for_class}")
                if not self.llm_manager.load_model(aphrodite_model_for_class):
                    logger.error(f"Failed to load Aphrodite model {aphrodite_model_for_class}"); return False
                logger.info(f"Aphrodite model {aphrodite_model_for_class} loaded successfully.")
            else:
                logger.info(f"Aphrodite classification model {aphrodite_model_for_class} already loaded.")
            return True
        else:
            logger.error("No valid LLM backend manager available.")
            return False

    def classify_documents(self, chunks: List[Dict[str, Any]], schema: Dict[str, Any],
                           multi_label_fields: Set[str], user_instructions: str) -> List[Dict[str, Any]]:
        """
        Classify document chunks according to the provided schema using the active backend.

        Args:
            chunks: List of document chunks (must contain 'text')
            schema: Dictionary defining the classification schema {'field_name': {'values': [], 'description': ''}}
            multi_label_fields: Set of field names that accept multiple values
            user_instructions: User-provided instructions or context for classification

        Returns:
            List of classified document dictionaries including original chunk data and 'classification' dict.
        """
        if not chunks: logger.warning("No chunks provided for classification"); return []
        if not self.llm_manager: logger.error("LLM Manager not available."); return []

        # Backend specific checks
        if self.is_aphrodite and not self.tokenizer:
             logger.error("Aphrodite backend: Tokenizer not loaded. Cannot perform classification.")
             return []
        if self.is_aphrodite and self.tokenizer and self.tokenizer.chat_template is None:
             logger.warning(f"Aphrodite backend: Tokenizer {self.model_name} has no chat template. Formatting may be incorrect.")
        if self.is_gemini and not PYDANTIC_AVAILABLE:
             logger.error("Gemini backend: Pydantic not available. Cannot create dynamic schema for classification.")
             return []

        if not self.ensure_model_loaded():
            logger.error(f"Failed to ensure model is ready for backend {LLM_BACKEND.upper()}")
            return []

        # Prepare schema information once
        schema_definition_formatted = ""
        for field_name, field_info in schema.items():
            allowed_values = field_info.get("values", [])
            description = field_info.get("description", "")
            values_str = ", ".join([f'"{val}"' for val in allowed_values])
            multi_label_str = " (allows multiple values)" if field_name in multi_label_fields else ""
            schema_definition_formatted += f"- {field_name}: {description}{multi_label_str}\n"
            schema_definition_formatted += f"  Allowed values: [{values_str}]\n\n"

        # Create example for the prompt
        example = {}
        for field_name, field_info in schema.items():
            allowed_values = field_info.get("values", [])
            if not allowed_values: continue
            if field_name in multi_label_fields:
                num_values = min(2, len(allowed_values))
                if num_values > 0: example[field_name] = random.sample(allowed_values, num_values)
                else: example[field_name] = [] # Empty list if no values
            else:
                if allowed_values: example[field_name] = allowed_values[0]
                else: example[field_name] = None # Use null if no values
        json_example = json.dumps(example, indent=2)

        classified_documents = []
        total_chunks = len(chunks)
        logger.info(f"Starting classification of {total_chunks} chunks using {LLM_BACKEND.upper()}...")

        # --- Backend Specific Processing ---
        if self.is_openrouter:
            # --- OpenRouter Path (Sequential Calls) ---
            json_schema_for_or = self.llm_manager._dict_schema_to_json_schema(schema, multi_label_fields)
            if not json_schema_for_or:
                logger.error("Failed to convert dict schema to JSON schema for OpenRouter classification.")
                return []

            processed_count = 0
            error_count = 0
            for i, chunk in enumerate(chunks):
                chunk_text = chunk.get('text', '')
                if not chunk_text.strip(): continue

                # Create OpenAI messages
                messages = self._format_classification_prompt(
                    chunk_text, schema_definition_formatted, json_example, user_instructions
                )

                if (i + 1) % 50 == 0: # Log progress
                    logger.info(f"Classifying chunk {i+1}/{total_chunks} via OpenRouter...")

                try:
                    response = self.llm_manager.extract_structured(
                        messages=messages,
                        json_schema=json_schema_for_or, # Pass the schema directly
                        model_name=self.model_name,
                        task_name="classification"
                    )

                    classification_data = {"error": "Classification failed"}
                    if response.get("status") == "success":
                        classification_data = response.get("result", {})
                        if isinstance(classification_data, dict):
                            processed_count += 1
                        else:
                            logger.warning(f"OpenRouter returned non-dict classification for chunk {chunk.get('chunk_id', i)}: {type(classification_data)}")
                            classification_data = {"error": "Invalid format received"}
                            error_count += 1
                    else:
                        logger.warning(f"OpenRouter classification failed for chunk {chunk.get('chunk_id', i)}: {response.get('error')}")
                        classification_data = {"error": response.get('error', 'Unknown classification error')}
                        error_count += 1

                    # Combine with original chunk data
                    classified_doc = {
                        "chunk_id": chunk.get("chunk_id", f"chunk_{i}"),
                        "document_id": chunk.get("document_id", "unknown"),
                        "file_name": chunk.get("file_name", "unknown"),
                        "page_num": chunk.get("page_num", None),
                        "text": chunk.get("text", ""),
                        "classification": classification_data,
                        **{f"class_{k}": v for k, v in classification_data.items()} # Flatten
                    }
                    classified_documents.append(classified_doc)

                except Exception as e:
                    error_count += 1
                    logger.error(f"Error calling OpenRouter for classification chunk {chunk.get('chunk_id', i)}: {e}", exc_info=True)
                    classified_doc = { # Add error entry
                        "chunk_id": chunk.get("chunk_id", f"chunk_{i}"), "document_id": chunk.get("document_id", "unknown"),
                        "file_name": chunk.get("file_name", "unknown"), "page_num": chunk.get("page_num", None),
                        "text": chunk.get("text", ""), "classification": {"error": str(e)}, "class_error": str(e)
                    }
                    classified_documents.append(classified_doc)

            logger.info(f"OpenRouter classification complete. Success: {processed_count}, Errors: {error_count}. Total results: {len(classified_documents)}")

        elif self.is_gemini:
            # --- Gemini Path (Sequential Calls) ---
            # Create the dynamic Pydantic model class ONCE
            DynamicClassificationModel = self._create_dynamic_classification_model(schema, multi_label_fields)
            if not DynamicClassificationModel:
                logger.error("Failed to create dynamic Pydantic model for Gemini classification.")
                return []

            processed_count = 0
            error_count = 0
            for i, chunk in enumerate(chunks):
                chunk_text = chunk.get('text', '')
                if not chunk_text.strip(): continue

                # Create the single prompt string for Gemini
                messages = self._format_classification_prompt(
                    chunk_text, schema_definition_formatted, json_example, user_instructions
                )
                prompt_string = messages[1]['content'] # Use user prompt content

                if (i + 1) % 50 == 0: # Log progress
                    logger.info(f"Classifying chunk {i+1}/{total_chunks} via Gemini...")

                try:
                    # Call Gemini manager's extract_structured
                    response = self.llm_manager.extract_structured(
                        prompt=prompt_string,
                        pydantic_schema=DynamicClassificationModel, # Pass the dynamic Pydantic class
                        model_name=self.model_name,
                        task_name="classification"
                    )

                    classification_data = {"error": "Classification failed"}
                    if response.get("status") == "success":
                        classification_data = response.get("result", {})
                        if isinstance(classification_data, dict):
                            processed_count += 1
                        else:
                            logger.warning(f"Gemini returned non-dict classification for chunk {chunk.get('chunk_id', i)}: {type(classification_data)}")
                            classification_data = {"error": "Invalid format received"}
                            error_count += 1
                    else:
                        logger.warning(f"Gemini classification failed for chunk {chunk.get('chunk_id', i)}: {response.get('error')}")
                        classification_data = {"error": response.get('error', 'Unknown classification error')}
                        error_count += 1

                    # Combine with original chunk data
                    classified_doc = {
                        "chunk_id": chunk.get("chunk_id", f"chunk_{i}"),
                        "document_id": chunk.get("document_id", "unknown"),
                        "file_name": chunk.get("file_name", "unknown"),
                        "page_num": chunk.get("page_num", None),
                        "text": chunk.get("text", ""),
                        "classification": classification_data,
                        **{f"class_{k}": v for k, v in classification_data.items()} # Flatten
                    }
                    classified_documents.append(classified_doc)

                except Exception as e:
                    error_count += 1
                    logger.error(f"Error calling Gemini for classification chunk {chunk.get('chunk_id', i)}: {e}", exc_info=True)
                    classified_doc = { # Add error entry
                        "chunk_id": chunk.get("chunk_id", f"chunk_{i}"), "document_id": chunk.get("document_id", "unknown"),
                        "file_name": chunk.get("file_name", "unknown"), "page_num": chunk.get("page_num", None),
                        "text": chunk.get("text", ""), "classification": {"error": str(e)}, "class_error": str(e)
                    }
                    classified_documents.append(classified_doc)

            logger.info(f"Gemini classification complete. Success: {processed_count}, Errors: {error_count}. Total results: {len(classified_documents)}")


        elif self.is_aphrodite:
            # --- Aphrodite Path (Batch Call) ---
            prompts_for_service = []
            chunk_mapping = []

            logger.info(f"Preparing classification prompts for {len(chunks)} chunks (Aphrodite)...")
            for chunk in chunks:
                chunk_text = chunk.get('text', '')
                if not chunk_text.strip(): continue

                messages = self._format_classification_prompt(
                    chunk_text, schema_definition_formatted, json_example, user_instructions
                )
                try:
                    final_prompt_string = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    prompts_for_service.append(final_prompt_string)
                    chunk_mapping.append(chunk)
                except Exception as template_err:
                    logger.error(f"Error applying chat template for chunk {chunk.get('chunk_id', 'N/A')}: {template_err}", exc_info=True)

            if not prompts_for_service:
                 logger.warning("No valid prompts generated for Aphrodite classification.")
                 return []

            logger.info(f"Sending {len(prompts_for_service)} formatted prompts to Aphrodite service for classification.")
            try:
                # Call Aphrodite's classify_chunks
                response = self.llm_manager.classify_chunks(
                    prompts=prompts_for_service,
                    schema_definition=schema, # Pass original schema dict
                    multi_label_fields=list(multi_label_fields)
                )

                if response.get("status") != "success":
                    logger.error(f"Error classifying chunks batch via Aphrodite: {response.get('error', 'Unknown error')}")
                    # Attempt to process partial results if available
                    results = response.get("results", [])
                    if not results: return [] # No results to process
                    logger.warning("Processing potentially partial results from failed Aphrodite batch.")
                else:
                    results = response.get("results", []) # Expect list of results (one per prompt)

                if len(results) != len(prompts_for_service):
                    logger.warning(f"Aphrodite result count mismatch: Expected {len(prompts_for_service)}, Got {len(results)}")
                    # Pad with errors for missing results
                    while len(results) < len(prompts_for_service): results.append({"status": "error", "error": "Missing result"})

                processed_count = 0
                error_count = 0
                for i in range(len(prompts_for_service)): # Iterate based on original prompt count
                    original_chunk = chunk_mapping[i]
                    classification_data = {"error": "Classification failed"}

                    # Check if result exists for this index
                    if i < len(results):
                        result_data = results[i]
                        if isinstance(result_data, dict) and result_data.get("status") == "success":
                            classification_data = result_data.get("result", {})
                            if isinstance(classification_data, dict):
                                processed_count += 1
                            else:
                                logger.warning(f"Aphrodite returned non-dict classification for chunk {original_chunk.get('chunk_id', i)}: {type(classification_data)}")
                                classification_data = {"error": "Invalid format received"}
                                error_count += 1
                        else:
                            error = result_data.get("error", "Unknown classification error") if isinstance(result_data, dict) else "Invalid result format"
                            logger.warning(f"Aphrodite classification error for chunk {original_chunk.get('chunk_id', i)}: {error}")
                            classification_data = {"error": error}
                            error_count += 1
                    else:
                        # This case handles padding if results were shorter than prompts
                        logger.warning(f"Missing result for chunk {original_chunk.get('chunk_id', i)} (index {i}).")
                        classification_data = {"error": "Missing result from batch"}
                        error_count += 1


                    # Combine with original chunk data
                    classified_doc = {
                        "chunk_id": original_chunk.get("chunk_id", f"chunk_{i}"),
                        "document_id": original_chunk.get("document_id", "unknown"),
                        "file_name": original_chunk.get("file_name", "unknown"),
                        "page_num": original_chunk.get("page_num", None),
                        "text": original_chunk.get("text", ""),
                        "classification": classification_data,
                        **{f"class_{k}": v for k, v in classification_data.items()} # Flatten
                    }
                    classified_documents.append(classified_doc)

                logger.info(f"Aphrodite classification complete. Success: {processed_count}, Errors: {error_count}. Total results: {len(classified_documents)}")

            except Exception as e:
                logger.error(f"Critical error during Aphrodite classification batch: {e}", exc_info=True)
                # Return whatever might have been processed before the error
                return classified_documents
        else:
            logger.error("No valid LLM backend configured for document classification.")
            return []

        return classified_documents


    def _format_classification_prompt(self, text: str, schema_definition: str,
                                      json_example: str, user_instructions: str) -> List[Dict[str, str]]:
        """
        Creates the structured message list for the classification task.
        (Used to generate 'user' content for all backends).

        Args:
            text: Text to classify
            schema_definition: String describing the schema fields and allowed values
            json_example: String showing example JSON output format
            user_instructions: User-provided instructions

        Returns:
            A list of dictionaries representing the conversation structure.
        """
        system_prompt = """You are an AI assistant that classifies text according to a specified schema.
You always output valid JSON that conforms exactly to the requested schema and allowed values.
You never include additional explanation text in your responses."""

        # User prompt suitable for all backends
        user_prompt = f"""Please classify the following text based on the provided schema and instructions.

## CLASSIFICATION SCHEMA:
{schema_definition}

## USER INSTRUCTIONS:
{user_instructions}

## OUTPUT FORMAT:
Your response must be a single JSON object matching this structure exactly. Use only the allowed values for each field.
{json_example}

## IMPORTANT:
- For fields allowing multiple values, return a JSON list of strings (e.g., ["value1", "value2"]).
- For fields allowing only a single value, return a single JSON string (e.g., "value1").
- If a classification cannot be determined confidently based on the text and instructions, use `null` for that field's value.
- Do NOT include any explanations or text outside the JSON object.

## TEXT TO CLASSIFY:
{text}
"""
        # For OpenRouter, return messages list. For Gemini/Aphrodite, the user prompt string will be extracted.
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return messages

    def export_to_csv(self, classified_docs: List[Dict[str, Any]], file_path: str) -> bool:
        """ Export classification results to CSV. (Unchanged) """
        try:
            if not classified_docs: logger.warning("No classified documents to export"); return False
            df = pd.DataFrame(classified_docs)
            # Optionally flatten the 'classification' column if needed, but current structure includes flattened fields
            # df_flat = pd.json_normalize(classified_docs, sep='_') # Alternative flattening
            df.to_csv(file_path, index=False)
            logger.info(f"Exported {len(classified_docs)} classified documents to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error exporting classification results to CSV: {e}")
            return False