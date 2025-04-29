
"""
Information extraction module for Anti-Corruption RAG System.
Uses the active LLM backend (Aphrodite, OpenRouter, or Gemini) to extract structured data
from document chunks based on user-defined schemas.
"""
import sys
import os
from pathlib import Path
import uuid
import yaml
import json
import time
import csv
from typing import List, Dict, Any, Optional, Union, Type, get_type_hints
import pandas as pd
import traceback # For better error logging

# Add project root to path
# Assuming this file is in src/core/extraction/
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

class InfoExtractor:
    """
    LLM-based information extractor for structured data using the active backend.
    """

    def __init__(self, model_name=None, debug=False):
        """
        Initialize information extractor.

        Args:
            model_name (str, optional): Name of the model to use for extraction.
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
            self.model_name = self.llm_manager.models.get("info_extraction", CONFIG["openrouter"]["info_extraction_model"])
        elif self.is_gemini: # Add Gemini case
            # Use the specific model from the example
            self.model_name = self.llm_manager.models.get("info_extraction", "gemini-2.5-pro-exp-03-25")
        elif self.is_aphrodite:
            # Use the standard extraction model for Aphrodite info extraction as well
            self.model_name = CONFIG["models"]["extraction_models"]["text_standard"]
        else:
            self.model_name = "default/unknown"
            logger.error("InfoExtractor initialized without a valid LLM manager!")

        self.extracted_data_path = ROOT_DIR / "data" / "extracted" / "info"
        self.extracted_data_path.mkdir(parents=True, exist_ok=True)

        # --- Load Tokenizer (Only needed for Aphrodite path) ---
        self.tokenizer = None
        if self.is_aphrodite and AutoTokenizer:
            try:
                # Use the specific model designated for info extraction
                aphrodite_model_for_info = CONFIG["models"]["extraction_models"]["text_standard"]
                logger.info(f"Loading tokenizer for Aphrodite info extraction model: {aphrodite_model_for_info}")
                self.tokenizer = AutoTokenizer.from_pretrained(aphrodite_model_for_info, trust_remote_code=True)
                if self.tokenizer.chat_template is None:
                    logger.warning(f"Tokenizer for {aphrodite_model_for_info} lacks chat_template. Manual formatting fallback.")
                else:
                    logger.info(f"Tokenizer for {aphrodite_model_for_info} loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load tokenizer for {aphrodite_model_for_info}: {e}", exc_info=True)
        elif self.is_aphrodite:
             logger.error("AutoTokenizer could not be imported. Aphrodite info extraction may fail.")


        logger.info(f"Initialized InfoExtractor with backend: {LLM_BACKEND.upper()}, model={self.model_name}, debug={debug}")
        log_memory_usage(logger)

    def _create_dynamic_pydantic_model(self, schema_dict: Dict[str, Any]) -> Optional[Type[BaseModel]]:
        """
        Dynamically creates a Pydantic model class based on the input schema dictionary.
        Wraps the dynamic model in a list structure as required by Gemini example.

        Args:
            schema_dict: Dictionary defining the extraction schema {'field_name': {'type': 'str', 'description': 'desc'}}

        Returns:
            A Pydantic BaseModel class representing List[DynamicModel] or None on failure.
        """
        if not PYDANTIC_AVAILABLE:
            logger.error("Pydantic is not available. Cannot create dynamic model for Gemini.")
            return None
        if not schema_dict:
            logger.error("Cannot create dynamic model from empty schema dictionary.")
            return None

        try:
            fields = {}
            for field_name, field_info in schema_dict.items():
                field_type = _get_python_type(field_info.get("type", "string"))
                description = field_info.get("description", "")
                # Use Optional for all fields to handle missing data gracefully
                # Pydantic v2: Optional[T] is equivalent to Union[T, None]
                # Default to None
                fields[field_name] = (Optional[field_type], Field(description=description, default=None))

            # Create the dynamic model for a single item
            DynamicItemModel = create_model("DynamicInfoItemModel", **fields)

            # Create the top-level model which is a list of these items
            # Mimics the structure: class DynamicInfoList(BaseModel): items: List[DynamicItemModel]
            DynamicListModel = create_model(
                "DynamicInfoListModel",
                items=(List[DynamicItemModel], Field(..., description="List of extracted information items"))
            )

            logger.info(f"Successfully created dynamic Pydantic model 'DynamicInfoListModel' with {len(fields)} fields per item.")
            return DynamicListModel

        except Exception as e:
            logger.error(f"Error creating dynamic Pydantic model: {e}", exc_info=True)
            return None

    def ensure_model_loaded(self):
        """Ensure the designated extraction model is ready in the active backend."""
        if self.is_openrouter:
            if self.llm_manager and self.llm_manager.client:
                logger.info(f"OpenRouter backend ready for info extraction (using model: {self.model_name}).")
                return True
            else:
                logger.error("OpenRouter manager or client not ready.")
                return False
        elif self.is_gemini: # Add Gemini check
             if self.llm_manager and self.llm_manager.client:
                 logger.info(f"Gemini backend ready for info extraction (using model: {self.model_name}).")
                 return True
             else:
                 logger.error("Gemini manager or client not ready.")
                 return False
        elif self.is_aphrodite:
            if not self.llm_manager.is_running():
                logger.info("Aphrodite service not running, starting it")
                if not self.llm_manager.start(): logger.error("Failed to start Aphrodite service"); return False
            status = self.llm_manager.get_status()
            # Use the specific model designated for info extraction
            aphrodite_model_for_info = CONFIG["models"]["extraction_models"]["text_standard"]
            if not status.get("model_loaded", False) or status.get("current_model") != aphrodite_model_for_info:
                logger.info(f"Loading Aphrodite info extraction model: {aphrodite_model_for_info}")
                if not self.llm_manager.load_model(aphrodite_model_for_info):
                    logger.error(f"Failed to load Aphrodite model {aphrodite_model_for_info}"); return False
                logger.info(f"Aphrodite model {aphrodite_model_for_info} loaded successfully.")
            else:
                logger.info(f"Aphrodite info extraction model {aphrodite_model_for_info} already loaded.")
            return True
        else:
            logger.error("No valid LLM backend manager available.")
            return False

    def extract_information(self, chunks: List[Dict[str, Any]], schema_dict: Dict[str, Any],
                           primary_key_field: str, primary_key_description: str, user_query: str) -> List[Dict[str, Any]]:
        """
        Extract structured information from chunks based on user-defined schema using the active backend.

        Args:
            chunks: List of document chunks (must contain 'text')
            schema_dict: Dictionary defining the extraction schema {'field_name': {'type': 'str', 'description': 'desc'}}
            primary_key_field: Name of the primary entity field (e.g., 'company')
            primary_key_description: Description of the primary entity (e.g., 'companies')
            user_query: User query describing the extraction task

        Returns:
            List of extracted information dictionaries, each including a '_source' key with chunk metadata.
        """
        if not chunks: logger.warning("No chunks provided for extraction"); return []
        if not self.llm_manager: logger.error("LLM Manager not available."); return []

        # Backend specific checks
        if self.is_aphrodite and not self.tokenizer:
             logger.error("Aphrodite backend: Tokenizer not loaded. Cannot perform extraction.")
             return []
        if self.is_aphrodite and self.tokenizer and self.tokenizer.chat_template is None:
             logger.warning(f"Aphrodite backend: Tokenizer {self.model_name} has no chat template. Formatting may be incorrect.")
        if self.is_gemini and not PYDANTIC_AVAILABLE:
             logger.error("Gemini backend: Pydantic not available. Cannot create dynamic schema for extraction.")
             return []

        if not self.ensure_model_loaded():
            logger.error(f"Failed to ensure model is ready for backend {LLM_BACKEND.upper()}")
            return []

        # Prepare schema information once
        schema_definition_str = ""
        for field_name, field_info in schema_dict.items():
            schema_definition_str += f"- {field_name}: {field_info['description']} (Type: {field_info['type']})\n"
        # Create example for the prompt (remains useful for all backends)
        json_schema_example_obj = {fn: f"<{fi['type']} value>" for fn, fi in schema_dict.items()}
        # IMPORTANT: The example output should be a LIST containing objects, matching the dynamic Pydantic model structure
        json_schema_example = json.dumps([json_schema_example_obj], indent=2)

        all_extracted_info = []
        total_chunks = len(chunks)
        logger.info(f"Starting information extraction from {total_chunks} chunks using {LLM_BACKEND.upper()}...")

        # --- Backend Specific Processing ---
        if self.is_openrouter:
            # --- OpenRouter Path (Sequential Calls) ---
            json_schema_for_or = self.llm_manager._dict_schema_to_json_schema(schema_dict)
            if not json_schema_for_or:
                logger.error("Failed to convert dict schema to JSON schema for OpenRouter.")
                return []
            # Wrap the properties in a top-level object with a key like "items" and type "array"
            # This matches the structure expected by the processing loop later.
            wrapped_schema_for_or = {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": json_schema_for_or # The original schema defines the items in the array
                    }
                },
                "required": ["items"]
            }

            processed_count = 0
            for i, chunk in enumerate(chunks):
                chunk_text = chunk.get('text', '')
                if not chunk_text.strip(): continue

                # Create OpenAI messages
                messages = self._format_extraction_prompt(
                    chunk_text, schema_definition_str, json_schema_example,
                    primary_key_field, primary_key_description, user_query
                )

                if (i + 1) % 50 == 0: # Log progress periodically
                    logger.info(f"Processing chunk {i+1}/{total_chunks} via OpenRouter...")

                try:
                    response = self.llm_manager.extract_structured(
                        messages=messages,
                        json_schema=wrapped_schema_for_or, # Use the wrapped schema
                        model_name=self.model_name,
                        task_name="info_extraction"
                    )

                    if response.get("status") == "success":
                        # Expecting {"items": [...]}
                        result_data_wrapper = response.get("result", {})
                        result_data = result_data_wrapper.get("items", []) # Extract the list

                        if not isinstance(result_data, list):
                            logger.warning(f"OpenRouter returned non-list result inside 'items' for chunk {chunk.get('chunk_id', i)}: {type(result_data)}. Wrapping.")
                            result_data = [result_data] if isinstance(result_data, dict) else []

                        items_extracted_from_chunk = 0
                        for item in result_data:
                            if isinstance(item, dict):
                                item_with_source = item.copy()
                                item_with_source['_source'] = {
                                    'chunk_id': chunk.get('chunk_id', 'unknown'),
                                    'document_id': chunk.get('document_id', 'unknown'),
                                    'file_name': chunk.get('file_name', 'unknown'),
                                    'page_num': chunk.get('page_num', None)
                                }
                                all_extracted_info.append(item_with_source)
                                items_extracted_from_chunk += 1
                        if items_extracted_from_chunk > 0: processed_count += 1
                        if self.debug and items_extracted_from_chunk > 0: logger.debug(f"Extracted {items_extracted_from_chunk} items from chunk {chunk.get('chunk_id', i)}")

                    else:
                        logger.warning(f"OpenRouter extraction failed for chunk {chunk.get('chunk_id', i)}: {response.get('error')}")

                except Exception as e:
                    logger.error(f"Error calling OpenRouter for chunk {chunk.get('chunk_id', i)}: {e}", exc_info=True)

            logger.info(f"OpenRouter extraction complete. Processed {processed_count} chunks successfully, yielding {len(all_extracted_info)} items.")

        elif self.is_gemini:
            # --- Gemini Path (Sequential Calls) ---
            # Create the dynamic Pydantic model class ONCE
            DynamicInfoListModel = self._create_dynamic_pydantic_model(schema_dict)
            if not DynamicInfoListModel:
                logger.error("Failed to create dynamic Pydantic model for Gemini extraction.")
                return []

            processed_count = 0
            for i, chunk in enumerate(chunks):
                chunk_text = chunk.get('text', '')
                if not chunk_text.strip(): continue

                # Create the single prompt string for Gemini
                # Use the same formatting function, but extract the user content
                messages = self._format_extraction_prompt(
                    chunk_text, schema_definition_str, json_schema_example,
                    primary_key_field, primary_key_description, user_query
                )
                # Combine system and user prompts into a single string for Gemini
                # A simple concatenation might work, or follow specific Gemini guidelines if available
                # For now, just use the user prompt content which contains all instructions.
                prompt_string = messages[1]['content'] # Index 1 is the user prompt

                if (i + 1) % 50 == 0: # Log progress periodically
                    logger.info(f"Processing chunk {i+1}/{total_chunks} via Gemini...")

                try:
                    # Call Gemini manager's extract_structured
                    response = self.llm_manager.extract_structured(
                        prompt=prompt_string,
                        pydantic_schema=DynamicInfoListModel, # Pass the dynamic Pydantic class
                        model_name=self.model_name,
                        task_name="info_extraction"
                    )

                    if response.get("status") == "success":
                        # Expecting {"items": [...]} as defined by DynamicInfoListModel
                        result_data_wrapper = response.get("result", {})
                        result_data = result_data_wrapper.get("items", []) # Extract the list

                        if not isinstance(result_data, list):
                            logger.warning(f"Gemini returned non-list result inside 'items' for chunk {chunk.get('chunk_id', i)}: {type(result_data)}. Wrapping.")
                            result_data = [result_data] if isinstance(result_data, dict) else []

                        items_extracted_from_chunk = 0
                        for item in result_data:
                            if isinstance(item, dict):
                                item_with_source = item.copy()
                                item_with_source['_source'] = {
                                    'chunk_id': chunk.get('chunk_id', 'unknown'),
                                    'document_id': chunk.get('document_id', 'unknown'),
                                    'file_name': chunk.get('file_name', 'unknown'),
                                    'page_num': chunk.get('page_num', None)
                                }
                                all_extracted_info.append(item_with_source)
                                items_extracted_from_chunk += 1
                        if items_extracted_from_chunk > 0: processed_count += 1
                        if self.debug and items_extracted_from_chunk > 0: logger.debug(f"Extracted {items_extracted_from_chunk} items from chunk {chunk.get('chunk_id', i)}")

                    else:
                        logger.warning(f"Gemini extraction failed for chunk {chunk.get('chunk_id', i)}: {response.get('error')}")

                except Exception as e:
                    logger.error(f"Error calling Gemini for chunk {chunk.get('chunk_id', i)}: {e}", exc_info=True)

            logger.info(f"Gemini extraction complete. Processed {processed_count} chunks successfully, yielding {len(all_extracted_info)} items.")


        elif self.is_aphrodite:
            # --- Aphrodite Path (Batch Call) ---
            prompts_for_service = []
            chunk_mapping = []

            logger.info(f"Preparing prompts for {len(chunks)} chunks (Aphrodite)...")
            for chunk in chunks:
                chunk_text = chunk.get('text', '')
                if not chunk_text.strip(): continue

                messages = self._format_extraction_prompt(
                    chunk_text, schema_definition_str, json_schema_example,
                    primary_key_field, primary_key_description, user_query
                )
                try:
                    # Apply chat template using the loaded tokenizer
                    final_prompt_string = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    prompts_for_service.append(final_prompt_string)
                    chunk_mapping.append(chunk)
                except Exception as template_err:
                    logger.error(f"Error applying chat template for chunk {chunk.get('chunk_id', 'N/A')}: {template_err}", exc_info=True)

            if not prompts_for_service:
                 logger.warning("No valid prompts generated for Aphrodite after applying templates.")
                 return []

            logger.info(f"Sending {len(prompts_for_service)} formatted prompts to Aphrodite service.")
            try:
                # Call Aphrodite's extract_info (which handles batching internally now)
                # Pass the original schema_dict for dynamic Pydantic model creation inside the worker
                all_results_data = []
                aphrodite_batch_size = 512 # Or get from config
                num_aphrodite_batches = (len(prompts_for_service) + aphrodite_batch_size - 1) // aphrodite_batch_size

                for i in range(0, len(prompts_for_service), aphrodite_batch_size):
                    batch_prompts = prompts_for_service[i:i+aphrodite_batch_size]
                    batch_chunks = chunk_mapping[i:i+aphrodite_batch_size]
                    logger.info(f"Processing Aphrodite info extraction batch {i//aphrodite_batch_size + 1}/{num_aphrodite_batches}")

                    # Call extract_info for the current batch
                    # NOTE: Aphrodite's extract_info needs modification to handle multiple prompts
                    # For now, we call it sequentially within the loop as a placeholder.
                    # A true batch implementation would pass all prompts at once.
                    # --- Placeholder Sequential Call ---
                    batch_results_temp = []
                    for idx, single_prompt in enumerate(batch_prompts):
                         single_chunk = batch_chunks[idx]
                         response = self.llm_manager.extract_info(
                             prompt=single_prompt, # Send one prompt
                             schema_definition=schema_dict # Pass original schema dict
                         )
                         batch_results_temp.append((response, single_chunk)) # Store response and chunk
                    # --- End Placeholder ---

                    # Process results from the (placeholder) batch
                    for response, original_chunk in batch_results_temp:
                        if response.get("status") != "success":
                            logger.warning(f"Aphrodite extraction error for chunk {original_chunk.get('chunk_id', 'N/A')}: {response.get('error', 'Unknown error')}")
                            continue # Skip failed chunk

                        result_data = response.get("result", []) # Expecting a list
                        if not isinstance(result_data, list):
                             logger.warning(f"Aphrodite returned non-list result for chunk {original_chunk.get('chunk_id', 'N/A')}: {type(result_data)}. Wrapping.")
                             result_data = [result_data] if isinstance(result_data, dict) else []

                        # Add source info to each item in the list
                        for item in result_data:
                            if isinstance(item, dict):
                                item_with_source = item.copy()
                                item_with_source['_source'] = {
                                    'chunk_id': original_chunk.get('chunk_id', 'unknown'),
                                    'document_id': original_chunk.get('document_id', 'unknown'),
                                    'file_name': original_chunk.get('file_name', 'unknown'),
                                    'page_num': original_chunk.get('page_num', None)
                                }
                                all_results_data.append(item_with_source) # Append directly to final list

                # Update all_extracted_info with results from all batches
                all_extracted_info = all_results_data
                processed_count = sum(1 for item in all_extracted_info if '_source' in item) # Approximate count

                logger.info(f"Aphrodite extraction complete. Processed approx {processed_count} chunks successfully, yielding {len(all_extracted_info)} items.")

            except Exception as e:
                logger.error(f"Critical error during Aphrodite info extraction batch: {e}", exc_info=True)
                return []
        else:
            logger.error("No valid LLM backend configured for information extraction.")
            return []

        return all_extracted_info


    def _format_extraction_prompt(self, text: str, schema_definition: str,
                                      json_schema_example: str, primary_key_field: str,
                                      primary_key_description: str, user_query: str) -> List[Dict[str, str]]:
            """
            Creates the structured message list for the information extraction task.
            (This is used to generate the 'user' content for all backends).

            Args:
                text: Text to extract information from
                schema_definition: String describing the schema fields
                json_schema_example: String showing example JSON output format (list of objects)
                primary_key_field: Name of the primary entity field
                primary_key_description: Description of the primary entity
                user_query: User query describing the extraction task

            Returns:
                A list of dictionaries representing the conversation structure (system, user).
            """
            # System prompt is generic for structured output
            system_prompt = f"""You are an AI assistant that extracts structured information from text according to a specified schema.
You always output valid JSON that conforms exactly to the requested schema ({primary_key_description}).
The output MUST be a JSON list containing objects, even if only one item is found.
You never include additional explanation text in your responses."""

            # User prompt contains all the specifics
            user_prompt = f"""I need you to extract structured information from the provided text.

## EXTRACTION TASK:
{user_query}
Extract data about {primary_key_description} from the text. Each distinct {primary_key_field} instance found should have its own entry (object) in the resulting JSON list.

## OUTPUT SCHEMA:
The data objects in the list should be structured with the following fields:
{schema_definition}

## OUTPUT FORMAT:
Your response should be formatted as valid JSON, specifically a list containing objects that match this structure exactly:
{json_schema_example}

## HANDLING MISSING DATA:
- For string fields: Use "" (empty string) or null for missing data
- For numeric fields: Use null for missing data
- For boolean fields: Use null for missing data

## IMPORTANT INSTRUCTIONS:
- Extract ALL instances of {primary_key_description} entities, even if some fields have missing data.
- Do NOT include any fields not specified in the schema.
- Do NOT include explanations or additional text outside the JSON structure.
- Use only the data explicitly mentioned in the text.
- If you're uncertain about a value, use the missing data convention (null or "").
- The final output MUST be a JSON list `[...]`. If no entities are found, return an empty list `[]`.

## TEXT TO ANALYZE:
{text}
"""
            # For OpenRouter, return messages list. For Gemini/Aphrodite, the user prompt string will be extracted.
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            return messages
