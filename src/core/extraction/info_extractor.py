"""
Information extraction module for Anti-Corruption RAG System.
Uses LLM-based extraction with Aphrodite service to extract structured data
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
from typing import List, Dict, Any, Optional, Union, Type
import pandas as pd

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))

# Import utils
from src.utils.logger import setup_logger
from src.utils.resource_monitor import log_memory_usage
from src.utils.aphrodite_service import get_service
from transformers import AutoTokenizer
import traceback
# Initialize logger
logger = setup_logger(__name__)

# Load configuration
CONFIG_PATH = ROOT_DIR / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

class InfoExtractor:
    """
    LLM-based information extractor for structured data.
    Uses a persistent Aphrodite service to extract information based on
    user-defined schemas.
    """

    def __init__(self, model_name=None, debug=False):
        """
        Initialize information extractor.

        Args:
            model_name (str, optional): Name of the model to use for extraction
            debug (bool, optional): Enable debugging output
        """
        self.debug = debug
        if model_name:
            self.model_name = model_name
        else:
            self.model_name = CONFIG["models"]["extraction_models"]["text_small"]

        self.extracted_data_path = ROOT_DIR / "data" / "extracted" / "info"
        self.extracted_data_path.mkdir(parents=True, exist_ok=True)

        self.aphrodite_service = get_service()

        # --- NEW: Load Tokenizer ---
        self.tokenizer = None
        try:
            logger.info(f"Loading tokenizer for model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            if self.tokenizer.chat_template is None:
                logger.warning(
                    f"Tokenizer for {self.model_name} does not have a chat_template defined. Falling back to manual formatting (potential errors).")
            else:
                logger.info(f"Tokenizer for {self.model_name} loaded successfully with chat template.")
        except Exception as e:
            logger.error(f"Failed to load tokenizer for {self.model_name}: {e}", exc_info=True)

        logger.info(f"Initialized InfoExtractor with model={self.model_name}, debug={debug}")
        log_memory_usage(logger)

    def ensure_model_loaded(self):
        """
        Ensure the designated extraction model is loaded in the service.

        Returns:
            bool: Success status
        """
        # Check if the service is running
        if not self.aphrodite_service.is_running():
            logger.info("Aphrodite service not running, starting it")
            if not self.aphrodite_service.start():
                logger.error("Failed to start Aphrodite service")
                return False

        # Check if the correct model is loaded
        status = self.aphrodite_service.get_status()
        if not status.get("model_loaded", False) or status.get("current_model") != self.model_name:
            logger.info(f"Loading designated extraction model: {self.model_name}")
            # Load model generically
            if not self.aphrodite_service.load_model(self.model_name):
                logger.error(f"Failed to load extraction model {self.model_name}")
                return False
            logger.info(f"Model {self.model_name} loaded successfully for information extraction.")
        else:
            logger.info(f"Extraction model {self.model_name} already loaded.")

        return True

    def extract_information(self, chunks: List[Dict[str, Any]], schema_dict: Dict[str, Any],
                           primary_key_field: str, primary_key_description: str, user_query: str) -> List[Dict[str, Any]]:
        """
        Extract structured information from chunks based on user-defined schema.

        Args:
            chunks: List of document chunks (must contain 'text')
            schema_dict: Dictionary defining the extraction schema
            primary_key_field: Name of the primary entity field (e.g., 'company')
            primary_key_description: Description of the primary entity
            user_query: User query describing the extraction task

        Returns:
            List of extracted information dictionaries
        """
        if not chunks:
            logger.warning("No chunks provided for extraction")
            return []

        if not self.tokenizer:
             logger.error("Tokenizer not loaded. Cannot perform extraction.")
             return []
        if self.tokenizer.chat_template is None:
             logger.error(f"Tokenizer {self.model_name} has no chat template. Cannot format prompts correctly.")
             return []

        if not self.ensure_model_loaded():
            logger.error("Failed to load model for extraction")
            return []

        batch_size = CONFIG.get("extraction", {}).get("information_extraction", {}).get("batch_size", 1024)
        logger.info(f"Using batch size of {batch_size} from config")

        schema_definition = ""
        for field_name, field_info in schema_dict.items():
            schema_definition += f"- {field_name}: {field_info['description']} (Type: {field_info['type']})\n"

        json_schema_example_obj = {
            field_name: f"<{field_info['type']} value>"
            for field_name, field_info in schema_dict.items()
        }
        json_schema_example = json.dumps([json_schema_example_obj], indent=2) # Example is a list containing one object

        all_extracted_info = []
        prompts_for_service = []
        chunk_mapping = [] # Store original chunk info corresponding to each prompt sent

        logger.info(f"Preparing prompts for {len(chunks)} chunks...")
        for chunk in chunks:
            chunk_text = chunk.get('text', '')
            if not chunk_text.strip():
                logger.warning(f"Skipping empty chunk {chunk.get('chunk_id', 'N/A')}")
                continue

            # 1. Create structured messages
            messages = self._format_extraction_prompt(
                chunk_text, schema_definition, json_schema_example,
                primary_key_field, primary_key_description, user_query
            )

            # 2. Apply chat template
            try:
                final_prompt_string = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True # Add assistant prompt marker
                )
                prompts_for_service.append(final_prompt_string)
                chunk_mapping.append(chunk) # Keep track of the chunk this prompt belongs to
            except Exception as template_err:
                logger.error(f"Error applying chat template for chunk {chunk.get('chunk_id', 'N/A')}: {template_err}", exc_info=True)
                # Skip this chunk

        if not prompts_for_service:
             logger.warning("No valid prompts were generated after applying templates.")
             return []

        logger.info(f"Sending {len(prompts_for_service)} formatted prompts for information extraction.")

        try:
            # Call the service's extract_info method (assuming it handles batches)
            # Note: The service method still needs the schema_dict for potential internal validation/guidance
            response = self.aphrodite_service.extract_info(
                prompts=prompts_for_service, # Send list of formatted strings
                schema_definition=schema_dict # Pass schema separately
            )

            if response.get("status") != "success":
                error_msg = response.get("error", "Unknown error")
                logger.error(f"Error extracting info batch: {error_msg}")
                return [] # Or handle partial results if possible

            results = response.get("results", [])
            if len(results) != len(prompts_for_service):
                logger.warning(f"Result count mismatch: Expected {len(prompts_for_service)}, Got {len(results)}")
                # Pad results if necessary
                while len(results) < len(prompts_for_service):
                    results.append({"error": "Missing result from service"})

            # Process results and map back to chunks
            processed_count = 0
            for i, result_data in enumerate(results):
                original_chunk = chunk_mapping[i] # Get the corresponding chunk

                if isinstance(result_data, dict) and result_data.get("error"):
                     logger.warning(f"Received error for chunk {original_chunk.get('chunk_id', i)}: {result_data['error']}")
                     continue

                # The service's extract_info should return a list of extracted items for each prompt
                # Ensure result_data is a list (of extracted items for this chunk)
                if not isinstance(result_data, list):
                     if isinstance(result_data, dict): # Handle case where service might return single dict instead of list
                          logger.warning(f"Expected list result for chunk {original_chunk.get('chunk_id', i)}, got dict. Wrapping in list.")
                          result_data = [result_data]
                     else:
                          logger.warning(f"Unexpected result type for chunk {original_chunk.get('chunk_id', i)}: {type(result_data)}. Content: {str(result_data)[:100]}")
                          continue # Skip this chunk's result

                # Process each extracted item (row) for the current chunk
                items_extracted_from_chunk = 0
                for item in result_data:
                    if isinstance(item, dict):
                        item_with_source = item.copy()
                        # Add source metadata from the original chunk
                        item_with_source['_source'] = {
                            'chunk_id': original_chunk.get('chunk_id', 'unknown'),
                            'document_id': original_chunk.get('document_id', 'unknown'),
                            'file_name': original_chunk.get('file_name', 'unknown'),
                            'page_num': original_chunk.get('page_num', None)
                        }
                        all_extracted_info.append(item_with_source)
                        items_extracted_from_chunk += 1
                    else:
                        logger.warning(f"Expected dict item within result list for chunk {original_chunk.get('chunk_id', i)}, got {type(item)}: {item}")

                if items_extracted_from_chunk > 0:
                     processed_count += 1
                     if self.debug: logger.debug(f"Extracted {items_extracted_from_chunk} items from chunk {original_chunk.get('chunk_id', i)}")


            logger.info(f"Successfully processed results for {processed_count} chunks, yielding {len(all_extracted_info)} total items.")
            return all_extracted_info

        except Exception as e:
            logger.error(f"Critical error during information extraction batch processing: {e}", exc_info=True)
            return [] # Return empty list on major failure
    def _format_extraction_prompt(self, text: str, schema_definition: str,
                                      json_schema_example: str, primary_key_field: str,
                                      primary_key_description: str, user_query: str) -> List[Dict[str, str]]:
            """
            Creates the structured message list for the information extraction task.

            Args:
                text: Text to extract information from
                schema_definition: Definition of the schema fields
                json_schema_example: Example of the JSON output format
                primary_key_field: Name of the primary entity field
                primary_key_description: Description of the primary entity
                user_query: User query describing the extraction task

            Returns:
                A list of dictionaries representing the conversation structure.
            """
            system_prompt = f"""You are an AI assistant that extracts structured information from text according to a specified schema.
    You always output valid JSON that conforms exactly to the requested schema ({primary_key_description}).
    You never include additional explanation text in your responses."""

            user_prompt = f"""I need you to extract structured information from the provided text.

    ## EXTRACTION TASK:
    Extract data about {primary_key_description} from the text. Each {primary_key_field} should have its own entry in the resulting table.

    ## OUTPUT SCHEMA:
    The data should be structured with the following fields:
    {schema_definition}

    ## OUTPUT FORMAT:
    Your response should be formatted as valid JSON, specifically a list containing objects that match this structure exactly:
    {json_schema_example}

    ## HANDLING MISSING DATA:
    - For string fields: Use "" (empty string) for missing data
    - For numeric fields: Use null for missing data
    - For boolean fields: Use null for missing data

    ## IMPORTANT INSTRUCTIONS:
    - Extract ALL instances of {primary_key_description} entities, even if some fields have missing data.
    - Do NOT include any fields not specified in the schema.
    - Do NOT include explanations or additional text outside the JSON structure.
    - Use only the data explicitly mentioned in the text.
    - If you're uncertain about a value, use the missing data convention.
    - The final output MUST be a JSON list `[...]`.

    ## USER QUERY:
    {user_query}

    ## TEXT TO ANALYZE:
    {text}
    """
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            return messages