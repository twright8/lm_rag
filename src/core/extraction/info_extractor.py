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
        # Set debug mode
        self.debug = debug

        # Set model name
        if model_name:
            self.model_name = model_name
        else:
            # Use the standard model as default
            self.model_name = CONFIG["models"]["extraction_models"]["text_small"]

        # Storage paths for results
        self.extracted_data_path = ROOT_DIR / "data" / "extracted" / "info"
        self.extracted_data_path.mkdir(parents=True, exist_ok=True)

        # Aphrodite service reference
        self.aphrodite_service = get_service()

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

        # Ensure model is loaded
        if not self.ensure_model_loaded():
            logger.error("Failed to load model for extraction")
            return []

        # Get batch size from config
        batch_size = CONFIG.get("extraction", {}).get("information_extraction", {}).get("batch_size", 1024)
        logger.info(f"Using batch size of {batch_size} from config")

        # Create schema definition for prompt
        schema_definition = ""
        for field_name, field_info in schema_dict.items():
            schema_definition += f"- {field_name}: {field_info['description']} (Type: {field_info['type']})\n"

        # Create JSON schema example with simple values
        json_schema_example = {
            field_name: f"<{field_info['type']} value>"
            for field_name, field_info in schema_dict.items()
        }
        json_schema_example = json.dumps([json_schema_example], indent=2)

        # Process chunks and extract information
        all_extracted_info = []

        try:
            # Process chunks
            for i in range(0, len(chunks), batch_size):
                # Get chunk batch
                batch_chunks = chunks[i:i+batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size} ({len(batch_chunks)} chunks)")

                # Process each chunk
                for chunk in batch_chunks:
                    chunk_id = chunk.get('chunk_id', f'unknown-{len(all_extracted_info)}')
                    text = chunk.get('text', '')

                    # Format the extraction prompt
                    formatted_prompt = self._format_extraction_prompt(
                        text, schema_definition, json_schema_example,
                        primary_key_field, primary_key_description, user_query
                    )

                    # Format as ChatML for Aphrodite
                    chat_prompt = f"""<|im_start|>system
You are an AI assistant that extracts structured information from text according to a specified schema.
You always output valid JSON that conforms exactly to the requested schema.
You never include additional explanation text in your responses.<|im_end|>
<|im_start|>user
{formatted_prompt}<|im_end|>
<|im_start|>assistant
"""

                    # Call the new extract_info method with the dynamic schema
                    response = self.aphrodite_service.extract_info(
                        prompt=chat_prompt,
                        schema_definition=schema_dict
                    )

                    if response.get("status") != "success":
                        error_msg = response.get("error", "Unknown error")
                        logger.error(f"Error extracting info from chunk {chunk_id}: {error_msg}")
                        continue

                    # Process results

                    # Ensure result is a list
                    # Process results
                    result_data = response.get("result", [])

                    # Ensure result is a list of dictionaries
                    if not isinstance(result_data, list):
                        if isinstance(result_data, dict):
                            # Single object
                            result_data = [result_data]
                        else:
                            logger.warning(f"Unexpected result type: {type(result_data)}")
                            result_data = []

                    # Process each individual item (these are the actual rows)
                    for item in result_data:
                        if len (item['items']) !=0:
                            for x in item['items']:
                                print(x)
                            # For each item, add the source information
                                if isinstance(x, dict):
                                    # Add source information
                                    item_with_source = x.copy()
                                    tmp = {
                                        'chunk_id': chunk.get('chunk_id', 'unknown'),
                                        'document_id': chunk.get('document_id', 'unknown'),
                                        'file_name': chunk.get('file_name', 'unknown'),
                                        'page_num': chunk.get('page_num', None)
                                    }
                                    x.update(tmp)
                                    all_extracted_info.append(item_with_source)
                                else:
                                    logger.warning(f"Expected dict item, got {type(item)}: {item}")
                        else:
                            pass
                    logger.info(f"Extracted {len(result_data)} items from chunk {chunk_id}")
            return all_extracted_info

        except Exception as e:
            logger.error(f"Error in information extraction: {e}", exc_info=True)
            return []


    def _format_extraction_prompt(self, text: str, schema_definition: str,
                                 json_schema_example: str, primary_key_field: str,
                                 primary_key_description: str, user_query: str) -> str:
        """
        Format the extraction prompt for the LLM.

        Args:
            text: Text to extract information from
            schema_definition: Definition of the schema fields
            json_schema_example: Example of the JSON output format
            primary_key_field: Name of the primary entity field
            primary_key_description: Description of the primary entity
            user_query: User query describing the extraction task

        Returns:
            str: Formatted prompt
        """
        # Format the full prompt - using the exact template specified
        prompt = f"""I need you to extract structured information from the provided text.

## EXTRACTION TASK:
Extract data about {primary_key_description} from the text. Each {primary_key_field} should have its own entry in the resulting table.

## OUTPUT SCHEMA:
The data should be structured with the following fields:
{schema_definition}

## OUTPUT FORMAT:
Your response should be formatted as valid JSON that matches this structure exactly:
{json_schema_example}

## HANDLING MISSING DATA:
- For string fields: Use "" (empty string) for missing data
- For numeric fields: Use null for missing data
- For boolean fields: Use null for missing data

## IMPORTANT INSTRUCTIONS:
- Extract ALL instances of {primary_key_field} entities, even if some fields have missing data
- Do NOT include any fields not specified in the schema
- Do NOT include explanations or additional text outside the JSON structure
- Use only the data explicitly mentioned in the text
- If you're uncertain about a value, use the missing data convention

## USER QUERY:
{user_query}

## TEXT TO ANALYZE:
{text}
"""
        return prompt