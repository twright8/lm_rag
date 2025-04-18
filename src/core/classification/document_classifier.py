"""
Document classification module for Anti-Corruption RAG System.
Uses LLM-based classification with Aphrodite service to classify document chunks
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
from typing import List, Dict, Any, Optional, Union, Type, Set
import pandas as pd

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))

# Import utils
from src.utils.logger import setup_logger
from src.utils.resource_monitor import log_memory_usage
from src.utils.aphrodite_service import get_service
from transformers import AutoTokenizer
import traceback # For better error logging
# Initialize logger
logger = setup_logger(__name__)

# Load configuration
CONFIG_PATH = ROOT_DIR / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)


class DocumentClassifier:
    """
    LLM-based document classifier for structured classification.
    Uses a persistent Aphrodite service to classify documents based on
    user-defined schemas.
    """

    def __init__(self, model_name=None, debug=False):
        """
        Initialize document classifier.

        Args:
            model_name (str, optional): Name of the model to use for classification
            debug (bool, optional): Enable debugging output
        """
        self.debug = debug
        if model_name:
            self.model_name = model_name
        else:
            self.model_name = CONFIG["models"]["extraction_models"]["text_standard"]

        self.classification_data_path = ROOT_DIR / "data" / "classification"
        self.classification_data_path.mkdir(parents=True, exist_ok=True)

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

        logger.info(f"Initialized DocumentClassifier with model={self.model_name}, debug={debug}")
        log_memory_usage(logger)
    def ensure_model_loaded(self):
        """
        Ensure the designated classification model is loaded in the service.

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
            logger.info(f"Loading designated classification model: {self.model_name}")
            # Load model generically
            if not self.aphrodite_service.load_model(self.model_name):
                logger.error(f"Failed to load classification model {self.model_name}")
                return False
            logger.info(f"Model {self.model_name} loaded successfully for document classification.")
        else:
            logger.info(f"Classification model {self.model_name} already loaded.")

        return True

    def classify_documents(self, chunks: List[Dict[str, Any]], schema: Dict[str, Any],
                           multi_label_fields: Set[str], user_instructions: str) -> List[Dict[str, Any]]:
        """
        Classify document chunks according to the provided schema.

        Args:
            chunks: List of document chunks (must contain 'text')
            schema: Dictionary defining the classification schema with fields and allowed values
            multi_label_fields: Set of field names that accept multiple values
            user_instructions: User-provided instructions or context for classification

        Returns:
            List of classified document dictionaries
        """
        if not chunks:
            logger.warning("No chunks provided for classification")
            return []

        if not self.tokenizer:
             logger.error("Tokenizer not loaded. Cannot perform classification.")
             return []
        if self.tokenizer.chat_template is None:
             logger.error(f"Tokenizer {self.model_name} has no chat template. Cannot format prompts correctly.")
             return []

        if not self.ensure_model_loaded():
            logger.error("Failed to load model for classification")
            return []

        try:
            schema_definition_formatted = ""
            for field_name, field_info in schema.items():
                allowed_values = field_info.get("values", [])
                description = field_info.get("description", "")
                values_str = ", ".join([f'"{val}"' for val in allowed_values])
                multi_label_str = " (allows multiple values)" if field_name in multi_label_fields else ""
                schema_definition_formatted += f"- {field_name}: {description}{multi_label_str}\n"
                schema_definition_formatted += f"  Allowed values: [{values_str}]\n\n"

            example = {}
            for field_name, field_info in schema.items():
                allowed_values = field_info.get("values", [])
                if not allowed_values: continue
                if field_name in multi_label_fields:
                    num_values = min(2, len(allowed_values))
                    if num_values > 0: example[field_name] = random.sample(allowed_values, num_values)
                else:
                    if allowed_values: example[field_name] = allowed_values[0]
            json_example = json.dumps(example, indent=2)

            prompts_for_service = []
            chunk_mapping = [] # Map prompt index back to original chunk

            logger.info(f"Preparing classification prompts for {len(chunks)} chunks...")
            for chunk in chunks:
                chunk_text = chunk.get('text', '')
                if not chunk_text.strip():
                    logger.warning(f"Skipping empty chunk {chunk.get('chunk_id', 'N/A')}")
                    continue

                # 1. Create structured messages
                messages = self._format_classification_prompt(
                    chunk_text, schema_definition_formatted, json_example, user_instructions
                )

                # 2. Apply chat template
                try:
                    final_prompt_string = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True # Add assistant prompt marker
                    )
                    prompts_for_service.append(final_prompt_string)
                    chunk_mapping.append(chunk)
                except Exception as template_err:
                    logger.error(f"Error applying chat template for chunk {chunk.get('chunk_id', 'N/A')}: {template_err}", exc_info=True)
                    # Skip this chunk

            if not prompts_for_service:
                 logger.warning("No valid prompts were generated after applying templates.")
                 return []

            logger.info(f"Sending {len(prompts_for_service)} formatted prompts for classification.")

            # Call the service's classify_chunks method
            # Note: Pass schema and multi_label_fields separately for potential internal validation/guidance
            response = self.aphrodite_service.classify_chunks(
                prompts=prompts_for_service, # List of formatted strings
                schema_definition=schema, # Original schema dict
                multi_label_fields=list(multi_label_fields) # Pass multi-label info
            )

            if response.get("status") != "success":
                error_msg = response.get("error", "Unknown error")
                logger.error(f"Error classifying chunks batch: {error_msg}")
                return [] # Or handle partial results

            results = response.get("results", [])
            if len(results) != len(prompts_for_service):
                logger.warning(f"Result count mismatch: Expected {len(prompts_for_service)}, Got {len(results)}")
                while len(results) < len(prompts_for_service):
                    results.append({"status": "error", "error": "Missing result from service"})

            # Combine classifications with original chunk data
            classified_documents = []
            processed_count = 0
            error_count = 0
            for i, result_data in enumerate(results):
                original_chunk = chunk_mapping[i] # Get the corresponding chunk

                if isinstance(result_data, dict) and result_data.get("status") == "success":
                    classification_data = result_data.get("result", {})
                    classified_doc = {
                        "chunk_id": original_chunk.get("chunk_id", f"chunk_{i}"),
                        "document_id": original_chunk.get("document_id", "unknown"),
                        "file_name": original_chunk.get("file_name", "unknown"),
                        "page_num": original_chunk.get("page_num", None),
                        "text": original_chunk.get("text", ""),
                        "classification": classification_data,
                        # Flatten fields for easier access if needed later
                        **{f"class_{k}": v for k, v in classification_data.items()}
                    }
                    classified_documents.append(classified_doc)
                    processed_count += 1
                else:
                    # Handle error cases
                    error = result_data.get("error", "Unknown classification error")
                    logger.warning(f"Classification error for chunk {original_chunk.get('chunk_id', i)}: {error}")
                    error_count += 1
                    classified_doc = {
                        "chunk_id": original_chunk.get("chunk_id", f"chunk_{i}"),
                        "document_id": original_chunk.get("document_id", "unknown"),
                        "file_name": original_chunk.get("file_name", "unknown"),
                        "page_num": original_chunk.get("page_num", None),
                        "text": original_chunk.get("text", ""),
                        "classification": {"error": error},
                        "class_error": error
                    }
                    classified_documents.append(classified_doc)

            logger.info(f"Classification complete. Success: {processed_count}, Errors: {error_count}. Total results: {len(classified_documents)}")
            return classified_documents

        except Exception as e:
            logger.error(f"Critical error during document classification: {e}", exc_info=True)
            return [] # Return empty list on major failure

    def export_to_csv(self, classified_docs: List[Dict[str, Any]], file_path: str) -> bool:
        """
        Export classification results to CSV.

        Args:
            classified_docs: List of classified documents
            file_path: Path to save the CSV file

        Returns:
            bool: Success status
        """
        try:
            if not classified_docs:
                logger.warning("No classified documents to export")
                return False

            # Create DataFrame from the classified documents
            df = pd.DataFrame(classified_docs)

            # Export to CSV
            df.to_csv(file_path, index=False)
            logger.info(f"Exported {len(classified_docs)} classified documents to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting classification results to CSV: {e}")
            return False