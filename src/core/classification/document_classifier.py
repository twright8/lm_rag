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
        # Set debug mode
        self.debug = debug

        # Set model name
        if model_name:
            self.model_name = model_name
        else:
            # Use the standard model as default
            self.model_name = CONFIG["models"]["extraction_models"]["text_standard"]

        # Storage paths for results
        self.classification_data_path = ROOT_DIR / "data" / "classification"
        self.classification_data_path.mkdir(parents=True, exist_ok=True)

        # Aphrodite service reference
        self.aphrodite_service = get_service()

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

        # Ensure model is loaded
        if not self.ensure_model_loaded():
            logger.error("Failed to load model for classification")
            return []

        try:
            # Create formatted schema definition for prompt
            schema_definition_formatted = ""
            for field_name, field_info in schema.items():
                allowed_values = field_info.get("values", [])
                description = field_info.get("description", "")

                values_str = ", ".join([f'"{val}"' for val in allowed_values])
                multi_label_str = " (allows multiple values)" if field_name in multi_label_fields else ""

                schema_definition_formatted += f"- {field_name}: {description}{multi_label_str}\n"
                schema_definition_formatted += f"  Allowed values: [{values_str}]\n\n"

            # Create JSON example based on schema and multi-label configuration
            example = {}
            for field_name, field_info in schema.items():
                allowed_values = field_info.get("values", [])
                if not allowed_values:
                    continue

                if field_name in multi_label_fields:
                    # For multi-label fields, choose 1-2 random values
                    num_values = min(2, len(allowed_values))
                    if num_values > 0:
                        example[field_name] = random.sample(allowed_values, num_values)
                else:
                    # For single-label fields, choose one value
                    if allowed_values:
                        example[field_name] = allowed_values[0]

            # Format the example JSON
            json_example = json.dumps(example, indent=2)

            # Prepare prompts for all chunks
            prompts = []
            for chunk in chunks:
                chunk_text = chunk.get('text', '')
                if not chunk_text.strip():
                    continue

                # Format the prompt for classification
                formatted_prompt = self._format_classification_prompt(
                    chunk_text, schema_definition_formatted, json_example, user_instructions
                )

                # Format as ChatML for Aphrodite
                chat_prompt = f"""<|im_start|>system
You are an expert document classifier. Your task is to analyze the provided text and categorize it according to the specified schema.
Always classify according to the schema exactly, using only the allowed values.
Provide your response in valid JSON format matching the required schema.
<|im_end|>
<|im_start|>user
{formatted_prompt}<|im_end|>
<|im_start|>assistant
"""
                prompts.append(chat_prompt)

            # Process chunks in batches
            logger.info(f"Processing {len(prompts)} classification prompts")

            # Call the classify_chunks method on the service
            response = self.aphrodite_service.classify_chunks(
                prompts=prompts,
                schema_definition=schema,
                multi_label_fields=list(multi_label_fields)
            )

            if response.get("status") != "success":
                error_msg = response.get("error", "Unknown error")
                logger.error(f"Error classifying chunks: {error_msg}")
                return []

            # Process classification results
            results = response.get("results", [])
            if len(results) != len(chunks):
                logger.warning(f"Mismatch between number of results ({len(results)}) and chunks ({len(chunks)})")

            # Combine classifications with original chunk data
            classified_documents = []
            for i, (chunk, result_data) in enumerate(zip(chunks, results)):
                if result_data.get("status") == "success":
                    # Extract the classification data
                    classification_data = result_data.get("result", {})

                    # Combine with chunk metadata
                    classified_doc = {
                        "chunk_id": chunk.get("chunk_id", f"chunk_{i}"),
                        "document_id": chunk.get("document_id", "unknown"),
                        "file_name": chunk.get("file_name", "unknown"),
                        "page_num": chunk.get("page_num", None),
                        "text": chunk.get("text", ""),
                        "classification": classification_data,
                        # Flatten classification fields for easier data handling
                        **{f"class_{k}": v for k, v in classification_data.items()}
                    }

                    classified_documents.append(classified_doc)
                else:
                    # Handle error cases
                    error = result_data.get("error", "Unknown classification error")
                    logger.warning(f"Classification error for chunk {i}: {error}")

                    # Add the chunk with error information
                    classified_doc = {
                        "chunk_id": chunk.get("chunk_id", f"chunk_{i}"),
                        "document_id": chunk.get("document_id", "unknown"),
                        "file_name": chunk.get("file_name", "unknown"),
                        "page_num": chunk.get("page_num", None),
                        "text": chunk.get("text", ""),
                        "classification": {"error": error},
                        "class_error": error
                    }

                    classified_documents.append(classified_doc)

            logger.info(f"Successfully classified {len(classified_documents)} chunks")
            return classified_documents

        except Exception as e:
            logger.error(f"Error in document classification: {e}", exc_info=True)
            return []

    def _format_classification_prompt(self, text: str, schema_definition: str,
                                      json_example: str, user_instructions: str) -> str:
        """
        Format the classification prompt for the LLM.

        Args:
            text: Text to classify
            schema_definition: Formatted schema definition
            json_example: Example JSON output
            user_instructions: User-provided instructions

        Returns:
            str: Formatted prompt
        """
        # Format the full prompt
        prompt = f"""I need you to classify the provided text according to the specified schema.

## CLASSIFICATION SCHEMA:
{schema_definition}

## EXPECTED OUTPUT FORMAT:
The classification should be provided as valid JSON that matches this example structure:
{json_example}

## USER INSTRUCTIONS:
{user_instructions}

## TEXT TO CLASSIFY:
{text}

Please classify this text according to the schema. Respond ONLY with valid JSON.
"""
        return prompt

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