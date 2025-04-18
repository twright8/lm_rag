"""
Document loading module for Anti-Corruption RAG System.
Handles PDF, DOCX, TXT, CSV, and XLSX files.
Uses Docling for PDF, DOCX, TXT conversion to Markdown, including OCR.
Spreadsheet files are handled separately using pandas.
"""
import sys
import os
from pathlib import Path
import uuid
import yaml
import pandas as pd
import torch # Add this import
from src.utils.resource_monitor import log_memory_usage
import gc
from typing import List, Dict, Any, Union, Optional

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))

# Import utils
from src.utils.logger import setup_logger

# Import Docling components
try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat, ConversionStatus
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice
    )
    from docling_core.types.doc import ImageRefMode # Correct import path
    DOCLING_AVAILABLE = True
except ImportError as e:
    setup_logger(__name__).warning(f"Docling library not found or import error: {e}. Docling functionality disabled.")
    DOCLING_AVAILABLE = False
    # Define dummy classes if Docling is not available to avoid runtime errors on load
    class DocumentConverter: pass
    class PdfFormatOption: pass
    class InputFormat: PDF = "PDF"; DOCX = "DOCX"; TXT = "TXT" # Dummy values
    class ConversionStatus: SUCCESS = "SUCCESS"; PARTIAL_SUCCESS = "PARTIAL_SUCCESS"; FAILURE = "FAILURE"
    class PdfPipelineOptions: pass
    class AcceleratorOptions: pass
    class AcceleratorDevice: AUTO = "AUTO"; CPU = "CPU"; CUDA = "CUDA"; MPS = "MPS"
    class ImageRefMode: PLACEHOLDER = "PLACEHOLDER"

# Initialize logger
logger = setup_logger(__name__)

# Load configuration
CONFIG_PATH = ROOT_DIR / "config.yaml"
try:
    with open(CONFIG_PATH, "r") as f:
        CONFIG = yaml.safe_load(f)
except FileNotFoundError:
    logger.error(f"Configuration file not found at {CONFIG_PATH}. Using default settings.")
    CONFIG = {}
except Exception as e:
    logger.error(f"Error loading configuration: {e}. Using default settings.")
    CONFIG = {}

class DocumentLoader:
    """
    Document loader using Docling for conversion and OCR (non-spreadsheets)
    and pandas for spreadsheets.
    """

    def __init__(self):
        """
        Initialize DocumentLoader. Configures Docling converter if available.
        """
        logger.info("Initializing DocumentLoader")
        self.docling_converter = None

        if not DOCLING_AVAILABLE:
            logger.error("Docling is not available. Document conversion for PDF, DOCX, TXT will fail.")
            return

        try:
            # Configure Docling based on config.yaml
            docling_config = CONFIG.get("docling", {})
            artifacts_path = docling_config.get("artifacts_path", None) # For offline models
            accel_device_str = docling_config.get("accelerator_device", "AUTO").upper()
            num_threads = docling_config.get("num_threads", -1)

            # Map string to AcceleratorDevice enum
            accel_device_map = {
                "AUTO": AcceleratorDevice.AUTO,
                "CPU": AcceleratorDevice.CPU,
                "CUDA": AcceleratorDevice.CUDA,
                "MPS": AcceleratorDevice.MPS,
            }
            accel_device = accel_device_map.get(accel_device_str, AcceleratorDevice.AUTO)
            if accel_device_str not in accel_device_map:
                 logger.warning(f"Invalid accelerator_device '{accel_device_str}' in config. Falling back to AUTO.")

            logger.info(f"Configuring Docling: Artifacts Path='{artifacts_path}', Device='{accel_device}', Threads={num_threads}")

            accelerator_options = AcceleratorOptions(
                device=accel_device,
                num_threads=num_threads
            )

            # Configure PDF pipeline options (can add more from config if needed)
            # OCR is typically enabled by default in Docling's pipeline
            pipeline_options = PdfPipelineOptions(
                artifacts_path=artifacts_path,
                accelerator_options=accelerator_options,
                do_ocr=True, # Explicitly enable OCR (often default)
                # ocr_options=... # Add specific OCR options if needed, e.g., language
                do_table_structure=True, # Enable table extraction
                generate_page_images=False # Don't need page images for Markdown export
            )
            # Add more pipeline options from config if necessary

            # Create the converter instance
            # We configure only for PDF here, Docling handles others automatically
            self.docling_converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                    # Docling will use default options for DOCX, TXT etc.
                }
            )
            logger.info("Docling DocumentConverter initialized successfully.")

        except Exception as e:
            logger.error(f"Failed to initialize Docling DocumentConverter: {e}", exc_info=True)
            self.docling_converter = None # Ensure it's None if init fails
    def shutdown(self):
        """
        Explicitly unloads the Docling converter and attempts to free memory,
        including clearing the CUDA cache if CUDA is available.
        Call this after you are finished processing all documents with this loader instance.
        """
        logger.info("===== SHUTTING DOWN DocumentLoader and Unloading Docling Models =====")
        log_memory_usage(logger) # Optional: log memory

        if not DOCLING_AVAILABLE:
            logger.info("Docling was not available, nothing to unload.")
            return

        if self.docling_converter is not None:
            logger.info("Unloading Docling DocumentConverter instance...")
            try:
                # Step 1: Remove the reference to the converter object
                # This makes it eligible for garbage collection.
                del self.docling_converter
                self.docling_converter = None
                logger.info("Docling converter reference removed.")

                # Step 2: Explicitly run garbage collection
                # This encourages Python to reclaim the memory sooner.
                logger.info("Running garbage collection...")
                gc.collect()
                logger.info("Garbage collection finished.")

                # Step 3: Clear PyTorch CUDA cache if CUDA is available
                # This is crucial for freeing GPU memory managed by PyTorch,
                # assuming Docling uses PyTorch for its CUDA backend.
                if torch.cuda.is_available():
                    logger.info("CUDA is available. Clearing PyTorch CUDA cache...")
                    torch.cuda.empty_cache()
                    logger.info("PyTorch CUDA cache cleared.")
                else:
                    logger.info("CUDA not available or Docling not using CUDA backend via PyTorch, skipping CUDA cache clear.")

                logger.info("Docling models should now be unloaded from memory.")

            except Exception as e:
                logger.error(f"An error occurred during Docling shutdown: {e}", exc_info=True)
                # Ensure the reference is cleared even if other steps fail
                self.docling_converter = None
        else:
            logger.info("Docling converter was already None or not initialized.")

        log_memory_usage(logger) # Optional: log memory
        logger.info("DocumentLoader shutdown complete.")
    def load_document_with_options(self, file_path: Union[str, Path], options: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        Load a document with custom options, primarily for spreadsheets.
        Routes to specific handlers based on file type.

        Args:
            file_path: Path to the document file
            options: Dictionary of options (used for spreadsheets)

        Returns:
            dict: Document data including content, or None on failure.
        """
        file_path = Path(file_path)
        file_name = file_path.name
        file_ext = file_path.suffix.lower()

        logger.info(f"Loading document: {file_name} (Type: {file_ext})")

        # --- Spreadsheet Handling ---
        if file_ext in [".csv", ".xlsx", ".xls"]:
            logger.info(f"Processing spreadsheet: {file_name}")
            # Initialize basic document data structure
            document_id = str(uuid.uuid4())
            document_data = self._initialize_document_data(document_id, file_name, file_path, file_ext)
            try:
                if options and 'selected_columns' in options:
                    logger.info("Using spreadsheet options for processing.")
                    return self._process_spreadsheet_with_options(file_path, document_data, options)
                else:
                    logger.info("Processing spreadsheet with default logic (all columns).")
                    return self._process_spreadsheet(file_path, document_data)
            except Exception as e:
                logger.error(f"Error processing spreadsheet {file_name}: {e}", exc_info=True)
                return None

        # --- Docling Handling (PDF, DOCX, TXT, etc.) ---
        elif file_ext in [".pdf", ".docx", ".txt", ".html", ".md"]: # Add other Docling supported types
             if not self.docling_converter:
                 logger.error(f"Docling converter not available. Cannot process {file_name}.")
                 return None
             logger.info(f"Processing with Docling: {file_name}")
             try:
                 return self._process_with_docling(file_path)
             except Exception as e:
                 logger.error(f"Error processing {file_name} with Docling: {e}", exc_info=True)
                 return None

        # --- Unsupported Type ---
        else:
            logger.warning(f"Unsupported file type: {file_ext} for file {file_name}")
            return None

    def load_document(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """
        Load a document from the given file path (wrapper for options).

        Args:
            file_path: Path to the document file

        Returns:
            dict: Document data including content, or None on failure.
        """
        # Call the main loading function without specific options
        return self.load_document_with_options(file_path, options=None)

    def _initialize_document_data(self, doc_id: str, file_name: str, file_path: Path, file_ext: str) -> Dict[str, Any]:
        """Helper to create the basic document data dictionary."""
        return {
            "document_id": doc_id,
            "file_name": file_name,
            "file_path": str(file_path),
            "file_type": file_ext,
            "content": [],
            "metadata": {
                "page_count": 0,
                "word_count": 0,
            }
        }

    def _process_with_docling(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Process a single document (PDF, DOCX, TXT) using Docling.

        Args:
            file_path: Path to the document file.

        Returns:
            dict: Document data dictionary or None on failure.
        """
        if not self.docling_converter:
            logger.error("Docling converter is not initialized.")
            return None

        file_name = file_path.name
        file_ext = file_path.suffix.lower()
        document_id = str(uuid.uuid4())
        document_data = self._initialize_document_data(document_id, file_name, file_path, file_ext)

        try:
            logger.info(f"Starting Docling conversion for {file_name}...")
            conv_result = self.docling_converter.convert(file_path, raises_on_error=False)

            if conv_result.status not in [ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS]:
                error_msg = f"Docling conversion failed for {file_name}. Status: {conv_result.status}."
                if conv_result.errors:
                    error_details = "; ".join([e.error_message for e in conv_result.errors])
                    error_msg += f" Errors: {error_details}"
                logger.error(error_msg)
                return None

            if conv_result.status == ConversionStatus.PARTIAL_SUCCESS:
                logger.warning(
                    f"Docling conversion partially successful for {file_name}. Errors: {[e.error_message for e in conv_result.errors]}")

            doc = conv_result.document
            if not doc:
                logger.error(
                    f"Docling conversion status was {conv_result.status} but document object is missing for {file_name}.")
                return None

            logger.info(f"Exporting {file_name} content to Markdown...")
            markdown_content = doc.export_to_markdown(
                image_mode=ImageRefMode.PLACEHOLDER,
                strict_text=False
            )

            document_data["content"] = [{
                "text": markdown_content,
                "page_num": None,
                "is_spreadsheet_row": False,
                "metadata": {
                    "docling_conversion_status": str(conv_result.status),  # Ensure status is string
                }
            }]

            # --- CORRECTED METADATA UPDATE ---
            page_count_value = 1  # Default value
            if hasattr(doc, 'num_pages'):
                try:
                    # Access the property/attribute or call the method
                    num_pages_attr = getattr(doc, 'num_pages')
                    if callable(num_pages_attr):
                        retrieved_value = num_pages_attr()  # Call if it's a method
                        logger.debug(f"Called doc.num_pages() method, got: {retrieved_value}")
                    else:
                        retrieved_value = num_pages_attr  # Access property/attribute value
                        logger.debug(f"Accessed doc.num_pages attribute/property, got: {retrieved_value}")

                    # Ensure the retrieved value is an integer
                    if isinstance(retrieved_value, int):
                        page_count_value = retrieved_value
                    else:
                        # Attempt conversion if possible, otherwise warn and use default
                        logger.warning(
                            f"doc.num_pages returned non-integer type '{type(retrieved_value).__name__}'. Attempting conversion.")
                        try:
                            page_count_value = int(retrieved_value)
                        except (ValueError, TypeError):
                            logger.error(
                                f"Could not convert doc.num_pages value '{retrieved_value}' to int. Using default 1.")
                            page_count_value = 1  # Fallback to default if conversion fails

                except Exception as e:
                    logger.error(f"Error accessing or calling doc.num_pages: {e}. Using default 1.", exc_info=True)
                    page_count_value = 1  # Fallback to default on any error
            else:
                logger.warning("Docling document object does not have 'num_pages' attribute. Using default 1.")
                page_count_value = 1

            document_data["metadata"]["page_count"] = page_count_value  # Assign the integer value
            document_data["metadata"]["word_count"] = len(markdown_content.split())  # Approximate word count
            # --- END CORRECTION ---

            logger.info(
                f"Docling processing successful for {file_name}. Word count: {document_data['metadata']['word_count']}, Page count: {page_count_value}")
            return document_data

        except Exception as e:
            logger.error(f"Unexpected error during Docling processing for {file_name}: {e}", exc_info=True)
            return None

    # --- Spreadsheet Processing Methods (Unchanged from original logic) ---

    def _process_spreadsheet_with_options(self, file_path: Path, document_data: Dict[str, Any],
                                          options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a spreadsheet with column selection options.
        Uses a fixed separator format with pipes and spaces.
        (Logic remains the same as provided in the prompt)
        """
        try:
            file_ext = file_path.suffix.lower()
            selected_columns = options.get('selected_columns', [])
            separator = "  |  " # Fixed separator
            include_column_names = True # Always include

            if file_ext == ".csv":
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)

            logger.info(f"Loaded spreadsheet {file_path.name} with {len(df)} rows and {len(df.columns)} columns")

            # Filter columns if selections provided and valid
            valid_selection = selected_columns and all(col in df.columns for col in selected_columns)
            if valid_selection:
                df = df[selected_columns]
                logger.info(f"Filtered to {len(selected_columns)} selected columns for {file_path.name}")
            else:
                if selected_columns: # Log if selection was attempted but invalid
                     logger.warning(f"Invalid column selection for {file_path.name}. Using all columns.")
                selected_columns = df.columns.tolist() # Use all columns if no valid selection
                logger.info(f"Using all {len(selected_columns)} columns for {file_path.name}")


            document_data["metadata"]["rows"] = len(df)
            document_data["metadata"]["columns"] = len(df.columns)
            document_data["metadata"]["column_names"] = selected_columns

            for idx, row in df.iterrows():
                fields = [f"{col}: {str(val)}" for col, val in row[selected_columns].items()]
                row_text = separator.join(fields)

                content_item = {
                    "text": row_text,
                    "row_idx": idx,
                    "is_spreadsheet_row": True,
                    "metadata": {
                        "row_idx": idx,
                        "file_name": document_data["file_name"],
                        "spreadsheet_columns": selected_columns
                    },
                    "images": [] # Spreadsheets don't have images in this context
                }
                document_data["content"].append(content_item)

            document_data["metadata"]["page_count"] = len(df) # Treat rows as pages
            total_words = sum(len(item.get("text", "").split()) for item in document_data["content"])
            document_data["metadata"]["word_count"] = total_words

            logger.info(f"Processed spreadsheet {file_path.name} into {len(df)} row-based content items")
            return document_data

        except Exception as e:
            logger.error(f"Error processing spreadsheet with options {file_path.name}: {e}", exc_info=True)
            raise # Re-raise to be caught by the calling function

    def _process_spreadsheet(self, file_path: Path, document_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a spreadsheet file (CSV, XLSX, XLS) without specific column options.
        Each row becomes a separate content item.
        (Logic remains the same as provided in the prompt)
        """
        try:
            file_ext = file_path.suffix.lower()

            if file_ext == ".csv":
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)

            logger.info(f"Loaded spreadsheet {file_path.name} with {len(df)} rows and {len(df.columns)} columns")

            document_data["metadata"]["rows"] = len(df)
            document_data["metadata"]["columns"] = len(df.columns)
            document_data["metadata"]["column_names"] = df.columns.tolist()

            for idx, row in df.iterrows():
                # Format row: "ColumnName1: Value1 | ColumnName2: Value2 | ..."
                row_text = " | ".join([f"{col}: {str(val)}" for col, val in row.items()])

                content_item = {
                    "text": row_text,
                    "row_idx": idx,
                    "is_spreadsheet_row": True,
                    "metadata": {
                        "file_name": document_data["file_name"],
                        "row_idx": idx,
                        "spreadsheet_columns": df.columns.tolist()
                    },
                    "images": []
                }
                document_data["content"].append(content_item)

            document_data["metadata"]["page_count"] = len(df) # Treat rows as pages
            total_words = sum(len(item.get("text", "").split()) for item in document_data["content"])
            document_data["metadata"]["word_count"] = total_words

            logger.info(f"Processed spreadsheet {file_path.name} into {len(df)} row-based content items")
            return document_data

        except Exception as e:
            logger.error(f"Error processing spreadsheet {file_path.name}: {e}", exc_info=True)
            raise # Re-raise