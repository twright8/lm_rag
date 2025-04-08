"""
Document loading module for Anti-Corruption RAG System.
Handles PDF, DOCX, TXT, CSV, and XLSX files.
"""
import sys
import os
from pathlib import Path
import uuid
import yaml
import fitz  # PyMuPDF
import pandas as pd
import pytesseract
from PIL import Image
import io
import docx
import csv
from typing import List, Dict, Any, Union, Optional

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))

# Import utils
from src.utils.logger import setup_logger

# Initialize logger
logger = setup_logger(__name__)

# Load configuration
CONFIG_PATH = ROOT_DIR / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

class DocumentLoader:
    """
    Document loader for loading various document types.
    """
    
    def __init__(self):
        """
        Initialize DocumentLoader.
        """
        logger.info("Initializing DocumentLoader")
        
        # OCR configuration
        self.use_ocr = True
        self.ocr_engine = CONFIG["document_processing"]["ocr_engine"]
        
        # Check if tesseract is available
        try:
            pytesseract.get_tesseract_version()
            logger.info(f"Tesseract OCR available: {pytesseract.get_tesseract_version()}")
        except Exception as e:
            logger.warning(f"Tesseract OCR not available: {e}")
            logger.warning("OCR functionality will be disabled")
            self.use_ocr = False

    def load_document_with_options(self, file_path: Union[str, Path], options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Load a document with custom options, primarily for spreadsheets.

        Args:
            file_path: Path to the document file
            options: Dictionary of options for processing
                - selected_columns: List of column names to include
                - separator: String to use between column values (default " | ")
                - include_column_names: Whether to include column names in the text (default True)

        Returns:
            dict: Document data including text content
        """
        file_path = Path(file_path)
        file_name = file_path.name
        file_ext = file_path.suffix.lower()

        logger.info(f"Loading document with options: {file_name}")

        try:
            # Initialize document data
            document_id = str(uuid.uuid4())
            document_data = {
                "document_id": document_id,
                "file_name": file_name,
                "file_path": str(file_path),
                "file_type": file_ext,
                "content": [],
                "metadata": {
                    "page_count": 0,
                    "word_count": 0,
                }
            }

            # Check if it's a spreadsheet and we have options
            if file_ext in [".csv", ".xlsx", ".xls"] and options:
                logger.info(f"Processing spreadsheet with options: {file_name}")
                document_data = self._process_spreadsheet_with_options(file_path, document_data, options)
            else:
                # Fall back to standard document loading
                document_data = self.load_document(file_path)

            return document_data

        except Exception as e:
            logger.error(f"Error loading document with options {file_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _process_spreadsheet_with_options(self, file_path: Path, document_data: Dict[str, Any],
                                          options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a spreadsheet with column selection options.
        Uses a fixed separator format with pipes and spaces.

        Args:
            file_path: Path to the spreadsheet file
            document_data: Document data dictionary
            options: Processing options (only needs selected_columns)

        Returns:
            dict: Updated document data
        """
        try:
            file_ext = file_path.suffix.lower()

            # Get selected columns
            selected_columns = options.get('selected_columns', [])

            # Always use pipe with spaces as separator
            separator = "  |  "

            # Always include column names in output
            include_column_names = True

            # Read spreadsheet
            if file_ext == ".csv":
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)

            logger.info(f"Loaded spreadsheet with {len(df)} rows and {len(df.columns)} columns")

            # Filter columns if selections provided
            if selected_columns and all(col in df.columns for col in selected_columns):
                df = df[selected_columns]
                logger.info(f"Filtered to {len(selected_columns)} selected columns")
            else:
                # If no valid columns selected or selection is invalid, use all columns
                selected_columns = df.columns.tolist()
                logger.info(f"Using all {len(selected_columns)} columns")

            # Store metadata
            document_data["metadata"]["rows"] = len(df)
            document_data["metadata"]["columns"] = len(df.columns)
            document_data["metadata"]["column_names"] = selected_columns

            # For each row, create a separate content item (which will become a chunk)
            for idx, row in df.iterrows():
                # Format with column names: "Column: Value | Column: Value"
                fields = [f"{col}: {str(val)}" for col, val in row[selected_columns].items()]

                # Join fields with the fixed separator
                row_text = separator.join(fields)

                # Create content item for this row
                content_item = {
                    "text": row_text,
                    "row_idx": idx,
                    "is_spreadsheet_row": True,
                    "metadata": {
                        "row_idx": idx,
                        "file_name": document_data["file_name"],
                        "spreadsheet_columns": selected_columns
                    },
                    "images": []
                }

                # Add to document data
                document_data["content"].append(content_item)

            # Update page count to reflect number of rows
            document_data["metadata"]["page_count"] = len(df)

            # Calculate word count (approximate)
            total_words = sum(len(item.get("text", "").split()) for item in document_data["content"])
            document_data["metadata"]["word_count"] = total_words

            logger.info(f"Processed spreadsheet into {len(df)} row-based content items")
            return document_data

        except Exception as e:
            logger.error(f"Error processing spreadsheet with options {file_path.name}: {e}")
            raise
    def load_document(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load a document from the given file path.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            dict: Document data including text content
        """
        file_path = Path(file_path)
        file_name = file_path.name
        file_ext = file_path.suffix.lower()
        
        logger.info(f"Loading document: {file_name}")
        
        try:
            # Initialize document data
            document_id = str(uuid.uuid4())
            document_data = {
                "document_id": document_id,
                "file_name": file_name,
                "file_path": str(file_path),
                "file_type": file_ext,
                "content": [],
                "metadata": {
                    "page_count": 0,
                    "word_count": 0,
                }
            }
            
            # Process by file type
            if file_ext == ".pdf":
                logger.info(f"Processing PDF: {file_name}")
                document_data = self._process_pdf(file_path, document_data)
                
            elif file_ext == ".docx":
                logger.info(f"Processing DOCX: {file_name}")
                document_data = self._process_docx(file_path, document_data)
                
            elif file_ext == ".txt":
                logger.info(f"Processing TXT: {file_name}")
                document_data = self._process_txt(file_path, document_data)
                
            elif file_ext in [".csv", ".xlsx", ".xls"]:
                logger.info(f"Processing spreadsheet: {file_name}")
                document_data = self._process_spreadsheet(file_path, document_data)
                
            else:
                logger.warning(f"Unsupported file type: {file_ext}")
                return None
            
            # Calculate word count
            total_words = sum(len(item.get("text", "").split()) for item in document_data["content"])
            document_data["metadata"]["word_count"] = total_words
            
            logger.info(f"Document loaded: {file_name}, Pages: {document_data['metadata']['page_count']}, Words: {total_words}")
            
            return document_data
            
        except Exception as e:
            logger.error(f"Error loading document {file_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _process_pdf(self, file_path: Path, document_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a PDF file.
        
        Args:
            file_path: Path to the PDF file
            document_data: Document data dictionary
            
        Returns:
            dict: Updated document data
        """
        try:
            # Open PDF
            pdf_document = fitz.open(file_path)
            page_count = len(pdf_document)
            
            # Update metadata
            document_data["metadata"]["page_count"] = page_count
            
            # Process each page
            for page_num, page in enumerate(pdf_document):
                page_text = page.get_text()
                
                # If page has no text or very little text, try OCR
                if len(page_text.strip()) < 50 and self.use_ocr:
                    logger.info(f"Page {page_num+1} has little text, attempting OCR")
                    page_text = self._ocr_pdf_page(page)
                
                # Get images
                images = []
                image_list = page.get_images(full=True)
                
                for img_index, img_info in enumerate(image_list):
                    xref = img_info[0]
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Store image for later use if needed
                    images.append({
                        "image_id": f"{document_data['document_id']}_p{page_num}_img{img_index}",
                        "image_bytes": image_bytes,
                        "image_type": base_image["ext"]
                    })
                
                # Add page content
                page_content = {
                    "page_num": page_num + 1,
                    "text": page_text,
                    "images": images
                }
                
                document_data["content"].append(page_content)
            
            # Close the document
            pdf_document.close()
            
            return document_data
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path.name}: {e}")
            raise
    
    def _ocr_pdf_page(self, page) -> str:
        """
        Perform OCR on a PDF page.
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            str: Extracted text
        """
        if not self.use_ocr:
            return ""
        
        try:
            # Convert page to an image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_bytes = pix.tobytes()
            
            # Use PIL to open the image
            img = Image.open(io.BytesIO(img_bytes))
            
            # Perform OCR
            text = pytesseract.image_to_string(img)
            
            logger.info(f"OCR extracted {len(text)} characters")
            return text
            
        except Exception as e:
            logger.error(f"Error performing OCR: {e}")
            return ""
    
    def _process_docx(self, file_path: Path, document_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a DOCX file.
        
        Args:
            file_path: Path to the DOCX file
            document_data: Document data dictionary
            
        Returns:
            dict: Updated document data
        """
        try:
            # Open DOCX
            doc = docx.Document(file_path)
            
            # Extract text
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            
            # Create content item
            content_item = {
                "text": text,
                "images": []
            }
            
            # Add to document data
            document_data["content"].append(content_item)
            document_data["metadata"]["page_count"] = 1
            
            return document_data
            
        except Exception as e:
            logger.error(f"Error processing DOCX {file_path.name}: {e}")
            raise
    
    def _process_txt(self, file_path: Path, document_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a TXT file.
        
        Args:
            file_path: Path to the TXT file
            document_data: Document data dictionary
            
        Returns:
            dict: Updated document data
        """
        try:
            # Read text file
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            
            # Create content item
            content_item = {
                "text": text,
                "images": []
            }
            
            # Add to document data
            document_data["content"].append(content_item)
            document_data["metadata"]["page_count"] = 1
            
            return document_data
            
        except Exception as e:
            logger.error(f"Error processing TXT {file_path.name}: {e}")
            raise

    def _process_spreadsheet(self, file_path: Path, document_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a spreadsheet file (CSV, XLSX, XLS).
        Each row becomes a separate content item with selected columns merged with pipe separators.

        Args:
            file_path: Path to the spreadsheet file
            document_data: Document data dictionary

        Returns:
            dict: Updated document data
        """
        try:
            file_ext = file_path.suffix.lower()

            # Read spreadsheet
            if file_ext == ".csv":
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)

            logger.info(f"Loaded spreadsheet with {len(df)} rows and {len(df.columns)} columns")

            # Store basic metadata about the spreadsheet
            document_data["metadata"]["rows"] = len(df)
            document_data["metadata"]["columns"] = len(df.columns)
            document_data["metadata"]["column_names"] = df.columns.tolist()

            # For each row, create a separate content item (which will become a chunk)
            for idx, row in df.iterrows():
                # Convert row to string with pipe separator between values
                # We'll use all columns by default, but this could be modified
                # to use only selected columns based on user input
                row_text = " | ".join([f"{col}: {str(val)}" for col, val in row.items()])

                # Create content item for this row
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

                # Add to document data
                document_data["content"].append(content_item)

            # Update page count to reflect number of rows (treated as "pages")
            document_data["metadata"]["page_count"] = len(df)

            logger.info(f"Processed spreadsheet into {len(df)} row-based content items")
            return document_data

        except Exception as e:
            logger.error(f"Error processing spreadsheet {file_path.name}: {e}")
            raise