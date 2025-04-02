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
            
            # Convert to string representation
            text = df.to_string()
            
            # Create content item
            content_item = {
                "text": text,
                "metadata": {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": df.columns.tolist()
                },
                "images": []
            }
            
            # Add to document data
            document_data["content"].append(content_item)
            document_data["metadata"]["page_count"] = 1
            
            return document_data
            
        except Exception as e:
            logger.error(f"Error processing spreadsheet {file_path.name}: {e}")
            raise
