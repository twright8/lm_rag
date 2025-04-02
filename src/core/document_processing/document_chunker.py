"""
Document chunking module for Anti-Corruption RAG System.
Uses semantic chunking with E5 embeddings.
"""
import sys
import os
from pathlib import Path
import uuid
import yaml
import torch
import gc
import re
import time
from typing import List, Dict, Any, Union, Optional

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))

# Import utils
from src.utils.logger import setup_logger
from src.utils.resource_monitor import log_memory_usage

# Initialize logger
logger = setup_logger(__name__)

# Load configuration
CONFIG_PATH = ROOT_DIR / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

class DocumentChunker:
    """
    Document chunker that implements semantic chunking.
    """
    
    def __init__(self):
        """
        Initialize document chunker.
        """
        self.chunk_size = CONFIG["document_processing"]["chunk_size"]
        self.chunk_overlap = CONFIG["document_processing"]["chunk_overlap"]
        self.embedding_model_name = CONFIG["models"]["chunking_model"]
        self.embedding_model = None
        
        logger.info(f"Initializing DocumentChunker with chunk_size={self.chunk_size}, "
                   f"chunk_overlap={self.chunk_overlap}, embedding_model={self.embedding_model_name}")
        
        log_memory_usage(logger)
    
    def load_model(self):
        """
        Load the embedding model if not already loaded.
        """
        if self.embedding_model is not None:
            logger.info("Embedding model already loaded, skipping load")
            return
        
        logger.info("===== STARTING MODEL LOAD =====")
        
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            
            # Ensure the TRANSFORMERS_CACHE env var is set and logged
            hf_cache = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
            transformers_cache = os.environ.get('TRANSFORMERS_CACHE', os.path.join(hf_cache, 'transformers'))
            
            logger.info(f"Loading embedding model for chunking: {self.embedding_model_name}...")
            logger.info(f"Using Hugging Face cache directory: {hf_cache}")
            logger.info(f"Using Transformers cache directory: {transformers_cache}")
            
            # Determine device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Configure HuggingFaceEmbeddings
            start_time = time.time()
            
            try:
                self.embedding_model = HuggingFaceEmbeddings(
                    model_name=self.embedding_model_name,
                    cache_folder=transformers_cache,
                    model_kwargs={"device": device},
                    encode_kwargs={"normalize_embeddings": True},
                )
            except Exception as e:
                logger.warning(f"Detailed configuration failed: {e}, falling back to minimal configuration")
                
                self.embedding_model = HuggingFaceEmbeddings(
                    model_name=self.embedding_model_name,
                )
            
            elapsed_time = time.time() - start_time
            
            logger.info(f"Embedding model loaded successfully in {elapsed_time:.2f} seconds")
            
            log_memory_usage(logger)
            
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def shutdown(self):
        """
        Unload embedding model to free up memory.
        """
        if self.embedding_model is not None:
            logger.info("===== STARTING MODEL UNLOAD =====")
            
            # Delete the model reference
            logger.info("Deleting model reference...")
            del self.embedding_model
            self.embedding_model = None
            
            # Clean up CUDA memory
            if torch.cuda.is_available():
                logger.info("Clearing CUDA cache...")
                torch.cuda.empty_cache()
            
            # Force garbage collection
            logger.info("Running garbage collection...")
            gc.collect()
            
            logger.info("===== MODEL UNLOAD COMPLETE =====")
            
            log_memory_usage(logger)
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a document using semantic chunking.
        
        Args:
            document (dict): Document data from DocumentLoader
            
        Returns:
            list: List of chunk dictionaries
        """
        start_time = time.time()
        
        doc_id = document.get('document_id', 'unknown')
        doc_name = document.get('file_name', 'unknown')
        doc_type = document.get('file_type', 'unknown')
        
        logger.info(f"Starting chunking for document: {doc_name} (ID: {doc_id}, Type: {doc_type})")
        
        chunks = []
        
        try:
            # Process each content item (page or section)
            content_items = document.get('content', [])
            total_items = len(content_items)
            
            logger.info(f"Document has {total_items} content items")
            
            for i, content_item in enumerate(content_items):
                page_num = content_item.get('page_num', None)
                text = content_item.get('text', '')
                
                # Skip empty content
                if not text.strip():
                    logger.info(f"Skipping empty content item {i+1}/{total_items}")
                    continue
                
                # Log item details
                page_info = f" (page {page_num})" if page_num else ""
                logger.info(f"Processing content item {i+1}/{total_items}{page_info}, size: {len(text)} chars")
                
                # Generate semantic chunks for this content
                content_chunks = self._semantic_chunking(text)
                
                # Create chunk objects with metadata
                for chunk_idx, chunk_text in enumerate(content_chunks):
                    chunk_id = str(uuid.uuid4())
                    
                    chunk = {
                        'chunk_id': chunk_id,
                        'document_id': doc_id,
                        'file_name': doc_name,
                        'text': chunk_text,
                        'page_num': page_num,
                        'chunk_idx': chunk_idx,
                        'metadata': {
                            'document_metadata': document.get('metadata', {}),
                            'file_type': doc_type,
                            'chunk_method': 'semantic'
                        }
                    }
                    
                    chunks.append(chunk)
            
            # Log results
            total_time = time.time() - start_time
            avg_chunk_size = sum(len(chunk.get('text', '')) for chunk in chunks) / len(chunks) if chunks else 0
            
            logger.info(f"Created {len(chunks)} chunks for document {doc_name} in {total_time:.2f}s. "
                        f"Average chunk size: {avg_chunk_size:.1f} characters")
            
            log_memory_usage(logger)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking document {doc_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def _semantic_chunking(self, text: str) -> List[str]:
        """
        Perform semantic chunking on text.
        
        Args:
            text (str): Text to chunk
            
        Returns:
            list: List of text chunks
        """
        try:
            # Step1: First split by natural boundaries for initial processing
            initial_chunks = self._split_by_recursive_boundaries(text)
            
            # Step 2: Apply semantic chunking
            from langchain_experimental.text_splitter import SemanticChunker
            
            # Ensure model is loaded
            if self.embedding_model is None:
                logger.warning("Embedding model not loaded, loading now...")
                self.load_model()
            
            # Create semantic chunker
            text_splitter = SemanticChunker(
                self.embedding_model,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=95.0
            )
            
            # Process chunks
            semantic_chunks = []
            
            for chunk in initial_chunks:
                # Only apply semantic chunking to larger chunks
                if len(chunk) > self.chunk_size:
                    try:
                        docs = text_splitter.create_documents([chunk])
                        results = [doc.page_content for doc in docs]
                        semantic_chunks.extend(results)
                    except Exception as chunk_error:
                        logger.warning(f"Error in semantic chunking: {chunk_error}")
                        # Fall back to sentence splitting for this chunk
                        semantic_chunks.extend(self._sentence_splitting(chunk))
                else:
                    # Keep small chunks as-is
                    semantic_chunks.append(chunk)
            
            # Step 3: Apply fallback splitting for any chunks still too large
            final_chunks = []
            
            for chunk in semantic_chunks:
                if len(chunk) > self.chunk_size * 1.5:
                    # Split oversized chunks using sentence boundaries
                    sentence_chunks = self._sentence_splitting(chunk)
                    final_chunks.extend(sentence_chunks)
                else:
                    final_chunks.append(chunk)
            
            logger.info(f"Semantic chunking created {len(final_chunks)} chunks from text of length {len(text)}")
            
            return final_chunks
            
        except Exception as e:
            logger.error(f"Error during semantic chunking: {e}")
            logger.warning("Falling back to basic chunking")
            return self._basic_chunking(text)
    
    def _split_by_recursive_boundaries(self, text: str) -> List[str]:
        """
        Split text by natural boundaries like paragraphs and sections.
        
        Args:
            text (str): Text to split
            
        Returns:
            list: List of text chunks split by natural boundaries
        """
        # First try splitting by multiple newlines (paragraphs/sections)
        if '\n\n\n' in text:
            return [chunk.strip() for chunk in text.split('\n\n\n') if chunk.strip()]
        
        # Then try double newlines (paragraphs)
        if '\n\n' in text:
            initial_splits = [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]
            
            # Check if any splits are still too large
            result = []
            for split in initial_splits:
                if len(split) > self.chunk_size * 2:
                    # Try splitting large paragraphs by headings or bullet points
                    subsplits = self._split_by_headings_or_bullets(split)
                    result.extend(subsplits)
                else:
                    result.append(split)
            return result
        
        # If no paragraph breaks, try splitting by headings or bullet points
        heading_splits = self._split_by_headings_or_bullets(text)
        if len(heading_splits) > 1:
            return heading_splits
        
        # Last resort: return the whole text as one chunk
        return [text]
    
    def _split_by_headings_or_bullets(self, text: str) -> List[str]:
        """
        Split text by headings or bullet points.
        
        Args:
            text (str): Text to split
            
        Returns:
            list: List of text chunks split by headings or bullets
        """
        # Try to identify heading patterns or bullet points
        heading_pattern = re.compile(r'\n[A-Z][^\n]{0,50}:\s*\n|\n\d+\.\s+[A-Z]|\n[•\-\*]\s+')
        
        splits = []
        last_end = 0
        
        for match in heading_pattern.finditer(text):
            # Don't split if match is at the beginning
            if match.start() > last_end:
                splits.append(text[last_end:match.start()])
                last_end = match.start()
        
        # Add the final chunk
        if last_end < len(text):
            splits.append(text[last_end:])
        
        # If we found meaningful splits, return them
        if len(splits) > 1:
            return [chunk.strip() for chunk in splits if chunk.strip()]
        
        # Otherwise return the original text
        return [text]
    
    def _sentence_splitting(self, text: str) -> List[str]:
        """
        Split text into sentences and combine into chunks under the target size.
        
        Args:
            text (str): Text to split
            
        Returns:
            list: List of text chunks
        """
        # Split text into sentences
        sentence_endings = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_endings, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Combine sentences into chunks under the target size
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > self.chunk_size and current_chunk:
                # Current chunk would exceed size limit, finalize it
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _basic_chunking(self, text: str) -> List[str]:
        """
        Fallback chunking method that splits text based on character count.
        
        Args:
            text (str): Text to chunk
            
        Returns:
            list: List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            # Find a good end point (preferably at paragraph or sentence boundary)
            end = min(start + self.chunk_size, len(text))
            
            # Try to find paragraph break
            if end < len(text):
                paragraph_break = text.rfind('\n\n', start, end)
                if paragraph_break != -1 and paragraph_break > start + (self.chunk_size // 2):
                    end = paragraph_break + 2
            
            # If no paragraph break, try sentence break
            if end < len(text) and end == start + self.chunk_size:
                sentence_break = max(
                    text.rfind('. ', start, end),
                    text.rfind('! ', start, end),
                    text.rfind('? ', start, end)
                )
                if sentence_break != -1 and sentence_break > start + (self.chunk_size // 2):
                    end = sentence_break + 2
            
            # Get the chunk and add to list
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position for next chunk, accounting for overlap
            start = end - self.chunk_overlap if end < len(text) else end
        
        logger.info(f"Basic chunking created {len(chunks)} chunks from text of length {len(text)}")
        return chunks
