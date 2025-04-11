# standalone_pdf_to_markdown.py remains the same as you provided it.

# --- document_chunker.py (Modified) ---

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
# Ensure ROOT_DIR is correctly pointing to your project root if running this file elsewhere
# If this script is inside src/core/indexing, adjust as needed:
# ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

# Import utils
from src.utils.logger import setup_logger
from src.utils.resource_monitor import log_memory_usage

# Import fallback tokenizers (used in sentence splitting fallback)
from src.core.indexing.fallback_tokenizers import simple_sent_tokenize #, simple_word_tokenize # word_tokenize wasn't used

# --- Langchain Imports ---
# Import necessary Langchain components
try:
    from langchain.text_splitter import MarkdownHeaderTextSplitter
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_huggingface import HuggingFaceEmbeddings
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    # Log detailed error during setup
    initial_logger = setup_logger(__name__) # Use temp logger if main one fails
    initial_logger.error(f"Required Langchain components not found: {e}. "
                         "Please install langchain, langchain_experimental, langchain_huggingface. "
                         "`pip install langchain langchain-experimental langchain-huggingface sentence-transformers`")
    LANGCHAIN_AVAILABLE = False
    # Define dummy classes to avoid import errors later if needed, though script should ideally exit
    class MarkdownHeaderTextSplitter: pass
    class SemanticChunker: pass
    class HuggingFaceEmbeddings: pass
# --- End Langchain Imports ---


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

# Default chunking settings if config is missing
DEFAULT_CHUNK_SIZE = CONFIG.get("document_processing", {}).get("chunk_size", 1000)
DEFAULT_CHUNK_OVERLAP = CONFIG.get("document_processing", {}).get("chunk_overlap", 100)
DEFAULT_EMBEDDING_MODEL = CONFIG.get("models", {}).get("chunking_model", "intfloat/multilingual-e5-small")
DEFAULT_SEMANTIC_THRESHOLD = CONFIG.get("document_processing", {}).get("semantic_chunking_threshold", 95.0) # Adjusted default percentile

# --- Define Markdown Headers for Splitting ---
# Standard headers to split on. Langchain's splitter includes the header in the chunk.
# Format: (Marker, Header Name)
DEFAULT_MARKDOWN_HEADERS_TO_SPLIT_ON = [
    ("#", "header_1"),
    ("##", "header_2"),
    ("###", "header_3"),
    ("####", "header_4"),
    ("#####", "header_5"), # Include H5 and H6 if needed
    ("######", "header_6"),
]

class DocumentChunker:
    """
    Document chunker implementing Markdown header splitting followed by semantic chunking.
    Uses Langchain's MarkdownHeaderTextSplitter and SemanticChunker.
    Spreadsheet rows are treated as individual chunks and bypass this logic.
    """

    def __init__(self):
        """
        Initialize document chunker.
        """
        if not LANGCHAIN_AVAILABLE:
            logger.critical("Langchain components could not be imported. DocumentChunker cannot function.")
            raise ImportError("Failed to import required Langchain components. Please check installation.")

        self.chunk_size = CONFIG.get("document_processing", {}).get("chunk_size", DEFAULT_CHUNK_SIZE)
        self.chunk_overlap = CONFIG.get("document_processing", {}).get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)
        self.embedding_model_name = CONFIG.get("models", {}).get("chunking_model", DEFAULT_EMBEDDING_MODEL)
        self.semantic_threshold = CONFIG.get("document_processing", {}).get("semantic_chunking_threshold", DEFAULT_SEMANTIC_THRESHOLD)
        self.markdown_headers = CONFIG.get("document_processing", {}).get("markdown_headers_to_split_on", DEFAULT_MARKDOWN_HEADERS_TO_SPLIT_ON)
        self.embedding_model = None # Langchain embedding model instance

        logger.info(f"Initializing DocumentChunker with chunk_size={self.chunk_size}, "
                   f"chunk_overlap={self.chunk_overlap}, embedding_model={self.embedding_model_name}, "
                   f"semantic_threshold={self.semantic_threshold}")
        logger.info(f"Markdown headers to split on: {self.markdown_headers}")

        log_memory_usage(logger)

    def load_model(self):
        """
        Load the embedding model for semantic chunking if not already loaded.
        """
        if not LANGCHAIN_AVAILABLE:
             logger.error("Cannot load model - Langchain is not available.")
             return

        if self.embedding_model is not None:
            logger.info("Semantic chunking embedding model already loaded.")
            return

        logger.info("===== LOADING SEMANTIC CHUNKING MODEL =====")
        try:
            # Ensure the TRANSFORMERS_CACHE env var is set and logged
            hf_cache = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
            transformers_cache = os.environ.get('TRANSFORMERS_CACHE', os.path.join(hf_cache, 'hub')) # Use 'hub' subdirectory common for HF models

            logger.info(f"Loading embedding model for semantic chunking: {self.embedding_model_name}...")
            logger.info(f"Using Hugging Face cache directory: {hf_cache}")
            logger.info(f"Using Transformers cache directory: {transformers_cache}")

            # Determine device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")

            # Configure HuggingFaceEmbeddings
            start_time = time.time()
            try:
                # Recommended settings for E5 models
                encode_kwargs = {"normalize_embeddings": True}
                model_kwargs = {"device": device}

                self.embedding_model = HuggingFaceEmbeddings(
                    model_name=self.embedding_model_name,
                    cache_folder=transformers_cache, # Point to the cache
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs,
                )
                # Test embedding generation on a small sample
                _ = self.embedding_model.embed_query("test")
                logger.info("Test embedding generated successfully.")

            except Exception as e:
                logger.warning(f"Detailed configuration failed: {e}, falling back to minimal configuration")
                self.embedding_model = HuggingFaceEmbeddings(
                    model_name=self.embedding_model_name,
                    model_kwargs={"device": device} # Still specify device
                )
                _ = self.embedding_model.embed_query("test") # Test again

            elapsed_time = time.time() - start_time
            logger.info(f"Semantic chunking embedding model loaded successfully in {elapsed_time:.2f} seconds")
            log_memory_usage(logger)

        except ImportError as ie:
             logger.error(f"ImportError loading model: {ie}. Make sure langchain_huggingface and sentence-transformers are installed.")
             self.embedding_model = None # Ensure it's None if loading fails
             raise # Re-raise to signal failure
        except Exception as e:
            logger.error(f"Error loading semantic chunking embedding model: {e}", exc_info=True)
            self.embedding_model = None # Ensure it's None if loading fails
            raise # Re-raise to signal failure

    def shutdown(self):
        """
        Unload embedding model to free up memory.
        """
        if self.embedding_model is not None:
            logger.info("===== UNLOADING SEMANTIC CHUNKING MODEL =====")
            try:
                # Access device via _client (HuggingFaceEmbeddings specific)
                model_device = None
                if hasattr(self.embedding_model, 'client') and hasattr(self.embedding_model.client, 'device'):
                    model_device = self.embedding_model.client.device
                    logger.info(f"Model was on device: {model_device}")
                elif hasattr(self.embedding_model, 'model_kwargs') and 'device' in self.embedding_model.model_kwargs:
                     model_device_str = self.embedding_model.model_kwargs['device']
                     model_device = torch.device(model_device_str) # Convert string to device obj
                     logger.info(f"Model was on device: {model_device}")
                else:
                    logger.warning("Could not determine model device before unloading.")

                # Delete the model reference
                logger.info("Deleting model reference...")
                # --- FIX: Ensure underlying model object is deleted if possible ---
                if hasattr(self.embedding_model, 'client'):
                    del self.embedding_model.client
                # --- End FIX ---
                del self.embedding_model
                self.embedding_model = None

                # Clean up CUDA memory if the model was on CUDA
                if model_device and model_device.type == 'cuda' and torch.cuda.is_available():
                    logger.info("Clearing CUDA cache...")
                    torch.cuda.empty_cache()

                # Force garbage collection
                logger.info("Running garbage collection...")
                gc.collect()
                logger.info("Semantic chunking model unloaded.")
                log_memory_usage(logger)
            except Exception as e:
                logger.error(f"Error during model shutdown: {e}", exc_info=True)
        else:
            logger.info("Semantic chunking model not loaded, nothing to shut down.")

    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a document using Markdown header splitting and semantic chunking.
        Handles spreadsheets separately.

        Args:
            document (dict): Document data from DocumentLoader

        Returns:
            list: List of chunk dictionaries, matching the required output format.
        """
        if not LANGCHAIN_AVAILABLE:
            logger.error("Cannot chunk document - Langchain is not available.")
            return []

        start_time = time.time()

        doc_id = document.get('document_id', 'unknown_doc_' + str(uuid.uuid4())[:8])
        doc_name = document.get('file_name', 'Unknown Filename')
        doc_type = document.get('file_type', 'unknown')
        doc_metadata = document.get('metadata', {}) # Original document metadata

        logger.info(f"Starting chunking for document: {doc_name} (ID: {doc_id}, Type: {doc_type})")

        final_chunks = []
        content_items = document.get('content', [])
        total_items = len(content_items)

        if not content_items:
            logger.warning(f"Document {doc_name} has no content items to process.")
            return []

        logger.info(f"Processing {total_items} content item(s) for {doc_name}.")

        # --- Pre-load model once if needed ---
        # Check if any non-spreadsheet items exist before loading
        needs_semantic = any(not item.get('is_spreadsheet_row', False) and item.get('text', '').strip() for item in content_items)
        if needs_semantic and self.embedding_model is None:
            try:
                self.load_model()
            except Exception as load_err:
                 logger.error(f"Failed to load embedding model required for chunking {doc_name}. Cannot proceed with semantic chunking.", exc_info=True)
                 # Decide how to proceed: either skip semantic chunking or fail entirely.
                 # For now, let's allow fallback to basic chunking.
                 pass # Error already logged, fallback will happen later if model is None

        # --- Process Content Items ---
        global_chunk_index = 0 # Track overall chunk index across all items
        for i, content_item in enumerate(content_items):
            item_text = content_item.get('text', '')
            item_metadata = content_item.get('metadata', {})
            page_num = content_item.get('page_num', None) # May be None for non-PDFs or full Markdown

            # --- Spreadsheet Row Handling ---
            if content_item.get('is_spreadsheet_row', False):
                row_idx = content_item.get('row_idx', global_chunk_index) # Use row_idx or sequential index
                chunk_id = str(uuid.uuid4())
                logger.debug(f"Processing spreadsheet row {row_idx} as a single chunk.")

                chunk = {
                    'chunk_id': chunk_id,
                    'document_id': doc_id,
                    'file_name': doc_name,
                    'text': item_text, # Use the pre-formatted row text directly
                    'original_text': item_text, # Original and text are the same for rows
                    'page_num': None, # No page number for spreadsheet rows
                    'row_idx': row_idx, # Include row index
                    'chunk_idx': global_chunk_index, # Overall index
                    'metadata': {
                        'document_metadata': doc_metadata, # Original doc metadata
                        'file_type': doc_type,
                        'chunk_method': 'spreadsheet_row', # Mark chunking method
                        'row_idx': row_idx,
                        'spreadsheet_columns': item_metadata.get('spreadsheet_columns', []),
                        # Add other relevant item metadata if needed
                    }
                }
                final_chunks.append(chunk)
                global_chunk_index += 1
                continue # Skip further chunking for this item

            # --- Markdown/Text Content Handling ---
            if not item_text or not item_text.strip():
                logger.info(f"Skipping empty content item {i + 1}/{total_items} for {doc_name}.")
                continue

            logger.info(f"Processing content item {i + 1}/{total_items} (length: {len(item_text)} chars) using Markdown Header + Semantic chunking.")

            # Perform Markdown Header splitting followed by semantic chunking
            try:
                # item_text is assumed to be Markdown here (from Docling export)
                item_chunks = self._semantic_chunking(item_text, item_metadata) # Pass item_metadata

                # Create final chunk dictionaries for each split part
                for chunk_idx_in_item, chunk_data in enumerate(item_chunks):
                    chunk_text = chunk_data.get("text")
                    chunk_headers = chunk_data.get("headers", {}) # Get headers from _semantic_chunking

                    if not chunk_text or not chunk_text.strip():
                        continue # Skip empty chunks resulting from splitting

                    chunk_id = str(uuid.uuid4())
                    chunk_metadata = {
                        'document_metadata': doc_metadata,
                        'file_type': doc_type,
                        'chunk_method': chunk_data.get("chunk_method", "markdown_semantic"), # Use method from chunk_data
                        'original_item_index': i, # Index of the content item this chunk came from
                        'chunk_index_within_item': chunk_idx_in_item, # Index relative to splits from this item
                        # Add header metadata if present
                        **chunk_headers,
                        # Add other relevant item metadata if needed
                        **{k: v for k, v in item_metadata.items() if k not in ['text', 'page_content']} # Merge item metadata safely
                    }

                    chunk = {
                        'chunk_id': chunk_id,
                        'document_id': doc_id,
                        'file_name': doc_name,
                        'text': chunk_text, # The final chunked text
                        'original_text': chunk_text, # Store the chunk itself as original for now
                        'page_num': page_num, # Carry over page num if available (might be None for full markdown export)
                        'row_idx': None, # Not a spreadsheet row
                        'chunk_idx': global_chunk_index, # Overall index
                        'metadata': chunk_metadata
                    }
                    final_chunks.append(chunk)
                    global_chunk_index += 1

            except Exception as e:
                 logger.error(f"Error during chunking pipeline for item {i+1} of {doc_name}: {e}", exc_info=True)
                 # Fallback to basic chunking for the *entire* item_text
                 logger.warning(f"Falling back to basic character chunking for item {i+1} due to error.")
                 basic_chunks = self._basic_chunking(item_text)
                 for chunk_idx_in_item, chunk_text in enumerate(basic_chunks):
                     if not chunk_text or not chunk_text.strip(): continue
                     chunk_id = str(uuid.uuid4())
                     chunk = {
                         'chunk_id': chunk_id, 'document_id': doc_id, 'file_name': doc_name,
                         'text': chunk_text, 'original_text': chunk_text, 'page_num': page_num,
                         'row_idx': None, 'chunk_idx': global_chunk_index,
                         'metadata': { 'document_metadata': doc_metadata, 'file_type': doc_type,
                                       'chunk_method': 'basic_fallback', 'original_item_index': i,
                                       'chunk_index_within_item': chunk_idx_in_item,
                                       **{k: v for k, v in item_metadata.items() if k not in ['text', 'page_content']} }
                     }
                     final_chunks.append(chunk)
                     global_chunk_index += 1


        # Log results
        total_time = time.time() - start_time
        avg_chunk_size = sum(len(chunk.get('text', '')) for chunk in final_chunks) / len(final_chunks) if final_chunks else 0
        spreadsheet_chunks = sum(1 for chunk in final_chunks if chunk.get('metadata', {}).get('chunk_method') == 'spreadsheet_row')
        semantic_chunks = sum(1 for chunk in final_chunks if chunk.get('metadata', {}).get('chunk_method') == 'markdown_semantic')
        basic_fallback_chunks = sum(1 for chunk in final_chunks if chunk.get('metadata', {}).get('chunk_method') == 'basic_fallback')
        sentence_fallback_chunks = sum(1 for chunk in final_chunks if chunk.get('metadata', {}).get('chunk_method') == 'sentence_fallback')


        logger.info(f"Chunking for {doc_name} completed in {total_time:.2f}s.")
        logger.info(f"Created {len(final_chunks)} total chunks: "
                    f"{spreadsheet_chunks} spreadsheet rows, "
                    f"{semantic_chunks} semantic chunks, "
                    f"{sentence_fallback_chunks} sentence fallback chunks, "
                    f"{basic_fallback_chunks} basic fallback chunks.")
        logger.info(f"Average chunk size: {avg_chunk_size:.1f} characters.")

        log_memory_usage(logger)
        return final_chunks

    # --- MODIFIED: Uses MarkdownHeaderTextSplitter then SemanticChunker ---
    def _semantic_chunking(self, text: str, item_metadata: Dict) -> List[Dict[str, Any]]:
        """
        Perform Markdown header splitting followed by semantic chunking.

        Args:
            text (str): Text content (Markdown expected) to chunk.
            item_metadata (dict): Metadata associated with this text item.

        Returns:
            list: List of dictionaries, each containing 'text', 'headers', and 'chunk_method'.
        """
        final_chunk_data = [] # Store dictionaries {text: str, headers: dict, chunk_method: str}

        try:
            # Step 1: Initial split using Langchain's Markdown Header Splitter
            logger.debug("Performing initial split using MarkdownHeaderTextSplitter...")
            markdown_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=self.markdown_headers,
                return_each_line=False # Keep paragraphs together
            )
            # split_text returns List[Document] where Document has page_content and metadata
            initial_docs = markdown_splitter.split_text(text)
            logger.info(f"MarkdownHeaderTextSplitter created {len(initial_docs)} initial structural chunks.")

            # If header splitting produced no results, but text exists, treat the whole text as one initial chunk
            if not initial_docs and text.strip():
                 from langchain_core.documents import Document # Import locally if needed
                 logger.warning("MarkdownHeaderTextSplitter found no headers; treating entire text as one chunk.")
                 initial_docs = [Document(page_content=text, metadata=item_metadata)] # Use item_metadata

            # Step 2: Apply semantic chunking *within* these initial structural chunks if needed
            semantic_chunker = None
            if self.embedding_model: # Only initialize if model loaded successfully
                 try:
                     semantic_chunker = SemanticChunker(
                         self.embedding_model,
                         breakpoint_threshold_type="percentile",
                         breakpoint_threshold_amount=self.semantic_threshold
                     )
                 except Exception as e:
                      logger.error(f"Failed to initialize SemanticChunker: {e}. Proceeding without semantic splitting.", exc_info=True)
                      semantic_chunker = None # Ensure it's None
            else:
                logger.warning("Embedding model not loaded. Skipping semantic chunking step.")


            processed_chunks = 0
            for i, doc in enumerate(initial_docs):
                initial_chunk_text = doc.page_content
                initial_chunk_headers = doc.metadata # Headers extracted by MarkdownHeaderTextSplitter

                if not initial_chunk_text or not initial_chunk_text.strip():
                    continue

                # Check if semantic chunking should be applied (chunk is large enough AND semantic_chunker exists)
                # Use chunk_size as a rough guide - semantic splitting adds overhead
                # Only split if chunk is significantly larger than the target size OR larger than some minimum
                needs_semantic_split = (semantic_chunker is not None and
                                        len(initial_chunk_text) > max(self.chunk_size, 500)) # Heuristic: apply if > chunk_size or > 500 chars

                sub_chunks_data = [] # Store results for this initial doc
                chunk_method = "markdown_semantic" # Default assumption

                if needs_semantic_split:
                    try:
                        logger.debug(f"Applying semantic chunking to initial chunk {i+1}/{len(initial_docs)} (length {len(initial_chunk_text)})...")
                        # SemanticChunker expects a list of strings, returns List[Document]
                        semantically_split_docs = semantic_chunker.create_documents([initial_chunk_text])
                        # Convert results back to our format
                        for sub_doc in semantically_split_docs:
                            sub_text = sub_doc.page_content.strip()
                            if sub_text:
                                # IMPORTANT: Preserve headers from the *initial* structural chunk
                                sub_chunks_data.append({"text": sub_text, "headers": initial_chunk_headers, "chunk_method": chunk_method})
                        processed_chunks += 1
                        logger.debug(f"Semantically split initial chunk {i+1} into {len(sub_chunks_data)} sub-chunks.")
                    except Exception as chunk_error:
                        logger.warning(f"Error during semantic chunking for initial chunk {i+1}: {chunk_error}. Using initial chunk as is.")
                        # Fallback: use the initial chunk text
                        sub_chunks_data.append({"text": initial_chunk_text, "headers": initial_chunk_headers, "chunk_method": chunk_method})
                else:
                    # Keep the initial chunk as is (either too small or semantic chunker unavailable)
                    logger.debug(f"Keeping initial chunk {i+1} (length {len(initial_chunk_text)}) as is (no semantic split).")
                    sub_chunks_data.append({"text": initial_chunk_text, "headers": initial_chunk_headers, "chunk_method": chunk_method})

                final_chunk_data.extend(sub_chunks_data)


            logger.debug(f"Applied semantic splitting to {processed_chunks}/{len(initial_docs)} initial structural chunks.")
            logger.info(f"Markdown header + semantic splitting resulted in {len(final_chunk_data)} potential chunks.")

            # Step 3: Final size check and fallback splitting (optional but recommended)
            # This ensures no chunk dramatically exceeds the desired size limit after semantic splitting.
            # We'll use the simpler _basic_chunking as a fallback here if chunks are still huge.
            post_processed_chunks = []
            oversized_count = 0
            max_len = self.chunk_size * 10 # Allow some flexibility over target chunk size

            for chunk_data in final_chunk_data:
                chunk_text = chunk_data["text"]
                if not chunk_text: continue

                if len(chunk_text) > max_len:
                    oversized_count += 1
                    logger.warning(f"Chunk (length {len(chunk_text)}) still exceeds size limit ({max_len}) after semantic split. Applying basic chunking fallback.")
                    # Split the oversized chunk using basic character splitting
                    basic_sub_chunks = self._basic_chunking(chunk_text)
                    for sub_chunk_text in basic_sub_chunks:
                        if sub_chunk_text.strip():
                            # Keep original headers, mark as basic fallback
                            post_processed_chunks.append({
                                "text": sub_chunk_text.strip(),
                                "headers": chunk_data["headers"],
                                "chunk_method": "basic_fallback" # Mark method
                            })
                else:
                    # Chunk is within acceptable size
                    post_processed_chunks.append(chunk_data)

            if oversized_count > 0:
                 logger.info(f"Applied basic chunking fallback to {oversized_count} oversized chunks.")

            logger.info(f"Final processing created {len(post_processed_chunks)} chunks from text of length {len(text)}.")
            return post_processed_chunks

        except ImportError as ie:
             logger.error(f"ImportError during chunking: {ie}. Langchain components missing?", exc_info=True)
             logger.warning("Falling back to basic chunking for the entire text.")
             basic_chunks = self._basic_chunking(text)
             return [{"text": t, "headers": {}, "chunk_method": "basic_fallback"} for t in basic_chunks if t.strip()]
        except Exception as e:
            logger.error(f"Error during Markdown/semantic chunking pipeline: {e}", exc_info=True)
            logger.warning("Falling back to basic chunking for the entire text.")
            basic_chunks = self._basic_chunking(text)
            return [{"text": t, "headers": {}, "chunk_method": "basic_fallback"} for t in basic_chunks if t.strip()]

    # --- REMOVED: _split_by_markdown_boundaries method is no longer needed ---
    # def _split_by_markdown_boundaries(self, text: str) -> List[str]:
    #     ... # Removed

    # --- Sentence Splitting Fallback (Kept as is, but less likely to be primary fallback) ---
    def _sentence_splitting(self, text: str) -> List[str]:
        """
        Fallback: Split text into sentences and combine into chunks under the target size.
        Uses NLTK if available, otherwise a simple regex splitter.

        Args:
            text (str): Text to split.

        Returns:
            list: List of text chunks based on sentences.
        """
        sentences = []
        try:
            # Try using NLTK first
            import nltk
            try:
                # Check if 'punkt' is available
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                logger.warning("NLTK 'punkt' tokenizer not found. Attempting download...")
                try:
                    nltk.download('punkt', quiet=True)
                    logger.info("NLTK 'punkt' downloaded successfully.")
                except Exception as download_err:
                    logger.warning(f"Failed to download 'punkt' ({download_err}). Falling back to simple regex sentence splitter.")
                    # Fall through to use simple_sent_tokenize below
                    raise ImportError("NLTK punkt download failed") # Force fallback

            sentences = nltk.sent_tokenize(text)
            logger.debug("Using NLTK sentence tokenizer.")

        except ImportError:
            logger.warning("NLTK not available or 'punkt' download failed. Falling back to simple regex sentence splitter.")
            sentences = simple_sent_tokenize(text) # Use fallback
        except Exception as nltk_err:
             logger.warning(f"Error using NLTK sent_tokenize: {nltk_err}. Falling back to simple regex sentence splitter.")
             sentences = simple_sent_tokenize(text) # Use fallback


        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return []

        # Combine sentences into chunks respecting self.chunk_size
        chunks = []
        current_chunk_sentences = []
        current_length = 0
        # Overlap logic for sentence splitting can be complex, often RecursiveCharacterTextSplitter
        # handles overlap more simply. We'll keep this simple combination logic without explicit overlap for now.

        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence)
            # Estimate length with separator space
            potential_length = current_length + (len(" ") if current_chunk_sentences else 0) + sentence_length

            if potential_length <= self.chunk_size or not current_chunk_sentences:
                # Add sentence if it fits or if the chunk is currently empty (even if sentence is long)
                current_chunk_sentences.append(sentence)
                current_length = potential_length
            else:
                # Current chunk is full, finalize it
                chunks.append(" ".join(current_chunk_sentences))
                # Start new chunk with the current sentence
                current_chunk_sentences = [sentence]
                current_length = sentence_length

            # Handle the last chunk
            if i == len(sentences) - 1 and current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))

        logger.debug(f"Sentence splitting created {len(chunks)} chunks.")
        return chunks


    # --- Basic Character Splitting Fallback (Kept as is) ---
    def _basic_chunking(self, text: str) -> List[str]:
        """
        Fallback chunking method that splits text based on character count with overlap.
        Tries to split at sensible boundaries (paragraphs, sentences) if possible within the window.

        Args:
            text (str): Text to chunk.

        Returns:
            list: List of text chunks.
        """
        logger.debug(f"Applying basic character-based chunking with size {self.chunk_size} and overlap {self.chunk_overlap}.")
        # Use Langchain's RecursiveCharacterTextSplitter for a robust basic fallback
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter

            # Use common separators, prioritizing paragraphs and sentences
            separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                is_separator_regex=False,
                separators=separators,
            )
            chunks = text_splitter.split_text(text)
            # Filter out any purely whitespace chunks that might occur
            chunks = [chunk for chunk in chunks if chunk.strip()]
            logger.info(f"RecursiveCharacterTextSplitter created {len(chunks)} basic chunks.")
            return chunks

        except ImportError:
            logger.warning("RecursiveCharacterTextSplitter not found (Langchain missing?). Using manual basic chunking.")
            # Manual fallback if Langchain isn't available (less robust)
            chunks = []
            start = 0
            text_len = len(text)

            while start < text_len:
                end = min(start + self.chunk_size, text_len)
                chunk_text = text[start:end].strip()

                if chunk_text:
                    chunks.append(chunk_text)

                # Calculate the start of the next chunk with overlap
                next_start = start + self.chunk_size - self.chunk_overlap
                # Ensure start advances, prevent infinite loop
                if next_start <= start:
                     start += 1 # Force advancement if stuck
                else:
                     start = next_start

                if start >= text_len:
                    break
            logger.info(f"Manual basic chunking created {len(chunks)} chunks.")
            return chunks
        except Exception as e:
             logger.error(f"Error during basic chunking: {e}. Returning single chunk.", exc_info=True)
             return [text] # Safest fallback

