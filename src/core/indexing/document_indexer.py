"""
Enhanced document indexing module with debugging and fallback mechanisms.
Modified to explicitly use enhanced text (with metadata) for embeddings.
"""
import sys
import os
from pathlib import Path
import yaml
import pickle
import torch
import gc
import time
import nltk
import numpy as np
from typing import List, Dict, Any, Union, Optional, Tuple
import socket
import json

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


class DocumentIndexer:
    """
    Enhanced document indexer with debugging and fallback mechanisms.
    Modified to explicitly use enhanced text (with metadata) for embeddings.
    """

    def __init__(self, debug_mode=False):
        """
        Initialize document indexer with debug mode option.

        Args:
            debug_mode (bool): Enable additional debugging
        """
        self.debug_mode = debug_mode

        # Vector DB configuration
        self.qdrant_host = CONFIG["qdrant"]["host"]
        self.qdrant_port = CONFIG["qdrant"]["port"]
        self.qdrant_collection = CONFIG["qdrant"]["collection_name"]

        # Add fallback settings
        self.fallback_to_bm25_only = CONFIG.get("indexing", {}).get("fallback_to_bm25_only", True)

        # Embedding model configuration
        self.embedding_model_name = CONFIG["models"]["embedding_model"]

        # BM25 configuration
        self.bm25_path = ROOT_DIR / CONFIG["storage"]["bm25_index_path"]
        self.bm25_path.parent.mkdir(parents=True, exist_ok=True)

        # Debug directory
        self.debug_dir = ROOT_DIR / "debug"
        if self.debug_mode:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

        # Initialize embedding model
        self.embed_register = None

        self._validate_config()

        logger.info(f"Initializing EnhancedDocumentIndexer with model={self.embedding_model_name}, "
                    f"qdrant={self.qdrant_host}:{self.qdrant_port}, "
                    f"collection={self.qdrant_collection}, "
                    f"debug_mode={self.debug_mode}")

        if self.debug_mode:
            # Log more detailed configuration info
            self._log_debug_info()

        log_memory_usage(logger)

    def _validate_config(self):
        """
        Validate configuration settings and log warnings for potential issues.
        """
        # Check if Qdrant hostname is valid
        try:
            # Try to resolve the hostname to check DNS
            if self.qdrant_host not in ('localhost', '127.0.0.1'):
                socket.gethostbyname(self.qdrant_host)
        except Exception as e:
            logger.warning(f"Unable to resolve Qdrant hostname '{self.qdrant_host}': {e}")
            logger.warning(
                "If using Docker, make sure the hostname matches your Docker container name or use 'localhost'")

        # Check if configuration paths are absolute
        if not os.path.isabs(str(self.bm25_path)):
            logger.warning(f"BM25 path {self.bm25_path} is not absolute, may cause issues")

    def _log_debug_info(self):
        """
        Log detailed debugging information.
        """
        logger.info(f"Debug mode ON - logging detailed information")

        # Log config file location
        logger.debug(f"Config file path: {CONFIG_PATH}")

        # Log directory information
        logger.debug(f"Root directory: {ROOT_DIR}")
        logger.debug(f"BM25 index path: {self.bm25_path}")
        logger.debug(f"Debug directory: {self.debug_dir}")

        # Log Qdrant connection info
        logger.debug(f"Qdrant connection: {self.qdrant_host}:{self.qdrant_port}")

        # Attempt to ping Qdrant host
        try:
            result = socket.gethostbyname(self.qdrant_host)
            logger.debug(f"Qdrant host resolution: {self.qdrant_host} -> {result}")
        except Exception as e:
            logger.debug(f"Qdrant host resolution failed: {e}")

        # Save all config to debug file
        debug_config_path = self.debug_dir / "config_dump.json"
        with open(debug_config_path, 'w') as f:
            json.dump(CONFIG, f, indent=2, default=str)

        logger.debug(f"Config dump saved to: {debug_config_path}")

    def load_model(self):
        """
        Load the embedding model if not already loaded.
        """
        if self.embed_register is not None:
            logger.info("Embedding model already loaded, skipping load")
            return

        logger.info("===== STARTING MODEL LOAD =====")

        try:
            from embed import BatchedInference

            # Determine device
            device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            logger.info(f"Device: {device}")

            # Time the model loading
            start_time = time.time()

            self.embed_register = BatchedInference(
                model_id=self.embedding_model_name,
                engine="torch",
                device=device
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
        Unload the embedding model to free up resources.
        """
        if self.embed_register is not None:
            logger.info("===== STARTING MODEL UNLOAD =====")

            try:
                # Stop the register
                logger.info("Stopping embed register...")
                self.embed_register.stop()
            except AttributeError as ae:
                logger.warning(f"Attribute error when stopping embed register: {ae}")
                # Continue with cleanup

            # Delete the register reference
            logger.info("Deleting register reference...")
            del self.embed_register
            self.embed_register = None

            # Clean up CUDA memory
            if torch.cuda.is_available():
                logger.info("Clearing CUDA cache...")
                torch.cuda.empty_cache()

            # Force garbage collection
            logger.info("Running garbage collection...")
            gc.collect()

            logger.info("===== MODEL UNLOAD COMPLETE =====")

            log_memory_usage(logger)

    def index_documents(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Index document chunks to Qdrant and BM25.

        Args:
            chunks (list): List of document chunks
        """
        if not chunks:
            logger.warning("No chunks to index")
            return

        logger.info(f"Indexing {len(chunks)} chunks")

        # Debug: save chunks to file
        if self.debug_mode:
            chunks_path = self.debug_dir / "chunks_to_index.json"
            try:
                # Create a copy with minimal info to avoid huge files
                minimal_chunks = []
                for chunk in chunks:
                    minimal_chunk = {
                        'chunk_id': chunk.get('chunk_id', ''),
                        'text_length': len(chunk.get('text', '')),
                        'metadata': chunk.get('metadata', {})
                    }
                    minimal_chunks.append(minimal_chunk)

                with open(chunks_path, 'w') as f:
                    json.dump(minimal_chunks, f, indent=2)
                logger.debug(f"Saved chunk info to: {chunks_path}")
            except Exception as e:
                logger.debug(f"Error saving chunks debug info: {e}")

        try:
            # Ensure embedding model is loaded
            self.load_model()

            # First always index to BM25 (this is local files, should work regardless)
            bm25_success = self._index_to_bm25(chunks)

            # Then try to index to Qdrant
            qdrant_success = False
            try:
                qdrant_success = self._index_to_qdrant(chunks)
            except Exception as e:
                logger.error(f"Error indexing to Qdrant: {e}")
                import traceback
                logger.error(traceback.format_exc())

                if self.debug_mode:
                    error_path = self.debug_dir / "qdrant_error.txt"
                    with open(error_path, 'w') as f:
                        f.write(f"Error: {str(e)}\n\n")
                        f.write(traceback.format_exc())
                    logger.debug(f"Saved Qdrant error to: {error_path}")

            # Log overall status
            if qdrant_success and bm25_success:
                logger.info(f"Indexing completed successfully for {len(chunks)} chunks (Qdrant and BM25)")
            elif bm25_success:
                if self.fallback_to_bm25_only:
                    logger.warning(f"Indexing completed with fallback mode (BM25 only)")
                else:
                    logger.error(f"Indexing incomplete - BM25 succeeded but Qdrant failed")
            else:
                logger.error(f"Indexing failed completely - neither Qdrant nor BM25 succeeded")

        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def _index_to_qdrant(self, chunks: List[Dict[str, Any]]) -> bool:
        """
        Index chunks to Qdrant.
        This version explicitly uses the enhanced text with metadata for embeddings.

        Args:
            chunks (list): List of document chunks

        Returns:
            bool: Success status
        """
        logger.info(f"Indexing {len(chunks)} chunks to Qdrant")

        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models as rest

            # Debug: test connection before full indexing
            if self.debug_mode:
                logger.debug(f"Testing Qdrant connection to {self.qdrant_host}:{self.qdrant_port}")
                try:
                    client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port, timeout=10)
                    # Simple ping/health check
                    response = client.http.health_api.health_check()
                    logger.debug(f"Qdrant connection test successful: {response}")
                except Exception as e:
                    logger.error(f"Qdrant connection test failed: {e}")
                    raise RuntimeError(f"Cannot connect to Qdrant server: {e}")

            # Connect to Qdrant
            client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port, timeout=30)

            # Generate embeddings for chunks
            logger.info("Generating embeddings...")

            # IMPORTANT: This is where we use the enhanced text with metadata for embeddings
            # We log this to make it explicit that we're using the enhanced text from entity_extractor
            texts = [chunk.get('text', '') for chunk in chunks]
            logger.info(f"Using enhanced text with summary, red flags, and entities for embedding generation")

            if self.debug_mode and texts:
                # Log a sample of the enhanced text for verification
                sample_text = texts[0][:200] + "..." if len(texts[0]) > 200 else texts[0]
                logger.debug(f"Sample enhanced text for embedding: {sample_text}")

            start_time = time.time()

            # Add validation for texts
            if not texts or len(texts) == 0:
                logger.error("No texts provided for embedding generation")
                return False

            # Use embed register to generate embeddings
            logger.info(f"Generating embeddings for {len(texts)} texts using model {self.embedding_model_name}")
            future = self.embed_register.embed(
                sentences=texts,
                model_id=self.embedding_model_name
            )

            # Get embeddings and validate - result() returns a tuple (embeddings, token_usage)
            result = future.result()

            # Check if result is a tuple (embeddings, token_usage) as expected from infinity embedding
            if isinstance(result, tuple) and len(result) >= 1:
                logger.info(f"Got result tuple from embed: {len(result)} items")
                embeddings = result[0]  # First item is the embeddings list
            else:
                embeddings = result  # Fall back to treating result as embeddings directly

            logger.info(f"Embeddings result type: {type(embeddings)}")

            # Validate embeddings
            if not embeddings or len(embeddings) == 0:
                logger.error("No embeddings returned from model")
                return False

            # Handle both list of arrays and single array cases
            if len(embeddings) == 1 and isinstance(embeddings, list):
                logger.info(f"Single embedding in a list format: {type(embeddings[0])}")

                # Special handling for potentially nested structure
                first_item = embeddings[0]

                # If first_item is an int or float, we have a problem
                if isinstance(first_item, (int, float)):
                    logger.error(f"Invalid embedding: got scalar value {first_item}")
                    return False

                # If first_item is a list, check its length to verify it's an actual embedding
                if isinstance(first_item, list) and len(first_item) < 2:
                    logger.error(f"Invalid embedding list: too short {first_item}")
                    return False

                # If first_item is a numpy array, check shape
                if hasattr(first_item, 'shape'):
                    logger.info(f"Embedding shape: {first_item.shape}")

            # More detailed logging of the embeddings structure
            logger.info(f"Embeddings type: {type(embeddings)}, length: {len(embeddings)}")
            if embeddings and len(embeddings) > 0:
                logger.info(f"First embedding type: {type(embeddings[0])}")

                # Log a short preview of the first embedding
                first_emb = embeddings[0]
                if hasattr(first_emb, 'tolist'):
                    preview = str(first_emb.tolist()[:3]) + "..." if len(first_emb) > 3 else str(first_emb.tolist())
                    logger.info(f"First embedding preview: {preview}")
                elif isinstance(first_emb, list):
                    preview = str(first_emb[:3]) + "..." if len(first_emb) > 3 else str(first_emb)
                    logger.info(f"First embedding preview: {preview}")

            elapsed_time = time.time() - start_time
            logger.info(f"Generated {len(embeddings)} embeddings in {elapsed_time:.2f}s")

            # Create or get collection
            vector_size = len(embeddings[0])
            logger.info(f"Vector size: {vector_size}")

            # Debug - save a sample embedding
            if self.debug_mode:
                try:
                    emb_sample_path = self.debug_dir / "embedding_sample.json"
                    with open(emb_sample_path, 'w') as f:
                        # Handle different embedding formats
                        first_emb = embeddings[0]

                        if hasattr(first_emb, 'shape') and hasattr(first_emb, 'tolist'):
                            # Tensor-like object
                            vector_sample = first_emb.tolist()[:10] + ['...'] if len(
                                first_emb) > 10 else first_emb.tolist()
                            sample_data = {
                                'type': type(first_emb).__name__,
                                'shape': list(first_emb.shape) if hasattr(first_emb.shape,
                                                                          '__iter__') else first_emb.shape,
                                'vector_sample': vector_sample,
                                'norm': float(np.linalg.norm(first_emb)),
                                'min': float(np.min(first_emb)),
                                'max': float(np.max(first_emb)),
                                'mean': float(np.mean(first_emb))
                            }
                        elif isinstance(first_emb, list):
                            # List object
                            vector_sample = first_emb[:10] + ['...'] if len(first_emb) > 10 else first_emb
                            sample_data = {
                                'type': 'list',
                                'length': len(first_emb),
                                'vector_sample': vector_sample,
                                'first_element_type': type(first_emb[0]).__name__ if first_emb else 'unknown'
                            }
                        else:
                            # Unknown format
                            sample_data = {
                                'type': type(first_emb).__name__,
                                'string_representation': str(first_emb)[:100]
                            }

                        json.dump(sample_data, f, indent=2)
                    logger.debug(f"Saved embedding sample to: {emb_sample_path}")
                except Exception as e:
                    logger.debug(f"Error saving embedding sample: {e}")

            # Check if collection exists
            try:
                collections = client.get_collections().collections
                collection_names = [collection.name for collection in collections]

                if self.qdrant_collection not in collection_names:
                    # Create collection
                    logger.info(f"Creating collection: {self.qdrant_collection}")
                    client.create_collection(
                        collection_name=self.qdrant_collection,
                        vectors_config=rest.VectorParams(
                            size=vector_size,
                            distance=rest.Distance.COSINE
                        )
                    )
            except Exception as e:
                logger.error(f"Error checking/creating collection: {e}")
                if self.debug_mode:
                    # Try to diagnose the connection more thoroughly
                    self._diagnose_qdrant_connection()
                raise

            # Prepare points for indexing
            points = []

            for i, (chunk, embedding_item) in enumerate(zip(chunks, embeddings)):
                # Extract metadata
                metadata = chunk.get('metadata', {})

                # Handle different possible types of embeddings
                try:
                    if isinstance(embedding_item, (int, float)):
                        logger.error(f"Embedding item {i} is a scalar value: {embedding_item}")
                        raise ValueError(f"Invalid embedding format: got scalar value at index {i}")

                    # Check if we have a numpy array
                    if hasattr(embedding_item, 'tolist'):
                        vector = embedding_item.tolist()
                        logger.debug(f"Converted numpy array to list for chunk {i}")
                    # Check if we already have a list
                    elif isinstance(embedding_item, list):
                        # Verify this is actually a vector and not another containing structure
                        if embedding_item and all(isinstance(x, (int, float)) for x in embedding_item[:5]):
                            vector = embedding_item
                            logger.debug(f"Using list embedding for chunk {i}")
                        else:
                            logger.error(f"Embedding item {i} is not a valid vector: {str(embedding_item)[:50]}")
                            raise ValueError(f"Invalid embedding format: not a numeric vector at index {i}")
                    else:
                        logger.error(f"Unknown embedding type at index {i}: {type(embedding_item)}")
                        raise ValueError(f"Unexpected embedding format: {type(embedding_item)}")

                    # Final check on vector dimensions
                    if len(vector) < 10:  # Arbitrary minimum size for a reasonable embedding
                        logger.error(f"Vector at index {i} is too small: {len(vector)} dimensions")
                        raise ValueError(f"Vector dimension too small: {len(vector)} at index {i}")
                except Exception as e:
                    logger.error(f"Error processing embedding for chunk {i}: {e}")
                    # Skip this item rather than failing the entire batch
                    continue

                try:
                    # Create the point
                    point = rest.PointStruct(
                        id=chunk.get('chunk_id', f"chunk_{i}"),
                        vector=vector,
                        payload={
                            'text': chunk.get('text', ''),  # This is the enhanced text with metadata
                            'original_text': chunk.get('original_text', chunk.get('text', '')),  # Original text only
                            'document_id': chunk.get('document_id', ''),
                            'file_name': chunk.get('file_name', ''),
                            'page_num': chunk.get('page_num', None),
                            'chunk_idx': chunk.get('chunk_idx', i),
                            'metadata': metadata
                        }
                    )

                    # Add the point to our collection
                    points.append(point)

                except Exception as e:
                    logger.error(f"Error creating point for chunk {i}: {e}")
                    # Skip problematic points
                    continue

            # Index points in batches
            batch_size = 100
            total_batches = (len(points) + batch_size - 1) // batch_size

            logger.info(f"Indexing {len(points)} points in {total_batches} batches")

            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]

                # Index batch
                client.upsert(
                    collection_name=self.qdrant_collection,
                    points=batch
                )

                logger.info(f"Indexed batch {i // batch_size + 1}/{total_batches} ({len(batch)} points)")

            logger.info(f"Indexed {len(points)} points to Qdrant")
            return True

        except Exception as e:
            logger.error(f"Error indexing to Qdrant: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _diagnose_qdrant_connection(self):
        """
        Run additional diagnostics on Qdrant connection.
        """
        logger.debug("Running Qdrant connection diagnostics...")

        # 1. Check DNS resolution
        try:
            ip_address = socket.gethostbyname(self.qdrant_host)
            logger.debug(f"DNS resolution: {self.qdrant_host} -> {ip_address}")
        except socket.gaierror as e:
            logger.debug(f"DNS resolution failed: {e}")

        # 2. Try TCP connection to the port
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5)
            result = s.connect_ex((self.qdrant_host, self.qdrant_port))
            if result == 0:
                logger.debug(f"TCP connection to {self.qdrant_host}:{self.qdrant_port} successful")
            else:
                logger.debug(f"TCP connection failed with error code: {result}")
            s.close()
        except Exception as e:
            logger.debug(f"Socket connection error: {e}")

        # 3. Check environment variables
        logger.debug(f"Environment variables:")
        for var in ['QDRANT_HOST', 'QDRANT_PORT', 'HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY']:
            logger.debug(f"  {var}: {os.environ.get(var, 'Not set')}")

        # 4. Save diagnostic info to file
        diagnostic_info = {
            'timestamp': time.time(),
            'qdrant_host': self.qdrant_host,
            'qdrant_port': self.qdrant_port,
            'qdrant_collection': self.qdrant_collection,
            'dns_resolution_attempted': True,
            'environment': {k: v for k, v in os.environ.items() if k.lower() in
                            ['qdrant_host', 'qdrant_port', 'http_proxy', 'https_proxy', 'no_proxy']}
        }

        try:
            diag_path = self.debug_dir / "qdrant_diagnostics.json"
            with open(diag_path, 'w') as f:
                json.dump(diagnostic_info, f, indent=2)
            logger.debug(f"Saved diagnostics to: {diag_path}")
        except Exception as e:
            logger.debug(f"Error saving diagnostics: {e}")

    def _index_to_bm25(self, chunks: List[Dict[str, Any]]) -> bool:
        """
        Index chunks to BM25. Uses enhanced text with metadata for better search results.

        Args:
            chunks (list): List of document chunks

        Returns:
            bool: Success status
        """
        logger.info(f"Indexing {len(chunks)} chunks to BM25")

        try:
            import nltk
            from rank_bm25 import BM25Okapi
            from nltk.tokenize import word_tokenize

            # Import our fallback tokenizers for use if needed
            from src.core.indexing.fallback_tokenizers import simple_sent_tokenize, simple_word_tokenize

            # Flag to track if we need to use fallback tokenizers
            use_fallback_tokenizers = False

            # Ensure NLTK tokenizer is available
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                logger.warning("NLTK punkt not found, attempting to download")
                try:
                    nltk.download('punkt', quiet=True)
                except Exception as e:
                    logger.warning(f"Could not download punkt: {e}. Will use fallback tokenizer.")
                    use_fallback_tokenizers = True

            # Manually download punkt if punkt_tab is used internally
            try:
                import nltk.tokenize.punkt
                if not hasattr(nltk.tokenize.punkt, '_PunktSentenceTokenizer'):
                    logger.warning("punkt_tab may be missing, attempting to download")
                    nltk.download('punkt_tab', quiet=True)
            except Exception as e:
                logger.warning(f"Error checking punkt_tab: {e}. Will try to download.")
                try:
                    nltk.download('punkt_tab', quiet=True)
                except Exception as e2:
                    logger.warning(f"Could not download punkt_tab: {e2}. Will use fallback tokenizer.")
                    use_fallback_tokenizers = True

            # Load stopwords if available, otherwise use empty set
            stop_words = set()
            try:
                from nltk.corpus import stopwords
                nltk.data.find('corpora/stopwords')
                stop_words = set(stopwords.words('english'))
            except (LookupError, ImportError):
                logger.warning("Stopwords not available. Proceeding without stopwords.")

            # Tokenize texts - IMPORTANT: Use the enhanced text with metadata here too
            logger.info("Tokenizing texts using enhanced text with metadata...")

            tokenized_texts = []
            for chunk in chunks:
                # Get the enhanced text with metadata
                text = chunk.get('text', '').lower()
                if use_fallback_tokenizers:
                    # Use our fallback tokenizer
                    tokens = simple_word_tokenize(text)
                    logger.info(f"Using fallback tokenizer: got {len(tokens)} tokens")
                else:
                    # Try NLTK's tokenizer, fall back if it fails
                    try:
                        tokens = word_tokenize(text)
                    except Exception as tokenize_err:
                        logger.warning(f"Error using word_tokenize: {tokenize_err}, falling back to custom tokenizer")
                        tokens = simple_word_tokenize(text)
                # Apply stopword filtering
                filtered_tokens = [token for token in tokens if token not in stop_words]
                tokenized_texts.append(filtered_tokens)

            # Create BM25 index
            logger.info("Creating BM25 index...")
            bm25_index = BM25Okapi(tokenized_texts)

            # Prepare metadata
            bm25_metadata = []
            for chunk in chunks:
                metadata = {
                    'chunk_id': chunk.get('chunk_id', ''),
                    'document_id': chunk.get('document_id', ''),
                    'file_name': chunk.get('file_name', ''),
                    'page_num': chunk.get('page_num', None),
                    'chunk_idx': chunk.get('chunk_idx', 0),
                    'text': chunk.get('text', ''),  # Store enhanced text with metadata
                    'original_text': chunk.get('original_text', chunk.get('text', ''))
                }
                bm25_metadata.append(metadata)

            # Save BM25 index
            logger.info(f"Saving BM25 index to {self.bm25_path}")

            # Create directory if it doesn't exist
            self.bm25_path.parent.mkdir(parents=True, exist_ok=True)

            # Save index, tokenized texts, and metadata
            with open(self.bm25_path, 'wb') as f:
                pickle.dump((bm25_index, tokenized_texts, bm25_metadata), f)

            # Save stopwords for consistency in retrieval
            stopwords_path = self.bm25_path.parent / "stopwords.pkl"
            with open(stopwords_path, 'wb') as f:
                pickle.dump(stop_words, f)

            logger.info(f"BM25 index saved successfully with {len(tokenized_texts)} documents")
            return True

        except Exception as e:
            logger.error(f"Error indexing to BM25: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False