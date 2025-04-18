# --- START OF REWRITTEN FILE: src/core/indexing/document_indexer.py ---
"""
Refactored document indexing module implementing end-to-end batching.
Handles OOM issues by processing embedding, point preparation, and Qdrant upsert
in configurable batches. Includes detailed logging and progress bars.
Supports BGE-M3 (dense, sparse) and E5/Dense models. ColBERT support removed for stability.
Uses settings from config.yaml.
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
from tqdm import tqdm # Import standard tqdm

# Qdrant models
from qdrant_client import QdrantClient, models as rest
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))

# Import utils
from src.utils.logger import setup_logger
from src.utils.resource_monitor import log_memory_usage
from src.core.indexing.fallback_tokenizers import simple_sent_tokenize, simple_word_tokenize

# Conditional imports for embedding models
try:
    from embed import BatchedInference # Assuming this is the library for E5/Dense
    BATCHED_INFERENCE_AVAILABLE = True
except ImportError:
    BATCHED_INFERENCE_AVAILABLE = False
try:
    from FlagEmbedding import BGEM3FlagModel
    FLAG_EMBEDDING_AVAILABLE = True
except ImportError:
    FLAG_EMBEDDING_AVAILABLE = False

# Import httpx for specific timeout exception
try:
    from httpx import TimeoutException as HTTPXTimeoutException
except ImportError:
    class HTTPXTimeoutException(Exception): pass

# Initialize logger
logger = setup_logger(__name__)

# Load configuration
CONFIG_PATH = ROOT_DIR / "config.yaml"
try:
    with open(CONFIG_PATH, "r") as f:
        CONFIG = yaml.safe_load(f)
except FileNotFoundError:
    logger.error(f"Configuration file not found at {CONFIG_PATH}. Using defaults.")
    CONFIG = {}
except Exception as e:
     logger.error(f"Error loading configuration: {e}. Using defaults.")
     CONFIG = {}


# --- Helper function for BGE-M3 sparse vectors ---
def _create_sparse_vector_qdrant_format(sparse_data: Dict[Union[str, int], float]) -> Optional[rest.SparseVector]:
    """Convert BGE-M3 sparse output to Qdrant sparse vector format."""
    if not sparse_data:
        return None
    sparse_indices = []
    sparse_values = []
    processed_count = 0
    skipped_non_digit = 0
    for key, value in sparse_data.items():
        try:
            float_value = float(value)
            if float_value > 0:
                if isinstance(key, str) and key.isdigit(): key_int = int(key)
                elif isinstance(key, int): key_int = key
                else: skipped_non_digit += 1; continue
                sparse_indices.append(key_int)
                sparse_values.append(float_value)
                processed_count += 1
        except (ValueError, TypeError) as e:
             logger.warning(f"Skipping invalid sparse data item: key='{key}', value='{value}', error='{e}'")
             continue
    if skipped_non_digit > 0: logger.warning(f"Skipped {skipped_non_digit} sparse vector keys that were not integers.")
    if not sparse_indices: logger.debug(f"Sparse data provided, but no valid positive entries found. Input keys sample: {list(sparse_data.keys())[:10]}"); return None
    logger.debug(f"Created sparse vector with {len(sparse_indices)} non-zero elements.")
    return rest.SparseVector(indices=sparse_indices, values=sparse_values)


# --- Main DocumentIndexer Class ---
class DocumentIndexer:
    """
    Indexes documents to BM25 and Qdrant using end-to-end batching for memory efficiency.
    Supports BGE-M3 (dense+sparse) and E5 (dense only). ColBERT is disabled.
    """

    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode or CONFIG.get("debug_mode", False) # Allow config override

        # --- Qdrant Configuration ---
        qdrant_config = CONFIG.get("qdrant", {})
        self.qdrant_host = qdrant_config.get("host", "localhost")
        self.qdrant_port = qdrant_config.get("port", 6333)
        self.qdrant_collection = qdrant_config.get("collection_name", "default_collection")
        self.qdrant_timeout = qdrant_config.get("timeout", 900)
        self.qdrant_upsert_batch_size = qdrant_config.get("upsert_batch_size", 128)
        self.qdrant_sparse_on_disk = qdrant_config.get("sparse_on_disk", True)

        # --- Indexing Configuration ---
        indexing_config = CONFIG.get("indexing", {})
        self.processing_batch_size = indexing_config.get("processing_batch_size", 64) # Batch for embed/prep
        self.fallback_to_bm25_only = indexing_config.get("fallback_to_bm25_only", True)
        self.auto_recreate_collection = indexing_config.get("auto_recreate_collection", False)

        # --- Embedding Model Configuration ---
        models_config = CONFIG.get("models", {})
        self.embedding_model_name = models_config.get("embedding_model", "BAAI/bge-m3") # Default to BGE-M3
        self.is_bge_m3 = "bge-m3" in self.embedding_model_name.lower()
        self.expected_vector_size = qdrant_config.get("vector_size", 1024) # Use Qdrant config first
        self.bge_m3_batch_size = models_config.get("bge_m3_batch_size", 12) # Batch size for BGE model itself
        self.bge_m3_max_length = models_config.get("bge_m3_max_length", 8192)

        # --- BM25 Configuration ---
        storage_config = CONFIG.get("storage", {})
        self.bm25_path = ROOT_DIR / storage_config.get("bm25_index_path", "data/bm25/index.pkl")
        self.bm25_path.parent.mkdir(parents=True, exist_ok=True)

        # --- Debug Directory ---
        self.debug_dir = ROOT_DIR / "debug"
        if self.debug_mode:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

        # --- Internal State ---
        self.embedding_model_instance = None
        self._validate_config()

        logger.info(f"Initializing DocumentIndexer:")
        logger.info(f"  - Processing Batch Size: {self.processing_batch_size}")
        logger.info(f"  - Embedding Model: {self.embedding_model_name} ({'BGE-M3 Mode (Dense + Sparse)' if self.is_bge_m3 else 'E5/Dense Mode'})")
        logger.info(f"  - Qdrant:")
        logger.info(f"    - Host: {self.qdrant_host}:{self.qdrant_port}")
        logger.info(f"    - Collection: {self.qdrant_collection}")
        logger.info(f"    - Timeout: {self.qdrant_timeout}s")
        logger.info(f"    - Upsert Batch Size: {self.qdrant_upsert_batch_size}")
        if self.is_bge_m3: logger.info(f"    - Sparse On Disk: {self.qdrant_sparse_on_disk}")
        logger.info(f"  - Expected Vector Size: {self.expected_vector_size}")
        logger.info(f"  - Auto Recreate Collection: {self.auto_recreate_collection}")
        logger.info(f"  - Debug Mode: {self.debug_mode}")

        if self.debug_mode: self._log_debug_info()
        log_memory_usage(logger)

    def _validate_config(self):
        """Validate config and library availability."""
        try: # Basic connectivity check hint
            if self.qdrant_host not in ('localhost', '127.0.0.1'): socket.gethostbyname(self.qdrant_host)
        except Exception as e: logger.warning(f"Unable to resolve Qdrant hostname '{self.qdrant_host}': {e}")
        if not os.path.isabs(str(self.bm25_path)): logger.warning(f"BM25 path {self.bm25_path} is not absolute.")

        if self.is_bge_m3 and not FLAG_EMBEDDING_AVAILABLE:
             logger.error("BGE-M3 specified, but 'FlagEmbedding' not installed. `pip install FlagEmbedding`")
             raise ImportError("FlagEmbedding library not installed.")
        if not self.is_bge_m3 and not BATCHED_INFERENCE_AVAILABLE:
             logger.error("E5/Dense model specified, but 'embed' library (BatchedInference) not found.")
             raise ImportError("BatchedInference library not installed.")

    def _log_debug_info(self):
        """Log detailed debugging information."""
        logger.debug(f"--- Debug Info ---")
        logger.debug(f"Config file path: {CONFIG_PATH}")
        logger.debug(f"Root directory: {ROOT_DIR}")
        logger.debug(f"BM25 index path: {self.bm25_path}")
        logger.debug(f"Debug directory: {self.debug_dir}")
        logger.debug(f"Qdrant connection: {self.qdrant_host}:{self.qdrant_port}, Timeout: {self.qdrant_timeout}")
        try: logger.debug(f"Qdrant host resolution: {self.qdrant_host} -> {socket.gethostbyname(self.qdrant_host)}")
        except Exception as e: logger.debug(f"Qdrant host resolution failed: {e}")
        debug_config_path = self.debug_dir / "config_dump_indexer.json"
        try:
            def safe_default(obj): return str(obj) if isinstance(obj, Path) else f"<<Non-serializable: {type(obj).__name__}>>"
            with open(debug_config_path, 'w') as f: json.dump(CONFIG, f, indent=2, default=safe_default)
            logger.debug(f"Config dump saved to: {debug_config_path}")
        except Exception as e: logger.error(f"Failed to save config dump: {e}")
        logger.debug(f"--- End Debug Info ---")

    def load_model(self):
        """Load the embedding model based on configuration."""
        if self.embedding_model_instance is not None:
            logger.info(f"Embedding model ({self.embedding_model_name}) already loaded.")
            return
        logger.info(f"===== LOADING EMBEDDING MODEL: {self.embedding_model_name} =====")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        start_time = time.time()
        try:
            if self.is_bge_m3:
                if not FLAG_EMBEDDING_AVAILABLE: raise ImportError("FlagEmbedding not installed")
                use_fp16 = torch.cuda.is_available() # Use FP16 only if CUDA is available
                logger.info(f"Loading BGE-M3 with use_fp16={use_fp16}")
                self.embedding_model_instance = BGEM3FlagModel(self.embedding_model_name, use_fp16=use_fp16, device=device)
                logger.info("Loaded BGE-M3 model using FlagEmbedding.")
                try: # Update expected vector size from loaded model
                    model_dim = self.embedding_model_instance.model.config.hidden_size
                    if model_dim != self.expected_vector_size:
                        logger.warning(f"Overriding expected vector size from config ({self.expected_vector_size}) with loaded BGE-M3 dimension: {model_dim}")
                        self.expected_vector_size = model_dim
                except Exception as e: logger.warning(f"Could not get vector size from BGE-M3 model config: {e}")
            else: # E5/Dense Mode
                if not BATCHED_INFERENCE_AVAILABLE: raise ImportError("BatchedInference not installed")
                self.embedding_model_instance = BatchedInference(model_id=self.embedding_model_name, engine="torch", device=device)
                logger.info("Loaded E5/Dense model using BatchedInference.")
                # Attempt to get size for E5 if library provides it, otherwise rely on config
                # e.g., if hasattr(self.embedding_model_instance, 'dim'): self.expected_vector_size = self.embedding_model_instance.dim

            elapsed_time = time.time() - start_time
            logger.info(f"Embedding model loaded in {elapsed_time:.2f} seconds")
            log_memory_usage(logger)
        except Exception as e:
            logger.error(f"Error loading embedding model {self.embedding_model_name}: {e}", exc_info=True)
            self.embedding_model_instance = None; raise

    def shutdown(self):
        """Unload the embedding model to free up resources."""
        if self.embedding_model_instance is not None:
            logger.info("===== STARTING EMBEDDING MODEL UNLOAD =====")
            model_name = self.embedding_model_name
            try:
                if self.is_bge_m3 and hasattr(self.embedding_model_instance, 'model'):
                    logger.info("Unloading BGE-M3 (FlagEmbedding) model components...")
                    if hasattr(self.embedding_model_instance.model, 'cpu'):
                        try: self.embedding_model_instance.model.cpu(); logger.debug("Moved BGE-M3 model to CPU.")
                        except Exception as cpu_err: logger.warning(f"Could not move BGE-M3 model to CPU: {cpu_err}")
                    if hasattr(self.embedding_model_instance, 'model'): del self.embedding_model_instance.model
                    if hasattr(self.embedding_model_instance, 'tokenizer'): del self.embedding_model_instance.tokenizer
                    logger.info("Deleted BGE-M3 model/tokenizer references.")
                elif not self.is_bge_m3 and hasattr(self.embedding_model_instance, 'stop'):
                    logger.info("Calling BatchedInference stop()...")
                    self.embedding_model_instance.stop(); logger.info("BatchedInference stopped.")
                else: logger.warning(f"No specific unload method found for model type of {model_name}.")
                del self.embedding_model_instance
                self.embedding_model_instance = None
                if torch.cuda.is_available(): logger.info("Clearing CUDA cache..."); torch.cuda.empty_cache()
                logger.info("Running garbage collection..."); gc.collect()
                logger.info("===== EMBEDDING MODEL UNLOAD COMPLETE =====")
                log_memory_usage(logger)
            except Exception as e:
                 logger.error(f"Error during model shutdown for {model_name}: {e}", exc_info=True)
                 self.embedding_model_instance = None
        else: logger.info("Embedding model already unloaded or not loaded.")

    def _generate_embeddings(self, texts: List[str]) -> Union[Dict[str, Any], List[np.ndarray]]:
        """
        Generate embeddings for a list of texts based on the loaded model type.
        For BGE-M3, returns dense and sparse. ColBERT is disabled.
        For E5/Dense, returns only dense.
        """
        if self.embedding_model_instance is None: raise RuntimeError("Embedding model not loaded.")
        if not texts: return {} if self.is_bge_m3 else [] # Handle empty input

        start_time_emb = time.time()
        logger.info(f"Generating embeddings for {len(texts)} texts using {self.embedding_model_name}...")
        try:
            if self.is_bge_m3:
                # *** MODIFIED: Disable ColBERT ***
                output = self.embedding_model_instance.encode(
                    texts, batch_size=self.bge_m3_batch_size, max_length=self.bge_m3_max_length,
                    return_dense=True, return_sparse=True, return_colbert_vecs=False # <-- Changed
                )
                # --- Validation (Updated) ---
                if not isinstance(output, dict): raise TypeError(f"BGE-M3 output type error: {type(output)}")
                # *** MODIFIED: Remove ColBERT from validation ***
                missing = [k for k in ['dense_vecs', 'lexical_weights'] if k not in output] # <-- Changed
                if missing: raise ValueError(f"BGE-M3 output missing keys: {missing}")
                # *** MODIFIED: Remove ColBERT from validation loop ***
                for key in ['dense_vecs', 'lexical_weights']: # <-- Changed
                    val = output.get(key)
                    if val is None or len(val) != len(texts):
                        v_len = len(val) if val is not None else 'None'
                        v_type = type(val)
                        msg = f"BGE-M3 '{key}' output error. Expected len {len(texts)}, Got {v_len} (Type: {v_type})"
                        logger.error(msg)
                        raise ValueError(msg)
                logger.info(f"BGE-M3 embeddings (dense, sparse) generated in {time.time() - start_time_emb:.2f}s")
                return output
            else: # E5/Dense
                result = self.embedding_model_instance.embed(sentences=texts) # Assuming embed method
                if hasattr(result, 'result') and callable(result.result): embeddings = result.result()
                else: embeddings = result
                dense_embeddings = embeddings[0] if isinstance(embeddings, tuple) and len(embeddings) >= 1 else embeddings

                if not isinstance(dense_embeddings, (list, np.ndarray)) or len(dense_embeddings) != len(texts):
                    raise ValueError(f"E5 output length mismatch: Got {len(dense_embeddings)} (Type: {type(dense_embeddings)}), Expected {len(texts)}")
                if len(dense_embeddings) > 0:
                    vec_len = len(dense_embeddings[0])
                    if vec_len != self.expected_vector_size:
                         logger.warning(f"E5 vector size mismatch: Got {vec_len}, Expected {self.expected_vector_size}.")
                logger.info(f"E5/Dense embeddings generated in {time.time() - start_time_emb:.2f}s")
                return dense_embeddings
        except Exception as e:
             logger.error(f"Error during embedding generation: {e}", exc_info=True)
             raise

    def _get_qdrant_vectors_config(self) -> Union[rest.VectorParams, Dict[str, rest.VectorParams]]:
        """
        Get the appropriate Qdrant vector configuration based on the model.
        BGE-M3 uses named "dense" vector. E5 uses default vector. ColBERT removed.
        """
        distance = rest.Distance.COSINE
        if self.is_bge_m3:
            # *** MODIFIED: Remove ColBERT ***
            return {
                "dense": rest.VectorParams(size=self.expected_vector_size, distance=distance),
                # "colbert": removed
            }
        else: # E5/Dense - Uses the default, unnamed vector configuration
            return rest.VectorParams(size=self.expected_vector_size, distance=distance)

    def _get_qdrant_sparse_config(self) -> Optional[Dict[str, rest.SparseVectorParams]]:
         """Get sparse vector config, only applicable for BGE-M3."""
         if self.is_bge_m3:
              logger.info(f"Configuring sparse vectors with on_disk={self.qdrant_sparse_on_disk}")
              return {"sparse": rest.SparseVectorParams(index=rest.SparseIndexParams(on_disk=self.qdrant_sparse_on_disk))}
         return None # No sparse vectors for E5

    def _ensure_qdrant_collection(self, client: QdrantClient) -> bool:
        """
        Checks if collection exists and has compatible config, recreates if necessary and allowed.
        Handles different schemas for BGE-M3 (dense+sparse) and E5 (dense only).
        """
        try:
            logger.debug(f"Ensuring Qdrant collection '{self.qdrant_collection}' exists and is compatible...")
            collections = client.get_collections().collections
            collection_exists = any(c.name == self.qdrant_collection for c in collections)
            needs_recreate = False
            target_vectors_config = self._get_qdrant_vectors_config()
            target_sparse_config = self._get_qdrant_sparse_config()

            if collection_exists:
                logger.info(f"Collection '{self.qdrant_collection}' exists. Verifying configuration...")
                collection_info = client.get_collection(collection_name=self.qdrant_collection)
                current_vectors_config = collection_info.config.params.vectors
                current_sparse_config = collection_info.config.params.sparse_vectors

                # --- Configuration Comparison Logic ---
                if self.is_bge_m3:
                    # Check if current config is a dict and has "dense" key (as required by target_vectors_config)
                    if not isinstance(current_vectors_config, dict) or "dense" not in current_vectors_config:
                        logger.warning("Collection vector config type is not Dict or missing 'dense' key, incompatible with BGE-M3 (dense+sparse).")
                        needs_recreate = True
                    # Check if "colbert" key exists unexpectedly (since we removed it)
                    elif "colbert" in current_vectors_config:
                         logger.warning("Collection has 'colbert' vector config, but ColBERT is now disabled.")
                         needs_recreate = True
                    # Check dense vector size
                    elif current_vectors_config["dense"].size != self.expected_vector_size:
                        logger.warning(f"Collection vector size mismatch (Dense:{current_vectors_config['dense'].size} vs Expected:{self.expected_vector_size}).")
                        needs_recreate = True
                    # Check sparse vector presence/absence
                    elif target_sparse_config and (not current_sparse_config or "sparse" not in current_sparse_config):
                        logger.warning("Collection missing required 'sparse' vector config for BGE-M3.")
                        needs_recreate = True
                    elif not target_sparse_config and current_sparse_config:
                         # This case shouldn't happen with current logic, but good to have
                         logger.warning("Collection has sparse config, but BGE-M3 sparse vectors are not configured (should be).")
                         needs_recreate = True
                    # Optional: Check sparse index params if needed (e.g., on_disk)
                    elif target_sparse_config and current_sparse_config and target_sparse_config["sparse"].index.on_disk != current_sparse_config["sparse"].index.on_disk:
                         logger.warning(f"Collection sparse index 'on_disk' mismatch (Current: {current_sparse_config['sparse'].index.on_disk} vs Target: {target_sparse_config['sparse'].index.on_disk}).")
                         needs_recreate = True

                else: # E5/Dense
                    # Check if current config is NOT a dict (E5 uses default unnamed vector)
                    if isinstance(current_vectors_config, dict):
                        logger.warning("Collection has named vector config (like BGE-M3), expected default vector for E5.")
                        needs_recreate = True
                    # Check vector size
                    elif current_vectors_config.size != self.expected_vector_size:
                        logger.warning(f"Collection vector size mismatch ({current_vectors_config.size} vs {self.expected_vector_size}).")
                        needs_recreate = True
                    # Check if sparse config exists unexpectedly
                    elif current_sparse_config:
                        logger.warning("Collection has sparse config, but current model (E5) is dense-only.")
                        needs_recreate = True
                    # Optional: Check distance metric

                if needs_recreate:
                    logger.warning(f"Recreation needed for collection '{self.qdrant_collection}'.")
                    if not self.auto_recreate_collection:
                        logger.error("Collection needs recreation, but 'auto_recreate_collection' is false. Aborting indexing.")
                        return False
                    logger.info(f"Recreating collection '{self.qdrant_collection}'...")
                    client.recreate_collection(
                        collection_name=self.qdrant_collection,
                        vectors_config=target_vectors_config,
                        sparse_vectors_config=target_sparse_config, # Pass None if E5
                        timeout=self.qdrant_timeout + 60 # Allow extra time for recreation
                    )
                    logger.info("Collection recreated.")
            else: # Collection doesn't exist
                logger.info(f"Creating collection '{self.qdrant_collection}'...")
                client.create_collection(
                    collection_name=self.qdrant_collection,
                    vectors_config=target_vectors_config,
                    sparse_vectors_config=target_sparse_config # Pass None if E5
                )
                logger.info("Collection created.")
            return True
        except Exception as e:
            logger.error(f"Error ensuring Qdrant collection: {e}", exc_info=True)
            return False


    def _process_and_index_qdrant_batch(self, processing_batch_chunks: List[Dict[str, Any]]) -> bool:
        """
        Processes a single batch: generates embeddings, prepares points, and upserts to Qdrant.
        Handles BGE-M3 (dense+sparse) and E5 (dense) point structures. ColBERT removed.
        """
        if not processing_batch_chunks:
            logger.debug("Empty chunk batch received, skipping Qdrant processing.")
            return True
        if self.embedding_model_instance is None:
             logger.error("Embedding model not loaded for Qdrant batch processing!")
             return False

        batch_start_time = time.time()
        num_chunks_in_batch = len(processing_batch_chunks)
        logger.info(f"Processing Qdrant batch with {num_chunks_in_batch} chunks...")

        try:
            # --- 1. Generate Embeddings for THIS Processing Batch ---
            texts = [chunk.get('text', '') for chunk in processing_batch_chunks]
            if not any(texts): logger.warning("Processing batch contains no text, skipping embedding."); return True
            embeddings_output = self._generate_embeddings(texts) # Handles both types

            # --- 2. Prepare Points for THIS Processing Batch ---
            points_to_upsert = []
            logger.info(f"Preparing {num_chunks_in_batch} points for Qdrant upsert...")
            skipped_count = 0
            preparation_start_time = time.time()

            for i, chunk in enumerate(processing_batch_chunks):
                chunk_id = chunk.get('chunk_id', f"generated_chunk_{batch_start_time}_{i}")
                payload = {'text': chunk.get('text', ''),'original_text': chunk.get('original_text', ''),
                           'document_id': chunk.get('document_id', ''), 'file_name': chunk.get('file_name', ''),
                           'page_num': chunk.get('page_num', None), 'chunk_idx': chunk.get('chunk_idx', i),
                           'metadata': chunk.get('metadata', {})}
                if 'file_name' in payload and 'file_name' not in payload['metadata']: payload['metadata']['file_name'] = payload['file_name']

                vector_payload = None # Initialize vector_payload
                text_length = len(payload.get('text', ''))
                sparse_vector_size = 0
                dense_ok = False
                try:
                    if self.is_bge_m3:
                        # *** MODIFIED: Get only dense and sparse ***
                        dense_vec = embeddings_output['dense_vecs'][i]
                        sparse_weights = embeddings_output['lexical_weights'][i]
                        # colbert_vecs removed

                        if dense_vec is None or len(dense_vec) != self.expected_vector_size:
                            raise ValueError(f"BGE-M3 Dense vector size mismatch/None")

                        # *** MODIFIED: Construct named vector payload ***
                        vector_payload = {"dense": dense_vec.tolist()} # Start with dense
                        qdrant_sparse = _create_sparse_vector_qdrant_format(sparse_weights)
                        if qdrant_sparse:
                            vector_payload["sparse"] = qdrant_sparse # Add sparse if valid
                            sparse_vector_size = len(qdrant_sparse.indices)
                        else:
                             # Optionally log if sparse vector creation failed for a chunk
                             logger.debug(f"Chunk {chunk_id}: No valid sparse vector created.")
                        # colbert vector removed
                        dense_ok = True
                        logger.debug(f"Prepared BGE-M3 point - ID: {chunk_id}, TextLen: {text_length}, DenseOK: {dense_ok}, SparseSize: {sparse_vector_size}")

                    else: # E5/Dense
                        dense_vec = embeddings_output[i]
                        if dense_vec is None or len(dense_vec) != self.expected_vector_size:
                            raise ValueError(f"E5 Dense vector size mismatch/None")
                        # *** MODIFIED: E5 uses default unnamed vector ***
                        vector_payload = list(dense_vec)
                        dense_ok = True
                        logger.debug(f"Prepared E5 point - ID: {chunk_id}, TextLen: {text_length}, DenseOK: {dense_ok}")

                    # Create PointStruct with the appropriate vector_payload format
                    points_to_upsert.append(rest.PointStruct(id=chunk_id, vector=vector_payload, payload=payload))

                except Exception as point_err:
                     logger.error(f"Error preparing chunk {chunk_id} for upsert: {point_err}", exc_info=False)
                     skipped_count += 1; continue
            preparation_duration = time.time() - preparation_start_time
            logger.info(f"Point preparation finished in {preparation_duration:.2f}s. Prepared: {len(points_to_upsert)}, Skipped: {skipped_count}")
            if skipped_count > 0: logger.warning(f"Skipped {skipped_count} chunks in this batch due to errors.")
            if not points_to_upsert: logger.warning("No valid points were prepared for this Qdrant batch."); return True

            # --- 3. Upsert THIS Batch to Qdrant (using internal client batches) ---
            client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port, timeout=self.qdrant_timeout)

            total_qdrant_batches = (len(points_to_upsert) + self.qdrant_upsert_batch_size - 1) // self.qdrant_upsert_batch_size
            logger.info(f"Upserting {len(points_to_upsert)} points to Qdrant server in {total_qdrant_batches} client batches (size: {self.qdrant_upsert_batch_size})...")

            batch_success = True # Assume success for the whole processing batch unless a client batch fails
            for i in tqdm(range(0, len(points_to_upsert), self.qdrant_upsert_batch_size),
                          desc=f"Upserting Qdrant Batches", total=total_qdrant_batches, unit="client_batch", leave=False, ncols=100):
                client_batch = points_to_upsert[i : i + self.qdrant_upsert_batch_size]
                client_batch_num = (i // self.qdrant_upsert_batch_size) + 1
                first_id = client_batch[0].id if client_batch else "N/A"
                logger.info(f"--> Sending Qdrant client batch {client_batch_num}/{total_qdrant_batches} ({len(client_batch)} points), starting with ID: {first_id}...")
                start_upsert_time = time.time()
                try:
                    response = client.upsert(collection_name=self.qdrant_collection, points=client_batch, wait=True)
                    end_upsert_time = time.time()
                    if response and response.status == rest.UpdateStatus.COMPLETED:
                         logger.info(f"<-- Successfully upserted Qdrant client batch {client_batch_num} in {end_upsert_time - start_upsert_time:.2f}s")
                    else:
                         logger.warning(f"<-- Qdrant client batch {client_batch_num} status not COMPLETED: {response.status if response else 'No response'}. Time: {end_upsert_time - start_upsert_time:.2f}s")
                         batch_success = False # Mark failure if any client batch fails
                except (HTTPXTimeoutException, ResponseHandlingException, ConnectionError, UnexpectedResponse) as net_err: # Catch more potential errors
                     logger.error(f"NETWORK/TIMEOUT ERROR upserting Qdrant client batch {client_batch_num}: {type(net_err).__name__} - {net_err}", exc_info=True)
                     batch_success = False; break # Stop processing this outer batch on network/timeout error
                except Exception as upsert_err:
                     logger.error(f"GENERAL ERROR upserting Qdrant client batch {client_batch_num}: {upsert_err}", exc_info=True)
                     batch_success = False; break # Stop processing this outer batch on general error

            # Log memory usage after a processing batch is done
            log_memory_usage(logger)
            gc.collect() # Explicitly request garbage collection

            total_batch_duration = time.time() - batch_start_time
            logger.info(f"Finished processing Qdrant batch in {total_batch_duration:.2f}s. Overall success for this batch: {batch_success}")
            return batch_success

        except Exception as e:
            logger.error(f"Unexpected error during single batch Qdrant processing: {e}", exc_info=True)
            return False

    def index_documents(self, all_chunks: List[Dict[str, Any]]) -> None:
        """
        Index document chunks to BM25 and Qdrant using end-to-end batching.
        Ensures Qdrant collection schema matches the selected embedding model (BGE-M3 or E5).
        """
        if not all_chunks: logger.warning("No chunks to index"); return
        num_chunks = len(all_chunks)
        logger.info(f"===== STARTING INDEXING PROCESS FOR {num_chunks} CHUNKS =====")
        start_indexing_time = time.time()

        # --- Debug Info ---
        if self.debug_mode:
            chunks_path = self.debug_dir / f"chunks_to_index_sample_{num_chunks}.json"
            try:
                minimal_chunks = [{'index': idx, 'chunk_id': c.get('chunk_id', f'missing_{idx}'),
                                   'text_length': len(c.get('text', '')), 'metadata_keys': list(c.get('metadata', {}).keys())}
                                  for idx, c in enumerate(all_chunks[:min(num_chunks, 50)])]
                with open(chunks_path, 'w') as f: json.dump(minimal_chunks, f, indent=2)
                logger.debug(f"Saved chunk info sample (first 50) to: {chunks_path}")
            except Exception as e: logger.error(f"Error saving chunks debug info: {e}")

        try:
            # --- 1. Load Model Once ---
            self.load_model()
            if self.embedding_model_instance is None: logger.error("Embedding model failed to load. Aborting."); return

            # --- 2. BM25 Indexing (Process all chunks at once) ---
            # Only relevant if E5 model might be used.
            logger.info("Starting BM25 indexing...")
            start_bm25_time = time.time()
            bm25_success = self._index_to_bm25(all_chunks)
            end_bm25_time = time.time()
            logger.info(f"BM25 indexing took {end_bm25_time - start_bm25_time:.2f}s (Success: {bm25_success})")

            # --- 3. Qdrant Indexing (Process in batches) ---
            overall_qdrant_success = True # Track success across all batches
            qdrant_processing_start_time = time.time()

            # Ensure collection exists AND IS COMPATIBLE before starting batch processing
            try:
                 client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port, timeout=self.qdrant_timeout)
                 if not self._ensure_qdrant_collection(client): # This now handles schema validation
                      logger.error("Failed to ensure Qdrant collection exists or is compatible with the current model. Aborting Qdrant indexing.")
                      overall_qdrant_success = False
                 del client # Close connection
                 gc.collect()
            except Exception as client_err:
                 logger.error(f"Failed to connect to Qdrant or ensure collection: {client_err}", exc_info=True)
                 overall_qdrant_success = False

            if overall_qdrant_success: # Only proceed if collection is okay
                logger.info(f"Starting Qdrant processing in batches of {self.processing_batch_size}...")
                total_processing_batches = (num_chunks + self.processing_batch_size - 1) // self.processing_batch_size

                for i in tqdm(range(0, num_chunks, self.processing_batch_size),
                              desc="Processing Chunks", total=total_processing_batches, unit="batch", ncols=100):
                    batch_chunks = all_chunks[i : i + self.processing_batch_size]
                    processing_batch_num = (i // self.processing_batch_size) + 1
                    logger.info(f"--- Starting Processing Batch {processing_batch_num}/{total_processing_batches} ({len(batch_chunks)} chunks) ---")

                    if not batch_chunks: continue

                    qdrant_batch_success = self._process_and_index_qdrant_batch(batch_chunks)

                    if not qdrant_batch_success:
                        overall_qdrant_success = False
                        logger.error(f"Failed to process Qdrant batch {processing_batch_num}. Check logs for details.")
                        if not self.fallback_to_bm25_only:
                            logger.error("Fallback to BM25 is disabled. Aborting remaining batches.")
                            break # Stop processing more batches if one fails and no fallback
                        else:
                             logger.warning("Fallback to BM25 is enabled. Continuing with next batch despite Qdrant failure.")
            else:
                 # This case handles if _ensure_qdrant_collection failed
                 logger.error("Skipping Qdrant processing due to earlier collection error.")
                 if not self.fallback_to_bm25_only:
                     logger.error("Qdrant skipped and fallback disabled. Indexing considered failed.")
                     # If BM25 succeeded earlier, the final status will reflect partial success.


            qdrant_processing_duration = time.time() - qdrant_processing_start_time
            logger.info(f"Qdrant processing finished in {qdrant_processing_duration:.2f}s (Overall Success: {overall_qdrant_success})")

            # --- Final Status Logging ---
            end_indexing_time = time.time()
            total_time = end_indexing_time - start_indexing_time
            logger.info(f"===== INDEXING PROCESS COMPLETE IN {total_time:.2f}s =====")
            if overall_qdrant_success and bm25_success:
                logger.info(f"RESULT: Successfully indexed {num_chunks} chunks to Qdrant and BM25.")
            elif bm25_success and not overall_qdrant_success:
                if self.fallback_to_bm25_only:
                    logger.warning(f"RESULT (Fallback): Indexed {num_chunks} chunks to BM25 only due to Qdrant errors.")
                else:
                    logger.error(f"RESULT (FAILED): BM25 succeeded but Qdrant failed (fallback disabled).")
            elif not bm25_success and overall_qdrant_success:
                 logger.error(f"RESULT (FAILED): Qdrant succeeded but BM25 failed. Check BM25 logs.")
            else: # Neither succeeded fully
                logger.error(f"RESULT (FAILED): Neither Qdrant nor BM25 succeeded completely.")

        except Exception as e:
            logger.error(f"Fatal error during index_documents orchestration: {e}", exc_info=True)
        finally:
             # Ensure model is unloaded after the whole indexing run
             self.shutdown()
             logger.info("Index_documents call finished.")


    def _index_to_bm25(self, chunks: List[Dict[str, Any]]) -> bool:
        """Index chunks to BM25. (Assumes processing all chunks at once is feasible)."""
        logger.info(f"Starting BM25 indexing for {len(chunks)} chunks...")
        if not chunks: logger.warning("No chunks provided for BM25 indexing."); return True
        try:
            from rank_bm25 import BM25Okapi
            use_fallback = False
            try: nltk.data.find('tokenizers/punkt'); nltk.data.find('corpora/stopwords')
            except LookupError:
                logger.warning("NLTK punkt/stopwords not found. Downloading...")
                try: nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)
                except Exception as e: logger.warning(f"NLTK download failed: {e}. Using fallback."); use_fallback = True
            stop_words = set() if use_fallback else set(nltk.corpus.stopwords.words('english'))

            logger.info("Tokenizing texts for BM25...")
            tokenized_texts = []
            for chunk in tqdm(chunks, desc="Tokenizing for BM25", unit="chunk", ncols=100, leave=False):
                text = chunk.get('text', '').lower()
                if not text: tokenized_texts.append([]); continue
                tokens = simple_word_tokenize(text) if use_fallback else nltk.tokenize.word_tokenize(text)
                filtered = [t for t in tokens if t.isalnum() and t not in stop_words]
                tokenized_texts.append(filtered if filtered else (tokens if tokens else [])) # Keep original if filter removes all

            logger.info("Creating BM25 index object...")
            bm25_index = BM25Okapi(tokenized_texts)
            logger.info("Preparing BM25 metadata...")
            # Store the tokenized texts along with the index and metadata
            bm25_metadata = [{'chunk_id': c.get('chunk_id',''), 'document_id': c.get('document_id',''),
                              'file_name': c.get('file_name',''), 'page_num': c.get('page_num'), 'chunk_idx': c.get('chunk_idx'),
                              'text': c.get('text', ''), # Store the full text used for tokenization
                              'original_text': c.get('original_text', '')} # Store original text if needed
                             for c in chunks]

            logger.info(f"Saving BM25 index, tokenized texts, and metadata to {self.bm25_path}")
            self.bm25_path.parent.mkdir(parents=True, exist_ok=True)
            # Save all three components: index, tokenized texts, metadata
            with open(self.bm25_path, 'wb') as f:
                pickle.dump((bm25_index, tokenized_texts, bm25_metadata), f)

            logger.info(f"BM25 index saved successfully ({len(tokenized_texts)} documents).")
            return True
        except Exception as e:
            logger.error(f"Error indexing to BM25: {e}", exc_info=True)
            return False

# --- END OF REWRITTEN FILE: src/core/indexing/document_indexer.py ---