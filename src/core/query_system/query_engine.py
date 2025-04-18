# --- START OF REWRITTEN FILE: src/core/query_system/query_engine.py ---
"""
Query engine for Anti-Corruption RAG System.
Handles retrieval and generation using persistent Aphrodite LLM service.
Supports BGE-M3 (dense+sparse hybrid search) and E5-Instruct (dense + BM25 fusion).
ColBERT support removed. Applies cross-encoder reranking universally.
Loads and uses dedicated chat tokenizer for LLM generation.
"""
import sys
import os
from pathlib import Path
import yaml
import pickle
import torch
import gc
import time
import uuid
import numpy as np
from typing import List, Dict, Any, Union, Optional, Tuple, Generator
import traceback

# Qdrant models
from qdrant_client import QdrantClient, models as rest

# Hugging Face Tokenizer
from transformers import AutoTokenizer

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))

# Import utils
from src.utils.logger import setup_logger
from src.utils.resource_monitor import log_memory_usage
from src.utils.qdrant_utils import scroll_with_filter_compatible
from src.utils.aphrodite_service import get_service

# Conditional imports for embedding models
try:
    from embed import BatchedInference
    BATCHED_INFERENCE_AVAILABLE = True
except ImportError:
    BATCHED_INFERENCE_AVAILABLE = False
try:
    from FlagEmbedding import BGEM3FlagModel
    FLAG_EMBEDDING_AVAILABLE = True
except ImportError:
    FLAG_EMBEDDING_AVAILABLE = False


# Initialize logger
logger = setup_logger(__name__)

# Load configuration
CONFIG_PATH = ROOT_DIR / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

# --- E5-INSTRUCT TASK DEFINITION (Only used if E5 model is loaded) ---
RETRIEVAL_TASK_INSTRUCTION = 'Given a web search query, retrieve relevant passages that answer the query'

# Helper function for BGE-M3 sparse vectors (copied from indexer)
def _create_sparse_vector_qdrant_format(sparse_data: Dict[Union[str, int], float]) -> Optional[rest.SparseVector]:
    """Convert BGE-M3 sparse output to Qdrant sparse vector format."""
    if not sparse_data:
        return None
    sparse_indices = []
    sparse_values = []
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
        except (ValueError, TypeError): continue
    if skipped_non_digit > 0: logger.warning(f"Skipped {skipped_non_digit} non-integer sparse keys.")
    if not sparse_indices: return None
    return rest.SparseVector(indices=sparse_indices, values=sparse_values)


class QueryEngine:
    """
    Query engine supporting BGE-M3 (dense+sparse) and E5-Instruct (dense+BM25) models.
    Uses a persistent Aphrodite service for generation.
    Correctly loads and uses chat tokenizer for LLM prompt formatting.
    """

    def __init__(self):
        """Initialize query engine."""
        # Qdrant Config
        self.qdrant_host = CONFIG["qdrant"]["host"]
        self.qdrant_port = CONFIG["qdrant"]["port"]
        self.qdrant_collection = CONFIG["qdrant"]["collection_name"]

        # Retrieval Config
        self.top_k_vector = CONFIG["retrieval"]["top_k_vector"]
        self.top_k_bm25 = CONFIG["retrieval"]["top_k_bm25"] # Used for E5 path
        self.top_k_hybrid = CONFIG["retrieval"]["top_k_hybrid"] # Used for fusion in both paths
        self.top_k_rerank = CONFIG["retrieval"]["top_k_rerank"]
        # Weights used for E5 (dense+bm25) and BGE-M3 (dense+sparse) fusion via RRF
        self.vector_weight = CONFIG["retrieval"]["vector_weight"]
        self.bm25_weight = CONFIG["retrieval"]["bm25_weight"] # Also used as sparse weight for BGE-M3 fusion
        self.use_reranking = CONFIG["retrieval"]["use_reranking"]
        self.min_score_threshold = CONFIG["retrieval"].get("minimum_score_threshold", 0.01)
        # Removed bge_m3_prefetch_limit as ColBERT prefetch is removed

        # Model Config
        self.llm_model_name = CONFIG["models"]["chat_model"]
        self.embedding_model_name = CONFIG["models"]["embedding_model"]
        self.reranking_model_name = CONFIG["models"].get("reranking_model", "BAAI/bge-reranker-v2-m3")

        # State
        self.is_bge_m3 = "bge-m3" in self.embedding_model_name.lower()
        self.embedding_model_instance = None
        self.reranker_model_instance = None
        self.chat_tokenizer = None
        self.aphrodite_service = get_service()
        self.bm25_index = None
        self.tokenized_texts = None # Added for BM25 loading
        self.bm25_metadata = None
        self.bm25_path = ROOT_DIR / CONFIG["storage"]["bm25_index_path"]

        logger.info(f"Initializing QueryEngine:")
        logger.info(f"  - Embedding Model: {self.embedding_model_name} ({'BGE-M3 Mode (Dense+Sparse)' if self.is_bge_m3 else 'E5/Dense Mode'})")
        logger.info(f"  - Reranking Model: {self.reranking_model_name}")
        logger.info(f"  - LLM (Chat) Model: {self.llm_model_name}")
        logger.info(f"  - Qdrant: {self.qdrant_host}:{self.qdrant_port}, Collection: {self.qdrant_collection}")
        logger.info(f"  - Cross-Encoder Reranking Enabled: {self.use_reranking}")
        logger.info(f"  - Fusion Weights (Vec/BM25_or_Sparse): {self.vector_weight}/{self.bm25_weight}")
        log_memory_usage(logger)

        # Validate library availability
        if self.is_bge_m3 and not FLAG_EMBEDDING_AVAILABLE:
             raise ImportError("BGE-M3 model specified, but 'FlagEmbedding' library is not installed.")
        if not self.is_bge_m3 and not BATCHED_INFERENCE_AVAILABLE:
             raise ImportError("E5-Instruct model specified, but 'embed' library (BatchedInference) is not installed.")
        if self.use_reranking and not BATCHED_INFERENCE_AVAILABLE: # Reranker uses BatchedInference
             raise ImportError("Reranking enabled, but 'embed' library (BatchedInference) is not installed.")

        self._load_chat_tokenizer()

    def _load_chat_tokenizer(self):
        """Loads the Hugging Face tokenizer for the chat LLM."""
        if self.chat_tokenizer:
            logger.info(f"Chat tokenizer for {self.llm_model_name} already loaded.")
            return True
        try:
            logger.info(f"Loading chat tokenizer for LLM: {self.llm_model_name}")
            self.chat_tokenizer = AutoTokenizer.from_pretrained(
                self.llm_model_name,
                trust_remote_code=True
            )
            if self.chat_tokenizer.chat_template is None:
                logger.warning(
                    f"Tokenizer for {self.llm_model_name} does not have a chat_template defined. "
                    "Prompt formatting might be incorrect or fail."
                )
            else:
                 logger.info(f"Chat tokenizer for {self.llm_model_name} loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to load chat tokenizer for {self.llm_model_name}: {e}", exc_info=True)
            self.chat_tokenizer = None
            return False

    def _format_query_for_e5(self, query: str) -> str:
        """Formats the raw user query with the instruction prefix for E5-instruct models."""
        logger.debug(f"Formatting query for E5 embedding: Task='{RETRIEVAL_TASK_INSTRUCTION}', Query='{query[:50]}...'")
        return f'Instruct: {RETRIEVAL_TASK_INSTRUCTION}\nQuery: {query}'

    def load_embedding_model(self):
        """Load the embedding and reranking models if not already loaded."""
        if self.embedding_model_instance is not None and (not self.use_reranking or self.reranker_model_instance is not None):
            logger.debug("Required embedding/reranking models already loaded.")
            return

        logger.info("===== LOADING EMBEDDING/RERANKING MODELS =====")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        start_time = time.time()

        try:
            # Load Embedding Model
            if self.embedding_model_instance is None:
                logger.info(f"Loading embedding model: {self.embedding_model_name} on {device}")
                if self.is_bge_m3:
                    if not FLAG_EMBEDDING_AVAILABLE: raise ImportError("FlagEmbedding not installed.")
                    self.embedding_model_instance = BGEM3FlagModel(
                        self.embedding_model_name,
                        use_fp16=True if device == "cuda" else False,
                        device=device
                    )
                    logger.info("Loaded BGE-M3 model using FlagEmbedding.")
                else: # E5 Instruct
                    if not BATCHED_INFERENCE_AVAILABLE: raise ImportError("BatchedInference not installed.")
                    self.embedding_model_instance = BatchedInference(
                        model_id=self.embedding_model_name,
                        engine="torch",
                        device=device
                    )
                    logger.info("Loaded E5-Instruct model using BatchedInference.")
            else:
                 logger.debug(f"Embedding model {self.embedding_model_name} already loaded.")

            # Load Reranker Model (always use BatchedInference if enabled)
            if self.use_reranking and self.reranking_model_name and self.reranker_model_instance is None:
                if not BATCHED_INFERENCE_AVAILABLE: raise ImportError("BatchedInference not installed for reranker.")
                logger.info(f"Loading reranking model: {self.reranking_model_name} on {device}")
                if not self.is_bge_m3 and self.embedding_model_instance is not None and \
                   hasattr(self.embedding_model_instance, 'model_id') and \
                   self.embedding_model_instance.model_id == self.reranking_model_name:
                     # Check if it's the same instance only for BatchedInference models
                    logger.warning("Reranker model is the same as E5 embedding model. Using the same instance.")
                    self.reranker_model_instance = self.embedding_model_instance
                else:
                    self.reranker_model_instance = BatchedInference(
                        model_id=self.reranking_model_name,
                        engine="torch",
                        device=device
                    )
                    logger.info("Loaded Reranker model using BatchedInference.")
            elif self.use_reranking and self.reranker_model_instance is not None:
                 logger.debug(f"Reranker model {self.reranking_model_name} already loaded.")
            elif not self.use_reranking:
                 logger.info("Cross-encoder reranking disabled.")

            elapsed_time = time.time() - start_time
            logger.info(f"Embedding/Reranking model loading completed in {elapsed_time:.2f} seconds")
            log_memory_usage(logger)

        except Exception as e:
            logger.error(f"Error loading embedding/reranking models: {e}", exc_info=True)
            self.embedding_model_instance = None
            self.reranker_model_instance = None
            raise

    def unload_models(self):
        """Unload the embedding, reranking, and chat tokenizer models."""
        logger.info("===== UNLOADING ALL QUERY ENGINE MODELS =====")
        unloaded_something = False

        # Unload Reranker Model First (especially if it might be shared with embedder)
        if self.reranker_model_instance is not None:
            logger.info(f"Unloading reranking model: {self.reranking_model_name}")
            try:
                # Check if the reranker instance is the same as the embedding instance (E5 case)
                is_shared_instance = (
                    not self.is_bge_m3 and
                    self.embedding_model_instance is not None and
                    self.reranker_model_instance is self.embedding_model_instance
                )

                if is_shared_instance:
                    logger.info("Reranker instance is shared with embedder. Will unload with embedder.")
                    # Don't delete/stop it here, let the embedding model unload handle it
                elif hasattr(self.reranker_model_instance, 'stop'):
                    self.reranker_model_instance.stop()
                    del self.reranker_model_instance
                    unloaded_something = True
                    logger.info(f"Reranker model {self.reranking_model_name} stopped and unloaded.")
                else:
                    # Fallback for objects without stop()
                    del self.reranker_model_instance
                    unloaded_something = True
                    logger.info(f"Reranker model {self.reranking_model_name} deleted (no stop method).")

            except Exception as e:
                logger.error(f"Error unloading reranker model: {e}", exc_info=True)
            finally:
                # Ensure the reference is cleared even if errors occurred, unless shared
                 if not locals().get('is_shared_instance', False):
                      self.reranker_model_instance = None


        # Unload Embedding Model
        if self.embedding_model_instance is not None:
            logger.info(f"Unloading embedding model: {self.embedding_model_name}")
            try:
                if self.is_bge_m3 and hasattr(self.embedding_model_instance, 'model'):
                    if hasattr(self.embedding_model_instance.model, 'cpu'): self.embedding_model_instance.model.cpu()
                    del self.embedding_model_instance.model; del self.embedding_model_instance.tokenizer
                elif not self.is_bge_m3 and hasattr(self.embedding_model_instance, 'stop'):
                    self.embedding_model_instance.stop()
                del self.embedding_model_instance
                self.embedding_model_instance = None
                unloaded_something = True
                logger.info(f"Embedding model {self.embedding_model_name} unloaded.")
            except Exception as e:
                 logger.error(f"Error unloading embedding model: {e}", exc_info=True)
                 self.embedding_model_instance = None # Ensure cleared on error

        # Unload Chat Tokenizer
        if self.chat_tokenizer is not None:
            logger.info(f"Unloading chat tokenizer: {self.llm_model_name}")
            try:
                del self.chat_tokenizer
                self.chat_tokenizer = None
                unloaded_something = True
                logger.info(f"Chat tokenizer {self.llm_model_name} unloaded.")
            except Exception as e:
                logger.error(f"Error unloading chat tokenizer: {e}", exc_info=True)


        if unloaded_something:
            if torch.cuda.is_available(): logger.info("Clearing CUDA cache..."); torch.cuda.empty_cache()
            logger.info("Running garbage collection..."); gc.collect()
            logger.info("===== MODEL UNLOAD COMPLETE =====")
            log_memory_usage(logger)
        else:
             logger.debug("No models were loaded, nothing to unload.")

    def ensure_llm_loaded(self):
        """Ensure the generative LLM model is loaded in the Aphrodite service."""
        if not self.aphrodite_service.is_running():
            logger.info("Aphrodite service not running, starting it")
            if not self.aphrodite_service.start():
                logger.error("Failed to start Aphrodite service")
                return False
        status = self.aphrodite_service.get_status()
        logger.info(f"Aphrodite service status: {status}")
        if not status.get("model_loaded", False) or status.get("current_model") != self.llm_model_name:
            logger.info(f"Loading LLM model for querying via Aphrodite: {self.llm_model_name}")
            if not self.aphrodite_service.load_model(self.llm_model_name):
                logger.error(f"Failed to load LLM model {self.llm_model_name} via Aphrodite")
                return False
            logger.info(f"LLM model {self.llm_model_name} loaded successfully via Aphrodite.")
        else:
             logger.info(f"LLM model {self.llm_model_name} already loaded via Aphrodite.")
        return True

    def _load_bm25_index(self) -> bool:
        """Load the BM25 index, tokenized texts, and metadata from file."""
        if self.bm25_index is not None:
            logger.debug("BM25 index already loaded.")
            return True
        try:
            if not self.bm25_path.exists():
                logger.warning(f"BM25 index file not found: {self.bm25_path}")
                self.bm25_index = None; self.tokenized_texts = None; self.bm25_metadata = None; return False

            logger.info(f"Loading BM25 index from {self.bm25_path}")
            with open(self.bm25_path, 'rb') as f:
                # Expecting tuple: (bm25_index, tokenized_texts, bm25_metadata)
                loaded_data = pickle.load(f)

            if not isinstance(loaded_data, tuple) or len(loaded_data) != 3:
                 logger.error(f"Invalid data format loaded from BM25 index file (expected tuple of 3): {self.bm25_path}")
                 self.bm25_index, self.tokenized_texts, self.bm25_metadata = None, None, None; return False

            self.bm25_index, self.tokenized_texts, self.bm25_metadata = loaded_data

            # Validate loaded components
            if not hasattr(self.bm25_index, 'get_scores'):
                 logger.error("Loaded BM25 index object lacks 'get_scores' method.")
                 self.bm25_index = None; self.tokenized_texts = None; self.bm25_metadata = None; return False
            if not isinstance(self.tokenized_texts, list):
                 logger.error("Loaded tokenized texts is not a list.")
                 self.bm25_index = None; self.tokenized_texts = None; self.bm25_metadata = None; return False
            if not isinstance(self.bm25_metadata, list):
                 logger.error("Loaded BM25 metadata is not a list.")
                 self.bm25_index = None; self.tokenized_texts = None; self.bm25_metadata = None; return False
            if len(self.tokenized_texts) != len(self.bm25_metadata):
                 logger.warning(f"Mismatch between tokenized texts ({len(self.tokenized_texts)}) and metadata ({len(self.bm25_metadata)}) counts.")
                 # Allow proceeding but log warning

            logger.info(f"BM25 index, texts, and metadata loaded successfully ({len(self.tokenized_texts)} documents).")
            return True
        except Exception as e:
            logger.error(f"Error loading BM25 index: {e}", exc_info=True)
            self.bm25_index, self.tokenized_texts, self.bm25_metadata = None, None, None; return False

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the Qdrant collection."""
        try:
            client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port, timeout=10)
            try: collections = client.get_collections().collections; collection_names = [c.name for c in collections]
            except Exception as conn_err: return {'exists': False, 'error': f"Connection error: {conn_err}"}

            if self.qdrant_collection in collection_names:
                info = client.get_collection(collection_name=self.qdrant_collection)
                # Determine vector size and distance based on actual config
                vector_size = 0
                distance_metric = "Unknown"
                has_dense = False
                has_sparse = False

                if isinstance(info.config.params.vectors, dict): # Named vectors (like BGE-M3 dense)
                    if 'dense' in info.config.params.vectors:
                        vector_size = info.config.params.vectors['dense'].size
                        distance_metric = str(info.config.params.vectors['dense'].distance)
                        has_dense = True
                elif info.config.params.vectors: # Default unnamed vector (like E5)
                    vector_size = info.config.params.vectors.size
                    distance_metric = str(info.config.params.vectors.distance)
                    has_dense = True

                if info.config.params.sparse_vectors and 'sparse' in info.config.params.sparse_vectors:
                    has_sparse = True

                schema_type = "Unknown"
                if has_dense and has_sparse: schema_type = "BGE-M3 (Dense+Sparse)"
                elif has_dense: schema_type = "E5/Dense Only"

                result = {
                    'exists': True,
                    'name': self.qdrant_collection,
                    'points_count': info.points_count,
                    'vector_size': vector_size,
                    'distance': distance_metric,
                    'schema_type': schema_type,
                    'has_sparse': has_sparse
                }
                logger.debug(f"Collection info retrieved: {result}")
                return result
            else:
                return {'exists': False}
        except Exception as e:
            logger.error(f"Unexpected error getting collection info: {e}", exc_info=True)
            return {'exists': False, 'error': str(e)}

    def clear_collection(self) -> bool:
        """Clear Qdrant collection, BM25 index, and extracted data."""
        try:
            try:
                client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
                collections = client.get_collections().collections; collection_names = [c.name for c in collections]
                if self.qdrant_collection in collection_names: client.delete_collection(collection_name=self.qdrant_collection); logger.info(f"Qdrant collection '{self.qdrant_collection}' deleted"); time.sleep(1)
                else: logger.warning(f"Qdrant collection '{self.qdrant_collection}' does not exist, skipping deletion.")
            except Exception as e: logger.error(f"Error clearing Qdrant collection: {e}", exc_info=True)

            # Clear BM25 files
            try:
                if self.bm25_path.exists(): self.bm25_path.unlink(); logger.info(f"BM25 index file deleted: {self.bm25_path}")
                # Also delete related stopwords file if it exists
                stopwords_path = self.bm25_path.parent / "stopwords.pkl"
                if stopwords_path.exists(): stopwords_path.unlink(); logger.info(f"Stopwords file deleted: {stopwords_path}")
            except Exception as e: logger.error(f"Error deleting BM25 files: {e}")

            # Clear extracted entities/relationships
            try:
                extracted_data_path = ROOT_DIR / CONFIG["storage"]["extracted_data_path"]
                entities_path = extracted_data_path / "entities.json"; relationships_path = extracted_data_path / "relationships.json"
                if entities_path.exists(): entities_path.unlink(); logger.info(f"Entities file deleted: {entities_path}")
                if relationships_path.exists(): relationships_path.unlink(); logger.info(f"Relationships file deleted: {relationships_path}")
            except Exception as e: logger.error(f"Error deleting extracted data files: {e}")

            # Reset internal state
            self.bm25_index = None; self.tokenized_texts = None; self.bm25_metadata = None
            logger.info("Internal BM25 state cleared.")
            return True
        except Exception as e:
            logger.error(f"Error clearing all data: {e}", exc_info=True)
            return False

    def get_chunks(self, limit: int = 20, search_text: str = None, document_filter: str = None) -> List[Dict[str, Any]]:
        """Retrieve chunks based on optional filters."""
        try:
            client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
            filter_conditions = []
            if search_text: filter_conditions.append(rest.FieldCondition(key="text", match=rest.MatchText(text=search_text)))
            if document_filter: filter_conditions.append(rest.FieldCondition(key="file_name", match=rest.MatchValue(value=document_filter)))
            filter_obj = rest.Filter(must=filter_conditions) if filter_conditions else None

            # Check collection existence before scrolling
            try:
                collections = client.get_collections().collections
                if self.qdrant_collection not in [c.name for c in collections]:
                    logger.warning(f"Collection {self.qdrant_collection} does not exist"); return []
            except Exception as conn_err:
                logger.error(f"Failed Qdrant connection checking collection for get_chunks: {conn_err}"); return []

            scroll_result, _ = scroll_with_filter_compatible(client=client, collection_name=self.qdrant_collection, limit=limit, with_payload=True, with_vectors=False, scroll_filter=filter_obj)

            points = scroll_result
            if not points: logger.info(f"No points found matching filter: text='{search_text}', doc='{document_filter}'"); return []

            chunks = []
            for point in points:
                payload = point.payload or {} # Ensure payload is a dict
                text = payload.get('text', '')
                original_text = payload.get('original_text', text)
                # Gather all metadata, prioritizing top-level keys over 'metadata' sub-dict
                metadata_from_payload = {k: v for k, v in payload.items() if k not in ['text', 'original_text', 'metadata']}
                if 'metadata' in payload and isinstance(payload['metadata'], dict):
                    # Merge, letting top-level keys overwrite sub-dict keys if conflicts exist
                    metadata_from_payload = {**payload['metadata'], **metadata_from_payload}

                # Ensure essential keys are present
                if 'chunk_id' not in metadata_from_payload: metadata_from_payload['chunk_id'] = point.id
                if 'file_name' not in metadata_from_payload: metadata_from_payload['file_name'] = 'Unknown'

                chunk = {'id': point.id, 'text': text, 'original_text': original_text, 'metadata': metadata_from_payload}
                chunks.append(chunk)
            logger.info(f"Retrieved {len(chunks)} chunks for exploration.")
            return chunks
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}", exc_info=True)
            return []

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks using appropriate strategy for the configured model:
        - BGE-M3: Dense Search + Sparse Search -> RRF Fusion -> Rerank
        - E5: Dense Search + BM25 Search -> RRF Fusion -> Rerank
        Applies cross-encoder reranking at the end for both.
        """
        try:
            start_time = time.time()
            # Ensure models are loaded
            if self.embedding_model_instance is None or (self.use_reranking and self.reranker_model_instance is None):
                logger.info("Embedding/reranking models not loaded, loading now...")
                self.load_embedding_model()

            # Load BM25 index if using E5 path and not already loaded
            if not self.is_bge_m3 and self.bm25_index is None:
                if not self._load_bm25_index():
                    logger.warning("BM25 index not available for E5 retrieval.")
                    # Proceed with vector search only for E5 in this case
                else:
                     logger.info("BM25 index loaded for E5 retrieval.")

            # Perform Initial Retrieval based on Model Type
            if self.is_bge_m3:
                logger.info("Performing BGE-M3 retrieval (Dense + Sparse)...")
                # _vector_search for BGE-M3 now handles dense+sparse search and fusion
                fused_results = self._vector_search(query)
                logger.info(f"BGE-M3 Dense+Sparse RRF fusion returned {len(fused_results)} candidates.")
            else: # E5 Instruct
                logger.info("Performing E5-Instruct + BM25 retrieval with RRF fusion...")
                bm25_results = self._bm25_search(query) if self.bm25_index is not None else []
                vector_results = self._vector_search(query) # _vector_search for E5 only does dense
                fused_results = self._fuse_results(bm25_results, vector_results)
                logger.info(f"E5+BM25 RRF fusion resulted in {len(fused_results)} candidates.")

            # Apply Cross-Encoder Reranking (if enabled)
            if self.use_reranking and fused_results and self.reranker_model_instance is not None:
                logger.info("Applying cross-encoder reranking...")
                final_results = self._rerank(query, fused_results)
            elif self.use_reranking:
                 logger.warning("Reranking enabled but model not loaded or no initial results. Skipping.")
                 # Limit results based on rerank k even if reranker didn't run
                 final_results = sorted(fused_results, key=lambda x: x['score'], reverse=True)[:self.top_k_rerank]
            else:
                logger.info("Cross-encoder reranking disabled.")
                # Limit results based on rerank k
                final_results = sorted(fused_results, key=lambda x: x['score'], reverse=True)[:self.top_k_rerank]

            elapsed_time = time.time() - start_time
            logger.info(f"Retrieval completed in {elapsed_time:.2f}s. Returning {len(final_results)} final results.")
            return final_results

        except Exception as e:
            logger.error(f"Error in retrieval orchestration: {e}", exc_info=True)
            return []

    def _bm25_search(self, query: str) -> List[Dict[str, Any]]:
        """Performs BM25 search using the loaded index."""
        if not self.bm25_index or not self.tokenized_texts or not self.bm25_metadata:
            logger.warning("BM25 components not loaded for search.")
            return []
        try:
            import nltk
            from nltk.tokenize import word_tokenize
            try: nltk.data.find('tokenizers/punkt')
            except LookupError: nltk.download('punkt', quiet=True)

            # Load stopwords (assuming they are saved alongside the index)
            stop_words = set()
            try:
                stopwords_path = self.bm25_path.parent / "stopwords.pkl"
                if stopwords_path.exists():
                    with open(stopwords_path, 'rb') as f: stop_words = pickle.load(f)
                else: # Fallback to NLTK defaults if file missing
                     nltk.download('stopwords', quiet=True)
                     stop_words = set(nltk.corpus.stopwords.words('english'))
            except Exception as e:
                 logger.warning(f"Error loading stopwords: {e}. Using NLTK defaults.")
                 try:
                      nltk.download('stopwords', quiet=True)
                      stop_words = set(nltk.corpus.stopwords.words('english'))
                 except: logger.error("Could not load NLTK stopwords either.")

            tokenized_query = word_tokenize(query.lower())
            filtered_query = [token for token in tokenized_query if token.isalnum() and token not in stop_words]
            if not filtered_query and tokenized_query: filtered_query = [t for t in tokenized_query if t.isalnum()]
            if not filtered_query: logger.warning("BM25 query empty after tokenization/filtering."); return []

            bm25_scores = self.bm25_index.get_scores(filtered_query)

            results = []
            doc_count = len(self.bm25_metadata)
            for i, score in enumerate(bm25_scores):
                # Ensure index is within bounds of metadata
                if score > 0 and i < doc_count:
                    metadata = self.bm25_metadata[i]
                    # Ensure metadata is a dictionary
                    if not isinstance(metadata, dict):
                        logger.warning(f"Skipping BM25 result at index {i} due to invalid metadata format: {type(metadata)}")
                        continue

                    text = metadata.get('text', '') # Get text used for indexing from metadata
                    original_text = metadata.get('original_text', text) # Fallback if original missing
                    chunk_id = metadata.get('chunk_id', f"bm25_doc_{i}") # Ensure chunk_id exists
                    # Reconstruct metadata dict for result, ensuring chunk_id is present
                    result_metadata = metadata.copy()
                    result_metadata['chunk_id'] = chunk_id

                    results.append({
                        'id': chunk_id,
                        'score': float(score),
                        'text': text,
                        'original_text': original_text,
                        'metadata': result_metadata
                    })

            results.sort(key=lambda x: x['score'], reverse=True)
            results = results[:self.top_k_bm25]
            logger.info(f"BM25 search returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Error in BM25 search: {e}", exc_info=True)
            return []

    def _vector_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Performs vector search based on the loaded model.
        - BGE-M3: Generates dense+sparse, searches both via Qdrant, fuses with RRF.
        - E5: Generates dense, searches via Qdrant.
        Returns results in a common dictionary format.
        """
        if self.embedding_model_instance is None:
            logger.error("Embedding model not loaded for vector search.")
            return []

        try:
            client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port, timeout=10)
            try: collections = client.get_collections().collections
            except Exception as conn_err: logger.error(f"Failed Qdrant connection: {conn_err}"); return []
            if self.qdrant_collection not in [c.name for c in collections]: logger.warning(f"Qdrant collection '{self.qdrant_collection}' does not exist."); return []

            # Helper function to format Qdrant results
            def format_qdrant_results(qdrant_list):
                formatted = []
                for point in qdrant_list:
                    payload = point.payload or {}
                    text = payload.get('text', '')
                    original_text = payload.get('original_text', text)
                    metadata_from_payload = {k: v for k, v in payload.items() if k not in ['text', 'original_text']}
                    chunk_id = metadata_from_payload.get('chunk_id', point.id) # Use point.id as fallback ID
                    metadata_from_payload['chunk_id'] = chunk_id
                    formatted.append({'id': point.id, 'score': float(point.score), 'text': text, 'original_text': original_text, 'metadata': metadata_from_payload})
                return formatted

            if self.is_bge_m3:
                logger.info("Generating BGE-M3 query embeddings (dense, sparse)...")
                # *** MODIFIED: Disable ColBERT ***
                query_outputs = self.embedding_model_instance.encode(
                    [query], return_dense=True, return_sparse=True, return_colbert_vecs=False # <-- Changed
                )
                dense_vec = query_outputs["dense_vecs"][0]
                sparse_weights = query_outputs["lexical_weights"][0]
                qdrant_sparse = _create_sparse_vector_qdrant_format(sparse_weights)

                dense_results_list = []
                sparse_results_list = []

                # Dense Search using NamedVector
                logger.info(f"Performing Qdrant BGE-M3 dense vector search (limit: {self.top_k_vector})...")
                try:
                    qdrant_dense_results = client.search(
                        collection_name=self.qdrant_collection,
                        query_vector=rest.NamedVector(name="dense", vector=dense_vec.tolist()),
                        limit=self.top_k_vector,
                        with_payload=True
                    )
                    if qdrant_dense_results: dense_results_list = qdrant_dense_results
                    logger.info(f"Qdrant BGE-M3 dense search returned {len(dense_results_list)} results.")
                except Exception as dense_search_err:
                     logger.error(f"Error during BGE-M3 dense search: {dense_search_err}", exc_info=True)

                # Sparse Search using NamedSparseVector (if sparse vector exists)
                if qdrant_sparse:
                    logger.info(f"Performing Qdrant BGE-M3 sparse vector search (limit: {self.top_k_vector})...")
                    try:
                        qdrant_sparse_results = client.search(
                            collection_name=self.qdrant_collection,
                            query_vector=rest.NamedSparseVector(name="sparse", vector=qdrant_sparse),
                            limit=self.top_k_vector,
                            with_payload=True
                        )
                        if qdrant_sparse_results: sparse_results_list = qdrant_sparse_results
                        logger.info(f"Qdrant BGE-M3 sparse search returned {len(sparse_results_list)} results.")
                    except Exception as sparse_search_err:
                        logger.error(f"Error during BGE-M3 sparse search: {sparse_search_err}", exc_info=True)
                else:
                    logger.warning("Skipping BGE-M3 sparse search as query sparse vector is empty.")

                # Format results
                formatted_dense_results = format_qdrant_results(dense_results_list)
                formatted_sparse_results = format_qdrant_results(sparse_results_list)

                # Fuse Dense and Sparse results using RRF
                logger.info("Fusing BGE-M3 dense and sparse results using RRF...")
                # Reuse _fuse_results: Treat sparse as "bm25" and dense as "vector" for its logic
                fused_vector_results = self._fuse_results(formatted_sparse_results, formatted_dense_results)
                logger.info(f"BGE-M3 RRF Fusion returned {len(fused_vector_results)} results")
                return fused_vector_results

            else: # E5 Instruct
                logger.info("Generating E5-Instruct query embedding (dense only)...")
                formatted_query = self._format_query_for_e5(query)
                future = self.embedding_model_instance.embed(sentences=[formatted_query], model_id=self.embedding_model_name)
                result = future.result()
                if isinstance(result, tuple) and len(result) >= 1: embeddings = result[0]
                else: embeddings = result
                if not embeddings: logger.error("Failed to generate E5 query embedding."); return []
                query_embedding = embeddings[0]
                if hasattr(query_embedding, 'tolist'): vector = query_embedding.tolist()
                elif isinstance(query_embedding, (list, np.ndarray)): vector = list(query_embedding)
                else: raise TypeError(f"Invalid E5 embedding type: {type(query_embedding)}")
                if not vector: logger.error("E5 query vector is empty."); return []

                logger.info(f"Performing Qdrant E5 dense vector search (limit: {self.top_k_vector})...")
                # E5 uses the default, unnamed vector
                qdrant_results = client.search(
                    collection_name=self.qdrant_collection,
                    query_vector=vector,
                    limit=self.top_k_vector,
                    with_payload=True
                )
                search_results = qdrant_results if qdrant_results else []
                logger.info(f"Qdrant E5 search returned {len(search_results)} results.")
                return format_qdrant_results(search_results) # Format and return directly

        except Exception as e:
            logger.error(f"Error in vector search (_vector_search): {e}", exc_info=True)
            return []

    def _fuse_results(self, list1: List[Dict[str, Any]], list2: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fuse results from two lists (e.g., BM25 & Vector, or Sparse & Dense) using RRF.
        Uses self.bm25_weight for list1 and self.vector_weight for list2.
        """
        if not list1 and not list2: return []
        # If one list is empty, return the other sorted and limited
        if not list1: return sorted(list2, key=lambda x: x['score'], reverse=True)[:self.top_k_hybrid]
        if not list2: return sorted(list1, key=lambda x: x['score'], reverse=True)[:self.top_k_hybrid]

        # Use rank-based RRF
        k = 60 # Standard RRF parameter
        dict1 = {} # Stores {id: {'rank': rank, ...}}
        dict2 = {} # Stores {id: {'rank': rank, ...}}

        # Assign ranks based on original scores (higher score = better rank)
        list1_sorted = sorted(list1, key=lambda x: x['score'], reverse=True)
        for rank, result in enumerate(list1_sorted):
            # Use 'id' which should be the consistent chunk_id
            doc_id = result.get('id')
            if doc_id and doc_id not in dict1:
                dict1[doc_id] = {'rank': rank + 1, **result}

        list2_sorted = sorted(list2, key=lambda x: x['score'], reverse=True)
        for rank, result in enumerate(list2_sorted):
            doc_id = result.get('id')
            if doc_id and doc_id not in dict2:
                 dict2[doc_id] = {'rank': rank + 1, **result}

        # Calculate RRF scores
        all_doc_ids = set(dict1.keys()) | set(dict2.keys())
        rrf_results = []

        for doc_id in all_doc_ids:
            rank1 = dict1.get(doc_id, {'rank': len(list1) * 2})['rank'] # Default rank if not found
            rank2 = dict2.get(doc_id, {'rank': len(list2) * 2})['rank'] # Default rank if not found

            # RRF formula using configured weights
            rrf_score = (self.bm25_weight / (k + rank1)) + (self.vector_weight / (k + rank2))

            # Get the full document info (prefer dict2=vector/dense if available)
            doc_info = dict2.get(doc_id) or dict1.get(doc_id)
            if not doc_info:
                logger.warning(f"Could not find document info for fused ID {doc_id}")
                continue

            # Ensure metadata exists and contains chunk_id
            metadata = doc_info.get('metadata', {})
            if not isinstance(metadata, dict): metadata = {} # Ensure it's a dict
            metadata['chunk_id'] = doc_id # Ensure ID is present

            rrf_results.append({
                 'id': doc_id,
                 'score': rrf_score, # Use RRF score for final ranking
                 'bm25_rank': rank1 if doc_id in dict1 else None, # Store original ranks
                 'vector_rank': rank2 if doc_id in dict2 else None,
                 'text': doc_info['text'],
                 'original_text': doc_info.get('original_text', doc_info['text']),
                 'metadata': metadata
            })

        # Sort by RRF score and limit
        rrf_results.sort(key=lambda x: x['score'], reverse=True)
        results = rrf_results[:self.top_k_hybrid]
        logger.info(f"RRF Fusion returned {len(results)} results")
        return results

    def _rerank(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank results using the cross-encoder model."""
        if not self.use_reranking: return results
        if self.reranker_model_instance is None: logger.error("Reranker model not loaded."); return results
        if not self.reranking_model_name: logger.warning("No reranking model name specified. Skipping."); return results
        if not results: return []

        try:
            # Use original_text if available, otherwise fallback to text
            texts_to_rerank = [result.get('original_text', result.get('text', '')) for result in results]
            if not any(texts_to_rerank): # Check if all texts are empty
                logger.warning("No text found in results to rerank.")
                return results

            logger.info(f"Reranking {len(texts_to_rerank)} results using cross-encoder: {self.reranking_model_name}")
            logger.debug(f"Reranking with original query: '{query[:100]}...'")

            # Call BatchedInference rerank
            future = self.reranker_model_instance.rerank(query=query, docs=texts_to_rerank, model_id=self.reranking_model_name)
            rerank_result = future.result()

            # Process results (handle potential tuple format)
            if isinstance(rerank_result, tuple): scores_data = rerank_result[0]
            else: scores_data = rerank_result

            # Ensure scores_data is a list of scores (handle index-score tuples if present)
            if isinstance(scores_data, list) and scores_data and isinstance(scores_data[0], tuple) and len(scores_data[0])==2:
                 # Sort by original index (first element of tuple) to align scores
                 scores_data.sort(key=lambda x: x[0])
                 scores_only = [score for _, score in scores_data]
            elif isinstance(scores_data, list): # Assume it's already a list of scores
                scores_only = scores_data
            else:
                logger.error(f"Unexpected reranker scores format: {type(scores_data)}. Skipping."); return results

            if len(scores_only) != len(results):
                logger.error(f"Reranker score count mismatch: got {len(scores_only)}, expected {len(results)}. Skipping."); return results

            # Update results with reranked scores
            reranked_results = []
            for i, score in enumerate(scores_only):
                result = results[i].copy() # Make a copy to avoid modifying original list
                result['original_score'] = result['score'] # Store pre-rerank score
                result['score'] = float(score) # Update with rerank score
                reranked_results.append(result)

            # Sort by new reranked score
            reranked_results.sort(key=lambda x: x['score'], reverse=True)

            # Filter by minimum score threshold
            filtered_results = [result for result in reranked_results if result['score'] >= self.min_score_threshold]

            # Limit to final rerank k
            final_reranked = filtered_results[:self.top_k_rerank]
            logger.info(f"Cross-encoder reranking completed. Returning {len(final_reranked)} results (after thresholding/limiting).")
            return final_reranked
        except Exception as e:
            logger.error(f"Error during cross-encoder reranking: {e}", exc_info=True)
            # Return original results sorted by initial score as fallback
            return sorted(results, key=lambda x: x['score'], reverse=True)[:self.top_k_rerank]

    def _create_chat_prompt(self, query: str, context: str) -> List[Dict[str, str]]:
            """
            Creates the structured message list for the chat model.
            """
            system_content = """You are an expert assistant specializing in anti-corruption investigations and analysis. Your responses should be detailed, factual, and based ONLY on the provided context. If the answer is not found in the context, state that clearly. Do not make assumptions or use external knowledge.

    When answering, cite your sources by referring to the numbers in square brackets corresponding to the context snippets provided below. For example, mention "[1]" if the information comes from the first context snippet. You can cite multiple sources like "[1], [3]"."""

            user_content = f"""Context:
    {context}

    Based ONLY on the context provided above, answer the following question:
    Question: {query}"""

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
            return messages

    def load_llm_model(self):
        """Ensure the designated LLM model (for chat) is loaded in the Aphrodite service."""
        logger.info(f"Requesting check/load of LLM model {self.llm_model_name} in Aphrodite service...")
        return self.ensure_llm_loaded()

    def query(self, query: str) -> Dict[str, Any]:
        """
        Query the system (non-streaming). Retrieves context, formats the prompt
        using chat templates, and generates a response.
        """
        try:
            # 1. Ensure LLM service and model are ready
            if not self.ensure_llm_loaded():
                 return {'answer': "Error: The Language Model service is not ready.", 'sources': []}

            # 1.5 Ensure Chat Tokenizer is Ready
            if self.chat_tokenizer is None:
                 logger.error(f"Chat tokenizer for {self.llm_model_name} is not loaded. Attempting to load again.")
                 if not self._load_chat_tokenizer():
                     return {"answer": f"Error: Could not load the required tokenizer for {self.llm_model_name}. Cannot process query.", "sources": []}
                 if self.chat_tokenizer.chat_template is None: # Re-check after load attempt
                     logger.error(f"Chat tokenizer for {self.llm_model_name} lacks a chat template.")
                     return {"answer": f"Sorry, the configured model ({self.llm_model_name}) doesn't support standard chat formatting. Cannot process query.", "sources": []}
            elif self.chat_tokenizer.chat_template is None:
                 logger.error(f"Chat tokenizer for {self.llm_model_name} lacks a chat template.")
                 return {"answer": f"Sorry, the configured model ({self.llm_model_name}) doesn't support standard chat formatting. Cannot process query.", "sources": []}


            # 2. Retrieval
            logger.info(f"Retrieving context for query: '{query[:50]}...'")
            start_retrieve = time.time()
            retrieval_results = self.retrieve(query) # Handles E5/BGE-M3 internally
            logger.info(f"Retrieval took {time.time() - start_retrieve:.2f}s, found {len(retrieval_results)} results for generation.")

            # 3. Format Context and Prepare Messages
            sources_for_llm = []
            context_texts = []
            for result in retrieval_results:
                # Prefer original_text for LLM context to maximize fidelity
                text_for_context = result.get('original_text', result.get('text', ''))
                context_texts.append(text_for_context)
                # Store both text and original_text in sources for potential display/debugging
                sources_for_llm.append({
                    'text': result.get('text', ''),
                    'original_text': result.get('original_text', ''),
                    'metadata': result.get('metadata', {}),
                    'score': result.get('score', 0.0)
                })

            context_for_prompt = "\n\n".join([f"[{i + 1}] {text}" for i, text in enumerate(context_texts)])
            if not context_for_prompt:
                 logger.warning("No context retrieved or context texts were empty.")
                 context_for_prompt = "No relevant context was found."
            elif context_texts:
                 logger.info(f"Using original_text for LLM context (first snippet preview: '{context_texts[0][:100]}...')")

            # Get base structured messages (system + user)
            base_messages = self._create_chat_prompt(query, context_for_prompt)
            # For non-streaming query, we typically don't include history unless explicitly passed
            final_messages_for_template = base_messages

            # 4. Apply Chat Template
            try:
                formatted_prompt = self.chat_tokenizer.apply_chat_template(
                    final_messages_for_template,
                    tokenize=False,
                    add_generation_prompt=True # Add prompt for assistant response
                )
                logger.debug(f"Formatted query prompt length: {len(formatted_prompt)}")
            except Exception as e:
                logger.error(f"Error applying chat template for query: {e}", exc_info=True)
                logger.error(traceback.format_exc())
                return {"answer": "Sorry, I encountered an error formatting the request for the AI model.", "sources": sources_for_llm}

            # 5. Call LLM Service
            logger.info("Sending chat generation request to Aphrodite service...")
            start_generate = time.time()
            response = self.aphrodite_service.generate_chat(prompt=formatted_prompt)
            logger.info(f"Chat generation via Aphrodite took {time.time() - start_generate:.2f}s")

            # 6. Process Response
            if response and response.get("status") == "success":
                answer = response.get("result", "Error: No answer content received.")
                logger.info(f"Received successful chat response (length: {len(answer)})")
                return {'answer': answer, 'sources': sources_for_llm}
            else:
                error_msg = response.get("error", "Unknown error") if response else "No response from service"
                logger.error(f"Error from Aphrodite service during chat generation: {error_msg}")
                return {'answer': f"Error generating response: {error_msg}", 'sources': sources_for_llm}

        except Exception as e:
            logger.error(f"Unexpected error in query method: {e}", exc_info=True)
            logger.error(traceback.format_exc())
            return {'answer': f"Error: An unexpected error occurred ({str(e)})", 'sources': []}