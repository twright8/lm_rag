# --- START OF REWRITTEN FILE: src/core/query_system/query_engine.py ---
"""
Query engine for Anti-Corruption RAG System.
Handles retrieval using appropriate strategy (Dense+Sparse or Dense+BM25).
Applies cross-encoder reranking.
Does NOT handle LLM generation directly.
Receives configuration via constructor.
"""
import sys
import os
from pathlib import Path
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

# Add project root to path *independently* if needed for sibling imports,
# but avoid importing config/setup modules directly.
# Assuming this file is in src/core/query_system/
try:
    _CURRENT_DIR = Path(__file__).resolve().parent
    _ROOT_DIR_QE = _CURRENT_DIR.parent.parent.parent.parent # Adjust based on actual structure if needed
    if str(_ROOT_DIR_QE) not in sys.path:
        sys.path.append(str(_ROOT_DIR_QE))
    # Import utils *after* potentially adding root to path
    from src.utils.logger import setup_logger
    from src.utils.resource_monitor import log_memory_usage
    from src.utils.qdrant_utils import scroll_with_filter_compatible
except ImportError as e:
    # Fallback logging if utils fail
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to import utils in QueryEngine: {e}. Check paths and dependencies.")
    # Define dummy functions if imports failed
    def log_memory_usage(logger, msg=None): pass
    def scroll_with_filter_compatible(*args, **kwargs): return ([], None)
else:
    # Initialize logger using the imported setup function
    logger = setup_logger(__name__)


# Conditional imports for embedding models
try:
    from embed import BatchedInference
    BATCHED_INFERENCE_AVAILABLE = True
except ImportError:
    BATCHED_INFERENCE_AVAILABLE = False
    logger.warning("BatchedInference (embed library) not found. E5/Reranker models unavailable.")
try:
    from FlagEmbedding import BGEM3FlagModel
    FLAG_EMBEDDING_AVAILABLE = True
except ImportError:
    FLAG_EMBEDDING_AVAILABLE = False
    logger.warning("FlagEmbedding library not found. BGE-M3 models unavailable.")


# --- E5-INSTRUCT TASK DEFINITION (Only used if E5 model is loaded) ---
RETRIEVAL_TASK_INSTRUCTION = 'Given a web search query, retrieve relevant passages that answer the query'

# Helper function for BGE-M3 sparse vectors (Unchanged)
def _create_sparse_vector_qdrant_format(sparse_data: Dict[Union[str, int], float]) -> Optional[rest.SparseVector]:
    if not sparse_data: return None
    sparse_indices = []; sparse_values = []; skipped_non_digit = 0
    for key, value in sparse_data.items():
        try:
            float_value = float(value)
            if float_value > 0:
                if isinstance(key, str) and key.isdigit(): key_int = int(key)
                elif isinstance(key, int): key_int = key
                else: skipped_non_digit += 1; continue
                sparse_indices.append(key_int); sparse_values.append(float_value)
        except (ValueError, TypeError): continue
    if skipped_non_digit > 0: logger.warning(f"Skipped {skipped_non_digit} non-integer sparse keys.")
    if not sparse_indices: return None
    return rest.SparseVector(indices=sparse_indices, values=sparse_values)


class QueryEngine:
    """
    Query engine focused on retrieval (Dense+Sparse or Dense+BM25) and reranking.
    Receives configuration via constructor.
    """

    def __init__(self, config: Dict[str, Any], root_dir: Path):
        """
        Initialize query engine with configuration.

        Args:
            config: The main application configuration dictionary.
            root_dir: The project's root directory path.
        """
        logger.info("Initializing QueryEngine...")
        self.config = config
        self.root_dir = root_dir

        # Qdrant Config
        qdrant_config = config.get("qdrant", {})
        self.qdrant_host = qdrant_config.get("host", "localhost")
        self.qdrant_port = qdrant_config.get("port", 6333)
        self.qdrant_collection = qdrant_config.get("collection_name", "default_collection")

        # Retrieval Config
        retrieval_config = config.get("retrieval", {})
        self.top_k_vector = retrieval_config.get("top_k_vector", 100)
        self.top_k_bm25 = retrieval_config.get("top_k_bm25", 100)
        self.top_k_hybrid = retrieval_config.get("top_k_hybrid", 50)
        self.top_k_rerank = retrieval_config.get("top_k_rerank", 10)
        self.vector_weight = float(retrieval_config.get("vector_weight", 0.7))
        self.bm25_weight = float(retrieval_config.get("bm25_weight", 0.3)) # Also sparse weight
        self.use_reranking = retrieval_config.get("use_reranking", True)
        self.min_score_threshold = float(retrieval_config.get("minimum_score_threshold", 0.01))

        # Model Config (Embedding/Reranking only)
        models_config = config.get("models", {})
        self.embedding_model_name = models_config.get("embedding_model", "BAAI/bge-m3") # Default if missing
        self.reranking_model_name = models_config.get("reranking_model", "BAAI/bge-reranker-v2-m3")
        self.llm_model_name = "N/A" # Set externally by app_setup after init

        # State
        self.is_bge_m3 = "bge-m3" in self.embedding_model_name.lower()
        self.embedding_model_instance = None
        self.reranker_model_instance = None
        self.bm25_index = None
        self.tokenized_texts = None
        self.bm25_metadata = None
        storage_config = config.get("storage", {})
        self.bm25_path = self.root_dir / storage_config.get("bm25_index_path", "data/bm25/index.pkl")

        logger.info(f"  - Embedding Model: {self.embedding_model_name} ({'BGE-M3 Mode' if self.is_bge_m3 else 'E5/Dense Mode'})")
        logger.info(f"  - Reranking Model: {self.reranking_model_name} (Enabled: {self.use_reranking})")
        logger.info(f"  - Qdrant: {self.qdrant_host}:{self.qdrant_port}, Collection: {self.qdrant_collection}")
        logger.info(f"  - Fusion Weights (Vec/BM25_or_Sparse): {self.vector_weight}/{self.bm25_weight}")
        log_memory_usage(logger)

        # Validate library availability
        if self.is_bge_m3 and not FLAG_EMBEDDING_AVAILABLE:
             raise ImportError("BGE-M3 model specified, but 'FlagEmbedding' library is not installed.")
        if not self.is_bge_m3 and not BATCHED_INFERENCE_AVAILABLE:
             raise ImportError("E5-Instruct model specified, but 'embed' library (BatchedInference) is not installed.")
        if self.use_reranking and not BATCHED_INFERENCE_AVAILABLE:
             raise ImportError("Reranking enabled, but 'embed' library (BatchedInference) is not installed.")

    # --- Methods below remain largely the same, using self.config where needed ---

    def _format_query_for_e5(self, query: str) -> str:
        """Formats the raw user query with the instruction prefix for E5-instruct models."""
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
            if self.embedding_model_instance is None:
                logger.info(f"Loading embedding model: {self.embedding_model_name} on {device}")
                if self.is_bge_m3:
                    if not FLAG_EMBEDDING_AVAILABLE: raise ImportError("FlagEmbedding not installed.")
                    self.embedding_model_instance = BGEM3FlagModel(self.embedding_model_name, use_fp16=(device == "cuda"), device=device)
                    logger.info("Loaded BGE-M3 model using FlagEmbedding.")
                else:
                    if not BATCHED_INFERENCE_AVAILABLE: raise ImportError("BatchedInference not installed.")
                    self.embedding_model_instance = BatchedInference(model_id=self.embedding_model_name, engine="torch", device=device)
                    logger.info("Loaded E5-Instruct model using BatchedInference.")
            if self.use_reranking and self.reranking_model_name and self.reranker_model_instance is None:
                if not BATCHED_INFERENCE_AVAILABLE: raise ImportError("BatchedInference not installed for reranker.")
                logger.info(f"Loading reranking model: {self.reranking_model_name} on {device}")
                is_shared_instance = (not self.is_bge_m3 and self.embedding_model_instance is not None and
                                      hasattr(self.embedding_model_instance, 'model_id') and
                                      self.embedding_model_instance.model_id == self.reranking_model_name)
                if is_shared_instance:
                    logger.warning("Reranker model is the same as E5 embedding model. Using the same instance.")
                    self.reranker_model_instance = self.embedding_model_instance
                else:
                    self.reranker_model_instance = BatchedInference(model_id=self.reranking_model_name, engine="torch", device=device)
                    logger.info("Loaded Reranker model using BatchedInference.")
            elif not self.use_reranking: logger.info("Cross-encoder reranking disabled.")
            elapsed_time = time.time() - start_time
            logger.info(f"Embedding/Reranking model loading completed in {elapsed_time:.2f} seconds")
            log_memory_usage(logger)
        except Exception as e:
            logger.error(f"Error loading embedding/reranking models: {e}", exc_info=True)
            self.embedding_model_instance = None; self.reranker_model_instance = None; raise

    def unload_models(self):
        """Unload the embedding and reranking models."""
        logger.info("===== UNLOADING QUERY ENGINE MODELS (Embed/Rerank) =====")
        unloaded_something = False
        if self.reranker_model_instance is not None:
            logger.info(f"Unloading reranking model: {self.reranking_model_name}")
            try:
                is_shared_instance = (not self.is_bge_m3 and self.embedding_model_instance is not None and self.reranker_model_instance is self.embedding_model_instance)
                if is_shared_instance: logger.info("Reranker instance shared, will unload with embedder.")
                elif hasattr(self.reranker_model_instance, 'stop'): self.reranker_model_instance.stop(); unloaded_something = True; logger.info("Reranker stopped.")
                else: del self.reranker_model_instance; unloaded_something = True; logger.info("Reranker deleted.")
            except Exception as e: logger.error(f"Error unloading reranker: {e}", exc_info=True)
            finally:
                 if not locals().get('is_shared_instance', False): self.reranker_model_instance = None
        if self.embedding_model_instance is not None:
            logger.info(f"Unloading embedding model: {self.embedding_model_name}")
            try:
                if self.is_bge_m3 and hasattr(self.embedding_model_instance, 'model'):
                    if hasattr(self.embedding_model_instance.model, 'cpu'): self.embedding_model_instance.model.cpu()
                    del self.embedding_model_instance.model; del self.embedding_model_instance.tokenizer
                elif not self.is_bge_m3 and hasattr(self.embedding_model_instance, 'stop'): self.embedding_model_instance.stop()
                del self.embedding_model_instance; self.embedding_model_instance = None; unloaded_something = True
                logger.info(f"Embedding model {self.embedding_model_name} unloaded.")
            except Exception as e: logger.error(f"Error unloading embedding model: {e}", exc_info=True); self.embedding_model_instance = None
        if unloaded_something:
            if torch.cuda.is_available(): logger.info("Clearing CUDA cache..."); torch.cuda.empty_cache()
            logger.info("Running garbage collection..."); gc.collect()
            logger.info("===== MODEL UNLOAD COMPLETE ====="); log_memory_usage(logger)
        else: logger.debug("No models were loaded, nothing to unload.")

    def _load_bm25_index(self) -> bool:
        """Load the BM25 index, tokenized texts, and metadata from file."""
        if self.bm25_index is not None: return True
        try:
            if not self.bm25_path.exists(): logger.warning(f"BM25 index file not found: {self.bm25_path}"); return False
            logger.info(f"Loading BM25 index from {self.bm25_path}")
            with open(self.bm25_path, 'rb') as f: loaded_data = pickle.load(f)
            if not isinstance(loaded_data, tuple) or len(loaded_data) != 3: logger.error("Invalid BM25 data format"); return False
            self.bm25_index, self.tokenized_texts, self.bm25_metadata = loaded_data
            if not hasattr(self.bm25_index, 'get_scores'): logger.error("BM25 index lacks 'get_scores'"); return False
            if not isinstance(self.tokenized_texts, list): logger.error("Tokenized texts not a list"); return False
            if not isinstance(self.bm25_metadata, list): logger.error("BM25 metadata not a list"); return False
            if len(self.tokenized_texts) != len(self.bm25_metadata): logger.warning("BM25 text/metadata count mismatch.")
            logger.info(f"BM25 index loaded successfully ({len(self.tokenized_texts)} documents).")
            return True
        except Exception as e: logger.error(f"Error loading BM25 index: {e}", exc_info=True); self.bm25_index = None; self.tokenized_texts = None; self.bm25_metadata = None; return False

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the Qdrant collection."""
        try:
            client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port, timeout=10)
            try: collections = client.get_collections().collections; collection_names = [c.name for c in collections]
            except Exception as conn_err: return {'exists': False, 'error': f"Connection error: {conn_err}"}
            if self.qdrant_collection in collection_names:
                info = client.get_collection(collection_name=self.qdrant_collection)
                vector_size = 0; distance_metric = "Unknown"; has_dense = False; has_sparse = False
                if isinstance(info.config.params.vectors, dict): # Named vectors
                    if 'dense' in info.config.params.vectors: vector_size = info.config.params.vectors['dense'].size; distance_metric = str(info.config.params.vectors['dense'].distance); has_dense = True
                elif info.config.params.vectors: # Default unnamed vector
                    vector_size = info.config.params.vectors.size; distance_metric = str(info.config.params.vectors.distance); has_dense = True
                if info.config.params.sparse_vectors and 'sparse' in info.config.params.sparse_vectors: has_sparse = True
                schema_type = "Unknown"
                if has_dense and has_sparse: schema_type = "BGE-M3 (Dense+Sparse)"
                elif has_dense: schema_type = "E5/Dense Only"
                result = {'exists': True, 'name': self.qdrant_collection, 'points_count': info.points_count, 'vector_size': vector_size, 'distance': distance_metric, 'schema_type': schema_type, 'has_sparse': has_sparse}
                return result
            else: return {'exists': False}
        except Exception as e: logger.error(f"Error getting collection info: {e}", exc_info=True); return {'exists': False, 'error': str(e)}

    def clear_collection(self) -> bool:
        """Clear Qdrant collection, BM25 index, and extracted data."""
        try:
            try:
                client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
                collections = client.get_collections().collections; collection_names = [c.name for c in collections]
                if self.qdrant_collection in collection_names: client.delete_collection(collection_name=self.qdrant_collection); logger.info(f"Qdrant collection '{self.qdrant_collection}' deleted"); time.sleep(1)
                else: logger.warning(f"Qdrant collection '{self.qdrant_collection}' does not exist.")
            except Exception as e: logger.error(f"Error clearing Qdrant collection: {e}", exc_info=True)
            try:
                if self.bm25_path.exists(): self.bm25_path.unlink(); logger.info(f"BM25 index file deleted: {self.bm25_path}")
                stopwords_path = self.bm25_path.parent / "stopwords.pkl"
                if stopwords_path.exists(): stopwords_path.unlink(); logger.info(f"Stopwords file deleted: {stopwords_path}")
            except Exception as e: logger.error(f"Error deleting BM25 files: {e}")
            try:
                # Use config passed during init to find extracted data path
                storage_config = self.config.get("storage", {})
                extracted_data_path_str = storage_config.get("extracted_data_path", "data/extracted")
                extracted_data_path = self.root_dir / extracted_data_path_str
                entities_path = extracted_data_path / "entities.json"; relationships_path = extracted_data_path / "relationships.json"
                if entities_path.exists(): entities_path.unlink(); logger.info(f"Entities file deleted: {entities_path}")
                if relationships_path.exists(): relationships_path.unlink(); logger.info(f"Relationships file deleted: {relationships_path}")
            except Exception as e: logger.error(f"Error deleting extracted data files: {e}")
            self.bm25_index = None; self.tokenized_texts = None; self.bm25_metadata = None; logger.info("Internal BM25 state cleared.")
            return True
        except Exception as e: logger.error(f"Error clearing all data: {e}", exc_info=True); return False

    def get_chunks(self, limit: int = 20, search_text: str = None, document_filter: str = None) -> List[Dict[str, Any]]:
        """Retrieve chunks based on optional filters."""
        try:
            client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
            filter_conditions = []
            if search_text: filter_conditions.append(rest.FieldCondition(key="text", match=rest.MatchText(text=search_text)))
            if document_filter: filter_conditions.append(rest.FieldCondition(key="file_name", match=rest.MatchValue(value=document_filter)))
            filter_obj = rest.Filter(must=filter_conditions) if filter_conditions else None
            try:
                collections = client.get_collections().collections
                if self.qdrant_collection not in [c.name for c in collections]: logger.warning(f"Collection {self.qdrant_collection} does not exist"); return []
            except Exception as conn_err: logger.error(f"Qdrant connection error: {conn_err}"); return []
            scroll_result, _ = scroll_with_filter_compatible(client=client, collection_name=self.qdrant_collection, limit=limit, with_payload=True, with_vectors=False, scroll_filter=filter_obj)
            points = scroll_result
            if not points: logger.info(f"No points found matching filter: text='{search_text}', doc='{document_filter}'"); return []
            chunks = []
            for point in points:
                payload = point.payload or {}; text = payload.get('text', ''); original_text = payload.get('original_text', text)
                metadata_from_payload = {k: v for k, v in payload.items() if k not in ['text', 'original_text', 'metadata']}
                if 'metadata' in payload and isinstance(payload['metadata'], dict): metadata_from_payload = {**payload['metadata'], **metadata_from_payload}
                if 'chunk_id' not in metadata_from_payload: metadata_from_payload['chunk_id'] = point.id
                if 'file_name' not in metadata_from_payload: metadata_from_payload['file_name'] = 'Unknown'
                chunk = {'id': point.id, 'text': text, 'original_text': original_text, 'metadata': metadata_from_payload}
                chunks.append(chunk)
            logger.info(f"Retrieved {len(chunks)} chunks for exploration.")
            return chunks
        except Exception as e: logger.error(f"Error retrieving chunks: {e}", exc_info=True); return []

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """ Retrieve relevant chunks using appropriate strategy and reranking. """
        try:
            start_time = time.time()
            if self.embedding_model_instance is None or (self.use_reranking and self.reranker_model_instance is None):
                logger.info("Loading embedding/reranking models for retrieval...")
                self.load_embedding_model()
            if not self.is_bge_m3 and self.bm25_index is None:
                if not self._load_bm25_index(): logger.warning("BM25 index not available for E5 retrieval.")
                else: logger.info("BM25 index loaded for E5 retrieval.")

            if self.is_bge_m3:
                logger.info("Performing BGE-M3 retrieval (Dense + Sparse)...")
                fused_results = self._vector_search(query) # Handles dense+sparse+fusion
                logger.info(f"BGE-M3 RRF fusion returned {len(fused_results)} candidates.")
            else: # E5 Instruct
                logger.info("Performing E5-Instruct + BM25 retrieval with RRF fusion...")
                bm25_results = self._bm25_search(query) if self.bm25_index is not None else []
                vector_results = self._vector_search(query) # E5 only does dense here
                fused_results = self._fuse_results(bm25_results, vector_results)
                logger.info(f"E5+BM25 RRF fusion resulted in {len(fused_results)} candidates.")

            if self.use_reranking and fused_results and self.reranker_model_instance is not None:
                logger.info("Applying cross-encoder reranking...")
                final_results = self._rerank(query, fused_results)
            elif self.use_reranking:
                 logger.warning("Reranking enabled but model not loaded or no initial results. Skipping.")
                 final_results = sorted(fused_results, key=lambda x: x['score'], reverse=True)[:self.top_k_rerank]
            else:
                logger.info("Cross-encoder reranking disabled.")
                final_results = sorted(fused_results, key=lambda x: x['score'], reverse=True)[:self.top_k_rerank]

            elapsed_time = time.time() - start_time
            logger.info(f"Retrieval completed in {elapsed_time:.2f}s. Returning {len(final_results)} final results.")
            return final_results
        except Exception as e: logger.error(f"Error in retrieval orchestration: {e}", exc_info=True); return []

    def _bm25_search(self, query: str) -> List[Dict[str, Any]]:
        """Performs BM25 search using the loaded index."""
        if not self.bm25_index or not self.tokenized_texts or not self.bm25_metadata: return []
        try:
            import nltk
            from nltk.tokenize import word_tokenize
            try: nltk.data.find('tokenizers/punkt')
            except LookupError: nltk.download('punkt', quiet=True)
            stop_words = set()
            try:
                stopwords_path = self.bm25_path.parent / "stopwords.pkl"
                if stopwords_path.exists():
                    with open(stopwords_path, 'rb') as f: stop_words = pickle.load(f)
                else: nltk.download('stopwords', quiet=True); stop_words = set(nltk.corpus.stopwords.words('english'))
            except Exception as e: logger.warning(f"Error loading stopwords: {e}. Using NLTK defaults."); nltk.download('stopwords', quiet=True); stop_words = set(nltk.corpus.stopwords.words('english'))
            tokenized_query = word_tokenize(query.lower()); filtered_query = [token for token in tokenized_query if token.isalnum() and token not in stop_words]
            if not filtered_query and tokenized_query: filtered_query = [t for t in tokenized_query if t.isalnum()]
            if not filtered_query: logger.warning("BM25 query empty after filtering."); return []
            bm25_scores = self.bm25_index.get_scores(filtered_query); results = []
            doc_count = len(self.bm25_metadata)
            for i, score in enumerate(bm25_scores):
                if score > 0 and i < doc_count:
                    metadata = self.bm25_metadata[i]
                    if not isinstance(metadata, dict): logger.warning(f"Skipping BM25 result {i}: invalid metadata type {type(metadata)}"); continue
                    text = metadata.get('text', ''); original_text = metadata.get('original_text', text)
                    chunk_id = metadata.get('chunk_id', f"bm25_doc_{i}"); result_metadata = metadata.copy(); result_metadata['chunk_id'] = chunk_id
                    results.append({'id': chunk_id, 'score': float(score), 'text': text, 'original_text': original_text, 'metadata': result_metadata})
            results.sort(key=lambda x: x['score'], reverse=True); results = results[:self.top_k_bm25]
            logger.info(f"BM25 search returned {len(results)} results")
            return results
        except Exception as e: logger.error(f"Error in BM25 search: {e}", exc_info=True); return []

    def _vector_search(self, query: str) -> List[Dict[str, Any]]:
        """ Performs vector search (Dense+Sparse for BGE-M3, Dense only for E5). """
        if self.embedding_model_instance is None: logger.error("Embedding model not loaded."); return []
        try:
            client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port, timeout=10)
            try: collections = client.get_collections().collections
            except Exception as conn_err: logger.error(f"Qdrant connection error: {conn_err}"); return []
            if self.qdrant_collection not in [c.name for c in collections]: logger.warning(f"Collection '{self.qdrant_collection}' does not exist."); return []
            def format_qdrant_results(qdrant_list):
                formatted = []
                for point in qdrant_list:
                    payload = point.payload or {}; text = payload.get('text', ''); original_text = payload.get('original_text', text)
                    metadata_from_payload = {k: v for k, v in payload.items() if k not in ['text', 'original_text']}
                    chunk_id = metadata_from_payload.get('chunk_id', point.id); metadata_from_payload['chunk_id'] = chunk_id
                    formatted.append({'id': point.id, 'score': float(point.score), 'text': text, 'original_text': original_text, 'metadata': metadata_from_payload})
                return formatted
            if self.is_bge_m3:
                logger.info("Generating BGE-M3 query embeddings (dense, sparse)...")
                query_outputs = self.embedding_model_instance.encode([query], return_dense=True, return_sparse=True, return_colbert_vecs=False)
                dense_vec = query_outputs["dense_vecs"][0]; sparse_weights = query_outputs["lexical_weights"][0]
                qdrant_sparse = _create_sparse_vector_qdrant_format(sparse_weights); dense_results_list = []; sparse_results_list = []
                logger.info(f"Performing Qdrant BGE-M3 dense search (limit: {self.top_k_vector})...")
                try:
                    qdrant_dense_results = client.search(collection_name=self.qdrant_collection, query_vector=rest.NamedVector(name="dense", vector=dense_vec.tolist()), limit=self.top_k_vector, with_payload=True)
                    if qdrant_dense_results: dense_results_list = qdrant_dense_results
                    logger.info(f"Dense search returned {len(dense_results_list)} results.")
                except Exception as dense_err: logger.error(f"BGE-M3 dense search error: {dense_err}", exc_info=True)
                if qdrant_sparse:
                    logger.info(f"Performing Qdrant BGE-M3 sparse search (limit: {self.top_k_vector})...")
                    try:
                        qdrant_sparse_results = client.search(collection_name=self.qdrant_collection, query_vector=rest.NamedSparseVector(name="sparse", vector=qdrant_sparse), limit=self.top_k_vector, with_payload=True)
                        if qdrant_sparse_results: sparse_results_list = qdrant_sparse_results
                        logger.info(f"Sparse search returned {len(sparse_results_list)} results.")
                    except Exception as sparse_err: logger.error(f"BGE-M3 sparse search error: {sparse_err}", exc_info=True)
                else: logger.warning("Skipping BGE-M3 sparse search (empty query vector).")
                formatted_dense = format_qdrant_results(dense_results_list); formatted_sparse = format_qdrant_results(sparse_results_list)
                logger.info("Fusing BGE-M3 dense and sparse results using RRF...")
                fused_vector_results = self._fuse_results(formatted_sparse, formatted_dense)
                logger.info(f"BGE-M3 RRF Fusion returned {len(fused_vector_results)} results")
                return fused_vector_results
            else: # E5 Instruct
                logger.info("Generating E5-Instruct query embedding (dense only)...")
                formatted_query = self._format_query_for_e5(query)
                future = self.embedding_model_instance.embed(sentences=[formatted_query], model_id=self.embedding_model_name)
                result = future.result(); embeddings = result[0] if isinstance(result, tuple) else result
                if not embeddings: logger.error("Failed to generate E5 query embedding."); return []
                query_embedding = embeddings[0]
                if hasattr(query_embedding, 'tolist'): vector = query_embedding.tolist()
                elif isinstance(query_embedding, (list, np.ndarray)): vector = list(query_embedding)
                else: raise TypeError(f"Invalid E5 embedding type: {type(query_embedding)}")
                if not vector: logger.error("E5 query vector is empty."); return []
                logger.info(f"Performing Qdrant E5 dense vector search (limit: {self.top_k_vector})...")
                qdrant_results = client.search(collection_name=self.qdrant_collection, query_vector=vector, limit=self.top_k_vector, with_payload=True)
                search_results = qdrant_results if qdrant_results else []
                logger.info(f"Qdrant E5 search returned {len(search_results)} results.")
                return format_qdrant_results(search_results)
        except Exception as e: logger.error(f"Error in vector search: {e}", exc_info=True); return []

    def _fuse_results(self, list1: List[Dict[str, Any]], list2: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """ Fuse results from two lists using RRF. """
        if not list1 and not list2: return []
        if not list1: return sorted(list2, key=lambda x: x['score'], reverse=True)[:self.top_k_hybrid]
        if not list2: return sorted(list1, key=lambda x: x['score'], reverse=True)[:self.top_k_hybrid]
        k = 60; dict1 = {}; dict2 = {}
        list1_sorted = sorted(list1, key=lambda x: x['score'], reverse=True)
        for rank, result in enumerate(list1_sorted):
            doc_id = result.get('id');
            if doc_id and doc_id not in dict1: dict1[doc_id] = {'rank': rank + 1, **result}
        list2_sorted = sorted(list2, key=lambda x: x['score'], reverse=True)
        for rank, result in enumerate(list2_sorted):
            doc_id = result.get('id');
            if doc_id and doc_id not in dict2: dict2[doc_id] = {'rank': rank + 1, **result}
        all_doc_ids = set(dict1.keys()) | set(dict2.keys()); rrf_results = []
        for doc_id in all_doc_ids:
            rank1 = dict1.get(doc_id, {'rank': len(list1) * 2})['rank']
            rank2 = dict2.get(doc_id, {'rank': len(list2) * 2})['rank']
            rrf_score = (self.bm25_weight / (k + rank1)) + (self.vector_weight / (k + rank2))
            doc_info = dict2.get(doc_id) or dict1.get(doc_id)
            if not doc_info: logger.warning(f"Could not find doc info for fused ID {doc_id}"); continue
            metadata = doc_info.get('metadata', {}); metadata['chunk_id'] = doc_id
            rrf_results.append({'id': doc_id, 'score': rrf_score, 'bm25_rank': rank1 if doc_id in dict1 else None, 'vector_rank': rank2 if doc_id in dict2 else None, 'text': doc_info['text'], 'original_text': doc_info.get('original_text', doc_info['text']), 'metadata': metadata})
        rrf_results.sort(key=lambda x: x['score'], reverse=True); results = rrf_results[:self.top_k_hybrid]
        logger.info(f"RRF Fusion returned {len(results)} results")
        return results

    def _rerank(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank results using the cross-encoder model."""
        if not self.use_reranking or self.reranker_model_instance is None or not self.reranking_model_name or not results: return results
        try:
            texts_to_rerank = [result.get('original_text', result.get('text', '')) for result in results]
            if not any(texts_to_rerank): logger.warning("No text found in results to rerank."); return results
            logger.info(f"Reranking {len(texts_to_rerank)} results using: {self.reranking_model_name}")
            future = self.reranker_model_instance.rerank(query=query, docs=texts_to_rerank, model_id=self.reranking_model_name)
            rerank_result = future.result(); scores_data = rerank_result[0] if isinstance(rerank_result, tuple) else rerank_result
            if isinstance(scores_data, list) and scores_data and isinstance(scores_data[0], tuple) and len(scores_data[0])==2:
                 scores_data.sort(key=lambda x: x[0]); scores_only = [score for _, score in scores_data]
            elif isinstance(scores_data, list): scores_only = scores_data
            else: logger.error(f"Unexpected reranker scores format: {type(scores_data)}. Skipping."); return results
            if len(scores_only) != len(results): logger.error(f"Reranker score count mismatch: got {len(scores_only)}, expected {len(results)}. Skipping."); return results
            reranked_results = []
            for i, score in enumerate(scores_only):
                result = results[i].copy(); result['original_score'] = result['score']; result['score'] = float(score); reranked_results.append(result)
            reranked_results.sort(key=lambda x: x['score'], reverse=True)
            filtered_results = [result for result in reranked_results if result['score'] >= self.min_score_threshold]
            final_reranked = filtered_results[:self.top_k_rerank]
            logger.info(f"Cross-encoder reranking completed. Returning {len(final_reranked)} results.")
            return final_reranked
        except Exception as e: logger.error(f"Error during cross-encoder reranking: {e}", exc_info=True); return sorted(results, key=lambda x: x['score'], reverse=True)[:self.top_k_rerank]

# --- END OF REWRITTEN FILE: src/core/query_system/query_engine.py ---