# --- START OF REWRITTEN FILE: src/core/query_system/query_engine.py ---

"""
Query engine for Anti-Corruption RAG System.
Handles retrieval and generation using persistent Aphrodite LLM service.
MODIFIED: Adapted for intfloat/multilingual-e5-large-instruct embedding model.
(Non-streaming version)
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
from typing import List, Dict, Any, Union, Optional, Tuple, Generator

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))

# Import utils
from src.utils.logger import setup_logger
from src.utils.resource_monitor import log_memory_usage
from src.utils.qdrant_utils import scroll_with_filter_compatible
# Import the service getter
from src.utils.aphrodite_service import get_service

# Initialize logger
logger = setup_logger(__name__)

# Load configuration
CONFIG_PATH = ROOT_DIR / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

# --- E5-INSTRUCT TASK DEFINITION ---
# Define the specific instruction for retrieval task used with E5-Instruct model
# This should match the task the model was trained/fine-tuned for.
# Reference: https://huggingface.co/intfloat/multilingual-e5-large-instruct
RETRIEVAL_TASK_INSTRUCTION = 'Given a web search query, retrieve relevant passages that answer the query'
# Consider moving this to config.yaml if you need more flexibility
# --- END OF E5-INSTRUCT TASK DEFINITION ---


class QueryEngine:
    """
    Query engine for retrieval and generation.
    Uses a persistent Aphrodite service in a child process.
    Adapted for E5-Instruct embedding model.
    (Non-streaming version)
    """

    def __init__(self):
        """Initialize query engine."""
        # Load configuration settings
        self.qdrant_host = CONFIG["qdrant"]["host"]
        self.qdrant_port = CONFIG["qdrant"]["port"]
        self.qdrant_collection = CONFIG["qdrant"]["collection_name"]
        self.top_k_vector = CONFIG["retrieval"]["top_k_vector"]
        self.top_k_bm25 = CONFIG["retrieval"]["top_k_bm25"]
        self.top_k_hybrid = CONFIG["retrieval"]["top_k_hybrid"]
        self.top_k_rerank = CONFIG["retrieval"]["top_k_rerank"]
        self.vector_weight = CONFIG["retrieval"]["vector_weight"]
        self.bm25_weight = CONFIG["retrieval"]["bm25_weight"]
        self.use_reranking = CONFIG["retrieval"]["use_reranking"]
        self.bm25_path = ROOT_DIR / CONFIG["storage"]["bm25_index_path"]
        # Use the designated chat model for querying (from config)
        self.llm_model_name = CONFIG["models"]["chat_model"]
        # Use the designated embedding model (from config - should be E5-Instruct)
        self.embedding_model_name = CONFIG["models"]["embedding_model"]
        # Use the designated reranking model (from config)
        self.reranking_model_name = CONFIG["models"].get("reranking_model", "BAAI/bge-reranker-v2-m3") # Default if missing

        # Initialize models and service reference
        self.embed_register = None # For embedding and reranking via embed library
        # Get the singleton service instance for Aphrodite (generative LLM)
        self.aphrodite_service = get_service()
        self.bm25_index = None
        self.tokenized_texts = None
        self.bm25_metadata = None

        logger.info(f"Initializing QueryEngine with:")
        logger.info(f"  - Embedding Model: {self.embedding_model_name}")
        logger.info(f"  - Reranking Model: {self.reranking_model_name}")
        logger.info(f"  - LLM (Chat) Model: {self.llm_model_name}")
        logger.info(f"  - Qdrant: {self.qdrant_host}:{self.qdrant_port}, Collection: {self.qdrant_collection}")
        logger.info(f"  - Reranking Enabled: {self.use_reranking}")
        log_memory_usage(logger)

    # --- NEW HELPER FUNCTION for E5-Instruct Query Formatting ---
    def _format_query_for_e5(self, query: str) -> str:
        """Formats the raw user query with the instruction prefix for E5-instruct models."""
        # Uses the globally defined instruction specific to the retrieval task
        logger.debug(f"Formatting query for E5 embedding: Task='{RETRIEVAL_TASK_INSTRUCTION}', Query='{query[:50]}...'")
        return f'Instruct: {RETRIEVAL_TASK_INSTRUCTION}\nQuery: {query}'
    # --- END OF NEW HELPER FUNCTION ---

    def load_embedding_model(self):
        """Load the embedding and reranking models via BatchedInference if not already loaded."""
        if self.embed_register is not None:
            logger.debug("Embedding/reranking models (via embed register) already loaded.")
            return
        logger.info("===== STARTING EMBEDDING/RERANKING MODEL LOAD (embed register) =====")
        try:
            from embed import BatchedInference # Ensure embed library is installed
            device = "cuda" if torch.cuda.is_available() else "cpu"
            models_to_load = [self.embedding_model_name] # Always load embedding model

            if self.use_reranking and self.reranking_model_name:
                logger.info(f"Loading embedding ({self.embedding_model_name}) and reranking ({self.reranking_model_name}) models on {device}")
                # Avoid adding duplicates if embedding and reranking models are somehow the same (shouldn't be)
                if self.reranking_model_name not in models_to_load:
                    models_to_load.append(self.reranking_model_name)
            else:
                logger.info(f"Loading embedding model ({self.embedding_model_name}) on {device}. Reranking disabled or no model specified.")

            start_time = time.time()
            self.embed_register = BatchedInference(
                model_id=models_to_load, # Pass list of models to load
                engine="torch", # Or "optimum" if configured/preferred
                device=device
            )
            elapsed_time = time.time() - start_time
            logger.info(f"Embed register with model(s) {models_to_load} loaded in {elapsed_time:.2f} seconds")
            log_memory_usage(logger)
        except ImportError:
             logger.error("Failed to import BatchedInference. Please install the 'embed' library (`pip install embed`).")
             self.embed_register = None
             raise # Re-raise error as it's critical
        except Exception as e:
            logger.error(f"Error loading embedding/reranking models via embed register: {e}", exc_info=True)
            self.embed_register = None # Ensure it's None on failure
            raise # Re-raise error

    def unload_embedding_model(self):
        """Unload the embedding/reranking models by stopping the BatchedInference register."""
        if self.embed_register is not None:
            logger.info("===== STARTING EMBEDDING/RERANKING MODEL UNLOAD (embed register) =====")
            try:
                if hasattr(self.embed_register, 'stop'):
                    logger.info("Calling embed_register.stop()...")
                    self.embed_register.stop()
                    logger.info("embed_register stopped.")
                else:
                     logger.warning("Embed register does not have a 'stop' method.")
            except Exception as e:
                logger.warning(f"Error stopping embed register: {e}")

            del self.embed_register
            self.embed_register = None

            if torch.cuda.is_available():
                logger.info("Clearing CUDA cache...")
                torch.cuda.empty_cache()
            logger.info("Running garbage collection...")
            gc.collect()
            logger.info("===== EMBEDDING/RERANKING MODEL UNLOAD COMPLETE =====")
            log_memory_usage(logger)
        else:
             logger.debug("Embedding/reranking models (embed register) already unloaded.")

    def ensure_llm_loaded(self):
        """
        Ensure the designated LLM model (for chat/querying) is loaded in the Aphrodite service.
        (No changes needed from original for E5 integration)
        """
        # Check if the service is running
        if not self.aphrodite_service.is_running():
            logger.info("Aphrodite service not running, starting it")
            if not self.aphrodite_service.start():
                logger.error("Failed to start Aphrodite service")
                return False

        # Check status to see if the correct model is loaded
        status = self.aphrodite_service.get_status()
        logger.info(f"Aphrodite service status: {status}")

        # If no model is loaded, or the wrong model is loaded, load the correct one
        if not status.get("model_loaded", False) or status.get("current_model") != self.llm_model_name:
            logger.info(f"Loading LLM model for querying via Aphrodite: {self.llm_model_name}")
            # Load generically, no is_chat flag needed
            if not self.aphrodite_service.load_model(self.llm_model_name):
                logger.error(f"Failed to load LLM model {self.llm_model_name} via Aphrodite")
                return False
            logger.info(f"LLM model {self.llm_model_name} loaded successfully via Aphrodite.")
        else:
             logger.info(f"LLM model {self.llm_model_name} already loaded via Aphrodite.")

        return True

    def _load_bm25_index(self) -> bool:
        """
        Load the BM25 index from disk.
        (No changes needed from original for E5 integration)
        """
        try:
            if not self.bm25_path.exists():
                logger.warning(f"BM25 index file not found: {self.bm25_path}")
                self.bm25_index = None # Ensure state reflects not loaded
                return False

            logger.info(f"Loading BM25 index from {self.bm25_path}")
            with open(self.bm25_path, 'rb') as f:
                # Order depends on how it was saved in document_indexer.py
                self.bm25_index, self.tokenized_texts, self.bm25_metadata = pickle.load(f)

            # Simple validation
            if not hasattr(self.bm25_index, 'get_scores') or not isinstance(self.tokenized_texts, list) or not isinstance(self.bm25_metadata, list):
                 logger.error(f"Invalid data format loaded from BM25 index file: {self.bm25_path}")
                 self.bm25_index, self.tokenized_texts, self.bm25_metadata = None, None, None
                 return False

            logger.info(f"BM25 index loaded with {len(self.tokenized_texts)} documents")
            return True

        except Exception as e:
            logger.error(f"Error loading BM25 index: {e}", exc_info=True)
            self.bm25_index, self.tokenized_texts, self.bm25_metadata = None, None, None
            return False

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the vector collection.
        (No changes needed from original for E5 integration, uses corrected Qdrant attribute access)
        """
        try:
            from qdrant_client import QdrantClient

            client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port, timeout=10)

            try:
                collections = client.get_collections().collections
                collection_names = [collection.name for collection in collections]
            except Exception as conn_err:
                 logger.error(f"Failed to connect or get collections from Qdrant: {conn_err}")
                 return {'exists': False, 'error': f"Connection error: {conn_err}"}

            if self.qdrant_collection in collection_names:
                collection_info = client.get_collection(collection_name=self.qdrant_collection)
                vector_params = collection_info.config.params.vectors
                vector_size = vector_params.size
                distance_metric = str(vector_params.distance)

                result = {
                    'exists': True,
                    'name': self.qdrant_collection,
                    'points_count': collection_info.points_count,
                    'vector_size': vector_size,
                    'distance': distance_metric
                }
                logger.debug(f"Collection info retrieved: {result}")
                return result
            else:
                logger.warning(f"Collection '{self.qdrant_collection}' does not exist in Qdrant.")
                return {'exists': False}

        except AttributeError as ae:
             logger.error(f"Attribute error getting collection info (check qdrant-client version): {ae}")
             try:
                 basic_info = client.get_collection(collection_name=self.qdrant_collection)
                 return {'exists': True, 'points_count': basic_info.points_count, 'error': f"AttributeError: {ae}"}
             except Exception: return {'exists': False, 'error': f"AttributeError: {ae}"}
        except Exception as e:
            logger.error(f"Unexpected error getting collection info: {e}", exc_info=True)
            return {'exists': False, 'error': str(e)}

    def clear_collection(self) -> bool:
        """
        Clear the vector collection and associated data.
        (No changes needed from original for E5 integration)
        """
        # --- Keep original implementation ---
        try:
            from qdrant_client import QdrantClient, models

            # 1. Delete Qdrant collection
            try:
                client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
                collections = client.get_collections().collections
                collection_names = [collection.name for collection in collections]

                if self.qdrant_collection in collection_names:
                    client.delete_collection(collection_name=self.qdrant_collection)
                    logger.info(f"Qdrant collection '{self.qdrant_collection}' deleted")
                    time.sleep(1) # Give qdrant a moment
                    # Optional: verify deletion, add retry if needed
                else:
                    logger.warning(f"Qdrant collection '{self.qdrant_collection}' does not exist, skipping deletion.")
            except Exception as e:
                logger.error(f"Error clearing Qdrant collection '{self.qdrant_collection}': {e}", exc_info=True)

            # 2. Delete BM25 index file and stopwords
            try:
                if self.bm25_path.exists():
                    self.bm25_path.unlink()
                    logger.info(f"BM25 index file deleted: {self.bm25_path}")
                stopwords_path = self.bm25_path.parent / "stopwords.pkl"
                if stopwords_path.exists():
                    stopwords_path.unlink()
                    logger.info(f"Stopwords file deleted: {stopwords_path}")
            except Exception as e:
                logger.error(f"Error deleting BM25 files: {e}")

            # 3. Delete extracted data files
            try:
                extracted_data_path = ROOT_DIR / CONFIG["storage"]["extracted_data_path"]
                entities_path = extracted_data_path / "entities.json"
                relationships_path = extracted_data_path / "relationships.json"
                if entities_path.exists():
                    entities_path.unlink()
                    logger.info(f"Entities file deleted: {entities_path}")
                if relationships_path.exists():
                    relationships_path.unlink()
                    logger.info(f"Relationships file deleted: {relationships_path}")
            except Exception as e:
                logger.error(f"Error deleting extracted data files: {e}")

            # Reset internal BM25 state
            self.bm25_index = None
            self.tokenized_texts = None
            self.bm25_metadata = None

            return True

        except Exception as e:
            logger.error(f"Error clearing all data: {e}", exc_info=True)
            return False
        # --- End of original implementation ---

    def get_chunks(self, limit: int = 20, search_text: str = None, document_filter: str = None) -> List[Dict[str, Any]]:
        """
        Get chunks from the vector database for exploration.
        (No changes needed from original for E5 integration)
        """
        # --- Keep original implementation ---
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models as rest

            client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
            filter_conditions = []
            if search_text:
                filter_conditions.append(rest.FieldCondition(key="text", match=rest.MatchText(text=search_text)))
            if document_filter:
                filter_conditions.append(rest.FieldCondition(key="file_name", match=rest.MatchValue(value=document_filter)))
            filter_obj = rest.Filter(must=filter_conditions) if filter_conditions else None

            collections = client.get_collections().collections
            if self.qdrant_collection not in [c.name for c in collections]:
                logger.warning(f"Collection {self.qdrant_collection} does not exist")
                return []

            scroll_result, next_offset = scroll_with_filter_compatible(
                client=client, collection_name=self.qdrant_collection, limit=limit,
                with_payload=True, with_vectors=False, scroll_filter=filter_obj
            )
            points = scroll_result

            if not points:
                logger.info(f"No points found matching filter: text='{search_text}', doc='{document_filter}'")
                return []

            chunks = []
            for point in points:
                text = point.payload.get('text', '')
                original_text = point.payload.get('original_text', text)
                chunk = {
                    'id': point.id,
                    'text': original_text, # Use original text for display
                    'metadata': {
                        'file_name': point.payload.get('file_name', 'Unknown'),
                        'page_num': point.payload.get('page_num', None),
                        'document_id': point.payload.get('document_id', 'Unknown'),
                        'chunk_id': point.payload.get('chunk_id', point.id)
                    }
                }
                chunks.append(chunk)
            return chunks

        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}", exc_info=True)
            return []
        # --- End of original implementation ---

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks using hybrid search (BM25 + vector).
        (No changes needed in flow for E5 integration, relies on modified _vector_search)
        """
        try:
            start_time = time.time()

            # Ensure embedding/reranking models are loaded via embed register
            if self.embed_register is None:
                logger.info("Embedding/reranking models not loaded, loading now...")
                self.load_embedding_model() # This will load both if needed

            # Load BM25 index if not already loaded
            if self.bm25_index is None:
                if not self._load_bm25_index():
                    logger.warning("BM25 index not available, using vector search only.")

            # Get BM25 results (uses original query)
            bm25_results = self._bm25_search(query) if self.bm25_index is not None else []

            # Get vector search results (uses formatted query via _vector_search modification)
            vector_results = self._vector_search(query)

            # Combine results using RRF
            hybrid_results = self._fuse_results(bm25_results, vector_results)

            # Apply reranking if enabled (uses original query)
            if self.use_reranking and hybrid_results and self.embed_register is not None:
                logger.info("Applying reranking...")
                final_results = self._rerank(query, hybrid_results)
            else:
                logger.info("Skipping reranking.")
                final_results = hybrid_results # Use fused results directly

            elapsed_time = time.time() - start_time
            logger.info(f"Retrieval completed in {elapsed_time:.2f}s. Returning {len(final_results)} final results.")

            return final_results

        except Exception as e:
            logger.error(f"Error in retrieval orchestration: {e}", exc_info=True)
            return []

    def _bm25_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Search using BM25.
        (No changes needed from original for E5 integration)
        """
        # --- Keep original implementation ---
        if not self.bm25_index or not self.tokenized_texts or not self.bm25_metadata:
             logger.warning("BM25 components not loaded, cannot perform BM25 search.")
             return []
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
            except Exception as e: logger.warning(f"Error loading stopwords: {e}.")

            tokenized_query = word_tokenize(query.lower())
            filtered_query = [token for token in tokenized_query if token not in stop_words]
            if not filtered_query and tokenized_query: filtered_query = tokenized_query

            bm25_scores = self.bm25_index.get_scores(filtered_query)

            results = []
            for i, score in enumerate(bm25_scores):
                if score > 0 and i < len(self.bm25_metadata) and i < len(self.tokenized_texts):
                    metadata = self.bm25_metadata[i]
                    original_text = metadata.get('original_text', ' '.join(self.tokenized_texts[i]))
                    chunk_id = metadata.get('chunk_id', f"bm25_doc_{i}")
                    metadata['chunk_id'] = chunk_id # Ensure it's there
                    results.append({
                        'index': i, 'score': float(score),
                        'text': ' '.join(self.tokenized_texts[i]), # Text used for BM25 indexing
                        'original_text': original_text,
                        'metadata': metadata, 'id': chunk_id
                    })

            results.sort(key=lambda x: x['score'], reverse=True)
            results = results[:self.top_k_bm25]
            logger.info(f"BM25 search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error in BM25 search: {e}", exc_info=True)
            return []
        # --- End of original implementation ---

    def _vector_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Search using vector search (Qdrant).
        MODIFIED: Formats the query for E5-Instruct before embedding.
        """
        if self.embed_register is None:
             logger.error("Embedding model (embed register) not loaded for vector search.")
             return []
        try:
            from qdrant_client import QdrantClient

            # --- MODIFICATION START ---
            # Format the query specifically for E5-instruct embedding
            formatted_query = self._format_query_for_e5(query) # Call the helper
            logger.info(f"Formatted E5 query for embedding: '{formatted_query[:100]}...'")

            # Generate query embedding using the formatted query
            logger.debug(f"Requesting embedding for formatted query via embed register (model: {self.embedding_model_name})")
            future = self.embed_register.embed(
                sentences=[formatted_query], # Pass the formatted query
                model_id=self.embedding_model_name # Specify the E5 model
            )
            # --- MODIFICATION END ---

            # Get embedding result (blocks until ready)
            result = future.result()

            # Handle potential tuple format (embeddings, token_usage)
            if isinstance(result, tuple) and len(result) >= 1:
                embeddings = result[0]
            else:
                embeddings = result

            # Validate embeddings
            if not embeddings or len(embeddings) == 0:
                logger.error("Failed to generate query embedding - empty result from embed register.")
                return []
            query_embedding = embeddings[0]
            if not isinstance(query_embedding, (list, tuple)) and not hasattr(query_embedding, 'tolist'):
                 logger.error(f"Invalid query embedding format received: {type(query_embedding)}")
                 return []

            # Connect to Qdrant
            client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port, timeout=10)

            # Check if collection exists
            try:
                collections = client.get_collections().collections
                if self.qdrant_collection not in [c.name for c in collections]:
                    logger.warning(f"Qdrant collection '{self.qdrant_collection}' does not exist.")
                    return []
            except Exception as conn_err:
                 logger.error(f"Failed to connect to Qdrant to check collection: {conn_err}")
                 return []

            # Prepare vector for search
            try:
                if hasattr(query_embedding, 'tolist'): vector = query_embedding.tolist()
                elif isinstance(query_embedding, (list, tuple)): vector = list(query_embedding)
                else: raise TypeError("Embedding is not list-like or convertible")

                if not vector or len(vector) < 10:
                    logger.error(f"Query vector appears invalid or too short (length: {len(vector)}).")
                    return []
            except Exception as e:
                logger.error(f"Error preparing query vector: {e}", exc_info=True)
                return []

            # Search in Qdrant
            search_result = client.search(
                collection_name=self.qdrant_collection,
                query_vector=vector,
                limit=self.top_k_vector,
                with_payload=True # Need payload for text and metadata
            )

            # Format results
            results = []
            for point in search_result:
                text = point.payload.get('text', '') # This is the text stored during indexing
                original_text = point.payload.get('original_text', text) # Prefer original if available
                metadata = {k: v for k, v in point.payload.items() if k not in ['text', 'original_text']}
                chunk_id = metadata.get('chunk_id', point.id)
                metadata['chunk_id'] = chunk_id # Ensure consistent ID key
                results.append({
                    'id': point.id, # Qdrant's point ID
                    'score': float(point.score),
                    'text': text, # Text stored in Qdrant (potentially with tags)
                    'original_text': original_text, # Original text (prefer for display/reranking)
                    'metadata': metadata
                })

            logger.info(f"Vector search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error in vector search: {e}", exc_info=True)
            return []

    def _fuse_results(self, bm25_results: List[Dict[str, Any]], vector_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fuse results from BM25 and vector search using Reciprocal Rank Fusion (RRF).
        (No changes needed from original for E5 integration, uses 'id' which should be chunk_id)
        """
        # --- Keep original implementation ---
        if not bm25_results and not vector_results: return []
        if not bm25_results: return sorted(vector_results, key=lambda x: x['score'], reverse=True)[:self.top_k_hybrid]
        if not vector_results: return sorted(bm25_results, key=lambda x: x['score'], reverse=True)[:self.top_k_hybrid]

        k = 60 # RRF constant
        bm25_dict, vector_dict = {}, {}

        bm25_sorted = sorted(bm25_results, key=lambda x: x['score'], reverse=True)
        for rank, result in enumerate(bm25_sorted):
            doc_id = result.get('id') # Use chunk_id stored in 'id' field
            if doc_id and doc_id not in bm25_dict:
                bm25_dict[doc_id] = {'rank': rank + 1, **result}

        vector_sorted = sorted(vector_results, key=lambda x: x['score'], reverse=True)
        for rank, result in enumerate(vector_sorted):
             doc_id = result.get('metadata', {}).get('chunk_id', result.get('id')) # Prioritize chunk_id from metadata
             if doc_id and doc_id not in vector_dict:
                vector_dict[doc_id] = {'rank': rank + 1, **result}

        all_doc_ids = set(bm25_dict.keys()) | set(vector_dict.keys())
        rrf_results = []
        for doc_id in all_doc_ids:
            bm25_rank = bm25_dict.get(doc_id, {'rank': len(bm25_results) * 2})['rank']
            vector_rank = vector_dict.get(doc_id, {'rank': len(vector_results) * 2})['rank']
            rrf_score = (self.bm25_weight / (k + bm25_rank)) + (self.vector_weight / (k + vector_rank))

            doc_info = vector_dict.get(doc_id) or bm25_dict.get(doc_id)
            if not doc_info: continue

            rrf_results.append({
                'id': doc_id, # Store the chunk_id
                'score': rrf_score,
                'bm25_rank': bm25_rank if doc_id in bm25_dict else None,
                'vector_rank': vector_rank if doc_id in vector_dict else None,
                'text': doc_info['text'], # Text from source (BM25 or Vector)
                'original_text': doc_info.get('original_text', doc_info['text']), # Prefer original
                'metadata': doc_info['metadata']
            })

        rrf_results.sort(key=lambda x: x['score'], reverse=True)
        results = rrf_results[:self.top_k_hybrid]
        logger.info(f"Fusion returned {len(results)} results")
        return results
        # --- End of original implementation ---

    def _rerank(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank results using a reranker model (e.g., BAAI/bge-reranker-v2-m3).
        Uses the ORIGINAL query and ORIGINAL document texts.
        """
        min_score_threshold = CONFIG["retrieval"].get("minimum_score_threshold", 0.1)

        if self.embed_register is None:
             logger.error("Reranker model (embed register) not loaded.")
             return results # Return original results if reranker isn't available
        if not self.reranking_model_name:
             logger.warning("No reranking model name specified in config. Skipping reranking.")
             return results
        if not results:
             return []

        try:
            # Use 'original_text' if available, otherwise fall back to 'text'
            texts_to_rerank = [result.get('original_text', result.get('text', '')) for result in results]

            if not texts_to_rerank:
                logger.warning("No text found in results to rerank.")
                return results

            logger.info(f"Reranking {len(texts_to_rerank)} results using model: {self.reranking_model_name}")
            logger.debug(f"Reranking with original query: '{query[:100]}...'")

            # Rerank using the ORIGINAL query and the specified reranker model
            future = self.embed_register.rerank(
                query=query, # Use original query
                docs=texts_to_rerank, # Use original texts
                model_id=self.reranking_model_name # Specify the reranker model
            )
            rerank_result = future.result() # Blocks until completion

            # Process reranker output (handle tuple or list of scores/tuples)
            if isinstance(rerank_result, tuple):
                scores_data = rerank_result[0]
            else:
                scores_data = rerank_result

            # Extract scores, handling [(index, score)] or [score] formats
            if isinstance(scores_data, list) and scores_data and isinstance(scores_data[0], tuple):
                scores_data.sort(key=lambda x: x[0]) # Sort by original index
                scores_only = [score for _, score in scores_data]
            elif isinstance(scores_data, list):
                scores_only = scores_data # Assume scores are already in order
            else:
                logger.error(f"Unexpected reranker scores format: {type(scores_data)}. Skipping reranking.")
                return results

            if len(scores_only) != len(results):
                logger.error(f"Reranker score count mismatch: got {len(scores_only)}, expected {len(results)}. Skipping reranking.")
                return results

            # Update results with reranked scores
            reranked_results = []
            for i, score in enumerate(scores_only):
                result = results[i].copy()
                result['original_score'] = result['score'] # Keep original fused score
                result['score'] = float(score) # Update with reranker score
                reranked_results.append(result)

            # Sort by new reranker score
            reranked_results.sort(key=lambda x: x['score'], reverse=True)
            filtered_results = [result for result in reranked_results if result['score'] >= min_score_threshold]

            # Limit to final top_k_rerank
            final_reranked = filtered_results[:self.top_k_rerank]

            logger.info(f"Reranking completed. Returning {len(final_reranked)} results.")
            return final_reranked

        except Exception as e:
            logger.error(f"Error during reranking: {e}", exc_info=True)
            return results # Return original results on error

    def _create_chat_prompt(self, query: str, context: str) -> str:
        """
        Create a prompt for the chat model.
        (No changes needed from original for E5 integration)
        """
        # --- Keep original implementation ---
        # This prompt structure is for the final generative LLM, not the embedder.
        prompt = f"""<|im_start|>system
You are an expert assistant specializing in anti-corruption investigations and analysis. Your responses should be detailed, factual, and based ONLY on the provided context. If the answer is not found in the context, state that clearly. Do not make assumptions or use external knowledge.

When answering, cite your sources by referring to the numbers in square brackets corresponding to the context snippets provided below. For example, mention "[1]" if the information comes from the first context snippet. You can cite multiple sources like "[1], [3]".<|im_end|>
<|im_start|>user
Context:
{context}

Based ONLY on the context provided above, answer the following question:
Question: {query}<|im_end|>
<|im_start|>assistant
"""
        return prompt
        # --- End of original implementation ---

    def load_llm_model(self):
        """
        Ensure the designated LLM model (for chat) is loaded in the Aphrodite service.
        (No changes needed from original for E5 integration)
        """
        # --- Keep original implementation ---
        logger.info(f"Requesting check/load of LLM model {self.llm_model_name} in Aphrodite service...")
        if not self.aphrodite_service.is_running():
            logger.info("Aphrodite service not running, attempting to start...")
            if not self.aphrodite_service.start():
                logger.error("Failed to start Aphrodite service for LLM loading.")
                return False

        success = self.aphrodite_service.load_model(self.llm_model_name)
        if success:
            logger.info(f"LLM model '{self.llm_model_name}' confirmed loaded in service.")
            return True
        else:
            logger.error(f"Failed to load/confirm LLM model '{self.llm_model_name}' in service.")
            return False
        # --- End of original implementation ---

    def query(self, query: str) -> Dict[str, Any]:
        """
        Query the system (non-streaming). Retrieves context and generates a response
        using the persistent Aphrodite service.
        (No changes needed in flow for E5 integration, relies on modified sub-methods)
        """
        try:
            # 1. Check if generative LLM service/model is ready
            if not self.ensure_llm_loaded():
                 logger.error("LLM service/model not available. Cannot generate response.")
                 # Provide a user-friendly error, maybe suggest starting the service.
                 return {'answer': "Error: The Language Model service is not ready. Please ensure it's started and the model is loaded (check sidebar status).", 'sources': []}

            # 2. Retrieve relevant context (using modified retrieve which handles E5 query)
            logger.info(f"Retrieving context for query: '{query[:50]}...'")
            start_retrieve = time.time()
            # retrieve() internally calls _vector_search (modified) and _rerank (verified)
            retrieval_results = self.retrieve(query)
            logger.info(f"Retrieval took {time.time() - start_retrieve:.2f}s, found {len(retrieval_results)} results for generation.")

            # 3. Prepare context for LLM
            sources = [
                {   # Use original_text for context quality, keep metadata and score
                    'text': result.get('original_text', result.get('text', '')),
                    'metadata': result.get('metadata', {}),
                    'score': result.get('score', 0.0)
                }
                for result in retrieval_results
            ]
            context_texts = [src['text'] for src in sources] # Text snippets for the prompt

            # Format context with source markers [1], [2], etc.
            context_for_prompt = "\n\n".join([f"[{i + 1}] {text}" for i, text in enumerate(context_texts)])
            if not context_for_prompt:
                logger.warning("No context retrieved for the query. LLM will be informed.")
                context_for_prompt = "No relevant context was found in the documents for this query."

            # 4. Create prompt for the generative LLM
            prompt = self._create_chat_prompt(query, context_for_prompt)
            logger.debug(f"Generated LLM prompt (truncated): {prompt[:200]}...")

            # 5. Generate response using the NON-STREAMING Aphrodite service method
            logger.info("Sending chat generation request to Aphrodite service...")
            start_generate = time.time()
            response = self.aphrodite_service.generate_chat(prompt=prompt) # Non-streaming call
            logger.info(f"Chat generation via Aphrodite took {time.time() - start_generate:.2f}s")

            # 6. Process response from Aphrodite
            if response and response.get("status") == "success":
                answer = response.get("result", "Error: No answer content received from LLM service.")
                logger.info(f"Received successful chat response (length: {len(answer)})")
                # Return final answer and the sources used to generate it
                return {'answer': answer, 'sources': sources}
            else:
                error_msg = response.get("error", "Unknown error during chat generation") if response else "No response from service"
                logger.error(f"Error from Aphrodite service during chat generation: {error_msg}")
                final_error_msg = f"Error generating response: {error_msg}"
                # Return error message but still include the sources that were retrieved
                return {'answer': final_error_msg, 'sources': sources}

        except Exception as e:
            logger.error(f"Unexpected error in query method: {e}", exc_info=True)
            return {'answer': f"Error: An unexpected error occurred during query processing ({str(e)})", 'sources': []}

# --- END OF REWRITTEN FILE: src/core/query_system/query_engine.py ---