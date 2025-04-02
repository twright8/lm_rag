"""
Embedding-based topic filtering module for Anti-Corruption RAG System.
Allows filtering large datasets based on semantic similarity to topics.
"""
import sys
from pathlib import Path
import torch
import numpy as np
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple, Set

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))

# Import utils
from src.utils.logger import setup_logger
from src.utils.resource_monitor import log_memory_usage

# Initialize logger
logger = setup_logger(__name__)


class EmbeddingFilter:
    """
    Embedding-based filter to retrieve the most relevant chunks from a large dataset.
    Uses E5-Instruct or similar embedding models for semantic similarity.
    """

    def __init__(self, query_engine):
        """
        Initialize the embedding filter with a query engine.

        Args:
            query_engine: QueryEngine instance with an initialized embedding model
        """
        self.query_engine = query_engine

        # Cache for document names (to avoid repeated Qdrant calls)
        self.document_names_cache = None

        logger.info("EmbeddingFilter initialized")
        log_memory_usage(logger)

    def get_document_names(self, refresh=False) -> List[str]:
        """
        Get unique document names from the vector database.

        Args:
            refresh: Whether to refresh the cache

        Returns:
            list: List of unique document names
        """
        if self.document_names_cache is not None and not refresh:
            return self.document_names_cache

        try:
            # Get a large number of chunks to find all document names
            all_chunks = self.query_engine.get_chunks(limit=1000)

            # Extract unique document names
            doc_names = sorted(list(set(
                c['metadata'].get('file_name', 'Unknown')
                for c in all_chunks
                if c['metadata'].get('file_name')
            )))

            self.document_names_cache = doc_names
            logger.info(f"Found {len(doc_names)} unique documents")
            return doc_names

        except Exception as e:
            logger.error(f"Error getting document names: {e}")
            return []

    def filter_by_topic(self,
                        topic_query: str,
                        top_k: int = 1000,
                        included_docs: Optional[Set[str]] = None,
                        excluded_docs: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
        """
        Filter chunks by semantic similarity to a topic.

        Args:
            topic_query: The topic description or query
            top_k: Number of top results to return
            included_docs: Set of document names to include (None = all)
            excluded_docs: Set of document names to exclude

        Returns:
            list: List of filtered chunks with similarity scores
        """
        start_time = time.time()
        logger.info(f"Filtering chunks by topic: '{topic_query}', top_k={top_k}")

        try:
            # Ensure embedding model is loaded
            if self.query_engine.embed_register is None:
                logger.info("Loading embedding model...")
                self.query_engine.load_embedding_model()

            # Format the query for E5-Instruct (reusing query_engine's method)
            formatted_query = self.query_engine._format_query_for_e5(topic_query)
            logger.info(f"Formatted query: '{formatted_query[:100]}...'")

            # Generate query embedding
            logger.debug(f"Generating embedding via embed register (model: {self.query_engine.embedding_model_name})")
            future = self.query_engine.embed_register.embed(
                sentences=[formatted_query],
                model_id=self.query_engine.embedding_model_name
            )

            # Get embedding result
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

            # Process the query embedding for Qdrant search
            if hasattr(query_embedding, 'tolist'):
                vector = query_embedding.tolist()
            elif isinstance(query_embedding, (list, tuple)):
                vector = list(query_embedding)
            else:
                raise TypeError("Embedding is not list-like or convertible")

            # Connect to Qdrant via query_engine
            from qdrant_client import QdrantClient
            # Import the Qdrant utility function for compatibility
            from src.utils.qdrant_utils import scroll_with_filter_compatible

            client = QdrantClient(
                host=self.query_engine.qdrant_host,
                port=self.query_engine.qdrant_port
            )

            # Create filter conditions if document filters provided
            filter_obj = None

            if included_docs or excluded_docs:
                from qdrant_client.http import models as rest

                filter_conditions = []

                # Include specific documents
                if included_docs:
                    filter_conditions.append(
                        rest.FieldCondition(
                            key="file_name",
                            match=rest.MatchAny(any=list(included_docs))
                        )
                    )

                # Exclude specific documents
                if excluded_docs:
                    filter_conditions.append(
                        rest.FieldCondition(
                            key="file_name",
                            match=rest.MatchAny(any=list(excluded_docs)),
                            must_not=True
                        )
                    )

                filter_obj = rest.Filter(must=filter_conditions)
                logger.info(
                    f"Created document filter with included_docs={included_docs}, excluded_docs={excluded_docs}")

            # Search in Qdrant with optional filtering
            try:
                # First try the method that works with newer versions of Qdrant client
                search_result = client.search(
                    collection_name=self.query_engine.qdrant_collection,
                    query_vector=vector,
                    limit=top_k,
                    with_payload=True,
                    filter=filter_obj
                )
            except (AssertionError, TypeError) as e:
                if "Unknown arguments" in str(e) and "filter" in str(e):
                    logger.info("Using alternative search method for older Qdrant client version")
                    # Try with query_filter instead (newer versions of Qdrant)
                    try:
                        search_result = client.search(
                            collection_name=self.query_engine.qdrant_collection,
                            query_vector=vector,
                            limit=top_k,
                            with_payload=True,
                            query_filter=filter_obj
                        )
                    except (AssertionError, TypeError):
                        # If that fails too, use the fallback method
                        # The compatible version using scroll with filter
                        logger.info("Using scroll_with_filter_compatible as fallback")
                        # We have to modify the approach since scroll has a slightly different behavior
                        scroll_result, _ = scroll_with_filter_compatible(
                            client=client,
                            collection_name=self.query_engine.qdrant_collection,
                            limit=top_k,
                            with_payload=True,
                            with_vectors=False,
                            scroll_filter=filter_obj  # This uses the utility function that handles compatibility
                        )
                        # Convert the scroll result to the expected format
                        search_result = scroll_result
                else:
                    # If it's a different type of error, re-raise it
                    raise

            # Format results
            results = []
            for point in search_result:
                text = point.payload.get('text', '')
                original_text = point.payload.get('original_text', text)
                metadata = {k: v for k, v in point.payload.items()
                            if k not in ['text', 'original_text']}
                chunk_id = metadata.get('chunk_id', point.id)
                metadata['chunk_id'] = chunk_id

                results.append({
                    'id': point.id,
                    'score': float(point.score),  # Similarity score
                    'text': text,
                    'original_text': original_text,
                    'metadata': metadata
                })

            elapsed_time = time.time() - start_time
            logger.info(f"Retrieved {len(results)} chunks in {elapsed_time:.2f}s")

            return results

        except Exception as e:
            logger.error(f"Error filtering by topic: {e}", exc_info=True)
            return []

    def export_results_to_csv(self, results: List[Dict[str, Any]], file_path: str) -> bool:
        """
        Export filtered results to a CSV file.

        Args:
            results: List of filtered chunks
            file_path: Path to save the CSV file

        Returns:
            bool: Success status
        """
        try:
            import pandas as pd

            # Create a list of rows for the DataFrame
            rows = []
            for result in results:
                row = {
                    'score': result['score'],
                    'document': result['metadata'].get('file_name', 'Unknown'),
                    'page': result['metadata'].get('page_num', 'N/A'),
                    'chunk_id': result['metadata'].get('chunk_id', result['id']),
                    'text': result.get('original_text', result.get('text', ''))
                }
                rows.append(row)

            # Create DataFrame
            df = pd.DataFrame(rows)

            # Export to CSV
            df.to_csv(file_path, index=False)
            logger.info(f"Exported {len(results)} results to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting results to CSV: {e}")
            return False