"""
Qdrant utilities for the Anti-Corruption RAG System.
Provides compatibility functions for different versions of the Qdrant client.
"""
from typing import Optional, Dict, Any, List, Union, Tuple
import logging
import inspect

logger = logging.getLogger(__name__)

# Log Qdrant client version at module import time
try:
    import qdrant_client
    QDRANT_VERSION = getattr(qdrant_client, '__version__', 'unknown')
    logger.info(f"Qdrant client version: {QDRANT_VERSION}")
except ImportError:
    logger.warning("Qdrant client not installed")
    QDRANT_VERSION = "unknown"
except Exception as e:
    logger.error(f"Error importing Qdrant client: {e}")
    QDRANT_VERSION = "unknown"


def scroll_with_filter_compatible(
        client,
        collection_name: str,
        limit: int = 10,
        with_payload: bool = True,
        with_vectors: bool = False,
        filter_obj: Optional[Dict[str, Any]] = None,
        scroll_filter=None  # Add this parameter to accept it properly
) -> Any:
    """
    Call scroll with filter in a way that's compatible with different Qdrant versions.

    Args:
        client: QdrantClient instance
        collection_name: Name of the collection
        limit: Maximum number of points to return
        with_payload: Whether to include payload
        with_vectors: Whether to include vectors
        filter_obj: Filter conditions (deprecated, use scroll_filter)
        scroll_filter: Filter conditions (new parameter name)

    Returns:
        Qdrant scroll results (format depends on Qdrant version)
    """
    # First check if the collection exists
    try:
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        if collection_name not in collection_names:
            logger.warning(f"Collection {collection_name} does not exist")
            return ([], None)  # Return empty result with correct tuple format
    except Exception as e:
        logger.error(f"Error checking collections: {e}")
        return ([], None)  # Return empty result with correct tuple format

    # Use scroll_filter if provided, otherwise use filter_obj
    actual_filter = scroll_filter if scroll_filter is not None else filter_obj

    # Directly check which parameters are supported by inspecting the function
    scroll_params = None
    try:
        # Get the function signature to see which parameters are available
        signature = inspect.signature(client.scroll)
        param_names = list(signature.parameters.keys())
        logger.info(f"Available scroll parameters: {param_names}")

        # For version 1.13.x, it should use 'scroll_filter' parameter
        if any(QDRANT_VERSION.startswith(v) for v in ['1.1', '1.2', '1.3']):
            logger.info(f"Using scroll_filter for Qdrant {QDRANT_VERSION}")
            scroll_params = {
                'collection_name': collection_name,
                'limit': limit,
                'with_payload': with_payload,
                'with_vectors': with_vectors
            }
            if actual_filter:
                scroll_params['scroll_filter'] = actual_filter
        elif 'scroll_filter' in param_names:
            logger.info("Using scroll_filter based on inspection")
            scroll_params = {
                'collection_name': collection_name,
                'limit': limit,
                'with_payload': with_payload,
                'with_vectors': with_vectors
            }
            if actual_filter:
                scroll_params['scroll_filter'] = actual_filter
        elif 'filter' in param_names:
            logger.info("Using filter based on inspection")
            scroll_params = {
                'collection_name': collection_name,
                'limit': limit,
                'with_payload': with_payload,
                'with_vectors': with_vectors
            }
            if actual_filter:
                scroll_params['filter'] = actual_filter
        else:
            # Default case, no filter
            logger.warning("No filter parameter found, using scroll without filter")
            scroll_params = {
                'collection_name': collection_name,
                'limit': limit,
                'with_payload': with_payload,
                'with_vectors': with_vectors
            }
    except Exception as e:
        logger.warning(f"Error inspecting scroll parameters: {e}, falling back to no filter")
        scroll_params = {
            'collection_name': collection_name,
            'limit': limit,
            'with_payload': with_payload,
            'with_vectors': with_vectors
        }

    # Execute the scroll call with the determined parameters
    try:
        logger.info(f"Calling scroll with parameters: {scroll_params}")
        result = client.scroll(**scroll_params)
        # Log success
        if hasattr(result, '__len__') and len(result) > 0:
            if isinstance(result, tuple) and len(result) > 0:
                logger.info(f"Scroll returned {len(result[0])} points")
            else:
                logger.info(f"Scroll returned {len(result)} points")
        return result
    except Exception as e:
        logger.error(f"Error in scroll with parameters {scroll_params}: {e}")
        # If we're here, all attempts failed - try one last time without any filter
        try:
            logger.warning("Trying one last time with minimal parameters")
            basic_params = {
                'collection_name': collection_name,
                'limit': limit,
                'with_payload': with_payload,
                'with_vectors': with_vectors
            }
            return client.scroll(**basic_params)
        except Exception as final_e:
            logger.error(f"Final scroll attempt failed: {final_e}")
            return ([], None)  # Return empty result with correct tuple format