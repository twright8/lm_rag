# app_processing.py
import streamlit as st
import pandas as pd
import gc
import torch
import traceback
import time

# Import necessary functions/variables from other modules
from app_setup import ROOT_DIR, CONFIG, logger, APHRODITE_SERVICE_AVAILABLE, get_service, get_or_create_query_engine
from app_chat import start_aphrodite_service # To start service if needed for processing
from src.utils.resource_monitor import log_memory_usage

# Import core processing modules
try:
    from src.core.document_processing.document_loader import DocumentLoader
    from src.core.document_processing.document_chunker import DocumentChunker
    from src.core.extraction.entity_extractor import EntityExtractor
    from src.core.indexing.document_indexer import DocumentIndexer
except ImportError as e:
     logger.error(f"Failed to import core processing modules: {e}", exc_info=True)
     # We can't proceed without these, maybe raise an error or handle gracefully
     st.error(f"Core processing components failed to load: {e}. Processing is disabled.")
     # Define dummy functions or raise to prevent NameErrors later if needed
     DocumentLoader = None
     DocumentChunker = None
     EntityExtractor = None
     DocumentIndexer = None


# --- Document Processing Pipeline ---
def process_documents_with_spreadsheet_options(uploaded_files, selected_llm_name, vl_pages, vl_process_all, spreadsheet_options):
    """
    Enhanced process_documents function that handles spreadsheet options.
    Orchestrates loading, chunking, extraction, and indexing.
    Updates QueryEngine LLM and relevant session state.

    Args:
        uploaded_files: List of Streamlit UploadedFile objects.
        selected_llm_name: Model name selected by user for processing AND querying.
        vl_pages: List of specific page numbers for visual processing.
        vl_process_all: Boolean indicating if all PDF pages should be visually processed.
        spreadsheet_options: Dictionary mapping spreadsheet filenames to selected columns.
    """
    # Check if core components loaded
    if not all([DocumentLoader, DocumentChunker, EntityExtractor, DocumentIndexer]):
        st.error("Core processing components are missing. Cannot process documents.")
        st.session_state.processing = False
        return

    try:
        # Ensure Aphrodite service is running
        if not APHRODITE_SERVICE_AVAILABLE:
            st.error("Aphrodite service module not available. Cannot process.")
            st.session_state.processing = False
            return

        logger.info(f"Starting document processing. Files: {[f.name for f in uploaded_files]}. LLM: {selected_llm_name}")
        log_memory_usage(logger, "Memory usage at start of processing")

        service = get_service()
        if not service.is_running():
            logger.info("Starting Aphrodite service for document processing")
            if not start_aphrodite_service(): # Function from app_chat
                st.session_state.processing_status = "Failed to start LLM service. Processing aborted."
                st.session_state.processing = False
                return
            # Give service a moment to start fully
            time.sleep(5)
            if not service.is_running(): # Double check
                 st.session_state.processing_status = "LLM service failed to stay running. Processing aborted."
                 st.session_state.processing = False
                 return

        # --- Document Loading ---
        st.session_state.processing_status = "Loading documents..."
        st.session_state.processing_progress = 0.10
        document_loader = DocumentLoader()
        documents = []
        temp_dir = ROOT_DIR / "temp"
        temp_dir.mkdir(exist_ok=True) # Ensure temp dir exists

        for i, file in enumerate(uploaded_files):
            file_path = temp_dir / file.name
            try:
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())

                # Handle spreadsheet options
                is_spreadsheet = file.name.lower().endswith(('.csv', '.xlsx', '.xls'))
                if is_spreadsheet and file.name in spreadsheet_options:
                    logger.info(f"Loading spreadsheet {file.name} with column options: {spreadsheet_options.get(file.name)}")
                    doc = document_loader.load_document_with_options(
                        file_path,
                        {
                            'selected_columns': spreadsheet_options.get(file.name, []),
                            'separator': "  |  ", # Fixed separator
                            'include_column_names': True # Always include column names
                        }
                    )
                else:
                    logger.info(f"Loading regular document: {file.name}")
                    doc = document_loader.load_document(file_path)

                if doc:
                    documents.append(doc)
                    content_count = len(doc.get('content', []))
                    logger.info(f"Document loaded: {file.name}, Content items: {content_count}")
                else:
                    logger.warning(f"Failed to load document from file: {file.name}")

            except Exception as e:
                logger.error(f"Error loading document {file.name}: {e}", exc_info=True)
                st.warning(f"Could not load {file.name}, skipping.")
            finally:
                 # Clean up temp file immediately after loading attempt
                 if file_path.exists():
                      try:
                           file_path.unlink()
                      except OSError as e:
                           logger.warning(f"Could not delete temp file {file_path}: {e}")


            progress = 0.10 + (0.10 * (i + 1) / len(uploaded_files))
            st.session_state.processing_progress = progress
            st.session_state.processing_status = f"Loaded document {i + 1}/{len(uploaded_files)}: {file.name}"

        # Shutdown loader and free memory
        logger.info("Shutting down document loader (Docling)...")
        if 'document_loader' in locals() and document_loader:
            document_loader.shutdown()
            del document_loader
            gc.collect()
            logger.info("Document loader shut down complete.")
            log_memory_usage(logger, "Memory usage after document loading")
        else:
             logger.warning("Document loader instance not found for shutdown.")

        if not documents:
            st.error("No documents were successfully loaded.")
            st.session_state.processing = False
            return

        # --- Chunking ---
        st.session_state.processing_status = "Chunking documents..."
        st.session_state.processing_progress = 0.25
        document_chunker = DocumentChunker()
        logger.info("Loading chunking model...")
        document_chunker.load_model()
        all_chunks = []

        for i, doc in enumerate(documents):
            doc_name = doc.get('file_name', 'Unknown')
            try:
                logger.info(f"Chunking document {i+1}/{len(documents)}: {doc_name}")
                doc_chunks = document_chunker.chunk_document(doc)
                chunk_count = len(doc_chunks)
                logger.info(f"Created {chunk_count} chunks for {doc_name}")
                all_chunks.extend(doc_chunks)

                progress = 0.25 + (0.15 * (i + 1) / len(documents))
                st.session_state.processing_progress = progress
                st.session_state.processing_status = f"Chunked document {i + 1}/{len(documents)}: {doc_name}"
            except Exception as e:
                logger.error(f"Error chunking document {doc_name}: {e}", exc_info=True)
                st.warning(f"Could not chunk {doc_name}, skipping.")

        logger.info(f"Shutting down chunking model...")
        document_chunker.shutdown()
        del document_chunker
        gc.collect()
        log_memory_usage(logger, "Memory usage after chunking")

        if not all_chunks:
            st.error("No chunks were generated from the loaded documents.")
            st.session_state.processing = False
            return

        logger.info(f"Total chunks generated: {len(all_chunks)}")

        # --- Entity Extraction ---
        st.session_state.processing_status = "Preparing entity extraction..."
        st.session_state.processing_progress = 0.40
        logger.info(f"Creating entity extractor with model {selected_llm_name}")
        entity_extractor = EntityExtractor(model_name=selected_llm_name, debug=True)

        st.session_state.processing_status = "Extracting entities..."
        st.session_state.processing_progress = 0.50

        # Determine visual chunks (logic remains the same)
        visual_chunk_ids = []
        if vl_pages or vl_process_all:
            for chunk in all_chunks:
                page_num = chunk.get('page_num')
                is_pdf = chunk.get('file_name', '').lower().endswith('.pdf')
                if is_pdf and page_num is not None and (vl_process_all or page_num in vl_pages):
                    visual_chunk_ids.append(chunk.get('chunk_id'))

        logger.info(f"Starting entity extraction on {len(all_chunks)} chunks ({len(visual_chunk_ids)} visual chunks)")

        # Process chunks (this ensures the selected_llm_name is loaded in the service)
        # This might take a while and involves LLM calls
        entity_extractor.process_chunks(all_chunks, visual_chunk_ids) # This handles loading model in service

        st.session_state.processing_progress = 0.75
        st.session_state.processing_status = "Saving extraction results..."
        entity_extractor.save_results()
        modified_chunks = entity_extractor.get_modified_chunks()
        logger.info(f"Entity extraction complete. Modified chunks: {len(modified_chunks)}")
        del entity_extractor # Free up extractor resources
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        log_memory_usage(logger, "Memory usage after entity extraction")


        # --- IMPORTANT: Update QueryEngine to use the same model ---
        try:
            query_engine = get_or_create_query_engine() # Function from app_setup
            if query_engine:
                query_engine.llm_model_name = selected_llm_name # Set the model for subsequent queries
                st.session_state.selected_llm_model_name = selected_llm_name # Update session state backup
                logger.info(f"QueryEngine LLM model updated to: {selected_llm_name}")
            else:
                 logger.error("Query engine not available, cannot update LLM model name.")
                 st.warning("Query engine unavailable. Queries might fail or use incorrect model.")
        except Exception as e:
            logger.error(f"Failed to update QueryEngine model name: {e}", exc_info=True)
            st.warning("Could not set the query model. Queries might use the default chat model.")

        # Update process info state to reflect the model loaded during extraction
        st.session_state.llm_model_loaded = True
        try:
            status = service.get_status()
            process_info = {
                "pid": service.process.pid if service.process else None,
                "model_name": status.get("current_model") # Should match selected_llm_name
            }
            if process_info["pid"] and process_info["model_name"]:
                st.session_state.aphrodite_process_info = process_info
                # Verify model consistency
                if process_info["model_name"] != selected_llm_name:
                    logger.warning(f"Model mismatch after extraction! Expected {selected_llm_name}, Service reports {process_info['model_name']}")
            else:
                logger.warning("Failed to update process info after extraction.")
        except Exception as e:
            logger.warning(f"Error updating Aphrodite process info after extraction: {e}")

        # --- Indexing ---
        st.session_state.processing_status = "Indexing documents..."
        st.session_state.processing_progress = 0.80
        document_indexer = DocumentIndexer()
        logger.info("Loading indexing model...")
        document_indexer.load_model()

        # Check if collection exists and clear if necessary (using query_engine)
        if query_engine:
            try:
                collection_info = query_engine.get_collection_info()
                if collection_info.get("exists", False):
                    # Get expected dimension from the indexer's model
                    expected_dim = document_indexer.get_embedding_dimension()
                    vector_size = collection_info.get("vector_size", 0)
                    if expected_dim is not None and vector_size != 0 and vector_size != expected_dim:
                        logger.warning(f"Clearing collection due to dimension mismatch (DB: {vector_size} vs Model: {expected_dim}).")
                        query_engine.clear_collection() # This clears vector DB, BM25, extracted files
                        # Re-check info after clearing
                        st.session_state.collection_info = query_engine.get_collection_info()
                    elif expected_dim is None:
                         logger.warning("Could not determine expected embedding dimension from indexer.")
            except Exception as e:
                logger.warning(f"Error checking/clearing collection: {e}")
        else:
             logger.error("Query engine not available, cannot check/clear collection before indexing.")


        # Index documents
        logger.info(f"Indexing {len(modified_chunks)} modified chunks")
        document_indexer.index_documents(modified_chunks) # This handles vector DB and BM25

        logger.info("Shutting down indexing model...")
        document_indexer.shutdown()
        del document_indexer
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        log_memory_usage(logger, "Memory usage after indexing")

        st.session_state.processing_progress = 0.95
        st.session_state.processing_status = "Indexing complete."

        # Complete - 100% progress
        st.session_state.processing_status = "Processing completed successfully!"
        st.session_state.processing_progress = 1.0

        # Update collection info one last time
        if query_engine:
            st.session_state.collection_info = query_engine.get_collection_info()
        logger.info("Document processing pipeline completed successfully!")
        log_memory_usage(logger, "Memory usage at end of processing")

    except Exception as e:
        logger.error(f"Fatal error during document processing pipeline: {e}", exc_info=True)
        st.session_state.processing_status = f"Error: {str(e)}"
        st.error(f"An unexpected error occurred during processing: {e}")
        log_memory_usage(logger, "Memory usage after processing error")
    finally:
        # Ensure processing flag is reset
        st.session_state.processing = False
        # Clean up any remaining temp files (optional, as they should be deleted after loading)
        temp_dir = ROOT_DIR / "temp"
        if temp_dir.exists():
             for item in temp_dir.iterdir():
                  try:
                       if item.is_file(): item.unlink()
                  except OSError as e:
                       logger.warning(f"Could not clean up temp file {item}: {e}")


# --- Data Clearing ---
def clear_all_data():
    """
    Clear all data from the system (vector DB, BM25, extracted files).
    Uses the QueryEngine's clear method.
    """
    try:
        logger.info("Clearing all indexed and extracted data...")
        query_engine = get_or_create_query_engine() # Function from app_setup

        if not query_engine:
             st.error("Query engine not available. Cannot clear data.")
             logger.error("Data clearing failed: Query engine instance is None.")
             return False # Indicate failure

        # The clear_collection method in QueryEngine should handle
        # deleting Qdrant collection, BM25 index files, and extracted data files.
        success = query_engine.clear_collection()

        if success:
             logger.info("Data clearing successful via QueryEngine.")
             st.success("Vector DB, BM25 index, and extracted data files cleared.")
        else:
             logger.error("Data clearing process reported errors via QueryEngine.")
             st.error("An error occurred during data clearing. Check application logs.")

        # Reset relevant session state that depends on data
        st.session_state.chat_history = [] # Legacy? Clear anyway.
        st.session_state.current_conversation_id = None # Clear active conversation
        st.session_state.active_conversation_data = None
        st.session_state.ui_chat_display = []
        # Clear any cached results that depend on data
        keys_to_clear = ["cluster_map_result", "topic_filter_results", "info_extraction_results", "classification_results", "node_similarity_results"]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        logger.info("Cleared relevant session state caches after data deletion.")


        # Refresh collection info state immediately
        try:
             # Re-getting the engine should update the info in session state
             get_or_create_query_engine()
             logger.info(f"Collection info after clearing: {st.session_state.get('collection_info')}")
        except Exception as e:
            logger.error(f"Error getting collection info after clear: {e}")
            # Manually set state if update fails
            st.session_state.collection_info = {"exists": False, "points_count": 0, "error": "Failed to refresh after clear"}

        # Note: Aphrodite service remains running, model state is untouched by data clear.
        if st.session_state.get("aphrodite_service_running"):
            logger.info("Aphrodite service remains running after data clear.")

        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        log_memory_usage(logger, "Memory usage after clearing data")
        return success

    except Exception as e:
        st.error(f"An unexpected error occurred while clearing data: {e}")
        logger.error(f"Clear data failed: {traceback.format_exc()}")
        log_memory_usage(logger, "Memory usage after data clearing error")
        return False # Indicate failure