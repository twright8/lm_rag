import sys
from pathlib import Path
import streamlit as st


# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))


from src.utils.resource_monitor import get_gpu_info, log_memory_usage
from src.utils.logger import setup_logger
# Import get_service carefully
import sys
from pathlib import Path
import yaml
import pandas as pd
import time
import json
import gc
import torch
import traceback
from typing import List, Dict, Any, Union, Optional
# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

# Import core modules AFTER setting path
from src.core.document_processing.document_loader import DocumentLoader
from src.core.document_processing.document_chunker import DocumentChunker
from src.core.extraction.entity_extractor import EntityExtractor
from src.core.indexing.document_indexer import DocumentIndexer
from src.core.query_system.query_engine import QueryEngine
# Add this new import for cluster_map functionality
# We'll conditionally import the visualization module when needed
# to avoid errors if dependencies are missing
try:
    from src.utils.aphrodite_service import get_service, AphroditeService
    APHRODITE_SERVICE_AVAILABLE = True
except ImportError as e:
    APHRODITE_SERVICE_AVAILABLE = False
    get_service = None # Placeholder
    AphroditeService = None # Placeholder
    setup_logger(__name__).error(f"Failed to import AphroditeService: {e}")


# Initialize logger
logger = setup_logger(__name__)

# Load configuration
CONFIG_PATH = ROOT_DIR / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

# Create data directories if they don't exist
EXTRACTED_DATA_PATH = ROOT_DIR / CONFIG["storage"]["extracted_data_path"]
BM25_DIR = ROOT_DIR / CONFIG["storage"]["bm25_index_path"]
EXTRACTED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
BM25_DIR.parent.mkdir(parents=True, exist_ok=True)
(ROOT_DIR / "temp").mkdir(parents=True, exist_ok=True) # Ensure temp dir exists

def initialize_app():
    """
    Initialize the Streamlit application state.
    Checks for existing Aphrodite process belonging to the current session
    and avoids terminating it during script reruns.
    """
    # Set page config early
    st.set_page_config(
        page_title="Tomni AI - NLP toolset",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    aphrodite_service_restored = False
    llm_model_restored = False # Renamed from chat_model_loaded
    service_instance = None

    if APHRODITE_SERVICE_AVAILABLE:
        try:
            service_instance = get_service()
        except Exception as e:
            logger.error(f"Failed to get AphroditeService instance during init: {e}")
            service_instance = None

        # --- Check for existing Aphrodite process tied to this session ---
        if "aphrodite_process_info" in st.session_state and st.session_state.aphrodite_process_info and service_instance:
            pid = st.session_state.aphrodite_process_info.get("pid")
            if pid:
                logger.info(f"Session state contains Aphrodite process info (PID: {pid}). Checking status...")
                try:
                    # Use the service's method to check its own process
                    if service_instance.is_running() and service_instance.process.pid == pid:
                        # Process from current session IS alive - KEEP IT
                        logger.info(f"Active Aphrodite process (PID: {pid}) confirmed running. Restoring session state.")
                        aphrodite_service_restored = True
                        # Check if model was loaded based on stored info
                        llm_model_restored = bool(st.session_state.aphrodite_process_info.get("model_name"))
                        if llm_model_restored:
                             logger.info(f"Model '{st.session_state.aphrodite_process_info.get('model_name')}' marked as restored.")
                        # Sync service state with Streamlit state
                        service_instance.current_model_info["name"] = st.session_state.aphrodite_process_info.get("model_name")

                    else:
                        # Stored PID but process is dead or mismatched - clear the stale state
                        logger.info(f"Stale or mismatched Aphrodite process info found (Expected PID: {pid}, Running: {service_instance.is_running()}, Actual PID: {service_instance.process.pid if service_instance.process else 'None'}). Clearing session state.")
                        st.session_state.aphrodite_process_info = None
                        aphrodite_service_restored = False
                        llm_model_restored = False
                        # Attempt to shutdown cleanly if the instance thinks it's running but PID mismatch
                        if service_instance.is_running():
                             service_instance.shutdown()
                except Exception as e:
                    logger.error(f"Error checking status of Aphrodite process PID {pid}: {e}. Clearing session state.", exc_info=True)
                    st.session_state.aphrodite_process_info = None
                    aphrodite_service_restored = False
                    llm_model_restored = False
            else:
                # Info dictionary exists but no PID - invalid state, clear it
                logger.warning("Aphrodite process info found in session state but no PID. Clearing state.")
                st.session_state.aphrodite_process_info = None
                aphrodite_service_restored = False
                llm_model_restored = False
        else:
             # No info stored for this session, assume not running
             if "aphrodite_process_info" not in st.session_state: # Only log if it's truly the first run
                 logger.info("No Aphrodite process info found in session state.")
             # Ensure state reflects not running if no info was present
             st.session_state.aphrodite_process_info = None
             aphrodite_service_restored = False
             llm_model_restored = False


    # --- Initialize standard session state variables using setdefault ---
    # This ensures they exist but doesn't overwrite if they were already set
    st.session_state.setdefault("processing", False)
    st.session_state.setdefault("processing_status", "")
    st.session_state.setdefault("processing_progress", 0.0)
    st.session_state.setdefault("documents", [])
    st.session_state.setdefault("query_engine", None)
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("selected_llm_model_name", None) # To store the model chosen during processing

    # Set/Reset Aphrodite states based on checks above or default if service unavailable
    st.session_state.setdefault("aphrodite_service_running", aphrodite_service_restored)
    st.session_state.setdefault("llm_model_loaded", llm_model_restored) # Renamed state variable
    # Ensure aphrodite_process_info is initialized if it wasn't already
    st.session_state.setdefault("aphrodite_process_info", None)

    # Initialize Query Engine and update collection info
    get_or_create_query_engine() # This also updates collection info

    # Log final initial state
    logger.debug(f"Initialized session state: service_running={st.session_state.aphrodite_service_running}, llm_model_loaded={st.session_state.llm_model_loaded}")

def apply_custom_styling():
    """
    Apply custom styling to the Streamlit app. (No changes needed here)
    """
    # Set custom theme colors from config
    theme_color = CONFIG["ui"]["theme_color"]
    secondary_color = CONFIG["ui"]["secondary_color"]
    accent_color = CONFIG["ui"]["accent_color"]

    # Apply custom CSS
    st.markdown(f"""
    <style>
    .main {{
        background-color: #FFFFFF;
    }}
    iframe{{
    min-height: 700px}}
    .stApp {{
        max-width: 1900px;
        margin: 0 auto;
    }}
    .sidebar .sidebar-content {{
        background-color: {theme_color};
    }}
    h1, h2, h3 {{
        color: {theme_color};
    }}
    .stButton>button {{
        background-color: {theme_color};
        color: white;
    }}
    .stButton>button:hover {{
        background-color: {secondary_color};
        color: white;
    }}
    .status-box {{
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }}
    .info-box {{
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }}
    .highlight {{
        background-color: #ffffcc;
        padding: 3px;
        border-radius: 3px;
    }}
    /* Node styles for relationship graph */
    .node-PERSON {{ background-color: #5DA5DA; border-radius: 50%; padding: 10px; color: white; text-align: center; }}
    .node-ORGANIZATION {{ background-color: #FAA43A; border-radius: 50%; padding: 10px; color: white; text-align: center; }}
    .node-GOVERNMENT_BODY {{ background-color: #60BD68; border-radius: 50%; padding: 10px; color: white; text-align: center; }}
    .node-COMMERCIAL_COMPANY {{ background-color: #F17CB0; border-radius: 50%; padding: 10px; color: white; text-align: center; }}
    .node-LOCATION {{ background-color: #B2912F; border-radius: 50%; padding: 10px; color: white; text-align: center; }}
    .node-POSITION {{ background-color: #B276B2; border-radius: 50%; padding: 10px; color: white; text-align: center; }}
    .node-MONEY {{ background-color: #DECF3F; border-radius: 50%; padding: 10px; color: black; text-align: center; }}
    .node-ASSET {{ background-color: #F15854; border-radius: 50%; padding: 10px; color: white; text-align: center; }}
    .node-EVENT {{ background-color: #4D4D4D; border-radius: 50%; padding: 10px; color: white; text-align: center; }}
    .node-Unknown {{ background-color: #999999; border-radius: 50%; padding: 10px; color: white; text-align: center; }}
    </style>
    """, unsafe_allow_html=True)

def render_header():
    """
    Render the application header. (No changes needed here)
    """
    st.title("🔍 Tomni AI - Anti-corruption toolset")
    st.markdown("""
    A semantic search and analysis system for anti-corruption investigations.
    Upload documents, extract entities and relationships, and query using natural language.
    """)

    # Add a horizontal rule
    st.markdown("---")


def render_sidebar():
    """
    Render the sidebar with navigation and system status.
    """
    with st.sidebar:
        st.title("Navigation")

        # Navigation links
        # Navigation links
        # --- Add to app.py in the list of pages in render_sidebar function ---
        page = st.radio(
            "Select Page",
            options=["Upload & Process", "Explore Data", "Query System", "Topic Filter", "Information Extraction",
                     "Cluster Map", "Document Classification", "Settings"],
            index=["Upload & Process", "Explore Data", "Query System", "Topic Filter", "Information Extraction",
                   "Cluster Map", "Document Classification", "Settings"].index(
                CONFIG["ui"]["default_page"] if CONFIG["ui"]["default_page"] in ["Upload & Process", "Explore Data",
                                                                                 "Query System", "Topic Filter",
                                                                                 "Information Extraction",
                                                                                 "Cluster Map",
                                                                                 "Document Classification",
                                                                                 "Settings"] else "Upload & Process")
        )

        # --- Add to main() function's page selection logic ---

        st.markdown("---")

        # System status
        st.subheader("System Status")

        # Resource monitoring section
        gpu_info = get_gpu_info()
        if gpu_info:
            vram_used = gpu_info.get("memory_used", 0)
            vram_total = gpu_info.get("memory_total", 1)
            if vram_total > 0:
                vram_percent = (vram_used / vram_total) * 100
                st.progress(vram_percent / 100, f"VRAM: {vram_used:.1f}GB / {vram_total:.1f}GB")
            else:
                st.info("GPU detected, but total memory reported as 0.")
        else:
            st.info("No GPU detected or CUDA not available.")


        # Collection info from Qdrant
        if "collection_info" in st.session_state and st.session_state.collection_info:
            info = st.session_state.collection_info
            count = info.get('points_count', 0)
            if info.get('exists', False):
                 st.info(f"Documents indexed: {count}")
            else:
                 st.warning("Vector DB collection not found.")

        # LLM status section
        st.subheader("LLM Status")

        # Display current status of the Aphrodite service
        # Use service's is_running() for definitive status
        service = get_service() if APHRODITE_SERVICE_AVAILABLE else None
        is_service_actually_running = service.is_running() if service else False
        st.session_state.aphrodite_service_running = is_service_actually_running # Sync state

        if is_service_actually_running:
            st.success("🟢 LLM service running")
            status = service.get_status(timeout=5) # Get fresh status
            current_model = status.get("current_model")
            model_is_loaded = status.get("model_loaded", False)
            st.session_state.llm_model_loaded = model_is_loaded # Sync state

            # Show model status
            if model_is_loaded:
                st.info(f"✓ Model loaded: {current_model}")
            else:
                st.warning("⚠️ LLM service running but no model loaded")

            # Add terminate button
            if st.button("Stop LLM Service", type="secondary", key="stop_llm"):
                with st.spinner("Stopping LLM service..."):
                    if terminate_aphrodite_service():
                         st.success("LLM service terminated")
                    else:
                         st.error("Failed to terminate LLM service.")
                st.rerun()
        else:
            st.warning("🔴 LLM service not running")
            st.session_state.llm_model_loaded = False # Ensure state reflects this

            # Add start button
            if st.button("Start LLM Service", type="primary", key="start_llm"):
                with st.spinner("Starting LLM service..."):
                    if start_aphrodite_service():
                        st.success("LLM service started")
                    else:
                         st.error("Failed to start LLM service.")
                st.rerun()

        # Actions section
        st.markdown("---")
        st.subheader("Actions")

        if st.button("Clear All Data", key="clear_data"):
            with st.spinner("Clearing all data..."):
                clear_all_data()
            st.success("All data cleared successfully!")
            st.rerun()

        # Add credits
        st.markdown("---")
        st.caption("© 2024 Anti-Corruption RAG System") # Year updated

    return page


def terminate_aphrodite_service():
    """
    Terminate the Aphrodite service process.
    """
    if not APHRODITE_SERVICE_AVAILABLE:
         st.error("Aphrodite service module not available.")
         return False
    try:
        logger.info("User requested Aphrodite service termination")
        service = get_service()
        success = service.shutdown()

        if success:
            logger.info("Aphrodite service successfully terminated")
        else:
            logger.warning("Aphrodite service shutdown may not have been fully successful")

        # Update states regardless of success to avoid stuck state
        st.session_state.llm_model_loaded = False # Renamed state
        st.session_state.aphrodite_service_running = False
        st.session_state.aphrodite_process_info = None

        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return success
    except Exception as e:
        logger.error(f"Error terminating Aphrodite service: {e}")
        import traceback
        logger.error(traceback.format_exc())

        # Update states even on error
        st.session_state.llm_model_loaded = False
        st.session_state.aphrodite_service_running = False
        st.session_state.aphrodite_process_info = None
        return False

def start_aphrodite_service():
    """
    Start the Aphrodite service process.
    """
    if not APHRODITE_SERVICE_AVAILABLE:
         st.error("Aphrodite service module not available.")
         return False
    try:
        service = get_service()

        # Start the service without loading a model yet
        if service.start():
            logger.info("Aphrodite service started successfully")
            st.session_state.aphrodite_service_running = True

            # Save process info (PID only needed now)
            process_info = {"pid": service.process.pid if service.process else None}
            if process_info["pid"]:
                st.session_state.aphrodite_process_info = process_info
                logger.info(f"Saved Aphrodite process info: PID={process_info.get('pid')}")
            else:
                 st.session_state.aphrodite_process_info = None
                 logger.warning("Aphrodite service started but failed to get PID.")


            # Don't set llm_model_loaded here, it happens on demand
            st.session_state.llm_model_loaded = False
            return True
        else:
            logger.error("Failed to start Aphrodite service")
            st.session_state.aphrodite_service_running = False
            st.session_state.llm_model_loaded = False
            st.session_state.aphrodite_process_info = None
            return False
    except Exception as e:
        logger.error(f"Error starting Aphrodite service: {e}")
        st.session_state.aphrodite_service_running = False
        st.session_state.llm_model_loaded = False
        st.session_state.aphrodite_process_info = None
        return False

def render_upload_page():
    """
    Render the upload and processing page.
    (Reintroducing model selection)
    """
    st.header("📄 Upload & Process Documents")

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload documents (PDF, DOCX, TXT, CSV, XLSX)",
        type=["pdf", "docx", "txt", "csv", "xlsx"],
        accept_multiple_files=True
    )

    # Model selection (Reintroduced)
    st.subheader("Processing Options")

    # Define model choices based on config keys
    extraction_model_options = {
        "Small Text (Faster)": CONFIG["models"]["extraction_models"]["text_small"],
        "Standard Text": CONFIG["models"]["extraction_models"]["text_standard"],
        # Add Visual Language if needed and supported by the backend logic
        # "Visual Language (PDF Images)": CONFIG["models"]["extraction_models"]["visual_language"],
    }
    # Use the previously selected model as default if available, else default to small
    default_model_display_name = "Small Text (Faster)" # Default choice
    if st.session_state.selected_llm_model_name:
        # Find the display name corresponding to the stored model name
        for name, model_val in extraction_model_options.items():
            if model_val == st.session_state.selected_llm_model_name:
                default_model_display_name = name
                break

    # Create map from display name -> actual model name
    model_name_map = {display: actual for display, actual in extraction_model_options.items()}

    selected_model_display_name = st.selectbox(
        "Select LLM for Processing & Querying",
        options=list(model_name_map.keys()),
        index=list(model_name_map.keys()).index(default_model_display_name) # Set default index
    )
    # Get the actual model name from the selected display name
    selected_llm_name = model_name_map[selected_model_display_name]

    # Visual processing options (keep UI but functionality isn't fully integrated in extractor)
    use_visual_processing = st.checkbox("Enable Visual Processing (Experimental for PDFs)")
    vl_page_numbers_str = ""
    if use_visual_processing:
         vl_page_numbers_str = st.text_input(
                "Specify Pages for Visual Processing",
                placeholder="e.g., 1, 3-5 (leave empty to try all PDF pages visually)"
            )


    # Processing button
    process_btn = st.button(
        "Process Documents",
        disabled=st.session_state.processing or not uploaded_files,
        type="primary"
    )

    # Show processing status
    if st.session_state.processing:
        st.markdown("### Processing Status")
        status_container = st.empty()
        progress_bar = st.progress(0)

        # Update progress bar and status message
        progress_bar.progress(st.session_state.processing_progress)
        status_container.info(st.session_state.processing_status)

    # Handle process button click
    if process_btn and uploaded_files:
        # Parse VL page numbers if provided
        vl_pages = []
        vl_process_all = False
        if use_visual_processing:
             # (Keep existing visual page parsing logic here)
            if not vl_page_numbers_str.strip():
                 vl_process_all = True
            else:
                try:
                    for part in vl_page_numbers_str.split(','):
                        part = part.strip()
                        if '-' in part:
                            start, end = map(int, part.split('-'))
                            if start > 0 and end >= start: vl_pages.extend(range(start, end + 1))
                            else: raise ValueError("Invalid page range")
                        elif part:
                            page_num = int(part)
                            if page_num > 0: vl_pages.append(page_num)
                            else: raise ValueError("Page number must be positive")
                except ValueError as e:
                    st.error(f"Invalid page numbers format: '{vl_page_numbers_str}'. Error: {e}")
                    return # Stop processing

        # Start processing
        st.session_state.processing = True
        st.session_state.processing_status = "Starting document processing..."
        st.session_state.processing_progress = 0.0

        # --- NEW: Store the selected model name ---
        st.session_state.selected_llm_model_name = selected_llm_name
        logger.info(f"Storing selected LLM for session: {selected_llm_name}")

        # Pass the selected extraction model and visual page info
        # Note: process_documents now needs to update QueryEngine
        process_documents(uploaded_files, selected_llm_name, vl_pages, vl_process_all)

        # Refresh the page to show updated status

def render_explore_page():
    """
    Render the data exploration page with the enhanced relationship graph.
    """
    st.header("🔎 Explore Data")

    tab1, tab2, tab3 = st.tabs(["Documents & Chunks", "Entities", "Relationships"])

    with tab1:
        render_document_explorer()

    with tab2:
        render_entity_explorer()

    with tab3:
        render_relationship_graph()  # This now includes all the enhanced visualizations

def render_document_explorer():
    """
    Render the document and chunk explorer. (No changes needed here)
    """
    st.subheader("Document Explorer")

    # Get documents and chunks from Qdrant
    query_engine = get_or_create_query_engine()

    # Document filter and search
    col1, col2 = st.columns([3, 1])
    with col1:
        search_text = st.text_input("Search within chunks", placeholder="Enter search terms...")
    with col2:
         # Provide unique document names for filtering
         try:
             all_chunks_for_filter = query_engine.get_chunks(limit=1000) # Get more chunks to find unique names
             doc_names = sorted(list(set(c['metadata'].get('file_name', 'Unknown') for c in all_chunks_for_filter if c['metadata'].get('file_name'))))
             doc_filter = st.selectbox("Filter by document", options=["All Documents"] + doc_names)
         except Exception as e:
              logger.error(f"Failed to get document names for filter: {e}")
              doc_filter = st.selectbox("Filter by document", options=["All Documents", "Error loading names"])

    # Get chunks from DB based on filter
    doc_filter_value = doc_filter if doc_filter != "All Documents" else None
    chunks = query_engine.get_chunks(limit=50, search_text=search_text if search_text else None, document_filter=doc_filter_value)

    if not chunks:
        if search_text or doc_filter_value:
             st.info("No documents match the current filter.")
        else:
             st.info("No documents found. Upload and process documents first.")
        return

    # Display chunks
    st.markdown(f"Displaying **{len(chunks)}** chunks:")
    for i, chunk in enumerate(chunks):
        with st.expander(f"Chunk {chunk['metadata'].get('chunk_id', chunk['id'])} - {chunk['metadata'].get('file_name', 'Unknown')}"):
            # Show original text (not the modified text with tags)
            st.markdown(chunk['text'])

            # Show metadata in a smaller font
            st.caption(f"Document: {chunk['metadata'].get('file_name', 'Unknown')} | Page: {chunk['metadata'].get('page_num', 'N/A')}")


def render_entity_explorer():
    """
    Render the entity explorer with tables of extracted entities.
    (No changes needed here)
    """
    st.subheader("Entity Explorer")

    # Load entities from file
    entities_file = EXTRACTED_DATA_PATH / "entities.json"

    if not entities_file.exists():
        st.info("No entities found. Upload and process documents first.")
        return

    try:
        with open(entities_file, "r", encoding='utf-8') as f: # Specify encoding
            entities = json.load(f)
    except Exception as e:
        st.error(f"Error loading entities: {e}")
        return

    if not entities:
         st.info("Entities file exists but is empty.")
         return

    # Display entity tables by type
    entity_types = sorted(list(set(entity.get("type", "Unknown") for entity in entities)))

    # Filter control
    selected_types = st.multiselect(
        "Filter by entity type",
        options=entity_types,
        default=entity_types
    )

    # Text search
    search_term = st.text_input("Search entities by name", placeholder="Enter search terms...")

    # Filter entities
    filtered_entities = [
        entity for entity in entities
        if entity.get("type", "Unknown") in selected_types and
        (not search_term or search_term.lower() in entity.get("name", "").lower())
    ]

    # Display as table
    if filtered_entities:
        entity_data = []
        for entity in filtered_entities:
            # Extract context for display
            context_info = entity.get("context", {})
            doc_id = context_info.get("document_id", "N/A")
            page_num = context_info.get("page_number", "N/A")
            chunk_ids = ", ".join(context_info.get("chunk_ids", []))

            entity_data.append({
                "Name": entity.get("name", ""),
                "Type": entity.get("type", "Unknown"),
                "Source Document": entity.get("source_document", "Unknown"),
                #"Description": entity.get("description", ""),
                "Page": page_num,
                "Chunk IDs": chunk_ids
            })

        st.dataframe(entity_data, use_container_width=True)
        st.caption(f"Displaying {len(filtered_entities)} of {len(entities)} total entities.")
    else:
        st.info("No entities match the current filters.")


def render_entity_connection_explorer(entities, relationships):
    """
    Render the entity connection explorer to visualize connections between two entities.

    Args:
        entities: List of entity dictionaries
        relationships: List of relationship dictionaries
    """
    import math
    import networkx as nx
    from pyvis.network import Network

    st.subheader("Entity Connection Explorer")

    # Create entity name to ID mapping for easier lookup
    entity_name_to_id = {}
    entity_id_to_name = {}
    entity_id_to_type = {}

    for entity in entities:
        entity_id = entity.get("id")
        entity_name = entity.get("name", "Unknown")
        entity_type = entity.get("type", "Unknown")

        # Store mappings
        if entity_id and entity_name:
            entity_name_to_id[entity_name] = entity_id
            entity_id_to_name[entity_id] = entity_name
            entity_id_to_type[entity_id] = entity_type

    # Get sorted list of entity names for dropdowns
    entity_names = sorted(entity_name_to_id.keys())

    # Controls
    col1, col2 = st.columns(2)

    with col1:
        # Source entity selection
        source_entity = st.selectbox(
            "Source Entity",
            options=entity_names,
            index=0 if entity_names else None,
            key="connection_source_entity"
        )

    with col2:
        # Target entity selection
        target_entity = st.selectbox(
            "Target Entity",
            options=entity_names,
            index=min(1, len(entity_names) - 1) if len(entity_names) > 1 else 0,
            key="connection_target_entity"
        )

    # Degree of separation slider
    max_degrees = 20
    degrees_of_separation = st.slider(
        "Degrees of Separation",
        min_value=1,
        max_value=max_degrees,
        value=2,
        key="connection_degrees"
    )

    # Button to generate visualization
    if st.button("Visualize Connection", key="visualize_connection_btn"):
        # Check if entities are selected and different
        if not source_entity or not target_entity:
            st.warning("Please select both source and target entities.")
            return

        if source_entity == target_entity:
            st.warning("Please select different entities for source and target.")
            return

        # Get entity IDs
        source_id = entity_name_to_id.get(source_entity)
        target_id = entity_name_to_id.get(target_entity)

        if not source_id or not target_id:
            st.error("Could not find IDs for the selected entities.")
            return

        # Create full graph for path finding
        G = nx.DiGraph()

        # Add all entities as nodes
        for entity in entities:
            entity_id = entity.get("id")
            if entity_id:
                G.add_node(
                    entity_id,
                    label=entity.get("name", "Unknown"),
                    type=entity.get("type", "Unknown")
                )

        # Add all relationships as edges
        for rel in relationships:
            source_rel_id = rel.get("source_entity_id", rel.get("from_entity_id"))
            target_rel_id = rel.get("target_entity_id", rel.get("to_entity_id"))
            rel_type = rel.get("type", rel.get("relationship_type", "RELATED_TO"))

            if source_rel_id and target_rel_id:
                G.add_edge(
                    source_rel_id,
                    target_rel_id,
                    type=rel_type
                )

        # Find all paths between source and target within degrees of separation
        # Convert to undirected for path finding (to find connections regardless of direction)
        UG = G.to_undirected()

        # Use modified BFS to find all paths within max length
        paths = []

        def find_all_paths_limited_length(graph, start, end, max_length):
            visited = {start: 0}  # node: distance from start
            queue = [(start, [start])]  # (node, path)
            all_paths = []

            while queue:
                (node, path) = queue.pop(0)

                # If we reached the target, add path to results
                if node == end:
                    all_paths.append(path)
                    continue

                # If we've reached max length, don't explore further on this path
                if visited[node] >= max_length:
                    continue

                # Explore neighbors
                for neighbor in graph.neighbors(node):
                    if neighbor not in visited or visited[neighbor] > visited[node] + 1:
                        visited[neighbor] = visited[node] + 1
                        if neighbor not in path:  # Avoid cycles
                            queue.append((neighbor, path + [neighbor]))

            return all_paths

        # Find paths
        paths = find_all_paths_limited_length(UG, source_id, target_id, degrees_of_separation)

        if not paths:
            st.warning(
                f"No connection found between {source_entity} and {target_entity} within {degrees_of_separation} degrees of separation.")
            return

        # Create a subgraph containing only the nodes and edges in the paths
        path_nodes = set()
        for path in paths:
            path_nodes.update(path)

        # Create visualization subgraph
        viz_graph = nx.DiGraph()

        # Add nodes from paths
        for node_id in path_nodes:
            if node_id in entity_id_to_name:
                viz_graph.add_node(
                    node_id,
                    label=entity_id_to_name[node_id],
                    type=entity_id_to_type.get(node_id, "Unknown"),
                    # Mark source and target nodes
                    is_source=(node_id == source_id),
                    is_target=(node_id == target_id)
                )

        # Add edges between nodes in paths (preserve direction from original graph)
        for u, v, data in G.edges(data=True):
            if u in path_nodes and v in path_nodes:
                viz_graph.add_edge(u, v, **data)

        # Create PyVis network for visualization
        net = Network(height="700px", width="100%", directed=True, notebook=True, cdn_resources='remote')

        # Node color map by type and role
        colors = {
            "PERSON": "#3B82F6",  # Blue
            "ORGANIZATION": "#10B981",  # Green
            "GOVERNMENT_BODY": "#60BD68",  # Green variant
            "COMMERCIAL_COMPANY": "#10B981",  # Green variant
            "LOCATION": "#F59E0B",  # Yellow/Orange
            "POSITION": "#8B5CF6",  # Purple
            "MONEY": "#EC4899",  # Pink
            "ASSET": "#EF4444",  # Red
            "EVENT": "#6366F1",  # Indigo
            "Unknown": "#9CA3AF"  # Light gray
        }

        # Special colors for source and target
        source_color = "#FF0000"  # Red for source
        target_color = "#00FF00"  # Green for target

        # Node shapes by type
        shapes = {
            "PERSON": "dot",
            "ORGANIZATION": "square",
            "GOVERNMENT_BODY": "triangle",
            "COMMERCIAL_COMPANY": "diamond",
            "LOCATION": "star",
            "POSITION": "ellipse",
            "MONEY": "hexagon",
            "ASSET": "box",
            "EVENT": "database",
            "Unknown": "dot"
        }

        # Add nodes to PyVis with visual attributes
        for node_id, attrs in viz_graph.nodes(data=True):
            entity_type = attrs.get("type", "Unknown")
            is_source = attrs.get("is_source", False)
            is_target = attrs.get("is_target", False)

            # Determine color and size
            if is_source:
                color = source_color
                size = 30  # Larger for source/target
                border_width = 3
            elif is_target:
                color = target_color
                size = 30
                border_width = 3
            else:
                color = colors.get(entity_type, colors["Unknown"])
                size = 20
                border_width = 1

            # Get shape
            shape = shapes.get(entity_type, shapes["Unknown"])

            # Create label with type
            label = attrs.get("label", "Unknown")
            title = f"{label} ({entity_type})"
            if is_source:
                title += " (Source)"
            elif is_target:
                title += " (Target)"

            # Add node with visual attributes
            net.add_node(
                node_id,
                label=label,
                title=title,
                color=color,
                shape=shape,
                size=size,
                borderWidth=border_width,
                font={'size': 16, 'face': 'Arial', 'color': 'black'}
            )

        # Add edges to PyVis with visual attributes
        for source, target, attrs in viz_graph.edges(data=True):
            rel_type = attrs.get("type", "RELATED_TO")

            # Highlight edges in shortest path
            is_in_shortest_path = False
            if len(paths) > 0 and len(paths[0]) > 0:
                # Check if this edge is in the first (shortest) path
                for i in range(len(paths[0]) - 1):
                    if paths[0][i] == source and paths[0][i + 1] == target:
                        is_in_shortest_path = True
                        break
                    if paths[0][i] == target and paths[0][i + 1] == source:
                        is_in_shortest_path = True
                        break

            # Set edge attributes
            if is_in_shortest_path:
                width = 4
                color = "#FF5500"  # Orange for shortest path
                dash = False
            else:
                width = 2
                color = "#666666"  # Gray for other connections
                dash = True

            # Add edge with visual attributes
            net.add_edge(
                source,
                target,
                title=rel_type,
                label=rel_type,
                width=width,
                color=color,
                dashes=dash,
                arrows={'to': {'enabled': True, 'scaleFactor': 0.5}}
            )

        # Set options with proper structure
        net.options = {
            "physics": {
                "enabled": True,
                "solver": "forceAtlas2Based",
                "forceAtlas2Based": {
                    "gravitationalConstant": -100,
                    "centralGravity": 0.2,
                    "springLength": 100,
                    "springConstant": 0.05,
                    "damping": 0.09
                },
                "stabilization": {
                    "enabled": True,
                    "iterations": 1000,
                    "updateInterval": 25
                }
            },
            "interaction": {
                "hover": True,
                "navigationButtons": True,
                "keyboard": {
                    "enabled": True
                }
            },
            "edges": {
                "smooth": {
                    "enabled": True,
                    "type": "dynamic"
                },
                "arrows": {
                    "to": {
                        "enabled": True,
                        "scaleFactor": 0.5
                    }
                }
            },
            "nodes": {
                "font": {
                    "size": 16,
                    "face": "Arial"
                }
            }
        }

        # Save to HTML file
        graph_html_path = ROOT_DIR / "temp" / "connection_graph.html"
        net.save_graph(str(graph_html_path))

        # Display number of paths found
        st.caption(
            f"Found {len(paths)} path(s) between {source_entity} and {target_entity} within {degrees_of_separation} degrees of separation")

        # Display shortest path details
        if paths:
            shortest_path = paths[0]
            path_description = " → ".join([entity_id_to_name.get(node_id, "Unknown") for node_id in shortest_path])
            st.info(f"**Shortest path:** {path_description} ({len(shortest_path) - 1} hops)")

        # Read HTML content and display
        with open(graph_html_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        st.components.v1.html(html_content, height=700, scrolling=True)


def render_entity_centered_explorer(entities, relationships):
    """
    Render the entity-centered explorer to visualize connections around a specific entity.

    Args:
        entities: List of entity dictionaries
        relationships: List of relationship dictionaries
    """
    import math
    import networkx as nx
    from pyvis.network import Network

    st.subheader("Entity-Centered Explorer")

    # Create entity name to ID mapping for easier lookup
    entity_name_to_id = {}
    entity_id_to_name = {}
    entity_id_to_type = {}

    for entity in entities:
        entity_id = entity.get("id")
        entity_name = entity.get("name", "Unknown")
        entity_type = entity.get("type", "Unknown")

        # Store mappings
        if entity_id and entity_name:
            entity_name_to_id[entity_name] = entity_id
            entity_id_to_name[entity_id] = entity_name
            entity_id_to_type[entity_id] = entity_type

    # Get sorted list of entity names for dropdowns
    entity_names = sorted(entity_name_to_id.keys())

    # Create controls
    col1, col2 = st.columns(2)

    with col1:
        # Search/filter for entity
        entity_search = st.text_input(
            "Search Entity",
            placeholder="Start typing to search...",
            key="centered_entity_search"
        )

        # Filter entity names based on search
        filtered_entities = [name for name in entity_names
                             if entity_search.lower() in name.lower()] if entity_search else entity_names

        # Select entity
        central_entity = st.selectbox(
            "Center Entity",
            options=filtered_entities,
            index=0 if filtered_entities else None,
            key="centered_entity_select"
        )

    with col2:
        # Connections depth slider
        max_depth = 15
        connection_depth = st.slider(
            "Connection Depth",
            min_value=1,
            max_value=max_depth,
            value=1,
            key="centered_connection_depth"
        )

        # Filter by relationship type
        rel_types = sorted(list(set(
            rel.get("type", rel.get("relationship_type", "Unknown"))
            for rel in relationships
        )))

        selected_rel_types = st.multiselect(
            "Filter by Relationship Type",
            options=rel_types,
            default=rel_types,
            key="centered_rel_type_filter"
        )

    # Button to generate visualization
    if st.button("Visualize Centered Network", key="visualize_centered_btn"):
        # Check if entity is selected
        if not central_entity:
            st.warning("Please select a center entity.")
            return

        # Get entity ID
        center_id = entity_name_to_id.get(central_entity)

        if not center_id:
            st.error("Could not find ID for the selected entity.")
            return

        # Create full graph
        G = nx.DiGraph()

        # Add all entities as nodes
        for entity in entities:
            entity_id = entity.get("id")
            if entity_id:
                G.add_node(
                    entity_id,
                    label=entity.get("name", "Unknown"),
                    type=entity.get("type", "Unknown")
                )

        # Add filtered relationships as edges
        for rel in relationships:
            source_rel_id = rel.get("source_entity_id", rel.get("from_entity_id"))
            target_rel_id = rel.get("target_entity_id", rel.get("to_entity_id"))
            rel_type = rel.get("type", rel.get("relationship_type", "Unknown"))

            # Skip if relationship type is filtered out
            if rel_type not in selected_rel_types:
                continue

            if source_rel_id and target_rel_id:
                G.add_edge(
                    source_rel_id,
                    target_rel_id,
                    type=rel_type
                )

        # For undirected traversal, create an undirected version of the graph
        UG = G.to_undirected()

        # Get nodes within n connections from the center entity
        nodes_within_n_connections = {center_id: 0}  # node_id: distance
        queue = [(center_id, 0)]  # (node_id, distance)

        while queue:
            node, distance = queue.pop(0)

            if distance < connection_depth:
                for neighbor in UG.neighbors(node):
                    if neighbor not in nodes_within_n_connections:
                        nodes_within_n_connections[neighbor] = distance + 1
                        queue.append((neighbor, distance + 1))

        # If no connections found
        if len(nodes_within_n_connections) <= 1:
            st.warning(f"No connections found for {central_entity} with the current filters.")
            return

        # Create visualization subgraph
        viz_graph = nx.DiGraph()

        # Add nodes from the neighborhood
        for node_id, distance in nodes_within_n_connections.items():
            if node_id in entity_id_to_name:
                viz_graph.add_node(
                    node_id,
                    label=entity_id_to_name[node_id],
                    type=entity_id_to_type.get(node_id, "Unknown"),
                    distance=distance,
                    is_center=(node_id == center_id)
                )

        # Add edges between nodes in the neighborhood (preserve direction)
        for u, v, data in G.edges(data=True):
            if u in nodes_within_n_connections and v in nodes_within_n_connections:
                viz_graph.add_edge(u, v, **data)

        # Create PyVis network for visualization
        net = Network(height="700px", width="100%", directed=True, notebook=True, cdn_resources='remote')

        # Node color map by type
        colors = {
            "PERSON": "#3B82F6",  # Blue
            "ORGANIZATION": "#10B981",  # Green
            "GOVERNMENT_BODY": "#60BD68",  # Green variant
            "COMMERCIAL_COMPANY": "#10B981",  # Green variant
            "LOCATION": "#F59E0B",  # Yellow/Orange
            "POSITION": "#8B5CF6",  # Purple
            "MONEY": "#EC4899",  # Pink
            "ASSET": "#EF4444",  # Red
            "EVENT": "#6366F1",  # Indigo
            "Unknown": "#9CA3AF"  # Light gray
        }

        # Center node color
        center_color = "#FF0000"  # Red for center

        # Node shapes by type
        shapes = {
            "PERSON": "dot",
            "ORGANIZATION": "square",
            "GOVERNMENT_BODY": "triangle",
            "COMMERCIAL_COMPANY": "diamond",
            "LOCATION": "star",
            "POSITION": "ellipse",
            "MONEY": "hexagon",
            "ASSET": "box",
            "EVENT": "database",
            "Unknown": "dot"
        }

        # Add nodes to PyVis with visual attributes
        for node_id, attrs in viz_graph.nodes(data=True):
            entity_type = attrs.get("type", "Unknown")
            distance = attrs.get("distance", 0)
            is_center = attrs.get("is_center", False)

            # Size based on distance from center (larger for center, smaller for distant nodes)
            size = 35 - (distance * 5) if distance <= 5 else 10

            # Determine color
            if is_center:
                color = center_color
                border_width = 3
            else:
                color = colors.get(entity_type, colors["Unknown"])
                border_width = 1

            # Get shape
            shape = shapes.get(entity_type, shapes["Unknown"])

            # Create label with type and distance
            label = attrs.get("label", "Unknown")
            if is_center:
                title = f"{label} ({entity_type}) - CENTER"
            else:
                title = f"{label} ({entity_type}) - {distance} step(s) away"

            # Add node with visual attributes
            net.add_node(
                node_id,
                label=label,
                title=title,
                color=color,
                shape=shape,
                size=size,
                borderWidth=border_width,
                font={'size': 16, 'face': 'Arial', 'color': 'black'}
            )

        # Add edges to PyVis with visual attributes
        for source, target, attrs in viz_graph.edges(data=True):
            rel_type = attrs.get("type", "RELATED_TO")

            # Check distances for gradient coloring
            source_distance = viz_graph.nodes[source].get("distance", 0)
            target_distance = viz_graph.nodes[target].get("distance", 0)

            # Determine edge width based on distance from center
            max_width = 5
            min_width = 1
            width = max(min_width, max_width - min(source_distance, target_distance))

            # Edge styling
            if source == center_id or target == center_id:
                # Direct connections to center
                color = "#FF0000"  # Red
                dash = False
            else:
                # Other connections
                color = "#666666"  # Gray
                dash = False

            # Add edge with visual attributes
            net.add_edge(
                source,
                target,
                title=rel_type,
                label=rel_type,
                width=width,
                color=color,
                dashes=dash,
                arrows={'to': {'enabled': True, 'scaleFactor': 0.5}}
            )

        # Configure layout based on center-focused radial approach
        net.options = {
            "physics": {
                "enabled": True,
                "solver": "forceAtlas2Based",
                "forceAtlas2Based": {
                    "gravitationalConstant": -200,
                    "centralGravity": 0.1,
                    "springLength": 150,
                    "springConstant": 0.08,
                    "damping": 0.09
                },
                "stabilization": {
                    "enabled": True,
                    "iterations": 1000,
                    "updateInterval": 25
                }
            },
            "interaction": {
                "hover": True,
                "navigationButtons": True,
                "keyboard": {
                    "enabled": True
                }
            },
            "edges": {
                "smooth": {
                    "enabled": True,
                    "type": "dynamic"
                },
                "arrows": {
                    "to": {
                        "enabled": True,
                        "scaleFactor": 0.5
                    }
                }
            },
            "nodes": {
                "font": {
                    "size": 16,
                    "face": "Arial"
                }
            }
        }

        # Save to HTML file
        graph_html_path = ROOT_DIR / "temp" / "centered_graph.html"
        net.save_graph(str(graph_html_path))

        # Display summary info
        st.caption(
            f"Showing {len(viz_graph.nodes)} entities and {len(viz_graph.edges)} relationships within {connection_depth} step(s) of {central_entity}")

        # Show connection counts by type
        connection_types = {}
        for _, _, data in viz_graph.edges(data=True):
            rel_type = data.get("type", "Unknown")
            connection_types[rel_type] = connection_types.get(rel_type, 0) + 1

        # Display connection type summary
        if connection_types:
            st.info("**Connection types:**")
            connection_summary = ", ".join(
                [f"{rel_type}: {count}" for rel_type, count in sorted(connection_types.items())])
            st.text(connection_summary)

        # Read HTML content and display
        with open(graph_html_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        st.components.v1.html(html_content, height=700, scrolling=True)


def render_relationship_graph():
    """
    Render the relationship network graph with additional exploration tabs.
    """
    st.subheader("Relationship Network Graph")

    # Apply custom CSS for taller graph
    custom_css = """
    <style>
    iframe {
        min-height: 750px !important;
        height: 750px !important;
    }
    .network-container, .vis-network {
        min-height: 750px !important;
        height: 750px !important;
    }
    div.vis-network div.vis-navigation {
        padding: 0;
        right: 10px;
        bottom: 10px;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    # Load data
    relationships_file = EXTRACTED_DATA_PATH / "relationships.json"
    entities_file = EXTRACTED_DATA_PATH / "entities.json"

    if not relationships_file.exists() or not entities_file.exists():
        st.info("No relationships or entities found. Upload and process documents first.")
        return

    try:
        with open(relationships_file, "r", encoding='utf-8') as f:
            relationships = json.load(f)
        with open(entities_file, "r", encoding='utf-8') as f:
            entities = json.load(f)
    except Exception as e:
        st.error(f"Error loading relationships or entities: {e}")
        return

    if not relationships or not entities:
        st.info("Entity or relationship data is empty.")
        return

    try:
        # Create tabs for the different visualizations and table
        overview_tab, connection_tab, centered_tab, table_tab = st.tabs([
            "Network Overview",
            "Connection Explorer",
            "Entity Explorer",
            "Relationship Table"
        ])

        # OVERVIEW TAB - Global network visualization
        with overview_tab:
            render_network_overview(entities, relationships)

        # CONNECTION TAB - Entity-to-Entity connection explorer
        with connection_tab:
            render_entity_connection_explorer(entities, relationships)

        # CENTERED TAB - Entity-centered exploration
        with centered_tab:
            render_entity_centered_explorer(entities, relationships)

        # TABLE TAB - Relationship table
        with table_tab:
            render_relationship_table(relationships, entities)

    except ImportError:
        st.error(
            "Please install PyVis and NetworkX (`pip install pyvis networkx pandas`) to view the relationship graph.")
    except Exception as e:
        st.error(f"Error generating relationship graph: {e}")
        logger.error(f"Graph generation failed: {traceback.format_exc()}")


def render_network_overview(entities, relationships):
    """
    Render the overview network graph (original visualization).

    Args:
        entities: List of entity dictionaries
        relationships: List of relationship dictionaries
    """
    import math
    import networkx as nx
    from pyvis.network import Network

    # Create columns for graph controls
    control_col1, control_col2, control_col3 = st.columns(3)

    with control_col1:
        # Limit number of entities
        top_entities_count = st.slider("Number of entities", 10, 100, min(30, len(entities)))

    with control_col2:
        # Filter by entity type
        entity_types = sorted(list(set(entity.get("type", "Unknown") for entity in entities)))
        selected_graph_types = st.multiselect(
            "Filter by entity type",
            options=entity_types,
            default=entity_types,
            key="entity_graph_type_filter"
        )

    with control_col3:
        # Graph physics options
        physics_enabled = st.checkbox("Enable physics", value=True)
        if physics_enabled:
            physics_solver = st.selectbox(
                "Physics solver",
                options=["forceAtlas2Based", "barnesHut", "repulsion"],
                index=1
            )
        else:
            physics_solver = "none"

    # Create entity lookup dict {entity_id: entity_data}
    entity_lookup = {entity.get("id"): entity for entity in entities if entity.get("id")}

    # Count entity mentions (inferred from relationship frequency)
    entity_mentions = {}
    for rel in relationships:
        source_id = rel.get("source_entity_id", rel.get("from_entity_id"))
        target_id = rel.get("target_entity_id", rel.get("to_entity_id"))
        if source_id:
            entity_mentions[source_id] = entity_mentions.get(source_id, 0) + 1
        if target_id:
            entity_mentions[target_id] = entity_mentions.get(target_id, 0) + 1

    # Filter entities by type and sort by mention count
    filtered_entities = [
        entity for entity in entities
        if entity.get("type", "Unknown") in selected_graph_types
    ]

    for entity in filtered_entities:
        entity_id = entity.get("id")
        entity["mention_count"] = entity_mentions.get(entity_id, 1)

    top_entities = sorted(
        filtered_entities,
        key=lambda e: e.get("mention_count", 1),
        reverse=True
    )[:top_entities_count]

    # Create list of top entity IDs for filtering relationships
    top_entity_ids = [entity.get("id") for entity in top_entities]

    # Filter relationships involving top entities
    filtered_relationships = [
        rel for rel in relationships
        if (rel.get("source_entity_id", rel.get("from_entity_id")) in top_entity_ids and
            rel.get("target_entity_id", rel.get("to_entity_id")) in top_entity_ids)
    ]

    # Create NetworkX graph
    G = nx.DiGraph()

    # Add nodes (entities)
    for entity in top_entities:
        entity_id = entity.get("id")
        entity_name = entity.get("name", "Unknown")
        entity_type = entity.get("type", "Unknown")
        mentions = entity.get("mention_count", 1)

        G.add_node(
            entity_id,
            label=entity_name,
            type=entity_type,
            weight=mentions,
            title=f"{entity_name} ({entity_type})\nMentions: {mentions}"
        )

    # Add edges (relationships)
    for rel in filtered_relationships:
        source_id = rel.get("source_entity_id", rel.get("from_entity_id"))
        target_id = rel.get("target_entity_id", rel.get("to_entity_id"))
        rel_type = rel.get("type", rel.get("relationship_type", "RELATED_TO"))

        # Default confidence to 0.7 if not available
        confidence = rel.get("confidence", 0.7)

        G.add_edge(
            source_id,
            target_id,
            type=rel_type,
            confidence=confidence,
            title=f"{rel_type} (conf: {confidence:.2f})"
        )

    # Create PyVis network with increased height
    net = Network(height="700px", width="100%", directed=True, notebook=True, cdn_resources='remote')

    # Node color map by type
    colors = {
        "PERSON": "#3B82F6",  # Blue
        "ORGANIZATION": "#10B981",  # Green
        "GOVERNMENT_BODY": "#60BD68",  # Green variant
        "COMMERCIAL_COMPANY": "#10B981",  # Green variant
        "LOCATION": "#F59E0B",  # Yellow/Orange
        "POSITION": "#8B5CF6",  # Purple
        "MONEY": "#EC4899",  # Pink
        "ASSET": "#EF4444",  # Red
        "EVENT": "#6366F1",  # Indigo
        "Unknown": "#9CA3AF"  # Light gray
    }

    # Node shapes by type
    shapes = {
        "PERSON": "dot",
        "ORGANIZATION": "square",
        "GOVERNMENT_BODY": "triangle",
        "COMMERCIAL_COMPANY": "diamond",
        "LOCATION": "star",
        "POSITION": "ellipse",
        "MONEY": "hexagon",
        "ASSET": "box",
        "EVENT": "database",
        "Unknown": "dot"
    }

    # Add nodes to PyVis with visual attributes
    for node_id, attrs in G.nodes(data=True):
        entity_type = attrs.get("type", "Unknown")
        mentions = attrs.get("weight", 1)

        # Size based on mention count (logarithmic scaling)
        size = 15 + (10 * math.log(mentions + 1))

        # Get color and shape from type maps
        color = colors.get(entity_type, colors["Unknown"])
        shape = shapes.get(entity_type, shapes["Unknown"])

        # Add node with visual attributes
        net.add_node(
            node_id,
            label=attrs.get("label", "Unknown"),
            title=attrs.get("title", ""),
            color=color,
            shape=shape,
            size=size,
            font={'size': min(14 + int(math.log(mentions + 1)), 24)}
        )

    # Add edges to PyVis with visual attributes
    for source, target, attrs in G.edges(data=True):
        rel_type = attrs.get("type", "RELATED_TO")
        confidence = attrs.get("confidence", 0.7)

        # Width based on confidence
        width = 1 + (confidence * 5)

        # Determine edge color (slightly darker than source node color)
        source_type = G.nodes[source].get('type', 'Unknown')
        source_color = colors.get(source_type, colors["Unknown"])

        # Create darker variant for edge color
        rgb = source_color.lstrip('#')
        r, g, b = tuple(int(rgb[i:i + 2], 16) for i in (0, 2, 4))
        edge_color = f"#{max(0, r - 30):02x}{max(0, g - 30):02x}{max(0, b - 30):02x}"

        # Add edge with visual attributes
        net.add_edge(
            source,
            target,
            title=attrs.get("title", rel_type),
            label=rel_type if confidence > 0.6 else "",
            width=width,
            color=edge_color,
            arrows={'to': {'enabled': True, 'scaleFactor': 0.5}}
        )

    # Dynamic physics options based on user selection
    physics_options = {
        "forceAtlas2Based": {
            "gravitationalConstant": -50,
            "centralGravity": 0.01,
            "springLength": 100,
            "springConstant": 0.08
        },
        "barnesHut": {
            "gravitationalConstant": -2000,
            "centralGravity": 0.3,
            "springLength": 95,
            "springConstant": 0.04,
            "damping": 0.09
        },
        "repulsion": {
            "nodeDistance": 100,
            "centralGravity": 0.2,
            "springLength": 200,
            "springConstant": 0.05,
            "damping": 0.09
        }
    }

    # Build physics configuration
    physics_config = {
        "enabled": physics_enabled,
        "solver": physics_solver,
        "stabilization": {"enabled": True, "iterations": 1000},
    }

    # Add solver-specific options if physics is enabled
    if physics_enabled and physics_solver in physics_options:
        for key, value in physics_options[physics_solver].items():
            physics_config[key] = value

    # Set options with proper structure
    net.options = {
        "physics": physics_config,
        "interaction": {"hover": True, "navigationButtons": True},
        "edges": {
            "smooth": {"enabled": True, "type": "dynamic"},
            "arrows": {"to": {"enabled": True}}
        }
    }

    # Save to HTML file
    graph_html_path = ROOT_DIR / "temp" / "relationship_graph.html"
    net.save_graph(str(graph_html_path))

    # Display number of entities and relationships
    st.caption(f"Displaying {len(top_entities)} entities and {len(filtered_relationships)} relationships")

    # Read HTML content and display with increased height
    with open(graph_html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    st.components.v1.html(html_content, height=700, scrolling=True)


def render_relationship_table(relationships, entities):
    """
    Render a table of relationships with filtering options.

    Args:
        relationships: List of relationship dictionaries
        entities: List of entity dictionaries
    """
    import pandas as pd

    if not relationships:
        st.info("No relationships have been extracted.")
        return

    st.subheader("Relationship Table")

    # Create entity lookup for getting names
    entity_lookup = {entity.get("id"): entity for entity in entities if entity.get("id")}

    # Filtering options
    col1, col2 = st.columns(2)

    with col1:
        # Get unique relationship types
        rel_types = sorted(list(set(rel.get("type", rel.get("relationship_type", "Unknown"))
                                    for rel in relationships)))
        selected_rel_types = st.multiselect(
            "Filter by relationship type",
            options=rel_types,
            default=rel_types,
            key="relationship_type_filter"
        )

    with col2:
        # Entity search field
        entity_search = st.text_input(
            "Search by entity name",
            placeholder="Enter entity name..."
        )



    # Create formatted relationship data
    rel_data = []
    for rel in relationships:
        # Get relationship type (handle different field names)
        rel_type = rel.get("type", rel.get("relationship_type", "Unknown"))

        # Get source and target entity IDs (handle different field names)
        source_id = rel.get("source_entity_id", rel.get("from_entity_id"))
        target_id = rel.get("target_entity_id", rel.get("to_entity_id"))

        # Get source and target entity names from lookup
        source_entity = entity_lookup.get(source_id, {})
        target_entity = entity_lookup.get(target_id, {})

        source_name = source_entity.get("name", "Unknown")
        target_name = target_entity.get("name", "Unknown")



        # Get document info
        document_id = rel.get("document_id", "Unknown")
        file_name = rel.get("file_name", document_id)

        # Add to relationship data
        rel_data.append({
            "Source": source_name,
            "Source Type": source_entity.get("type", "Unknown"),
            "Relationship": rel_type,
            "Target": target_name,
            "Target Type": target_entity.get("type", "Unknown"),
            "Document": file_name
        })

    # Convert to DataFrame
    rel_df = pd.DataFrame(rel_data)

    # Apply filters
    filtered_df = rel_df[
        rel_df["Relationship"].isin(selected_rel_types)
        ]

    # Apply entity search if provided
    if entity_search:
        search_term = entity_search.lower()
        filtered_df = filtered_df[
            filtered_df["Source"].str.lower().str.contains(search_term) |
            filtered_df["Target"].str.lower().str.contains(search_term)
            ]

    # Display relationship count
    st.markdown(f"**Showing {len(filtered_df)} of {len(rel_df)} relationships**")

    # Display relationship table
    st.dataframe(
        filtered_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Source": st.column_config.TextColumn("Source", width="medium"),
            "Source Type": st.column_config.TextColumn("Source Type", width="small"),
            "Relationship": st.column_config.TextColumn("Relationship", width="medium"),
            "Target": st.column_config.TextColumn("Target", width="medium"),
            "Target Type": st.column_config.TextColumn("Target Type", width="small"),
            "Document": st.column_config.TextColumn("Document", width="medium")
        }
    )

def render_query_page():
    """
    Render the query and chat interface.
    (Now relies on QueryEngine having the correct model set)
    """
    st.header("💬 Query System")

    # Check if there's data to query
    query_engine = get_or_create_query_engine() # Ensures QE exists
    collection_info = query_engine.get_collection_info()

    if not collection_info.get("exists", False) or collection_info.get("points_count", 0) == 0:
        st.warning("No indexed documents found. Please upload and process documents first.")
        return

    # Check Aphrodite service status
    if not APHRODITE_SERVICE_AVAILABLE:
         st.error("Aphrodite service module is not available. Cannot query.")
         return

    service = get_service()
    if not service.is_running():
        st.warning("LLM service is not running. Please start it from the sidebar.")
        if st.button("Start LLM Service Now"):
             start_aphrodite_service()
             st.rerun()
        return

    # --- Check if the model designated by QueryEngine is loaded ---
    # QueryEngine's llm_model_name should have been updated after processing
    target_llm_name = query_engine.llm_model_name
    st.info(f"Querying using model: `{target_llm_name}`") # Inform user

    status = service.get_status(timeout=5)
    is_model_loaded_in_service = status.get("model_loaded", False)
    current_model_in_service = status.get("current_model")

    # Check if the correct model is loaded
    if not is_model_loaded_in_service or current_model_in_service != target_llm_name:
        st.info(f"LLM service needs model '{target_llm_name}'. Currently loaded: '{current_model_in_service}'.")
        with st.spinner(f"Loading LLM model ({target_llm_name})..."):
            # Use the query engine's method to load the designated model
            success = query_engine.load_llm_model() # Loads the model stored in query_engine.llm_model_name

        if success:
            st.session_state.llm_model_loaded = True
            st.success("LLM model loaded successfully!")
            # Update process info state
            try:
                status_after_load = service.get_status()
                process_info = {
                    "pid": service.process.pid if service.process else None,
                    "model_name": status_after_load.get("current_model")
                }
                if process_info["pid"] and process_info["model_name"]:
                    st.session_state.aphrodite_process_info = process_info
                else:
                     logger.warning("Failed to update process info after model load.")
            except Exception as e:
                logger.warning(f"Error saving process info after model load: {e}")
            st.rerun() # Rerun to reflect loaded state
        else:
            st.error(f"Failed to load LLM model ({target_llm_name}). Please check logs.")
            st.session_state.llm_model_loaded = False
            # Update process info state to reflect no model loaded
            if st.session_state.aphrodite_process_info:
                 st.session_state.aphrodite_process_info["model_name"] = None
            return
    else:
        # Correct model is already loaded
        st.session_state.llm_model_loaded = True


    # --- Chat Interface (No changes needed below this point in this function) ---
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"]) # Use markdown for potential formatting
            if message.get("sources"):
                with st.expander("View sources"):
                    for i, source in enumerate(message["sources"]):
                        st.markdown(f"**Source {i + 1} (Score: {source['score']:.2f}):**")
                        st.markdown(f"> {source['text']}") # Blockquote for context
                        meta = source['metadata']
                        st.caption(f"Document: {meta.get('file_name', 'N/A')} | Page: {meta.get('page_num', 'N/A')} | Chunk: {meta.get('chunk_id', 'N/A')}")
                        st.markdown("---")

    # Chat input
    prompt = st.chat_input("Ask a question about the documents...")

    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            sources_placeholder = st.empty()
            with st.spinner("Generating response..."):
                response_data = query_engine.query(prompt)

            full_response = response_data.get('answer', "Error: No answer received.")
            sources = response_data.get('sources', [])
            message_placeholder.markdown(full_response)

            if sources:
                 with sources_placeholder.expander("View sources", expanded=False):
                    for i, source in enumerate(sources):
                        st.markdown(f"**Source {i + 1} (Score: {source['score']:.2f}):**")
                        st.markdown(f"> {source['text']}")
                        meta = source['metadata']
                        st.caption(f"Document: {meta.get('file_name', 'N/A')} | Page: {meta.get('page_num', 'N/A')} | Chunk: {meta.get('chunk_id', 'N/A')}")
                        st.markdown("---")

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": full_response,
                "sources": sources
            })


def render_settings_page():
    """
    Render the settings page. (No changes needed related to model loading logic)
    """
    st.header("⚙️ Settings")

    # Display current configuration
    with st.expander("Current Configuration", expanded=False):
        st.json(CONFIG) # Display the whole config for reference

    st.subheader("Update Settings")

    # Retrieval settings
    st.markdown("#### Retrieval Settings")
    col1, col2 = st.columns(2)
    with col1:
        top_k_vector = st.slider("Vector Search Results (top_k_vector)", 1, 50, CONFIG["retrieval"]["top_k_vector"], key="s_tkv")
        top_k_bm25 = st.slider("BM25 Search Results (top_k_bm25)", 1, 50, CONFIG["retrieval"]["top_k_bm25"], key="s_tkb")
        top_k_hybrid = st.slider("Hybrid Search Results (top_k_hybrid)", 1, 30, CONFIG["retrieval"]["top_k_hybrid"], key="s_tkh")
        top_k_rerank = st.slider("Reranked Results (top_k_rerank)", 1, 20, CONFIG["retrieval"]["top_k_rerank"], key="s_tkr")
    with col2:
        vector_weight = st.slider("Vector Weight (RRF)", 0.0, 1.0, float(CONFIG["retrieval"]["vector_weight"]), step=0.05, key="s_vw")
        bm25_weight = st.slider("BM25 Weight (RRF)", 0.0, 1.0, float(CONFIG["retrieval"]["bm25_weight"]), step=0.05, key="s_bw")
        use_reranking = st.checkbox("Use Reranking", value=CONFIG["retrieval"]["use_reranking"], key="s_ur")

    # Extraction settings
    st.markdown("#### Extraction Settings")
    dedup_threshold = st.slider("Entity Deduplication Threshold (%)", 0, 100, CONFIG["extraction"]["deduplication_threshold"], key="s_dt")

    # LLM shared settings (Aphrodite)
    st.markdown("#### LLM Shared Settings (Aphrodite Service)")
    col3, col4 = st.columns(2)
    with col3:
        max_model_len = st.slider("Max Model Context Length", 1024, 16384, CONFIG["aphrodite"]["max_model_len"], step=1024, key="s_mml")
        # Ensure quantization value exists in options before setting index
        quant_options = ["none", "fp8", "gptq", "awq"] # Example options, adjust based on Aphrodite support
        current_quant = CONFIG["aphrodite"]["quantization"]
        quant_index = quant_options.index(current_quant) if current_quant in quant_options else 0
        quantization = st.selectbox("Quantization", options=quant_options, index=quant_index, key="s_q")

    with col4:
        # Checkbox values might be missing in older configs, provide defaults
        use_flash_attention = st.checkbox("Use Flash Attention", value=CONFIG["aphrodite"].get("use_flash_attention", True), key="s_ufa")
        compile_model = st.checkbox("Compile Model (Experimental)", value=CONFIG["aphrodite"].get("compile_model", False), key="s_cm")

    # Extraction-specific LLM parameters
    st.markdown("#### Extraction Parameters (Aphrodite Service)")
    col5, col6 = st.columns(2)
    with col5:
        extraction_temperature = st.slider("Extraction Temperature", 0.0, 1.0, float(CONFIG["aphrodite"].get("extraction_temperature", 0.1)), step=0.05, key="s_et")
    with col6:
        extraction_max_new_tokens = st.slider("Extraction Max Tokens", 256, 4096, CONFIG["aphrodite"].get("extraction_max_new_tokens", 1024), step=128, key="s_emt")

    # Chat-specific LLM parameters
    st.markdown("#### Chat Parameters (Aphrodite Service)")
    col7, col8 = st.columns(2)
    with col7:
        chat_temperature = st.slider("Chat Temperature", 0.0, 1.5, float(CONFIG["aphrodite"].get("chat_temperature", 0.7)), step=0.05, key="s_ct")
        top_p = st.slider("Top P", 0.1, 1.0, float(CONFIG["aphrodite"].get("top_p", 0.9)), step=0.05, key="s_tp")
    with col8:
        chat_max_new_tokens = st.slider("Chat Max Tokens", 256, 4096, CONFIG["aphrodite"].get("chat_max_new_tokens", 1024), step=128, key="s_cmt")


    # Save button
    if st.button("Save Settings", type="primary"):
        # Update config dictionary with new values
        CONFIG["retrieval"]["top_k_vector"] = top_k_vector
        CONFIG["retrieval"]["top_k_bm25"] = top_k_bm25
        CONFIG["retrieval"]["top_k_hybrid"] = top_k_hybrid
        CONFIG["retrieval"]["top_k_rerank"] = top_k_rerank
        CONFIG["retrieval"]["vector_weight"] = float(vector_weight)
        CONFIG["retrieval"]["bm25_weight"] = float(bm25_weight)
        CONFIG["retrieval"]["use_reranking"] = use_reranking
        CONFIG["extraction"]["deduplication_threshold"] = dedup_threshold
        CONFIG["aphrodite"]["max_model_len"] = max_model_len
        CONFIG["aphrodite"]["quantization"] = quantization
        CONFIG["aphrodite"]["use_flash_attention"] = use_flash_attention
        CONFIG["aphrodite"]["compile_model"] = compile_model
        CONFIG["aphrodite"]["extraction_temperature"] = float(extraction_temperature)
        CONFIG["aphrodite"]["extraction_max_new_tokens"] = extraction_max_new_tokens
        CONFIG["aphrodite"]["chat_temperature"] = float(chat_temperature)
        CONFIG["aphrodite"]["chat_max_new_tokens"] = chat_max_new_tokens
        CONFIG["aphrodite"]["top_p"] = float(top_p)

        # Save to file
        try:
            with open(CONFIG_PATH, "w") as f:
                yaml.dump(CONFIG, f, default_flow_style=False, sort_keys=False)
            st.success("Settings saved successfully!")

            # Inform user that LLM service needs to be restarted for some changes
            if st.session_state.aphrodite_service_running:
                st.warning("**Note:** LLM service might need to be restarted for changes like Max Model Length, Quantization, Flash Attention, or Compile Model to take full effect. Use the sidebar to Stop/Start the service.")
            # Rerun to reflect changes if needed, or just update internal state
            # For now, just show success message. User needs to manage restarts.

        except Exception as e:
            st.error(f"Failed to save settings: {e}")
            logger.error(f"Error saving config file: {e}")


def process_documents(uploaded_files, selected_llm_name, vl_pages, vl_process_all):
    """
    Process uploaded documents using sequential processing.
    UPDATED: Also updates the QueryEngine to use the selected LLM.

    Args:
        uploaded_files: List of uploaded file objects
        selected_llm_name: Model name selected by user for processing AND querying
        vl_pages: List of specific page numbers for visual processing
        vl_process_all: Boolean indicating if all PDF pages should be visually processed
    """
    try:
        # ... (keep existing initialization, service checks, document loading, chunking) ...
        # Ensure Aphrodite service is running
        if not APHRODITE_SERVICE_AVAILABLE:
            st.error("Aphrodite service module not available. Cannot process.")
            st.session_state.processing = False
            return
        service = get_service()
        if not service.is_running():
            logger.info("Starting Aphrodite service for document processing")
            if not start_aphrodite_service():
                st.session_state.processing_status = "Failed to start LLM service. Processing aborted."
                st.session_state.processing = False
                return

        # --- Document Loading ---
        st.session_state.processing_status = "Loading documents..."
        st.session_state.processing_progress = 0.10
        document_loader = DocumentLoader()
        documents = []
        temp_dir = ROOT_DIR / "temp"
        temp_dir.mkdir(exist_ok=True)
        for i, file in enumerate(uploaded_files):
            file_path = temp_dir / file.name
            try:
                with open(file_path, "wb") as f: f.write(file.getbuffer())
                doc = document_loader.load_document(file_path)
                if doc: documents.append(doc)
                else: logger.warning(f"Failed to load document from file: {file.name}")
            except Exception as e:
                 logger.error(f"Error loading document {file.name}: {e}")
                 st.warning(f"Could not load {file.name}, skipping.")
            progress = 0.10 + (0.10 * (i + 1) / len(uploaded_files))
            st.session_state.processing_progress = progress
            st.session_state.processing_status = f"Loaded document {i + 1}/{len(uploaded_files)}: {file.name}"
        if not documents:
             st.error("No documents were successfully loaded.")
             st.session_state.processing = False
             return

        # --- Chunking ---
        st.session_state.processing_status = "Chunking documents..."
        st.session_state.processing_progress = 0.25
        document_chunker = DocumentChunker()
        document_chunker.load_model()
        all_chunks = []
        for i, doc in enumerate(documents):
            try:
                doc_chunks = document_chunker.chunk_document(doc)
                all_chunks.extend(doc_chunks)
                progress = 0.25 + (0.15 * (i + 1) / len(documents))
                st.session_state.processing_progress = progress
                st.session_state.processing_status = f"Chunked document {i + 1}/{len(documents)}: {doc.get('file_name', 'Unknown')}"
            except Exception as e:
                 logger.error(f"Error chunking document {doc.get('file_name', 'Unknown')}: {e}")
        document_chunker.shutdown()
        if not all_chunks:
             st.error("No chunks were generated.")
             st.session_state.processing = False
             return

        # --- Entity Extraction ---
        st.session_state.processing_status = "Preparing entity extraction..."
        st.session_state.processing_progress = 0.40
        # Use the selected LLM name for extraction
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
        # Process (this ensures the selected_llm_name is loaded in the service)
        entity_extractor.process_chunks(all_chunks, visual_chunk_ids)
        st.session_state.processing_progress = 0.75
        st.session_state.processing_status = "Saving extraction results..."
        entity_extractor.save_results()
        modified_chunks = entity_extractor.get_modified_chunks()

        # --- IMPORTANT: Update QueryEngine to use the same model ---
        try:
            query_engine = get_or_create_query_engine()
            query_engine.llm_model_name = selected_llm_name # Set the model for subsequent queries
            st.session_state.selected_llm_model_name = selected_llm_name # Update session state backup
            logger.info(f"QueryEngine LLM model updated to: {selected_llm_name}")
        except Exception as e:
            logger.error(f"Failed to update QueryEngine model name: {e}")
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
        document_indexer.load_model()
        # (Keep collection check/clear logic)
        try:
            collection_info = query_engine.get_collection_info()
            if collection_info.get("exists", False):
                 expected_dim = 1024 # Make dynamic based on embedding model if needed
                 vector_size = collection_info.get("vector_size", 0)
                 if vector_size != 0 and vector_size != expected_dim:
                    logger.warning(f"Clearing collection due to dimension mismatch ({vector_size} vs {expected_dim}).")
                    query_engine.clear_collection()
        except Exception as e:
            logger.warning(f"Error checking/clearing collection: {e}")
        # Index documents
        document_indexer.index_documents(modified_chunks)
        document_indexer.shutdown()
        st.session_state.processing_progress = 0.95
        st.session_state.processing_status = "Indexing complete."

        # Complete - 100% progress
        st.session_state.processing_status = "Processing completed successfully!"
        st.session_state.processing_progress = 1.0
        query_engine = get_or_create_query_engine() # Update collection info one last time
        st.session_state.collection_info = query_engine.get_collection_info()

    except Exception as e:
        logger.error(f"Fatal error processing documents: {e}", exc_info=True)
        st.session_state.processing_status = f"Error: {str(e)}"
        st.error(f"An unexpected error occurred: {e}")
    finally:
        st.session_state.processing = False


def get_or_create_query_engine():
    """
    Get existing query engine or create a new one.
    UPDATED: Sets the engine's model based on session state if available.
    """
    if "query_engine" not in st.session_state or st.session_state.query_engine is None:
        logger.info("Creating new QueryEngine instance.")
        st.session_state.query_engine = QueryEngine()
        # On creation, set its model based on session state (if processing ran) or config default
        if st.session_state.get("selected_llm_model_name"):
            st.session_state.query_engine.llm_model_name = st.session_state.selected_llm_model_name
            logger.info(f"QueryEngine created, using session LLM: {st.session_state.selected_llm_model_name}")
        else:
            # Falls back to the default chat_model defined in QueryEngine.__init__
             logger.info(f"QueryEngine created, using default LLM: {st.session_state.query_engine.llm_model_name}")

    # Update collection info
    try:
        collection_info = st.session_state.query_engine.get_collection_info()
        st.session_state.collection_info = collection_info
    except Exception as e:
        logger.error(f"Failed to get collection info: {e}")
        st.session_state.collection_info = {"exists": False, "points_count": 0, "error": str(e)}

    # Sync Aphrodite status (keep existing sync logic)
    if APHRODITE_SERVICE_AVAILABLE:
        service = get_service()
        is_service_actually_running = service.is_running()
        if st.session_state.aphrodite_service_running != is_service_actually_running:
             st.session_state.aphrodite_service_running = is_service_actually_running
        if is_service_actually_running:
             status = service.get_status(timeout=5)
             model_loaded = status.get("model_loaded", False)
             current_model = status.get("current_model")
             if st.session_state.llm_model_loaded != model_loaded or st.session_state.aphrodite_process_info.get("model_name") != current_model:
                   st.session_state.llm_model_loaded = model_loaded
                   if st.session_state.aphrodite_process_info: st.session_state.aphrodite_process_info["model_name"] = current_model
                   else: st.session_state.aphrodite_process_info = {"pid": service.process.pid if service.process else None, "model_name": current_model}
        else:
            st.session_state.llm_model_loaded = False
            st.session_state.aphrodite_process_info = None

    return st.session_state.query_engine


# Removed load_chat_model function as load_llm_model in query_engine handles generic loading

def clear_all_data():
    """
    Clear all data from the system (vector DB, BM25, extracted files).
    """
    try:
        logger.info("Clearing all indexed and extracted data...")
        # Use query engine's clear method
        query_engine = get_or_create_query_engine()
        success = query_engine.clear_collection() # This deletes Qdrant, BM25, extracted files

        if success:
             logger.info("Data clearing successful.")
        else:
             logger.error("Data clearing process reported errors.")
             st.error("An error occurred during data clearing. Check logs.")

        # Reset relevant session state
        st.session_state.chat_history = []
        st.session_state.collection_info = {"exists": False, "points_count": 0} # Assume cleared

        # Refresh collection info state
        try:
             query_engine = get_or_create_query_engine() # Re-gets engine and updates info
             logger.info(f"Collection info after clearing: {st.session_state.collection_info}")
        except Exception as e:
            logger.error(f"Error getting collection info after clear: {e}")

        # Note: Aphrodite service remains running, model state is untouched by data clear.
        if st.session_state.aphrodite_service_running:
            logger.info("Aphrodite service remains running after data clear.")

    except Exception as e:
        st.error(f"An error occurred while clearing data: {e}")
        logger.error(f"Clear data failed: {traceback.format_exc()}")


def render_cluster_map_page():
    """
    Render the cluster map visualization page with improved DataMapPlot support.
    """
    st.header("🔍 Document Cluster Map")

    # Import cluster_map module
    try:
        from src.core.visualization import cluster_map
        cluster_map_available = True
    except ImportError as e:
        st.error(f"Cluster map module not available: {e}")
        st.warning("Make sure the cluster_map.py module is installed in the src/core/visualization directory.")
        cluster_map_available = False
        return

    # Check if dependencies are available
    if not cluster_map.check_dependencies():
        st.error("Required dependencies for cluster mapping are not available.")
        st.warning(
            "Please install the required packages: `pip install cuml umap-learn hdbscan bertopic plotly scikit-learn`")
        return

    # Get query engine
    query_engine = get_or_create_query_engine()
    collection_info = query_engine.get_collection_info()

    if not collection_info.get("exists", False) or collection_info.get("points_count", 0) == 0:
        st.warning("No indexed documents found. Please upload and process documents first.")
        return

    # Add configuration controls
    with st.expander("Clustering Configuration"):
        # Create tabs for better organization
        config_tabs = st.tabs(["UMAP & HDBSCAN", "Topic Parameters", "Visualization"])

        with config_tabs[0]:
            # UMAP Parameters
            st.subheader("UMAP Parameters")
            col1, col2 = st.columns(2)

            with col1:
                umap_n_neighbors = st.slider(
                    "n_neighbors",
                    min_value=5,
                    max_value=50,
                    value=CONFIG["clustering"]["umap"].get("n_neighbors", 15),
                    step=1,
                    help="Number of neighbors to consider for manifold approximation"
                )

                umap_min_dist = st.slider(
                    "min_dist",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(CONFIG["clustering"]["umap"].get("min_dist", 0.0)),
                    step=0.05,
                    help="Minimum distance between embedded points"
                )

            with col2:
                umap_n_components = st.slider(
                    "n_components",
                    min_value=2,
                    max_value=10,
                    value=CONFIG["clustering"]["umap"].get("n_components", 5),
                    step=1,
                    help="Dimension of the embedded space"
                )

                umap_metric_options = ["cosine", "euclidean", "manhattan", "correlation"]
                umap_metric = st.selectbox(
                    "metric",
                    options=umap_metric_options,
                    index=umap_metric_options.index(CONFIG["clustering"]["umap"].get("metric", "cosine")),
                    help="Distance metric to use"
                )

            # HDBSCAN Parameters
            st.subheader("HDBSCAN Parameters")
            col1, col2 = st.columns(2)

            with col1:
                min_cluster_size = st.slider(
                    "min_cluster_size",
                    min_value=5,
                    max_value=100,
                    value=CONFIG["clustering"]["hdbscan"].get("min_cluster_size", 10),
                    step=5,
                    help="Minimum number of documents to form a cluster"
                )

            with col2:
                min_samples = st.slider(
                    "min_samples",
                    min_value=1,
                    max_value=20,
                    value=CONFIG["clustering"]["hdbscan"].get("min_samples", 5),
                    step=1,
                    help="Number of samples in a neighborhood for a point to be considered a core point"
                )

        with config_tabs[1]:
            # Topic Parameters
            st.subheader("Topic Parameters")
            col1, col2 = st.columns(2)

            with col1:
                nr_topics_options = ["auto"] + [str(i) for i in range(1, 51)]
                nr_topics = st.selectbox(
                    "Number of Topics",
                    options=nr_topics_options,
                    index=nr_topics_options.index(str(CONFIG["clustering"]["topics"].get("nr_topics", "auto"))),
                    help="Number of topics to find, 'auto' for automatic detection"
                )

            with col2:
                # Seed topic input
                seed_topics = st.text_area(
                    "Seed Topics (comma-separated keywords per line)",
                    value="\n".join(
                        [",".join(topic) for topic in CONFIG["clustering"]["topics"].get("seed_topic_list", [[]])]),
                    help="Enter comma-separated keywords for each topic, one topic per line"
                )

            # Parse seed topics
            seed_topic_list = []
            if seed_topics:
                for line in seed_topics.strip().split("\n"):
                    if line.strip():
                        seed_topic_list.append([word.strip() for word in line.split(",") if word.strip()])

        with config_tabs[2]:
            # Visualization Options
            st.subheader("Visualization Options")

            # Select visualization type
            visualization_type = st.radio(
                "Visualization Type",
                options=["Plotly", "DataMapPlot"],
                index=0 if CONFIG["clustering"].get("visualization_type", "plotly").lower() == "plotly" else 1,
                help="Choose visualization library for cluster map"
            )

            # Include outliers option - Move to main UI for better visibility
            # Will be handled outside the config expander

            # DataMapPlot specific options (show only when DataMapPlot is selected)
            if visualization_type == "DataMapPlot":
                # Check if datamapplot is available
                try:
                    import datamapplot
                    datamapplot_available = True
                except ImportError:
                    st.warning("DataMapPlot not installed. Install with: `pip install datamapplot`")
                    datamapplot_available = False

                if datamapplot_available:
                    st.subheader("DataMapPlot Options")

                    col1, col2 = st.columns(2)
                    with col1:
                        darkmode = st.checkbox(
                            "Dark Mode",
                            value=CONFIG["clustering"].get("datamapplot", {}).get("darkmode", False),
                            help="Use dark mode for the visualization"
                        )

                        cvd_safer = st.checkbox(
                            "CVD-Safer Palette",
                            value=CONFIG["clustering"].get("datamapplot", {}).get("cvd_safer", True),
                            help="Use a color vision deficiency (CVD) safer color palette"
                        )

                        enable_toc = st.checkbox(
                            "Enable Table of Contents",
                            value=CONFIG["clustering"].get("datamapplot", {}).get("enable_table_of_contents", True),
                            help="Show topic hierarchy as a table of contents"
                        )

                    with col2:
                        cluster_boundaries = st.checkbox(
                            "Show Cluster Boundaries",
                            value=CONFIG["clustering"].get("datamapplot", {}).get("cluster_boundary_polygons", True),
                            help="Draw boundary lines around clusters"
                        )

                        color_labels = st.checkbox(
                            "Color Label Text",
                            value=CONFIG["clustering"].get("datamapplot", {}).get("color_label_text", True),
                            help="Use colors for label text based on cluster colors"
                        )

                        marker_size = st.slider(
                            "Marker Size",
                            min_value=3,
                            max_value=15,
                            value=CONFIG["clustering"].get("datamapplot", {}).get("marker_size", 8),
                            help="Size of data points in visualization"
                        )

                    # Font selection
                    fonts = ["Arial", "Helvetica", "Roboto", "Times New Roman",
                             "Georgia", "Courier New", "Playfair Display SC", "Open Sans"]

                    current_font = CONFIG["clustering"].get("datamapplot", {}).get("font_family", "Arial")
                    font_index = fonts.index(current_font) if current_font in fonts else 0

                    font_family = st.selectbox(
                        "Font Family",
                        options=fonts,
                        index=font_index,
                        help="Font family for text elements"
                    )

                    # Add polygon alpha slider for glow effect around clusters
                    polygon_alpha = st.slider(
                        "Cluster Boundary Opacity",
                        min_value=0.05,
                        max_value=5.00,
                        value=CONFIG["clustering"].get("datamapplot", {}).get("polygon_alpha", 2.5),
                        step=0.05,
                        help="Opacity of the cluster boundary polygons (higher values make clusters more prominent)"
                    )

        # Update configuration button
        if st.button("Update Configuration"):
            # Create a copy of the configuration
            if "clustering" not in CONFIG:
                CONFIG["clustering"] = {}

            # Update UMAP parameters
            if "umap" not in CONFIG["clustering"]:
                CONFIG["clustering"]["umap"] = {}

            CONFIG["clustering"]["umap"].update({
                "n_neighbors": umap_n_neighbors,
                "n_components": umap_n_components,
                "min_dist": umap_min_dist,
                "metric": umap_metric
            })

            # Update HDBSCAN parameters
            if "hdbscan" not in CONFIG["clustering"]:
                CONFIG["clustering"]["hdbscan"] = {}

            CONFIG["clustering"]["hdbscan"].update({
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "prediction_data": True
            })

            # Update topic parameters
            if "topics" not in CONFIG["clustering"]:
                CONFIG["clustering"]["topics"] = {}

            CONFIG["clustering"]["topics"].update({
                "nr_topics": nr_topics,
                "seed_topic_list": seed_topic_list
            })

            # Update visualization type
            CONFIG["clustering"]["visualization_type"] = visualization_type.lower()

            # Update DataMapPlot settings if that's the selected visualization
            if visualization_type == "DataMapPlot":
                if "datamapplot" not in CONFIG["clustering"]:
                    CONFIG["clustering"]["datamapplot"] = {}

                CONFIG["clustering"]["datamapplot"].update({
                    "darkmode": darkmode,
                    "cvd_safer": cvd_safer,
                    "enable_table_of_contents": enable_toc,
                    "cluster_boundary_polygons": cluster_boundaries,
                    "color_label_text": color_labels,
                    "marker_size": marker_size,
                    "font_family": font_family,
                    "height": 800,
                    "width": "100%",
                    "polygon_alpha": polygon_alpha
                })

            # Save to file
            try:
                with open(CONFIG_PATH, "w") as f:
                    yaml.dump(CONFIG, f, default_flow_style=False, sort_keys=False)
                st.success("Configuration updated successfully!")
            except Exception as e:
                st.error(f"Failed to save configuration: {e}")

    # Include outliers option (moved outside the expander for better visibility)
    include_outliers = st.checkbox(
        "Include Outliers/Noise Points (Topic -1)",
        value=False,
        help="When enabled, includes outlier documents (Topic -1) in the visualization."
    )

    # Add generation button
    generate_btn = st.button(
        "Generate Cluster Map",
        type="primary",
        help="Generate a cluster map visualization from document embeddings"
    )

    # Create placeholder for progress and visualization
    progress_placeholder = st.empty()
    vis_placeholder = st.empty()

    # Store cluster map result in session state to avoid recomputing
    if "cluster_map_result" not in st.session_state:
        st.session_state.cluster_map_result = None

    # Generate cluster map when button is clicked
    if generate_btn:
        with progress_placeholder.container():
            with st.spinner("Extracting embeddings and generating cluster map..."):
                # Generate cluster map - pass the include_outliers parameter
                result, message = cluster_map.generate_cluster_map(
                    query_engine,
                    include_outliers=include_outliers
                )

                if result:
                    st.session_state.cluster_map_result = result
                    st.success(message)
                else:
                    st.error(message)

    # Display cluster map if available
    if st.session_state.cluster_map_result:
        result = st.session_state.cluster_map_result

        # Display cluster map info
        with vis_placeholder.container():
            st.subheader("Cluster Map")

            # Show statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Documents", result["document_count"])
            with col2:
                st.metric("Clustered Documents", result["clustered_count"])
            with col3:
                st.metric("Outliers", f"{result['outlier_count']} ({result['outlier_percentage']:.1f}%)")
            with col4:
                st.metric("Topics", result["topic_count"])

            # Display visualization based on type
            if result.get("is_datamap", False):
                # DataMapPlot visualization
                html_content = result["figure"]._repr_html_()  # or however it exposes HTML
                st.components.v1.html(html_content, height=800, scrolling=True)
            else:
                # Plotly visualization
                st.plotly_chart(result["figure"], use_container_width=True)

            # Display topic info table
            with st.expander("Topic Information", expanded=False):
                # Filter out topic -1 from the display if not including outliers
                if not include_outliers and not result.get("include_outliers", False):
                    filtered_topic_info = result["topic_info"][result["topic_info"]["Topic"] != -1]
                else:
                    filtered_topic_info = result["topic_info"]

                st.dataframe(
                    filtered_topic_info,
                    use_container_width=True,
                    column_config={
                        "Topic": st.column_config.NumberColumn("Topic", width="small"),
                        "Count": st.column_config.NumberColumn("Count", width="small"),
                        "Name": st.column_config.TextColumn("Name", width="medium"),
                        "Representation": st.column_config.TextColumn("Representation", width="large")
                    }
                )


def render_topic_filter_page():
    """
    Render the topic filtering page.
    """
    st.header("🧮 Topic Filter")

    # Import the embedding filter module
    try:
        from src.core.filtering.embedding_filter import EmbeddingFilter
        # Get query engine
        query_engine = get_or_create_query_engine()
        # Create embedding filter
        embedding_filter = EmbeddingFilter(query_engine)
    except ImportError as e:
        st.error(f"Embedding filter module not available: {e}")
        st.warning("Make sure the embedding_filter.py module is installed in the src/core/filtering directory.")
        return
    except Exception as e:
        st.error(f"Error initializing embedding filter: {e}")
        return

    # Check if there's data to filter
    collection_info = query_engine.get_collection_info()
    if not collection_info.get("exists", False) or collection_info.get("points_count", 0) == 0:
        st.warning("No indexed documents found. Please upload and process documents first.")
        return

    # Create form for filter parameters
    with st.form("topic_filter_form"):
        # Topic input
        topic_query = st.text_area(
            "Enter topic description or query",
            placeholder="Describe the topic you want to filter for...",
            height=100
        )

        # Get document names
        try:
            document_names = embedding_filter.get_document_names()
        except Exception as e:
            st.error(f"Error getting document names: {e}")
            document_names = []

        # Document selection
        col1, col2 = st.columns(2)

        with col1:
            # Default to all documents selected
            selected_docs = st.multiselect(
                "Filter by Documents (select none for all)",
                options=document_names,
                default=[],
                help="Select specific documents to include (leave empty to include all)"
            )

        with col2:
            # Parameters
            top_k = st.slider(
                "Number of results",
                min_value=10,
                max_value=50000,
                value=1000,
                step=10,
                help="Maximum number of chunks to retrieve"
            )

        # Run filter button
        submit_button = st.form_submit_button("Run Topic Filter", type="primary")

    # Store results in session state to preserve across page reloads
    if "topic_filter_results" not in st.session_state:
        st.session_state.topic_filter_results = None

    # Process form submission
    if submit_button and topic_query:
        with st.spinner("Filtering chunks by topic..."):
            # Determine included/excluded documents
            included_docs = set(selected_docs) if selected_docs else None

            # Run the filter
            results = embedding_filter.filter_by_topic(
                topic_query=topic_query,
                top_k=top_k,
                included_docs=included_docs,
                excluded_docs=None
            )

            # Store results in session state
            st.session_state.topic_filter_results = results

            if results:
                st.success(f"Found {len(results)} relevant chunks!")
            else:
                st.warning("No relevant chunks found. Try a different topic query or filter settings.")

    # Display results if available
    if st.session_state.topic_filter_results:
        results = st.session_state.topic_filter_results

        # Create a DataFrame for display
        import pandas as pd

        # Create a list of rows for the DataFrame
        rows = []
        for result in results:
            row = {
                'Score': f"{result['score']:.4f}",
                'Document': result['metadata'].get('file_name', 'Unknown'),
                'Page': result['metadata'].get('page_num', 'N/A'),
                'Text': result.get('original_text', result.get('text', ''))[:200] + "..."  # Truncate text for display
            }
            rows.append(row)

        # Create DataFrame
        df = pd.DataFrame(rows)

        # Show results count and export option
        col1, col2 = st.columns([3, 1])

        with col1:
            st.subheader(f"Results ({len(results)} chunks)")



        # Show results
        st.dataframe(
            df,
            use_container_width=True,
            column_config={
                "Score": st.column_config.NumberColumn("Score", format="%.4f", width="small"),
                "Document": st.column_config.TextColumn("Document", width="medium"),
                "Page": st.column_config.TextColumn("Page", width="small"),
                "Text": st.column_config.TextColumn("Text", width="large")
            }
        )

        # Text view of selected result
        st.subheader("Text View")
        selected_index = st.selectbox("Select chunk to view", options=range(len(results)))

        if selected_index is not None:
            selected_result = results[selected_index]

            # Show metadata
            meta = selected_result['metadata']
            st.markdown(f"**Document:** {meta.get('file_name', 'Unknown')}")
            st.markdown(f"**Page:** {meta.get('page_num', 'N/A')}")
            st.markdown(f"**Score:** {selected_result['score']:.4f}")

            # Show full text
            st.markdown("**Full Text:**")
            st.markdown(selected_result.get('original_text', selected_result.get('text', '')))


def render_info_extraction_page():
    """
    Render the information extraction page.
    """
    st.header("📊 Information Extraction")

    # Import the info extractor
    try:
        from src.core.extraction.info_extractor import InfoExtractor
        # Get query engine for document retrieval
        query_engine = get_or_create_query_engine()
    except ImportError as e:
        st.error(f"Information extraction module not available: {e}")
        st.warning("Make sure the info_extractor.py module is installed in the src/core/extraction directory.")
        return
    except Exception as e:
        st.error(f"Error initializing information extractor: {e}")
        return

    # Check if there's data to extract from
    collection_info = query_engine.get_collection_info()
    if not collection_info.get("exists", False) or collection_info.get("points_count", 0) == 0:
        st.warning("No indexed documents found. Please upload and process documents first.")
        return

    # Check Aphrodite service status
    if not APHRODITE_SERVICE_AVAILABLE:
        st.error("Aphrodite service module is not available. Cannot perform extraction.")
        return

    service = get_service()
    if not service.is_running():
        st.warning("LLM service is not running. Please start it from the sidebar.")
        if st.button("Start LLM Service Now"):
            start_aphrodite_service()
            st.rerun()
        return

    # Create tabs for schema definition and extraction
    schema_tab, extract_tab, results_tab = st.tabs(["Define Schema", "Extract Information", "View Results"])

    # Store schema in session state
    if "info_extraction_schema" not in st.session_state:
        st.session_state.info_extraction_schema = {
            "fields": [{"name": "entity", "type": "string", "description": "Name of the primary entity"}],
            "primary_key": "entity",
            "primary_key_description": "entities mentioned in the text",
            "user_query": "Extract all entities mentioned in the text"
        }

    # Store extraction results in session state
    if "info_extraction_results" not in st.session_state:
        st.session_state.info_extraction_results = None

    # Schema definition tab
    with schema_tab:
        st.subheader("Define Extraction Schema")

        # Description of what this does
        st.markdown("""
        This feature allows you to extract structured information from documents using a custom schema.
        Define the fields you want to extract, and the system will use AI to identify and organize this information.

        **Instructions:**
        1. Define your schema fields (name, type, description)
        2. Specify the primary key field (the main entity being extracted)
        3. Provide a description of what you're extracting
        4. Go to the Extract Information tab to run the extraction
        """)

        # Primary key field (simplified)
        primary_key_options = [field["name"] for field in st.session_state.info_extraction_schema["fields"]]
        # Check if current primary key exists in options, if not set to first field
        current_primary_key = st.session_state.info_extraction_schema["primary_key"]
        if current_primary_key not in primary_key_options and primary_key_options:
            current_primary_key = primary_key_options[0]
            st.session_state.info_extraction_schema["primary_key"] = current_primary_key

        primary_key_index = 0
        if primary_key_options:
            try:
                primary_key_index = primary_key_options.index(current_primary_key)
            except ValueError:
                pass

        # A single combined field for selecting the primary entity
        primary_key = st.selectbox(
            "Primary Entity Field",
            options=primary_key_options,
            index=primary_key_index,
            help="This field defines the main entity that each row in your results will represent (e.g., 'person', 'company', 'transaction')"
        )

        # We'll generate the description automatically from the field name
        # This will be used in the prompt (e.g., "entity" → "entities")
        primary_key_description = f"{primary_key}s" if primary_key else "entities"

        # User query
        user_query = st.text_area(
            "Extraction Query",
            value=st.session_state.info_extraction_schema["user_query"],
            help="Describe in detail what information you want to extract",
            height=100
        )

        # Schema fields
        st.subheader("Schema Fields")

        # Available field types
        field_types = ["string", "number", "integer", "boolean", "date"]

        # Create a container for the fields
        fields_container = st.container()

        # Add field button
        if st.button("Add Field"):
            st.session_state.info_extraction_schema["fields"].append({
                "name": f"field_{len(st.session_state.info_extraction_schema['fields'])}",
                "type": "string",
                "description": "Description of the field"
            })

        # Display and edit fields
        with fields_container:
            updated_fields = []

            for i, field in enumerate(st.session_state.info_extraction_schema["fields"]):
                col1, col2, col3, col4 = st.columns([2, 2, 5, 1])

                with col1:
                    field_name = st.text_input(
                        "Field Name",
                        value=field["name"],
                        key=f"field_name_{i}"
                    )

                with col2:
                    field_type = st.selectbox(
                        "Type",
                        options=field_types,
                        index=field_types.index(field["type"]) if field["type"] in field_types else 0,
                        key=f"field_type_{i}"
                    )

                with col3:
                    field_description = st.text_input(
                        "Description",
                        value=field["description"],
                        key=f"field_desc_{i}"
                    )

                with col4:
                    # Don't allow removing the last field
                    if len(st.session_state.info_extraction_schema["fields"]) > 1:
                        remove = st.button("🗑️", key=f"remove_{i}")
                    else:
                        remove = False

                if not remove:
                    updated_fields.append({
                        "name": field_name,
                        "type": field_type,
                        "description": field_description
                    })

            # Update fields in session state
            st.session_state.info_extraction_schema["fields"] = updated_fields

        # Save schema button
        if st.button("Save Schema", type="primary"):
            # Update primary key and description
            st.session_state.info_extraction_schema["primary_key"] = primary_key
            st.session_state.info_extraction_schema["primary_key_description"] = primary_key_description
            st.session_state.info_extraction_schema["user_query"] = user_query

            # Check field name uniqueness
            field_names = [field["name"] for field in st.session_state.info_extraction_schema["fields"]]
            if len(field_names) != len(set(field_names)):
                st.error("Field names must be unique!")
            else:
                st.success("Schema saved successfully!")

    # Extraction tab
    with extract_tab:
        st.subheader("Extract Information")

        # Show current schema
        with st.expander("Current Schema", expanded=True):
            # Display field information in a table
            field_data = []
            for field in st.session_state.info_extraction_schema["fields"]:
                field_data.append({
                    "Name": field["name"],
                    "Type": field["type"],
                    "Description": field["description"],
                    "Primary Key": "✓" if field["name"] == st.session_state.info_extraction_schema[
                        "primary_key"] else ""
                })

            st.dataframe(field_data)
            st.caption(f"Primary Key Description: {st.session_state.info_extraction_schema['primary_key_description']}")
            st.caption(f"Extraction Query: {st.session_state.info_extraction_schema['user_query']}")

        # Document selection for extraction
        st.subheader("Select Documents")

        # Fetch all document names
        try:
            max_chunks_for_names = CONFIG.get("extraction", {}).get("information_extraction", {}).get("max_chunks",
                                                                                                      1000)
            all_chunks = query_engine.get_chunks(limit=max_chunks_for_names)
            doc_names = sorted(list(set(c['metadata'].get('file_name', 'Unknown')
                                        for c in all_chunks
                                        if c['metadata'].get('file_name'))))
        except Exception as e:
            logger.error(f"Failed to get document names: {e}")
            doc_names = []

        # Allow user to select documents
        selected_docs = st.multiselect(
            "Select documents to extract from",
            options=doc_names,
            default=[],
            help="Leave empty to extract from all documents"
        )

        # Model selection
        extraction_model_options = {
            "Small Text (Faster)": CONFIG["models"]["extraction_models"]["text_small"],
            "Standard Text": CONFIG["models"]["extraction_models"]["text_standard"],
        }

        selected_model_display_name = st.selectbox(
            "Select LLM for Extraction",
            options=list(extraction_model_options.keys()),
            index=1  # Default to Standard
        )

        # Get the actual model name
        selected_model_name = extraction_model_options[selected_model_display_name]

        # Run extraction button
        extract_btn = st.button(
            "Run Extraction",
            type="primary",
            help="Extract information from selected documents"
        )

        # Show extraction progress
        extraction_progress = st.empty()

        # Run extraction when button is clicked
        if extract_btn:
            # Create info extractor with selected model
            info_extractor = InfoExtractor(model_name=selected_model_name)

            # Fetch chunks
            # Fetch chunks
            # Fetch chunks
            with extraction_progress.container():
                with st.spinner("Fetching document chunks..."):
                    # Get the max chunks limit from config
                    max_chunks = CONFIG.get("extraction", {}).get("information_extraction", {}).get("max_chunks", 10000)

                    # Process based on document selection
                    if selected_docs:
                        # Get chunks from selected documents only
                        chunks = []
                        for doc_name in selected_docs:
                            doc_chunks = query_engine.get_chunks(limit=max_chunks // len(selected_docs),
                                                                 document_filter=doc_name)
                            chunks.extend(doc_chunks)
                            # Limit total chunks
                            if len(chunks) >= max_chunks:
                                st.warning(
                                    f"Reached maximum chunk limit ({max_chunks}). Some documents may be partially processed.")
                                chunks = chunks[:max_chunks]
                                break
                    else:
                        # Get all chunks when no documents are selected
                        chunks = query_engine.get_chunks(limit=max_chunks)

                    st.info(
                        f"Processing {len(chunks)} chunks from {len(selected_docs) if selected_docs else 'all'} documents")

            # Prepare schema for extraction
            schema_dict = {
                field["name"]: {
                    "type": field["type"],
                    "description": field["description"]
                }
                for field in st.session_state.info_extraction_schema["fields"]
            }

            # Run extraction
            with extraction_progress.container():
                with st.spinner("Extracting information..."):
                    # Run extraction
                    results = info_extractor.extract_information(
                        chunks=chunks,
                        schema_dict=schema_dict,
                        primary_key_field=st.session_state.info_extraction_schema["primary_key"],
                        primary_key_description=st.session_state.info_extraction_schema["primary_key_description"],
                        user_query=st.session_state.info_extraction_schema["user_query"]
                    )

                    # Store results in session state
                    st.session_state.info_extraction_results = results

                    if results:
                        st.success(f"Successfully extracted {len(results)} items!")
                    else:
                        st.warning("No information was extracted. Try adjusting your schema or query.")

    # Results tab
    # Results tab
    # In the Results tab of the render_info_extraction_page function

    # Results tab
    with results_tab:
        st.subheader("Extraction Results")

        # Check if results exist
        if not st.session_state.info_extraction_results:
            st.info("No extraction results available. Run an extraction first.")
            return

        # Display results
        results = st.session_state.info_extraction_results

        # Create flattened data for display
        flattened_rows = []

        for item in results:
            # Extract the main content fields
            row_data = {k: v for k, v in item.items() if k != '_source'}

            # Add source fields with prefixes
            if '_source' in item and isinstance(item['_source'], dict):
                for src_key, src_value in item['_source'].items():
                    row_data[f'source_{src_key}'] = src_value

            flattened_rows.append(row_data)

        # Create DataFrame for display
        if flattened_rows:
            df = pd.DataFrame(flattened_rows)

            # Reorder columns to put source fields at the end
            source_cols = [col for col in df.columns if col.startswith('source_')]
            other_cols = [col for col in df.columns if not col.startswith('source_')]

            if other_cols and source_cols:
                df = df[other_cols + source_cols]
        else:
            df = pd.DataFrame()

        # Show statistics
        unique_docs = set()
        for item in results:
            if '_source' in item and isinstance(item['_source'], dict) and 'file_name' in item['_source']:
                unique_docs.add(item['_source']['file_name'])

        st.info(f"Found {len(results)} items from {len(unique_docs)} documents")

        # Export options
        col1, col2 = st.columns([1, 3])

        with col1:
            # Export to CSV
            if st.button("Export to CSV"):
                # Create filename
                export_time = int(time.time())
                export_path = str(ROOT_DIR / "temp" / f"extraction_results_{export_time}.csv")

                # Export
                info_extractor = InfoExtractor()
                success = info_extractor.export_to_csv(results, export_path)

                if success:
                    # Create download link
                    with open(export_path, "r") as f:
                        csv_data = f.read()

                    import base64
                    b64 = base64.b64encode(csv_data.encode()).decode()
                    href = f'<a href="data:text/csv;base64,{b64}" download="extraction_results.csv">Download CSV</a>'
                    st.markdown(href, unsafe_allow_html=True)
                else:
                    st.error("Failed to export results")

        # Display results table
        if not df.empty:
            st.dataframe(df)
        else:
            st.warning("No data to display. Extraction results might be empty or malformed.")


def render_classification_page():
    """
    Render the document classification page.
    """
    st.header("🏷️ Document Classification")

    # Import the document classifier
    try:
        from src.core.classification.document_classifier import DocumentClassifier
        # Get query engine for document retrieval
        query_engine = get_or_create_query_engine()
    except ImportError as e:
        st.error(f"Document classification module not available: {e}")
        st.warning("Make sure the document_classifier.py module is installed in the src/core/classification directory.")
        return
    except Exception as e:
        st.error(f"Error initializing document classifier: {e}")
        return

    # Check if there's data to classify
    collection_info = query_engine.get_collection_info()
    if not collection_info.get("exists", False) or collection_info.get("points_count", 0) == 0:
        st.warning("No indexed documents found. Please upload and process documents first.")
        return

    # Check Aphrodite service status
    if not APHRODITE_SERVICE_AVAILABLE:
        st.error("Aphrodite service module is not available. Cannot perform classification.")
        return

    service = get_service()
    if not service.is_running():
        st.warning("LLM service is not running. Please start it from the sidebar.")
        if st.button("Start LLM Service Now"):
            start_aphrodite_service()
            st.rerun()
        return

    # Create tabs for schema definition and classification
    schema_tab, classify_tab, results_tab = st.tabs(["Define Schema", "Classify Documents", "View Results"])

    # Store schema in session state
    if "classification_schema" not in st.session_state:
        st.session_state.classification_schema = {
            "fields": [{"name": "category", "type": "string", "values": ["General", "Legal", "Financial", "Personal"],
                        "description": "Document category"}],
            "multi_label_fields": ["category"],
            "user_instructions": "Classify the document by its main topic"
        }

    # Store classification results in session state
    if "classification_results" not in st.session_state:
        st.session_state.classification_results = None

    # Schema definition tab
    with schema_tab:
        st.subheader("Define Classification Schema")

        # Description of what this does
        st.markdown("""
        This feature allows you to classify documents using custom categories.
        Define the fields and values you want to use for classification, and the system will
        use AI to categorize your documents according to your schema.

        **Instructions:**
        1. Define your classification fields, allowed values, and descriptions
        2. Specify if each field should support multiple values
        3. Provide instructions for the classification
        4. Go to the Classify Documents tab to run the classification
        """)

        # Schema fields
        st.subheader("Classification Fields")

        # Create a container for the fields
        fields_container = st.container()

        # Add field button
        if st.button("Add Field"):
            st.session_state.classification_schema["fields"].append({
                "name": f"field_{len(st.session_state.classification_schema['fields'])}",
                "type": "string",
                "values": ["Value1", "Value2", "Value3"],
                "description": "Description of the field"
            })

        # Display and edit fields
        with fields_container:
            updated_fields = []
            updated_multi_label = []

            for i, field in enumerate(st.session_state.classification_schema["fields"]):
                st.markdown(f"### Field {i + 1}")
                col1, col2 = st.columns(2)

                with col1:
                    field_name = st.text_input(
                        "Field Name",
                        value=field["name"],
                        key=f"field_name_{i}"
                    )

                with col2:
                    field_description = st.text_input(
                        "Description",
                        value=field["description"],
                        key=f"field_desc_{i}"
                    )

                # Allowed values as a text area for easier editing
                current_values = field.get("values", [])
                values_str = ", ".join(current_values)
                values_input = st.text_area(
                    "Allowed Values (comma-separated)",
                    value=values_str,
                    key=f"field_values_{i}",
                    help="Enter allowed values separated by commas"
                )

                # Parse values from input
                values_list = [v.strip() for v in values_input.split(",") if v.strip()]

                # Multi-label option
                is_multi_label = st.checkbox(
                    "Allow multiple values",
                    value=field["name"] in st.session_state.classification_schema.get("multi_label_fields", []),
                    key=f"field_multi_{i}",
                    help="Check this if multiple values can be assigned to this field"
                )

                # Option to remove field
                remove = st.button("Remove Field", key=f"remove_{i}")

                if not remove:
                    updated_field = {
                        "name": field_name,
                        "type": "string",  # Always string for classification
                        "values": values_list,
                        "description": field_description
                    }
                    updated_fields.append(updated_field)

                    if is_multi_label:
                        updated_multi_label.append(field_name)

                st.markdown("---")

        # User instructions for classification
        st.subheader("Classification Instructions")
        user_instructions = st.text_area(
            "Instructions",
            value=st.session_state.classification_schema.get("user_instructions", ""),
            height=100,
            help="Provide specific instructions for how to classify documents"
        )

        # Save schema button
        if st.button("Save Schema", type="primary"):
            # Update schema in session state
            st.session_state.classification_schema["fields"] = updated_fields
            st.session_state.classification_schema["multi_label_fields"] = updated_multi_label
            st.session_state.classification_schema["user_instructions"] = user_instructions

            # Check field name uniqueness
            field_names = [field["name"] for field in updated_fields]
            if len(field_names) != len(set(field_names)):
                st.error("Field names must be unique!")
            else:
                st.success("Schema saved successfully!")

                # Generate example output
                example = {}
                for field in updated_fields:
                    field_name = field["name"]
                    values = field.get("values", [])
                    if values:
                        if field_name in updated_multi_label:
                            # Multi-label example with 1-2 values
                            num_values = min(2, len(values))
                            example[field_name] = random.sample(values, num_values) if num_values > 0 else []
                        else:
                            # Single label example
                            example[field_name] = values[0] if values else ""

                # Show the example output
                st.subheader("Example Output")
                st.json(example)

    # Classification tab
    with classify_tab:
        st.subheader("Classify Documents")

        # Show current schema
        with st.expander("Current Schema", expanded=True):
            # Display field information in a table
            field_data = []
            for field in st.session_state.classification_schema["fields"]:
                field_data.append({
                    "Name": field["name"],
                    "Description": field["description"],
                    "Allowed Values": ", ".join(field.get("values", [])),
                    "Multi-label": "✓" if field["name"] in st.session_state.classification_schema.get(
                        "multi_label_fields", []) else ""
                })

            st.dataframe(field_data)
            st.caption(f"Instructions: {st.session_state.classification_schema.get('user_instructions', '')}")

        # Document selection for classification
        st.subheader("Select Documents")

        # Fetch all document names
        try:
            # Get the max documents limit from config
            max_chunks_for_names = CONFIG.get("classification", {}).get("max_chunks_for_listing", 1000)
            all_chunks = query_engine.get_chunks(limit=max_chunks_for_names)
            doc_names = sorted(list(set(c['metadata'].get('file_name', 'Unknown')
                                        for c in all_chunks
                                        if c['metadata'].get('file_name'))))
        except Exception as e:
            logger.error(f"Failed to get document names: {e}")
            doc_names = []

        # Allow user to select documents
        selected_docs = st.multiselect(
            "Select documents to classify",
            options=doc_names,
            default=[],
            help="Leave empty to classify all documents"
        )

        # Model selection
        classification_model_options = {
            "Small Text (Faster)": CONFIG["models"]["extraction_models"]["text_small"],
            "Standard Text": CONFIG["models"]["extraction_models"]["text_standard"],
        }

        selected_model_display_name = st.selectbox(
            "Select LLM for Classification",
            options=list(classification_model_options.keys()),
            index=1  # Default to Standard
        )

        # Get the actual model name
        selected_model_name = classification_model_options[selected_model_display_name]

        # Batch size setting for classification
        batch_size = st.slider(
            "Max chunks to classify",
            min_value=10,
            max_value=10000,
            value=1000,
            step=100,
            help="Maximum number of chunks to classify"
        )

        # Run classification button
        classify_btn = st.button(
            "Run Classification",
            type="primary",
            help="Classify selected documents according to schema"
        )

        # Show classification progress
        classification_progress = st.empty()

        # Run classification when button is clicked
        if classify_btn:
            # Validate schema
            if not st.session_state.classification_schema["fields"]:
                st.error("No classification fields defined. Please define your schema first.")
                return

            # Check if all fields have values
            invalid_fields = []
            for field in st.session_state.classification_schema["fields"]:
                if not field.get("values", []):
                    invalid_fields.append(field["name"])

            if invalid_fields:
                st.error(
                    f"The following fields have no values defined: {', '.join(invalid_fields)}. Please add values to all fields.")
                return

            # Create document classifier with selected model
            document_classifier = DocumentClassifier(model_name=selected_model_name)

            # Get chunks to classify
            with classification_progress.container():
                with st.spinner("Fetching document chunks..."):
                    # Process based on document selection
                    if selected_docs:
                        # Get chunks from selected documents only
                        chunks = []
                        for doc_name in selected_docs:
                            doc_chunks = query_engine.get_chunks(limit=batch_size // len(selected_docs),
                                                                 document_filter=doc_name)
                            chunks.extend(doc_chunks)
                            # Limit total chunks
                            if len(chunks) >= batch_size:
                                st.warning(
                                    f"Reached maximum chunk limit ({batch_size}). Some documents may be partially processed.")
                                chunks = chunks[:batch_size]
                                break
                    else:
                        # Get all chunks when no documents are selected
                        chunks = query_engine.get_chunks(limit=batch_size)

                    st.info(
                        f"Processing {len(chunks)} chunks from {len(selected_docs) if selected_docs else 'all'} documents")

            # Prepare schema for classification
            schema_dict = {}
            multi_label_fields = set(st.session_state.classification_schema.get("multi_label_fields", []))

            for field in st.session_state.classification_schema["fields"]:
                schema_dict[field["name"]] = {
                    "type": field["type"],
                    "description": field["description"],
                    "values": field.get("values", [])
                }

            # Run classification
            with classification_progress.container():
                with st.spinner("Classifying documents..."):
                    # Run classification
                    results = document_classifier.classify_documents(
                        chunks=chunks,
                        schema=schema_dict,
                        multi_label_fields=multi_label_fields,
                        user_instructions=st.session_state.classification_schema.get("user_instructions", "")
                    )

                    # Store results in session state
                    st.session_state.classification_results = results

                    if results:
                        st.success(f"Successfully classified {len(results)} chunks!")
                    else:
                        st.warning("No documents were classified. Check logs for errors.")

    # Results tab
    with results_tab:
        st.subheader("Classification Results")

        # Check if results exist
        if not st.session_state.classification_results:
            st.info("No classification results available. Run a classification first.")
            return

        # Display results
        results = st.session_state.classification_results

        # Create DataFrame for display
        if results:
            # Extract all classification fields
            class_fields = []
            for result in results:
                if "classification" in result:
                    class_fields.extend(result["classification"].keys())
            class_fields = sorted(list(set(class_fields)))

            # Create data for display
            display_data = []
            for result in results:
                row = {
                    "Document": result.get("file_name", "Unknown"),
                    "Page": result.get("page_num", "N/A"),
                    "Text": result.get("text", "")[:200] + "..." if len(result.get("text", "")) > 200 else result.get(
                        "text", ""),
                }

                # Add classification fields
                classification = result.get("classification", {})
                for field in class_fields:
                    if field in classification:
                        value = classification[field]
                        if isinstance(value, list):
                            row[field] = ", ".join(value)
                        else:
                            row[field] = value
                    else:
                        row[field] = ""

                display_data.append(row)

            df = pd.DataFrame(display_data)

            # Show statistics
            unique_docs = set()
            for result in results:
                if "file_name" in result and result["file_name"]:
                    unique_docs.add(result["file_name"])

            st.info(f"Found {len(results)} classified chunks from {len(unique_docs)} documents")

            # Export options
            col1, col2 = st.columns([1, 3])

            with col1:
                # Export to CSV
                if st.button("Export to CSV"):
                    # Create filename
                    export_time = int(time.time())
                    export_path = str(ROOT_DIR / "temp" / f"classification_results_{export_time}.csv")

                    # Export
                    document_classifier = DocumentClassifier()
                    success = document_classifier.export_to_csv(results, export_path)

                    if success:
                        # Create download link
                        with open(export_path, "r") as f:
                            csv_data = f.read()

                        import base64
                        b64 = base64.b64encode(csv_data.encode()).decode()
                        href = f'<a href="data:text/csv;base64,{b64}" download="classification_results.csv">Download CSV</a>'
                        st.markdown(href, unsafe_allow_html=True)
                    else:
                        st.error("Failed to export results")

            # Display results table
            if not df.empty:
                st.dataframe(df)
            else:
                st.warning("No data to display. Classification results might be empty or malformed.")

            # Show detailed view of a selected result
            st.subheader("Detailed View")

            if results:
                # Select a result to view
                result_index = st.selectbox("Select result to view", range(len(results)))
                selected_result = results[result_index]

                # Display full text
                st.markdown("**Full Text:**")
                st.text(selected_result.get("text", ""))

                # Display classification
                st.markdown("**Classification:**")
                st.json(selected_result.get("classification", {}))

def main():
    """
    Main application entry point.
    """
    # Initialize app state and handle process restoration *first*
    initialize_app()

    # Apply custom styling
    apply_custom_styling()

    # Render header
    render_header()

    # Render sidebar and get selected page
    page = render_sidebar()

    # Render selected page
    if page == "Upload & Process":
        render_upload_page()
    elif page == "Explore Data":
        render_explore_page()
    elif page == "Query System":
        render_query_page()
    elif page == "Cluster Map":
        render_cluster_map_page()
    elif page == "Topic Filter":
        render_topic_filter_page()
    elif page == "Information Extraction":
        render_info_extraction_page()
    elif page == "Document Classification":
        render_classification_page()
    elif page == "Settings":
        render_settings_page()

if __name__ == "__main__":
    # Ensure multiprocessing start method is set globally before any processes might be spawned
    # It's generally recommended to set this at the entry point of the main script.
    try:
        import multiprocessing
        # Check if the start method is already set, potentially by the service module
        current_method = multiprocessing.get_start_method(allow_none=True)
        if current_method != 'spawn':
             logger.info(f"Setting multiprocessing start method to 'spawn'. Current method: {current_method}")
             # Force=True can be risky if other libraries expect a different method
             multiprocessing.set_start_method('spawn', force=True)
        else:
             logger.info("Multiprocessing start method already set to 'spawn'.")
    except Exception as e:
         # Log error but proceed; default method might work or might cause issues later.
         logger.error(f"Could not set multiprocessing start method to 'spawn': {e}. Using default: {multiprocessing.get_start_method(allow_none=True)}")


    main()