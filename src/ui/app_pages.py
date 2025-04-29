# --- START OF REWRITTEN app_pages.py ---

# app_pages.py
import streamlit as st
import pandas as pd
import json
import time
import base64
import io
import yaml
from datetime import datetime
import ast # For cluster map download helper
from typing import Dict, Type, List, Optional, Set, Union, Any # Import Type for Pydantic models

# Import necessary functions/variables from other modules
from app_setup import (
    ROOT_DIR, CONFIG, logger, EXTRACTED_DATA_PATH,
    get_or_create_query_engine, get_conversation_store,
    APHRODITE_SERVICE_AVAILABLE, IS_OPENROUTER_ACTIVE, IS_GEMINI_ACTIVE, LLM_BACKEND # Import backend flags
)
from app_processing import process_documents_with_spreadsheet_options # For upload page

# Load configuration
CONFIG_PATH = ROOT_DIR / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

# Import chat functions conditionally
if IS_OPENROUTER_ACTIVE or IS_GEMINI_ACTIVE: # API Backends
    from app_chat import handle_chat_message, save_current_conversation # API backends use the same handler
    # Define dummy functions for Aphrodite service control if API backend is active
    def start_aphrodite_service(): st.error("Aphrodite backend not active."); return False
    def terminate_aphrodite_service(): st.error("Aphrodite backend not active."); return False
    logger.info(f"Imported {LLM_BACKEND.upper()} chat handler.")
elif LLM_BACKEND == "aphrodite" and APHRODITE_SERVICE_AVAILABLE: # Aphrodite Backend
    from app_chat import handle_chat_message, start_aphrodite_service, terminate_aphrodite_service, save_current_conversation
    logger.info("Imported Aphrodite chat handlers.")
else:
    # Define dummy functions if no backend is available
    def handle_chat_message(prompt): st.error("No LLM backend available for chat.")
    def start_aphrodite_service(): st.error("Aphrodite backend not available."); return False
    def terminate_aphrodite_service(): st.error("Aphrodite backend not available."); return False
    logger.warning("No chat handlers imported as no backend is available.")

from app_visuals import render_network_overview_tab, render_connection_explorer_tab, render_entity_centered_tab, render_network_metrics_tab # For explore page

# Import core modules needed by pages conditionally
cluster_map_available = False
try:
    # Use the renamed llm_representation module
    from src.core.visualization.llm_representation import LLMRepresentation
    from src.core.visualization.cluster_map import create_download_dataframe, check_dependencies, generate_cluster_map
    cluster_map_available = True
except ImportError as e:
    logger.warning(f"Cluster map module or LLMRepresentation not fully available: {e}. Cluster map page may be limited.")
    def create_download_dataframe(docs_df, topic_info): return pd.DataFrame()
    def check_dependencies(): return False
    def generate_cluster_map(query_engine, include_outliers): return None, "Cluster map module not loaded."
    LLMRepresentation = None # Placeholder if import fails

embedding_filter_available = False
try:
    from src.core.filtering.embedding_filter import EmbeddingFilter
    embedding_filter_available = True
except ImportError as e:
    logger.warning(f"Embedding filter module not available: {e}. Topic filter page disabled.")
    EmbeddingFilter = None

info_extractor_available = False
try:
    from src.core.extraction.info_extractor import InfoExtractor
    info_extractor_available = True
except ImportError as e:
    logger.warning(f"Info extractor module not available: {e}. Info extraction page disabled.")
    InfoExtractor = None

doc_classifier_available = False
try:
    from src.core.classification.document_classifier import DocumentClassifier
    doc_classifier_available = True
except ImportError as e:
    logger.warning(f"Document classifier module not available: {e}. Classification page disabled.")
    DocumentClassifier = None

# Import Pydantic for dynamic models (needed here for helper functions)
try:
    from pydantic import BaseModel, Field, create_model
    # Import Literal from typing for broader compatibility
    from typing import Literal
    from pydantic_core import PydanticUndefined
    PYDANTIC_AVAILABLE_PAGES = True
except ImportError:
    BaseModel = None
    Field = None
    create_model = None
    Literal = None
    List = None
    Optional = None
    PydanticUndefined = None
    PYDANTIC_AVAILABLE_PAGES = False
    logger.error("Pydantic not available in app_pages. Dynamic schema creation will fail.")

# --- Helper Functions for Dynamic Pydantic Models ---
# (Moved here from InfoExtractor/Classifier to be accessible before instantiation)

def _get_python_type_for_dynamic(type_str: str) -> Type:
    """Convert string type name to Python type for Pydantic."""
    type_map = {
        "string": str, "number": float, "integer": int,
        "boolean": bool, "date": str, "array": list, "object": dict
    }
    return type_map.get(type_str.lower(), str)

def _create_dynamic_info_pydantic_model(schema_dict: Dict[str, Any]) -> Optional[Type[BaseModel]]:
    """Dynamically creates List[Item] Pydantic model for Info Extraction."""
    if not PYDANTIC_AVAILABLE_PAGES or not schema_dict: return None
    try:
        fields = {}
        for field_name, field_info in schema_dict.items():
            field_type = _get_python_type_for_dynamic(field_info.get("type", "string"))
            description = field_info.get("description", "")
            fields[field_name] = (Optional[field_type], Field(description=description, default=None))
        DynamicItemModel = create_model("DynamicInfoItemModel", **fields)
        # Define the list model structure as required by Gemini example
        DynamicListModel = create_model(
            "DynamicInfoListModel",
            items=(List[DynamicItemModel], Field(..., description="List of extracted information items"))
        )
        return DynamicListModel
    except Exception as e:
        logger.error(f"Error creating dynamic info Pydantic model: {e}", exc_info=True)
        return None

def _create_dynamic_classification_pydantic_model(schema: Dict[str, Any], multi_label_fields: set) -> Optional[Type[BaseModel]]:
    """Dynamically creates Pydantic model for Classification."""
    if not PYDANTIC_AVAILABLE_PAGES or not schema: return None
    try:
        fields = {}
        for field_name, field_info in schema.items():
            allowed_values = field_info.get("values", [])
            description = field_info.get("description", "")
            if not allowed_values: continue
            is_multi_label = field_name in multi_label_fields
            # Ensure all values are strings for Literal
            str_values = tuple(str(v) for v in allowed_values)
            if not str_values: continue

            # Use Literal from typing
            from typing import Literal as TypingLiteral

            if is_multi_label:
                field_type = Optional[List[TypingLiteral[str_values]]]
            else:
                field_type = Optional[TypingLiteral[str_values]]
            fields[field_name] = (field_type, Field(description=description, default=None))

        if not fields: return None
        DynamicClassificationModel = create_model("DynamicClassificationModel", **fields)
        return DynamicClassificationModel
    except Exception as e:
        logger.error(f"Error creating dynamic classification Pydantic model: {e}", exc_info=True)
        return None


# --- Page: Upload & Process ---

def render_spreadsheet_options(uploaded_files):
    """ Render simplified options for processing spreadsheet files. """
    if not uploaded_files: return {}
    spreadsheet_files = [f for f in uploaded_files if f.name.lower().endswith(('.csv', '.xlsx', '.xls'))]
    if not spreadsheet_files: return {}
    st.subheader("Spreadsheet Processing Options")
    st.info("For each spreadsheet, select columns to include. Each row becomes one chunk.")
    column_selections = {}
    for file in spreadsheet_files:
        with st.expander(f"Configure {file.name}", expanded=True):
            try:
                # Create temp dir if it doesn't exist
                temp_dir = ROOT_DIR / "temp"
                temp_dir.mkdir(exist_ok=True)
                temp_file_path = temp_dir / f"preview_{file.name}"
                with open(temp_file_path, "wb") as f: f.write(file.getbuffer())
                if file.name.lower().endswith('.csv'): df = pd.read_csv(temp_file_path, nrows=5)
                else: df = pd.read_excel(temp_file_path, nrows=5)
                columns = df.columns.tolist()
                st.markdown("#### Data Preview"); st.dataframe(df.head(3), use_container_width=True)
                st.markdown("#### Column Selection"); st.markdown("Select columns. Each row becomes: `Col1Name: Value1 | Col2Name: Value2`")
                select_all = st.checkbox(f"Select All Columns for {file.name}", value=True, key=f"select_all_{file.name}")
                if select_all: selected_columns = columns; st.info(f"All {len(columns)} columns selected.")
                else: selected_columns = st.multiselect(f"Select columns for {file.name}", options=columns, default=[], key=f"columns_{file.name}")
                column_selections[file.name] = selected_columns
                if temp_file_path.exists(): temp_file_path.unlink()
            except Exception as e:
                st.error(f"Error reading preview for {file.name}: {e}")
                column_selections[file.name] = []
                if 'temp_file_path' in locals() and temp_file_path.exists(): temp_file_path.unlink()
    return column_selections

def render_upload_page():
    """ Render the upload and processing page. """
    st.header("📄 Upload & Process Documents")

    uploaded_files = st.file_uploader(
        "Upload documents (PDF, DOCX, TXT, CSV, XLSX)",
        type=["pdf", "docx", "txt", "csv", "xlsx"],
        accept_multiple_files=True
    )

    st.subheader("Processing Options")

    # Model selection (Primarily affects Aphrodite backend)
    model_help_text = "Selects the LLM used for entity/relationship extraction during processing. "
    llm_select_disabled = False
    default_model_display_name = "N/A"
    model_name_map = {}
    selected_llm_name = "N/A" # Initialize

    if IS_OPENROUTER_ACTIVE:
        model_help_text += " (Note: OpenRouter uses models defined in config for extraction)."
        selected_llm_name = CONFIG.get('openrouter',{}).get('extraction_model','N/A')
        st.info(f"OpenRouter backend active. Extraction model configured as: `{selected_llm_name}`")
        llm_select_disabled = True
    elif IS_GEMINI_ACTIVE:
        model_help_text += " (Note: Gemini uses models defined in config for extraction)."
        # Use the specific model from the example
        selected_llm_name = CONFIG.get('gemini',{}).get('extraction_model','gemini-2.5-pro-exp-03-25')
        st.info(f"Gemini backend active. Extraction model configured as: `{selected_llm_name}`")
        llm_select_disabled = True
    else: # Aphrodite
        model_help_text += " Also sets the default model for the Query System if Aphrodite is active."
        extraction_model_options = {
            "Small Text (Faster)": CONFIG["models"]["extraction_models"]["text_small"],
            "Standard Text": CONFIG["models"]["extraction_models"]["text_standard"],
            "Large Text": CONFIG["models"]["extraction_models"]["text_large"],
        }
        default_model_display_name = "Small Text (Faster)" # Aphrodite default
        stored_model_name = st.session_state.get("selected_llm_model_name") # Model used in last processing run
        if stored_model_name:
            for name, model_val in extraction_model_options.items():
                if model_val == stored_model_name: default_model_display_name = name; break
        model_name_map = {display: actual for display, actual in extraction_model_options.items()}
        selected_model_display_name = st.selectbox(
            "Select LLM for Processing",
            options=list(model_name_map.keys()),
            index=list(model_name_map.keys()).index(default_model_display_name),
            key="upload_llm_select",
            help=model_help_text,
            disabled=llm_select_disabled
        )
        selected_llm_name = model_name_map[selected_model_display_name]

    # Visual processing options (Unchanged)
    use_visual_processing = st.checkbox("Enable Visual Processing (Experimental for PDFs)", key="upload_visual_cb")
    vl_page_numbers_str = ""
    if use_visual_processing:
        vl_page_numbers_str = st.text_input("Specify Pages for Visual Processing", placeholder="e.g., 1, 3-5 (leave empty for all)", key="upload_visual_pages")

    # Spreadsheet Options (Unchanged)
    spreadsheet_options = {}
    if uploaded_files and any(f.name.lower().endswith(('.csv', '.xlsx', '.xls')) for f in uploaded_files):
        spreadsheet_options = render_spreadsheet_options(uploaded_files)

    # Processing button
    process_btn = st.button("Process Documents", disabled=st.session_state.get("processing", False) or not uploaded_files, type="primary", key="upload_process_btn")

    # Show processing status (Unchanged)
    if st.session_state.get("processing", False):
        st.markdown("### Processing Status")
        status_container = st.empty(); progress_bar = st.progress(st.session_state.get("processing_progress", 0.0))
        status_container.info(st.session_state.get("processing_status", "Initializing..."))

    # Handle process button click
    if process_btn and uploaded_files:
        vl_pages = []; vl_process_all = False
        if use_visual_processing:
            if not vl_page_numbers_str.strip(): vl_process_all = True
            else:
                try:
                    for part in vl_page_numbers_str.split(','):
                        part = part.strip()
                        if '-' in part: start, end = map(int, part.split('-')); vl_pages.extend(range(start, end + 1))
                        elif part: vl_pages.append(int(part))
                    vl_pages = sorted(list(set(vl_pages)))
                except ValueError as e: st.error(f"Invalid page numbers format: '{vl_page_numbers_str}'. Error: {e}"); return

        st.session_state.processing = True
        st.session_state.processing_status = "Starting document processing..."
        st.session_state.processing_progress = 0.0
        # Store the model name *intended* for processing (might differ based on backend)
        st.session_state.selected_llm_model_name = selected_llm_name
        logger.info(f"Starting processing with LLM: {selected_llm_name} (Backend: {LLM_BACKEND.upper()})")
        if spreadsheet_options: logger.info(f"Processing with spreadsheet options: {spreadsheet_options}")

        # Call the processing function (it uses the active LLM manager internally)
        process_documents_with_spreadsheet_options(
            uploaded_files, selected_llm_name, vl_pages, vl_process_all, spreadsheet_options
        )
        st.rerun()


# --- Page: Explore Data ---

def render_document_explorer():
    """ Render the document and chunk explorer. """
    st.subheader("Document Explorer")
    query_engine = get_or_create_query_engine()
    if not query_engine: st.warning("Query Engine not available."); return
    col1, col2 = st.columns([3, 1])
    with col1: search_text = st.text_input("Search within chunks", placeholder="Enter search terms...", key="doc_search")
    with col2:
         doc_names = ["All Documents"]
         try:
             # Increase limit slightly to get more doc names if needed
             all_chunks_for_filter = query_engine.get_chunks(limit=2000)
             unique_names = sorted(list(set(c['metadata'].get('file_name', 'Unknown') for c in all_chunks_for_filter if c['metadata'].get('file_name'))))
             doc_names.extend(unique_names)
         except Exception as e: logger.error(f"Failed to get document names: {e}"); doc_names.append("Error loading names")
         doc_filter = st.selectbox("Filter by document", options=doc_names, key="doc_filter")
    doc_filter_value = doc_filter if doc_filter != "All Documents" else None
    try: chunks = query_engine.get_chunks(limit=100, search_text=search_text if search_text else None, document_filter=doc_filter_value)
    except Exception as e: st.error(f"Error retrieving chunks: {e}"); chunks = []
    if not chunks: st.info("No documents or chunks match the current filter/search criteria."); return
    st.markdown(f"Displaying **{min(len(chunks), 100)}** chunks:")
    for i, chunk in enumerate(chunks[:100]):
        meta = chunk.get('metadata', {}); chunk_id_display = meta.get('chunk_id', chunk.get('id', f'chunk_{i}'))
        file_name_display = meta.get('file_name', 'Unknown'); expander_title = f"Chunk {chunk_id_display} - {file_name_display}"
        with st.expander(expander_title):
            page_num_display = meta.get('page_num', 'N/A'); row_idx_display = meta.get('row_idx', None)
            st.markdown("##### Original Text"); original_text = chunk.get('original_text', chunk.get('text', ''))
            st.markdown(f"> {original_text}" if original_text else "_No text content available_")
            location_info = f"Row: {row_idx_display}" if row_idx_display is not None else f"Page: {page_num_display}"
            st.caption(f"Document: {file_name_display} | {location_info}")
            st.markdown("---"); st.markdown("##### Extracted Metadata")
            summary = meta.get('extracted_summary', 'Not available'); red_flags = meta.get('extracted_red_flags', 'None detected')
            types_list = meta.get('extracted_entity_types', []); types_text = ", ".join(types_list) if types_list else 'None detected'
            st.caption(f"**Summary:** {summary}"); st.caption(f"**Red Flags:** {red_flags}"); st.caption(f"**Entity Types:** {types_text}")

def render_entity_explorer():
    """ Render the entity explorer with tables of extracted entities. """
    st.subheader("Entity Explorer")
    entities_file = EXTRACTED_DATA_PATH / "entities.json"
    if not entities_file.exists(): st.info("No entities found. Process documents first."); return
    try:
        with open(entities_file, "r", encoding='utf-8') as f: entities = json.load(f)
    except Exception as e: st.error(f"Error loading entities: {e}"); return
    if not entities: st.info("Entities file exists but is empty."); return
    entity_types = sorted(list(set(entity.get("type", "Unknown") for entity in entities)))
    selected_types = st.multiselect("Filter by entity type", options=entity_types, default=entity_types, key="ent_type_filter")
    search_term = st.text_input("Search entities by name", placeholder="Enter search terms...", key="ent_search")
    filtered_entities = [entity for entity in entities if entity.get("type", "Unknown") in selected_types and (not search_term or search_term.lower() in entity.get("name", "").lower())]
    if filtered_entities:
        entity_data = []
        for entity in filtered_entities:
            context_info = entity.get("context", {}); page_num = context_info.get("page_number", "N/A"); chunk_ids = ", ".join(context_info.get("chunk_ids", []))
            entity_data.append({"Name": entity.get("name", ""), "Type": entity.get("type", "Unknown"), "Source Document": entity.get("source_document", "Unknown"), "Page": page_num, "Chunk IDs": chunk_ids})
        st.dataframe(entity_data, use_container_width=True, hide_index=True)
        st.caption(f"Displaying {len(filtered_entities)} of {len(entities)} total entities.")
    else: st.info("No entities match the current filters.")

def render_relationship_table(relationships, entities):
    """ Render a table of relationships with filtering options. """
    st.subheader("Relationship Table")
    if not relationships: st.info("No relationships have been extracted."); return
    entity_lookup = {entity.get("id"): entity for entity in entities if entity.get("id")}
    col1, col2 = st.columns(2)
    with col1:
        rel_types = sorted(list(set(rel.get("type", rel.get("relationship_type", "Unknown")) for rel in relationships)))
        selected_rel_types = st.multiselect("Filter by relationship type", options=rel_types, default=rel_types, key="rel_type_filter")
    with col2: entity_search = st.text_input("Search by entity name", placeholder="Enter entity name...", key="rel_entity_search")
    rel_data = []
    for rel in relationships:
        rel_type = rel.get("type", rel.get("relationship_type", "Unknown")); source_id = rel.get("source_entity_id", rel.get("from_entity_id")); target_id = rel.get("target_entity_id", rel.get("to_entity_id"))
        source_entity = entity_lookup.get(source_id, {}); target_entity = entity_lookup.get(target_id, {}); source_name = source_entity.get("name", "Unknown"); target_name = target_entity.get("name", "Unknown")
        file_name = rel.get("file_name", rel.get("document_id", "Unknown")); rel_desc = rel.get("description","N/A")
        rel_data.append({"Source": source_name, "Source Type": source_entity.get("type", "Unknown"), "Relationship": rel_type, "Description": rel_desc, "Target": target_name, "Target Type": target_entity.get("type", "Unknown"), "Document": file_name})
    rel_df = pd.DataFrame(rel_data); filtered_df = rel_df[rel_df["Relationship"].isin(selected_rel_types)]
    if entity_search:
        search_term = entity_search.lower()
        filtered_df = filtered_df[filtered_df["Source"].astype(str).str.lower().str.contains(search_term) | filtered_df["Target"].astype(str).str.lower().str.contains(search_term) | filtered_df["Description"].astype(str).str.lower().str.contains(search_term)]
    st.markdown(f"**Showing {len(filtered_df)} of {len(rel_df)} relationships**")
    st.dataframe(filtered_df, hide_index=True, use_container_width=True, column_config={"Source": st.column_config.TextColumn(width="medium"), "Source Type": st.column_config.TextColumn(width="small"), "Relationship": st.column_config.TextColumn(width="medium"), "Description": st.column_config.TextColumn(width="large"), "Target": st.column_config.TextColumn(width="medium"), "Target Type": st.column_config.TextColumn(width="small"), "Document": st.column_config.TextColumn(width="medium")})

def render_explore_page():
    """ Render the data exploration page. """
    st.header("🔎 Explore Data")
    relationships_file = EXTRACTED_DATA_PATH / "relationships.json"; entities_file = EXTRACTED_DATA_PATH / "entities.json"
    relationships, entities = [], []; load_error = False
    if relationships_file.exists() and entities_file.exists():
        try:
            with open(relationships_file, "r", encoding='utf-8') as f: relationships = json.load(f)
            with open(entities_file, "r", encoding='utf-8') as f: entities = json.load(f)
        except Exception as e: st.error(f"Error loading graph data: {e}"); load_error = True
    else: st.info("Entity or relationship files not found. Process documents first."); load_error = True
    tab1, tab2, tab3 = st.tabs(["Documents & Chunks", "Entities", "Relationships"])
    with tab1: render_document_explorer()
    with tab2: render_entity_explorer()
    with tab3:
        if load_error: st.warning("Cannot display relationships as entity/relationship data failed to load or is missing.")
        else:
            st.subheader("Relationship Network Analysis"); st.markdown("Explore extracted entities and relationships. Use the document filter to focus the analysis.")
            st.markdown("<style>iframe { min-height: 700px !important; height: 700px !important; }</style>", unsafe_allow_html=True)
            try: # Document Filter for Graphs
                st.markdown("---"); st.markdown("### Document Filter (for Graphs)")
                all_doc_names = sorted(list(set(ent.get("source_document", ent.get('context', {}).get('file_name')) for ent in entities if ent.get("source_document", ent.get('context', {}).get('file_name'))).union(set(rel.get("file_name") for rel in relationships if rel.get("file_name")))))
                selected_docs_graph = st.multiselect("Filter graphs by documents (select none for all)", options=all_doc_names, default=[], key="graph_document_filter")
                if selected_docs_graph:
                    filtered_rels = [r for r in relationships if r.get("file_name") in selected_docs_graph]
                    related_ent_ids = set(r.get("source_entity_id", r.get("from_entity_id")) for r in filtered_rels) | set(r.get("target_entity_id", r.get("to_entity_id")) for r in filtered_rels)
                    filtered_ents = [e for e in entities if e.get("id") in related_ent_ids]
                    st.success(f"Filtered graph to {len(filtered_rels)} relationships and {len(filtered_ents)} entities.")
                else: filtered_rels, filtered_ents = relationships, entities; st.info(f"Analyzing all {len(relationships)} relationships and {len(entities)} entities.")
                st.markdown("---")
            except Exception as e: st.error(f"Error applying document filter for graphs: {e}"); filtered_rels, filtered_ents = relationships, entities
            if not filtered_ents: st.warning("No entities found based on the current document filter for graphs.")
            else: # Graph Exploration Tabs
                graph_tab_titles = ["📊 Network Overview", "🔗 Connection Explorer", "🎯 Entity Centered View", "📈 Network Metrics", "📄 Relationship Table"]
                g_tab1, g_tab2, g_tab3, g_tab4, g_tab5 = st.tabs(graph_tab_titles)
                with g_tab1: render_network_overview_tab(filtered_ents, filtered_rels)
                with g_tab2: render_connection_explorer_tab(filtered_ents, filtered_rels)
                with g_tab3: render_entity_centered_tab(filtered_ents, filtered_rels)
                with g_tab4: render_network_metrics_tab(filtered_ents, filtered_rels)
                with g_tab5: render_relationship_table(filtered_rels, filtered_ents)


# --- Page: Query System ---

def render_query_page():
    """ Render the query and chat interface, adapting to the active LLM backend. """
    st.header("💬 Query System")

    query_engine = get_or_create_query_engine()
    conversation_store = get_conversation_store()
    if not query_engine or not conversation_store: st.error("Query system or conversation store failed to initialize."); return

    # Determine LLM availability based on the active backend
    llm_available = False
    llm_status_message = ""
    active_manager = st.session_state.get("active_llm_manager") # Get manager initialized in app_setup

    if IS_OPENROUTER_ACTIVE:
        if active_manager and active_manager.client:
            llm_available = True
            model_name = active_manager.models.get("chat", "N/A")
            llm_status_message = f"Using OpenRouter API (Chat Model: {model_name})."
        else:
            llm_status_message = "OpenRouter backend active, but client/API key might be missing or invalid."
    elif IS_GEMINI_ACTIVE: # Add Gemini check
        if active_manager and active_manager.client:
            llm_available = True
            model_name = active_manager.models.get("chat", "N/A")
            llm_status_message = f"Using Gemini API (Chat Model: {model_name})."
        else:
            llm_status_message = "Gemini backend active, but client/API key might be missing or invalid."
    elif LLM_BACKEND == "aphrodite" and APHRODITE_SERVICE_AVAILABLE: # Check specific backend
        is_service_running = st.session_state.get("aphrodite_service_running", False)
        is_model_loaded = st.session_state.get("llm_model_loaded", False)
        if is_service_running and is_model_loaded:
            llm_available = True
            model_name = st.session_state.get("aphrodite_process_info", {}).get("model_name", "Unknown")
            llm_status_message = f"Using local LLM service ({model_name})."
        elif is_service_running:
            llm_status_message = "Local LLM service running, but no model loaded. Process documents or use sidebar/settings."
        else:
            llm_status_message = "Local LLM service not running. Please start it from the sidebar."
    else:
        llm_status_message = f"No LLM service ({LLM_BACKEND.upper()}) is configured or available."

    if not llm_available:
        st.warning(f"LLM not available: {llm_status_message}")
        # Add button to start Aphrodite if applicable
        if LLM_BACKEND == "aphrodite" and APHRODITE_SERVICE_AVAILABLE and not st.session_state.get("aphrodite_service_running"):
             if st.button("Start LLM Service Now"): start_aphrodite_service(); st.rerun()

    # --- UI Tabs (Unchanged structure) ---
    conv_tab, chat_tab = st.tabs(["Manage Conversations", "Current Chat"])

    # --- Conversation Management Tab (Unchanged logic) ---
    with conv_tab:
        st.subheader("Conversations")
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("✨ New Conversation", use_container_width=True, type="primary"):
                new_conv_id = conversation_store.create_conversation()
                if new_conv_id:
                    new_conv_data = conversation_store.get_conversation(new_conv_id)
                    st.session_state.current_conversation_id = new_conv_id; st.session_state.active_conversation_data = new_conv_data
                    st.session_state.ui_chat_display = []; st.session_state.retrieval_enabled_for_next_turn = True
                    st.success("New conversation started!"); st.rerun()
                else: st.error("Failed to create new conversation.")
        conversations = conversation_store.list_conversations()
        if conversations:
            st.write(f"You have {len(conversations)} saved conversations:")
            for conv in conversations:
                conv_id = conv['id']; is_active = (st.session_state.get("current_conversation_id") == conv_id)
                list_col1, list_col2, list_col3 = st.columns([4, 1, 1])
                with list_col1:
                    title = conv.get("title", f"Conversation {conv_id[:6]}"); msg_count = conv.get("message_count", 0)
                    last_upd = datetime.fromtimestamp(conv.get("last_updated", 0)).strftime("%Y-%m-%d %H:%M")
                    display_title = f"**{title}**" if is_active else title
                    st.markdown(f"{display_title} ({msg_count} msgs) - *{last_upd}*", unsafe_allow_html=True)
                with list_col2:
                    if not is_active:
                        if st.button("Load", key=f"load_{conv_id}", use_container_width=True):
                            loaded_data = conversation_store.get_conversation(conv_id)
                            if loaded_data:
                                st.session_state.current_conversation_id = conv_id; st.session_state.active_conversation_data = loaded_data
                                st.session_state.ui_chat_display = []
                                for msg in loaded_data.get("messages", []):
                                    ui_msg = {"role": msg["role"], "content": msg["content"]}
                                    if msg.get("used_context"): ui_msg["sources"] = msg["used_context"]
                                    if msg.get("thinking_process"): ui_msg["thinking"] = msg["thinking_process"] # Keep for loading old data
                                    st.session_state.ui_chat_display.append(ui_msg)
                                st.session_state.retrieval_enabled_for_next_turn = False
                                st.success(f"Loaded: {loaded_data.get('title')}"); st.rerun()
                            else: st.error(f"Failed to load conversation {conv_id}.")
                    else: st.write("*(Active)*")
                with list_col3:
                    # Use unique key for delete confirmation
                    delete_confirm_key = f"del_confirm_{conv_id}_{int(time.time())}"
                    if st.button("Delete", key=f"del_{conv_id}", use_container_width=True, type="secondary"):
                        if st.checkbox("Confirm Delete?", key=delete_confirm_key):
                            if conversation_store.delete_conversation(conv_id):
                                if is_active: st.session_state.current_conversation_id = None; st.session_state.active_conversation_data = None; st.session_state.ui_chat_display = []; st.session_state.retrieval_enabled_for_next_turn = True
                                st.success("Conversation deleted."); st.rerun()
                            else: st.error("Failed to delete conversation.")
                        else:
                            st.warning("Check confirmation box to delete.")
                st.divider()
        else: st.info("No saved conversations yet.")

    # --- Current Chat Tab (Unchanged logic, uses handle_chat_message) ---
    with chat_tab:
        active_conv_data = st.session_state.get("active_conversation_data")
        if active_conv_data:
            st.subheader(f"Chat: {active_conv_data.get('title', 'Untitled')}")
            control_cols = st.columns([3, 1, 1])
            with control_cols[0]:
                new_title = st.text_input("Rename:", value=active_conv_data.get("title", ""), label_visibility="collapsed", placeholder="Rename Conversation...", key="conv_rename")
                if new_title and new_title != active_conv_data.get("title"): active_conv_data["title"] = new_title.strip(); save_current_conversation()
            with control_cols[1]:
                if st.button("End Conversation", use_container_width=True, key="conv_end"):
                    save_current_conversation(); st.session_state.current_conversation_id = None; st.session_state.active_conversation_data = None; st.session_state.ui_chat_display = []; st.session_state.retrieval_enabled_for_next_turn = True
                    st.success("Conversation ended."); st.rerun()
            st.divider()
            chat_container = st.container()
            with chat_container:
                if not st.session_state.ui_chat_display: st.info("Ask a question to start the chat!")
                for message in st.session_state.ui_chat_display:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                        if message.get("thinking"): # Display old thinking if loaded
                            with st.expander("💭 Reasoning Process", expanded=False): st.markdown(f'<div class="thinking-box">{message["thinking"]}</div>', unsafe_allow_html=True)
                        if message.get("sources"):
                            with st.expander("View Sources Used", expanded=False):
                                for i, source in enumerate(message["sources"]):
                                    idx = source.get("source_index", i + 1); score = source.get('score', 0.0); st.markdown(f"**Source {idx} (Score: {score:.2f}):**")
                                    st.markdown(f"> {source.get('text', '')}"); meta = source.get('metadata', {}); st.caption(f"Doc: {meta.get('file_name', 'N/A')} | Page: {meta.get('page_num', 'N/A')} | Chunk: {meta.get('chunk_id', 'N/A')}"); st.markdown("---")
            st.divider()
            input_cols = st.columns([4, 1])
            with input_cols[1]: st.checkbox("Enable RAG", key="retrieval_enabled_for_next_turn", help="Check *before* sending message to retrieve context. Turns OFF automatically.")
            if prompt := st.chat_input("Ask a question...", disabled=not llm_available):
                handle_chat_message(prompt) # Calls the central handler
            elif not llm_available: st.chat_input("LLM not available...", disabled=True)
        else: st.info("Please start a new conversation or load an existing one from the 'Manage Conversations' tab.")


# --- Page: Settings ---

def render_settings_page():
    """ Render the settings page, adapting to the active LLM backend. """
    st.header("⚙️ Settings")

    with st.expander("Current Configuration", expanded=False): st.json(CONFIG)
    st.subheader("Update Settings")

    # Determine active backend for conditional display
    active_backend = CONFIG.get("llm_backend", "aphrodite").lower()

    settings_tabs = st.tabs(["Retrieval", "LLM Backend", "Extraction"])

    # Retrieval settings (Unchanged)
    with settings_tabs[0]:
        st.markdown("#### Retrieval Settings")
        col1, col2 = st.columns(2)
        with col1:
            top_k_vector = st.slider("Vector Results", 1, 50, CONFIG["retrieval"]["top_k_vector"], key="s_tkv")
            top_k_bm25 = st.slider("BM25 Results", 1, 50, CONFIG["retrieval"]["top_k_bm25"], key="s_tkb")
            top_k_hybrid = st.slider("Hybrid Results", 1, 30, CONFIG["retrieval"]["top_k_hybrid"], key="s_tkh")
            top_k_rerank = st.slider("Reranked Results", 1, 20, CONFIG["retrieval"]["top_k_rerank"], key="s_tkr")
        with col2:
            vector_weight = st.slider("Vector Weight (RRF)", 0.0, 1.0, float(CONFIG["retrieval"]["vector_weight"]), 0.05, key="s_vw")
            bm25_weight = st.slider("BM25 Weight (RRF)", 0.0, 1.0, float(CONFIG["retrieval"]["bm25_weight"]), 0.05, key="s_bw")
            use_reranking = st.checkbox("Use Reranking", value=CONFIG["retrieval"]["use_reranking"], key="s_ur")
            min_score = st.slider("Min Score Threshold", 0.0, 1.0, float(CONFIG["retrieval"].get("minimum_score_threshold", 0.01)), 0.01, key="s_ms")

    # LLM Backend Settings (Conditional Display)
    with settings_tabs[1]:
        st.markdown("#### LLM Backend Selection")
        backend_options = ["aphrodite", "openrouter", "gemini"] # Add gemini
        try:
            current_backend_index = backend_options.index(active_backend)
        except ValueError:
            logger.warning(f"Configured backend '{active_backend}' not in options. Defaulting to Aphrodite.")
            current_backend_index = 0 # Default to aphrodite index

        backend_choice = st.radio(
            "Select LLM Backend",
            options=backend_options,
            index=current_backend_index,
            key="s_backend_select",
            horizontal=True
        )
        st.markdown("---")

        if backend_choice == "openrouter":
            st.markdown("#### OpenRouter API Settings")
            or_config = CONFIG.get("openrouter", {}) # Get current or empty dict
            api_key = st.text_input("API Key", value=or_config.get("api_key", ""), type="password", key="s_or_key")
            base_url = st.text_input("API Base URL", value=or_config.get("base_url", "https://openrouter.ai/api/v1"), key="s_or_url")
            site_url = st.text_input("Site URL (Optional Header)", value=or_config.get("site_url", ""), key="s_or_siteurl")
            site_title = st.text_input("Site Title (Optional Header)", value=or_config.get("site_title", "Anti-Corruption Tool"), key="s_or_sitetitle")

            st.markdown("##### Model Selection (OpenRouter)")
            col_or1, col_or2 = st.columns(2)
            with col_or1:
                or_chat_model = st.text_input("Chat Model", value=or_config.get("chat_model", "mistralai/mistral-7b-instruct:free"), key="s_or_m_chat")
                or_extract_model = st.text_input("Entity Extraction Model", value=or_config.get("extraction_model", "mistralai/mistral-7b-instruct:free"), key="s_or_m_extract")
                or_info_model = st.text_input("Info Extraction Model", value=or_config.get("info_extraction_model", "mistralai/mistral-7b-instruct:free"), key="s_or_m_info")
            with col_or2:
                or_class_model = st.text_input("Classification Model", value=or_config.get("classification_model", "mistralai/mistral-7b-instruct:free"), key="s_or_m_class")
                or_topic_model = st.text_input("Topic Labeling Model", value=or_config.get("topic_labeling_model", "mistralai/mistral-7b-instruct:free"), key="s_or_m_topic")

            st.markdown("##### Default Generation Parameters (OpenRouter)")
            col_or3, col_or4 = st.columns(2)
            with col_or3: or_temp = st.slider("Default Temperature", 0.0, 1.5, float(or_config.get("temperature", 0.7)), 0.05, key="s_or_temp")
            with col_or4: or_max_tokens = st.slider("Default Max Tokens", 256, 8192, or_config.get("max_tokens", 1024), 128, key="s_or_maxt")

        elif backend_choice == "gemini": # Add Gemini settings section
            st.markdown("#### Google Gemini API Settings")
            gem_config = CONFIG.get("gemini", {}) # Get current or empty dict
            api_key_gem = st.text_input("API Key", value=gem_config.get("api_key", ""), type="password", key="s_gem_key")

            st.markdown("##### Model Selection (Gemini)")
            col_gem1, col_gem2 = st.columns(2)
            with col_gem1:
                gem_chat_model = st.text_input("Chat Model", value=gem_config.get("chat_model", "gemini-1.5-flash-latest"), key="s_gem_m_chat")
                # Use the specific model from the example for extraction tasks
                gem_extract_model = st.text_input("Entity Extraction Model", value=gem_config.get("extraction_model", "gemini-2.5-pro-exp-03-25"), key="s_gem_m_extract")
                gem_info_model = st.text_input("Info Extraction Model", value=gem_config.get("info_extraction_model", "gemini-2.5-pro-exp-03-25"), key="s_gem_m_info")
            with col_gem2:
                gem_class_model = st.text_input("Classification Model", value=gem_config.get("classification_model", "gemini-2.5-pro-exp-03-25"), key="s_gem_m_class")
                gem_topic_model = st.text_input("Topic Labeling Model", value=gem_config.get("topic_labeling_model", "gemini-1.5-flash-latest"), key="s_gem_m_topic")

            st.markdown("##### Default Generation Parameters (Gemini)")
            col_gem3, col_gem4 = st.columns(2)
            with col_gem3: gem_temp = st.slider("Default Temperature", 0.0, 1.0, float(gem_config.get("temperature", 0.7)), 0.05, key="s_gem_temp") # Gemini temp range often 0-1
            with col_gem4: gem_max_tokens = st.slider("Default Max Tokens", 256, 8192, gem_config.get("max_tokens", 2048), 128, key="s_gem_maxt")

        elif backend_choice == "aphrodite":
            st.markdown("#### Aphrodite Service Settings")
            aph_config = CONFIG.get("aphrodite", {}) # Get current or empty dict
            col3, col4 = st.columns(2)
            with col3:
                max_model_len = st.slider("Max Model Length", 1024, 16384, aph_config.get("max_model_len", 4096), 1024, key="s_mml")
                quant_options = ["none", "fp8", "fp5", "fp4", "fp6"] # Adjust based on Aphrodite support
                current_quant = aph_config.get("quantization", "fp5")
                quant_index = quant_options.index(current_quant) if current_quant in quant_options else 0
                quantization = st.selectbox("Quantization", options=quant_options, index=quant_index, key="s_q")
            with col4:
                enforce_eager = st.checkbox("Enforce Eager", value=aph_config.get("enforce_eager", True), key="s_aeager")
                cpu_offload_gb = st.number_input("CPU Offload GB (0=disable)", min_value=0, value=aph_config.get("cpu_offload_gb", 0), step=4, key="s_cpu_offload")
                swap_space = st.number_input("Swap Space GB", min_value=0, value=aph_config.get("swap_space", 4), step=4, key="s_swap")


            st.markdown("##### Task Parameters (Aphrodite)")
            col5, col6 = st.columns(2)
            with col5:
                extraction_temperature = st.slider("Extraction Temp", 0.0, 1.0, float(aph_config.get("extraction_temperature", 0.1)), 0.05, key="s_et")
                chat_temperature = st.slider("Chat Temp", 0.0, 1.5, float(aph_config.get("chat_temperature", 0.7)), 0.05, key="s_ct")
                top_p = st.slider("Top P (Chat)", 0.1, 1.0, float(aph_config.get("top_p", 0.9)), 0.05, key="s_tp")
            with col6:
                extraction_max_new_tokens = st.slider("Extraction Max Tokens", 256, 4096, aph_config.get("extraction_max_new_tokens", 1024), 128, key="s_emt")
                chat_max_new_tokens = st.slider("Chat Max Tokens", 256, 4096, aph_config.get("chat_max_new_tokens", 1024), 128, key="s_cmt")
                info_extraction_temp = st.slider("Info Extraction Temp", 0.0, 1.0, float(aph_config.get("info_extraction_temperature", 0.2)), 0.05, key="s_iet")
                info_extraction_max_tokens = st.slider("Info Extraction Max Tokens", 256, 4096, aph_config.get("info_extraction_max_tokens", 2048), 128, key="s_iemt")
                classification_temp = st.slider("Classification Temp", 0.0, 1.0, float(aph_config.get("classification_temperature", 0.1)), 0.05, key="s_clt")
                classification_max_tokens = st.slider("Classification Max Tokens", 64, 1024, aph_config.get("classification_max_tokens", 256), 64, key="s_clmt")


    # Extraction settings (Deduplication - Unchanged)
    with settings_tabs[2]:
        st.markdown("#### Entity Deduplication Settings")
        # Example: Use the first threshold as a representative setting
        person_threshold = CONFIG["deduplication"]["similarity_thresholds"]["PERSON"]["token_set_ratio"]
        dedup_threshold = st.slider("Person Name Similarity Threshold (%)", 0, 100, person_threshold, key="s_dt_person")
        st.caption("Note: More detailed deduplication settings are in config.yaml.")


    # Save button
    if st.button("Save Settings", type="primary"):
        # Update config dictionary
        CONFIG["llm_backend"] = backend_choice # Save selected backend
        CONFIG["retrieval"].update({
            "top_k_vector": top_k_vector, "top_k_bm25": top_k_bm25, "top_k_hybrid": top_k_hybrid,
            "top_k_rerank": top_k_rerank, "vector_weight": float(vector_weight), "bm25_weight": float(bm25_weight),
            "use_reranking": use_reranking, "minimum_score_threshold": float(min_score)
        })
        # Update backend-specific settings
        if backend_choice == "openrouter":
            if "openrouter" not in CONFIG: CONFIG["openrouter"] = {}
            CONFIG["openrouter"].update({
                "api_key": api_key, "base_url": base_url, "site_url": site_url, "site_title": site_title,
                "chat_model": or_chat_model, "extraction_model": or_extract_model,
                "info_extraction_model": or_info_model, "classification_model": or_class_model,
                "topic_labeling_model": or_topic_model, "temperature": float(or_temp), "max_tokens": or_max_tokens
            })
        elif backend_choice == "gemini": # Add Gemini save logic
            if "gemini" not in CONFIG: CONFIG["gemini"] = {}
            CONFIG["gemini"].update({
                "api_key": api_key_gem,
                "chat_model": gem_chat_model, "extraction_model": gem_extract_model,
                "info_extraction_model": gem_info_model, "classification_model": gem_class_model,
                "topic_labeling_model": gem_topic_model, "temperature": float(gem_temp), "max_tokens": gem_max_tokens
            })
        elif backend_choice == "aphrodite":
            if "aphrodite" not in CONFIG: CONFIG["aphrodite"] = {}
            CONFIG["aphrodite"].update({
                "max_model_len": max_model_len, "quantization": quantization, "enforce_eager": enforce_eager,
                "cpu_offload_gb": cpu_offload_gb, "swap_space": swap_space,
                "extraction_temperature": float(extraction_temperature), "extraction_max_new_tokens": extraction_max_new_tokens,
                "chat_temperature": float(chat_temperature), "chat_max_new_tokens": chat_max_new_tokens, "top_p": float(top_p),
                "info_extraction_temperature": float(info_extraction_temp), "info_extraction_max_tokens": info_extraction_max_tokens,
                "classification_temperature": float(classification_temp), "classification_max_tokens": classification_max_tokens
            })
        # Update deduplication setting example
        CONFIG["deduplication"]["similarity_thresholds"]["PERSON"]["token_set_ratio"] = dedup_threshold

        # Save to file
        try:
            with open(CONFIG_PATH, "w") as f: yaml.dump(CONFIG, f, default_flow_style=False, sort_keys=False)
            st.success("Settings saved successfully!")
            st.warning("**Note:** Application might need a restart for backend changes or significant LLM parameter changes to take full effect.")
            # Clear manager state to force reinitialization on next action
            if "active_llm_manager" in st.session_state:
                del st.session_state.active_llm_manager
                logger.info("Cleared active LLM manager state due to settings change.")
            st.rerun() # Rerun to reflect changes immediately
        except Exception as e:
            st.error(f"Failed to save settings: {e}")
            logger.error(f"Error saving config file: {e}")


# --- Page: Cluster Map ---

def render_cluster_map_page():
    """ Render the cluster map visualization page. """
    st.header("🗺️ Document Cluster Map")
    if not cluster_map_available: st.error("Cluster map module or dependencies unavailable."); return
    if not check_dependencies(): st.error("Required libraries for cluster mapping missing."); return
    query_engine = get_or_create_query_engine(); collection_info = query_engine.get_collection_info() if query_engine else {}
    if not collection_info.get("exists", False) or collection_info.get("points_count", 0) == 0: st.warning("No documents found. Process documents first."); return
    with st.expander("Clustering Configuration"): # Config UI (Unchanged)
        config_tabs = st.tabs(["UMAP & HDBSCAN", "Topic Parameters", "Visualization"])
        with config_tabs[0]:
            st.subheader("UMAP Parameters"); col1, col2 = st.columns(2)
            with col1: umap_n_neighbors = st.slider("n_neighbors", 5, 50, CONFIG["clustering"]["umap"].get("n_neighbors", 15), 1, key="umap_nn"); umap_min_dist = st.slider("min_dist", 0.0, 1.0, float(CONFIG["clustering"]["umap"].get("min_dist", 0.0)), 0.05, key="umap_md")
            with col2: umap_n_components = st.slider("n_components", 2, 10, CONFIG["clustering"]["umap"].get("n_components", 5), 1, key="umap_nc"); umap_metric_options = ["cosine", "euclidean", "manhattan", "correlation"]; umap_metric = st.selectbox("metric", umap_metric_options, index=umap_metric_options.index(CONFIG["clustering"]["umap"].get("metric", "cosine")), key="umap_met")
            st.subheader("HDBSCAN Parameters"); col1, col2 = st.columns(2)
            with col1: min_cluster_size = st.slider("min_cluster_size", 2, 100, CONFIG["clustering"]["hdbscan"].get("min_cluster_size", 10), 5, key="hdb_mcs")
            with col2: min_samples = st.slider("min_samples", 1, 20, CONFIG["clustering"]["hdbscan"].get("min_samples", 5), 1, key="hdb_ms")
        with config_tabs[1]:
            st.subheader("Topic Parameters"); col1, col2 = st.columns(2)
            with col1: nr_topics_options = ["auto"] + [str(i) for i in range(1, 51)]; nr_topics = st.selectbox("Number of Topics", nr_topics_options, index=nr_topics_options.index(str(CONFIG["clustering"]["topics"].get("nr_topics", "auto"))), key="topic_nr")
            with col2: seed_topics_str = "\n".join([",".join(topic) for topic in CONFIG["clustering"]["topics"].get("seed_topic_list", [[]])]); seed_topics = st.text_area("Seed Topics (keywords per line)", value=seed_topics_str, key="topic_seed")
            seed_topic_list = [[word.strip() for word in line.split(",") if word.strip()] for line in seed_topics.strip().split("\n") if line.strip()]
        with config_tabs[2]:
            st.subheader("Visualization Options"); current_vis_type = CONFIG["clustering"].get("visualization_type", "plotly").lower(); st.markdown(f"Current Type: `{current_vis_type.upper()}`"); vis_options = ["plotly", "datamapplot", "static_datamapplot"]; vis_labels = ["Plotly", "Interactive DataMapPlot", "Static DataMapPlot"]
            try: vis_index = vis_options.index(current_vis_type)
            except ValueError: vis_index = 0
            visualization_type = st.radio("Select Visualization Type", vis_labels, index=vis_index, key="vis_type_select"); selected_vis_type = vis_options[vis_labels.index(visualization_type)]
            if selected_vis_type != current_vis_type: st.warning(f"Type will change to `{selected_vis_type.upper()}`. Click 'Update Configuration' and 'Generate Cluster Map'.")
            if "datamapplot" in selected_vis_type:
                try: import datamapplot; datamapplot_ok = True
                except ImportError: st.warning("DataMapPlot not installed (`pip install datamapplot`)."); datamapplot_ok = False
                if datamapplot_ok:
                    config_key = "datamapplot" if selected_vis_type == "datamapplot" else "static_datamapplot"; st.subheader(f"{visualization_type} Options"); col1, col2 = st.columns(2)
                    with col1: darkmode = st.checkbox("Dark Mode", value=CONFIG["clustering"].get(config_key, {}).get("darkmode", False), key="dmp_dark"); cvd_safer = st.checkbox("CVD-Safer Palette", value=CONFIG["clustering"].get(config_key, {}).get("cvd_safer", True), key="dmp_cvd"); enable_toc = st.checkbox("Enable TOC", value=CONFIG["clustering"].get(config_key, {}).get("enable_table_of_contents", True), key="dmp_toc") if selected_vis_type == "datamapplot" else None
                    with col2: cluster_boundaries = st.checkbox("Show Boundaries", value=CONFIG["clustering"].get(config_key, {}).get("cluster_boundary_polygons", True), key="dmp_bound") if selected_vis_type == "datamapplot" else None; color_labels = st.checkbox("Color Label Text", value=CONFIG["clustering"].get(config_key, {}).get("color_label_text", True), key="dmp_clabel"); marker_size = st.slider("Marker Size", 3, 15, CONFIG["clustering"].get(config_key, {}).get("marker_size", 8), key="dmp_msize")
                    fonts = ["Oswald", "Helvetica", "Roboto", "Times New Roman", "Georgia", "Courier New", "Playfair Display SC", "Open Sans"]; current_font = CONFIG["clustering"].get(config_key, {}).get("font_family", "Oswald"); font_index = fonts.index(current_font) if current_font in fonts else 0; font_family = st.selectbox("Font Family", fonts, index=font_index, key="dmp_font")
                    polygon_alpha = st.slider("Boundary Opacity", 0.05, 5.00, CONFIG["clustering"].get(config_key, {}).get("polygon_alpha", 2.5), 0.05, key="dmp_alpha") if selected_vis_type == "datamapplot" else None
                    dpi = st.slider("Plot DPI", 72, 600, CONFIG["clustering"].get(config_key, {}).get("dpi", 300), 1, key="dmp_dpi") if selected_vis_type == "static_datamapplot" else None
        if st.button("Update Configuration", key="cluster_update_config"): # Update Config Button (Unchanged)
            if "clustering" not in CONFIG: CONFIG["clustering"] = {}
            CONFIG["clustering"]["umap"] = {"n_neighbors": umap_n_neighbors, "n_components": umap_n_components, "min_dist": umap_min_dist, "metric": umap_metric}
            CONFIG["clustering"]["hdbscan"] = {"min_cluster_size": min_cluster_size, "min_samples": min_samples, "prediction_data": True}
            CONFIG["clustering"]["topics"] = {"nr_topics": nr_topics, "seed_topic_list": seed_topic_list}
            CONFIG["clustering"]["visualization_type"] = selected_vis_type; logger.info(f"Updating CONFIG visualization_type to: {selected_vis_type}")
            if "datamapplot" in selected_vis_type and datamapplot_ok:
                 config_key_int = "datamapplot"; config_key_stat = "static_datamapplot"
                 if config_key_int not in CONFIG["clustering"]: CONFIG["clustering"][config_key_int] = {}
                 if config_key_stat not in CONFIG["clustering"]: CONFIG["clustering"][config_key_stat] = {}
                 CONFIG["clustering"][config_key_int].update({"darkmode": darkmode, "cvd_safer": cvd_safer, "enable_table_of_contents": enable_toc, "cluster_boundary_polygons": cluster_boundaries, "color_label_text": color_labels, "marker_size": marker_size, "font_family": font_family, "height": 800, "width": "100%", "polygon_alpha": polygon_alpha})
                 CONFIG["clustering"][config_key_stat].update({"darkmode": darkmode, "cvd_safer": cvd_safer, "color_label_text": color_labels, "marker_size": marker_size, "font_family": font_family, "dpi": dpi})
            try:
                with open(CONFIG_PATH, "w") as f:
                    yaml.dump(CONFIG, f, default_flow_style=False, sort_keys=False)
                st.success("Configuration updated successfully!"); st.session_state.pop('cluster_map_result', None)
            except Exception as e: st.error(f"Failed to save configuration: {e}")
    st.subheader("Document Selection"); document_options = [] # Document Selection (Unchanged)
    if query_engine:
        try: chunks = query_engine.get_chunks(limit=1000); document_options = sorted(list(set(c['metadata'].get('file_name', 'Unknown') for c in chunks if c['metadata'].get('file_name'))))
        except Exception as e: st.error(f"Error retrieving document names: {e}")
    if document_options: selected_documents = st.multiselect("Select documents for clustering (empty=all)", options=document_options, default=[], key="cluster_docs"); st.info(f"Clustering limited to {len(selected_documents)} selected documents.") if selected_documents else None
    else: st.info("No documents available for selection."); selected_documents = []
    st.session_state.selected_documents_for_clustering = selected_documents
    include_outliers = st.checkbox("Include Outliers/Noise Points (Topic -1)", value=False, key="cluster_outliers")
    generate_btn = st.button("Generate Cluster Map", type="primary", key="cluster_generate")
    progress_placeholder = st.empty(); vis_placeholder = st.empty()
    if generate_btn: # Generate Button (Unchanged)
        current_config_vis_type = CONFIG["clustering"].get("visualization_type", "plotly").lower()
        with progress_placeholder.container():
            st.info(f"Using visualization type: {current_config_vis_type.upper()}")
            with st.spinner(f"Generating cluster map using {current_config_vis_type.upper()}..."):
                result, message = generate_cluster_map(query_engine, include_outliers=include_outliers)
                if result: st.session_state.cluster_map_result = result; st.success(message)
                else: st.error(message); st.session_state.pop('cluster_map_result', None)
    if 'cluster_map_result' in st.session_state and st.session_state.cluster_map_result: # Display Results (Unchanged)
        result = st.session_state.cluster_map_result
        with vis_placeholder.container():
            st.subheader("Cluster Map"); col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Total Docs", result.get("document_count", "N/A"))
            with col2: st.metric("Clustered Docs", result.get("clustered_count", "N/A"))
            with col3: st.metric("Outliers", f"{result.get('outlier_count', 'N/A')} ({result.get('outlier_percentage', 0):.1f}%)")
            with col4: st.metric("Topics", result.get("topic_count", "N/A"))
            if result.get("is_static", False): st.pyplot(result["figure"])
            elif result.get("is_datamap", False): st.components.v1.html(result["figure"]._repr_html_(), height=800, scrolling=True)
            else: st.plotly_chart(result["figure"], use_container_width=True)
            with st.expander("Topic Information", expanded=False):
                topic_info_df = result.get("topic_info", pd.DataFrame())
                if not topic_info_df.empty:
                    # Clean names before display
                    cleaned_names = []
                    for name in topic_info_df['Name']:
                        if isinstance(name, str):
                            cleaned_name = name.replace("Topic Name:", "").replace("_", " ").strip()
                            if len(cleaned_name) > 100 or '_type_' in cleaned_name:
                                 # Find corresponding topic ID to display instead
                                 topic_id = topic_info_df[topic_info_df['Name'] == name]['Topic'].iloc[0]
                                 cleaned_name = f"Topic {topic_id}"
                        else:
                            cleaned_name = str(name) # Convert non-strings
                        cleaned_names.append(cleaned_name)
                    topic_info_df['Name'] = cleaned_names

                    display_topic_info = topic_info_df[topic_info_df["Topic"] != -1] if not result.get("include_outliers_setting", False) else topic_info_df
                    st.dataframe(display_topic_info, use_container_width=True, column_config={ "Topic": st.column_config.NumberColumn(width="small"), "Count": st.column_config.NumberColumn(width="small"), "Name": st.column_config.TextColumn(width="medium"), "Representation": st.column_config.TextColumn(width="large") })
            st.subheader("Download Results")
            if 'topic_info' in result and 'docs_df' in result:
                # Pass the original topic_info (before cleaning for display) to the download function
                download_df = create_download_dataframe(result['docs_df'], result['topic_info'])
                col1, col2 = st.columns(2)
                with col1: csv = download_df.to_csv(index=False).encode('utf-8'); st.download_button(label="Download CSV", data=csv, file_name='cluster_results.csv', mime='text/csv', use_container_width=True)
                with col2:
                    buffer = io.BytesIO();
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer: download_df.to_excel(writer, sheet_name='Cluster Results', index=False); result['topic_info'].to_excel(writer, sheet_name='Topics Overview', index=False)
                    st.download_button(label="Download Excel", data=buffer, file_name='cluster_results.xlsx', mime='application/vnd.ms-excel', use_container_width=True)
                with st.expander("Preview Download Data", expanded=False): st.dataframe(download_df.head(10))


# --- Page: Topic Filter ---

def render_topic_filter_page():
    """ Render the topic filtering page. (Unchanged) """
    st.header("🧮 Topic Filter")
    if not embedding_filter_available: st.error("Embedding filter module not available."); return
    query_engine = get_or_create_query_engine();
    if not query_engine: st.error("Query Engine not available."); return
    try: embedding_filter = EmbeddingFilter(query_engine)
    except Exception as e: st.error(f"Error initializing embedding filter: {e}"); return
    collection_info = query_engine.get_collection_info()
    if not collection_info.get("exists", False) or collection_info.get("points_count", 0) == 0: st.warning("No indexed documents found."); return
    with st.form("topic_filter_form"):
        topic_query = st.text_area("Enter topic description or query", placeholder="Describe the topic...", height=100)
        try: document_names = embedding_filter.get_document_names()
        except Exception as e: st.error(f"Error getting document names: {e}"); document_names = []
        col1, col2 = st.columns(2)
        with col1: selected_docs = st.multiselect("Filter by Documents (empty=all)", options=document_names, default=[])
        with col2: top_k = st.slider("Number of results", 10, 5000, 100, 10)
        submit_button = st.form_submit_button("Run Topic Filter", type="primary")
    if "topic_filter_results" not in st.session_state: st.session_state.topic_filter_results = None
    if submit_button and topic_query:
        with st.spinner("Filtering chunks by topic..."):
            included_docs = set(selected_docs) if selected_docs else None
            results = embedding_filter.filter_by_topic(topic_query=topic_query, top_k=top_k, included_docs=included_docs)
            st.session_state.topic_filter_results = results
            if results: st.success(f"Found {len(results)} relevant chunks!")
            else: st.warning("No relevant chunks found.")
    if st.session_state.topic_filter_results:
        results = st.session_state.topic_filter_results
        rows = [{'Score': f"{r['score']:.4f}", 'Document': r['metadata'].get('file_name', 'N/A'), 'Page': r['metadata'].get('page_num', 'N/A'), 'Text': r.get('original_text', r.get('text', ''))[:200] + "..."} for r in results]
        df = pd.DataFrame(rows); st.subheader(f"Results ({len(results)} chunks)")
        st.dataframe(df, use_container_width=True, column_config={"Score": st.column_config.NumberColumn(format="%.4f", width="small"), "Document": st.column_config.TextColumn(width="medium"), "Page": st.column_config.TextColumn(width="small"), "Text": st.column_config.TextColumn(width="large")})
        csv_data = df.to_csv(index=False).encode('utf-8'); st.download_button(label="Download Results as CSV", data=csv_data, file_name="topic_filter_results.csv", mime="text/csv")
        st.subheader("Text View")
        if results:
            selected_index = st.selectbox("Select chunk to view", options=range(len(results)), key="topic_view_select")
            if selected_index is not None:
                selected_result = results[selected_index]; meta = selected_result['metadata']
                st.markdown(f"**Document:** {meta.get('file_name', 'N/A')} | **Page:** {meta.get('page_num', 'N/A')} | **Score:** {selected_result['score']:.4f}")
                st.markdown("**Full Text:**"); st.text_area("Chunk Text", selected_result.get('original_text', selected_result.get('text', '')), height=300, key="topic_text_view", disabled=True)


# --- Page: Information Extraction ---

def render_info_extraction_page():
    """ Render the information extraction page, checking active backend status. """
    st.header("📊 Information Extraction")
    if not info_extractor_available: st.error("Information extraction module not available."); return
    query_engine = get_or_create_query_engine();
    if not query_engine: st.error("Query Engine not available."); return
    collection_info = query_engine.get_collection_info()
    if not collection_info.get("exists", False) or collection_info.get("points_count", 0) == 0: st.warning("No indexed documents found."); return

    # Check LLM backend readiness
    llm_ready = False; active_manager = st.session_state.get("active_llm_manager")
    if IS_OPENROUTER_ACTIVE: llm_ready = bool(active_manager and active_manager.client)
    elif IS_GEMINI_ACTIVE: llm_ready = bool(active_manager and active_manager.client) # Add Gemini check
    elif LLM_BACKEND == "aphrodite" and APHRODITE_SERVICE_AVAILABLE: llm_ready = st.session_state.get("aphrodite_service_running", False) and st.session_state.get("llm_model_loaded", False)
    if not llm_ready: st.warning(f"The active LLM backend ({LLM_BACKEND.upper()}) is not ready for extraction."); return

    # State initialization (Unchanged)
    if "info_extraction_schema" not in st.session_state: st.session_state.info_extraction_schema = {"fields": [{"name": "entity", "type": "string", "description": "Primary entity"}], "primary_key": "entity", "primary_key_description": "entities", "user_query": "Extract entities"}
    if "info_extraction_results" not in st.session_state: st.session_state.info_extraction_results = None

    schema_tab, extract_tab, results_tab = st.tabs(["Define Schema", "Extract Information", "View Results"])

    with schema_tab: # Schema Definition UI (Unchanged)
        st.subheader("Define Extraction Schema"); st.markdown("Define fields to extract structured information.")
        schema = st.session_state.info_extraction_schema
        pk_options = [f["name"] for f in schema["fields"]]; pk_index = pk_options.index(schema["primary_key"]) if schema["primary_key"] in pk_options else 0
        primary_key = st.selectbox("Primary Entity Field", options=pk_options, index=pk_index, key="ie_pk"); primary_key_description = f"{primary_key}s"
        user_query = st.text_area("Extraction Query", value=schema["user_query"], height=100, key="ie_query")
        st.subheader("Schema Fields"); field_types = ["string", "number", "integer", "boolean", "date"]; fields_container = st.container()
        if st.button("Add Field", key="ie_add_field"): schema["fields"].append({"name": f"field_{len(schema['fields'])}", "type": "string", "description": ""})
        with fields_container:
            updated_fields = []
            for i, field in enumerate(schema["fields"]):
                col1, col2, col3, col4 = st.columns([2, 2, 5, 1])
                with col1: field_name = st.text_input("Name", field["name"], key=f"ie_fn_{i}")
                with col2: field_type = st.selectbox("Type", field_types, index=field_types.index(field["type"]) if field["type"] in field_types else 0, key=f"ie_ft_{i}")
                with col3: field_description = st.text_input("Description", field["description"], key=f"ie_fd_{i}")
                remove = False; can_remove = len(schema["fields"]) > 1 and field["name"] != primary_key
                with col4: remove = st.button("🗑️", key=f"ie_rem_{i}", help="Remove field") if can_remove else st.button("🗑️", key=f"ie_rem_{i}", disabled=True, help="Cannot remove primary key or last field")
                if not remove: updated_fields.append({"name": field_name, "type": field_type, "description": field_description})
            schema["fields"] = updated_fields
        if st.button("Save Schema", type="primary", key="ie_save"):
            schema["primary_key"] = primary_key; schema["primary_key_description"] = primary_key_description; schema["user_query"] = user_query
            field_names = [f["name"] for f in schema["fields"]]
            if len(field_names) != len(set(field_names)): st.error("Field names must be unique!")
            elif primary_key not in field_names: st.error(f"Primary key '{primary_key}' is not a defined field!")
            else: st.success("Schema saved!"); st.rerun()

    with extract_tab: # Extraction Execution UI (Uses InfoExtractor which handles backend)
        st.subheader("Extract Information")
        with st.expander("Current Schema", expanded=True):
            schema = st.session_state.info_extraction_schema
            field_data = [{"Name": f["name"], "Type": f["type"], "Description": f["description"], "Primary Key": "✓" if f["name"] == schema["primary_key"] else ""} for f in schema["fields"]]
            st.dataframe(field_data, hide_index=True); st.caption(f"Primary Key Desc: {schema['primary_key_description']} | Query: {schema['user_query']}")
        st.subheader("Select Documents")
        try: max_chunks_for_names = CONFIG.get("extraction", {}).get("information_extraction", {}).get("max_chunks_for_listing", 1000); all_chunks = query_engine.get_chunks(limit=max_chunks_for_names); doc_names = sorted(list(set(c['metadata'].get('file_name', 'N/A') for c in all_chunks if c['metadata'].get('file_name'))))
        except Exception as e: logger.error(f"Failed to get doc names: {e}"); doc_names = []
        selected_docs = st.multiselect("Select documents (empty=all)", options=doc_names, default=[], key="ie_docs")
        # Model selection (display only, actual model used depends on backend config)
        model_display_name = "N/A"
        if IS_OPENROUTER_ACTIVE: model_display_name = CONFIG.get("openrouter", {}).get("info_extraction_model", "N/A")
        elif IS_GEMINI_ACTIVE: model_display_name = CONFIG.get("gemini", {}).get("info_extraction_model", "gemini-2.5-pro-exp-03-25") # Use specific model
        else: model_display_name = CONFIG.get("models", {}).get("extraction_models", {}).get("text_standard", "N/A") # Aphrodite uses standard model here
        st.info(f"Using LLM: `{model_display_name}` (Backend: {LLM_BACKEND.upper()})")
        max_chunks_to_process = st.slider("Max Chunks to Process", 10, 5000, 500, 10, key="ie_max_chunks")
        extract_btn = st.button("Run Extraction", type="primary", key="ie_run"); extraction_progress = st.empty()
        if extract_btn:
            if not InfoExtractor: st.error("InfoExtractor module not loaded."); return
            info_extractor = InfoExtractor() # Uses active manager internally
            with extraction_progress.container():
                with st.spinner("Fetching chunks..."):
                    chunks_to_process = [];
                    if selected_docs:
                        limit_per_doc = max(1, max_chunks_to_process // len(selected_docs)) if selected_docs else max_chunks_to_process
                        for doc_name in selected_docs: doc_chunks = query_engine.get_chunks(limit=limit_per_doc, document_filter=doc_name); chunks_to_process.extend(doc_chunks);
                    else: chunks_to_process = query_engine.get_chunks(limit=max_chunks_to_process)
                    chunks_to_process = chunks_to_process[:max_chunks_to_process] # Hard limit
                    st.info(f"Processing {len(chunks_to_process)} chunks...")
            with extraction_progress.container():
                with st.spinner("Extracting information..."):
                    schema_dict = {f["name"]: {"type": f["type"], "description": f["description"]} for f in schema["fields"]}
                    results = info_extractor.extract_information(chunks=chunks_to_process, schema_dict=schema_dict, primary_key_field=schema["primary_key"], primary_key_description=schema["primary_key_description"], user_query=schema["user_query"])
                    st.session_state.info_extraction_results = results
                    if results: st.success(f"Successfully extracted {len(results)} items!")
                    else: st.warning("No information extracted.")

    with results_tab: # Results Display (Unchanged)
        st.subheader("Extraction Results"); results = st.session_state.info_extraction_results
        if not results: st.info("No extraction results available."); return
        flattened_rows = []
        for item in results:
            row_data = {k: v for k, v in item.items() if k != '_source'}
            if '_source' in item and isinstance(item['_source'], dict):
                for src_key, src_value in item['_source'].items(): row_data[f'source_{src_key}'] = src_value
            flattened_rows.append(row_data)
        df = pd.DataFrame(flattened_rows) if flattened_rows else pd.DataFrame()
        unique_docs = set(item['_source']['file_name'] for item in results if '_source' in item and 'file_name' in item['_source'])
        st.info(f"Found {len(results)} items from {len(unique_docs)} documents")
        if not df.empty:
            csv_data = df.to_csv(index=False).encode('utf-8'); st.download_button("Export Results as CSV", csv_data, "extraction_results.csv", "text/csv", key="ie_export")
            st.dataframe(df, hide_index=True)
        else: st.warning("No data to display.")


# --- Page: Document Classification ---

def render_classification_page():
    """ Render the document classification page, checking active backend status. """
    st.header("🏷️ Document Classification")
    if not doc_classifier_available: st.error("Document classification module not available."); return
    query_engine = get_or_create_query_engine();
    if not query_engine: st.error("Query Engine not available."); return
    collection_info = query_engine.get_collection_info()
    if not collection_info.get("exists", False) or collection_info.get("points_count", 0) == 0: st.warning("No indexed documents found."); return

    # Check LLM backend readiness
    llm_ready = False; active_manager = st.session_state.get("active_llm_manager")
    if IS_OPENROUTER_ACTIVE: llm_ready = bool(active_manager and active_manager.client)
    elif IS_GEMINI_ACTIVE: llm_ready = bool(active_manager and active_manager.client) # Add Gemini check
    elif LLM_BACKEND == "aphrodite" and APHRODITE_SERVICE_AVAILABLE: llm_ready = st.session_state.get("aphrodite_service_running", False) and st.session_state.get("llm_model_loaded", False)
    if not llm_ready: st.warning(f"The active LLM backend ({LLM_BACKEND.upper()}) is not ready for classification."); return

    # State initialization (Unchanged)
    if "classification_schema" not in st.session_state: st.session_state.classification_schema = {"fields": [{"name": "category", "type": "string", "values": ["General", "Legal", "Financial"], "description": "Doc category"}], "multi_label_fields": ["category"], "user_instructions": "Classify by main topic"}
    if "classification_results" not in st.session_state: st.session_state.classification_results = None

    schema_tab, classify_tab, results_tab = st.tabs(["Define Schema", "Classify Documents", "View Results"])

    with schema_tab: # Schema Definition UI (Unchanged)
        st.subheader("Define Classification Schema"); st.markdown("Define categories and instructions for classifying documents.")
        schema = st.session_state.classification_schema; st.subheader("Classification Fields"); fields_container = st.container()
        if st.button("Add Field", key="clf_add"): schema["fields"].append({"name": f"field_{len(schema['fields'])}", "type": "string", "values": [], "description": ""})
        with fields_container:
            updated_fields = []; updated_multi_label = []
            for i, field in enumerate(schema["fields"]):
                st.markdown(f"--- \n**Field {i+1}**"); col1, col2 = st.columns(2)
                with col1: field_name = st.text_input("Name", field["name"], key=f"clf_fn_{i}")
                with col2: field_description = st.text_input("Description", field["description"], key=f"clf_fd_{i}")
                values_str = ", ".join(field.get("values", [])); values_input = st.text_area("Allowed Values (comma-separated)", value=values_str, key=f"clf_fv_{i}")
                values_list = [v.strip() for v in values_input.split(",") if v.strip()]
                is_multi_label = st.checkbox("Allow multiple values", value=(field["name"] in schema.get("multi_label_fields", [])), key=f"clf_fm_{i}")
                remove = st.button("Remove Field", key=f"clf_rem_{i}")
                if not remove: updated_fields.append({"name": field_name, "type": "string", "values": values_list, "description": field_description}); updated_multi_label.append(field_name) if is_multi_label else None
            schema["fields"] = updated_fields; schema["multi_label_fields"] = updated_multi_label
        st.subheader("Classification Instructions"); schema["user_instructions"] = st.text_area("Instructions", value=schema.get("user_instructions", ""), height=100, key="clf_instr")
        if st.button("Save Schema", type="primary", key="clf_save"):
            field_names = [f["name"] for f in schema["fields"]]
            if len(field_names) != len(set(field_names)): st.error("Field names must be unique!")
            elif any(not f.get("values") for f in schema["fields"]): st.error("All fields must have at least one allowed value!")
            else: st.success("Schema saved!"); st.rerun()

    with classify_tab: # Classification Execution UI (Uses DocumentClassifier which handles backend)
        st.subheader("Classify Documents")
        with st.expander("Current Schema", expanded=True):
            schema = st.session_state.classification_schema
            field_data = [{"Name": f["name"], "Description": f["description"], "Values": ", ".join(f.get("values", [])), "Multi-label": "✓" if f["name"] in schema.get("multi_label_fields", []) else ""} for f in schema["fields"]]
            st.dataframe(field_data, hide_index=True); st.caption(f"Instructions: {schema.get('user_instructions', '')}")
        st.subheader("Select Documents")
        try: max_chunks_for_names = CONFIG.get("classification", {}).get("max_chunks_for_listing", 1000); all_chunks = query_engine.get_chunks(limit=max_chunks_for_names); doc_names = sorted(list(set(c['metadata'].get('file_name', 'N/A') for c in all_chunks if c['metadata'].get('file_name'))))
        except Exception as e: logger.error(f"Failed to get doc names: {e}"); doc_names = []
        selected_docs = st.multiselect("Select documents (empty=all)", options=doc_names, default=[], key="clf_docs")
        # Model display
        model_display_name = "N/A"
        if IS_OPENROUTER_ACTIVE: model_display_name = CONFIG.get("openrouter", {}).get("classification_model", "N/A")
        elif IS_GEMINI_ACTIVE: model_display_name = CONFIG.get("gemini", {}).get("classification_model", "gemini-2.5-pro-exp-03-25") # Use specific model
        else: model_display_name = CONFIG.get("models", {}).get("extraction_models", {}).get("text_standard", "N/A")
        st.info(f"Using LLM: `{model_display_name}` (Backend: {LLM_BACKEND.upper()})")
        batch_size = st.slider("Max Chunks to Classify", 10, 5000, 500, 10, key="clf_batch")
        classify_btn = st.button("Run Classification", type="primary", key="clf_run"); classification_progress = st.empty()
        if classify_btn:
            if not DocumentClassifier: st.error("DocumentClassifier module not loaded."); return
            if not schema["fields"] or any(not f.get("values") for f in schema["fields"]): st.error("Schema invalid or incomplete."); return
            document_classifier = DocumentClassifier() # Uses active manager internally
            with classification_progress.container():
                with st.spinner("Fetching chunks..."):
                    chunks_to_process = [];
                    if selected_docs:
                        limit_per_doc = max(1, batch_size // len(selected_docs)) if selected_docs else batch_size
                        for doc_name in selected_docs: doc_chunks = query_engine.get_chunks(limit=limit_per_doc, document_filter=doc_name); chunks_to_process.extend(doc_chunks);
                    else: chunks_to_process = query_engine.get_chunks(limit=batch_size)
                    chunks_to_process = chunks_to_process[:batch_size]
                    st.info(f"Processing {len(chunks_to_process)} chunks...")
            with classification_progress.container():
                with st.spinner("Classifying documents..."):
                    schema_dict = {f["name"]: {"type": "string", "description": f["description"], "values": f.get("values", [])} for f in schema["fields"]}
                    multi_label_fields = set(schema.get("multi_label_fields", []))
                    results = document_classifier.classify_documents(chunks=chunks_to_process, schema=schema_dict, multi_label_fields=multi_label_fields, user_instructions=schema.get("user_instructions", ""))
                    st.session_state.classification_results = results
                    if results: st.success(f"Successfully classified {len(results)} chunks!")
                    else: st.warning("No documents classified.")

    with results_tab: # Results Display (Unchanged)
        st.subheader("Classification Results"); results = st.session_state.classification_results
        if not results: st.info("No classification results available."); return
        if results:
            class_fields = sorted(list(set(k for r in results if "classification" in r for k in r["classification"])))
            display_data = []
            for result in results:
                row = {"Document": result.get("file_name", "N/A"), "Page": result.get("page_num", "N/A"), "Text": result.get("text", "")[:100] + "..."}
                classification = result.get("classification", {});
                for field in class_fields: value = classification.get(field); row[field] = ", ".join(value) if isinstance(value, list) else value if value is not None else ""
                display_data.append(row)
            df = pd.DataFrame(display_data)
            unique_docs = set(r["file_name"] for r in results if "file_name" in r)
            st.info(f"Found {len(results)} classified chunks from {len(unique_docs)} documents")
            if not df.empty:
                csv_data = df.to_csv(index=False).encode('utf-8'); st.download_button("Export Results as CSV", csv_data, "classification_results.csv", "text/csv", key="clf_export")
                st.dataframe(df, hide_index=True)
                st.subheader("Detailed View"); result_index = st.selectbox("Select result to view", range(len(results)), key="clf_view_select")
                if result_index is not None: selected = results[result_index]; st.text_area("Full Text", selected.get("text", ""), height=200, disabled=True); st.json(selected.get("classification", {}))
            else: st.warning("No data to display.")

# --- END OF REWRITTEN app_pages.py ---