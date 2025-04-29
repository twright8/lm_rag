# --- START OF MODIFIED app_ui_core.py ---

# app_ui_core.py
import streamlit as st
import time # Import time for potential key generation if needed elsewhere

# Import necessary functions/variables from other modules
from app_setup import CONFIG, logger, APHRODITE_SERVICE_AVAILABLE, IS_OPENROUTER_ACTIVE, IS_GEMINI_ACTIVE, LLM_BACKEND # Import new flags
# Import functions conditionally or handle absence
if LLM_BACKEND == "aphrodite" and APHRODITE_SERVICE_AVAILABLE:
    from app_chat import start_aphrodite_service, terminate_aphrodite_service
else:
    # Define dummy functions if Aphrodite is not active/available
    def start_aphrodite_service(): st.error("Aphrodite backend not active."); return False
    def terminate_aphrodite_service(): st.error("Aphrodite backend not active."); return False

from app_processing import clear_all_data # For sidebar button
from src.utils.resource_monitor import get_gpu_info # For sidebar status

# --- Styling ---
def apply_custom_styling():
    """ Apply custom styling to the Streamlit app. (Unchanged) """
    theme_color = CONFIG.get("ui", {}).get("theme_color", "#4e8cff")
    secondary_color = CONFIG.get("ui", {}).get("secondary_color", "#3a7be8")
    accent_color = CONFIG.get("ui", {}).get("accent_color", "#ff6b6b")
    st.markdown(f"""
    <style>
    iframe {{ min-height: 900px; height: 700px !important; }}
    .main {{ background-color: #FFFFFF; }}
    .stApp {{ max-width: 1900px; margin: 0 auto; }}
    .sidebar .sidebar-content {{ background-color: {theme_color}; }}
    h1, h2, h3 {{ color: {theme_color}; }}
    .stButton>button[kind="primary"] {{ background-color: {theme_color}; color: white; border: none; }}
    .stButton>button[kind="primary"]:hover {{ background-color: {secondary_color}; color: white; }}
    .stButton>button[kind="secondary"] {{ border: 1px solid #d3d3d3; }}
    .stButton>button[kind="secondary"]:hover {{ border: 1px solid {theme_color}; color: {theme_color}; }}
    .status-box {{ padding: 15px; border-radius: 5px; margin-bottom: 15px; }}
    .info-box {{ background-color: #f0f2f6; padding: 15px; border-radius: 5px; margin-bottom: 15px; }}
    .highlight {{ background-color: #ffffcc; padding: 3px; border-radius: 3px; }}
    .node-PERSON {{ background-color: {CONFIG.get("visualization", {}).get("node_colors", {}).get("PERSON", "#5DA5DA")}; border-radius: 50%; padding: 10px; color: white; text-align: center; }}
    .node-ORGANIZATION {{ background-color: {CONFIG.get("visualization", {}).get("node_colors", {}).get("ORGANIZATION", "#FAA43A")}; border-radius: 50%; padding: 10px; color: white; text-align: center; }}
    .node-GOVERNMENT_BODY {{ background-color: {CONFIG.get("visualization", {}).get("node_colors", {}).get("GOVERNMENT_BODY", "#60BD68")}; border-radius: 50%; padding: 10px; color: white; text-align: center; }}
    .node-COMMERCIAL_COMPANY {{ background-color: {CONFIG.get("visualization", {}).get("node_colors", {}).get("COMMERCIAL_COMPANY", "#F17CB0")}; border-radius: 50%; padding: 10px; color: white; text-align: center; }}
    .node-LOCATION {{ background-color: {CONFIG.get("visualization", {}).get("node_colors", {}).get("LOCATION", "#B2912F")}; border-radius: 50%; padding: 10px; color: white; text-align: center; }}
    .node-POSITION {{ background-color: {CONFIG.get("visualization", {}).get("node_colors", {}).get("POSITION", "#B276B2")}; border-radius: 50%; padding: 10px; color: white; text-align: center; }}
    .node-MONEY {{ background-color: {CONFIG.get("visualization", {}).get("node_colors", {}).get("MONEY", "#DECF3F")}; border-radius: 50%; padding: 10px; color: black; text-align: center; }}
    .node-ASSET {{ background-color: {CONFIG.get("visualization", {}).get("node_colors", {}).get("ASSET", "#F15854")}; border-radius: 50%; padding: 10px; color: white; text-align: center; }}
    .node-EVENT {{ background-color: {CONFIG.get("visualization", {}).get("node_colors", {}).get("EVENT", "#4D4D4D")}; border-radius: 50%; padding: 10px; color: white; text-align: center; }}
    .node-Unknown {{ background-color: {CONFIG.get("visualization", {}).get("node_colors", {}).get("Unknown", "#999999")}; border-radius: 50%; padding: 10px; color: white; text-align: center; }}
    .thinking-box {{ background-color: #f0f7ff; border-left: 5px solid #2196F3; padding: 15px; margin-bottom: 15px; border-radius: 4px; font-family: monospace; max-height: 300px; overflow-y: auto; white-space: pre-wrap; word-wrap: break-word; }}
    .thinking-title {{ font-weight: bold; color: #2196F3; margin-bottom: 8px; }}
    .btn {{ display: inline-block; padding: 8px 16px; margin: 5px 0; background-color: {theme_color}; color: white; text-decoration: none; border-radius: 4px; border: none; font-weight: 500; cursor: pointer; text-align: center; transition: background-color 0.3s; }}
    .btn:hover {{ background-color: {secondary_color}; }}
    .datamapplot-progress-container, .progress-label {{ display: none !important; }}
    .datamapplot-tooltip {{ opacity: 0.95 !important; pointer-events: none !important; z-index: 9999 !important; position: absolute !important; box-shadow: 0 2px 10px rgba(0,0,0,0.15) !important; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 24px; }}
    .stTabs [data-baseweb="tab"] {{ height: 50px; white-space: pre-wrap; background-color: #F0F2F6; border-radius: 4px 4px 0px 0px; gap: 8px; padding: 10px; }}
    .stTabs [aria-selected="true"] {{ background-color: #FFFFFF; }}
    </style>
    """, unsafe_allow_html=True)

# --- Header ---
def render_header():
    """ Render the application header. (Unchanged) """
    st.title("🔍 Tomni AI - Anti-corruption toolset")
    st.markdown("""
    A semantic search and analysis system for anti-corruption investigations.
    Upload documents, extract entities and relationships, and query using natural language.
    """)
    st.markdown("---")

# --- Sidebar ---
def render_sidebar():
    """
    Render the sidebar with navigation and system status, adapting to the active LLM backend.
    Includes a robust two-step confirmation for data clearing.
    """
    with st.sidebar:
        st.title("Navigation")
        page_options = ["Upload & Process", "Explore Data", "Query System", "Topic Filter", "Information Extraction", "Cluster Map", "Document Classification", "Settings"]
        default_page = CONFIG.get("ui", {}).get("default_page", "Upload & Process")
        try: default_index = page_options.index(default_page)
        except ValueError: logger.warning(f"Default page '{default_page}' not found."); default_index = 0
        page = st.radio("Select Page", options=page_options, index=default_index, key="page_selection")
        st.markdown("---")

        # System status
        st.subheader("System Status")

        # Resource monitoring (GPU only relevant for Aphrodite)
        if LLM_BACKEND == "aphrodite": # Check specific backend
            try:
                gpu_info = get_gpu_info()
                if gpu_info:
                    # Use reserved memory as a better indicator of total GPU usage by the process
                    vram_used = gpu_info.get("memory_reserved", 0); vram_total = gpu_info.get("memory_total", 1)
                    if vram_total > 0: st.progress((vram_used / vram_total), f"VRAM: {vram_used:.1f}GB / {vram_total:.1f}GB")
                    else: st.info("GPU detected, but total memory reported as 0.")
                else: st.info("No GPU detected or CUDA not available.")
            except Exception as e: logger.warning(f"Could not display GPU info: {e}"); st.info("GPU status unavailable.")
        else:
            st.info("GPU monitoring disabled (API backend).")


        # Collection info (remains the same)
        collection_info = st.session_state.get("collection_info", {"exists": False, "points_count": 0})
        count = collection_info.get('points_count', 0)
        if collection_info.get('exists', False): st.info(f"Documents indexed: {count}")
        else:
             st.warning("Vector DB collection not found.")
             if collection_info.get("error"): st.caption(f"DB Error: {collection_info['error']}")

        # LLM status section (Adapts based on backend)
        st.subheader("LLM Backend Status")

        if IS_OPENROUTER_ACTIVE:
            st.success(f"🔵 Backend: OpenRouter API")
            or_config = CONFIG.get("openrouter", {})
            chat_model = or_config.get("chat_model", "N/A")
            api_key_present = bool(or_config.get("api_key"))
            st.info(f"Chat Model: `{chat_model}`")
            if api_key_present: st.success("✓ API Key Present")
            else: st.error("❌ API Key Missing in config!")
            # No start/stop buttons for OpenRouter
        elif IS_GEMINI_ACTIVE: # Add Gemini status
            st.success(f"✨ Backend: Google Gemini API")
            gemini_config = CONFIG.get("gemini", {})
            chat_model = gemini_config.get("chat_model", "N/A")
            api_key_present = bool(gemini_config.get("api_key"))
            st.info(f"Chat Model: `{chat_model}`")
            if api_key_present: st.success("✓ API Key Present")
            else: st.error("❌ API Key Missing in config!")
            # No start/stop buttons for Gemini
        elif LLM_BACKEND == "aphrodite" and APHRODITE_SERVICE_AVAILABLE: # Aphrodite Backend
            st.info(f"⚙️ Backend: Local Aphrodite Service")
            is_service_running = st.session_state.get("aphrodite_service_running", False)
            is_model_loaded = st.session_state.get("llm_model_loaded", False)
            process_info = st.session_state.get("aphrodite_process_info")
            current_model = process_info.get("model_name") if process_info else "N/A"

            if is_service_running:
                st.success("🟢 LLM service running")
                if is_model_loaded: st.info(f"✓ Model loaded: {current_model}")
                else: st.warning("⚠️ Service running but no model loaded")
                if st.button("Stop LLM Service", type="secondary", key="stop_llm_sidebar"):
                    with st.spinner("Stopping LLM service..."):
                        if terminate_aphrodite_service(): st.success("LLM service terminated")
                        else: st.error("Failed to terminate LLM service.")
                    st.rerun()
            else:
                st.warning("🔴 LLM service not running")
                if st.button("Start LLM Service", type="primary", key="start_llm_sidebar"):
                    with st.spinner("Starting LLM service..."):
                        if start_aphrodite_service(): st.success("LLM service started")
                        else: st.error("Failed to start LLM service.")
                    st.rerun()
        else: # No valid backend
             st.error("🔴 No LLM Backend Available")
             st.caption(f"Check configuration (backend: {LLM_BACKEND}) and installations.")


        # Actions section (Modified Clear Data Logic)
        st.markdown("---")
        st.subheader("Actions")

        # Initialize confirmation state if it doesn't exist
        st.session_state.setdefault("show_delete_confirmation", False)

        # Button to initiate deletion
        if st.button("Clear All Data", key="clear_data_sidebar_initiate", type="secondary"):
            st.session_state.show_delete_confirmation = True # Set flag to show confirmation

        # Conditional confirmation section
        if st.session_state.show_delete_confirmation:
            st.warning("**Are you sure?** This will permanently delete the vector database, BM25 index, and all extracted entities/relationships.")
            # Confirmation button
            if st.button("⚠️ Yes, Delete Everything", key="clear_data_sidebar_confirm", type="primary"):
                with st.spinner("Clearing all data..."):
                    success = clear_all_data() # Call the actual clearing function
                if success:
                    st.success("All data cleared successfully!")
                else:
                    st.error("Data clearing failed. Check logs.")
                st.session_state.show_delete_confirmation = False # Hide confirmation after action
                st.rerun() # Rerun to update UI (e.g., collection info)
            # Optional: Add a cancel button
            if st.button("Cancel", key="clear_data_sidebar_cancel"):
                st.session_state.show_delete_confirmation = False # Hide confirmation
                st.rerun()

        st.markdown("---")
        st.caption("© 2024 Anti-Corruption RAG System")

    return page

# --- END OF MODIFIED app_ui_core.py ---