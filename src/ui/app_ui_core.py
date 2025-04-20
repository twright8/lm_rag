# app_ui_core.py
import streamlit as st

# Import necessary functions/variables from other modules
from app_setup import CONFIG, logger, APHRODITE_SERVICE_AVAILABLE, get_service
from app_chat import start_aphrodite_service, terminate_aphrodite_service # For sidebar buttons
from app_processing import clear_all_data # For sidebar button
from src.utils.resource_monitor import get_gpu_info # For sidebar status

# --- Styling ---
def apply_custom_styling():
    """
    Apply custom styling to the Streamlit app.
    """
    # Set custom theme colors from config
    theme_color = CONFIG.get("ui", {}).get("theme_color", "#4e8cff") # Provide defaults
    secondary_color = CONFIG.get("ui", {}).get("secondary_color", "#3a7be8")
    accent_color = CONFIG.get("ui", {}).get("accent_color", "#ff6b6b")

    # Apply custom CSS
    st.markdown(f"""
    <style>
    iframe {{
        min-height: 900px; /* Ensure graph iframes have enough height */
        height: 700px !important; /* Force height for PyVis */
    }}
    .main {{
        background-color: #FFFFFF;
    }}
    .stApp {{
        max-width: 1900px;
        margin: 0 auto;
    }}
    .sidebar .sidebar-content {{
        background-color: {theme_color}; /* Use theme color for sidebar */
    }}
    h1, h2, h3 {{
        color: {theme_color}; /* Use theme color for headers */
    }}
    .stButton>button {{
        /* Default button style */
    }}
    .stButton>button[kind="primary"] {{
        background-color: {theme_color}; /* Primary buttons use theme color */
        color: white;
        border: none;
    }}
    .stButton>button[kind="primary"]:hover {{
        background-color: {secondary_color};
        color: white;
    }}
    .stButton>button[kind="secondary"] {{
        /* Style secondary buttons if needed */
        border: 1px solid #d3d3d3;
    }}
    .stButton>button[kind="secondary"]:hover {{
        border: 1px solid {theme_color};
        color: {theme_color};
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

    /* Thinking box styling from query page */
    .thinking-box {{
        background-color: #f0f7ff;
        border-left: 5px solid #2196F3;
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 4px;
        font-family: monospace;
        max-height: 300px;
        overflow-y: auto;
        white-space: pre-wrap; /* Ensure line breaks are respected */
        word-wrap: break-word; /* Ensure long lines wrap */
    }}
    .thinking-title {{
        font-weight: bold;
        color: #2196F3;
        margin-bottom: 8px;
    }}
    /* Download button styling from cluster map */
    .btn {{
        display: inline-block;
        padding: 8px 16px;
        margin: 5px 0;
        background-color: {theme_color}; /* Use theme color */
        color: white;
        text-decoration: none;
        border-radius: 4px;
        border: none;
        font-weight: 500;
        cursor: pointer;
        text-align: center;
        transition: background-color 0.3s;
    }}
    .btn:hover {{
        background-color: {secondary_color}; /* Use secondary color */
    }}
    /* Hide datamapplot progress */
    .datamapplot-progress-container, .progress-label {{
        display: none !important;
    }}
    /* Ensure datamapplot tooltips are visible */
    .datamapplot-tooltip {{
        opacity: 0.95 !important;
        pointer-events: none !important;
        z-index: 9999 !important;
        position: absolute !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.15) !important;
    }}
    /* Relationship graph tab styling */
    .stTabs [data-baseweb="tab-list"] {{ gap: 24px; }}
    .stTabs [data-baseweb="tab"] {{ height: 50px; white-space: pre-wrap; background-color: #F0F2F6; border-radius: 4px 4px 0px 0px; gap: 8px; padding: 10px; }}
    .stTabs [aria-selected="true"] {{ background-color: #FFFFFF; }}

    </style>
    """, unsafe_allow_html=True)

# --- Header ---
def render_header():
    """
    Render the application header.
    """
    st.title("🔍 Tomni AI - Anti-corruption toolset")
    st.markdown("""
    A semantic search and analysis system for anti-corruption investigations.
    Upload documents, extract entities and relationships, and query using natural language.
    """)
    st.markdown("---") # Add a horizontal rule

# --- Sidebar ---
def render_sidebar():
    """
    Render the sidebar with navigation and system status.
    Returns the selected page name.
    """
    with st.sidebar:
        st.title("Navigation")

        # Define page options
        page_options = [
            "Upload & Process",
            "Explore Data",
            "Query System",
            "Topic Filter",
            "Information Extraction",
            "Cluster Map",
            "Document Classification",
            "Settings"
        ]

        # Determine default page index
        default_page = CONFIG.get("ui", {}).get("default_page", "Upload & Process")
        try:
            default_index = page_options.index(default_page)
        except ValueError:
            logger.warning(f"Default page '{default_page}' not found in options. Defaulting to first page.")
            default_index = 0

        # Navigation radio buttons
        page = st.radio(
            "Select Page",
            options=page_options,
            index=default_index,
            key="page_selection" # Add a key for potential state access if needed
        )

        st.markdown("---")

        # System status
        st.subheader("System Status")

        # Resource monitoring section
        try:
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
        except Exception as e:
            logger.warning(f"Could not display GPU info: {e}")
            st.info("GPU status unavailable.")


        # Collection info from Qdrant (using session state)
        collection_info = st.session_state.get("collection_info", {"exists": False, "points_count": 0})
        count = collection_info.get('points_count', 0)
        if collection_info.get('exists', False):
             st.info(f"Documents indexed: {count}")
        else:
             st.warning("Vector DB collection not found.")
             if collection_info.get("error"):
                  st.caption(f"DB Error: {collection_info['error']}")

        # LLM status section
        st.subheader("LLM Status")

        if not APHRODITE_SERVICE_AVAILABLE:
            st.error("LLM Service Module Unavailable")
        else:
            # Display current status based on session state (synced by app_setup)
            is_service_running = st.session_state.get("aphrodite_service_running", False)
            is_model_loaded = st.session_state.get("llm_model_loaded", False)
            process_info = st.session_state.get("aphrodite_process_info")
            current_model = process_info.get("model_name") if process_info else "N/A"

            if is_service_running:
                st.success("🟢 LLM service running")
                if is_model_loaded:
                    st.info(f"✓ Model loaded: {current_model}")
                else:
                    st.warning("⚠️ LLM service running but no model loaded")

                # Add terminate button
                if st.button("Stop LLM Service", type="secondary", key="stop_llm_sidebar"):
                    with st.spinner("Stopping LLM service..."):
                        if terminate_aphrodite_service(): # Function from app_chat
                             st.success("LLM service terminated")
                        else:
                             st.error("Failed to terminate LLM service.")
                    st.rerun() # Rerun to update status display
            else:
                st.warning("🔴 LLM service not running")

                # Add start button
                if st.button("Start LLM Service", type="primary", key="start_llm_sidebar"):
                    with st.spinner("Starting LLM service..."):
                        if start_aphrodite_service(): # Function from app_chat
                            st.success("LLM service started")
                        else:
                             st.error("Failed to start LLM service.")
                    st.rerun() # Rerun to update status display

        # Actions section
        st.markdown("---")
        st.subheader("Actions")

        if st.button("Clear All Data", key="clear_data_sidebar", type="secondary"):
            # Confirmation dialog
            confirm = st.checkbox("Confirm data deletion?", value=False, key="confirm_delete")
            if confirm:
                with st.spinner("Clearing all data..."):
                    clear_all_data() # Function from app_processing
                st.success("All data cleared successfully!")
                # Uncheck confirmation box after deletion
                st.session_state.confirm_delete = False
                st.rerun()
            elif st.session_state.get("confirm_delete"): # Check if button was clicked without confirmation
                 st.warning("Please check the confirmation box to proceed with data deletion.")


        # Add credits
        st.markdown("---")
        st.caption("© 2024 Anti-Corruption RAG System")

    return page # Return the selected page name