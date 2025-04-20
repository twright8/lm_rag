# app_setup.py
import sys
from pathlib import Path
import yaml
import streamlit as st
import gc
import torch
import traceback

# --- Core Paths and Configuration ---
try:
    # This assumes app_setup.py is in the same directory as the original app.py
    # which is likely ROOT_DIR/streamlit_app/app.py
    # Adjust if your structure is different (e.g., ROOT_DIR/app.py)
    CURRENT_DIR = Path(__file__).resolve().parent
    # Go up three levels from streamlit_app/app/ -> project root
    ROOT_DIR = CURRENT_DIR # Adjust this if app_setup.py is not in ROOT_DIR/streamlit_app/app
    if not (ROOT_DIR / "config.yaml").exists():
        # Try alternative common structure: ROOT_DIR/app.py
        ROOT_DIR = CURRENT_DIR.parent
        if not (ROOT_DIR / "config.yaml").exists():
             # Try another level up if needed
             ROOT_DIR = CURRENT_DIR.parent.parent
             if not (ROOT_DIR / "config.yaml").exists():
                  ROOT_DIR = CURRENT_DIR.parent.parent.parent # Original assumption
                  if not (ROOT_DIR / "config.yaml").exists():
                       raise FileNotFoundError("Could not locate project root directory containing config.yaml")

    sys.path.append(str(ROOT_DIR))

    CONFIG_PATH = ROOT_DIR / "config.yaml"
    with open(CONFIG_PATH, "r") as f:
        CONFIG = yaml.safe_load(f)

    # Create data directories if they don't exist
    EXTRACTED_DATA_PATH = ROOT_DIR / CONFIG["storage"]["extracted_data_path"]
    BM25_DIR = ROOT_DIR / CONFIG["storage"]["bm25_index_path"]
    EXTRACTED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    BM25_DIR.parent.mkdir(parents=True, exist_ok=True)
    (ROOT_DIR / "temp").mkdir(parents=True, exist_ok=True) # Ensure temp dir exists

except Exception as e:
    st.error(f"Fatal Error initializing paths or loading config: {e}")
    st.stop() # Stop execution if basic setup fails

# --- Logger Setup ---
# Import logger setup function AFTER potentially adding ROOT_DIR to path
try:
    from src.utils.logger import setup_logger
    logger = setup_logger(__name__)
    logger.info(f"Project Root Directory set to: {ROOT_DIR}")
    logger.info("Configuration loaded successfully.")
except ImportError as e:
     # Fallback basic logging if setup_logger fails
     import logging
     logging.basicConfig(level=logging.INFO)
     logger = logging.getLogger(__name__)
     logger.error(f"Failed to import setup_logger: {e}. Using basic logging.")
     logger.info(f"Project Root Directory set to: {ROOT_DIR}")
except Exception as e:
     import logging
     logging.basicConfig(level=logging.INFO)
     logger = logging.getLogger(__name__)
     logger.error(f"Error setting up logger: {e}. Using basic logging.")
     logger.info(f"Project Root Directory set to: {ROOT_DIR}")


# --- Conditional Aphrodite Service Import ---
try:
    from src.utils.aphrodite_service import get_service, AphroditeService
    APHRODITE_SERVICE_AVAILABLE = True
    logger.info("Aphrodite service modules imported.")
except ImportError as e:
    APHRODITE_SERVICE_AVAILABLE = False
    get_service = None # Placeholder
    AphroditeService = None # Placeholder
    logger.error(f"Failed to import AphroditeService: {e}. LLM service features will be disabled.")


# --- Core Module Imports (AFTER PATH SETUP) ---
try:
    from src.core.query_system.query_engine import QueryEngine
    from src.utils.conversation_store import ConversationStore
    from src.utils.resource_monitor import log_memory_usage # Keep for state init logging
except ImportError as e:
     logger.error(f"Failed to import core modules: {e}", exc_info=True)
     st.error(f"Core application components failed to load: {e}. Please check installation and paths.")
     st.stop()


# --- Application State Initialization ---
def initialize_app_state():
    """
    Initialize the Streamlit application state.
    Checks for existing Aphrodite process belonging to the current session
    and avoids terminating it during script reruns.
    Sets default session state values.
    """
    # Set page config early (moved from original initialize_app to main app.py)
    # st.set_page_config(...)

    aphrodite_service_restored = False
    llm_model_restored = False
    service_instance = None

    # --- Aphrodite Status Sync ---
    if APHRODITE_SERVICE_AVAILABLE:
        try:
            service = get_service()
            is_service_actually_running = service.is_running()

            if is_service_actually_running:
                status = service.get_status(timeout=5)
                current_model_loaded = status.get("model_loaded", False)
                current_model_name = status.get("current_model")
                current_pid = service.process.pid if service.process else None

                session_model_loaded = st.session_state.get("llm_model_loaded", False)
                session_info = st.session_state.get("aphrodite_process_info")
                session_model_name = session_info.get("model_name") if session_info else None
                session_pid = session_info.get("pid") if session_info else None

                needs_update = (
                    st.session_state.get("aphrodite_service_running") is not True or
                    session_model_loaded != current_model_loaded or
                    session_model_name != current_model_name or
                    session_pid != current_pid
                )

                if needs_update:
                    logger.info(f"Syncing Aphrodite status: Running=True, Model Loaded={current_model_loaded}, Model={current_model_name}, PID={current_pid}")
                    st.session_state.aphrodite_service_running = True
                    st.session_state.llm_model_loaded = current_model_loaded
                    st.session_state.aphrodite_process_info = {
                        "pid": current_pid,
                        "model_name": current_model_name
                    }
                    aphrodite_service_restored = True # Mark as restored based on sync
                    llm_model_restored = current_model_loaded # Mark as restored based on sync
            else:
                if (st.session_state.get("aphrodite_service_running") is True or
                    st.session_state.get("llm_model_loaded") is True or
                    st.session_state.get("aphrodite_process_info") is not None):
                    logger.info("Syncing Aphrodite status: Service not running. Clearing state.")
                    st.session_state.aphrodite_service_running = False
                    st.session_state.llm_model_loaded = False
                    st.session_state.aphrodite_process_info = None

        except Exception as sync_err:
            logger.error(f"Error during Aphrodite status sync: {sync_err}", exc_info=True)
            st.session_state.aphrodite_service_running = False
            st.session_state.llm_model_loaded = False
            st.session_state.aphrodite_process_info = None

    # --- Initialize Standard Session State Variables ---
    st.session_state.setdefault("processing", False)
    st.session_state.setdefault("processing_status", "")
    st.session_state.setdefault("processing_progress", 0.0)
    st.session_state.setdefault("documents", []) # Still needed? Maybe remove if not used directly
    st.session_state.setdefault("query_engine", None)
    st.session_state.setdefault("chat_history", []) # Legacy? ConversationStore handles history now
    st.session_state.setdefault("selected_llm_model_name", None)
    st.session_state.setdefault("collection_info", {"exists": False, "points_count": 0}) # Init collection info

    # Set/Reset Aphrodite states based on checks above or default if service unavailable
    st.session_state.setdefault("aphrodite_service_running", aphrodite_service_restored)
    st.session_state.setdefault("llm_model_loaded", llm_model_restored)
    st.session_state.setdefault("aphrodite_process_info", None if not aphrodite_service_restored else st.session_state.get("aphrodite_process_info"))

    # --- Initialize ConversationStore Singleton ---
    if "conversation_store" not in st.session_state:
        logger.info("Initializing ConversationStore...")
        try:
            st.session_state.conversation_store = ConversationStore(ROOT_DIR)
        except Exception as e:
             logger.error(f"Failed to initialize ConversationStore: {e}", exc_info=True)
             st.error("Failed to initialize conversation storage. Chat functionality may be limited.")
             st.session_state.conversation_store = None # Ensure it's None on failure
    else:
        logger.debug("ConversationStore already initialized.")

    # --- UI State & Active Conversation State ---
    st.session_state.setdefault("ui_chat_display", [])
    st.session_state.setdefault("current_conversation_id", None)
    st.session_state.setdefault("active_conversation_data", None)
    st.session_state.setdefault("retrieval_enabled_for_next_turn", True)

    # --- Initialize Query Engine and update collection info ---
    # This needs to be called after basic state is set up
    get_or_create_query_engine()

    # Log final initial state
    logger.debug(
        f"Initialized session state: service_running={st.session_state.aphrodite_service_running}, "
        f"llm_model_loaded={st.session_state.llm_model_loaded}, "
        f"collection_exists={st.session_state.collection_info.get('exists')}"
    )
    log_memory_usage(logger, "Memory usage after state initialization")


# --- Query Engine Management ---
def get_or_create_query_engine():
    """
    Get existing query engine or create a new one.
    Sets the engine's model based on session state if available.
    Updates collection info in session state.
    """
    if "query_engine" not in st.session_state or st.session_state.query_engine is None:
        logger.info("Creating new QueryEngine instance.")
        try:
            st.session_state.query_engine = QueryEngine()
            # On creation, set its model based on session state (if processing ran) or config default
            selected_model = st.session_state.get("selected_llm_model_name")
            if selected_model:
                st.session_state.query_engine.llm_model_name = selected_model
                logger.info(f"QueryEngine created, using session LLM: {selected_model}")
            else:
                # Falls back to the default chat_model defined in QueryEngine.__init__
                logger.info(f"QueryEngine created, using default LLM: {st.session_state.query_engine.llm_model_name}")
        except Exception as e:
             logger.error(f"Failed to create QueryEngine: {e}", exc_info=True)
             st.error(f"Failed to initialize the query system: {e}")
             st.session_state.query_engine = None
             return None # Return None on failure

    # Update collection info if query engine exists
    if st.session_state.query_engine:
        try:
            collection_info = st.session_state.query_engine.get_collection_info()
            st.session_state.collection_info = collection_info
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            st.session_state.collection_info = {"exists": False, "points_count": 0, "error": str(e)}
    else:
        # Ensure collection info reflects no engine
        st.session_state.collection_info = {"exists": False, "points_count": 0, "error": "Query engine not available"}


    # Sync Aphrodite status (redundant if called after initialize_app_state, but safe)
    if APHRODITE_SERVICE_AVAILABLE:
        try:
            service = get_service()
            is_service_actually_running = service.is_running()
            if st.session_state.get("aphrodite_service_running") != is_service_actually_running:
                 st.session_state.aphrodite_service_running = is_service_actually_running
                 logger.info(f"Query Engine check: Synced service running state to {is_service_actually_running}")

            if is_service_actually_running:
                 status = service.get_status(timeout=5)
                 model_loaded = status.get("model_loaded", False)
                 current_model = status.get("current_model")
                 current_pid = service.process.pid if service.process else None
                 session_info = st.session_state.get("aphrodite_process_info", {})

                 if (st.session_state.get("llm_model_loaded") != model_loaded or
                     session_info.get("model_name") != current_model or
                     session_info.get("pid") != current_pid):

                       st.session_state.llm_model_loaded = model_loaded
                       st.session_state.aphrodite_process_info = {"pid": current_pid, "model_name": current_model}
                       logger.info(f"Query Engine check: Synced model state (Loaded={model_loaded}, Name={current_model}, PID={current_pid})")
            else:
                # Ensure state is cleared if service is not running
                if st.session_state.get("llm_model_loaded") or st.session_state.get("aphrodite_process_info"):
                    st.session_state.llm_model_loaded = False
                    st.session_state.aphrodite_process_info = None
                    logger.info("Query Engine check: Cleared model state as service is not running.")
        except Exception as e:
             logger.warning(f"Query Engine check: Error syncing Aphrodite status: {e}")


    return st.session_state.query_engine

# --- Helper to get ConversationStore instance ---
def get_conversation_store() -> ConversationStore | None:
    """Safely retrieves the ConversationStore instance from session state."""
    store = st.session_state.get("conversation_store")
    if not store:
        logger.error("ConversationStore not found in session state.")
    return store