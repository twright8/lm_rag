import sys
from pathlib import Path
import yaml
import streamlit as st
import gc
import torch
import traceback
import logging # Use standard logging here

# --- Core Paths and Configuration ---
try:
    # Assume app_setup.py is in ROOT_DIR/src/ui/
    CURRENT_DIR = Path(__file__).resolve().parent
    # Go up two levels from src/ui/ -> project root
    ROOT_DIR = CURRENT_DIR.parent.parent
    if not (ROOT_DIR / "config.yaml").exists():
        # Try alternative common structure: ROOT_DIR/app.py (go up one level)
        ROOT_DIR = CURRENT_DIR.parent
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
    logger = setup_logger(__name__) # Use standard logger setup
    logger.info(f"Project Root Directory set to: {ROOT_DIR}")
    logger.info("Configuration loaded successfully.")
except ImportError as e:
     # Fallback basic logging if setup_logger fails
     logging.basicConfig(level=logging.INFO)
     logger = logging.getLogger(__name__)
     logger.error(f"Failed to import setup_logger: {e}. Using basic logging.")
     logger.info(f"Project Root Directory set to: {ROOT_DIR}")
except Exception as e:
     logging.basicConfig(level=logging.INFO)
     logger = logging.getLogger(__name__)
     logger.error(f"Error setting up logger: {e}. Using basic logging.")
     logger.info(f"Project Root Directory set to: {ROOT_DIR}")


# --- LLM Backend Selection and Initialization ---
LLM_BACKEND = CONFIG.get("llm_backend", "aphrodite").lower() # Default to aphrodite
IS_OPENROUTER_ACTIVE = (LLM_BACKEND == "openrouter")
IS_GEMINI_ACTIVE = (LLM_BACKEND == "gemini") # Add Gemini flag
logger.info(f"Selected LLM Backend: {LLM_BACKEND.upper()}")

# --- Conditional Aphrodite Service Import ---
APHRODITE_SERVICE_AVAILABLE = False
AphroditeService = None
get_service = None
if LLM_BACKEND == "aphrodite": # Check specific backend
    try:
        from src.utils.aphrodite_service import get_service, AphroditeService
        APHRODITE_SERVICE_AVAILABLE = True
        logger.info("Aphrodite service modules imported (backend is Aphrodite).")
    except ImportError as e:
        logger.error(f"Failed to import AphroditeService: {e}. Local LLM features will be disabled.")
        LLM_BACKEND = "none" # Indicate failure
else:
    logger.info(f"Aphrodite service modules not imported (backend is {LLM_BACKEND.upper()}).")

# --- Conditional OpenRouter Manager Import ---
OPENROUTER_MANAGER_AVAILABLE = False
OpenRouterManager = None
if IS_OPENROUTER_ACTIVE:
    try:
        from src.utils.openrouter_manager import OpenRouterManager
        OPENROUTER_MANAGER_AVAILABLE = True
        logger.info("OpenRouterManager module imported (backend is OpenRouter).")
    except ImportError as e:
        logger.error(f"Failed to import OpenRouterManager: {e}. OpenRouter features will be disabled.")
        IS_OPENROUTER_ACTIVE = False # Disable if import fails
        LLM_BACKEND = "none" # Indicate no backend available
else:
    logger.info(f"OpenRouterManager module not imported (backend is {LLM_BACKEND.upper()}).")

# --- Conditional Gemini Manager Import ---
GEMINI_MANAGER_AVAILABLE = False
GeminiManager = None
if IS_GEMINI_ACTIVE:
    try:
        from src.utils.gemini_manager import GeminiManager
        GEMINI_MANAGER_AVAILABLE = True
        logger.info("GeminiManager module imported (backend is Gemini).")
    except ImportError as e:
        logger.error(f"Failed to import GeminiManager: {e}. Gemini features will be disabled.")
        IS_GEMINI_ACTIVE = False # Disable if import fails
        LLM_BACKEND = "none" # Indicate no backend available
else:
    logger.info(f"GeminiManager module not imported (backend is {LLM_BACKEND.upper()}).")


# --- Core Module Imports (AFTER PATH SETUP) ---
try:
    from src.core.query_system.query_engine import QueryEngine
    from src.utils.conversation_store import ConversationStore
    from src.utils.resource_monitor import log_memory_usage # Keep for state init logging
except ImportError as e:
     logger.error(f"Failed to import core modules: {e}", exc_info=True)
     st.error(f"Core application components failed to load: {e}. Please check installation and paths.")
     st.stop()


# --- Active LLM Manager Initialization ---
def initialize_active_llm_manager():
    """Initializes and stores the active LLM manager instance based on config."""
    if "active_llm_manager" in st.session_state and st.session_state.active_llm_manager is not None:
        logger.debug("Active LLM manager already initialized.")
        return st.session_state.active_llm_manager

    manager = None
    if IS_OPENROUTER_ACTIVE:
        if OPENROUTER_MANAGER_AVAILABLE:
            logger.info("Initializing OpenRouterManager...")
            try:
                if not CONFIG.get("openrouter", {}).get("api_key"):
                    logger.error("OpenRouter API key is missing in config.yaml.")
                    st.error("OpenRouter API key is missing. Please add it in config.yaml.")
                    manager = None
                else:
                    manager = OpenRouterManager(CONFIG["openrouter"])
                    if not manager.client:
                        logger.error("OpenRouterManager client initialization failed.")
                        st.error("Failed to initialize OpenRouter client. Check API key and connection.")
                        manager = None
                    else:
                        logger.info("OpenRouterManager initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize OpenRouterManager: {e}", exc_info=True)
                st.error(f"Error initializing OpenRouter: {e}")
                manager = None
        else:
            logger.error("OpenRouter backend selected, but manager module failed to import.")
            st.error("OpenRouter backend selected, but manager failed to load.")
            manager = None
    elif IS_GEMINI_ACTIVE: # Add Gemini case
        if GEMINI_MANAGER_AVAILABLE:
            logger.info("Initializing GeminiManager...")
            try:
                if not CONFIG.get("gemini", {}).get("api_key"):
                    logger.error("Gemini API key is missing in config.yaml.")
                    st.error("Gemini API key is missing. Please add it in config.yaml.")
                    manager = None
                else:
                    manager = GeminiManager(CONFIG["gemini"])
                    if not manager.client: # Check if client initialization failed
                        logger.error("GeminiManager client initialization failed.")
                        st.error("Failed to initialize Gemini client. Check API key and connection.")
                        manager = None
                    else:
                        logger.info("GeminiManager initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize GeminiManager: {e}", exc_info=True)
                st.error(f"Error initializing Gemini: {e}")
                manager = None
        else:
            logger.error("Gemini backend selected, but manager module failed to import.")
            st.error("Gemini backend selected, but manager failed to load.")
            manager = None
    elif LLM_BACKEND == "aphrodite": # Check specific backend name
        if APHRODITE_SERVICE_AVAILABLE:
            logger.info("Initializing AphroditeService instance...")
            manager = get_service() # Get the singleton Aphrodite service instance
            logger.info("Using AphroditeService instance.")
        else:
            logger.error("Aphrodite backend selected, but service module failed to import.")
            st.error("Aphrodite backend selected, but service failed to load.")
            manager = None
    else:
        logger.error(f"No valid LLM backend configured or available: {LLM_BACKEND}. Check config.yaml.")
        st.error(f"No LLM backend ({LLM_BACKEND}) is available. Please check configuration and installations.")
        manager = None

    st.session_state.active_llm_manager = manager
    return manager

def get_active_llm_manager():
    """Retrieves the initialized active LLM manager instance."""
    if "active_llm_manager" not in st.session_state:
        logger.warning("Attempted to get LLM manager before initialization.")
        return initialize_active_llm_manager() # Attempt initialization if not found
    return st.session_state.active_llm_manager


# --- Application State Initialization ---
def initialize_app_state():
    """
    Initialize the Streamlit application state.
    Handles backend-specific state initialization.
    """
    global IS_OPENROUTER_ACTIVE, IS_GEMINI_ACTIVE # Ensure global flags are accessible

    # Initialize the active LLM manager first
    initialize_active_llm_manager()

    # --- Aphrodite Status Sync (Only if Aphrodite is the backend) ---
    aphrodite_service_restored = False
    llm_model_restored = False
    if LLM_BACKEND == "aphrodite" and APHRODITE_SERVICE_AVAILABLE: # Check specific backend
        try:
            service = get_service() # Should return the singleton instance
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
                # If service isn't running, clear related state
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
    st.session_state.setdefault("query_engine", None)
    st.session_state.setdefault("collection_info", {"exists": False, "points_count": 0}) # Init collection info
    st.session_state.setdefault("selected_llm_model_name", None) # Tracks model selected during upload/processing

    # Set/Reset Aphrodite states based on checks above or default if service unavailable/not active
    st.session_state.setdefault("aphrodite_service_running", aphrodite_service_restored if LLM_BACKEND == "aphrodite" else False)
    st.session_state.setdefault("llm_model_loaded", llm_model_restored if LLM_BACKEND == "aphrodite" else False) # API backends don't "load" models locally
    st.session_state.setdefault("aphrodite_process_info", None if LLM_BACKEND != "aphrodite" or not aphrodite_service_restored else st.session_state.get("aphrodite_process_info"))

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
        f"Initialized session state: Backend={LLM_BACKEND.upper()}, "
        f"Aphrodite Running={st.session_state.aphrodite_service_running}, "
        f"Aphrodite Model Loaded={st.session_state.llm_model_loaded}, "
        f"Collection Exists={st.session_state.collection_info.get('exists')}"
    )
    log_memory_usage(logger, "Memory usage after state initialization")


# --- Query Engine Management ---
def get_or_create_query_engine():
    """
    Get existing query engine or create a new one.
    Sets the engine's model based on session state if available.
    Updates collection info in session state.
    Injects the active LLM manager.
    """
    if "query_engine" not in st.session_state or st.session_state.query_engine is None:
        logger.info("Creating new QueryEngine instance.")
        try:
            # Pass config and root_dir to constructor
            st.session_state.query_engine = QueryEngine(config=CONFIG, root_dir=ROOT_DIR)

            # Set the LLM model name *used by the active backend* (for display/logging)
            llm_name_to_set = None
            if IS_OPENROUTER_ACTIVE:
                llm_name_to_set = CONFIG.get("openrouter", {}).get("chat_model")
                logger.info(f"QueryEngine created, using OpenRouter backend (Chat Model: {llm_name_to_set})")
            elif IS_GEMINI_ACTIVE: # Add Gemini case
                llm_name_to_set = CONFIG.get("gemini", {}).get("chat_model")
                logger.info(f"QueryEngine created, using Gemini backend (Chat Model: {llm_name_to_set})")
            else: # Aphrodite backend
                selected_model = st.session_state.get("selected_llm_model_name")
                if selected_model:
                    llm_name_to_set = selected_model
                    logger.info(f"QueryEngine created, using Aphrodite backend (Session LLM: {llm_name_to_set})")
                else:
                    llm_name_to_set = CONFIG["models"]["chat_model"] # Fallback to Aphrodite default
                    logger.info(f"QueryEngine created, using Aphrodite backend (Default LLM: {llm_name_to_set})")

            # Store the determined LLM name in the query engine instance
            if llm_name_to_set:
                st.session_state.query_engine.llm_model_name = llm_name_to_set
            else:
                 logger.warning("Could not determine LLM model name for QueryEngine.")

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

    # Sync Aphrodite status (only if Aphrodite is active)
    if LLM_BACKEND == "aphrodite" and APHRODITE_SERVICE_AVAILABLE: # Check specific backend
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

# --- END OF MODIFIED app_setup.py ---