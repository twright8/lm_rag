# app.py (Main Entry Point)
import streamlit as st
from pathlib import Path
import sys
import multiprocessing
import traceback # Keep for initial error handling

# --- Initial Setup and Module Imports ---
# Attempt to set up basic paths and import the setup module first.
try:
    # Assume app.py is in the same folder as the other app_*.py files
    CURRENT_DIR = Path(__file__).resolve().parent
    # Add current directory to path to find app_*.py modules
    sys.path.append(str(CURRENT_DIR))

    # Import setup functions and variables AFTER adding CURRENT_DIR to path
    from app_setup import (
        ROOT_DIR, CONFIG, logger, initialize_app_state,
        APHRODITE_SERVICE_AVAILABLE # Needed for multiprocessing check potentially
    )
    SETUP_SUCCESS = True
except Exception as e:
    # Fallback if initial setup fails
    SETUP_SUCCESS = False
    st.set_page_config(page_title="Error", layout="wide")
    st.error(f"Fatal Error during application setup: {e}")
    st.error("Could not load core configuration or logger.")
    st.code(traceback.format_exc())
    st.stop() # Halt execution

# --- Import UI and Page Modules (only if setup was successful) ---
if SETUP_SUCCESS:
    try:
        from app_ui_core import apply_custom_styling, render_header, render_sidebar
        from app_pages import (
            render_upload_page,
            render_explore_page,
            render_query_page,
            render_cluster_map_page,
            render_topic_filter_page,
            render_info_extraction_page,
            render_classification_page,
            render_settings_page
        )
        MODULE_IMPORT_SUCCESS = True
    except ImportError as e:
        MODULE_IMPORT_SUCCESS = False
        logger.error(f"Failed to import UI/Page modules: {e}", exc_info=True)
        st.error(f"Application UI components failed to load: {e}")
        st.info("Please ensure all `app_*.py` files are present in the same directory.")
        st.stop()
else:
    MODULE_IMPORT_SUCCESS = False # Should already be stopped, but for clarity


# --- Main Application Logic ---
def main():
    """
    Main application entry point.
    Initializes state, renders UI shell, and routes to the selected page.
    """
    # 1. Initialize State (from app_setup)
    # This handles session state defaults, Aphrodite status sync, query engine creation etc.
    initialize_app_state()

    # 2. Apply Styling (from app_ui_core)
    apply_custom_styling()

    # 3. Render Header (from app_ui_core)
    render_header()

    # 4. Render Sidebar and Get Page Selection (from app_ui_core)
    # The sidebar function now handles its internal logic (status, buttons)
    selected_page = render_sidebar()

    # 5. Page Routing (calls functions from app_pages)
    logger.debug(f"Routing to page: {selected_page}")
    if selected_page == "Upload & Process":
        render_upload_page()
    elif selected_page == "Explore Data":
        render_explore_page()
    elif selected_page == "Query System":
        render_query_page()
    elif selected_page == "Cluster Map":
        render_cluster_map_page()
    elif selected_page == "Topic Filter":
        render_topic_filter_page()
    elif selected_page == "Information Extraction":
        render_info_extraction_page()
    elif selected_page == "Document Classification":
        render_classification_page()
    elif selected_page == "Settings":
        render_settings_page()
    else:
        # Fallback if page name is unexpected
        st.error(f"Unknown page selected: {selected_page}")
        logger.error(f"Attempted to route to unknown page: {selected_page}")
        render_upload_page() # Default to upload page


# --- Application Entry Point ---
if __name__ == "__main__":
    # Ensure setup and module imports were successful before proceeding
    if SETUP_SUCCESS and MODULE_IMPORT_SUCCESS:
        # Set multiprocessing start method (important for libraries like Aphrodite)
        try:
            # Check if the service is available and might need spawn
            needs_spawn = APHRODITE_SERVICE_AVAILABLE # Assume spawn needed if service might be used

            if needs_spawn:
                current_method = multiprocessing.get_start_method(allow_none=True)
                if current_method != 'spawn':
                    logger.info(f"Setting multiprocessing start method to 'spawn'. Current method: {current_method}")
                    # Force=True might be necessary if method was set differently elsewhere
                    multiprocessing.set_start_method('spawn', force=True)
                else:
                    logger.info("Multiprocessing start method already set to 'spawn'.")
            else:
                 logger.info("Skipping multiprocessing start method check (Aphrodite service not available).")

        except Exception as e:
            # Log error but proceed; default method might work or might cause issues later.
            logger.error(f"Could not set multiprocessing start method to 'spawn': {e}. Using default: {multiprocessing.get_start_method(allow_none=True)}")

        # Run the main Streamlit application logic
        try:
            main()
        except Exception as app_err:
             logger.error(f"Unhandled error in main application loop: {app_err}", exc_info=True)
             st.error("An unexpected error occurred in the application.")
             st.code(traceback.format_exc())

    # If setup or imports failed, the errors would have been shown earlier,
    # and st.stop() would have halted execution.