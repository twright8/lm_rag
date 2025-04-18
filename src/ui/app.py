import sys
from pathlib import Path
import streamlit as st
import base64
import io
from datetime import datetime
import networkx as nx
from collections import deque
import uuid
# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

import random
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
from transformers import AutoTokenizer
import traceback # For better error logging
from typing import List, Dict, Any, Union, Optional
# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

# Import core modules AFTER setting path
from src.core.document_processing.document_loader import DocumentLoader
from src.core.document_processing.document_chunker import DocumentChunker
from src.core.extraction.entity_extractor import EntityExtractor
from src.core.indexing.document_indexer import DocumentIndexer
from src.core.visualization.cluster_map import create_download_dataframe
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
    llm_model_restored = False  # Renamed from chat_model_loaded
    service_instance = None

    if APHRODITE_SERVICE_AVAILABLE:
        try:
            service = get_service()
            is_service_actually_running = service.is_running()

            # Sync state based on actual service status
            if is_service_actually_running:
                status = service.get_status(timeout=5)
                current_model_loaded = status.get("model_loaded", False)
                current_model_name = status.get("current_model")
                current_pid = service.process.pid if service.process else None

                # Get current session state values safely
                session_model_loaded = st.session_state.get("llm_model_loaded", False)
                session_info = st.session_state.get("aphrodite_process_info") # Use .get()
                session_model_name = session_info.get("model_name") if session_info else None
                session_pid = session_info.get("pid") if session_info else None

                # Check if any state mismatches require an update
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
                    # Always update/create the info dict when syncing and service is running
                    st.session_state.aphrodite_process_info = {
                        "pid": current_pid,
                        "model_name": current_model_name
                    }
            else:
                # Service is not running, ensure state reflects this
                if (st.session_state.get("aphrodite_service_running") is True or
                    st.session_state.get("llm_model_loaded") is True or
                    st.session_state.get("aphrodite_process_info") is not None):

                    logger.info("Syncing Aphrodite status: Service not running. Clearing state.")
                    st.session_state.aphrodite_service_running = False
                    st.session_state.llm_model_loaded = False
                    st.session_state.aphrodite_process_info = None

        except Exception as sync_err:
            logger.error(f"Error during Aphrodite status sync: {sync_err}", exc_info=True)
            # Attempt to reset state on error to prevent inconsistent state
            st.session_state.aphrodite_service_running = False
            st.session_state.llm_model_loaded = False
            st.session_state.aphrodite_process_info = None

    # --- Initialize standard session state variables using setdefault ---
    # This ensures they exist but doesn't overwrite if they were already set
    st.session_state.setdefault("processing", False)
    st.session_state.setdefault("processing_status", "")
    st.session_state.setdefault("processing_progress", 0.0)
    st.session_state.setdefault("documents", [])
    st.session_state.setdefault("query_engine", None)
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("selected_llm_model_name", None)  # To store the model chosen during processing

    # Set/Reset Aphrodite states based on checks above or default if service unavailable
    st.session_state.setdefault("aphrodite_service_running", aphrodite_service_restored)
    st.session_state.setdefault("llm_model_loaded", llm_model_restored)  # Renamed state variable
    # Ensure aphrodite_process_info is initialized if it wasn't already
    st.session_state.setdefault("aphrodite_process_info", None)

    # --- Initialize ConversationStore Singleton ---
    if "conversation_store" not in st.session_state:
        logger.info("Initializing ConversationStore...")
        from src.utils.conversation_store import ConversationStore
        st.session_state.conversation_store = ConversationStore(ROOT_DIR)
    else:
        logger.debug("ConversationStore already initialized.")

    # UI State for Chat Display
    st.session_state.setdefault("ui_chat_display", [])  # List of messages purely for UI rendering

    # Active Conversation State
    st.session_state.setdefault("current_conversation_id", None)  # ID of the loaded conversation
    st.session_state.setdefault("active_conversation_data", None)  # Holds the entire dict of the loaded conversation

    # Retrieval Control State
    # This flag determines if the *next* query submitted by the user will trigger RAG.
    # It's set True when a new conversation starts, False otherwise, and the user toggle controls it.
    st.session_state.setdefault("retrieval_enabled_for_next_turn", True)

    # Initialize Query Engine and update collection info
    get_or_create_query_engine()  # This also updates collection info

    # Log final initial state
    logger.debug(
        f"Initialized session state: service_running={st.session_state.aphrodite_service_running}, llm_model_loaded={st.session_state.llm_model_loaded}")

def save_current_conversation():
    """Safely saves the currently active conversation data to disk via ConversationStore."""
    conv_id = st.session_state.get("current_conversation_id")
    conv_data = st.session_state.get("active_conversation_data")
    store = st.session_state.get("conversation_store")

    if conv_id and conv_data and store:
        logger.debug(f"Attempting to save conversation {conv_id}...")
        success = store.save_conversation(conv_id, conv_data)
        if success:
            logger.debug(f"Conversation {conv_id} saved successfully.")
        else:
            logger.error(f"Failed to save conversation {conv_id}.")
            # Optionally notify the user in the UI if save fails critically
            # st.toast("Error: Failed to save conversation state.", icon="❌")
    elif conv_id or conv_data:
        logger.warning("Attempted to save conversation but ID, data, or store was missing.")


# In app.py

# Add imports if not already present
from transformers import AutoTokenizer
import traceback # For better error logging

def generate_llm_response(
    llm_manager, query_engine, conversation_history, current_prompt, context_sources,
    message_placeholder, thinking_placeholder
    ) -> Dict[str, Any]:
    """
    Generates response using the LLM (Aphrodite/DeepSeek).
    Handles formatting input appropriately: uses chat templates for Aphrodite via local tokenizer,
    constructs a raw prompt string for DeepSeek API.
    Manages streaming updates to UI placeholders for DeepSeek.

    Args:
        llm_manager: Instance of AphroditeService or DeepSeekManager.
        query_engine: Instance of QueryEngine.
        conversation_history: List of previous message dicts [{'role': ..., 'content': ...}].
        current_prompt: The user's latest prompt string.
        context_sources: List of source dicts from RAG (can be empty).
        message_placeholder: Streamlit st.empty() object for the main answer.
        thinking_placeholder: Streamlit st.empty() object for the thinking process.

    Returns:
        Dictionary containing the *final* state after generation:
        - 'final_answer': The complete generated response string after streaming.
        - 'final_thinking': The complete reasoning string (if any).
        - 'error': Error message string if generation failed.
    """
    logger.info("Initiating LLM response generation...")
    final_data = {"final_answer": "", "final_thinking": "", "error": None}
    tokenizer = None # Initialize tokenizer variable (only used for Aphrodite)
    model_name = None # Store the determined model name
    full_prompt_for_llm = None # Initialize raw prompt string
    messages_for_template = None # Initialize structured messages

    # --- Determine LLM Type and Prepare Input ---
    is_deepseek = hasattr(llm_manager, 'generate') and CONFIG.get("deepseek", {}).get("use_api", False)

    try:
        # --- Prepare Context and History (Common Logic) ---
        max_turns = CONFIG.get("conversation", {}).get("max_history_turns", 5)
        history_subset = conversation_history[-(max_turns * 2):] # Get last N turns

        context_str = ""
        if context_sources:
            logger.info(f"Providing {len(context_sources)} context sources to LLM.")
            context_texts = [src.get('original_text', src.get('text', '')) for src in context_sources]
            context_str = "## Context Documents:\n" + "\n\n".join([f"[{i + 1}] {text}" for i, text in enumerate(context_texts)])
        else:
            logger.info("No context provided to LLM for this turn.")
            context_str = "## Context Documents:\nNo context documents were retrieved or provided for this query."

        # --- Input Formatting: DeepSeek vs Aphrodite ---
        if is_deepseek:
            # --- DeepSeek Path: Construct Raw Prompt String ---
            use_reasoner = CONFIG.get("deepseek", {}).get("use_reasoner", False)
            model_name = CONFIG["deepseek"]["reasoning_model"] if use_reasoner else CONFIG["deepseek"]["chat_model"]
            logger.info(f"Using DeepSeek model: {model_name}. Constructing raw prompt.")

            # Define system prompt (adjust if DeepSeek uses specific system tags)
            # Assuming a simple system prompt block for now.
            system_block = f"""System: You are an expert assistant specializing in anti-corruption investigations and analysis. Your responses should be detailed, factual, and based ONLY on the provided context if available. If the answer is not found in the context or conversation history, state that clearly. Do not make assumptions or use external knowledge. Cite sources using [index] notation where applicable based on the provided context."""

            # Format history into a string block
            history_block = "## Conversation History:\n"
            if history_subset:
                 history_block += "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history_subset if "role" in msg and "content" in msg])
            else:
                 history_block += "No previous conversation history."

            # Construct the final user query block including context
            user_query_block = f"""{context_str}

Based on the conversation history and the context provided above (if any), answer the following question:
Question: {current_prompt}"""

            # Combine all parts into a single string for DeepSeek
            # (Adjust separators like \n\n as needed based on how DeepSeek best parses)
            full_prompt_for_llm = f"{system_block}\n\n{history_block}\n\n{user_query_block}\n\nAssistant:" # Add prompt for assistant turn
            logger.debug(f"Constructed Raw Prompt for DeepSeek (sample):\n{full_prompt_for_llm[:500]}...")

        else:
            # --- Aphrodite Path: Prepare Structured Messages & Load Tokenizer ---
            if not hasattr(llm_manager, 'get_status'):
                 final_data['error'] = "Error: Invalid LLM manager provided (not DeepSeek or Aphrodite)."
                 message_placeholder.error(final_data['error'])
                 return final_data

            status = llm_manager.get_status()
            model_name = status.get("current_model")
            if not model_name:
                 final_data['error'] = "Error: Aphrodite service has no model loaded."
                 message_placeholder.error(final_data['error'])
                 return final_data
            logger.info(f"Using Aphrodite model: {model_name}. Preparing structured messages.")

            # Load tokenizer for the Aphrodite model
            try:
                 tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                 if tokenizer.chat_template is None:
                      logger.error(f"Tokenizer for {model_name} lacks a chat template. Cannot format prompt.")
                      final_data['error'] = f"Error: Tokenizer for {model_name} lacks a chat template."
                      message_placeholder.error(final_data['error'])
                      return final_data
            except Exception as e:
                 logger.error(f"Failed to load tokenizer for {model_name}: {e}", exc_info=True)
                 final_data['error'] = f"Error loading tokenizer for {model_name}: {e}"
                 message_placeholder.error(final_data['error'])
                 return final_data

            # Prepare structured messages for the template
            system_content = """You are an expert assistant specializing in anti-corruption investigations and analysis. Your responses should be detailed, factual, and based ONLY on the provided context if available. If the answer is not found in the context or conversation history, state that clearly. Do not make assumptions or use external knowledge. Cite sources using [index] notation where applicable based on the provided context."""
            user_content = f"""{context_str}

Based on the conversation history and the context provided above (if any), answer the following question:
Question: {current_prompt}"""

            messages_for_template = [{"role": "system", "content": system_content}]
            # Add validated history messages
            messages_for_template.extend([
                {"role": msg["role"], "content": msg["content"]}
                for msg in history_subset if "role" in msg and "content" in msg
            ])
            messages_for_template.append({"role": "user", "content": user_content})

            # Apply the template now
            try:
                full_prompt_for_llm = tokenizer.apply_chat_template(
                    messages_for_template,
                    tokenize=False,
                    add_generation_prompt=True # Crucial for assistant response
                )
                logger.debug(f"Applied chat template for {model_name}. Result length: {len(full_prompt_for_llm)}")
            except Exception as e:
                logger.error(f"Error applying chat template for model {model_name}: {e}", exc_info=True)
                final_data['error'] = f"Error formatting prompt: {str(e)}"
                message_placeholder.error(final_data['error'])
                return final_data

    except Exception as e:
        logger.error(f"Error during input preparation: {e}", exc_info=True)
        final_data['error'] = f"Input Preparation Error: {str(e)}"
        message_placeholder.error(final_data['error'])
        return final_data

    # --- 3. Initiate LLM Call & Handle Streaming/Response ---
    llm_called = False
    try:
        # --- DeepSeek Streaming Path ---
        if is_deepseek:
            llm_called = True
            logger.info("Using DeepSeekManager - initiating streaming generation with raw prompt.")
            if full_prompt_for_llm is None: # Should not happen if logic above is correct
                 final_data['error'] = "Error: Raw prompt for DeepSeek was not generated."
                 message_placeholder.error(final_data['error'])
                 return final_data

            # (Callback definition remains the same)
            full_response_buffer = ""
            full_thinking_buffer = ""
            thinking_displayed = False
            def deepseek_stream_callback(token_or_thinking):
                nonlocal full_response_buffer, full_thinking_buffer, thinking_displayed
                try:
                    if isinstance(token_or_thinking, dict) and token_or_thinking.get("type") == "thinking":
                        thinking_chunk = token_or_thinking.get("content", "")
                        if thinking_chunk:
                            full_thinking_buffer += thinking_chunk
                            if not thinking_displayed and thinking_chunk.strip():
                                thinking_displayed = True
                                with thinking_placeholder.container():
                                    st.markdown('<div class="thinking-title">💭 Reasoning Process (Live):</div>', unsafe_allow_html=True)
                                    st.markdown(f'<div class="thinking-box">{full_thinking_buffer}▌</div>', unsafe_allow_html=True)
                            elif thinking_displayed:
                                with thinking_placeholder.container():
                                    st.markdown('<div class="thinking-title">💭 Reasoning Process (Live):</div>', unsafe_allow_html=True)
                                    st.markdown(f'<div class="thinking-box">{full_thinking_buffer}▌</div>', unsafe_allow_html=True)
                    elif isinstance(token_or_thinking, str):
                        full_response_buffer += token_or_thinking
                        message_placeholder.markdown(full_response_buffer + "▌")
                except Exception as callback_e:
                     logger.error(f"Error in deepseek_stream_callback: {callback_e}", exc_info=True)

            # Pass the raw prompt string to DeepSeekManager's generate method
            aggregated_answer = llm_manager.generate(
                prompt=full_prompt_for_llm, # Pass the raw string
                stream_callback=deepseek_stream_callback
                # messages argument is likely ignored if prompt is provided
            )
            # (Post-streaming finalization remains the same)
            if isinstance(aggregated_answer, str) and "Error:" in aggregated_answer:
                 final_data['error'] = aggregated_answer
                 message_placeholder.error(aggregated_answer)
            else:
                 final_data['final_answer'] = aggregated_answer
                 final_data['final_thinking'] = full_thinking_buffer
                 message_placeholder.markdown(final_data['final_answer'])
                 if thinking_displayed:
                     with thinking_placeholder.container():
                          with st.expander("💭 Reasoning Process", expanded=False):
                              st.markdown(f'<div class="thinking-box">{final_data["final_thinking"]}</div>', unsafe_allow_html=True)

        # --- Aphrodite Non-Streaming Path ---
        elif full_prompt_for_llm is not None: # Check if template was applied successfully for Aphrodite
            llm_called = True
            logger.info("Using AphroditeService - initiating non-streaming generation with templated prompt.")
            # Pass the fully formatted prompt string to generate_chat
            aphrodite_response = llm_manager.generate_chat(prompt=full_prompt_for_llm)

            if aphrodite_response.get("status") == "success":
                final_data['final_answer'] = aphrodite_response.get("result", "").strip()
                message_placeholder.markdown(final_data['final_answer'])
                final_data['final_thinking'] = None
            else:
                final_data['error'] = aphrodite_response.get("error", "Unknown error from Aphrodite service")
                message_placeholder.error(f"Error: {final_data['error']}")
        else:
             # This case handles if prompt generation failed earlier but wasn't caught
             final_data['error'] = "Error: Could not generate final prompt string for LLM."
             message_placeholder.error(final_data['error'])

        # Final validation
        if not final_data['final_answer'] and not final_data['error'] and llm_called:
            final_data['error'] = "Error: LLM returned no answer content."
            message_placeholder.error(final_data['error'])

    except Exception as e:
        logger.error(f"Critical error during LLM generation coordination: {e}", exc_info=True)
        final_data['error'] = f"LLM Generation System Error: {str(e)}"
        final_data['final_answer'] = f"Sorry, a critical error occurred while trying to generate the response."
        message_placeholder.error(f"Error: {final_data['error']}")

    logger.info(f"LLM generation process finished. Final Answer length: {len(final_data.get('final_answer', ''))}. Error: {final_data.get('error')}")
    return final_data

# In app.py

# (Keep save_current_conversation and generate_llm_response functions as previously provided)

# --- Rewritten handle_chat_message function ---
def handle_chat_message(prompt: str, query_engine, llm_manager):
    """
    Orchestrates processing a user's chat message. Handles retrieval based on user
    toggle state AT THE TIME OF SUBMISSION, calls the LLM, updates state, and saves.

    Args:
        prompt: The user's input string.
        query_engine: Instance of QueryEngine.
        llm_manager: Instance of the active LLM service manager.
    """
    logger.info(f"Handling chat message: '{prompt[:50]}...'")

    # --- 1. Check for Active Conversation ---
    if not st.session_state.get("active_conversation_data"):
        logger.warning("handle_chat_message called without active conversation.")
        st.error("Please start or load a conversation first.")
        return

    # --- 2. Get Current Conversation State ---
    try:
        # Work directly with session state for simplicity here, save logic handles persistence
        active_conv_data = st.session_state.active_conversation_data
        # Make a temporary copy of messages for this turn's processing if needed,
        # but append directly to the session state list at the end.
        conversation_history_for_llm = active_conv_data.get("messages", [])[:] # Shallow copy for passing to LLM
    except Exception as e:
         logger.error(f"Failed to access active conversation data: {e}", exc_info=True)
         st.error("Internal error: Could not access conversation data.")
         return

    # --- 3. Determine Retrieval Need for THIS turn ---
    # **** IMPORTANT: Read the state variable's value AS IT WAS WHEN THE USER SUBMITTED ****
    retrieve_now = st.session_state.retrieval_enabled_for_next_turn
    logger.info(f"Retrieval decision for this turn: {'ENABLED' if retrieve_now else 'DISABLED'} (based on checkbox state when prompt was submitted).")
    # **** REMOVED THE PROBLEMATIC LINE: st.session_state.retrieval_enabled_for_next_turn = False ****
    # The reset happens implicitly because the checkbox default value is False on the next run unless clicked again.

    # --- 4. Add User Message to History (Internal State and UI Display) ---
    user_message_id = f"msg_user_{len(active_conv_data.get('messages', [])) + 1}"
    user_message = {
        "role": "user",
        "content": prompt,
        "timestamp": time.time(),
        "id": user_message_id
    }
    # Append to the *actual* session state list
    st.session_state.active_conversation_data["messages"].append(user_message)
    # Also update the list used for immediate UI rendering
    st.session_state.ui_chat_display.append({"role": "user", "content": prompt})
    # Render the user message instantly (this part should be outside the handler, in the main page logic)
    # The rerun triggered by chat_input submission handles display usually.
    # If not, you might need to manually render it *before* calling the handler.

    # --- 5. Perform Retrieval (Conditional) ---
    sources = []
    retrieval_info_msg = "Retrieval skipped (toggle was off)."
    if retrieve_now:
        retrieval_info_msg = "Performing RAG retrieval..."
        logger.info(retrieval_info_msg)
        retrieval_status_placeholder = st.empty() # Display status temporarily
        with retrieval_status_placeholder.status("Retrieving context...", expanded=True):
            try:
                st.write("Searching relevant documents...")
                sources = query_engine.retrieve(prompt)
                retrieval_info_msg = f"Retrieved {len(sources)} relevant sources."
                st.write(retrieval_info_msg)
                logger.info(retrieval_info_msg)
            except Exception as e:
                logger.error(f"Error during retrieval: {e}", exc_info=True)
                retrieval_info_msg = f"Error during retrieval: {e}"
                st.error(retrieval_info_msg)
                sources = []
        time.sleep(0.5)
        retrieval_status_placeholder.empty()
    else:
        logger.info(retrieval_info_msg)
        # Display caption only if skipping, otherwise status handles it
        st.caption(retrieval_info_msg)

    # --- 6. Initiate LLM Response Generation ---
    # Create placeholders *before* the assistant message context
    message_placeholder = st.empty()
    thinking_placeholder = st.empty()
    sources_placeholder = st.empty()

    # Generate response within the assistant's chat message context
    with st.chat_message("assistant"):
        # Re-assign placeholders within this context if needed, or ensure they are accessible
        message_placeholder = st.empty() # Re-scope placeholder inside chat_message
        thinking_placeholder = st.empty()
        sources_placeholder = st.empty()

        with st.spinner("Assistant is thinking..."):
            final_result = generate_llm_response(
                llm_manager=llm_manager,
                query_engine=query_engine,
                # Pass the history *before* adding the potential assistant message
                conversation_history=active_conv_data["messages"][:-1] if active_conv_data["messages"] else [],
                current_prompt=prompt,
                context_sources=sources,
                message_placeholder=message_placeholder,
                thinking_placeholder=thinking_placeholder
            )

    # --- Generation Complete ---
    final_answer = final_result.get("final_answer", "")
    final_thinking = final_result.get("final_thinking")
    error = final_result.get("error")

    # Populate sources expander if needed (outside chat_message context likely better)
    if not error and sources and CONFIG.get("conversation", {}).get("persist_retrieved_context", True):
         with sources_placeholder.expander("View Sources Used", expanded=False):
             # (Same source display logic as before)
             for i, source in enumerate(sources):
                 st.markdown(f"**Source {i + 1} (Score: {source.get('score', 0.0):.2f}):**")
                 st.markdown(f"> {source.get('original_text', source.get('text', ''))}")
                 meta = source.get('metadata', {})
                 st.caption(f"Doc: {meta.get('file_name', 'N/A')} | Page: {meta.get('page_num', 'N/A')} | Chunk: {meta.get('chunk_id', 'N/A')}")
                 st.markdown("---")

    # --- 7. Add Final Assistant Message to Conversation History ---
    assistant_message_id = f"msg_asst_{len(active_conv_data['messages'])}" # ID relates to the turn
    assistant_message = {
        "role": "assistant",
        "content": final_answer if not error else f"Error generating response: {error}",
        "timestamp": time.time(),
        "id": assistant_message_id
    }
    if sources and not error and CONFIG.get("conversation", {}).get("persist_retrieved_context", True):
        assistant_message["used_context"] = [ # Store context used
            # (Same context structuring logic as before)
            {"text": s.get('original_text', s.get('text', '')),
             "metadata": s.get('metadata', {}),
             "score": s.get('score', 0.0),
             "source_index": i+1}
            for i, s in enumerate(sources)
        ]
    if final_thinking:
        assistant_message["thinking_process"] = final_thinking

    # Append the final assistant message to the *actual* session state data
    st.session_state.active_conversation_data["messages"].append(assistant_message)

    # --- 8. Update UI Display List ---
    st.session_state.ui_chat_display.append({ # Add the complete assistant message info
        "role": "assistant",
        "content": assistant_message["content"],
        "thinking": final_thinking,
        "sources": assistant_message.get("used_context") # Use context stored in the message
    })

    # --- 9. Auto-Save Conversation ---
    if CONFIG.get("conversation", {}).get("auto_save_on_turn", True):
        save_current_conversation()

    logger.info("Finished handling chat message cycle.")

    # --- Optional: Rerun may not be needed now ---
    # The state updates should trigger a natural rerun by Streamlit
    # st.rerun()

def render_spreadsheet_options(uploaded_files):
    """
    Render simplified options for processing spreadsheet files.
    Returns a dictionary mapping file names to selected columns.
    """
    if not uploaded_files:
        return {}

    # Check if any spreadsheet files are uploaded
    spreadsheet_files = [f for f in uploaded_files if f.name.lower().endswith(('.csv', '.xlsx', '.xls'))]
    if not spreadsheet_files:
        return {}

    st.subheader("Spreadsheet Processing Options")
    st.info(
        "Spreadsheet files detected. For each file, select which columns to include in the processing. Each row will become one chunk with the selected columns combined.")

    column_selections = {}

    for file in spreadsheet_files:
        with st.expander(f"Configure {file.name}", expanded=True):
            # Sample the file to get column names
            try:
                # Create a temporary file
                temp_file = ROOT_DIR / "temp" / file.name
                with open(temp_file, "wb") as f:
                    f.write(file.getbuffer())

                # Read the column names
                if file.name.lower().endswith('.csv'):
                    df = pd.read_csv(temp_file, nrows=5)  # Just read a few rows to get columns
                else:
                    df = pd.read_excel(temp_file, nrows=5)  # Just read a few rows to get columns

                columns = df.columns.tolist()

                # Display a preview of the data
                st.markdown("#### Data Preview")
                st.dataframe(df.head(3), use_container_width=True)

                # Simple explanation of what happens
                st.markdown("""
                #### Column Selection
                Select which columns to include in processing. Each row will become one separate chunk, 
                with values from the selected columns combined using a pipe separator.

                For example, if you select "Name" and "Description", each row will be formatted as:
                `Name: John Smith  |  Description: Project proposal`
                """)

                # Column selection with "Select All" option
                select_all = st.checkbox(f"Select All Columns for {file.name}", value=True)

                if select_all:
                    selected_columns = columns
                    st.info(f"All {len(columns)} columns will be included")
                else:
                    selected_columns = st.multiselect(
                        f"Select columns to include for {file.name}",
                        options=columns,
                        default=[],
                        key=f"columns_{file.name}"
                    )

                # Store the selection
                column_selections[file.name] = selected_columns

            except Exception as e:
                st.error(f"Error reading {file.name}: {e}")
                column_selections[file.name] = []

    return column_selections


# To be called from render_upload_page
def process_documents_with_spreadsheet_options(uploaded_files, selected_llm_name, vl_pages, vl_process_all,
                                               spreadsheet_options):
    """
    Enhanced process_documents function that handles spreadsheet options.
    This replaces the original process_documents function with support for spreadsheet column selection.
    """
    try:
        # Ensure Aphrodite service is running
        if not APHRODITE_SERVICE_AVAILABLE:
            st.error("Aphrodite service module not available. Cannot process.")
            st.session_state.processing = False
            return
        
        logger.info(f"Starting document processing with spreadsheet options. Files: {[f.name for f in uploaded_files]}")
        
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
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())

                # If this is a spreadsheet and we have options for it, pass them to the loader
                if file.name.lower().endswith(('.csv', '.xlsx', '.xls')) and file.name in spreadsheet_options:
                    logger.info(f"Loading spreadsheet {file.name} with column options: {spreadsheet_options.get(file.name)}")
                    # Use the enhanced loader with spreadsheet options
                    doc = document_loader.load_document_with_options(
                        file_path,
                        {
                            'selected_columns': spreadsheet_options.get(file.name, []),
                            'separator': "  |  ", # Fixed separator
                            'include_column_names': True # Always include column names
                        }
                    )
                    logger.info(f"Loaded spreadsheet {file.name} successfully with options")
                else:
                    logger.info(f"Loading regular document: {file.name}")
                    doc = document_loader.load_document(file_path)

                if doc:
                    documents.append(doc)
                    # Log document details
                    content_count = len(doc.get('content', []))
                    logger.info(f"Document loaded: {file.name}, Content items: {content_count}")
                else:
                    logger.warning(f"Failed to load document from file: {file.name}")
            except Exception as e:
                logger.error(f"Error loading document {file.name}: {e}")
                st.warning(f"Could not load {file.name}, skipping.")

            progress = 0.10 + (0.10 * (i + 1) / len(uploaded_files))
            st.session_state.processing_progress = progress
            st.session_state.processing_status = f"Loaded document {i + 1}/{len(uploaded_files)}: {file.name}"
        logger.info("All documents loaded. Shutting down document loader (Docling) to free resources...")
        if 'document_loader' in locals() and document_loader: # Check if loader exists
            document_loader.shutdown()
            del document_loader # Explicitly delete the reference to potentially help GC
            gc.collect() # Encourage garbage collection
            logger.info("Document loader shut down complete.")
            log_memory_usage(logger) # Optional logging
        else:
             logger.warning("Document loader instance not found or already deleted before shutdown call.")
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
            try:
                logger.info(f"Chunking document {i+1}/{len(documents)}: {doc.get('file_name', 'Unknown')}")
                
                # Check if it's a spreadsheet
                is_spreadsheet = doc.get('file_type', '').lower() in ['.csv', '.xlsx', '.xls']
                if is_spreadsheet:
                    spreadsheet_rows = sum(1 for item in doc.get('content', []) if item.get('is_spreadsheet_row', False))
                    logger.info(f"Processing spreadsheet with {spreadsheet_rows} rows")
                
                doc_chunks = document_chunker.chunk_document(doc)
                chunk_count = len(doc_chunks)
                logger.info(f"Created {chunk_count} chunks for {doc.get('file_name', 'Unknown')}")
                
                all_chunks.extend(doc_chunks)
                progress = 0.25 + (0.15 * (i + 1) / len(documents))
                st.session_state.processing_progress = progress
                st.session_state.processing_status = f"Chunked document {i + 1}/{len(documents)}: {doc.get('file_name', 'Unknown')}"
            except Exception as e:
                logger.error(f"Error chunking document {doc.get('file_name', 'Unknown')}: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        logger.info(f"Shutting down chunking model...")
        document_chunker.shutdown()
        
        if not all_chunks:
            st.error("No chunks were generated.")
            st.session_state.processing = False
            return
            
        logger.info(f"Total chunks generated: {len(all_chunks)}")

        # --- Entity Extraction ---
        st.session_state.processing_status = "Preparing entity extraction..."
        st.session_state.processing_progress = 0.40
        # Use the selected LLM name for extraction
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
        
        # Log extraction details
        logger.info(f"Starting entity extraction on {len(all_chunks)} chunks ({len(visual_chunk_ids)} visual chunks)")
        
        # Process (this ensures the selected_llm_name is loaded in the service)
        entity_extractor.process_chunks(all_chunks, visual_chunk_ids)
        st.session_state.processing_progress = 0.75
        st.session_state.processing_status = "Saving extraction results..."
        entity_extractor.save_results()
        modified_chunks = entity_extractor.get_modified_chunks()
        logger.info(f"Entity extraction complete. Modified chunks: {len(modified_chunks)}")

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
        logger.info("Loading indexing model...")
        document_indexer.load_model()
        
        # Check if collection exists and clear if necessary
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
        logger.info(f"Indexing {len(modified_chunks)} chunks")
        document_indexer.index_documents(modified_chunks)
        document_indexer.shutdown()
        st.session_state.processing_progress = 0.95
        st.session_state.processing_status = "Indexing complete."

        # Complete - 100% progress
        st.session_state.processing_status = "Processing completed successfully!"
        st.session_state.processing_progress = 1.0
        query_engine = get_or_create_query_engine() # Update collection info one last time
        st.session_state.collection_info = query_engine.get_collection_info()
        logger.info("Document processing completed successfully!")

    except Exception as e:
        logger.error(f"Fatal error processing documents: {e}", exc_info=True)
        st.session_state.processing_status = f"Error: {str(e)}"
        st.error(f"An unexpected error occurred: {e}")
    finally:
        st.session_state.processing = False

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
    iframe {{
        min-height:900px
        }}
    .main {{
        background-color: #FFFFFF;
    }}
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
    Render the upload and processing page with spreadsheet options.
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
        "Large Text": CONFIG["models"]["extraction_models"]["text_large"],

    }
    # Use the previously selected model as default if available, else default to small
    default_model_display_name = "Small Text (Faster)"  # Default choice
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
        index=list(model_name_map.keys()).index(default_model_display_name)  # Set default index
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

    # NEW: Spreadsheet Options
    # Only show if uploaded files include spreadsheets
    spreadsheet_options = {}
    if uploaded_files and any(f.name.lower().endswith(('.csv', '.xlsx', '.xls')) for f in uploaded_files):
        spreadsheet_options = render_spreadsheet_options(uploaded_files)

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
                            if start > 0 and end >= start:
                                vl_pages.extend(range(start, end + 1))
                            else:
                                raise ValueError("Invalid page range")
                        elif part:
                            page_num = int(part)
                            if page_num > 0:
                                vl_pages.append(page_num)
                            else:
                                raise ValueError("Page number must be positive")
                except ValueError as e:
                    st.error(f"Invalid page numbers format: '{vl_page_numbers_str}'. Error: {e}")
                    return # Stop processing

        # Start processing
        st.session_state.processing = True
        st.session_state.processing_status = "Starting document processing..."
        st.session_state.processing_progress = 0.0

        # --- Store the selected model name ---
        st.session_state.selected_llm_model_name = selected_llm_name
        logger.info(f"Storing selected LLM for session: {selected_llm_name}")
        
        # Log spreadsheet options if any
        has_spreadsheets = any(f.name.lower().endswith(('.csv', '.xlsx', '.xls')) for f in uploaded_files)
        if has_spreadsheets:
            logger.info(f"Processing with spreadsheet options: {spreadsheet_options}")
        
        # Use the enhanced processing function with spreadsheet options
        try:
            logger.info(f"Starting document processing for {len(uploaded_files)} files")
            process_documents_with_spreadsheet_options(
                uploaded_files, 
                selected_llm_name, 
                vl_pages, 
                vl_process_all,
                spreadsheet_options
            )
            logger.info("Document processing call completed")
        except Exception as e:
            logger.error(f"Error in document processing: {e}", exc_info=True)
            st.error(f"An unexpected error occurred: {e}")
            st.session_state.processing = False

        # Refresh the page to show updated status
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

# --- START OF RELEVANT FUNCTION: render_document_explorer (in app.py) ---

def render_document_explorer():
    """
    Render the document and chunk explorer.
    Always shows original text and extracted metadata (if available).
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
             # Increase limit slightly to catch more potential doc names
             all_chunks_for_filter = query_engine.get_chunks(limit=1500)
             doc_names = sorted(list(set(c['metadata'].get('file_name', 'Unknown')
                                        for c in all_chunks_for_filter
                                        if c['metadata'].get('file_name'))))
             if not doc_names:
                  logger.warning("No document names found in retrieved chunks metadata.")
             doc_filter = st.selectbox("Filter by document", options=["All Documents"] + doc_names)
         except Exception as e:
              logger.error(f"Failed to get document names for filter: {e}")
              doc_filter = st.selectbox("Filter by document", options=["All Documents", "Error loading names"])

    # Get chunks from DB based on filter
    doc_filter_value = doc_filter if doc_filter != "All Documents" else None
    # Fetch slightly more chunks for display flexibility
    chunks = query_engine.get_chunks(limit=100, search_text=search_text if search_text else None, document_filter=doc_filter_value)

    if not chunks:
        if search_text or doc_filter_value:
             st.info("No documents match the current filter.")
        else:
             st.info("No documents found. Upload and process documents first.")
        return

    # Display chunks
    st.markdown(f"Displaying **{len(chunks)}** chunks:")
    for i, chunk in enumerate(chunks):
        # Use chunk_id for unique expander key if available
        chunk_id_display = chunk['metadata'].get('chunk_id', chunk.get('id', f'chunk_{i}'))
        file_name_display = chunk['metadata'].get('file_name', 'Unknown')
        expander_title = f"Chunk {chunk_id_display} - {file_name_display}"

        with st.expander(expander_title):
            meta = chunk.get('metadata', {})
            page_num_display = meta.get('page_num', 'N/A')
            row_idx_display = meta.get('row_idx', None) # Check for spreadsheet row index

            # --- Display Original Text ---
            st.markdown("##### Original Text")
            original_text = chunk.get('original_text', chunk.get('text', '')) # Fallback to 'text' if original is missing
            st.markdown(original_text if original_text else "_No text content available_")

            # --- Display Location Info ---
            if row_idx_display is not None:
                 location_info = f"Row: {row_idx_display}"
            else:
                 location_info = f"Page: {page_num_display}"
            st.caption(f"Document: {file_name_display} | {location_info}")

            # --- Always Display Extracted Metadata Section ---
            st.markdown("---")
            st.markdown("##### Extracted Metadata")

            # Get metadata values with defaults
            summary = meta.get('extracted_summary', 'Not available')
            red_flags = meta.get('extracted_red_flags', 'None detected')
            entity_types_list = meta.get('extracted_entity_types', [])
            types_text = ", ".join(entity_types_list) if entity_types_list else 'None detected'

            # Display using captions for consistent style
            st.caption(f"**Summary:** {summary}")
            st.caption(f"**Red Flags:** {red_flags}")
            st.caption(f"**Entity Types:** {types_text}")



# (Ensure necessary imports like nx, Network, json etc. are present)
def render_connection_explorer_tab(entities, relationships):
    """
    Render the entity connection explorer tab using MultiDiGraph.
    Visualizes paths between two entities.

    Args:
        entities: List of entity dictionaries (potentially filtered)
        relationships: List of relationship dictionaries (potentially filtered)
    """
    try:
        import math
        import networkx as nx
        from pyvis.network import Network
        from collections import deque # For efficient BFS
        import json # Ensure json is imported
        import traceback
        import random # For edge colors
    except ImportError as e:
        st.error(f"Missing library required for graph visualization: {e}. Please install PyVis and NetworkX.")
        return

    st.markdown("#### Entity-to-Entity Connection Path")
    st.markdown("Find and visualize the shortest paths connecting two specific entities.")

    if not entities:
        st.warning("No entities available for selection.")
        return

    # (Keep entity lookup maps and UI controls: source/target select, max path length, physics)
    entity_name_to_id = {e.get("name"): e.get("id") for e in entities if e.get("name") and e.get("id")}
    entity_id_to_name = {v: k for k, v in entity_name_to_id.items()}
    entity_id_to_type = {e.get("id"): e.get("type", "Unknown") for e in entities if e.get("id")}
    entity_names = sorted(entity_name_to_id.keys())

    if len(entity_names) < 2:
        st.warning("Need at least two entities to find a connection.")
        return

    st.markdown("**Connection Controls**")
    col1, col2, col3 = st.columns(3)
    with col1:
        source_entity_name = st.selectbox("Source Entity", options=entity_names, index=0, key="conn_source")
    with col2:
        default_target_index = 1 if len(entity_names) > 1 else 0
        if entity_names[default_target_index] == source_entity_name and len(entity_names) > 1: default_target_index = 0
        target_entity_name = st.selectbox("Target Entity", options=entity_names, index=default_target_index, key="conn_target")
    with col3:
        degrees_of_separation = st.slider("Max Path Length (hops)", min_value=1, max_value=10, value=3, key="conn_degrees", help="Maximum number of relationship steps allowed in the path.")

    st.markdown("**Physics & Layout Controls**")
    physics_col1, physics_col2, physics_col3 = st.columns(3)
    with physics_col1:
        physics_enabled = st.toggle("Enable Physics Simulation", value=True, key="conn_physics_toggle")
        physics_solver = st.selectbox("Physics Solver", options=["barnesHut", "forceAtlas2Based", "repulsion"], index=1, key="conn_physics_solver")
    with physics_col2:
        grav_constant = st.slider( "Node Repulsion Strength", min_value=-20000, max_value=-100, value=-5000, step=500, key="conn_grav_constant")
        central_gravity = st.slider( "Central Gravity", min_value=0.0, max_value=1.0, value=0.2, step=0.05, key="conn_central_gravity")
    with physics_col3:
        spring_length = st.slider( "Ideal Edge Length", min_value=50, max_value=500, value=100, step=10, key="conn_spring_length")
        spring_constant = st.slider( "Edge Stiffness", min_value=0.005, max_value=0.5, value=0.06, step=0.005, format="%.3f", key="conn_spring_constant")


    if st.button("Visualize Connection Path", key="visualize_connection_btn", type="primary"):
        if source_entity_name == target_entity_name:
            st.warning("Source and Target entities must be different.")
            return

        source_id = entity_name_to_id.get(source_entity_name)
        target_id = entity_name_to_id.get(target_entity_name)

        if not source_id or not target_id:
            st.error("Could not find IDs for selected entities.")
            return

        # --- Build Full Graph for Path Finding ---
        with st.spinner("Finding connection paths..."):
            # ***** CHANGE HERE: Use MultiDiGraph *****
            G_path = nx.MultiDiGraph()

            nodes_in_rels = set()
            rel_lookup = {} # Store descriptions keyed by (source, target, type) - maybe less useful now

            # Add nodes involved in any relationship first
            for rel in relationships:
                 s_id = rel.get("source_entity_id", rel.get("from_entity_id"))
                 t_id = rel.get("target_entity_id", rel.get("to_entity_id"))
                 if s_id and t_id:
                     nodes_in_rels.add(s_id)
                     nodes_in_rels.add(t_id)
                     # Store relationship details if needed later, keyed uniquely
                     # rel_lookup[rel.get("id", str(uuid.uuid4()))] = rel # Store full rel keyed by its ID

            # Add only relevant nodes to the pathfinding graph
            for entity in entities:
                e_id = entity.get("id")
                if e_id in nodes_in_rels:
                     # Add node only if not already present
                     if not G_path.has_node(e_id):
                         G_path.add_node(e_id, label=entity.get("name"), type=entity.get("type"))

            # Add edges
            for rel in relationships:
                s_id = rel.get("source_entity_id", rel.get("from_entity_id"))
                t_id = rel.get("target_entity_id", rel.get("to_entity_id"))
                r_type = rel.get("type", rel.get("relationship_type", "RELATED_TO"))
                r_desc = rel.get("description", "N/A") # Get description
                rel_id_key = rel.get("id", str(uuid.uuid4())) # Unique key for the edge

                # Add edge only if both nodes exist in our graph
                if G_path.has_node(s_id) and G_path.has_node(t_id):
                     # ***** CHANGE HERE: Directly add edge *****
                     G_path.add_edge(s_id, t_id, key=rel_id_key, type=r_type, description=r_desc)

            # --- Path Finding (Undirected BFS up to max depth on Simple Graph) ---
            # Pathfinding algorithms like shortest_path often work best on simple graphs.
            # Convert the MultiDiGraph to a simple Graph for path finding, ignoring edge types/multiplicity for the path structure itself.
            # We'll retrieve the specific edge details later for visualization.
            UG_simple = nx.Graph(G_path) # Convert to simple undirected graph

            paths_found = []
            if UG_simple.has_node(source_id) and UG_simple.has_node(target_id): # Check node existence in simple graph
                if nx.has_path(UG_simple, source_id, target_id):
                     try:
                        # Find all shortest paths in the simple graph
                        shortest_paths = list(nx.all_shortest_paths(UG_simple, source=source_id, target=target_id))
                        # Filter paths by max length if necessary
                        paths_found = [p for p in shortest_paths if len(p) - 1 <= degrees_of_separation]
                        if not paths_found and shortest_paths:
                             st.warning(f"Shortest path(s) have {len(shortest_paths[0])-1} hops, exceeding the limit of {degrees_of_separation}.")
                     except nx.NetworkXNoPath:
                        paths_found = []
                     except Exception as path_err:
                        st.error(f"Error finding paths: {path_err}")
                        logger.error(f"Path finding error: {traceback.format_exc()}")
                        paths_found = []
            else:
                st.warning(f"Source or target entity not found in the graph after filtering/processing.")


            if not paths_found:
                st.warning(f"No connection path found between '{source_entity_name}' and '{target_entity_name}' within {degrees_of_separation} hops.")
                return

            # --- Create Visualization Subgraph (MultiDiGraph) ---
            path_nodes = set(node for path in paths_found for node in path)
            # ***** CHANGE HERE: Use MultiDiGraph for visualization *****
            viz_graph = nx.MultiDiGraph()

            # Add nodes from paths (same logic)
            for node_id in path_nodes:
                 if node_id in entity_id_to_name:
                    viz_graph.add_node(
                        node_id,
                        label=entity_id_to_name[node_id],
                        type=entity_id_to_type.get(node_id, "Unknown"),
                        is_source=(node_id == source_id),
                        is_target=(node_id == target_id)
                    )

            # Add edges *between* nodes in paths, preserving all original relationships
            # Iterate through the *original* G_path (MultiDiGraph) to get all parallel edges
            for u, v, key, data in G_path.edges(data=True, keys=True):
                 if u in path_nodes and v in path_nodes:
                     # Check if this specific edge segment (u,v) is part of any found shortest path sequence
                     # Note: A pair (u,v) might be in a path, but not *this specific* parallel edge if multiple exist.
                     # We'll add *all* edges between nodes that are part of *any* path for context.
                     # Highlighting the specific shortest path edges will be done during PyVis rendering.
                     is_segment_in_a_path = any(
                         (u == path[i] and v == path[i+1]) or (v == path[i] and u == path[i+1]) # Undirected check for segment
                         for path in paths_found for i in range(len(path)-1)
                     )
                     if is_segment_in_a_path:
                         viz_graph.add_edge(u, v, key=key, type=data.get("type"), description=data.get("description"))


        # --- Create PyVis Network ---
        with st.spinner("Rendering connection path..."):
            net = Network(height="700px", width="100%", directed=True, notebook=True, cdn_resources='remote', heading="")

            # (Keep node styling: colors, shapes, source/target highlighting)
            colors = CONFIG.get("visualization", {}).get("node_colors", {}) # Use centrally defined colors
            shapes = CONFIG.get("visualization", {}).get("node_shapes", {})
            source_color = "#FF4444" # Bright Red
            target_color = "#44FF44" # Bright Green

            for node_id, attrs in viz_graph.nodes(data=True):
                 entity_type = attrs.get("type", "Unknown")
                 is_source = attrs.get("is_source", False)
                 is_target = attrs.get("is_target", False)

                 if is_source: color, border_width, size, title_suffix = source_color, 3, 30, " (Source)"
                 elif is_target: color, border_width, size, title_suffix = target_color, 3, 30, " (Target)"
                 else: color, border_width, size, title_suffix = colors.get(entity_type, colors.get("Unknown", "#9CA3AF")), 1, 20, ""

                 shape = shapes.get(entity_type, shapes.get("Unknown", "dot"))
                 label = attrs.get("label", "Unknown")
                 title = f"{label}\nType: {entity_type}{title_suffix}"
                 net.add_node(node_id, label=label, title=title, color=color, shape=shape, size=size, borderWidth=border_width)


            # --- Add Edges to PyVis (Modified) ---
            # Define styles for relationship types
            rel_type_styles = {
                "WORKS_FOR": {"color": "#ff5733", "dashes": False}, "OWNS": {"color": "#33ff57", "dashes": [5, 5]},
                "LOCATED_IN": {"color": "#3357ff", "dashes": False}, "CONNECTED_TO": {"color": "#ff33a1", "dashes": [2, 2]},
                "MET_WITH": {"color": "#f4f70a", "dashes": False}, "DEFAULT": {"color": "#AAAAAA", "dashes": False}
            }

            # Identify edges belonging to the *first* shortest path found for highlighting
            shortest_path_segments = set()
            if paths_found:
                shortest_p = paths_found[0]
                for i in range(len(shortest_p) - 1):
                    # Store both directed segments for highlighting
                    shortest_path_segments.add((shortest_p[i], shortest_p[i+1]))
                    # shortest_path_segments.add((shortest_p[i+1], shortest_p[i])) # Only needed if highlighting undirected path

            edge_count = 0
            # Iterate through edges in the viz_graph (MultiDiGraph)
            # Use keys=True if you need the unique key, otherwise it's optional
            for source, target, attrs in viz_graph.edges(data=True):
                 edge_count += 1
                 rel_type = attrs.get("type", "RELATED_TO")
                 description = attrs.get("description", "N/A")
                 edge_title = f"Type: {rel_type}\nDescription: {description}"

                 # Check if this directed edge segment is part of the highlighted shortest path
                 is_shortest = (source, target) in shortest_path_segments

                 # Get base style for the relationship type
                 style = rel_type_styles.get(rel_type, rel_type_styles["DEFAULT"]).copy() # Get a copy to modify

                 # Highlight shortest path edges
                 if is_shortest:
                     style['color'] = "#FF6347" # Tomato color for shortest path
                     style['width'] = 3
                     style['dashes'] = False
                 else:
                     # Apply default width or potentially vary based on type
                     style['width'] = style.get('width', 1.5) # Use type's width or default 1.5

                 # Add edge to PyVis
                 net.add_edge(
                    source, target,
                    title=edge_title,
                    label="", # Keep labels clean
                    width=style.get('width'),
                    color=style.get('color'),
                    dashes=style.get('dashes'),
                    arrows={'to': {'enabled': True, 'scaleFactor': 0.6}}
                 )

            # --- (Keep PyVis options setup and rendering logic) ---
            pyvis_options = _build_pyvis_options(
                physics_enabled, physics_solver, spring_length, spring_constant, central_gravity, grav_constant
            )
            net.set_options(json.dumps(pyvis_options))

            graph_html_path = ROOT_DIR / "temp" / "connection_graph.html"
            try:
                net.save_graph(str(graph_html_path))
                with open(graph_html_path, "r", encoding="utf-8") as f: html_content = f.read()

                st.markdown(f"**Found {len(paths_found)} shortest path(s) (length: {len(paths_found[0])-1} hops):**")
                path_desc = " → ".join([entity_id_to_name.get(node_id, "?") for node_id in paths_found[0]])
                st.success(f"Path: {path_desc}")
                if len(paths_found) > 1:
                     with st.expander(f"Show {len(paths_found)-1} other shortest path(s)"):
                         for i, path in enumerate(paths_found[1:], 1):
                             path_desc_other = " → ".join([entity_id_to_name.get(node_id, "?") for node_id in path])
                             st.text(f"Path {i+1}: {path_desc_other}")

                st.components.v1.html(html_content, height=710, scrolling=False)
                # Update edge count reporting
                st.caption(f"Displaying {viz_graph.number_of_nodes()} entities and {viz_graph.number_of_edges()} relationships involved in the connection path(s).")

            except Exception as render_err:
                st.error(f"Failed to render connection graph: {render_err}")
                logger.error(f"PyVis rendering failed for connection graph: {traceback.format_exc()}")

def render_entity_connection_explorer(entities, relationships):
    """
    Render the entity connection explorer to visualize connections between two entities.
    Enhanced with node distance controls.

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

    # NEW: Node distance controls
    st.markdown("### Node Distance Settings")
    distance_col1, distance_col2, distance_col3 = st.columns(3)

    with distance_col1:
        # Spring length controls node distance
        spring_length = st.slider(
            "Spring Length",
            min_value=50,
            max_value=500,
            value=100,
            step=10,
            help="Higher values increase distance between nodes",
            key="connection_spring_length"
        )

    with distance_col2:
        # Spring constant affects how strongly nodes are pulled together
        spring_constant = st.slider(
            "Spring Constant",
            min_value=0.01,
            max_value=0.5,
            value=0.05,
            step=0.01,
            help="Lower values make connections more flexible",
            key="connection_spring_constant"
        )

    with distance_col3:
        # Gravitational Constant controls how strongly nodes repel each other
        grav_constant = st.slider(
            "Gravitational Constant",
            min_value=-1000,
            max_value=-50,
            value=-100,
            step=50,
            help="More negative values increase repulsion between nodes",
            key="connection_grav_constant"
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
                    "gravitationalConstant": grav_constant,
                    "centralGravity": 0.2,
                    "springLength": spring_length,
                    "springConstant": spring_constant,
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


def render_network_metrics(entities, relationships):
    """
    Render a comprehensive analysis of network metrics useful for investigations.
    Includes centrality measures, community detection, influential nodes, etc.

    Args:
        entities: List of entity dictionaries
        relationships: List of relationship dictionaries
    """
    import networkx as nx
    import pandas as pd
    import matplotlib.pyplot as plt
    from collections import Counter

    st.subheader("Network Analysis Metrics")
    st.markdown("""
    This section provides advanced network analysis metrics to help identify 
    key entities, patterns, and structures in the relationship network.
    """)

    # Create NetworkX graph for analysis
    G = nx.DiGraph()
    UG = nx.Graph()  # Undirected version for some metrics

    # Entity lookups for node information
    entity_lookup = {entity.get("id"): entity for entity in entities if entity.get("id")}

    # Add nodes
    for entity in entities:
        entity_id = entity.get("id")
        if entity_id:
            G.add_node(
                entity_id,
                label=entity.get("name", "Unknown"),
                type=entity.get("type", "Unknown")
            )
            UG.add_node(
                entity_id,
                label=entity.get("name", "Unknown"),
                type=entity.get("type", "Unknown")
            )

    # Add edges
    for rel in relationships:
        source_id = rel.get("source_entity_id", rel.get("from_entity_id"))
        target_id = rel.get("target_entity_id", rel.get("to_entity_id"))
        rel_type = rel.get("type", rel.get("relationship_type", "RELATED_TO"))

        if source_id and target_id:
            G.add_edge(source_id, target_id, type=rel_type)
            UG.add_edge(source_id, target_id, type=rel_type)

    # Create tabs for different metric types
    metrics_tab1, metrics_tab2, metrics_tab3, metrics_tab4 = st.tabs([
        "Centrality & Influence",
        "Communities & Clusters",
        "Relationship Patterns",
        "Investigative Insights"
    ])

    # Tab 1: Centrality and Influence Metrics
    with metrics_tab1:
        st.markdown("### Centrality & Influence Analysis")
        st.markdown("""
        These metrics identify the most central and influential entities in the network:
        - **Degree Centrality**: Entities with the most direct connections
        - **Betweenness Centrality**: Entities that serve as bridges between other entities
        - **Eigenvector Centrality**: Entities connected to other highly connected entities
        - **PageRank**: Influential entities based on the entire network structure
        """)

        # Calculate various centrality measures
        with st.spinner("Calculating centrality metrics..."):
            # Degree Centrality (in and out for directed graph)
            in_degree = dict(G.in_degree())
            out_degree = dict(G.out_degree())

            # More complex centrality measures
            try:
                betweenness = nx.betweenness_centrality(G)
                eigenvector = nx.eigenvector_centrality_numpy(G)
                pagerank = nx.pagerank(G)

                # Calculate harmonic centrality for UG (use undirected for this)
                harmonic = nx.harmonic_centrality(UG)
            except Exception as e:
                st.warning(f"Some advanced centrality metrics couldn't be calculated: {e}")
                betweenness = {}
                eigenvector = {}
                pagerank = {}
                harmonic = {}

        # Create a dataframe for display with all metrics
        centrality_data = []

        for node_id in G.nodes():
            entity = entity_lookup.get(node_id, {})
            centrality_data.append({
                'Entity Name': entity.get('name', 'Unknown'),
                'Entity Type': entity.get('type', 'Unknown'),
                'In-Degree': in_degree.get(node_id, 0),
                'Out-Degree': out_degree.get(node_id, 0),
                'Total Connections': in_degree.get(node_id, 0) + out_degree.get(node_id, 0),
                'Betweenness': round(betweenness.get(node_id, 0), 4),
                'Eigenvector': round(eigenvector.get(node_id, 0), 4),
                'PageRank': round(pagerank.get(node_id, 0), 4),
                'Harmonic': round(harmonic.get(node_id, 0), 4),
                'Node ID': node_id
            })

        # Create DataFrame and sort by total connections
        df_centrality = pd.DataFrame(centrality_data)
        df_centrality = df_centrality.sort_values(by='Total Connections', ascending=False)

        # Key Players Section (Top 10 for each metric)
        st.markdown("#### 🔍 Key Players by Centrality Metrics")
        key_players_col1, key_players_col2 = st.columns(2)

        with key_players_col1:
            st.markdown("**Most Connected Entities**")
            df_most_connected = df_centrality.head(10)[['Entity Name', 'Entity Type', 'Total Connections']]
            st.dataframe(df_most_connected, hide_index=True)

            st.markdown("**Top Brokers (Betweenness)**")
            df_brokers = df_centrality.sort_values(by='Betweenness', ascending=False).head(10)[
                ['Entity Name', 'Entity Type', 'Betweenness']]
            st.dataframe(df_brokers, hide_index=True)

        with key_players_col2:
            st.markdown("**Most Influential (PageRank)**")
            df_influential = df_centrality.sort_values(by='PageRank', ascending=False).head(10)[
                ['Entity Name', 'Entity Type', 'PageRank']]
            st.dataframe(df_influential, hide_index=True)

            st.markdown("**Connected to Important Entities (Eigenvector)**")
            df_eigenvector = df_centrality.sort_values(by='Eigenvector', ascending=False).head(10)[
                ['Entity Name', 'Entity Type', 'Eigenvector']]
            st.dataframe(df_eigenvector, hide_index=True)

        # Full Centrality Table with filtering
        st.markdown("#### Full Centrality Analysis")
        centrality_search = st.text_input("Search entities:", key="centrality_search")

        # Apply filtering based on search
        filtered_df = df_centrality
        if centrality_search:
            filtered_df = df_centrality[df_centrality['Entity Name'].str.contains(centrality_search, case=False)]

        st.dataframe(filtered_df, hide_index=True)

    # Tab 2: Communities and Clusters
    with metrics_tab2:
        st.markdown("### Community Detection & Cluster Analysis")
        st.markdown("""
        These metrics help identify cohesive groups, clusters, and communities within the network:
        - **Community Detection**: Groups of entities that are densely connected
        - **Structural Analysis**: Identification of cliques and core groups
        - **Network Density**: Overall connectedness of the network
        """)

        try:
            # Calculate community detection (use undirected graph)
            if len(UG.nodes()) > 0:
                # Community detection using Louvain method
                try:
                    import community as community_louvain

                    partition = community_louvain.best_partition(UG)
                    # Convert community ID to integer for better display
                    partition = {k: int(v) for k, v in partition.items()}

                    # Count communities and their sizes
                    community_sizes = Counter(partition.values())
                    num_communities = len(community_sizes)

                    st.success(f"Identified {num_communities} distinct communities in the network")

                    # Create a dataframe with community information
                    community_data = []
                    for node_id, community_id in partition.items():
                        entity = entity_lookup.get(node_id, {})
                        community_data.append({
                            'Entity Name': entity.get('name', 'Unknown'),
                            'Entity Type': entity.get('type', 'Unknown'),
                            'Community ID': community_id,
                            'Community Size': community_sizes[community_id],
                            'Node ID': node_id
                        })

                    # Create DataFrame
                    df_communities = pd.DataFrame(community_data)

                    # Show community distribution
                    st.markdown("#### Community Distribution")
                    community_summary = []
                    for comm_id, size in sorted(community_sizes.items()):
                        # Get most common entity types in this community
                        comm_entities = df_communities[df_communities['Community ID'] == comm_id]
                        type_counts = Counter(comm_entities['Entity Type'])
                        most_common_types = ", ".join(
                            [f"{type_name} ({count})" for type_name, count in type_counts.most_common(3)])

                        # Find central entity in community (using degree within community)
                        comm_members = [node_id for node_id, comm in partition.items() if comm == comm_id]
                        subgraph = UG.subgraph(comm_members)

                        if subgraph.number_of_nodes() > 0:
                            degree_dict = dict(subgraph.degree())
                            central_node_id = max(degree_dict, key=degree_dict.get)
                            central_entity = entity_lookup.get(central_node_id, {}).get('name', 'Unknown')
                        else:
                            central_entity = "Unknown"

                        community_summary.append({
                            'Community ID': comm_id,
                            'Size': size,
                            'Central Entity': central_entity,
                            'Main Entity Types': most_common_types
                        })

                    # Display community summary
                    st.dataframe(pd.DataFrame(community_summary), hide_index=True)

                    # Community members
                    selected_community = st.selectbox(
                        "Select community to view members:",
                        options=sorted(community_sizes.keys())
                    )

                    if selected_community is not None:
                        st.markdown(f"#### Community {selected_community} Members")
                        community_members = df_communities[df_communities['Community ID'] == selected_community]
                        community_members = community_members.sort_values(by='Entity Type')
                        st.dataframe(community_members[['Entity Name', 'Entity Type']], hide_index=True)

                except ImportError:
                    st.warning(
                        "Community detection requires the 'python-louvain' package. Install with `pip install python-louvain`")

            # Clique Analysis
            st.markdown("#### Clique Analysis")
            try:
                # Find cliques (groups where everyone is connected to everyone else)
                cliques = list(nx.find_cliques(UG))
                # Filter to only show larger cliques
                significant_cliques = [c for c in cliques if len(c) >= 3]

                if significant_cliques:
                    st.success(
                        f"Found {len(significant_cliques)} significant cliques (groups of 3+ fully-connected entities)")

                    clique_data = []
                    for i, clique in enumerate(significant_cliques):
                        # Get entity names and types
                        entities_in_clique = []
                        types_in_clique = []
                        for node_id in clique:
                            entity = entity_lookup.get(node_id, {})
                            entities_in_clique.append(entity.get('name', 'Unknown'))
                            types_in_clique.append(entity.get('type', 'Unknown'))

                        clique_data.append({
                            'Clique ID': i + 1,
                            'Size': len(clique),
                            'Members': ", ".join(entities_in_clique),
                            'Entity Types': ", ".join(sorted(set(types_in_clique)))
                        })

                    # Display cliques
                    st.dataframe(pd.DataFrame(clique_data), hide_index=True)
                else:
                    st.info("No significant cliques found (groups of 3+ fully-connected entities)")

            except Exception as e:
                st.warning(f"Could not compute clique analysis: {e}")

            # Network Density
            st.markdown("#### Network Density")
            # Calculate overall network density
            density = nx.density(G)
            st.markdown(f"**Network Density**: {density:.4f}")

            if density < 0.1:
                st.markdown("This is a **sparse network** with relatively few connections between entities.")
            elif density < 0.3:
                st.markdown("This is a **moderately connected network** with some clustering.")
            else:
                st.markdown("This is a **densely connected network** with many relationships between entities.")

        except Exception as e:
            st.error(f"Error in community analysis: {e}")

    # Tab 3: Relationship Patterns
    with metrics_tab3:
        st.markdown("### Relationship Patterns")
        st.markdown("""
        This section analyzes the types and patterns of relationships in the network:
        - **Relationship Type Distribution**: Frequency of different relationship types
        - **Entity-Relationship Patterns**: How different entities relate to each other
        - **Directionality Analysis**: Incoming vs outgoing relationships
        """)

        # Relationship Type Distribution
        rel_types = [rel.get("type", rel.get("relationship_type", "Unknown")) for rel in relationships]
        rel_type_counts = Counter(rel_types)

        st.markdown("#### Relationship Type Distribution")

        rel_type_data = []
        for rel_type, count in rel_type_counts.most_common():
            rel_type_data.append({
                'Relationship Type': rel_type,
                'Count': count,
                'Percentage': f"{count / len(relationships) * 100:.1f}%"
            })

        st.dataframe(pd.DataFrame(rel_type_data), hide_index=True)

        # Entity Type to Relationship Type Analysis
        st.markdown("#### Entity Type to Relationship Type Patterns")

        entity_rel_patterns = {}
        for rel in relationships:
            source_id = rel.get("source_entity_id", rel.get("from_entity_id"))
            target_id = rel.get("target_entity_id", rel.get("to_entity_id"))
            rel_type = rel.get("type", rel.get("relationship_type", "Unknown"))

            source_entity = entity_lookup.get(source_id, {})
            target_entity = entity_lookup.get(target_id, {})

            source_type = source_entity.get('type', 'Unknown')
            target_type = target_entity.get('type', 'Unknown')

            # Create pattern key
            pattern = f"{source_type} → {rel_type} → {target_type}"
            entity_rel_patterns[pattern] = entity_rel_patterns.get(pattern, 0) + 1

        # Display patterns
        pattern_data = []
        for pattern, count in sorted(entity_rel_patterns.items(), key=lambda x: x[1], reverse=True):
            pattern_data.append({
                'Pattern': pattern,
                'Count': count
            })

        st.dataframe(pd.DataFrame(pattern_data), hide_index=True)

        # Directionality Analysis
        st.markdown("#### Entity Role Analysis (Source vs Target)")

        # Count how many times each entity appears as source vs target
        entity_source_counts = {}
        entity_target_counts = {}

        for rel in relationships:
            source_id = rel.get("source_entity_id", rel.get("from_entity_id"))
            target_id = rel.get("target_entity_id", rel.get("to_entity_id"))

            entity_source_counts[source_id] = entity_source_counts.get(source_id, 0) + 1
            entity_target_counts[target_id] = entity_target_counts.get(target_id, 0) + 1

        # Identify entities that are primarily sources, primarily targets, or balanced
        role_data = []
        for entity in entities:
            entity_id = entity.get('id')
            if not entity_id:
                continue

            source_count = entity_source_counts.get(entity_id, 0)
            target_count = entity_target_counts.get(entity_id, 0)
            total = source_count + target_count

            if total == 0:
                continue

            source_pct = source_count / total * 100
            target_pct = target_count / total * 100

            # Determine primary role
            if source_count > 2 * target_count:
                role = "Primarily Source"
            elif target_count > 2 * source_count:
                role = "Primarily Target"
            else:
                role = "Balanced"

            role_data.append({
                'Entity Name': entity.get('name', 'Unknown'),
                'Entity Type': entity.get('type', 'Unknown'),
                'Outgoing': source_count,
                'Incoming': target_count,
                'Total': total,
                'Outgoing %': f"{source_pct:.1f}%",
                'Incoming %': f"{target_pct:.1f}%",
                'Primary Role': role
            })

        # Sort by total relationships
        role_df = pd.DataFrame(role_data)
        role_df = role_df.sort_values(by='Total', ascending=False)

        st.dataframe(role_df, hide_index=True)

    # Tab 4: Investigative Insights
    with metrics_tab4:
        st.markdown("### Investigative Insights")
        st.markdown("""
        This section provides specific insights that may be useful for investigations:
        - **Shortest Paths**: Find shortest paths between entities of interest
        - **Key Brokers**: Entities that connect different communities
        - **Unusual Patterns**: Potential anomalies in the network
        """)

        # Shortest Path Finder
        st.markdown("#### 🔍 Shortest Path Finder")
        st.markdown("Find the shortest path between any two entities in the network")

        # Entity selection
        path_col1, path_col2 = st.columns(2)

        with path_col1:
            start_entity = st.selectbox(
                "Start Entity:",
                options=[entity.get('name', 'Unknown') for entity in entities if entity.get('id')],
                key="sp_start_entity"
            )

        with path_col2:
            end_entity = st.selectbox(
                "End Entity:",
                options=[entity.get('name', 'Unknown') for entity in entities if entity.get('id')],
                key="sp_end_entity"
            )

        # Find path when button is clicked
        if st.button("Find Path", key="find_path_btn"):
            if start_entity and end_entity and start_entity != end_entity:
                # Map names to IDs
                start_id = None
                end_id = None

                for entity in entities:
                    if entity.get('name') == start_entity:
                        start_id = entity.get('id')
                    if entity.get('name') == end_entity:
                        end_id = entity.get('id')

                if start_id and end_id:
                    try:
                        # Use undirected graph to find paths regardless of direction
                        if nx.has_path(UG, start_id, end_id):
                            # Find shortest path
                            path = nx.shortest_path(UG, start_id, end_id)

                            # Format path with entity names and relationship types
                            path_description = []
                            for i in range(len(path) - 1):
                                from_id = path[i]
                                to_id = path[i + 1]

                                # Get entity names
                                from_name = entity_lookup.get(from_id, {}).get('name', 'Unknown')
                                to_name = entity_lookup.get(to_id, {}).get('name', 'Unknown')

                                # Find relationship type (check both directions)
                                rel_type = None
                                if G.has_edge(from_id, to_id):
                                    rel_type = G.edges[from_id, to_id].get('type', 'Related to')
                                    direction = "→"
                                elif G.has_edge(to_id, from_id):
                                    rel_type = G.edges[to_id, from_id].get('type', 'Related to')
                                    direction = "←"
                                else:
                                    rel_type = "Connected to"
                                    direction = "—"

                                path_description.append(f"{from_name} {direction} [{rel_type}] {direction} {to_name}")

                            # Display path
                            st.success(f"Found path with {len(path) - 1} steps")
                            for step in path_description:
                                st.markdown(f"- {step}")
                        else:
                            st.warning(f"No path exists between {start_entity} and {end_entity}")
                    except Exception as e:
                        st.error(f"Error finding path: {e}")
                else:
                    st.error("Could not find IDs for the selected entities")

        # Key Brokers Analysis
        st.markdown("#### 🔑 Key Brokers")
        st.markdown("""
        Entities that connect otherwise disconnected groups are critical in investigations.
        These "brokers" often control the flow of information, resources, or influence between groups.
        """)

        try:
            # Calculate betweenness again if needed (we already did this earlier)
            if not betweenness:
                betweenness = nx.betweenness_centrality(G)

            # Find bridges between communities (if community detection worked)
            if 'partition' in locals():
                bridges = []

                # Check each relationship to see if it connects different communities
                for rel in relationships:
                    source_id = rel.get("source_entity_id", rel.get("from_entity_id"))
                    target_id = rel.get("target_entity_id", rel.get("to_entity_id"))
                    rel_type = rel.get("type", rel.get("relationship_type", "Unknown"))

                    # Skip if missing necessary data
                    if not source_id or not target_id:
                        continue

                    # Check if this relationship bridges communities
                    if source_id in partition and target_id in partition:
                        source_community = partition[source_id]
                        target_community = partition[target_id]

                        if source_community != target_community:
                            # Get entity names
                            source_entity = entity_lookup.get(source_id, {})
                            target_entity = entity_lookup.get(target_id, {})

                            bridges.append({
                                'Source Entity': source_entity.get('name', 'Unknown'),
                                'Source Type': source_entity.get('type', 'Unknown'),
                                'Source Community': source_community,
                                'Relationship': rel_type,
                                'Target Entity': target_entity.get('name', 'Unknown'),
                                'Target Type': target_entity.get('type', 'Unknown'),
                                'Target Community': target_community
                            })

                # Display bridge relationships
                if bridges:
                    st.success(f"Found {len(bridges)} relationships that bridge different communities")
                    st.dataframe(pd.DataFrame(bridges), hide_index=True)
                else:
                    st.info("No community-bridging relationships found")
        except Exception as e:
            st.warning(f"Could not complete bridge analysis: {e}")

        # Unusual Patterns
        st.markdown("#### 🚩 Unusual Patterns")
        st.markdown("""
        Identifying unusual or anomalous patterns can reveal key insights.
        """)

        # Identify isolated clusters and unusual connections
        try:
            # Find connected components (isolated subgraphs)
            components = list(nx.connected_components(UG))

            if len(components) > 1:
                st.warning(f"Found {len(components)} disconnected subnetworks - this may indicate information silos")

                # List components
                component_data = []
                for i, component in enumerate(components):
                    # Skip giant component
                    if len(component) > 0.8 * UG.number_of_nodes():
                        continue

                    # Get entity names
                    entities_in_component = []
                    types_in_component = []
                    for node_id in component:
                        entity = entity_lookup.get(node_id, {})
                        entities_in_component.append(entity.get('name', 'Unknown'))
                        types_in_component.append(entity.get('type', 'Unknown'))

                    component_data.append({
                        'Component ID': i + 1,
                        'Size': len(component),
                        'Entities': ", ".join(entities_in_component),
                        'Entity Types': ", ".join(sorted(set(types_in_component)))
                    })

                # Display isolated components
                if component_data:
                    st.subheader("Isolated Subnetworks")
                    st.dataframe(pd.DataFrame(component_data), hide_index=True)

            # Identify entities with unusual relationship patterns
            unusual_entities = []
            avg_type_ratio = {}

            # Calculate average ratio of relationship types for each entity type
            for entity in entities:
                entity_id = entity.get('id')
                entity_type = entity.get('type', 'Unknown')

                if not entity_id:
                    continue

                # Get relationships for this entity
                entity_rels = []
                for rel in relationships:
                    source_id = rel.get("source_entity_id", rel.get("from_entity_id"))
                    target_id = rel.get("target_entity_id", rel.get("to_entity_id"))
                    rel_type = rel.get("type", rel.get("relationship_type", "Unknown"))

                    if source_id == entity_id or target_id == entity_id:
                        entity_rels.append(rel_type)

                # Count relationship types
                rel_type_counts = Counter(entity_rels)

                # Add to average calculation
                if entity_type not in avg_type_ratio:
                    avg_type_ratio[entity_type] = {}

                for rel_type, count in rel_type_counts.items():
                    if rel_type not in avg_type_ratio[entity_type]:
                        avg_type_ratio[entity_type][rel_type] = []

                    avg_type_ratio[entity_type][rel_type].append(count)

            # Calculate averages
            type_rel_averages = {}
            for entity_type, rel_counts in avg_type_ratio.items():
                type_rel_averages[entity_type] = {}
                for rel_type, counts in rel_counts.items():
                    type_rel_averages[entity_type][rel_type] = sum(counts) / len(counts)

            # Find entities with unusual patterns
            for entity in entities:
                entity_id = entity.get('id')
                entity_type = entity.get('type', 'Unknown')
                entity_name = entity.get('name', 'Unknown')

                if not entity_id or entity_type not in type_rel_averages:
                    continue

                # Get relationships for this entity
                entity_rels = []
                for rel in relationships:
                    source_id = rel.get("source_entity_id", rel.get("from_entity_id"))
                    target_id = rel.get("target_entity_id", rel.get("to_entity_id"))
                    rel_type = rel.get("type", rel.get("relationship_type", "Unknown"))

                    if source_id == entity_id or target_id == entity_id:
                        entity_rels.append(rel_type)

                # Count relationship types
                rel_type_counts = Counter(entity_rels)

                # Compare to averages
                unusual_ratios = []
                for rel_type, avg in type_rel_averages[entity_type].items():
                    entity_count = rel_type_counts.get(rel_type, 0)

                    # Check if significantly different from average
                    if avg > 0 and (entity_count > 2 * avg or (entity_count < avg / 2 and entity_count > 0)):
                        unusual_ratios.append({
                            'Relationship Type': rel_type,
                            'Entity Count': entity_count,
                            'Average for Type': avg,
                            'Difference': f"{((entity_count - avg) / avg * 100):.1f}%"
                        })

                # Add to unusual entities if any unusual ratios found
                if unusual_ratios:
                    unusual_entities.append({
                        'Entity Name': entity_name,
                        'Entity Type': entity_type,
                        'Unusual Patterns': unusual_ratios
                    })

            # Display unusual entities
            if unusual_entities:
                st.subheader("Entities with Unusual Relationship Patterns")
                for entity in unusual_entities[:10]:  # Limit to top 10
                    st.markdown(f"**{entity['Entity Name']}** ({entity['Entity Type']})")
                    for pattern in entity['Unusual Patterns']:
                        st.markdown(
                            f"- {pattern['Relationship Type']}: {pattern['Entity Count']} vs avg {pattern['Average for Type']:.1f} ({pattern['Difference']} difference)")
        except Exception as e:
            st.warning(f"Could not complete unusual pattern analysis: {e}")

        # Add additional insights based on network structure
        st.markdown("#### 💡 Additional Investigative Insights")

        insights = []

        # Check for strongly connected components
        try:
            strongly_connected = list(nx.strongly_connected_components(G))
            if len(strongly_connected) > 1:
                insights.append(
                    f"Found {len(strongly_connected)} strongly connected components - potential closed loops of influence or resources.")
        except Exception:
            pass

        # Calculate reciprocity (mutual relationships)
        try:
            reciprocity = nx.reciprocity(G)
            if reciprocity > 0.3:
                insights.append(
                    f"High reciprocity ({reciprocity:.2f}) - many mutual/bidirectional relationships, indicating potential collusion or cooperation.")
            elif reciprocity < 0.1:
                insights.append(
                    f"Low reciprocity ({reciprocity:.2f}) - predominantly one-way relationships, possibly indicating hierarchical structure.")
        except Exception:
            pass

        # Check for hubs and authorities
        try:
            hubs, authorities = nx.hits(G)

            # Find top hubs and authorities
            top_hubs = sorted(hubs.items(), key=lambda x: x[1], reverse=True)[:3]
            top_authorities = sorted(authorities.items(), key=lambda x: x[1], reverse=True)[:3]

            hub_names = [entity_lookup.get(node_id, {}).get('name', 'Unknown') for node_id, _ in top_hubs]
            authority_names = [entity_lookup.get(node_id, {}).get('name', 'Unknown') for node_id, _ in top_authorities]

            insights.append(f"Key hubs (entities with many outgoing connections): {', '.join(hub_names)}")
            insights.append(f"Key authorities (entities with many incoming connections): {', '.join(authority_names)}")
        except Exception:
            pass

        # Display insights
        for insight in insights:
            st.markdown(f"- {insight}")
# (Ensure necessary imports like nx, Network, json etc. are present)
def render_entity_centered_tab(entities, relationships):
    """
    Render the entity-centered explorer tab using MultiDiGraph.
    Visualizes connections around a specific entity.

    Args:
        entities: List of entity dictionaries (potentially filtered)
        relationships: List of relationship dictionaries (potentially filtered)
    """
    try:
        import math
        import networkx as nx
        from pyvis.network import Network
        from collections import deque # For BFS
        import json # Ensure json is imported
        import traceback
        import random # For edge colors
    except ImportError as e:
        st.error(f"Missing library required for graph visualization: {e}. Please install PyVis and NetworkX.")
        return

    st.markdown("#### Entity-Centered Network View")
    st.markdown("Explore the immediate neighborhood around a selected entity.")

    if not entities:
        st.warning("No entities available for selection.")
        return

    # (Keep entity lookup maps and UI controls: center entity, depth, rel filter, physics)
    entity_name_to_id = {e.get("name"): e.get("id") for e in entities if e.get("name") and e.get("id")}
    entity_id_to_name = {v: k for k, v in entity_name_to_id.items()}
    entity_id_to_type = {e.get("id"): e.get("type", "Unknown") for e in entities if e.get("id")}
    entity_names = sorted(entity_name_to_id.keys())

    st.markdown("**View Controls**")
    col1, col2, col3 = st.columns(3)
    with col1:
        center_entity_name = st.selectbox("Center Entity", options=entity_names, index=0, key="center_entity")
    with col2:
        connection_depth = st.slider("Connection Depth (hops)", min_value=1, max_value=5, value=1, key="center_depth", help="How many steps away from the center entity to display.")
    with col3:
        rel_types_available = sorted(list(set(rel.get("type", rel.get("relationship_type", "Unknown")) for rel in relationships)))
        selected_rel_types = st.multiselect("Filter Relationship Types", options=rel_types_available, default=rel_types_available, key="center_rel_filter")

    st.markdown("**Physics & Layout Controls**")
    physics_col1, physics_col2, physics_col3 = st.columns(3)
    with physics_col1:
        physics_enabled = st.toggle("Enable Physics Simulation", value=True, key="center_physics_toggle")
        physics_solver = st.selectbox("Physics Solver", options=["barnesHut", "forceAtlas2Based", "repulsion"], index=0, key="center_physics_solver")
    with physics_col2:
        grav_constant = st.slider( "Node Repulsion Strength", min_value=-20000, max_value=-100, value=-6000, step=500, key="center_grav_constant")
        central_gravity = st.slider( "Central Gravity", min_value=0.0, max_value=1.0, value=0.15, step=0.05, key="center_central_gravity")
    with physics_col3:
        spring_length = st.slider( "Ideal Edge Length", min_value=50, max_value=500, value=150, step=10, key="center_spring_length")
        spring_constant = st.slider( "Edge Stiffness", min_value=0.005, max_value=0.5, value=0.07, step=0.005, format="%.3f", key="center_spring_constant")


    if st.button("Visualize Centered Network", key="visualize_centered_btn", type="primary"):
        center_id = entity_name_to_id.get(center_entity_name)
        if not center_id:
            st.error("Could not find ID for the selected center entity.")
            return

        # --- Build Full Graph (Filtered by selected relationship types) ---
        with st.spinner(f"Building neighborhood graph around '{center_entity_name}'..."):
            # ***** CHANGE HERE: Use MultiDiGraph *****
            G_full = nx.MultiDiGraph()
            nodes_in_rels = set()

            # Add nodes involved in relevant relationships first
            for rel in relationships:
                 s_id = rel.get("source_entity_id", rel.get("from_entity_id"))
                 t_id = rel.get("target_entity_id", rel.get("to_entity_id"))
                 r_type = rel.get("type", rel.get("relationship_type", "RELATED_TO"))
                 # Filter by selected relationship types
                 if r_type in selected_rel_types:
                     if s_id and t_id:
                         nodes_in_rels.add(s_id)
                         nodes_in_rels.add(t_id)

            # Add only relevant nodes
            for entity in entities:
                 e_id = entity.get("id")
                 if e_id in nodes_in_rels:
                     if not G_full.has_node(e_id): # Add node only if not present
                         G_full.add_node(e_id, label=entity.get("name"), type=entity.get("type"))

            # Add filtered edges
            for rel in relationships:
                 s_id = rel.get("source_entity_id", rel.get("from_entity_id"))
                 t_id = rel.get("target_entity_id", rel.get("to_entity_id"))
                 r_type = rel.get("type", rel.get("relationship_type", "RELATED_TO"))
                 r_desc = rel.get("description", "N/A") # Get description
                 rel_id_key = rel.get("id", str(uuid.uuid4())) # Unique key

                 # Add edge only if both nodes exist and type is selected
                 if r_type in selected_rel_types and G_full.has_node(s_id) and G_full.has_node(t_id):
                      # ***** CHANGE HERE: Directly add edge *****
                      G_full.add_edge(s_id, t_id, key=rel_id_key, type=r_type, description=r_desc)

            # --- Find Neighborhood (Undirected BFS on Simple Graph) ---
            # Convert to simple graph for neighborhood finding
            UG_simple = nx.Graph(G_full)
            nodes_in_neighborhood = {center_id: 0} # node_id: distance
            queue = deque([(center_id, 0)])

            if UG_simple.has_node(center_id): # Check center exists in simple graph
                 while queue:
                    curr_node, dist = queue.popleft()
                    if dist >= connection_depth: continue
                    # Iterate neighbors in the simple graph
                    for neighbor in UG_simple.neighbors(curr_node):
                        if neighbor not in nodes_in_neighborhood:
                            nodes_in_neighborhood[neighbor] = dist + 1
                            queue.append((neighbor, dist + 1))
            else:
                 st.warning(f"Center entity '{center_entity_name}' not found in the graph after filtering.")


            if len(nodes_in_neighborhood) <= 1 and connection_depth > 0:
                st.warning(f"No connections found for '{center_entity_name}' within {connection_depth} hop(s) with the selected relationship types.")
                return

            # --- Create Visualization Subgraph (MultiDiGraph) ---
            # ***** CHANGE HERE: Use MultiDiGraph *****
            viz_graph = nx.MultiDiGraph()
            # Add nodes in neighborhood (same logic)
            for node_id, distance in nodes_in_neighborhood.items():
                 if node_id in entity_id_to_name: # Ensure node exists in original data
                    viz_graph.add_node(
                        node_id,
                        label=entity_id_to_name[node_id],
                        type=entity_id_to_type.get(node_id, "Unknown"),
                        distance=distance,
                        is_center=(node_id == center_id)
                    )

            # Add edges *between* nodes within the neighborhood from the full MultiDiGraph
            edge_count = 0
            # Iterate through the *original* G_full (MultiDiGraph)
            for u, v, key, data in G_full.edges(data=True, keys=True):
                 # Check if both ends are in the calculated neighborhood
                 if u in nodes_in_neighborhood and v in nodes_in_neighborhood:
                     # Add the edge with its original attributes and key
                     viz_graph.add_edge(u, v, key=key, type=data.get("type"), description=data.get("description"))
                     edge_count +=1

        # --- Create PyVis Network ---
        with st.spinner("Rendering centered visualization..."):
            net = Network(height="700px", width="100%", directed=True, notebook=True, cdn_resources='remote', heading="")

            # (Keep node styling: colors, shapes, center highlighting, distance-based size)
            colors = CONFIG.get("visualization", {}).get("node_colors", {})
            shapes = CONFIG.get("visualization", {}).get("node_shapes", {})
            center_color = "#FF0000" # Bright Red

            for node_id, attrs in viz_graph.nodes(data=True):
                 entity_type = attrs.get("type", "Unknown"); distance = attrs.get("distance", 0); is_center = attrs.get("is_center", False)
                 size = max(12, 35 - (distance * 6))
                 if is_center: color, border_width, title_suffix = center_color, 3, " (Center)"
                 else: color, border_width, title_suffix = colors.get(entity_type, colors.get("Unknown", "#9CA3AF")), 1, f" ({distance} hop{'s' if distance != 1 else ''})"
                 shape = shapes.get(entity_type, shapes.get("Unknown", "dot"))
                 label = attrs.get("label", "Unknown")
                 title = f"{label}\nType: {entity_type}{title_suffix}"
                 net.add_node(node_id, label=label, title=title, color=color, shape=shape, size=size, borderWidth=border_width)

            # --- Add Edges to PyVis (Modified) ---
            rel_type_styles = { # Consistent styles as overview
                "WORKS_FOR": {"color": "#ff5733", "dashes": False, "width": 2}, "OWNS": {"color": "#33ff57", "dashes": [5, 5], "width": 2},
                "LOCATED_IN": {"color": "#3357ff", "dashes": False, "width": 1.5}, "CONNECTED_TO": {"color": "#ff33a1", "dashes": [2, 2], "width": 1.5},
                "MET_WITH": {"color": "#f4f70a", "dashes": False, "width": 1.5}, "DEFAULT": {"color": "#B0B0B0", "dashes": False, "width": 1.0} # Lighter default for non-direct
            }

            # Iterate through edges in viz_graph (MultiDiGraph)
            for source, target, attrs in viz_graph.edges(data=True):
                 rel_type = attrs.get("type", "RELATED_TO")
                 description = attrs.get("description", "N/A")
                 edge_title = f"Type: {rel_type}\nDescription: {description}"

                 # Get base style
                 style = rel_type_styles.get(rel_type, rel_type_styles["DEFAULT"]).copy()

                 # Highlight edges connected directly to the center
                 is_direct_connection = (source == center_id or target == center_id)
                 if is_direct_connection:
                     style['color'] = "#FF6A6A" # Light red for direct
                     style['width'] = max(style.get('width', 1.0), 2.0) # Make direct connections slightly thicker
                     style['dashes'] = False
                 else:
                     # Use type-specific style or default for indirect connections
                     pass # style already holds type-specific or default

                 net.add_edge(
                     source, target,
                     title=edge_title,
                     label="",
                     width=style.get('width'),
                     color=style.get('color'),
                     dashes=style.get('dashes'),
                     arrows={'to': {'enabled': True, 'scaleFactor': 0.6}}
                     )

            # --- (Keep PyVis options setup and rendering logic) ---
            pyvis_options = _build_pyvis_options(
                physics_enabled, physics_solver, spring_length, spring_constant, central_gravity, grav_constant
            )
            net.set_options(json.dumps(pyvis_options))

            graph_html_path = ROOT_DIR / "temp" / "centered_graph.html"
            try:
                 net.save_graph(str(graph_html_path))
                 with open(graph_html_path, "r", encoding="utf-8") as f: html_content = f.read()
                 st.components.v1.html(html_content, height=710, scrolling=False)
                 # Update edge count reporting
                 st.caption(f"Displaying neighborhood: {viz_graph.number_of_nodes()} entities and {viz_graph.number_of_edges()} relationships within {connection_depth} hop(s) of '{center_entity_name}'.")
            except Exception as render_err:
                 st.error(f"Failed to render centered graph: {render_err}")
                 logger.error(f"PyVis rendering failed for centered graph: {traceback.format_exc()}")

def render_network_metrics_tab(entities, relationships):
    """
    Render the network metrics analysis tab with sub-tabs for different metric categories.

    Args:
        entities: List of entity dictionaries (potentially filtered)
        relationships: List of relationship dictionaries (potentially filtered)
    """
    st.markdown("#### Advanced Network Analysis")
    st.markdown("""
    Dive deeper into the network structure with quantitative metrics. Identify key players,
    cohesive groups, potential hidden connections, and structural anomalies.
    """)

    if not entities or not relationships:
        st.info("Insufficient data for network metrics calculation (need both entities and relationships).")
        return

    # Inside render_network_metrics_tab

    @st.cache_data(ttl=3600)
    def build_analysis_graphs(_entities, _relationships):
        logger.info("Building NetworkX multi-graphs for analysis...")
        # ***** CHANGE HERE *****
        G = nx.MultiDiGraph()  # Directed multi-graph
        UG = nx.MultiGraph()  # Undirected multi-graph

        # (Keep entity_lookup creation logic)
        entity_lookup = {}
        valid_entity_count = 0
        for entity in _entities:
            e_id = entity.get("id")
            if e_id:
                entity_data = {"id": e_id, "label": entity.get("name", f"Unknown Entity {e_id[:4]}"),
                               "type": entity.get("type", "Unknown")}
                entity_data.update(entity)
                entity_lookup[e_id] = entity_data
                valid_entity_count += 1
            else:
                logger.warning(f"Skipping entity due to missing ID: {str(entity)[:100]}...")
        logger.info(f"Created entity lookup with {valid_entity_count} valid entities.")

        # Add nodes (same logic)
        for e_id, entity_data in entity_lookup.items():
            node_attrs = {"label": entity_data["label"], "type": entity_data["type"]}
            if not G.has_node(e_id):
                G.add_node(e_id, **node_attrs)
                UG.add_node(e_id, **node_attrs)

        # Add edges (modified for MultiGraph)
        valid_edge_count = 0
        skipped_edge_count = 0
        for rel in _relationships:
            s_id = rel.get("source_entity_id", rel.get("from_entity_id"))
            t_id = rel.get("target_entity_id", rel.get("to_entity_id"))
            if s_id in entity_lookup and t_id in entity_lookup:
                r_type = rel.get("type", rel.get("relationship_type"))
                if r_type and isinstance(r_type, str) and r_type.strip() and r_type.upper() != "UNKNOWN":
                    edge_attrs = {"type": r_type, "description": rel.get("description")}  # Add description
                    rel_id_key = rel.get("id", str(uuid.uuid4()))  # Unique key
                    # ***** CHANGE HERE: Directly add edge *****
                    G.add_edge(s_id, t_id, key=rel_id_key, **edge_attrs)
                    # Add undirected edge - MultiGraph handles parallel edges
                    UG.add_edge(s_id, t_id, key=rel_id_key, **edge_attrs)
                    valid_edge_count += 1
                else:
                    skipped_edge_count += 1
            else:
                skipped_edge_count += 1
        logger.info(
            f"Analysis multi-graphs built: Directed ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges), Undirected ({UG.number_of_nodes()} nodes, {UG.number_of_edges()} edges). Skipped {skipped_edge_count} edges.")
        if skipped_edge_count > G.number_of_edges() * 0.1 and G.number_of_edges() > 0: logger.warning(
            f"Skipped a significant number of edges ({skipped_edge_count})...")
        return G, UG, entity_lookup  # Return MultiDiGraph, MultiGraph, lookup



    try:
        import networkx as nx
        import pandas as pd
        from collections import Counter
        # Import community detection library
        try:
            import community as community_louvain
            louvain_available = True
        except ImportError:
            louvain_available = False
            logger.warning("Python-louvain library not found. Community detection will be unavailable. Install with: pip install python-louvain")
        # Import gensim for Node2Vec
        try:
            from node2vec import Node2Vec
            gensim_available = True
        except ImportError:
            gensim_available = False
            logger.warning("Gensim library not found. Node Similarity (Node2Vec) will be unavailable. Install with: pip install gensim")

        # Build graphs using caching
        G, UG, entity_lookup = build_analysis_graphs(entities, relationships)

        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
             st.warning("The filtered data resulted in an empty graph. Cannot calculate metrics.")
             return

        # --- Create Sub-Tabs for Metrics ---
        centrality_tab, community_tab, links_tab, similarity_tab = st.tabs([
            "👑 Centrality & Influence",
            "👥 Communities & Groups",
            "🔮 Link Prediction Hints",
            "🤝 Node Similarity (Embeddings)"
        ])

        # --- Centrality Tab ---
        with centrality_tab:
            render_centrality_metrics(G, UG, entity_lookup) # Pass both graphs if needed

        # --- Community Tab ---
        with community_tab:
            if louvain_available:
                render_community_metrics(UG, entity_lookup) # Use undirected for communities
            else:
                st.warning("Community detection requires the 'python-louvain' library. Please install it (`pip install python-louvain`).")

        # --- Link Prediction Hints Tab ---
        with links_tab:
            render_link_prediction_hints(UG, entity_lookup) # Use undirected for link prediction scores

        # --- Node Similarity Tab ---
        with similarity_tab:
            if gensim_available:
                render_node_similarity(UG, entity_lookup) # Use undirected for Node2Vec
            else:
                st.warning("Node similarity requires the 'gensim' library. Please install it (`pip install gensim`).")

    except Exception as e:
        st.error(f"An error occurred during network metrics calculation: {e}")
        logger.error(f"Network metrics calculation failed: {traceback.format_exc()}")


def render_node_similarity(UG: nx.Graph, entity_lookup: Dict):
    """Renders the Node Similarity (Embeddings) sub-tab using the 'node2vec' library."""
    import pandas as pd

    # Check for the correct library
    try:
        from node2vec import Node2Vec # Check specific library import
        node2vec_available = True
    except ImportError:
        node2vec_available = False

    st.markdown("##### Find Structurally Similar Entities")
    st.markdown("""
    Node Embeddings learn vector representations of entities based on their network neighborhood. Entities with similar vectors often play similar roles. This uses the Node2Vec algorithm via the `node2vec` library.
    - **Node2Vec:** Learns embeddings by simulating random walks on the graph.
    - **Cosine Similarity:** Measures similarity between entity vectors (closer to 1 is more similar).
    """)

    if not node2vec_available:
        st.warning("Node similarity requires the 'node2vec' library. Please install it (`pip install node2vec`). Note: This library also depends on `gensim`.")
        return

    # --- Node2Vec Model Training & Similarity Search ---
    if UG.number_of_nodes() < 5:
         st.warning("Graph is too small (< 5 nodes) for meaningful Node2Vec training.")
         return

    st.markdown("**Node2Vec Parameters** (Adjust then click 'Find Similar Entities')")
    n2v_col1, n2v_col2, n2v_col3 = st.columns(3)
    with n2v_col1:
         n2v_dims = st.slider("Embedding Dimensions", min_value=16, max_value=128, value=64, step=16, key="n2v_dims")
         n2v_walk_len = st.slider("Walk Length", min_value=10, max_value=80, value=30, step=5, key="n2v_walklen")
    with n2v_col2:
         n2v_num_walks = st.slider("Number of Walks / Node", min_value=10, max_value=200, value=50, step=10, key="n2v_numwalks")
         n2v_window = st.slider("Window Size (Word2Vec)", min_value=2, max_value=10, value=5, step=1, key="n2v_window")
    with n2v_col3:
         # Note about workers=1 on Windows moved to train function logging
         pass

    st.markdown("**Similarity Search**")
    search_col1, search_col2 = st.columns([3,1])
    with search_col1:
         # Ensure labels exist before creating options
         entity_names = sorted([data['label'] for _, data in UG.nodes(data=True) if data and 'label' in data])
         if not entity_names:
              st.warning("No entity names found in the graph nodes.")
              return
         target_entity_name = st.selectbox(
             "Find entities similar to:", options=entity_names, key="n2v_target"
         )
    with search_col2:
         top_n_similar = st.number_input("Number of similar results", min_value=1, max_value=50, value=10, step=1, key="n2v_topn")

    if st.button("Find Similar Entities", key="n2v_find_btn", type="primary"):
        target_node_id = None
        target_node_str = None # We need the string representation used in wv
        for node_id, data in UG.nodes(data=True):
            if data and data.get('label') == target_entity_name:
                target_node_id = node_id
                target_node_str = str(node_id) # Assume node ID is directly convertible to string key used by wv
                break

        if target_node_id is None:
            st.error(f"Could not find node ID for entity: {target_entity_name}")
            return

        with st.spinner(f"Calculating embeddings and finding entities similar to '{target_entity_name}'..."):
            # Call the updated train_node2vec_model function
            # Pass the necessary parameters from sliders
            # Using workers=4 as default, user might need workers=1 on Windows (noted in logs)
            wv, node_map = train_node2vec_model(
                UG,
                dimensions=n2v_dims,
                walk_length=n2v_walk_len,
                num_walks=n2v_num_walks,
                window=n2v_window,
                workers=4 # Default worker count
            )

            if wv is None or node_map is None:
                 st.error("Failed to get Node2Vec model. Cannot perform similarity search.")
                 # Clear previous results if training failed
                 if 'node_similarity_results' in st.session_state: del st.session_state.node_similarity_results
                 return

            # Ensure the target node string (used as key in wv) exists
            if target_node_str not in wv:
                # If direct string conversion didn't work, try finding it in the returned node_map
                found_in_map = False
                for orig_id, str_id in node_map.items():
                     if orig_id == target_node_id:
                          target_node_str = str_id
                          if target_node_str in wv:
                              found_in_map = True
                              break
                if not found_in_map:
                     st.error(f"'{target_entity_name}' (ID: {target_node_id}, Key Attempt: '{target_node_str}') not found in the Node2Vec model vocabulary. It might be isolated or training failed.")
                     logger.warning(f"Node ID {target_node_id} mapped to '{target_node_str}', which is not in WV keys. WV keys sample: {list(wv.index_to_key[:10])}...")
                     if 'node_similarity_results' in st.session_state: del st.session_state.node_similarity_results
                     return

            # Find most similar nodes using the potentially updated target_node_str
            try:
                 similar_nodes = wv.most_similar(target_node_str, topn=top_n_similar + 5) # Get a few extra to filter self and potential errors

                 similarity_results = []
                 # Map string IDs back to original IDs using the reverse map derived from node_map
                 reverse_node_map = {v: k for k, v in node_map.items()}

                 count = 0
                 for node_str, similarity_score in similar_nodes:
                     if count >= top_n_similar: break # Stop once we have enough results

                     original_id = reverse_node_map.get(node_str)
                     if original_id == target_node_id: continue # Skip self
                     if original_id is None:
                          logger.warning(f"Could not map node string '{node_str}' back to an original ID.")
                          continue

                     if original_id in entity_lookup:
                         entity_data = entity_lookup[original_id]
                         similarity_results.append({
                             'Similar Entity': entity_data.get('label', 'Unknown'),
                             'Type': entity_data.get('type', 'Unknown'),
                             'Similarity Score': round(similarity_score, 4),
                             'Node ID': original_id
                         })
                         count += 1
                     else:
                          logger.warning(f"Original ID '{original_id}' (from node_str '{node_str}') not found in entity_lookup.")


                 df_similar = pd.DataFrame(similarity_results) # Already limited by the loop
                 st.session_state.node_similarity_results = df_similar
                 st.session_state.node_similarity_target = target_entity_name # Store target name
                 st.success(f"Found {len(df_similar)} entities similar to '{target_entity_name}'.")

            except KeyError as ke:
                 st.error(f"Entity key '{target_node_str}' not found in the embedding model's vocabulary: {ke}")
                 logger.error(f"KeyError during similarity search for key '{target_node_str}': {ke}")
                 if 'node_similarity_results' in st.session_state: del st.session_state.node_similarity_results
            except Exception as e:
                 st.error(f"An error occurred during similarity search: {e}")
                 logger.error(f"Node similarity search error: {traceback.format_exc()}")
                 if 'node_similarity_results' in st.session_state: del st.session_state.node_similarity_results

    # --- Display Results ---
    if 'node_similarity_results' in st.session_state:
        df_results = st.session_state.node_similarity_results
        target_display_name = st.session_state.get('node_similarity_target', 'the selected entity')
        st.markdown(f"**Entities Structurally Similar to '{target_display_name}'**")
        if not df_results.empty:
            st.dataframe(
                df_results[['Similar Entity', 'Type', 'Similarity Score']],
                hide_index=True, use_container_width=True,
                 column_config={ "Similarity Score": st.column_config.NumberColumn(format="%.4f")}
            )
        else:
             st.info(f"No similar entities found for '{target_display_name}' based on the current model and parameters.")
    else:
         st.info("Select an entity and click 'Find Similar Entities'. Training the embedding model may take a minute or two depending on graph size and parameters.")

def render_centrality_metrics(G: nx.DiGraph, UG: nx.Graph, entity_lookup: Dict): # Accepts the lookup
    """Renders the Centrality & Influence analysis sub-tab."""
    import pandas as pd
    import networkx as nx

    st.markdown("##### Identify Key Entities")
    st.markdown("""
    Centrality metrics help pinpoint influential nodes within the network.
    - **Degree (In/Out/Total):** Number of direct connections. High degree nodes are local hubs.
    - **Betweenness:** Measures how often a node lies on the shortest paths between other nodes. High betweenness nodes act as bridges or brokers.
    - **Eigenvector:** Measures influence based on connections to other influential nodes. High eigenvector nodes are connected to well-connected nodes.
    - **PageRank:** Google's algorithm, similar to Eigenvector, measures influence based on link structure.
    """)


    if G.number_of_nodes() == 0:
        st.warning("Graph is empty. Cannot calculate centrality.")
        return

    with st.spinner("Calculating centrality metrics..."):
        try:
            # ... (calculations as before) ...
            in_degree = dict(G.in_degree())
            out_degree = dict(G.out_degree())
            k_betweenness = min(100, G.number_of_nodes() // 2) if G.number_of_nodes() > 20 else None
            if k_betweenness is not None and k_betweenness <= 0: k_betweenness = None
            betweenness = nx.betweenness_centrality(G, k=k_betweenness, normalized=True, weight=None)
            # ... (Robust Eigenvector calculation) ...
            eigenvector = {} # Initialize
            if G.number_of_edges() > 0:
                 try:
                     largest_cc = max(nx.weakly_connected_components(G), key=len)
                     G_conn = G.subgraph(largest_cc)
                     if G_conn.number_of_nodes() > 1:
                         try:
                             eigenvector_conn = nx.eigenvector_centrality_numpy(G_conn, max_iter=1000, tol=1e-03)
                             eigenvector = {node: eigenvector_conn.get(node, 0.0) for node in G.nodes()}
                         except (nx.PowerIterationFailedConvergence, nx.NetworkXError) as eig_err:
                             logger.warning(f"Eigenvector centrality failed: {eig_err}. Assigning 0.")
                             eigenvector = {node: 0.0 for node in G.nodes()}
                     else: eigenvector = {node: 0.0 for node in G.nodes()}
                 except Exception as eig_gen_err:
                      logger.warning(f"Eigenvector prep error: {eig_gen_err}. Assigning 0.")
                      eigenvector = {node: 0.0 for node in G.nodes()}
            else: eigenvector = {node: 0.0 for node in G.nodes()}

            pagerank = nx.pagerank(G, alpha=0.85)

            centrality_data = []
            processed_ids_debug = set() # For debugging lookup issues
            for node_id in G.nodes():
                # *** Use the PASSED entity_lookup ***
                entity = entity_lookup.get(node_id) # Use .get, default is None
                processed_ids_debug.add(node_id)

                # *** Check if entity lookup was successful ***
                if entity:
                    entity_label = entity.get('label', f"Unknown Entity {node_id[:4]}")
                    entity_type = entity.get('type', 'Unknown')
                else:
                    # This case should ideally not happen if build_analysis_graphs is correct
                    logger.warning(f"Node ID {node_id} found in graph but not in entity_lookup!")
                    entity_label = f"Missing Label {node_id[:4]}"
                    entity_type = "Missing Type"

                in_d = in_degree.get(node_id, 0)
                out_d = out_degree.get(node_id, 0)
                total_degree = in_d + out_d
                centrality_data.append({
                    'Entity': entity_label, # Use potentially defaulted label
                    'Type': entity_type,     # Use potentially defaulted type
                    'In-Degree': in_d, 'Out-Degree': out_d, 'Total Degree': total_degree,
                    'Betweenness': round(betweenness.get(node_id, 0), 5),
                    'Eigenvector': round(eigenvector.get(node_id, 0), 5),
                    'PageRank': round(pagerank.get(node_id, 0), 5),
                    'Node ID': node_id
                })
            df_centrality = pd.DataFrame(centrality_data)

            # Debugging: Compare graph nodes and lookup keys
            graph_nodes_set = set(G.nodes())
            lookup_keys_set = set(entity_lookup.keys())
            if graph_nodes_set != lookup_keys_set:
                 logger.warning(f"Mismatch between graph nodes ({len(graph_nodes_set)}) and lookup keys ({len(lookup_keys_set)}).")
                 logger.warning(f"Nodes in graph but not lookup: {graph_nodes_set - lookup_keys_set}")
                 logger.warning(f"Keys in lookup but not graph: {lookup_keys_set - graph_nodes_set}")


            if df_centrality.empty and G.number_of_nodes() > 0:
                 st.warning("Centrality calculations resulted in an empty dataframe, though graph has nodes.")
                 return

        except Exception as e:
            st.error(f"Failed to calculate centrality metrics: {e}")
            logger.error(f"Centrality calculation error: {traceback.format_exc()}")
            return

    # --- Display Key Players & Full Table ---
    # (Keep the display logic, ensuring it handles potentially empty df_centrality gracefully)
    st.markdown("**Key Players Summary (Top 5)**")
    if not df_centrality.empty:
        kp_col1, kp_col2, kp_col3 = st.columns(3)
        # (Metrics display code as before, using the potentially fixed df_centrality)
        with kp_col1:
            max_degree_val = df_centrality['Total Degree'].max()
            top_degree_entity = df_centrality.loc[df_centrality['Total Degree'].idxmax()]['Entity'] if max_degree_val > 0 else "N/A"
            st.metric("Highest Total Degree", top_degree_entity, f"{max_degree_val} connections")
            st.dataframe(df_centrality.nlargest(5, 'Total Degree')[['Entity', 'Type', 'Total Degree']], hide_index=True, use_container_width=True)
        with kp_col2:
             max_betweenness_val = df_centrality['Betweenness'].max()
             top_betweenness_entity = df_centrality.loc[df_centrality['Betweenness'].idxmax()]['Entity'] if max_betweenness_val > 0 else "N/A"
             st.metric("Top Broker (Betweenness)", top_betweenness_entity, f"{max_betweenness_val:.3f} score")
             st.dataframe(df_centrality.nlargest(5, 'Betweenness')[['Entity', 'Type', 'Betweenness']], hide_index=True, use_container_width=True)
        with kp_col3:
            max_pagerank_val = df_centrality['PageRank'].max()
            top_pagerank_entity = df_centrality.loc[df_centrality['PageRank'].idxmax()]['Entity'] if max_pagerank_val > 0 else "N/A"
            st.metric("Most Influential (PageRank)", top_pagerank_entity, f"{max_pagerank_val:.3f} score")
            st.dataframe(df_centrality.nlargest(5, 'PageRank')[['Entity', 'Type', 'PageRank']], hide_index=True, use_container_width=True)
    else:
        st.info("No centrality data to display.")

    st.markdown("**Full Centrality Data**")
    if not df_centrality.empty:
        filt_col1, filt_col2 = st.columns([2,1])
        # (Filter controls as before)
        with filt_col1: search_entity = st.text_input("Search Entity Name", key="cent_search")
        with filt_col2: min_degree = st.number_input("Min Total Degree", min_value=0, value=0, step=1, key="cent_min_degree")

        filtered_df = df_centrality[df_centrality['Total Degree'] >= min_degree]
        if search_entity:
            if 'Entity' in filtered_df.columns:
                 filtered_df = filtered_df[filtered_df['Entity'].astype(str).str.contains(search_entity, case=False, na=False)]

        st.dataframe(
            filtered_df.sort_values(by='Total Degree', ascending=False),
            # (Column config as before)
             hide_index=True, use_container_width=True,
             column_config={ "Betweenness": st.column_config.NumberColumn(format="%.5f"), "Eigenvector": st.column_config.NumberColumn(format="%.5f"), "PageRank": st.column_config.NumberColumn(format="%.5f"), "Node ID": None }
            )
        st.caption(f"Displaying {len(filtered_df)} of {len(df_centrality)} entities.")
    else:
        st.info("No centrality data to display.")

def render_community_metrics(UG: nx.Graph, entity_lookup: Dict):
    """Renders the Communities & Groups analysis sub-tab."""
    import pandas as pd
    import networkx as nx
    from collections import Counter
    try:
        import community as community_louvain
        louvain_available = True
    except ImportError:
        louvain_available = False

    if not louvain_available:
        st.warning("Community detection requires 'python-louvain'. Install with `pip install python-louvain`.")
        return

    st.markdown("##### Detect Cohesive Groups")
    st.markdown("""
    Community detection algorithms identify groups of nodes that are more densely connected internally than with the rest of the network. These can represent organizations, families, or operational clusters.
    - **Louvain Method:** A popular algorithm for finding high-modularity partitions (communities).
    - **Modularity:** A score indicating the quality of the detected community structure (higher is generally better, typically > 0.3).
    """)

    # --- Calculate Communities ---
    with st.spinner("Detecting communities using Louvain method..."):
        try:
            # Compute the best partition using Louvain
            partition = community_louvain.best_partition(UG)
            modularity = community_louvain.modularity(partition, UG)

            # Process partition results
            community_sizes = Counter(partition.values())
            num_communities = len(community_sizes)
            st.success(f"Detected {num_communities} communities with a modularity score of {modularity:.4f}.")

            community_data = []
            for node_id, comm_id in partition.items():
                entity = entity_lookup.get(node_id, {})
                community_data.append({
                    'Entity': entity.get('label', 'Unknown'),
                    'Type': entity.get('type', 'Unknown'),
                    'Community ID': comm_id,
                    'Community Size': community_sizes[comm_id],
                    'Node ID': node_id
                })
            df_communities = pd.DataFrame(community_data)

        except Exception as e:
            st.error(f"Failed to detect communities: {e}")
            logger.error(f"Community detection error: {traceback.format_exc()}")
            return

    # --- Display Community Summary ---
    st.markdown("**Community Overview**")
    # Filter out very small communities for the summary table
    min_comm_size_display = st.slider("Min Community Size to Display", min_value=1, max_value=max(50, max(community_sizes.values()) // 2), value=max(3, min(5,max(community_sizes.values()))), key="comm_min_size")

    summary_data = []
    displayed_community_ids = set()
    for comm_id, size in sorted(community_sizes.items(), key=lambda item: item[1], reverse=True):
         if size >= min_comm_size_display:
            displayed_community_ids.add(comm_id)
            comm_nodes = df_communities[df_communities['Community ID'] == comm_id]
            type_counts = Counter(comm_nodes['Type'])
            most_common_types = ", ".join([f"{t} ({c})" for t, c in type_counts.most_common(3)])

            # Find most central node within the community (by degree within the community subgraph)
            subgraph = UG.subgraph(comm_nodes['Node ID'].tolist())
            if subgraph.number_of_nodes() > 0:
                degrees_in_subgraph = dict(subgraph.degree())
                # Check if degrees_in_subgraph is not empty before finding max
                if degrees_in_subgraph:
                    central_node_id = max(degrees_in_subgraph, key=degrees_in_subgraph.get)
                    central_entity_name = entity_lookup.get(central_node_id, {}).get('label', 'Unknown')
                else:
                    central_entity_name = "N/A (isolated node)"
            else:
                central_entity_name = "N/A"


            summary_data.append({
                 'ID': comm_id,
                 'Size': size,
                 'Top Types': most_common_types,
                 'Most Connected Internal Node': central_entity_name
            })

    st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)

    # --- Explore Specific Community ---
    st.markdown("**Explore Community Members**")
    if displayed_community_ids:
        selected_comm_id = st.selectbox(
            "Select Community ID to View Members",
            options=sorted(list(displayed_community_ids)),
            key="comm_select"
            )
        if selected_comm_id is not None:
             members_df = df_communities[df_communities['Community ID'] == selected_comm_id][['Entity', 'Type']].sort_values(by='Type')
             st.dataframe(members_df, hide_index=True, use_container_width=True)
    else:
         st.info("No communities large enough to display based on the current size filter.")

def render_link_prediction_hints(UG: nx.Graph, entity_lookup: Dict):
    """Renders the Link Prediction Hints sub-tab."""
    import pandas as pd
    import networkx as nx

    st.markdown("##### Suggest Potential Connections")
    st.markdown("""
    Link prediction algorithms suggest pairs of nodes that are *not* currently connected but are likely to be, based on the network structure. This can help uncover hidden or missing relationships.
    - **Adamic-Adar Index:** Predicts links based on the number of shared neighbors, weighting rarer neighbors more heavily. Higher scores suggest a higher likelihood of connection.
    """)

    num_nodes = UG.number_of_nodes()
    if num_nodes > 1500: # Limit computation for very large graphs
        st.warning(f"Graph has {num_nodes} nodes. Link prediction calculation is limited to avoid performance issues. Results may be incomplete.")
        limit_nodes = True
    else:
        limit_nodes = False

    # --- Calculate Potential Links ---
    # Select top N nodes by degree to focus calculation if graph is large
    nodes_to_consider = UG.nodes()
    if limit_nodes:
        degrees = dict(UG.degree())
        nodes_to_consider = sorted(degrees, key=degrees.get, reverse=True)[:1500]


    # Parameters for calculation
    num_suggestions = st.slider("Number of Potential Links to Suggest", min_value=10, max_value=200, value=50, step=10, key="lp_num")

    if st.button("Calculate Potential Links", key="lp_calc_btn"):
        with st.spinner("Calculating Adamic-Adar scores for potential links..."):
            try:
                # Generate pairs of non-connected nodes within the considered set
                logger.info("Converting MultiGraph to simple Graph for Adamic-Adar calculation.")
                UG_simple = nx.Graph(UG)

                # Generate pairs of non-connected nodes using the simple graph
                non_edges = nx.non_edges(UG_simple)  # Use nx utility for non-edges


                if not list(non_edges):  # Check if iterator is empty
                    st.info("No potential links to evaluate (graph might be fully connected or too small).")
                    # Handle appropriately, maybe clear session state
                    if 'link_prediction_results' in st.session_state: del st.session_state.link_prediction_results
                    return  # Exit if no non-edges
                # Calculate Adamic-Adar Index for non-edges
                predictions = nx.adamic_adar_index(UG_simple, non_edges)

                # Store results
                link_suggestions = []
                for u, v, score in predictions:
                    entity_u = entity_lookup.get(u, {})
                    entity_v = entity_lookup.get(v, {})
                    link_suggestions.append({
                        'Entity 1': entity_u.get('label', 'Unknown'),
                        'Type 1': entity_u.get('type', 'Unknown'),
                        'Entity 2': entity_v.get('label', 'Unknown'),
                        'Type 2': entity_v.get('type', 'Unknown'),
                        'Adamic-Adar Score': round(score, 4),
                        'Node ID 1': u,
                        'Node ID 2': v
                    })

                # Sort by score and get top N
                df_suggestions = pd.DataFrame(link_suggestions)
                df_suggestions = df_suggestions.nlargest(num_suggestions, 'Adamic-Adar Score')

                # Store in session state
                st.session_state.link_prediction_results = df_suggestions
                st.success(f"Calculated scores for {len(non_edges)} potential links. Showing top {min(num_suggestions, len(df_suggestions))}.")

            except Exception as e:
                st.error(f"Failed to calculate link prediction scores: {e}")
                logger.error(f"Link prediction error: {traceback.format_exc()}")
                if 'link_prediction_results' in st.session_state:
                    del st.session_state.link_prediction_results


    # --- Display Results ---
    if 'link_prediction_results' in st.session_state:
        df_results = st.session_state.link_prediction_results
        st.markdown("**Top Potential Links (Higher score suggests higher likelihood)**")
        st.dataframe(
            df_results[['Entity 1', 'Type 1', 'Entity 2', 'Type 2', 'Adamic-Adar Score']],
            hide_index=True,
            use_container_width=True,
             column_config={
                 "Adamic-Adar Score": st.column_config.NumberColumn(format="%.4f"),
             }
            )
    else:
         st.info("Click 'Calculate Potential Links' to generate suggestions.")


@st.cache_data(ttl=3600, show_spinner=False) # Cache Node2Vec model for an hour
def train_node2vec_model(_UG, dimensions=64, walk_length=30, num_walks=100, window=5, workers=4):
    """Trains a Node2Vec model on the graph using the 'node2vec' library. Cached function."""
    try:
        # *** Import the correct library ***
        from node2vec import Node2Vec
        import networkx as nx # Ensure networkx is imported here if not globally
    except ImportError:
        logger.error("The 'node2vec' library is required for node similarity. Please install it: pip install node2vec")
        st.error("Node Similarity feature requires 'node2vec'. Please install it (`pip install node2vec`) and restart.")
        return None, None

    logger.info(f"Preparing graph for Node2Vec training...")
    # The node2vec library internally handles node types, but using strings is safer if IDs are complex.
    # However, let's try passing the original graph directly first, as the library might handle it.
    # If issues arise, we can revert to string conversion.

    if _UG.number_of_nodes() == 0 or _UG.number_of_edges() == 0:
         logger.warning("Graph is empty or has no edges. Cannot train Node2Vec.")
         return None, None

    # Map original IDs to integers if they are not already suitable for the library
    # (Let's assume UUIDs or strings might need mapping for some implementations)
    node_list = list(_UG.nodes())
    node_map_int_to_orig = {i: node_id for i, node_id in enumerate(node_list)}
    node_map_orig_to_int = {node_id: i for i, node_id in node_map_int_to_orig.items()}

    # Create a new graph with integer node IDs for node2vec library if needed
    # Some versions might handle string IDs directly. Testing with original graph first.
    # If direct use fails, uncomment the integer mapping graph creation.
    # int_graph = nx.Graph()
    # int_graph.add_nodes_from(range(len(node_list)))
    # int_graph.add_edges_from([(node_map_orig_to_int[u], node_map_orig_to_int[v]) for u, v in _UG.edges()])
    # graph_to_train = int_graph
    logger.info("Converting input graph to simple Graph for Node2Vec compatibility.")
    graph_to_train = nx.Graph(_UG)  # Use a simple, undirected graph
    if graph_to_train.number_of_nodes() == 0 or graph_to_train.number_of_edges() == 0:
        logger.warning("Simple graph became empty after conversion. Cannot train Node2Vec.")
        return None, None
    logger.info(f"Training Node2Vec model (dim={dimensions}, walk_length={walk_length}, num_walks={num_walks})...")

    # Train Node2Vec model using the node2vec library's API
    try:
        # Instantiate Node2Vec object
        # NOTE: workers > 1 might cause issues on Windows. Add comment.
        # Consider making 'workers' an advanced setting if problems persist.
        # p=1, q=1 simulates DeepWalk. Adjust p & q to tune BFS vs DFS exploration.
        logger.info(f"Using {workers} workers for Node2Vec. (May require workers=1 on Windows if issues occur)")
        node2vec_model = Node2Vec(
            graph_to_train,
            dimensions=dimensions,
            walk_length=walk_length,
            num_walks=num_walks,
            p=1, # Return parameter
            q=1, # In-out parameter
            workers=workers, # ** Note: May need workers=1 on Windows **
            seed=42, # for reproducibility
            quiet=True # Suppress verbose output during walk generation
        )

        # Train the model (this calls gensim's Word2Vec internally)
        # Pass window size, min_count, etc., here.
        # dimensions and workers are derived from the Node2Vec object.
        model = node2vec_model.fit(
            window=window,
            min_count=1,
            sg=1, # Use skip-gram
            epochs=10, # Standard number of epochs
            batch_words=4 # Standard batch size for Word2Vec
        )

        logger.info("Node2Vec model training complete.")
        # We need the Word2VecKeyedVectors part for similarity lookups
        # Also return the original_id -> integer_id map if we used integer IDs
        # If using original graph, node_map can just map original ID to itself as string if needed by wv
        final_node_map = {orig_id: str(orig_id) for orig_id in graph_to_train.nodes()}

        return model.wv, final_node_map # Return KeyedVectors and the map used (orig_id -> string)

    except Exception as e:
         logger.error(f"Node2Vec training failed: {e}", exc_info=True)
         # Add more specific error message if possible
         if "received duplicated nodes" in str(e).lower():
              logger.error("Node2Vec Error Detail: The graph might contain duplicate node IDs which is unexpected.")
         elif "must be provided" in str(e).lower():
              logger.error(f"Node2Vec Error Detail: A required parameter might be missing or invalid. Check parameters: dims={dimensions}, wl={walk_length}, nw={num_walks}")
         st.error(f"Node2Vec training failed: {e}")
         return None, None

def render_relationship_graph():
    """
    Render the relationship network graph with additional exploration tabs.
    Acts as the main container for different graph views and analyses.
    """
    st.subheader("Relationship Network Analysis")
    st.markdown("""
    Explore the extracted entities and relationships through various interactive visualizations and analytical metrics.
    Use the document filter below to focus the analysis on specific sources.
    """)

    # Apply custom CSS for graph height and styling
    custom_css = """
    <style>
    iframe {
        min-height: 700px !important; /* Adjusted height */
        height: 700px !important;
    }
    .network-container, .vis-network {
        min-height: 700px !important;
        height: 700px !important;
    }
    div.vis-network div.vis-navigation {
        padding: 0;
        right: 10px;
        bottom: 10px;
    }
    /* Styling for metric tabs */
    .stTabs [data-baseweb="tab-list"] {
		gap: 24px;
	}
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F0F2F6;
        border-radius: 4px 4px 0px 0px;
        gap: 8px;
        padding: 10px;
    }
    .stTabs [aria-selected="true"] {
  		background-color: #FFFFFF;
	}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    # --- Load Data ---
    relationships_file = EXTRACTED_DATA_PATH / "relationships.json"
    entities_file = EXTRACTED_DATA_PATH / "entities.json"

    if not relationships_file.exists() or not entities_file.exists():
        st.info("No relationships or entities found. Upload and process documents first.")
        return

    try:
        # Use caching for loading data
        @st.cache_data
        def load_graph_data(rels_path, ents_path):
            logger.info("Loading relationships and entities from disk...")
            with open(rels_path, "r", encoding='utf-8') as f:
                rels = json.load(f)
            with open(ents_path, "r", encoding='utf-8') as f:
                ents = json.load(f)
            logger.info(f"Loaded {len(ents)} entities and {len(rels)} relationships.")
            return rels, ents

        relationships, entities = load_graph_data(relationships_file, entities_file)

    except Exception as e:
        st.error(f"Error loading relationships or entities: {e}")
        logger.error(f"Graph data loading failed: {traceback.format_exc()}")
        return

    if not relationships or not entities:
        st.info("Entity or relationship data is empty.")
        return

    # --- Document Filter ---
    try:
        st.markdown("---")
        st.markdown("### Document Filter")

        # Extract unique document names efficiently
        @st.cache_data
        def get_unique_doc_names(_entities, _relationships):
            docs = set()
            for entity in _entities:
                docs.add(entity.get("source_document", entity.get('context', {}).get('file_name')))
            for rel in _relationships:
                docs.add(rel.get("file_name"))
            # Remove None or empty strings and sort
            return sorted([doc for doc in docs if doc])

        all_documents = get_unique_doc_names(entities, relationships)

        # Multiselect for filtering
        selected_documents = st.multiselect(
            "Filter by documents (select none to analyze all)",
            options=all_documents,
            default=[],
            key="graph_document_filter"
        )

        # Apply filter
        if selected_documents:
            # Filter relationships first
            filtered_relationships = [
                rel for rel in relationships if rel.get("file_name") in selected_documents
            ]
            # Get IDs of entities involved in the filtered relationships
            related_entity_ids = set()
            for rel in filtered_relationships:
                related_entity_ids.add(rel.get("source_entity_id", rel.get("from_entity_id")))
                related_entity_ids.add(rel.get("target_entity_id", rel.get("to_entity_id")))
            # Filter entities
            filtered_entities = [
                entity for entity in entities if entity.get("id") in related_entity_ids
            ]
            st.success(
                f"Filtered to {len(filtered_relationships)} relationships and {len(filtered_entities)} entities from {len(selected_documents)} selected document(s).")
        else:
            # No filter applied
            filtered_entities = entities
            filtered_relationships = relationships
            st.info(f"Analyzing all {len(relationships)} relationships and {len(entities)} entities.")
        st.markdown("---")

    except Exception as e:
        st.error(f"Error applying document filter: {e}")
        logger.error(f"Document filter failed: {traceback.format_exc()}")
        # Fallback to using all data if filter fails
        filtered_entities = entities
        filtered_relationships = relationships

    # Check if filtered data exists
    if not filtered_entities and selected_documents:
        st.warning("No entities found for the selected documents.")
        return


    # --- Render Tabs ---
    try:
        overview_tab, connection_tab, centered_tab, metrics_tab, table_tab = st.tabs([
            "📊 Network Overview",
            "🔗 Connection Explorer",
            "🎯 Entity Centered View",
            "📈 Network Metrics",
            "📄 Relationship Table"
        ])

        # Pass filtered data to each tab rendering function
        with overview_tab:
            render_network_overview_tab(filtered_entities, filtered_relationships)

        with connection_tab:
            render_connection_explorer_tab(filtered_entities, filtered_relationships)

        with centered_tab:
            render_entity_centered_tab(filtered_entities, filtered_relationships)

        with metrics_tab:
            render_network_metrics_tab(filtered_entities, filtered_relationships)

        with table_tab:
            # Use the existing table rendering function, passing filtered data
            render_relationship_table(filtered_relationships, filtered_entities)

    except ImportError as ie_err:
         st.error(f"Missing required library: {ie_err}. Please install it.")
         st.info("You might need to run: pip install pyvis networkx pandas python-louvain gensim")
         logger.error(f"Import error in graph rendering: {ie_err}")
    except Exception as e:
        st.error(f"An unexpected error occurred while rendering the relationship tabs: {e}")
        logger.error(f"Graph rendering failed: {traceback.format_exc()}")
def _build_pyvis_options(physics_enabled: bool, solver: str, spring_length: int, spring_constant: float, central_gravity: float, grav_constant: int) -> Dict:
    """
    Builds the PyVis physics options dictionary based on user settings and selected solver.

    Args:
        physics_enabled (bool): Whether physics simulation is enabled.
        solver (str): The selected physics solver ('forceAtlas2Based', 'barnesHut', 'repulsion').
        spring_length (int): Desired distance between connected nodes.
        spring_constant (float): Stiffness of the connections.
        central_gravity (float): Attraction force towards the center (0 to 1).
        grav_constant (int): Repulsion force between nodes (negative value).

    Returns:
        Dict: Configuration dictionary for PyVis options.
    """
    if not physics_enabled:
        # Return options with physics explicitly disabled
        return {
            "physics": {"enabled": False},
            "interaction": { # Keep interactions enabled even if physics is off
                "hover": True,
                "navigationButtons": True,
                "tooltipDelay": 300,
                "keyboard": {"enabled": True}
            },
             "edges": {
                "smooth": False, # No smoothing needed if static
                "arrows": {"to": {"enabled": True, "scaleFactor": 0.7}}
            },
            "nodes": {
                 "font": {"size": 14, "face": "Arial"}
            }
        }


    physics_config = {
        "enabled": True,
        "solver": solver,
        "stabilization": { # Enhanced stabilization settings
            "enabled": True,
            "iterations": 2000, # More iterations for better stability initially
            "updateInterval": 50,
            "onlyDynamicEdges": False,
            "fit": True
        },
        "adaptiveTimestep": True,
        "minVelocity": 0.75
    }

    # Solver-specific parameters - ensure they match vis.js documentation
    if solver == "forceAtlas2Based":
        physics_config["forceAtlas2Based"] = {
            "gravitationalConstant": grav_constant,
            "centralGravity": central_gravity,
            "springLength": spring_length,
            "springConstant": spring_constant,
            "damping": 0.4,
            "avoidOverlap": 0.6 # Slightly increase overlap avoidance
        }
        # forceAtlas2Based doesn't use barnesHut settings
        if "barnesHut" in physics_config: del physics_config["barnesHut"]
        if "repulsion" in physics_config: del physics_config["repulsion"]

    elif solver == "barnesHut":
        physics_config["barnesHut"] = {
            "gravitationalConstant": grav_constant,
            "centralGravity": central_gravity,
            "springLength": spring_length,
            "springConstant": spring_constant,
            "damping": 0.09,
            "avoidOverlap": 0.1
        }
        if "forceAtlas2Based" in physics_config: del physics_config["forceAtlas2Based"]
        if "repulsion" in physics_config: del physics_config["repulsion"]

    elif solver == "repulsion":
        physics_config["repulsion"] = {
            "centralGravity": central_gravity,
            "springLength": spring_length, # Used indirectly by nodeDistance
            "springConstant": spring_constant,
            # repulsion solver uses nodeDistance for repulsion, gravitationalConstant isn't directly used here
            "nodeDistance": int(spring_length * 1.5), # Make node distance related to spring length
            "damping": 0.09
        }
        if "forceAtlas2Based" in physics_config: del physics_config["forceAtlas2Based"]
        if "barnesHut" in physics_config: del physics_config["barnesHut"]


    options = {
        "physics": physics_config,
        "interaction": {
            "hover": True,
            "navigationButtons": True,
            "tooltipDelay": 300,
            "keyboard": {"enabled": True}
            },
        "edges": {
            "smooth": {"enabled": True, "type": "dynamic"},
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.7}}
            },
        "nodes": {
             "font": {"size": 14, "face": "Arial"}
        }
    }
    return options
# (Import necessary modules at the top of the file if not already present)
# import networkx as nx
# import random # For color variation example

def render_network_overview_tab(entities, relationships):
    """
    Render the overview network graph tab using MultiDiGraph.
    Enhanced with layout algorithm choice, better styling, physics controls, and tooltips.

    Args:
        entities: List of entity dictionaries (potentially filtered)
        relationships: List of relationship dictionaries (potentially filtered)
    """
    try:
        import math
        import networkx as nx # Ensure nx is imported
        from pyvis.network import Network
        import json
        import traceback
        import numpy as np
        import random # For generating distinct edge colors/styles

    except ImportError as e:
        st.error(f"Missing library required for graph visualization: {e}. Please install PyVis, NetworkX, and NumPy.")
        return

    # --- (Keep existing UI controls: sliders, filters, physics) ---
    st.markdown("#### Global Network Visualization")
    st.markdown("Visualize the connections between the most prominent entities based on relationship frequency.")

    # --- Controls ---
    st.markdown("**Visualization Controls**")
    control_col1, control_col2, control_col3 = st.columns(3)
    with control_col1:
        max_nodes = len(entities)
        top_entities_count = st.slider(
            "Max Entities to Display", min_value=10, max_value=max(max_nodes, 300),
            value=min(50, max_nodes), step=5, key="overview_max_entities",
            help=f"Adjust the maximum number of entities shown (Total available: {max_nodes}). Lower values improve performance."
        )
    with control_col2:
        entity_types = sorted(list(set(entity.get("type", "Unknown") for entity in entities)))
        default_types = st.session_state.get("overview_entity_type_filter_default", entity_types)
        selected_graph_types = st.multiselect(
            "Filter by Entity Type", options=entity_types, default=default_types,
            key="overview_entity_type_filter", help="Select entity types to include in the visualization."
        )
        st.session_state.overview_entity_type_filter_default = selected_graph_types
    with control_col3:
         layout_algorithm = st.selectbox(
              "Layout Algorithm",
              options=["PyVis Physics", "Kamada-Kawai (Static)"], index=0, key="overview_layout_algo",
              help="'PyVis Physics' uses interactive simulation. 'Kamada-Kawai' pre-calculates positions for potentially clearer but static layout (best for < 150 nodes)."
         )
         disable_physics_controls = (layout_algorithm == "Kamada-Kawai (Static)")

    # --- Physics Controls ---
    st.markdown("**Physics & Layout Controls** (Only active for 'PyVis Physics' layout)")
    physics_col1, physics_col2, physics_col3 = st.columns(3)
    with physics_col1:
         physics_enabled_toggle = st.toggle("Enable Physics Simulation", value=True, key="overview_physics_toggle", disabled=disable_physics_controls)
         physics_enabled = physics_enabled_toggle and not disable_physics_controls
         physics_solver = st.selectbox(
                "Physics Solver", options=["barnesHut", "forceAtlas2Based", "repulsion"], index=0,
                key="overview_physics_solver", disabled=disable_physics_controls or not physics_enabled,
                help="Algorithm for PyVis physics simulation."
            )
    with physics_col2:
        grav_constant = st.slider( "Node Repulsion", min_value=-30000, max_value=-100, value=-8000, step=500, key="overview_grav_constant", disabled=disable_physics_controls or not physics_enabled)
        central_gravity = st.slider( "Central Gravity", min_value=0.0, max_value=1.0, value=0.1, step=0.05, key="overview_central_gravity", disabled=disable_physics_controls or not physics_enabled)
    with physics_col3:
        spring_length = st.slider( "Edge Length", min_value=50, max_value=600, value=150, step=10, key="overview_spring_length", disabled=disable_physics_controls or not physics_enabled)
        spring_constant = st.slider( "Edge Stiffness", min_value=0.005, max_value=0.5, value=0.04, step=0.005, format="%.3f", key="overview_spring_constant", disabled=disable_physics_controls or not physics_enabled)


    # --- Build Graph ---
    with st.spinner("Building graph..."):
        # (Keep entity filtering logic based on top_entities_count and selected_graph_types)
        entity_lookup = {entity.get("id"): entity for entity in entities if entity.get("id")}
        entity_mentions = {}
        for rel in relationships:
            source_id = rel.get("source_entity_id", rel.get("from_entity_id"))
            target_id = rel.get("target_entity_id", rel.get("to_entity_id"))
            if source_id in entity_lookup:
                entity_mentions[source_id] = entity_mentions.get(source_id, 0) + 1
            if target_id in entity_lookup:
                entity_mentions[target_id] = entity_mentions.get(target_id, 0) + 1

        filtered_entities_by_type = [
            entity for entity in entities
            if entity.get("type", "Unknown") in selected_graph_types and entity.get("id") in entity_lookup
        ]
        for entity in filtered_entities_by_type:
            entity["mention_count"] = entity_mentions.get(entity.get("id"), 1)
        top_entities = sorted(filtered_entities_by_type, key=lambda e: e.get("mention_count", 1), reverse=True)[:top_entities_count]
        top_entity_ids = {entity.get("id") for entity in top_entities}

        # ***** CHANGE HERE: Use MultiDiGraph *****
        G = nx.MultiDiGraph()
        node_attributes_added = set() # Track added nodes to prevent duplicates

        # Add nodes (same logic)
        for entity in top_entities:
             node_id = entity.get("id")
             if node_id not in node_attributes_added:
                 G.add_node(
                    node_id,
                    label=entity.get("name", "Unknown"),
                    type=entity.get("type", "Unknown"), # Store type attribute correctly
                    mention_count=entity.get("mention_count", 1),
                    title=f"{entity.get('name', 'Unknown')}\nType: {entity.get('type', 'Unknown')}\nMentions (all docs): {entity.get('mention_count', 1)}"
                 )
                 node_attributes_added.add(node_id)

        # Add edges
        edge_count = 0
        skipped_edges = 0
        for rel in relationships:
            source_id = rel.get("source_entity_id", rel.get("from_entity_id"))
            target_id = rel.get("target_entity_id", rel.get("to_entity_id"))
            # Only add edge if BOTH source and target nodes are in our TOP entities graph
            if G.has_node(source_id) and G.has_node(target_id):
                rel_type = rel.get("type", rel.get("relationship_type"))
                description = rel.get("description", None) # Get the description
                if rel_type and isinstance(rel_type, str) and rel_type.strip() and rel_type.upper() != "UNKNOWN":
                     # ***** CHANGE HERE: Directly add edge, no has_edge check *****
                     # Add edge with type AND description as attributes
                     G.add_edge(source_id, target_id, key=rel.get("id", str(uuid.uuid4())), # Use original rel ID or new UUID as key for multi-edge uniqueness
                                type=rel_type, description=description)
                     edge_count += 1
                else:
                    skipped_edges += 1
            # We don't increment skipped_edges here if nodes aren't in top set,
            # as that's expected filtering behaviour. Only skip if rel_type is invalid.

    if skipped_edges > 0:
         st.caption(f"ℹ️ Skipped {skipped_edges} relationships with missing/invalid types.")

    # --- (Keep Kamada-Kawai layout calculation logic if selected) ---
    node_positions = None
    if layout_algorithm == "Kamada-Kawai (Static)":
         if G.number_of_nodes() == 0:
              st.warning("Graph is empty, cannot calculate layout.")
         elif G.number_of_nodes() > 150:
              st.warning("Kamada-Kawai layout may be slow for > 150 nodes.")

         if G.number_of_nodes() > 1:
             with st.spinner("Calculating Kamada-Kawai layout..."):
                 try:
                     # Use largest weakly connected component for layout
                     # Note: kamada_kawai works best on connected graphs. MultiDiGraph might have parallel edges affecting distance.
                     # Consider converting to SimpleGraph for layout if results are poor: G_simple = nx.Graph(G)
                     largest_cc_nodes = max(nx.weakly_connected_components(G), key=len)
                     subgraph_for_layout = G.subgraph(largest_cc_nodes)

                     # Convert subgraph to simple graph for layout robustness if needed
                     # subgraph_simple = nx.Graph(subgraph_for_layout)
                     # if subgraph_simple.number_of_nodes() > 1: node_positions_comp = nx.kamada_kawai_layout(subgraph_simple)

                     if subgraph_for_layout.number_of_nodes() > 1:
                         # Try layout directly on the potentially multi-edge subgraph
                         node_positions_comp = nx.kamada_kawai_layout(subgraph_for_layout)

                         node_positions = {
                             node: node_positions_comp.get(node, np.array([0.0, 0.0])) for node in G.nodes() # Use numpy array default
                         }
                         logger.info(f"Kamada-Kawai layout calculated for {len(node_positions_comp)} nodes.")
                         if len(largest_cc_nodes) < G.number_of_nodes():
                              st.caption(f"Layout applied to the largest component ({len(largest_cc_nodes)} nodes). Other {G.number_of_nodes() - len(largest_cc_nodes)} nodes placed at origin.")
                     else:
                         st.warning("Largest connected component has <= 1 node. Cannot apply Kamada-Kawai layout.")
                         node_positions = None
                 except Exception as layout_err:
                     st.error(f"Failed to compute Kamada-Kawai layout: {layout_err}")
                     logger.error(f"Kamada-Kawai layout error: {traceback.format_exc()}")
                     node_positions = None
         else:
             st.info("Graph has <= 1 node, layout calculation skipped.")
             node_positions = None

    # --- Create PyVis Network ---
    if G.number_of_nodes() > 0:
        with st.spinner("Rendering visualization..."):
            net = Network(height="700px", width="100%", directed=True, notebook=True, cdn_resources='remote', heading="")

            # --- (Keep node color/shape setup) ---
            default_colors = {"PERSON": "#3B82F6", "ORGANIZATION": "#10B981", "GOVERNMENT_BODY": "#60BD68", "COMMERCIAL_COMPANY": "#F17CB0", "LOCATION": "#F59E0B", "POSITION": "#8B5CF6", "MONEY": "#EC4899", "ASSET": "#EF4444", "EVENT": "#6366F1", "Unknown": "#9CA3AF"}
            default_shapes = {"PERSON": "dot", "ORGANIZATION": "square", "GOVERNMENT_BODY": "triangle", "COMMERCIAL_COMPANY": "diamond", "LOCATION": "star", "POSITION": "ellipse", "MONEY": "hexagon", "ASSET": "box", "EVENT": "database", "Unknown": "dot"}
            colors = CONFIG.get("visualization", {}).get("node_colors", default_colors)
            shapes = CONFIG.get("visualization", {}).get("node_shapes", default_shapes)

            # Add nodes to PyVis (same logic, including positioning)
            for node_id, attrs in G.nodes(data=True):
                entity_type = attrs.get("type", "Unknown") # Get type from graph attributes
                mentions = attrs.get("mention_count", 1)
                size = max(10, min(35, 10 + 5 * math.log1p(mentions)))
                color = colors.get(entity_type, colors.get("Unknown", "#9CA3AF"))
                shape = shapes.get(entity_type, shapes.get("Unknown", "dot")) # Get shape based on type

                pos = None
                if node_positions is not None and node_id in node_positions:
                     pos_val = node_positions[node_id]
                     if isinstance(pos_val, (np.ndarray, list)) and len(pos_val) >= 2 and all(isinstance(coord, (int, float, np.number)) for coord in pos_val):
                         pos = pos_val
                     else: logger.warning(f"Invalid position data for node {node_id}: {pos_val}. Skipping position.")
                pos_x = pos[0] * 1000 if pos is not None else None
                pos_y = pos[1] * 1000 if pos is not None else None

                net.add_node(
                    node_id, label=attrs.get("label", "Unknown"), title=attrs.get("title", ""),
                    color=color, shape=shape, size=size,
                    font={'size': max(10, min(18, 11 + int(math.log1p(mentions))))},
                    x=pos_x, y=pos_y
                )

            # --- Add Edges to PyVis (Modified for MultiDiGraph) ---
            # Define some basic styling variations for different relationship types
            rel_type_styles = {
                "WORKS_FOR": {"color": "#ff5733", "dashes": False, "width": 2},
                "OWNS": {"color": "#33ff57", "dashes": [5, 5], "width": 2},
                "LOCATED_IN": {"color": "#3357ff", "dashes": False, "width": 1.5},
                "CONNECTED_TO": {"color": "#ff33a1", "dashes": [2, 2], "width": 1.5},
                "MET_WITH": {"color": "#f4f70a", "dashes": False, "width": 1.5},
                "DEFAULT": {"color": "#A0A0A0", "dashes": False, "width": 1.0} # Default style
            }
            # Generate random-ish colors for unmapped types (optional)
            # def get_random_color(): return f"#{random.randint(0, 0xFFFFFF):06x}"

            # Iterate through edges (MultiDiGraph yields all parallel edges)
            # Use G.edges(data=True, keys=True) to get the unique key if needed, but usually not required for PyVis add_edge
            for source, target, attrs in G.edges(data=True):
                rel_type = attrs.get("type", "UNKNOWN") # Get type from edge attributes
                description = attrs.get("description", "N/A") # Get description
                edge_title = f"Type: {rel_type}\nDescription: {description}"

                # Get style based on relationship type
                style = rel_type_styles.get(rel_type, rel_type_styles["DEFAULT"])
                # Or use random color: style['color'] = get_random_color()

                net.add_edge(
                    source, target,
                    title=edge_title, # Tooltip shows type and description
                    label="", # Keep edge labels clean unless needed
                    color=style.get("color"),
                    width=style.get("width"),
                    dashes=style.get("dashes", False), # Set dashes
                    opacity=0.7, # Keep opacity consistent
                    arrows={'to': {'enabled': True, 'scaleFactor': 0.6}}
                )

            # --- (Keep PyVis options setup and rendering logic) ---
            pyvis_options = _build_pyvis_options(
                physics_enabled and layout_algorithm != "Kamada-Kawai (Static)",
                physics_solver, spring_length, spring_constant, central_gravity, grav_constant
            )
            net.set_options(json.dumps(pyvis_options))

            graph_html_path = ROOT_DIR / "temp" / "overview_graph.html"
            try:
                 net.save_graph(str(graph_html_path))
                 with open(graph_html_path, "r", encoding="utf-8") as f:
                    html_content = f.read()
                 st.components.v1.html(html_content, height=710, scrolling=False)
                 # Update edge count reporting for MultiDiGraph
                 st.caption(f"Displaying {G.number_of_nodes()} entities and {G.number_of_edges()} relationships (including parallels). Layout: {layout_algorithm}.")
            except Exception as render_err:
                 st.error(f"Failed to render graph: {render_err}")
                 logger.error(f"PyVis rendering failed: {traceback.format_exc()}")
    else:
        st.info("No nodes to display based on current filters.")

# --- END OF MODIFIED FUNCTION: render_network_overview_tab ---
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
        rel_desc = rel.get("description","Unknown")
        # Add to relationship data
        rel_data.append({
            "Source": source_name,
            "Source Type": source_entity.get("type", "Unknown"),
            "Relationship": rel_type,
            "Relationship Desc":rel_desc,
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
            "Relationship Desc": st.column_config.TextColumn("Description", width="small"),
            "Target": st.column_config.TextColumn("Target", width="medium"),
            "Target Type": st.column_config.TextColumn("Target Type", width="small"),
            "Document": st.column_config.TextColumn("Document", width="medium")
        }
    )


def render_query_page():
    """
    Render the query and chat interface with conversation management.
    """
    st.header("💬 Query System")

    # Add custom CSS for the thinking box
    st.markdown("""
    <style>
    .thinking-box {
        background-color: #f0f7ff;
        border-left: 5px solid #2196F3;
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 4px;
        font-family: monospace;
        max-height: 300px;
        overflow-y: auto;
    }
    .thinking-title {
        font-weight: bold;
        color: #2196F3;
        margin-bottom: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Ensure necessary components (QueryEngine, LLM Manager) are ready
    query_engine = get_or_create_query_engine()  # Assumes this function exists and works
    # Determine which LLM manager to use (adapt this logic)
    use_deepseek = CONFIG.get("deepseek", {}).get("use_api", False)
    llm_manager = None
    if use_deepseek:
        # Initialize DeepSeek manager if needed
        from src.utils.deepseek_manager import DeepSeekManager
        deepseek_manager = DeepSeekManager(CONFIG)
        if not deepseek_manager.client:
            st.warning("DeepSeek API not properly configured. Check settings.")
            return
        llm_manager = deepseek_manager
    else:
        # If not using DeepSeek, check Aphrodite service status
        if not APHRODITE_SERVICE_AVAILABLE:
            st.error("Aphrodite service module is not available. Cannot query.")
            return

        # Get service is already defined in app.py (referenced in other places)
        service = get_service()
        if not service.is_running():
            st.warning("LLM service is not running. Please start it from the sidebar.")
            if st.button("Start LLM Service Now"):
                start_aphrodite_service()
                st.rerun()
            return

        # Ensure the model is loaded
        if not st.session_state.get("llm_model_loaded", False):
            st.warning("Aphrodite service is running, but no model is loaded. Please load a model from Settings.")
            return

        llm_manager = service

    if not llm_manager:
        st.error("No valid LLM service is available or configured.")
        return

    # Get ConversationStore
    conversation_store = st.session_state.get("conversation_store")
    if not conversation_store:
        st.error("Conversation storage system failed to initialize.")
        return

    # --- UI Tabs ---
    conv_tab, chat_tab = st.tabs(["Manage Conversations", "Current Chat"])

    # --- Conversation Management Tab ---
    with conv_tab:
        st.subheader("Conversations")

        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("✨ New Conversation", use_container_width=True, type="primary"):
                new_conv_id = conversation_store.create_conversation()
                if new_conv_id:
                    new_conv_data = conversation_store.get_conversation(new_conv_id)
                    st.session_state.current_conversation_id = new_conv_id
                    st.session_state.active_conversation_data = new_conv_data
                    st.session_state.ui_chat_display = []  # Clear UI display
                    st.session_state.retrieval_enabled_for_next_turn = True  # Enable RAG for first turn
                    st.success("New conversation started!")
                    # Switch to the chat tab automatically (optional)
                    # st.experimental_set_query_params(tab="chat") # May require specific Streamlit version/handling
                    st.rerun()  # Rerun to reflect changes and potentially switch tab focus
                else:
                    st.error("Failed to create new conversation.")

        conversations = conversation_store.list_conversations()
        if conversations:
            st.write(f"You have {len(conversations)} saved conversations:")
            # Display conversations in reverse chronological order (newest first)
            for conv in conversations:
                conv_id = conv['id']
                is_active = (st.session_state.get("current_conversation_id") == conv_id)
                bg_color = "#e0f7fa" if is_active else "transparent"  # Highlight active conversation

                with st.container():  # Use container for styling potential
                    # st.markdown(f"<div style='background-color: {bg_color}; padding: 10px; border-radius: 5px;'>", unsafe_allow_html=True)
                    list_col1, list_col2, list_col3 = st.columns([4, 1, 1])
                    with list_col1:
                        title = conv.get("title", "Untitled")
                        msg_count = conv.get("message_count", 0)
                        last_upd = datetime.fromtimestamp(conv.get("last_updated", 0)).strftime("%Y-%m-%d %H:%M")
                        display_title = f"**{title}**" if is_active else title
                        st.markdown(f"{display_title} ({msg_count} messages) - *{last_upd}*", unsafe_allow_html=True)
                    with list_col2:
                        if not is_active:
                            if st.button("Load", key=f"load_{conv_id}", use_container_width=True):
                                loaded_data = conversation_store.get_conversation(conv_id)
                                if loaded_data:
                                    st.session_state.current_conversation_id = conv_id
                                    st.session_state.active_conversation_data = loaded_data
                                    # Rebuild UI display from loaded messages
                                    st.session_state.ui_chat_display = []
                                    for msg in loaded_data.get("messages", []):
                                        ui_msg = {"role": msg["role"], "content": msg["content"]}
                                        # Include sources/thinking if stored and needed for display
                                        if msg.get("used_context"):
                                            ui_msg["sources"] = msg["used_context"]
                                        if msg.get("thinking_process"):
                                            ui_msg["thinking"] = msg["thinking_process"]
                                        st.session_state.ui_chat_display.append(ui_msg)

                                    st.session_state.retrieval_enabled_for_next_turn = False  # Default OFF when loading
                                    st.success(f"Loaded: {loaded_data.get('title')}")
                                    st.rerun()
                                else:
                                    st.error(f"Failed to load conversation {conv_id}.")
                        else:
                            st.write("*(Active)*")  # Indicate active conversation
                    with list_col3:
                        if st.button("Delete", key=f"del_{conv_id}", use_container_width=True, type="secondary"):
                            if conversation_store.delete_conversation(conv_id):
                                if is_active:  # If deleting the active one, clear state
                                    st.session_state.current_conversation_id = None
                                    st.session_state.active_conversation_data = None
                                    st.session_state.ui_chat_display = []
                                    st.session_state.retrieval_enabled_for_next_turn = True  # Reset for potential new conv
                                st.success("Conversation deleted.")
                                st.rerun()
                            else:
                                st.error("Failed to delete conversation.")
                    # st.markdown("</div>", unsafe_allow_html=True)
                    st.divider()  # Separator between conversations
        else:
            st.info("No saved conversations yet.")

    # --- Current Chat Tab ---
    with chat_tab:
        active_conv_data = st.session_state.get("active_conversation_data")

        if active_conv_data:
            # Display Conversation Controls
            st.subheader(f"Chat: {active_conv_data.get('title', 'Untitled')}")
            control_cols = st.columns([3, 1, 1])
            with control_cols[0]:
                new_title = st.text_input("Rename:", value=active_conv_data.get("title", ""),
                                          label_visibility="collapsed", placeholder="Rename Conversation...")
                if new_title and new_title != active_conv_data.get("title"):
                    active_conv_data["title"] = new_title.strip()
                    save_current_conversation()  # Save immediately on rename
                    # No rerun needed usually, title updates automatically if bound correctly
            with control_cols[1]:
                if st.button("End Conversation", use_container_width=True):
                    save_current_conversation()  # Save before ending
                    st.session_state.current_conversation_id = None
                    st.session_state.active_conversation_data = None
                    st.session_state.ui_chat_display = []
                    st.session_state.retrieval_enabled_for_next_turn = True  # Ready for new conv
                    st.success("Conversation ended.")
                    st.rerun()

            st.divider()

            # Display Chat History (using ui_chat_display for rendering)
            chat_container = st.container()  # Use a container for potentially fixed height scrolling later
            with chat_container:
                if not st.session_state.ui_chat_display:
                    st.info("Ask a question to start the chat!")

                for message in st.session_state.ui_chat_display:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                        # Display thinking/sources if available in the message dict
                        if message.get("thinking"):
                            with st.expander("💭 Reasoning Process", expanded=False):
                                st.markdown(f'<div class="thinking-box">{message["thinking"]}</div>',
                                            unsafe_allow_html=True)
                        if message.get("sources"):
                            with st.expander("View Sources Used", expanded=False):
                                for i, source in enumerate(message["sources"]):
                                    # Use source_index if stored, otherwise use loop index
                                    idx = source.get("source_index", i + 1)
                                    score = source.get('score', 0.0)
                                    st.markdown(f"**Source {idx} (Score: {score:.2f}):**")
                                    st.markdown(f"> {source.get('text', '')}")
                                    meta = source.get('metadata', {})
                                    st.caption(
                                        f"Doc: {meta.get('file_name', 'N/A')} | Page: {meta.get('page_num', 'N/A')} | Chunk: {meta.get('chunk_id', 'N/A')}")
                                    st.markdown("---")

            st.divider()

            # Chat Input Area
            input_cols = st.columns([4, 1])
            with input_cols[1]:
                # Retrieval Toggle - **IMPORTANT:** Key matches session state variable
                with input_cols[1]:  # Keep the column layout if you have it
                    st.checkbox(
                        "Enable RAG",
                        key="retrieval_enabled_for_next_turn",  # Key links it to session state
                        # NO value=... parameter here. Let Streamlit manage it via the key.
                        help="Check this box *before* sending your message to retrieve context. It turns OFF automatically for the next turn."
                    )

            if prompt := st.chat_input("Ask a question..."):
                # When the user submits, chat_input returns the text content in 'prompt'.
                # The script reruns, and this block executes.
                # The state of the checkbox ('retrieval_enabled_for_next_turn') reflects
                # how the user left it *before* submitting.

                # Call the central handler function
                handle_chat_message(prompt, query_engine, llm_manager)

                # IMPORTANT: Remove any explicit st.rerun() immediately after calling
                # handle_chat_message here in the main input handling block.
                # Let handle_chat_message complete its state updates. Streamlit's
                # natural rerun cycle after state changes within the handler
                # (or an optional st.rerun() at the *very end* of handle_chat_message
                # if absolutely necessary) should manage the UI refresh.
                # Adding a rerun here can prematurely interrupt the handler or cause issues.

        else:
            st.info("Please start a new conversation or load an existing one from the 'Manage Conversations' tab.")


def render_settings_page():
    """
    Render the settings page with DeepSeek options.
    """
    st.header("⚙️ Settings")

    # Display current configuration
    with st.expander("Current Configuration", expanded=False):
        st.json(CONFIG)  # Display the whole config for reference

    st.subheader("Update Settings")

    # Create tabs for different settings categories
    settings_tabs = st.tabs(["Retrieval", "LLM", "DeepSeek API", "Extraction"])

    # Retrieval settings
    with settings_tabs[0]:
        st.markdown("#### Retrieval Settings")
        col1, col2 = st.columns(2)
        with col1:
            top_k_vector = st.slider("Vector Search Results (top_k_vector)", 1, 50, CONFIG["retrieval"]["top_k_vector"],
                                     key="s_tkv")
            top_k_bm25 = st.slider("BM25 Search Results (top_k_bm25)", 1, 50, CONFIG["retrieval"]["top_k_bm25"],
                                   key="s_tkb")
            top_k_hybrid = st.slider("Hybrid Search Results (top_k_hybrid)", 1, 30, CONFIG["retrieval"]["top_k_hybrid"],
                                     key="s_tkh")
            top_k_rerank = st.slider("Reranked Results (top_k_rerank)", 1, 20, CONFIG["retrieval"]["top_k_rerank"],
                                     key="s_tkr")
        with col2:
            vector_weight = st.slider("Vector Weight (RRF)", 0.0, 1.0, float(CONFIG["retrieval"]["vector_weight"]),
                                      step=0.05, key="s_vw")
            bm25_weight = st.slider("BM25 Weight (RRF)", 0.0, 1.0, float(CONFIG["retrieval"]["bm25_weight"]), step=0.05,
                                    key="s_bw")
            use_reranking = st.checkbox("Use Reranking", value=CONFIG["retrieval"]["use_reranking"], key="s_ur")
            min_score = st.slider("Minimum Score Threshold", 0.0, 1.0,
                                  float(CONFIG["retrieval"].get("minimum_score_threshold", 0.01)), step=0.01,
                                  key="s_ms")

    # LLM shared settings (Aphrodite)
    with settings_tabs[1]:
        st.markdown("#### LLM Shared Settings (Aphrodite Service)")
        col3, col4 = st.columns(2)
        with col3:
            max_model_len = st.slider("Max Model Context Length", 1024, 16384, CONFIG["aphrodite"]["max_model_len"],
                                      step=1024, key="s_mml")
            # Ensure quantization value exists in options before setting index
            quant_options = [None, "fp8", "fp5", "fp4", "fp6"]  # Example options, adjust based on Aphrodite support
            current_quant = CONFIG["aphrodite"]["quantization"]
            quant_index = quant_options.index(current_quant) if current_quant in quant_options else 0
            quantization = st.selectbox("Quantization", options=quant_options, index=quant_index, key="s_q")

        with col4:
            # Checkbox values might be missing in older configs, provide defaults
            use_flash_attention = st.checkbox("Use Flash Attention",
                                              value=CONFIG["aphrodite"].get("use_flash_attention", True), key="s_ufa")
            compile_model = st.checkbox("Compile Model (Experimental)",
                                        value=CONFIG["aphrodite"].get("compile_model", False), key="s_cm")

        # Extraction-specific LLM parameters
        st.markdown("#### Extraction Parameters (Aphrodite Service)")
        col5, col6 = st.columns(2)
        with col5:
            extraction_temperature = st.slider("Extraction Temperature", 0.0, 1.0,
                                               float(CONFIG["aphrodite"].get("extraction_temperature", 0.1)), step=0.05,
                                               key="s_et")
        with col6:
            extraction_max_new_tokens = st.slider("Extraction Max Tokens", 256, 4096,
                                                  CONFIG["aphrodite"].get("extraction_max_new_tokens", 1024), step=128,
                                                  key="s_emt")

        # Chat-specific LLM parameters
        st.markdown("#### Chat Parameters (Aphrodite Service)")
        col7, col8 = st.columns(2)
        with col7:
            chat_temperature = st.slider("Chat Temperature", 0.0, 1.5,
                                         float(CONFIG["aphrodite"].get("chat_temperature", 0.7)), step=0.05, key="s_ct")
            top_p = st.slider("Top P", 0.1, 1.0, float(CONFIG["aphrodite"].get("top_p", 0.9)), step=0.05, key="s_tp")
        with col8:
            chat_max_new_tokens = st.slider("Chat Max Tokens", 256, 4096,
                                            CONFIG["aphrodite"].get("chat_max_new_tokens", 1024), step=128, key="s_cmt")

    # DeepSeek API settings (new)
    with settings_tabs[2]:
        st.markdown("#### DeepSeek API Settings")

        # Initialize default values if deepseek section doesn't exist
        if "deepseek" not in CONFIG:
            CONFIG["deepseek"] = {
                "use_api": False,
                "api_key": "",
                "api_url": "https://api.deepseek.com/v1",
                "use_reasoner": False,
                "chat_model": "deepseek-chat",
                "reasoning_model": "deepseek-reasoner",
                "temperature": 0.6,
                "max_tokens": 4096
            }

        use_deepseek = st.checkbox(
            "Use DeepSeek API",
            value=CONFIG["deepseek"].get("use_api", False),
            help="Enable DeepSeek API for generation instead of local models"
        )

        # Only show these settings if DeepSeek is enabled
        if use_deepseek:
            col9, col10 = st.columns(2)

            with col9:
                api_key = st.text_input(
                    "API Key",
                    value=CONFIG["deepseek"].get("api_key", ""),
                    type="password",
                    help="Your DeepSeek API key"
                )

                api_url = st.text_input(
                    "API URL",
                    value=CONFIG["deepseek"].get("api_url", "https://api.deepseek.com/v1"),
                    help="DeepSeek API endpoint URL"
                )

                use_reasoner = st.checkbox(
                    "Use DeepSeek Reasoner",
                    value=CONFIG["deepseek"].get("use_reasoner", False),
                    help="Enable reasoning capabilities (shows thinking process)"
                )

            with col10:
                chat_model = st.text_input(
                    "Chat Model Name",
                    value=CONFIG["deepseek"].get("chat_model", "deepseek-chat"),
                    help="Model name for regular chat"
                )

                reasoning_model = st.text_input(
                    "Reasoning Model Name",
                    value=CONFIG["deepseek"].get("reasoning_model", "deepseek-reasoner"),
                    help="Model name for reasoning (used when reasoner is enabled)"
                )

            col11, col12 = st.columns(2)

            with col11:
                ds_temperature = st.slider(
                    "DeepSeek Temperature",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(CONFIG["deepseek"].get("temperature", 0.7)),
                    step=0.05,
                    help="Sampling temperature (higher = more random)"
                )

            with col12:
                ds_max_tokens = st.slider(
                    "DeepSeek Max Tokens",
                    min_value=256,
                    max_value=4096,
                    value=CONFIG["deepseek"].get("max_tokens", 1024),
                    step=128,
                    help="Maximum number of tokens to generate"
                )

    # Extraction settings
    with settings_tabs[3]:
        st.markdown("#### Extraction Settings")
        dedup_threshold = st.slider("Entity Deduplication Threshold (%)", 0, 100,
                                    CONFIG["extraction"]["deduplication_threshold"], key="s_dt")

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
        CONFIG["retrieval"]["minimum_score_threshold"] = float(min_score)

        CONFIG["aphrodite"]["max_model_len"] = max_model_len
        CONFIG["aphrodite"]["quantization"] = quantization
        CONFIG["aphrodite"]["use_flash_attention"] = use_flash_attention
        CONFIG["aphrodite"]["compile_model"] = compile_model
        CONFIG["aphrodite"]["extraction_temperature"] = float(extraction_temperature)
        CONFIG["aphrodite"]["extraction_max_new_tokens"] = extraction_max_new_tokens
        CONFIG["aphrodite"]["chat_temperature"] = float(chat_temperature)
        CONFIG["aphrodite"]["chat_max_new_tokens"] = chat_max_new_tokens
        CONFIG["aphrodite"]["top_p"] = float(top_p)

        CONFIG["extraction"]["deduplication_threshold"] = dedup_threshold

        # Update DeepSeek settings
        if "deepseek" not in CONFIG:
            CONFIG["deepseek"] = {}

        CONFIG["deepseek"]["use_api"] = use_deepseek

        if use_deepseek:
            CONFIG["deepseek"]["api_key"] = api_key
            CONFIG["deepseek"]["api_url"] = api_url
            CONFIG["deepseek"]["use_reasoner"] = use_reasoner
            CONFIG["deepseek"]["chat_model"] = chat_model
            CONFIG["deepseek"]["reasoning_model"] = reasoning_model
            CONFIG["deepseek"]["temperature"] = float(ds_temperature)
            CONFIG["deepseek"]["max_tokens"] = ds_max_tokens

        # Save to file
        try:
            with open(CONFIG_PATH, "w") as f:
                yaml.dump(CONFIG, f, default_flow_style=False, sort_keys=False)
            st.success("Settings saved successfully!")

            # Inform user that LLM service needs to be restarted for some changes
            if st.session_state.aphrodite_service_running:
                st.warning(
                    "**Note:** LLM service might need to be restarted for changes like Max Model Length, Quantization, Flash Attention, or Compile Model to take full effect. Use the sidebar to Stop/Start the service.")

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

    # Required imports for downloading
    import base64
    import io
    import ast
    import pandas as pd
    import numpy as np
    import time

    # Add custom CSS to hide the progress indicator
    st.markdown("""
    <style>
    /* Hide the "Point Data: 100%" progress indicator */
    .datamapplot-progress-container {
        display: none !important;
    }

    /* Hide other progress elements that might appear */
    .progress-label {
        display: none !important;
    }

    /* Make sure tooltips are displayed properly */
    .datamapplot-tooltip {
        opacity: 0.95 !important;
        pointer-events: none !important;
        z-index: 9999 !important;
        position: absolute !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.15) !important;
    }

    /* Download button styling */
    .btn {
        display: inline-block;
        padding: 8px 16px;
        margin: 5px 0;
        background-color: #4e8cff;
        color: white;
        text-decoration: none;
        border-radius: 4px;
        border: none;
        font-weight: 500;
        cursor: pointer;
        text-align: center;
        transition: background-color 0.3s;
    }
    .btn:hover {
        background-color: #3a7be8;
    }
    </style>
    """, unsafe_allow_html=True)

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
                    min_value=2,
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

            # Create a divider for emphasis
            st.markdown("---")

            # Show current visualization type with emphasis
            current_vis_type = CONFIG["clustering"].get("visualization_type", "plotly").lower()
            st.markdown(f"### Current Visualization Type: `{current_vis_type.upper()}`")

            # Select visualization type with clear options
            visualization_options = ["plotly", "datamapplot", "static_datamapplot"]
            visualization_labels = ["Plotly", "Interactive DataMapPlot", "Static DataMapPlot"]

            # Find index of current visualization type
            try:
                vis_index = visualization_options.index(current_vis_type)
            except ValueError:
                vis_index = 0  # Default to Plotly if current type not found

            visualization_type = st.radio(
                "Select Visualization Type",
                options=visualization_labels,
                index=vis_index,
                key="vis_type_selection",
                help="Choose visualization library for cluster map"
            )

            # Map selection back to internal type
            selected_vis_type = visualization_options[visualization_labels.index(visualization_type)]

            # Make it very clear if there's a change
            if selected_vis_type != current_vis_type:
                st.warning(f"""
                **Visualization type will change from `{current_vis_type.upper()}` to `{selected_vis_type.upper()}`**

                You must click 'Update Configuration' and then 'Generate Cluster Map' to apply this change.
                """)

                # Add a direct button to make it easier
                if st.button("Update Visualization Type Now", key="quick_vis_update"):
                    # Update config
                    if "clustering" not in CONFIG:
                        CONFIG["clustering"] = {}
                    CONFIG["clustering"]["visualization_type"] = selected_vis_type

                    # Save to file
                    try:
                        with open(CONFIG_PATH, "w") as f:
                            yaml.dump(CONFIG, f, default_flow_style=False, sort_keys=False)
                        st.success(f"✅ Visualization type updated to {selected_vis_type}!")

                        # Force clear previous results
                        if 'cluster_map_result' in st.session_state:
                            del st.session_state.cluster_map_result

                        # Add a rerun after a short delay
                        time.sleep(0.5)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to update configuration: {e}")

            # DataMapPlot specific options (show for both interactive and static)
            if "datamapplot" in selected_vis_type:
                # Check if datamapplot is available
                try:
                    import datamapplot
                    datamapplot_available = True
                except ImportError:
                    st.warning("DataMapPlot not installed. Install with: `pip install datamapplot`")
                    datamapplot_available = False

                if datamapplot_available:
                    # Get the appropriate config key based on visualization type
                    config_key = "datamapplot" if selected_vis_type == "datamapplot" else "static_datamapplot"

                    st.subheader(f"{visualization_type} Options")

                    col1, col2 = st.columns(2)
                    with col1:
                        darkmode = st.checkbox(
                            "Dark Mode",
                            value=CONFIG["clustering"].get(config_key, {}).get("darkmode", False),
                            help="Use dark mode for the visualization"
                        )

                        cvd_safer = st.checkbox(
                            "CVD-Safer Palette",
                            value=CONFIG["clustering"].get(config_key, {}).get("cvd_safer", True),
                            help="Use a color vision deficiency (CVD) safer color palette"
                        )

                        if selected_vis_type == "datamapplot":  # Interactive-only options
                            enable_toc = st.checkbox(
                                "Enable Table of Contents",
                                value=CONFIG["clustering"].get(config_key, {}).get("enable_table_of_contents", True),
                                help="Show topic hierarchy as a table of contents"
                            )

                    with col2:
                        if selected_vis_type == "datamapplot":  # Interactive-only options
                            cluster_boundaries = st.checkbox(
                                "Show Cluster Boundaries",
                                value=CONFIG["clustering"].get(config_key, {}).get("cluster_boundary_polygons", True),
                                help="Draw boundary lines around clusters"
                            )

                        color_labels = st.checkbox(
                            "Color Label Text",
                            value=CONFIG["clustering"].get(config_key, {}).get("color_label_text", True),
                            help="Use colors for label text based on cluster colors"
                        )

                        marker_size = st.slider(
                            "Marker Size",
                            min_value=3,
                            max_value=15,
                            value=CONFIG["clustering"].get(config_key, {}).get("marker_size", 8),
                            help="Size of data points in visualization"
                        )

                    # Font selection
                    fonts = ["Oswald", "Helvetica", "Roboto", "Times New Roman",
                             "Georgia", "Courier New", "Playfair Display SC", "Open Sans"]

                    current_font = CONFIG["clustering"].get(config_key, {}).get("font_family", "Oswald")
                    font_index = fonts.index(current_font) if current_font in fonts else 0

                    font_family = st.selectbox(
                        "Font Family",
                        options=fonts,
                        index=font_index,
                        help="Font family for text elements"
                    )

                    # Add polygon alpha slider for interactive version
                    if selected_vis_type == "datamapplot":
                        polygon_alpha = st.slider(
                            "Cluster Boundary Opacity",
                            min_value=0.05,
                            max_value=5.00,
                            value=CONFIG["clustering"].get(config_key, {}).get("polygon_alpha", 2.5),
                            step=0.05,
                            help="Opacity of the cluster boundary polygons (higher values make clusters more prominent)"
                        )

                    # Add DPI slider for static version
                    if selected_vis_type == "static_datamapplot":
                        dpi = st.slider(
                            "Plot DPI",
                            min_value=72,
                            max_value=600,
                            value=CONFIG["clustering"].get(config_key, {}).get("dpi", 300),
                            step=1,
                            help="Dots per inch for the static plot (higher values create larger, more detailed images)"
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

            # Update visualization type and log it
            CONFIG["clustering"]["visualization_type"] = selected_vis_type
            logger.info(f"Updating CONFIG visualization_type to: {selected_vis_type}")

            # Update DataMapPlot settings if that's the selected visualization
            if "datamapplot" in selected_vis_type:
                # Interactive DataMapPlot settings
                if "datamapplot" not in CONFIG["clustering"]:
                    CONFIG["clustering"]["datamapplot"] = {}

                CONFIG["clustering"]["datamapplot"].update({
                    "darkmode": darkmode,
                    "cvd_safer": cvd_safer,
                    "enable_table_of_contents": enable_toc if selected_vis_type == "datamapplot" else True,
                    "cluster_boundary_polygons": cluster_boundaries if selected_vis_type == "datamapplot" else True,
                    "color_label_text": color_labels,
                    "marker_size": marker_size,
                    "font_family": font_family,
                    "height": 800,
                    "width": "100%",
                    "polygon_alpha": polygon_alpha if selected_vis_type == "datamapplot" else 2.5
                })

                # Static DataMapPlot settings
                if "static_datamapplot" not in CONFIG["clustering"]:
                    CONFIG["clustering"]["static_datamapplot"] = {}

                CONFIG["clustering"]["static_datamapplot"].update({
                    "darkmode": darkmode,
                    "cvd_safer": cvd_safer,
                    "color_label_text": color_labels,
                    "marker_size": marker_size,
                    "font_family": font_family,
                    "dpi": dpi if selected_vis_type == "static_datamapplot" else 300,
                })

            # Save to file
            try:
                with open(CONFIG_PATH, "w") as f:
                    yaml.dump(CONFIG, f, default_flow_style=False, sort_keys=False)
                st.success("Configuration updated successfully!")

                # Clear previous results when configuration is updated
                if 'cluster_map_result' in st.session_state:
                    st.session_state.pop('cluster_map_result')

            except Exception as e:
                st.error(f"Failed to save configuration: {e}")

    # Document Selection
    st.subheader("Document Selection")

    # Get document names for selection
    query_engine = get_or_create_query_engine()
    collection_info = query_engine.get_collection_info()

    if not collection_info.get("exists", False) or collection_info.get("points_count", 0) == 0:
        st.warning("No documents found in the vector database. Please process documents first.")
        document_options = []
    else:
        try:
            # Get up to 1000 chunks to extract unique document names
            chunks = query_engine.get_chunks(limit=1000)
            document_options = sorted(list(set(c['metadata'].get('file_name', 'Unknown')
                                               for c in chunks if c['metadata'].get('file_name'))))

            if not document_options:
                st.warning("No document names found in the chunks metadata.")
        except Exception as e:
            st.error(f"Error retrieving document names: {e}")
            document_options = []

    # Document selection widget
    if document_options:
        selected_documents = st.multiselect(
            "Select documents to include in clustering (leave empty to include all)",
            options=document_options,
            default=[],
            help="Only data from selected documents will be included in the clustering. If no documents are selected, all will be used."
        )

        if selected_documents:
            st.info(f"Clustering will be limited to {len(selected_documents)} selected documents.")
    else:
        st.info("No documents available for selection.")
        selected_documents = []

    # Store in session state for use in generate_cluster_map
    st.session_state.selected_documents_for_clustering = selected_documents

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

    # Handle visualization type change
    if 'previous_vis_type' not in st.session_state:
        st.session_state.previous_vis_type = current_vis_type

    # Modified approach: Don't clear results on type change, just update the type
    if st.session_state.previous_vis_type != current_vis_type:
        # Only update the type without clearing results
        logger.info(f"Visualization type changed from {st.session_state.previous_vis_type} to {current_vis_type}")
        st.session_state.previous_vis_type = current_vis_type

        # We'll still need to regenerate, but we don't need to create a new LLM instance
        # The updated create_topic_model function will reuse the existing LLM instance
        if 'cluster_map_result' in st.session_state:
            # Optional visual feedback that regeneration is needed
            st.info("You changed visualization type. Click 'Generate Cluster Map' to update the visualization.")

    # Generate cluster map when button is clicked
    if generate_btn:
        current_config_vis_type = CONFIG["clustering"].get("visualization_type", "plotly").lower()

        with progress_placeholder.container():
            st.info(f"Using visualization type: {current_config_vis_type.upper()}")
            with st.spinner(f"Generating cluster map using {current_config_vis_type.upper()} visualization..."):
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
                    # Clear any previous results on error
                    if 'cluster_map_result' in st.session_state:
                        st.session_state.pop('cluster_map_result')

    # Display cluster map if available
    if 'cluster_map_result' in st.session_state and st.session_state.cluster_map_result:
        result = st.session_state.cluster_map_result

        # Fix for topic names that might be truncated or malformed
        if 'topic_info' in result:
            # Check if topic names need cleaning
            for i, row in result['topic_info'].iterrows():
                # Check for malformed or extremely long topic names
                if len(str(row['Name'])) > 100 or '_type_' in str(row['Name']):
                    # Replace with a clean, generic name
                    result['topic_info'].at[i, 'Name'] = f"Topic {row['Topic']}"

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
            if result.get("is_static", False):
                st.pyplot(result["figure"])  # For static matplotlib figures
            elif result.get("is_datamap", False):
                try:
                    html_content = result["figure"]._repr_html_()
                    st.components.v1.html(html_content, height=800, scrolling=True)
                except Exception as e:
                    st.error(f"Error displaying DataMapPlot: {e}")
                    st.warning("Try switching to Plotly visualization or regenerating the map.")
            else:
                try:
                    st.plotly_chart(result["figure"], use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying Plotly chart: {e}")
                    st.warning("Try switching to DataMapPlot visualization or regenerating the map.")

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

        # Download functionality
    if 'cluster_map_result' in st.session_state and st.session_state.cluster_map_result:
        st.subheader("Download Results")

        result = st.session_state.cluster_map_result

        if 'topic_info' in result:
            # Create download DataFrame
            download_df = create_download_dataframe(result.get('docs_df', pd.DataFrame()), result['topic_info'])

            col1, col2 = st.columns(2)

            with col1:
                # CSV download
                csv = download_df.to_csv(index=False)
                csv_b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{csv_b64}" download="cluster_results.csv" class="btn">Download CSV</a>'
                st.markdown(href, unsafe_allow_html=True)

            with col2:
                # Excel download
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    download_df.to_excel(writer, sheet_name='Cluster Results', index=False)
                    # Create a sheet with topic info
                    result['topic_info'].to_excel(writer, sheet_name='Topics Overview', index=False)

                    # Get workbook and add formatting
                    workbook = writer.book

                    # Format for Topics Overview sheet
                    topic_sheet = writer.sheets['Topics Overview']
                    topic_sheet.set_column('A:A', 10)  # Topic ID
                    topic_sheet.set_column('B:B', 15)  # Count
                    topic_sheet.set_column('C:D', 25)  # Name and Representation

                    # Format for main results sheet
                    results_sheet = writer.sheets['Cluster Results']
                    results_sheet.set_column('A:A', 20)  # Document Name
                    results_sheet.set_column('B:B', 10)  # Page Number
                    results_sheet.set_column('C:C', 10)  # Topic ID
                    results_sheet.set_column('D:D', 25)  # Topic Name
                    results_sheet.set_column('E:E', 40)  # Topic Keywords
                    results_sheet.set_column('F:F', 15)  # Probability
                    results_sheet.set_column('G:G', 50)  # Text Content

                # Save and get value
                writer.close()
                excel_data = buffer.getvalue()

                # Create download link
                excel_b64 = base64.b64encode(excel_data).decode()
                href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{excel_b64}" download="cluster_results.xlsx" class="btn">Download Excel</a>'
                st.markdown(href, unsafe_allow_html=True)

            # Add sample data preview toggle
            with st.expander("Preview Download Data", expanded=False):
                st.dataframe(download_df.head(10))

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