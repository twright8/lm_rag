import streamlit as st
import time
import gc
import torch
import traceback
import json # For parsing potential JSON strings
from typing import List, Dict, Any, Union, Optional, Callable

# Import necessary functions/variables from other modules
from app_setup import (
    ROOT_DIR, CONFIG, logger, APHRODITE_SERVICE_AVAILABLE,
    get_or_create_query_engine, get_conversation_store,
    get_active_llm_manager, IS_OPENROUTER_ACTIVE, IS_GEMINI_ACTIVE # Import new flags
)
from src.utils.resource_monitor import log_memory_usage

# Import LLM-related modules conditionally based on backend
if IS_OPENROUTER_ACTIVE:
    from src.utils.openrouter_manager import OpenRouterManager
    AutoTokenizer = None # Not needed for OpenRouter prompt formatting
    AphroditeService = None # Placeholder
    GeminiManager = None # Placeholder
    logger.info("OpenRouter backend active for chat.")
elif IS_GEMINI_ACTIVE: # Add Gemini case
    from src.utils.gemini_manager import GeminiManager
    AutoTokenizer = None # Not needed for Gemini prompt formatting
    AphroditeService = None # Placeholder
    OpenRouterManager = None # Placeholder
    logger.info("Gemini backend active for chat.")
elif APHRODITE_SERVICE_AVAILABLE:
    from src.utils.aphrodite_service import AphroditeService, get_service
    from transformers import AutoTokenizer # Needed for Aphrodite templating
    OpenRouterManager = None # Placeholder
    GeminiManager = None # Placeholder
    logger.info("Aphrodite backend active for chat.")
else:
    # No backend available
    AutoTokenizer = None
    AphroditeService = None
    OpenRouterManager = None
    GeminiManager = None
    logger.warning("No LLM backend (Aphrodite, OpenRouter, or Gemini) available for chat.")

# Import DeepSeek manager if configured (kept for potential future use, but OpenRouter is primary API)
if CONFIG.get("deepseek", {}).get("use_api", False):
    try:
        from src.utils.deepseek_manager import DeepSeekManager
    except ImportError as e:
        logger.error(f"Failed to import DeepSeekManager: {e}. DeepSeek API will not be available.")
        DeepSeekManager = None # Placeholder
else:
    DeepSeekManager = None # Placeholder


# --- LLM Service Management (Aphrodite - Keep for sidebar control) ---
# These functions are now primarily called from app_ui_core.py (sidebar)

def start_aphrodite_service():
    """ Start the Aphrodite service process. """
    if IS_OPENROUTER_ACTIVE or IS_GEMINI_ACTIVE: # Check both API backends
         st.error("Cannot start Aphrodite service when an API backend (OpenRouter/Gemini) is active.")
         logger.error("Attempted to start Aphrodite service while an API backend is active.")
         return False
    if not APHRODITE_SERVICE_AVAILABLE:
         st.error("Aphrodite service module not available.")
         logger.error("Attempted to start Aphrodite service, but module is not available.")
         return False
    try:
        service = get_service() # Get the singleton instance
        if service.is_running():
             logger.info("Aphrodite service is already running.")
             st.session_state.aphrodite_service_running = True
             return True

        logger.info("Attempting to start Aphrodite service...")
        if service.start():
            logger.info("Aphrodite service started successfully via start()")
            st.session_state.aphrodite_service_running = True
            time.sleep(2) # Allow time for PID registration
            pid = service.process.pid if service.process else None
            process_info = {"pid": pid}
            if pid: st.session_state.aphrodite_process_info = process_info
            else: logger.warning("Aphrodite service started but failed to get PID.")
            st.session_state.llm_model_loaded = False # Model loads on demand
            log_memory_usage(logger, "Memory usage after starting LLM service")
            return True
        else:
            logger.error("Failed to start Aphrodite service (service.start() returned False).")
            st.session_state.aphrodite_service_running = False
            st.session_state.llm_model_loaded = False
            st.session_state.aphrodite_process_info = None
            return False
    except Exception as e:
        logger.error(f"Error starting Aphrodite service: {e}", exc_info=True)
        st.session_state.aphrodite_service_running = False
        st.session_state.llm_model_loaded = False
        st.session_state.aphrodite_process_info = None
        return False

def terminate_aphrodite_service():
    """ Terminate the Aphrodite service process. """
    if IS_OPENROUTER_ACTIVE or IS_GEMINI_ACTIVE: # Check both API backends
         st.error("Cannot terminate Aphrodite service when an API backend (OpenRouter/Gemini) is active.")
         logger.error("Attempted to terminate Aphrodite service while an API backend is active.")
         return False
    if not APHRODITE_SERVICE_AVAILABLE:
         st.error("Aphrodite service module not available.")
         logger.error("Attempted to terminate Aphrodite service, but module is not available.")
         return False
    try:
        logger.info("User requested Aphrodite service termination")
        service = get_service()
        if not service.is_running():
             logger.info("Aphrodite service is not running, termination request ignored.")
             st.session_state.aphrodite_service_running = False
             st.session_state.llm_model_loaded = False
             st.session_state.aphrodite_process_info = None
             return True

        success = service.shutdown()
        if success: logger.info("Aphrodite service successfully terminated via shutdown()")
        else: logger.warning("Aphrodite service shutdown command did not report full success.")

        # Update states regardless
        st.session_state.llm_model_loaded = False
        st.session_state.aphrodite_service_running = False
        st.session_state.aphrodite_process_info = None
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache(); logger.info("Cleared CUDA cache.")
        log_memory_usage(logger, "Memory usage after terminating LLM service")
        return success
    except Exception as e:
        logger.error(f"Error terminating Aphrodite service: {e}", exc_info=True)
        st.session_state.llm_model_loaded = False
        st.session_state.aphrodite_service_running = False
        st.session_state.aphrodite_process_info = None
        return False


# --- Conversation Management ---

def save_current_conversation():
    """Safely saves the currently active conversation data to disk via ConversationStore."""
    conv_id = st.session_state.get("current_conversation_id")
    conv_data = st.session_state.get("active_conversation_data")
    store = get_conversation_store() # Use helper from app_setup

    if conv_id and conv_data and store:
        logger.debug(f"Attempting to save conversation {conv_id}...")
        success = store.save_conversation(conv_id, conv_data)
        if success: logger.debug(f"Conversation {conv_id} saved successfully.")
        else:
            logger.error(f"Failed to save conversation {conv_id}.")
            st.toast(f"Error: Failed to save conversation '{conv_data.get('title', conv_id)}'.", icon="❌")
    elif conv_id or conv_data:
        logger.warning("Attempted to save conversation but ID, data, or store was missing.")


# --- LLM Response Generation ---

def generate_llm_response(
    active_llm_manager: Union[AphroditeService, OpenRouterManager, GeminiManager, None], # Add Gemini
    query_engine, # Pass QueryEngine instance (currently unused here)
    conversation_history: List[Dict[str, Any]],
    current_prompt: str,
    context_sources: List[Dict[str, Any]],
    message_placeholder, # Streamlit placeholder for the main answer
    thinking_placeholder # Streamlit placeholder for the thinking process
    ) -> Dict[str, Any]:
    """
    Generates response using the active LLM backend (Aphrodite/OpenRouter/Gemini).
    Handles formatting input appropriately:
    - Aphrodite: Uses chat templates via local tokenizer.
    - OpenRouter: Constructs OpenAI messages list.
    - Gemini: Constructs single prompt string (or list for streaming).
    Manages streaming updates to UI placeholders for OpenRouter/Gemini.

    Args:
        active_llm_manager: Instance of AphroditeService, OpenRouterManager, or GeminiManager, or None.
        query_engine: Instance of QueryEngine (passed but currently unused).
        conversation_history: List of previous message dicts [{'role': ..., 'content': ...}].
        current_prompt: The user's latest prompt string.
        context_sources: List of source dicts from RAG (can be empty).
        message_placeholder: Streamlit st.empty() object for the main answer.
        thinking_placeholder: Streamlit st.empty() object for the thinking process.

    Returns:
        Dictionary containing the *final* state after generation:
        - 'final_answer': The complete generated response string.
        - 'final_thinking': The complete reasoning string (None for OpenRouter/Aphrodite/Gemini).
        - 'error': Error message string if generation failed.
    """
    logger.info("Initiating LLM response generation...")
    final_data = {"final_answer": "", "final_thinking": None, "error": None} # Thinking not supported by default
    tokenizer = None
    model_name = None
    full_prompt_for_llm = None # Used by Aphrodite/Gemini (non-streaming)
    messages_for_llm = None # Used by OpenRouter
    contents_for_gemini_stream = None # Used by Gemini (streaming)

    if active_llm_manager is None:
        final_data['error'] = "Error: No active LLM manager available."
        message_placeholder.error(final_data['error'])
        logger.error("generate_llm_response called with no active_llm_manager.")
        return final_data

    # --- Determine LLM Backend Type ---
    is_openrouter = isinstance(active_llm_manager, OpenRouterManager) if OpenRouterManager else False
    is_aphrodite = isinstance(active_llm_manager, AphroditeService) if AphroditeService else False
    is_gemini = isinstance(active_llm_manager, GeminiManager) if GeminiManager else False # Add Gemini check

    try:
        # --- Prepare Context and History (Common Logic) ---
        max_turns = CONFIG.get("conversation", {}).get("max_history_turns", 5)
        valid_history = [msg for msg in conversation_history if isinstance(msg, dict) and "role" in msg and "content" in msg]
        history_subset = valid_history[-(max_turns * 2):] # Get last N turns (user + assistant)

        context_str = ""
        if context_sources:
            logger.info(f"Providing {len(context_sources)} context sources to LLM.")
            valid_sources = [src for src in context_sources if isinstance(src, dict)]
            # Use original_text for context if available
            context_texts = [src.get('original_text', src.get('text', '')) for src in valid_sources]
            context_str = "## Context Documents:\n" + "\n\n".join([f"[{i + 1}] {text}" for i, text in enumerate(context_texts)])
        else:
            logger.info("No context provided to LLM for this turn.")
            context_str = "## Context Documents:\nNo context documents were retrieved or provided for this query."

        # Define system prompt (consistent across backends)
        system_content = """You are an expert assistant specializing in anti-corruption investigations and analysis. Your responses should be detailed, factual, and based ONLY on the provided context if available. If the answer is not found in the context or conversation history, state that clearly. Do not make assumptions or use external knowledge. Cite sources using [index] notation where applicable based on the provided context."""

        # --- Input Formatting: OpenRouter vs Aphrodite vs Gemini ---
        if is_openrouter:
            # --- OpenRouter Path: Construct OpenAI Messages List ---
            model_name = active_llm_manager.models.get("chat", "Default OpenRouter Chat")
            logger.info(f"Using OpenRouter model: {model_name}. Constructing messages list.")

            user_content = f"""{context_str}

Based on the conversation history and the context provided above (if any), answer the following question:
Question: {current_prompt}"""

            messages_for_llm = [{"role": "system", "content": system_content}]
            messages_for_llm.extend(history_subset) # Add validated history
            messages_for_llm.append({"role": "user", "content": user_content})

            logger.debug(f"Constructed Messages for OpenRouter (sample):\n{messages_for_llm}")

        elif is_gemini:
            # --- Gemini Path: Construct Prompt String(s) ---
            model_name = active_llm_manager.models.get("chat", "gemini-1.5-flash-latest")
            logger.info(f"Using Gemini model: {model_name}. Constructing prompt string(s).")

            # Format history and context into a single string for the prompt
            # Simple alternating format for history
            history_str = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in history_subset])
            user_content_for_prompt = f"""{context_str}

## Conversation History:
{history_str}

## Current Question:
{current_prompt}

## Your Answer:"""

            # Combine system and user content for the final prompt string
            # (Gemini's generate_content takes a single string or list)
            full_prompt_for_llm = f"System: {system_content}\n\n{user_content_for_prompt}"
            # For streaming, Gemini expects a list, use the same content
            contents_for_gemini_stream = [full_prompt_for_llm]

            logger.debug(f"Constructed Prompt String for Gemini (start): {full_prompt_for_llm[:300]}...")
            logger.debug(f"Constructed Contents List for Gemini Stream (start): {contents_for_gemini_stream[0][:300]}...")


        elif is_aphrodite:
            # --- Aphrodite Path: Prepare Structured Messages & Load Tokenizer ---
            status = active_llm_manager.get_status()
            model_name = status.get("current_model")
            if not model_name:
                 final_data['error'] = "Error: Aphrodite service has no model loaded."
                 message_placeholder.error(final_data['error'])
                 return final_data
            logger.info(f"Using Aphrodite model: {model_name}. Preparing structured messages.")

            # Load tokenizer for the Aphrodite model
            try:
                 if AutoTokenizer is None: raise ImportError("Transformers AutoTokenizer not available.")
                 tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

                 if tokenizer.chat_template is None:
                      logger.error(f"Tokenizer for {model_name} lacks a chat template. Cannot format prompt.")
                      logger.warning("Attempting basic prompt concatenation as fallback.")
                      history_str = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in history_subset])
                      user_content = f"{context_str}\n\nHistory:\n{history_str}\n\nUser: {current_prompt}\nAssistant:"
                      full_prompt_for_llm = f"System: {system_content}\n{user_content}"
                      logger.debug(f"Using fallback concatenation for {model_name}. Length: {len(full_prompt_for_llm)}")
                 else:
                      # Prepare structured messages for the template
                      user_content = f"""{context_str}

Based on the conversation history and the context provided above (if any), answer the following question:
Question: {current_prompt}"""

                      messages_for_template = [{"role": "system", "content": system_content}]
                      messages_for_template.extend(history_subset) # Add validated history
                      messages_for_template.append({"role": "user", "content": user_content})

                      # Apply the template
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

            except ImportError as e:
                 logger.error(f"ImportError during tokenizer loading: {e}")
                 final_data['error'] = f"Error loading tokenizer components: {e}"
                 message_placeholder.error(final_data['error'])
                 return final_data
            except Exception as e:
                 logger.error(f"Failed to load tokenizer for {model_name}: {e}", exc_info=True)
                 final_data['error'] = f"Error loading tokenizer for {model_name}: {e}"
                 message_placeholder.error(final_data['error'])
                 return final_data
        else:
             final_data['error'] = "Error: Unsupported or unavailable LLM manager type."
             message_placeholder.error(final_data['error'])
             logger.error(f"Unsupported llm_manager type: {type(active_llm_manager)}")
             return final_data

    except Exception as e:
        logger.error(f"Error during input preparation: {e}", exc_info=True)
        final_data['error'] = f"Input Preparation Error: {str(e)}"
        message_placeholder.error(final_data['error'])
        return final_data

    # --- 3. Initiate LLM Call & Handle Streaming/Response ---
    llm_called = False
    try:
        # --- OpenRouter Streaming Path ---
        if is_openrouter:
            if messages_for_llm is None:
                 final_data['error'] = "Error: Messages list for OpenRouter was not generated."
                 message_placeholder.error(final_data['error']); return final_data

            llm_called = True
            logger.info("Using OpenRouterManager - initiating streaming generation.")
            thinking_placeholder.empty() # Clear thinking placeholder (not supported)

            full_response_buffer = ""
            def openrouter_stream_callback(token):
                nonlocal full_response_buffer
                try:
                    if isinstance(token, str):
                        full_response_buffer += token
                        message_placeholder.markdown(full_response_buffer + "▌")
                except Exception as callback_e:
                     logger.error(f"Error in openrouter_stream_callback: {callback_e}", exc_info=True)
                     st.toast(f"UI Update Error: {callback_e}", icon="⚠️")

            # Call OpenRouterManager's generate method with the messages list
            response = active_llm_manager.generate_chat(
                messages=messages_for_llm,
                stream_callback=openrouter_stream_callback
            )

            # Final update after streaming finishes
            if response.get("status") == "success":
                 final_data['final_answer'] = response.get("result", "")
                 message_placeholder.markdown(final_data['final_answer']) # Final answer without cursor
            else:
                 final_data['error'] = response.get("error", "Unknown OpenRouter error")
                 message_placeholder.error(f"Error: {final_data['error']}")

        # --- Gemini Streaming Path ---
        elif is_gemini:
            if contents_for_gemini_stream is None:
                 final_data['error'] = "Error: Contents list for Gemini stream was not generated."
                 message_placeholder.error(final_data['error']); return final_data

            llm_called = True
            logger.info("Using GeminiManager - initiating streaming generation.")
            thinking_placeholder.empty() # Clear thinking placeholder

            full_response_buffer = ""
            def gemini_stream_callback(token):
                nonlocal full_response_buffer
                try:
                    if isinstance(token, str):
                        full_response_buffer += token
                        message_placeholder.markdown(full_response_buffer + "▌")
                except Exception as callback_e:
                     logger.error(f"Error in gemini_stream_callback: {callback_e}", exc_info=True)
                     st.toast(f"UI Update Error: {callback_e}", icon="⚠️")

            # Call GeminiManager's generate method with the prompt string
            # Pass the callback for streaming
            response = active_llm_manager.generate_chat(
                prompt=contents_for_gemini_stream[0], # Pass the single prompt string from the list
                stream_callback=gemini_stream_callback,
                model_name=model_name # Pass model name if needed
            )

            # Final update after streaming finishes
            if response.get("status") == "success":
                 final_data['final_answer'] = response.get("result", "")
                 message_placeholder.markdown(final_data['final_answer']) # Final answer without cursor
            else:
                 final_data['error'] = response.get("error", "Unknown Gemini error")
                 message_placeholder.error(f"Error: {final_data['error']}")


        # --- Aphrodite Non-Streaming Path ---
        elif is_aphrodite:
            if full_prompt_for_llm is None:
                 final_data['error'] = "Error: Final prompt string for Aphrodite was not generated."
                 message_placeholder.error(final_data['error']); return final_data

            llm_called = True
            logger.info("Using AphroditeService - initiating non-streaming generation.")
            thinking_placeholder.empty() # Clear thinking placeholder

            # Pass the fully formatted prompt string to generate_chat
            aphrodite_response = active_llm_manager.generate_chat(prompt=full_prompt_for_llm)

            if isinstance(aphrodite_response, dict) and aphrodite_response.get("status") == "success":
                final_data['final_answer'] = aphrodite_response.get("result", "").strip()
                message_placeholder.markdown(final_data['final_answer'])
            else:
                error_msg = "Unknown error from Aphrodite service"
                if isinstance(aphrodite_response, dict): error_msg = aphrodite_response.get("error", error_msg)
                elif isinstance(aphrodite_response, str): error_msg = aphrodite_response # Handle error string
                final_data['error'] = error_msg
                message_placeholder.error(f"Error: {final_data['error']}")
        else:
             # Should not be reached
             final_data['error'] = "Error: LLM Manager configuration issue."
             message_placeholder.error(final_data['error'])
             logger.error("Reached unexpected state in LLM call logic.")

        # Final validation
        if not final_data['final_answer'] and not final_data['error'] and llm_called:
            final_data['error'] = "Error: LLM returned no answer content."
            message_placeholder.warning(final_data['error'])

    except Exception as e:
        logger.error(f"Critical error during LLM generation coordination: {e}", exc_info=True)
        final_data['error'] = f"LLM Generation System Error: {str(e)}"
        if not final_data['final_answer']: final_data['final_answer'] = f"Sorry, a critical error occurred: {str(e)}"
        message_placeholder.error(f"Error: {final_data['error']}")
        thinking_placeholder.empty()

    logger.info(f"LLM generation process finished. Final Answer length: {len(final_data.get('final_answer', ''))}. Error: {final_data.get('error')}")
    return final_data


# --- Chat Message Handling Orchestration ---

def handle_chat_message(prompt: str):
    """
    Orchestrates processing a user's chat message using the active LLM backend.
    Handles retrieval, calls the LLM, updates state, and saves.

    Args:
        prompt: The user's input string.
    """
    logger.info(f"Handling chat message: '{prompt[:50]}...'")

    # --- 1. Get Dependencies ---
    query_engine = get_or_create_query_engine()
    conversation_store = get_conversation_store()
    active_conv_data = st.session_state.get("active_conversation_data")
    llm_manager = get_active_llm_manager() # Get the initialized manager

    if not query_engine or not conversation_store or not active_conv_data:
        logger.error("handle_chat_message called without necessary components (engine, store, or active conversation).")
        st.error("Cannot process message. Please ensure system is initialized and a conversation is active.")
        return

    # Check if the LLM manager is ready
    llm_ready = False
    llm_status_message = ""
    if isinstance(llm_manager, OpenRouterManager):
        if llm_manager.client:
            llm_ready = True
            llm_status_message = f"OpenRouter ready (Model: {llm_manager.models.get('chat')})."
        else:
            llm_status_message = "OpenRouter manager initialized but client failed (check API key?)."
    elif isinstance(llm_manager, GeminiManager): # Add Gemini check
        if llm_manager.client:
            llm_ready = True
            llm_status_message = f"Gemini ready (Model: {llm_manager.models.get('chat')})."
        else:
            llm_status_message = "Gemini manager initialized but client failed (check API key?)."
    elif isinstance(llm_manager, AphroditeService):
        if llm_manager.is_running() and st.session_state.get("llm_model_loaded"):
            llm_ready = True
            model_name = st.session_state.get("aphrodite_process_info", {}).get("model_name", "Unknown")
            llm_status_message = f"Local LLM service ready (Model: {model_name})."
        elif llm_manager.is_running():
            llm_status_message = "Local LLM service running, but no model loaded."
        else:
            llm_status_message = "Local LLM service not running."
    else:
        llm_status_message = "No valid LLM manager found."

    if not llm_ready:
        st.warning(f"LLM not ready: {llm_status_message}")
        # Optionally add buttons to start service etc.
        if LLM_BACKEND == "aphrodite" and APHRODITE_SERVICE_AVAILABLE and not st.session_state.get("aphrodite_service_running"):
             if st.button("Start LLM Service Now"): start_aphrodite_service(); st.rerun()
        return # Don't proceed if LLM isn't ready

    # --- 2. Get Current Conversation State ---
    try:
        conversation_history_for_llm = active_conv_data.get("messages", [])[:] # Shallow copy
    except Exception as e:
         logger.error(f"Failed to access active conversation data: {e}", exc_info=True)
         st.error("Internal error: Could not access conversation data."); return

    # --- 3. Determine Retrieval Need ---
    retrieve_now = st.session_state.get("retrieval_enabled_for_next_turn", False)
    logger.info(f"Retrieval decision for this turn: {'ENABLED' if retrieve_now else 'DISABLED'}.")

    # --- 4. Add User Message ---
    user_message_id = f"msg_user_{len(active_conv_data.get('messages', [])) + 1}"
    user_message = {"role": "user", "content": prompt, "timestamp": time.time(), "id": user_message_id}
    st.session_state.active_conversation_data["messages"].append(user_message)
    st.session_state.ui_chat_display.append({"role": "user", "content": prompt})
    # User message renders on rerun after chat_input

    # --- 5. Perform Retrieval (Conditional) ---
    sources = []
    retrieval_info_msg = "Retrieval skipped (toggle was off)."
    retrieval_status_placeholder = st.empty()

    if retrieve_now:
        retrieval_info_msg = "Performing RAG retrieval..."
        logger.info(retrieval_info_msg)
        with retrieval_status_placeholder.status("Retrieving context...", expanded=False):
            try:
                st.write("Searching relevant documents...")
                sources = query_engine.retrieve(prompt)
                retrieval_info_msg = f"Retrieved {len(sources)} relevant source(s)."
                st.write(retrieval_info_msg)
                logger.info(retrieval_info_msg)
            except Exception as e:
                logger.error(f"Error during retrieval: {e}", exc_info=True)
                retrieval_info_msg = f"Error during retrieval: {e}"
                st.error(retrieval_info_msg); sources = []
        if "Error" not in retrieval_info_msg:
             time.sleep(0.5); retrieval_status_placeholder.empty()
    else:
        logger.info(retrieval_info_msg)
        retrieval_status_placeholder.caption(retrieval_info_msg)

    # --- 6. Initiate LLM Response Generation ---
    message_placeholder_container = st.empty()
    thinking_placeholder_container = st.empty()
    sources_placeholder_container = st.empty()

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        thinking_placeholder = st.empty() # Will likely remain empty

        with st.spinner("Assistant is thinking..."):
            final_result = generate_llm_response(
                active_llm_manager=llm_manager, # Pass the active manager
                query_engine=query_engine,
                conversation_history=st.session_state.active_conversation_data["messages"], # Includes user msg
                current_prompt=prompt,
                context_sources=sources,
                message_placeholder=message_placeholder,
                thinking_placeholder=thinking_placeholder
            )

    # --- Generation Complete ---
    final_answer = final_result.get("final_answer", "")
    final_thinking = final_result.get("final_thinking") # Likely None
    error = final_result.get("error")
    retrieval_status_placeholder.empty() # Clear status message

    # Populate sources expander
    if not error and sources and CONFIG.get("conversation", {}).get("persist_retrieved_context", True):
         with sources_placeholder_container.expander("View Sources Used", expanded=False):
             for i, source in enumerate(sources):
                 if isinstance(source, dict):
                     score = source.get('score', 0.0)
                     text = source.get('original_text', source.get('text', '')) # Prefer original
                     meta = source.get('metadata', {})
                     st.markdown(f"**Source {i + 1} (Score: {score:.2f}):**")
                     st.markdown(f"> {text}")
                     st.caption(f"Doc: {meta.get('file_name', 'N/A')} | Page: {meta.get('page_num', 'N/A')} | Chunk: {meta.get('chunk_id', 'N/A')}")
                     st.markdown("---")
                 else: st.warning(f"Source {i+1} has unexpected format: {type(source)}")

    # --- 7. Add Final Assistant Message to History ---
    assistant_message_id = f"msg_asst_{len(active_conv_data['messages'])}"
    assistant_message = {
        "role": "assistant",
        "content": final_answer if not error else f"Error generating response: {error}",
        "timestamp": time.time(),
        "id": assistant_message_id
    }
    persist_context = CONFIG.get("conversation", {}).get("persist_retrieved_context", True)
    if sources and not error and persist_context:
        assistant_message["used_context"] = [
            {"text": s.get('original_text', s.get('text', '')), # Store original text used
             "metadata": s.get('metadata', {}), "score": s.get('score', 0.0), "source_index": i+1}
            for i, s in enumerate(sources) if isinstance(s, dict)
        ]
    # No thinking process to store for OpenRouter/Aphrodite/Gemini

    st.session_state.active_conversation_data["messages"].append(assistant_message)

    # --- 8. Update UI Display List ---
    st.session_state.ui_chat_display.append({
        "role": "assistant",
        "content": assistant_message["content"],
        "thinking": None, # No thinking
        "sources": assistant_message.get("used_context")
    })

    # --- 9. Auto-Save Conversation ---
    if CONFIG.get("conversation", {}).get("auto_save_on_turn", True):
        save_current_conversation()

    logger.info("Finished handling chat message cycle.")

    # --- 10. Trigger Rerun ---
    # Only rerun if the message wasn't an error display
    if not error:
        st.rerun()
