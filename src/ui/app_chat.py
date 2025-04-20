# app_chat.py
import streamlit as st
import time
import gc
import torch
import traceback
from typing import List, Dict, Any, Union, Optional

# Import necessary functions/variables from other modules
from app_setup import ROOT_DIR, CONFIG, logger, APHRODITE_SERVICE_AVAILABLE, get_service, get_or_create_query_engine, get_conversation_store
from src.utils.resource_monitor import log_memory_usage

# Import LLM-related modules conditionally
if APHRODITE_SERVICE_AVAILABLE:
    from transformers import AutoTokenizer
    from src.utils.aphrodite_service import AphroditeService # For type hinting if needed
else:
    AutoTokenizer = None # Placeholder
    AphroditeService = None # Placeholder

# Import DeepSeek manager if configured
if CONFIG.get("deepseek", {}).get("use_api", False):
    try:
        from src.utils.deepseek_manager import DeepSeekManager
    except ImportError as e:
        logger.error(f"Failed to import DeepSeekManager: {e}. DeepSeek API will not be available.")
        DeepSeekManager = None # Placeholder
else:
    DeepSeekManager = None # Placeholder


# --- LLM Service Management (Aphrodite) ---

def start_aphrodite_service():
    """
    Start the Aphrodite service process.
    """
    if not APHRODITE_SERVICE_AVAILABLE:
         st.error("Aphrodite service module not available.")
         logger.error("Attempted to start Aphrodite service, but module is not available.")
         return False
    try:
        service = get_service()
        if service.is_running():
             logger.info("Aphrodite service is already running.")
             # Ensure state reflects reality
             st.session_state.aphrodite_service_running = True
             # Optionally sync model state here too if needed
             return True

        logger.info("Attempting to start Aphrodite service...")
        # Start the service without loading a model yet
        if service.start():
            logger.info("Aphrodite service started successfully via start()")
            st.session_state.aphrodite_service_running = True

            # Give it a moment to initialize fully before getting PID
            time.sleep(2)
            pid = service.process.pid if service.process else None

            # Save process info (PID only needed now)
            process_info = {"pid": pid}
            if process_info["pid"]:
                st.session_state.aphrodite_process_info = process_info
                logger.info(f"Saved Aphrodite process info: PID={process_info.get('pid')}")
            else:
                 st.session_state.aphrodite_process_info = None
                 logger.warning("Aphrodite service started but failed to get PID.")

            # Don't set llm_model_loaded here, it happens on demand during processing/querying
            st.session_state.llm_model_loaded = False
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
    """
    Terminate the Aphrodite service process.
    """
    if not APHRODITE_SERVICE_AVAILABLE:
         st.error("Aphrodite service module not available.")
         logger.error("Attempted to terminate Aphrodite service, but module is not available.")
         return False
    try:
        logger.info("User requested Aphrodite service termination")
        service = get_service()
        if not service.is_running():
             logger.info("Aphrodite service is not running, termination request ignored.")
             # Ensure state is correct
             st.session_state.aphrodite_service_running = False
             st.session_state.llm_model_loaded = False
             st.session_state.aphrodite_process_info = None
             return True # Considered successful as it's already stopped

        success = service.shutdown()

        if success:
            logger.info("Aphrodite service successfully terminated via shutdown()")
        else:
            # This might happen if the process ended unexpectedly or shutdown timed out
            logger.warning("Aphrodite service shutdown command did not report full success. Process might already be gone or unresponsive.")

        # Update states regardless of success to avoid stuck state
        st.session_state.llm_model_loaded = False
        st.session_state.aphrodite_service_running = False
        st.session_state.aphrodite_process_info = None

        # Force garbage collection and clear CUDA cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache after LLM service termination.")
        log_memory_usage(logger, "Memory usage after terminating LLM service")

        return success # Return the reported success status
    except Exception as e:
        logger.error(f"Error terminating Aphrodite service: {e}", exc_info=True)
        # Update states even on error to prevent inconsistent UI
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
        # Ensure messages are up-to-date before saving (though they should be)
        # conv_data["messages"] = st.session_state.ui_chat_display # Or use active_conversation_data directly
        success = store.save_conversation(conv_id, conv_data)
        if success:
            logger.debug(f"Conversation {conv_id} saved successfully.")
        else:
            logger.error(f"Failed to save conversation {conv_id}.")
            st.toast(f"Error: Failed to save conversation '{conv_data.get('title', conv_id)}'.", icon="❌")
    elif conv_id or conv_data:
        logger.warning("Attempted to save conversation but ID, data, or store was missing.")


# --- LLM Response Generation ---

def generate_llm_response(
    llm_manager: Union[AphroditeService, DeepSeekManager, None], # Accept potential None
    query_engine, # Pass QueryEngine instance if needed (e.g., for context verification - currently unused here)
    conversation_history: List[Dict[str, Any]],
    current_prompt: str,
    context_sources: List[Dict[str, Any]],
    message_placeholder, # Streamlit placeholder for the main answer
    thinking_placeholder # Streamlit placeholder for the thinking process
    ) -> Dict[str, Any]:
    """
    Generates response using the LLM (Aphrodite/DeepSeek).
    Handles formatting input appropriately: uses chat templates for Aphrodite via local tokenizer,
    constructs a raw prompt string for DeepSeek API.
    Manages streaming updates to UI placeholders for DeepSeek.

    Args:
        llm_manager: Instance of AphroditeService or DeepSeekManager, or None if unavailable.
        query_engine: Instance of QueryEngine (passed but currently unused in this function).
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
    tokenizer = None
    model_name = None
    full_prompt_for_llm = None
    messages_for_template = None

    if llm_manager is None:
        final_data['error'] = "Error: No LLM manager available (Aphrodite or DeepSeek)."
        message_placeholder.error(final_data['error'])
        logger.error("generate_llm_response called with no llm_manager.")
        return final_data

    # --- Determine LLM Type and Prepare Input ---
    # Check if it's DeepSeek by checking the type/class name or specific attributes
    is_deepseek = DeepSeekManager is not None and isinstance(llm_manager, DeepSeekManager)

    try:
        # --- Prepare Context and History (Common Logic) ---
        max_turns = CONFIG.get("conversation", {}).get("max_history_turns", 5)
        # Ensure history is a list of dicts with 'role' and 'content'
        valid_history = [msg for msg in conversation_history if isinstance(msg, dict) and "role" in msg and "content" in msg]
        history_subset = valid_history[-(max_turns * 2):] # Get last N turns (user + assistant)

        context_str = ""
        if context_sources:
            logger.info(f"Providing {len(context_sources)} context sources to LLM.")
            # Ensure context sources are dicts and have text
            valid_sources = [src for src in context_sources if isinstance(src, dict)]
            context_texts = [src.get('original_text', src.get('text', '')) for src in valid_sources]
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

            system_block = f"""System: You are an expert assistant specializing in anti-corruption investigations and analysis. Your responses should be detailed, factual, and based ONLY on the provided context if available. If the answer is not found in the context or conversation history, state that clearly. Do not make assumptions or use external knowledge. Cite sources using [index] notation where applicable based on the provided context."""

            history_block = "## Conversation History:\n"
            if history_subset:
                 history_block += "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history_subset])
            else:
                 history_block += "No previous conversation history."

            user_query_block = f"""{context_str}

Based on the conversation history and the context provided above (if any), answer the following question:
Question: {current_prompt}"""

            full_prompt_for_llm = f"{system_block}\n\n{history_block}\n\n{user_query_block}\n\nAssistant:"
            logger.debug(f"Constructed Raw Prompt for DeepSeek (sample):\n{full_prompt_for_llm[:500]}...")

        elif APHRODITE_SERVICE_AVAILABLE and isinstance(llm_manager, AphroditeService):
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
                 # Ensure AutoTokenizer is available
                 if AutoTokenizer is None:
                      raise ImportError("Transformers AutoTokenizer not available.")
                 tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                 if tokenizer.chat_template is None:
                      logger.error(f"Tokenizer for {model_name} lacks a chat template. Cannot format prompt.")
                      # Attempt fallback to basic concatenation if template missing
                      logger.warning("Attempting basic prompt concatenation as fallback.")
                      system_content = "System: You are an expert assistant..." # Keep system prompt simple
                      user_content = f"{context_str}\n\nHistory:\n" + "\n".join([f"{m['role']}: {m['content']}" for m in history_subset]) + f"\n\nUser: {current_prompt}\nAssistant:"
                      full_prompt_for_llm = system_content + "\n" + user_content
                      logger.debug(f"Using fallback concatenation for {model_name}. Length: {len(full_prompt_for_llm)}")

                 else:
                      # Prepare structured messages for the template
                      system_content = """You are an expert assistant specializing in anti-corruption investigations and analysis. Your responses should be detailed, factual, and based ONLY on the provided context if available. If the answer is not found in the context or conversation history, state that clearly. Do not make assumptions or use external knowledge. Cite sources using [index] notation where applicable based on the provided context."""
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
             # This case handles if llm_manager is neither DeepSeek nor Aphrodite, or Aphrodite is unavailable
             final_data['error'] = "Error: Unsupported or unavailable LLM manager type."
             message_placeholder.error(final_data['error'])
             logger.error(f"Unsupported llm_manager type: {type(llm_manager)}")
             return final_data


    except Exception as e:
        logger.error(f"Error during input preparation: {e}", exc_info=True)
        final_data['error'] = f"Input Preparation Error: {str(e)}"
        message_placeholder.error(final_data['error'])
        return final_data

    # --- 3. Initiate LLM Call & Handle Streaming/Response ---
    llm_called = False
    try:
        if full_prompt_for_llm is None:
             final_data['error'] = "Error: Final prompt string for LLM was not generated."
             message_placeholder.error(final_data['error'])
             logger.error("LLM call aborted because full_prompt_for_llm is None.")
             return final_data

        # --- DeepSeek Streaming Path ---
        if is_deepseek:
            llm_called = True
            logger.info("Using DeepSeekManager - initiating streaming generation with raw prompt.")

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
                            # Update thinking placeholder only if content changes significantly
                            if not thinking_displayed and thinking_chunk.strip(): thinking_displayed = True
                            if thinking_displayed:
                                with thinking_placeholder.container():
                                    st.markdown('<div class="thinking-title">💭 Reasoning Process (Live):</div>', unsafe_allow_html=True)
                                    st.markdown(f'<div class="thinking-box">{full_thinking_buffer}▌</div>', unsafe_allow_html=True)
                    elif isinstance(token_or_thinking, str):
                        full_response_buffer += token_or_thinking
                        message_placeholder.markdown(full_response_buffer + "▌")
                except Exception as callback_e:
                     logger.error(f"Error in deepseek_stream_callback: {callback_e}", exc_info=True)
                     # Avoid crashing the main thread due to UI errors in callback
                     st.toast(f"UI Update Error: {callback_e}", icon="⚠️")


            # Pass the raw prompt string to DeepSeekManager's generate method
            aggregated_answer = llm_manager.generate(
                prompt=full_prompt_for_llm, # Pass the raw string
                stream_callback=deepseek_stream_callback
            )

            # Final update after streaming finishes
            if isinstance(aggregated_answer, str) and aggregated_answer.startswith("Error:"):
                 final_data['error'] = aggregated_answer
                 message_placeholder.error(aggregated_answer)
            else:
                 final_data['final_answer'] = aggregated_answer if isinstance(aggregated_answer, str) else str(aggregated_answer)
                 final_data['final_thinking'] = full_thinking_buffer
                 message_placeholder.markdown(final_data['final_answer']) # Final answer without cursor
                 if thinking_displayed:
                     with thinking_placeholder.container(): # Replace live thinking with expander
                          with st.expander("💭 Reasoning Process", expanded=False):
                              st.markdown(f'<div class="thinking-box">{final_data["final_thinking"]}</div>', unsafe_allow_html=True)
                 else:
                      thinking_placeholder.empty() # Clear if no thinking was displayed


        # --- Aphrodite Non-Streaming Path ---
        elif APHRODITE_SERVICE_AVAILABLE and isinstance(llm_manager, AphroditeService):
            llm_called = True
            logger.info("Using AphroditeService - initiating non-streaming generation with formatted prompt.")
            # Pass the fully formatted prompt string to generate_chat
            aphrodite_response = llm_manager.generate_chat(prompt=full_prompt_for_llm) # Uses generate_chat

            if isinstance(aphrodite_response, dict) and aphrodite_response.get("status") == "success":
                final_data['final_answer'] = aphrodite_response.get("result", "").strip()
                message_placeholder.markdown(final_data['final_answer'])
                final_data['final_thinking'] = None # Aphrodite generate_chat doesn't provide thinking stream
                thinking_placeholder.empty() # Clear thinking placeholder
            else:
                error_msg = "Unknown error from Aphrodite service"
                if isinstance(aphrodite_response, dict):
                    error_msg = aphrodite_response.get("error", error_msg)
                elif isinstance(aphrodite_response, str):
                     error_msg = aphrodite_response # Handle case where service returns error string directly
                final_data['error'] = error_msg
                message_placeholder.error(f"Error: {final_data['error']}")
                thinking_placeholder.empty() # Clear thinking placeholder
        else:
             # Should not be reached if initial checks are correct
             final_data['error'] = "Error: LLM Manager configuration issue."
             message_placeholder.error(final_data['error'])
             logger.error("Reached unexpected state in LLM call logic.")


        # Final validation
        if not final_data['final_answer'] and not final_data['error'] and llm_called:
            final_data['error'] = "Error: LLM returned no answer content."
            message_placeholder.warning(final_data['error']) # Use warning for empty content

    except Exception as e:
        logger.error(f"Critical error during LLM generation coordination: {e}", exc_info=True)
        final_data['error'] = f"LLM Generation System Error: {str(e)}"
        # Avoid overwriting potentially useful partial answer if error happens late
        if not final_data['final_answer']:
             final_data['final_answer'] = f"Sorry, a critical error occurred: {str(e)}"
        message_placeholder.error(f"Error: {final_data['error']}")
        thinking_placeholder.empty() # Clear thinking on critical error

    logger.info(f"LLM generation process finished. Final Answer length: {len(final_data.get('final_answer', ''))}. Error: {final_data.get('error')}")
    return final_data


# --- Chat Message Handling Orchestration ---

def handle_chat_message(prompt: str):
    """
    Orchestrates processing a user's chat message. Handles retrieval based on user
    toggle state AT THE TIME OF SUBMISSION, calls the LLM, updates state, and saves.

    Args:
        prompt: The user's input string.
    """
    logger.info(f"Handling chat message: '{prompt[:50]}...'")

    # --- 1. Get Dependencies ---
    query_engine = get_or_create_query_engine()
    conversation_store = get_conversation_store()
    active_conv_data = st.session_state.get("active_conversation_data")

    if not query_engine or not conversation_store or not active_conv_data:
        logger.error("handle_chat_message called without necessary components (engine, store, or active conversation).")
        st.error("Cannot process message. Please ensure system is initialized and a conversation is active.")
        return

    # Determine which LLM manager to use
    use_deepseek = CONFIG.get("deepseek", {}).get("use_api", False)
    llm_manager = None
    if use_deepseek:
        if DeepSeekManager:
            # Initialize DeepSeek manager if needed (or get existing instance)
            # This assumes DeepSeekManager handles its own singleton or state management if needed
            deepseek_manager = DeepSeekManager(CONFIG) # Re-creating might be inefficient, consider singleton pattern
            if not deepseek_manager.client:
                st.warning("DeepSeek API not properly configured. Check settings.")
                return
            llm_manager = deepseek_manager
        else:
             st.error("DeepSeek API is enabled but the manager failed to load.")
             return
    elif APHRODITE_SERVICE_AVAILABLE:
        service = get_service()
        if service.is_running() and st.session_state.get("llm_model_loaded"):
            llm_manager = service
        else:
            st.warning("Local LLM service is not ready (running and model loaded). Cannot generate response.")
            return # Don't proceed if local LLM isn't ready
    else:
        st.error("No valid LLM service is available or configured.")
        return

    # --- 2. Get Current Conversation State ---
    try:
        # Work directly with session state dictionary
        conversation_history_for_llm = active_conv_data.get("messages", [])[:] # Shallow copy for passing to LLM
    except Exception as e:
         logger.error(f"Failed to access active conversation data: {e}", exc_info=True)
         st.error("Internal error: Could not access conversation data.")
         return

    # --- 3. Determine Retrieval Need for THIS turn ---
    retrieve_now = st.session_state.get("retrieval_enabled_for_next_turn", False) # Default to False if key missing
    logger.info(f"Retrieval decision for this turn: {'ENABLED' if retrieve_now else 'DISABLED'} (based on checkbox state when prompt was submitted).")
    # The state variable 'retrieval_enabled_for_next_turn' is managed by the checkbox key directly.
    # It will reflect the user's choice at the time of submission. It naturally resets on rerun unless checked again.

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
    # Render the user message instantly - This happens naturally on rerun after chat_input

    # --- 5. Perform Retrieval (Conditional) ---
    sources = []
    retrieval_info_msg = "Retrieval skipped (toggle was off)."
    retrieval_status_placeholder = st.empty() # Placeholder for status messages

    if retrieve_now:
        retrieval_info_msg = "Performing RAG retrieval..."
        logger.info(retrieval_info_msg)
        with retrieval_status_placeholder.status("Retrieving context...", expanded=False): # Use status context manager
            try:
                st.write("Searching relevant documents...")
                sources = query_engine.retrieve(prompt) # Assumes retrieve returns list of dicts
                retrieval_info_msg = f"Retrieved {len(sources)} relevant source(s)."
                st.write(retrieval_info_msg)
                logger.info(retrieval_info_msg)
            except Exception as e:
                logger.error(f"Error during retrieval: {e}", exc_info=True)
                retrieval_info_msg = f"Error during retrieval: {e}"
                st.error(retrieval_info_msg) # Show error within status
                sources = []
        # Status context manager automatically closes, no need to empty manually unless error
        if "Error" in retrieval_info_msg:
             pass # Keep error message displayed by status context
        else:
             time.sleep(0.5) # Brief pause before clearing success message
             retrieval_status_placeholder.empty()

    else:
        logger.info(retrieval_info_msg)
        # Display caption only if skipping, otherwise status handles it
        retrieval_status_placeholder.caption(retrieval_info_msg) # Show skipped message briefly

    # --- 6. Initiate LLM Response Generation ---
    # Create placeholders *before* the assistant message context
    # These will be passed *into* the chat_message context
    message_placeholder_container = st.empty()
    thinking_placeholder_container = st.empty()
    sources_placeholder_container = st.empty() # For the sources expander

    # Generate response within the assistant's chat message context
    with st.chat_message("assistant"):
        # Re-scope placeholders inside chat_message context for direct updates
        message_placeholder = st.empty()
        thinking_placeholder = st.empty()
        # sources_placeholder = st.empty() # Expander created later if needed

        with st.spinner("Assistant is thinking..."):
            # Pass the necessary placeholders to the generation function
            final_result = generate_llm_response(
                llm_manager=llm_manager,
                query_engine=query_engine,
                # Pass the history *including* the user message just added
                conversation_history=st.session_state.active_conversation_data["messages"],
                current_prompt=prompt,
                context_sources=sources,
                message_placeholder=message_placeholder,
                thinking_placeholder=thinking_placeholder
            )

    # --- Generation Complete ---
    final_answer = final_result.get("final_answer", "")
    final_thinking = final_result.get("final_thinking")
    error = final_result.get("error")

    # Clear the temporary status message placeholder now that generation is done
    retrieval_status_placeholder.empty()

    # Populate sources expander if needed (outside chat_message context)
    if not error and sources and CONFIG.get("conversation", {}).get("persist_retrieved_context", True):
         with sources_placeholder_container.expander("View Sources Used", expanded=False):
             for i, source in enumerate(sources):
                 # Ensure source is a dict before accessing keys
                 if isinstance(source, dict):
                     score = source.get('score', 0.0)
                     text = source.get('original_text', source.get('text', ''))
                     meta = source.get('metadata', {})
                     st.markdown(f"**Source {i + 1} (Score: {score:.2f}):**")
                     st.markdown(f"> {text}")
                     st.caption(f"Doc: {meta.get('file_name', 'N/A')} | Page: {meta.get('page_num', 'N/A')} | Chunk: {meta.get('chunk_id', 'N/A')}")
                     st.markdown("---")
                 else:
                      st.warning(f"Source {i+1} has unexpected format: {type(source)}")

    # --- 7. Add Final Assistant Message to Conversation History ---
    assistant_message_id = f"msg_asst_{len(active_conv_data['messages'])}" # ID relates to the turn
    assistant_message = {
        "role": "assistant",
        "content": final_answer if not error else f"Error generating response: {error}",
        "timestamp": time.time(),
        "id": assistant_message_id
    }
    # Persist context and thinking if available and configured
    persist_context = CONFIG.get("conversation", {}).get("persist_retrieved_context", True)
    if sources and not error and persist_context:
        # Store structured context used
        assistant_message["used_context"] = [
            {"text": s.get('original_text', s.get('text', '')),
             "metadata": s.get('metadata', {}),
             "score": s.get('score', 0.0),
             "source_index": i+1}
            for i, s in enumerate(sources) if isinstance(s, dict) # Ensure s is dict
        ]
    if final_thinking:
        assistant_message["thinking_process"] = final_thinking

    # Append the final assistant message to the *actual* session state data
    st.session_state.active_conversation_data["messages"].append(assistant_message)

    # --- 8. Update UI Display List ---
    # Add the complete assistant message info for rendering in the next cycle
    st.session_state.ui_chat_display.append({
        "role": "assistant",
        "content": assistant_message["content"],
        "thinking": final_thinking,
        "sources": assistant_message.get("used_context") # Use context stored in the message
    })

    # --- 9. Auto-Save Conversation ---
    if CONFIG.get("conversation", {}).get("auto_save_on_turn", True):
        save_current_conversation()

    logger.info("Finished handling chat message cycle.")

    # --- 10. Trigger Rerun ---
    # A rerun is needed to display the assistant's message, thinking, and sources expander correctly.
    # Streamlit's natural rerun after state updates might be sufficient, but an explicit
    # rerun ensures the UI reflects the final state immediately after processing.
    st.rerun()