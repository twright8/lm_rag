# --- START OF NEW FILE src/utils/gemini_manager.py ---
"""
Google Gemini API manager for handling API calls, streaming, and structured output.
"""
import sys
import json
import time
import traceback
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Union, Type

# Add project root to path if necessary (adjust based on actual structure)
try:
    ROOT_DIR = Path(__file__).resolve().parent.parent.parent
    if str(ROOT_DIR) not in sys.path: # Avoid adding duplicates
        sys.path.append(str(ROOT_DIR))
    from src.utils.logger import setup_logger
    logger = setup_logger(__name__)
except ImportError:
    # Fallback if run standalone or path issues
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.warning("Could not import standard logger setup, using basic logging.")

# Attempt to import Google Generative AI library
try:
    from google import genai
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    logger.error("Google Generative AI library not found. Please install with: pip install google-generativeai")
    genai = None # Placeholder
    google_exceptions = None
    GOOGLE_GENAI_AVAILABLE = False

# Attempt to import Pydantic for schema definition
try:
    from pydantic import BaseModel, Field, create_model
    from pydantic_core import PydanticUndefined
    PYDANTIC_AVAILABLE = True
except ImportError:
    BaseModel = None
    Field = None
    create_model = None
    PydanticUndefined = None
    PYDANTIC_AVAILABLE = False
    logger.error("Pydantic not found. Structured output generation will fail.")


class GeminiManager:
    """Manager for Google Gemini API interactions."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Gemini manager with configuration.

        Args:
            config: Dictionary containing Gemini settings from config.yaml
        """
        logger.info("Initializing GeminiManager...")
        self.config = config
        self.api_key = config.get("api_key")
        # Use the specific model from the example for extraction tasks
        self.models = {
            "chat": config.get("chat_model", "gemini-1.5-flash-latest"),
            "extraction": config.get("extraction_model", "gemini-2.5-pro-exp-03-25"),
            "info_extraction": config.get("info_extraction_model", "gemini-2.5-pro-exp-03-25"),
            "classification": config.get("classification_model", "gemini-2.5-pro-exp-03-25"),
            "topic_labeling": config.get("topic_labeling_model", "gemini-1.5-flash-latest"),
        }
        self.default_temperature = config.get("temperature", 0.7)
        self.default_max_tokens = config.get("max_tokens", 2048)

        self.client = None
        if not GOOGLE_GENAI_AVAILABLE:
            logger.error("Google Generative AI library failed to import. GeminiManager cannot function.")
            return
        if not PYDANTIC_AVAILABLE:
            logger.error("Pydantic library failed to import. GeminiManager cannot generate structured output.")
            # Allow init but structured calls will fail later

        if not self.api_key:
            logger.error("Gemini API key is missing in configuration.")
            # Don't raise error here, allow instantiation but client will be None
            return
        else:
            masked_key = self.api_key[:7] + "..." + self.api_key[-4:] if len(self.api_key) > 11 else "Key present (short)"
            logger.info(f"Gemini API Key found (masked: {masked_key})")

        try:
            # Configure the client using the API key
            self.client = genai.Client(api_key=self.api_key)
            # Test connection by listing models (optional, but good practice)
            # self.client.models.list()
            logger.info("Gemini manager initialized. Google Generative AI client created successfully.")
        except Exception as e:
            logger.error(f"Error initializing Google Generative AI client for Gemini: {e}", exc_info=True)
            self.client = None # Ensure client is None on error

    def _get_model_for_task(self, task_name: str, model_override: Optional[str] = None) -> str:
        """Get the appropriate model name for a given task."""
        if model_override:
            return model_override
        # Use the specific model from the example for extraction tasks
        if task_name in ["extraction", "info_extraction", "classification"]:
            return self.models.get(task_name, "gemini-2.5-pro-exp-03-25")
        return self.models.get(task_name, self.models["chat"]) # Default to chat model

    def _handle_api_error(self, error: Exception, context: str = "API call") -> Dict[str, Any]:
        """Standardized error handling for Gemini API calls."""
        error_type = type(error).__name__
        error_message = str(error)
        logger.error(f"Gemini {context} error ({error_type}): {error_message}", exc_info=True)

        # Fallback for general errors
        return {"status": "error", "error": f"An unexpected error occurred: {error_message}"}

    def generate_chat(self, prompt: str, model_name: Optional[str] = None, stream_callback: Optional[Callable] = None, **kwargs) -> Dict[str, Any]:
        """
        Generate chat response using Gemini, supporting streaming.

        Args:
            prompt: The user's prompt string (including formatted history if needed).
            model_name: Optional override for the model name.
            stream_callback: Optional function to call with each received token chunk.
            **kwargs: Additional generation parameters (temperature, max_tokens).

        Returns:
            Dictionary with status and result (full response string).
        """
        if not self.client:
            logger.error("Cannot generate chat: Gemini client not initialized.")
            return {"status": "error", "error": "Gemini client not initialized."}

        selected_model = self._get_model_for_task("chat", model_name)
        temperature = kwargs.get("temperature", self.default_temperature)
        max_tokens = kwargs.get("max_tokens", self.default_max_tokens)

        # Prepare generation config for Gemini
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            # Add other relevant parameters like top_p, top_k if needed
        )

        # Gemini expects 'contents' as a list for streaming, single string for non-streaming
        # Based on user examples, we will use a list for streaming and single string for non-streaming.
        # The calling function (app_chat) needs to format history into the prompt string.
        contents_for_api = [prompt] if stream_callback else prompt

        logger.info(f"Generating chat with Gemini model: {selected_model} (Streaming: {bool(stream_callback)})")
        logger.debug(f"Prompt (start): {prompt[:200]}...")
        logger.debug(f"Params: temp={temperature}, max_tokens={max_tokens}")

        try:
            if stream_callback:
                # Streaming generation using generate_content_stream
                full_response = ""
                logger.debug("Attempting to create streaming chat completion...")

                logger.debug("Streaming chat completion request sent.")
                for chunk in self.client.models.generate_content_stream(
                    model=selected_model,
                    contents=contents_for_api, # Pass as list
                    config=generation_config
                ):
                    if hasattr(chunk, 'text') and chunk.text:
                        token = chunk.text
                        full_response += token
                        try:
                            stream_callback(token)
                        except Exception as cb_err:
                            logger.error(f"Error in stream_callback: {cb_err}", exc_info=True)
                            # Continue streaming even if callback fails
                    # Gemini stream chunks might contain other info (e.g., safety ratings)
                    # We only care about the text content here.
                logger.info(f"Gemini streaming finished. Response length: {len(full_response)}")
                return {"status": "success", "result": full_response}
            else:
                # Non-streaming generation using generate_content
                logger.debug("Attempting to create non-streaming chat completion...")
                response = self.client.models.generate_content(
                    model=selected_model,
                    contents=contents_for_api, # Pass as single string
                    config=generation_config
                )
                logger.debug("Non-streaming chat completion request successful.")
                content = response.text if hasattr(response, 'text') else ""
                logger.info(f"Gemini non-streaming finished. Response length: {len(content)}")
                return {"status": "success", "result": content.strip()}

        except Exception as e:
            logger.error(f"Exception during Gemini chat generation API call: {type(e).__name__}", exc_info=True)
            return self._handle_api_error(e, "chat generation")

    def extract_structured(self, prompt: str, pydantic_schema: Type[BaseModel], model_name: Optional[str] = None, task_name: str = "extraction", **kwargs) -> Dict[str, Any]:
        """
        Extract structured data using Gemini's JSON Schema mode via Pydantic.

        Args:
            prompt: The prompt string containing instructions and text to analyze.
            pydantic_schema: The Pydantic model class defining the desired output structure.
            model_name: Optional override for the model name.
            task_name: The type of task (e.g., 'extraction', 'info_extraction', 'classification') to select the correct model.
            **kwargs: Additional generation parameters (temperature, max_tokens).

        Returns:
            Dictionary with status and parsed JSON result or error.
        """
        if not self.client:
            logger.error("Cannot extract structured data: Gemini client not initialized.")
            return {"status": "error", "error": "Gemini client not initialized."}
        if not PYDANTIC_AVAILABLE or pydantic_schema is None:
             logger.error("Cannot extract structured data: Pydantic or schema is not available.")
             return {"status": "error", "error": "Pydantic schema required for structured extraction."}

        # Use the specific model from the example for extraction tasks
        selected_model = self._get_model_for_task(task_name, model_name)
        # Use lower temperature for structured tasks
        temperature = kwargs.get("temperature", 0.1)
        max_tokens = kwargs.get("max_tokens", self.default_max_tokens)

        # Prepare generation config for Gemini, including structured output config
        # Adhere strictly to the user's example structure
        if not "2.5-flash" in selected_model:
            generation_config = genai.types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                response_mime_type='application/json',
                response_schema=pydantic_schema,
            )
        else:#
            print("NONONONO THINKY")
            generation_config = genai.types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                response_mime_type='application/json',
                response_schema=pydantic_schema,
                thinking_config=genai.types.ThinkingConfig(
                    thinking_budget=0,
                ),
            )

        logger.info(f"Extracting structured data with Gemini model: {selected_model}")
        logger.debug(f"Prompt (start): {prompt[:200]}...")
        logger.debug(f"Pydantic Schema: {pydantic_schema.__name__}")
        logger.debug(f"Params: temp={temperature}, max_tokens={max_tokens}")

        try:
            logger.debug(f"Attempting to call Gemini generate_content for structured extraction (model: {selected_model})...")
            response = self.client.models.generate_content(
                model=selected_model,
                contents=prompt, # Pass single string as per example
                config=generation_config,
            )
            logger.debug("Gemini generate_content call successful for structured extraction.")

            # Extract the JSON content string
            if hasattr(response, 'text') and response.text:
                json_string = response.text
                logger.debug(f"Raw JSON string from Gemini: {json_string}")

                try:
                    # Parse the JSON string
                    parsed_json = json.loads(json_string)
                    logger.info("Successfully parsed structured JSON response from Gemini.")
                    # The Gemini API should have already validated against the schema
                    return {"status": "success", "result": parsed_json}

                except json.JSONDecodeError as json_err:
                    logger.error(f"Failed to parse JSON response from Gemini: {json_err}")
                    logger.error(f"JSON string that failed: {json_string}")
                    return {"status": "error", "error": f"Model returned invalid JSON content: {json_err}", "raw_output": json_string}
            else:
                logger.error("Gemini response did not contain expected text content.")
                try:
                    raw_resp_str = str(response) # Attempt basic string conversion
                    logger.error(f"Raw response object received: {raw_resp_str}")
                except Exception as log_err:
                    logger.error(f"Could not serialize raw response object for logging: {log_err}")
                return {"status": "error", "error": "Model response was empty or malformed."}

        except Exception as e:
            logger.error(f"Exception during Gemini structured extraction API call: {type(e).__name__}", exc_info=True)
            # Check if the error is related to schema incompatibility
            if "response_schema" in str(e).lower() or "unsupported" in str(e).lower():
                 logger.error(f"Model {selected_model} might not support the provided schema or structured output mode.")
                 return {"status": "error", "error": f"Model {selected_model} failed with the provided schema."}
            return self._handle_api_error(e, f"{task_name} structured extraction")

# --- END OF NEW FILE src/utils/gemini_manager.py ---