# --- START OF NEW FILE src/utils/openrouter_manager.py ---

"""
OpenRouter API manager for handling API calls, streaming, and structured output.
Uses the official OpenAI Python library.
"""
import sys
import json
import time
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Union, Type, Literal

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

# Attempt to import OpenAI library
try:
    from openai import OpenAI, APIError, RateLimitError, APIConnectionError, APITimeoutError
except ImportError:
    logger.error("OpenAI library not found. Please install with: pip install openai")
    OpenAI = None # Placeholder
    APIError = RateLimitError = APIConnectionError = APITimeoutError = Exception # Fallback exceptions

# Attempt to import Pydantic for schema conversion helper
try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except ImportError:
    BaseModel = None
    PYDANTIC_AVAILABLE = False
    logger.warning("Pydantic not found. Schema conversion from Pydantic models will not work.")


class OpenRouterManager:
    """Manager for OpenRouter API interactions."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the OpenRouter manager with configuration.

        Args:
            config: Dictionary containing OpenRouter settings from config.yaml
                    (api_key, base_url, models, temperature, max_tokens, etc.)
        """
        logger.info("Initializing OpenRouterManager...") # Added log
        self.config = config
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url", "https://openrouter.ai/api/v1")
        self.models = {
            "chat": config.get("chat_model", "mistralai/mistral-7b-instruct:free"),
            "extraction": config.get("extraction_model", "mistralai/mistral-7b-instruct:free"),
            "info_extraction": config.get("info_extraction_model", "mistralai/mistral-7b-instruct:free"),
            "classification": config.get("classification_model", "mistralai/mistral-7b-instruct:free"),
            "topic_labeling": config.get("topic_labeling_model", "mistralai/mistral-7b-instruct:free"),
        }
        self.default_temperature = config.get("temperature", 0.7)
        # --- ADJUSTED DEFAULT MAX TOKENS ---
        # Read from config, but maybe use a smaller default if not set
        self.default_max_tokens = config.get("max_tokens", 2048) # Use a more reasonable default
        logger.info(f"Default max_tokens set to: {self.default_max_tokens}")
        # --- END ADJUSTMENT ---
        self.site_url = config.get("site_url")
        self.site_title = config.get("site_title")

        self.client = None
        if OpenAI is None:
            logger.error("OpenAI library failed to import. OpenRouterManager cannot function.")
            return

        # --- Added Logging ---
        if not self.api_key:
            logger.error("OpenRouter API key is missing in configuration.")
            # Don't raise error here, allow instantiation but client will be None
            return
        else:
            # Log partial key for verification without exposing the full key
            masked_key = self.api_key[:7] + "..." + self.api_key[-4:] if len(self.api_key) > 11 else "Key present (short)"
            logger.info(f"OpenRouter API Key found (masked: {masked_key})")
        # --- End Added Logging ---

        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            # --- Added Logging ---
            if self.client:
                 logger.info(f"OpenRouter manager initialized. Base URL: {self.base_url}. OpenAI client created successfully.")
            else:
                 # This case might not happen if OpenAI() raises an error instead of returning None
                 logger.error("OpenAI client initialization returned None unexpectedly.")
            # --- End Added Logging ---
            # Optionally test connection here if desired
            # self.client.models.list()
        except Exception as e:
            logger.error(f"Error initializing OpenAI client for OpenRouter: {e}", exc_info=True)
            self.client = None # Ensure client is None on error

    def _prepare_headers(self) -> Dict[str, str]:
        """Prepare optional headers for OpenRouter API calls."""
        headers = {}
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_title:
            headers["X-Title"] = self.site_title
        return headers

    def _get_model_for_task(self, task_name: str, model_override: Optional[str] = None) -> str:
        """Get the appropriate model name for a given task."""
        if model_override:
            return model_override
        return self.models.get(task_name, self.models["chat"]) # Default to chat model

    def _handle_api_error(self, error: Exception, context: str = "API call") -> Dict[str, Any]:
        """Standardized error handling for API calls."""
        error_type = type(error).__name__
        error_message = str(error)
        logger.error(f"OpenRouter {context} error ({error_type}): {error_message}", exc_info=True)
        # Provide more specific messages for common errors
        if isinstance(error, RateLimitError):
            return {"status": "error", "error": "API rate limit exceeded. Please try again later."}
        elif isinstance(error, APIConnectionError):
            return {"status": "error", "error": "Network error connecting to OpenRouter API."}
        elif isinstance(error, APITimeoutError):
            return {"status": "error", "error": "Request to OpenRouter API timed out."}
        elif isinstance(error, APIError): # General API error (includes auth, not found etc.)
             # Check for authentication error specifically if possible (depends on OpenAI lib version)
             if "authentication" in error_message.lower() or getattr(error, 'status_code', 0) == 401:
                 return {"status": "error", "error": "OpenRouter authentication failed. Check your API key."}
             # --- ADDED: Check for context length errors ---
             elif getattr(error, 'status_code', 0) == 400 and ("context_length" in error_message.lower() or "token limit" in error_message.lower()):
                  logger.error(f"Context length exceeded for model. Error: {error_message}")
                  return {"status": "error", "error": "Input prompt plus requested output tokens exceed model's context limit."}
             # --- END ADDED ---
             return {"status": "error", "error": f"OpenRouter API error: {error_message}"}
        else:
            return {"status": "error", "error": f"An unexpected error occurred: {error_message}"}

    def generate_chat(self, messages: List[Dict[str, str]], stream_callback: Optional[Callable] = None, model_name: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Generate chat response using OpenRouter, supporting streaming.

        Args:
            messages: List of message dictionaries (e.g., [{"role": "user", "content": "..."}]).
            stream_callback: Optional function to call with each received token chunk.
            model_name: Optional override for the model name.
            **kwargs: Additional generation parameters (temperature, max_tokens).

        Returns:
            Dictionary with status and result (full response string).
        """
        if not self.client:
            logger.error("Cannot generate chat: OpenRouter client not initialized.") # Added log
            return {"status": "error", "error": "OpenRouter client not initialized."}

        selected_model = self._get_model_for_task("chat", model_name)
        temperature = kwargs.get("temperature", self.default_temperature)
        # --- Use default_max_tokens directly ---
        max_tokens = kwargs.get("max_tokens", self.default_max_tokens)
        # --- End Use default_max_tokens directly ---
        extra_headers = self._prepare_headers()

        logger.info(f"Generating chat with OpenRouter model: {selected_model} (Streaming: {bool(stream_callback)})")
        logger.debug(f"Messages: {messages}")
        logger.debug(f"Params: temp={temperature}, max_tokens={max_tokens}")

        try:
            if stream_callback:
                # Streaming generation
                full_response = ""
                logger.debug("Attempting to create streaming chat completion...") # Added log
                stream = self.client.chat.completions.create(
                    model=selected_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                    extra_headers=extra_headers
                )
                logger.debug("Streaming chat completion request sent.") # Added log
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content is not None:
                        token = chunk.choices[0].delta.content
                        full_response += token
                        try:
                            stream_callback(token)
                        except Exception as cb_err:
                            logger.error(f"Error in stream_callback: {cb_err}", exc_info=True)
                            # Continue streaming even if callback fails
                logger.info(f"OpenRouter streaming finished. Response length: {len(full_response)}")
                return {"status": "success", "result": full_response}
            else:
                # Non-streaming generation
                logger.debug("Attempting to create non-streaming chat completion...") # Added log
                response = self.client.chat.completions.create(
                    model=selected_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False,
                    extra_headers=extra_headers
                )
                logger.debug("Non-streaming chat completion request successful.") # Added log
                content = response.choices[0].message.content
                logger.info(f"OpenRouter non-streaming finished. Response length: {len(content)}")
                return {"status": "success", "result": content.strip()}

        except Exception as e:
            # --- Enhanced Logging ---
            logger.error(f"Exception during OpenRouter chat generation API call: {type(e).__name__}", exc_info=True)
            # --- End Enhanced Logging ---
            return self._handle_api_error(e, "chat generation")

    def extract_structured(self, messages: List[Dict[str, str]], json_schema: Dict, model_name: Optional[str] = None, task_name: str = "extraction", **kwargs) -> Dict[str, Any]:
        """
        Extract structured data using OpenRouter's JSON Schema mode.

        Args:
            messages: List of message dictionaries.
            json_schema: The JSON schema definition for the desired output.
            model_name: Optional override for the model name.
            task_name: The type of task (e.g., 'extraction', 'info_extraction', 'classification') to select the correct model.
            **kwargs: Additional generation parameters (temperature, max_tokens).

        Returns:
            Dictionary with status and parsed JSON result or error.
        """

        logger.info(f"Entering extract_structured for task: {task_name}")  # Added log
        if not self.client:
            logger.error("Cannot extract structured data: OpenRouter client not initialized.") # Added log
            return {"status": "error", "error": "OpenRouter client not initialized."}

        selected_model = self._get_model_for_task(task_name, model_name)
        # Use lower temperature for structured tasks by default
        temperature = kwargs.get("temperature", 0.1)
        # --- Use default_max_tokens directly, don't double it here ---
        # Let the config value be the intended max output for structured tasks too
        max_tokens = kwargs.get("max_tokens", self.default_max_tokens)
        logger.info(f"Using max_tokens for output: {max_tokens}") # Log the actual value used
        # --- End Use default_max_tokens directly ---
        extra_headers = self._prepare_headers()

        # Prepare response_format for JSON Schema mode
        # --- MODIFIED: Turn strict OFF based on user feedback ---
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": f"{task_name}_schema", # Name for the schema
                "strict": True, # Set strict to False
                "schema": json_schema # The actual schema definition
            }
        }
        # --- END MODIFIED ---

        logger.info(f"Extracting structured data with OpenRouter model: {selected_model}")
        logger.debug(f"Messages: {messages}")
        logger.debug(f"JSON Schema: {json.dumps(json_schema, indent=2)}")
        logger.debug(f"Params: temp={temperature}, max_tokens={max_tokens}")

        try:
            # --- Added Logging ---
            print(messages)
            print(response_format)
            logger.debug(f"Attempting to call OpenRouter chat completions create for structured extraction (model: {selected_model})...")
            # --- End Added Logging ---
            response = self.client.chat.completions.create(
                model=selected_model,
                messages=messages,
                temperature=0.1,
                max_tokens=max_tokens,
                response_format=response_format,
                stream=False, # Structured output typically non-streaming
            )
            # --- Added Logging ---
            print(response.choices[0].message.content)
            logger.debug("OpenRouter chat completions create call successful for structured extraction.")
            # --- End Added Logging ---

            # Extract the JSON content
            print(response.choices[0].message)
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                json_string = response.choices[0].message.content
                logger.debug(f"Raw string from OpenRouter: {json_string}") # Keep this log

                # --- CLEANING STEP (from previous fix) ---
                cleaned_json_string = json_string.strip() # Remove leading/trailing whitespace
                if cleaned_json_string.startswith("```json"):
                    cleaned_json_string = cleaned_json_string[7:].lstrip()
                elif cleaned_json_string.startswith("```"):
                     cleaned_json_string = cleaned_json_string[3:].lstrip()

                if cleaned_json_string.endswith("```"):
                    cleaned_json_string = cleaned_json_string[:-3].rstrip()
                cleaned_json_string = cleaned_json_string.strip()
                logger.debug(f"Cleaned string before parsing: {cleaned_json_string}")
                # --- END CLEANING STEP ---

                try:
                    # --- PARSE CLEANED STRING ---
                    parsed_json = json.loads(cleaned_json_string)
                    # --- END PARSE CLEANED STRING ---

                    # --- REVISED NESTED PARSING ---
                    logger.debug("Checking for nested JSON strings or placeholders...")
                    fully_parsed = True # Flag to track if parsing succeeded

                    # Check and parse entity_relationship_list items
                    entity_list = parsed_json.get('entity_relationship_list')
                    if isinstance(entity_list, list):
                        logger.debug(f"Processing 'entity_relationship_list' with {len(entity_list)} items.")
                        parsed_list = []
                        items_skipped = 0
                        for i, item_val in enumerate(entity_list):
                            if isinstance(item_val, str):
                                # Check if it's a placeholder like 'string' or actual JSON
                                item_str_cleaned = item_val.strip()
                                if item_str_cleaned.startswith('{') or item_str_cleaned.startswith('['):
                                    # Looks like JSON, try parsing
                                    try:
                                        parsed_item = json.loads(item_str_cleaned)
                                        parsed_list.append(parsed_item)
                                    except json.JSONDecodeError as inner_err:
                                        logger.error(f"Failed to parse nested JSON string in entity_relationship_list at index {i}: {inner_err}. String: '{item_val}'")
                                        fully_parsed = False
                                        items_skipped += 1
                                else:
                                    # It's a string but doesn't look like JSON (e.g., 'string') - skip it
                                    logger.warning(f"Skipping non-JSON string placeholder '{item_val}' in entity_relationship_list at index {i}.")
                                    items_skipped += 1
                            elif isinstance(item_val, dict):
                                # Already a dictionary, add it directly
                                parsed_list.append(item_val)
                            elif item_val is None:
                                # Skip None values silently or log if needed
                                logger.debug(f"Skipping None value in entity_relationship_list at index {i}.")
                                items_skipped += 1
                            else:
                                # Unexpected type
                                logger.warning(f"Skipping unexpected type '{type(item_val)}' in entity_relationship_list at index {i}.")
                                items_skipped += 1

                        # Replace the original list with the newly parsed/filtered list
                        parsed_json['entity_relationship_list'] = parsed_list
                        if items_skipped > 0:
                             logger.info(f"Skipped {items_skipped} invalid/placeholder items in 'entity_relationship_list'. Kept {len(parsed_list)}.")
                        if not fully_parsed:
                             logger.warning("Partial failure parsing nested JSON strings in 'entity_relationship_list'.")

                    # Check and parse metadata
                    metadata_val = parsed_json.get('metadata')
                    if isinstance(metadata_val, str):
                        metadata_str_cleaned = metadata_val.strip()
                        if metadata_str_cleaned.startswith('{') or metadata_str_cleaned.startswith('['):
                            logger.info("Detected JSON string in 'metadata'. Parsing...")
                            try:
                                parsed_metadata = json.loads(metadata_str_cleaned)
                                parsed_json['metadata'] = parsed_metadata
                                logger.info("Successfully parsed nested string in 'metadata'.")
                            except json.JSONDecodeError as meta_err:
                                logger.error(f"Failed to parse nested JSON string in metadata: {meta_err}. String: '{metadata_val}'")
                                fully_parsed = False
                                parsed_json['metadata'] = None # Set to None on failure
                        else:
                            # It's a string but doesn't look like JSON
                            logger.warning(f"Metadata value is a non-JSON string placeholder: '{metadata_val}'. Setting metadata to None.")
                            parsed_json['metadata'] = None
                    elif metadata_val is None:
                         logger.debug("Metadata field is None.")
                    elif not isinstance(metadata_val, dict):
                         logger.warning(f"Metadata field has unexpected type '{type(metadata_val)}'. Setting to None.")
                         parsed_json['metadata'] = None


                    if not fully_parsed:
                         # If any inner parsing failed, return an error status but include partially parsed data
                         logger.error("Returning error status due to partial failure in parsing nested JSON.")
                         return {"status": "error", "error": "Failed to fully parse nested JSON content from model.", "result": parsed_json, "raw_output": json_string}

                    # --- END REVISED NESTED PARSING ---

                    logger.info("Successfully parsed structured JSON response (including nested).")
                    return {"status": "success", "result": parsed_json} # Return the fully parsed structure

                except json.JSONDecodeError as json_err:
                    # --- MODIFIED LOGGING ---
                    logger.error(f"Failed to parse JSON response from OpenRouter (even after cleaning): {json_err}")
                    logger.error(f"Cleaned JSON string that failed: {cleaned_json_string}") # Log the cleaned string
                    logger.error(f"Original raw string: {json_string}") # Log original for comparison
                    # Return error with original raw output for context
                    return {"status": "error", "error": f"Model returned invalid JSON content: {json_err}", "raw_output": json_string}
                    # --- END MODIFIED LOGGING ---
            else:
                # --- ADDED LOGGING for empty/malformed ---
                logger.error("OpenRouter response did not contain expected content structure (choices/message/content).")
                try:
                    # Log the raw response if possible to see the structure
                    raw_resp_str = response.model_dump_json(indent=2) if hasattr(response, 'model_dump_json') else str(response)
                    logger.error(f"Raw response object received: {raw_resp_str}")
                except Exception as log_err:
                    logger.error(f"Could not serialize raw response object for logging: {log_err}")
                # --- END ADDED LOGGING ---
                return {"status": "error", "error": "Model response was empty or malformed."}

        except Exception as e:
            # --- Enhanced Logging ---
            logger.error(f"Exception during OpenRouter structured extraction API call: {type(e).__name__}", exc_info=True)
            # --- End Enhanced Logging ---
            # Check if the error is due to the model not supporting response_format
            # Note: This check might be less relevant now strict=False
            if "response_format is not supported" in str(e):
                 logger.error(f"Model {selected_model} does not support JSON Schema mode (response_format).")
                 return {"status": "error", "error": f"Model {selected_model} does not support structured JSON output."}
            # Use the updated _handle_api_error which now checks for context length errors
            return self._handle_api_error(e, f"{task_name} structured extraction")

    # --- Schema Conversion Helpers ---

    def _pydantic_to_json_schema(self, pydantic_model: Type[BaseModel]) -> Optional[Dict[str, Any]]:
        """
        Convert a Pydantic model to a JSON Schema dictionary.

        Args:
            pydantic_model: The Pydantic model class.

        Returns:
            JSON Schema dictionary or None if Pydantic is not available or conversion fails.
        """
        if not PYDANTIC_AVAILABLE or pydantic_model is None:
            logger.error("Pydantic not available or model is None, cannot convert to JSON Schema.")
            return None
        try:
            # Use Pydantic's built-in method
            schema = pydantic_model.model_json_schema()
            # Remove title if present, as it might not be needed/supported everywhere
            schema.pop("title", None)
            logger.debug(f"Converted Pydantic model {pydantic_model.__name__} to JSON Schema.")
            return schema
        except Exception as e:
            logger.error(f"Error converting Pydantic model {pydantic_model.__name__} to JSON Schema: {e}", exc_info=True)
            return None

    def _dict_schema_to_json_schema(self, dict_schema: Dict[str, Dict], multi_label_fields: Optional[set] = None, required_fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Convert internal dictionary-based schema (used by InfoExtractor, DocumentClassifier)
        to the JSON Schema format required by OpenRouter.

        Args:
            dict_schema: The input schema definition (e.g., {"field_name": {"type": "string", "values": [], "description": ""}}).
            multi_label_fields: Set of field names that should be treated as arrays (for classification).
            required_fields: Optional list of field names that should be marked as required. If None, all fields are required.

        Returns:
            JSON Schema dictionary.
        """
        properties = {}
        multi_label = multi_label_fields or set()

        type_mapping = {
            "string": "string",
            "number": "number",
            "integer": "integer",
            "boolean": "boolean",
            "date": "string", # Represent dates as strings in JSON schema
            # Arrays/Objects might need more complex handling if nested schemas are involved
        }

        for field_name, field_info in dict_schema.items():
            json_type = type_mapping.get(field_info.get("type", "string").lower(), "string")
            prop_schema = {"type": json_type}

            if "description" in field_info and field_info["description"]:
                prop_schema["description"] = field_info["description"]

            allowed_values = field_info.get("values") # Used for classification enums

            if field_name in multi_label:
                # Multi-label field: becomes an array of enums
                prop_schema["type"] = "array"
                item_schema = {"type": json_type} # Base type of items in array
                if allowed_values:
                    item_schema["enum"] = allowed_values
                prop_schema["items"] = item_schema
            elif allowed_values:
                # Single-label field with specific values: becomes an enum
                prop_schema["enum"] = allowed_values

            properties[field_name] = prop_schema

        # Determine required fields
        final_required = required_fields if required_fields is not None else list(dict_schema.keys())

        json_schema_output = {
            "type": "object",
            "properties": properties,
            "required": final_required,
            # "additionalProperties": False # Consider if strictness is needed
        }
        logger.debug("Converted dictionary schema to JSON Schema.")
        return json_schema_output

# --- END OF MODIFIED FILE src/utils/openrouter_manager.py ---