# --- START OF MODIFIED aphrodite_service.py ---

"""
Aphrodite LLM service in a separate process.
Handles LLM operations in isolation with clean termination capability.
Uses a single loaded model instance, applying parameters dynamically.
(Non-streaming version)
"""
import sys
import os
from pathlib import Path
import yaml
import time
import json
import logging
import multiprocessing
# Set the multiprocessing start method to 'spawn' to avoid CUDA re-init issues
# Ensure this runs early, ideally at the main script entry point if possible
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # Might already be set, ignore
    pass
from multiprocessing import Process, Queue
import queue # Use 'queue' instead of multiprocessing.Queue for Empty exception
import torch
import signal
import traceback
from typing import Dict, Any, Optional, List, Tuple

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

# Import utils - don't import logger directly to avoid duplicate handlers
from src.utils.resource_monitor import log_memory_usage

# Set up logger manually for this module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aphrodite_service")

# Load configuration
CONFIG_PATH = ROOT_DIR / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

# Global variables for the worker process
global_llm = None
global_logits_processor = None # Will be created once if needed

def init_worker_logging():
    """Initialize logging for the worker process."""
    log_dir = ROOT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "aphrodite_worker.log")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    # Prevent duplicate handlers if re-initialized
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.addHandler(file_handler)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    root_logger.addHandler(console)
    logger.info("Worker process logging initialized")

def load_aphrodite_model(model_name: str, attempt=0):
    """Load an Aphrodite model generically and create JSON processor."""
    global global_llm, global_logits_processor
    try:
        from aphrodite import LLM

        max_model_len = CONFIG["aphrodite"]["max_model_len"]
        quantization = CONFIG["aphrodite"]["quantization"]
        logger.info(f"Loading Aphrodite model: {model_name}")
        logger.info(f"Parameters: max_model_len={max_model_len}, quantization={quantization}")

        try:
            # Attempt to load the LLM
            llm = LLM(
                model=model_name,
                max_model_len=max_model_len,
                quantization=quantization if quantization != "none" else None,
                dtype="bfloat16", # Consider making this configurable if needed
                gpu_memory_utilization=0.95, # Adjust if needed
                enforce_eager=True, # Often needed for stability/correctness
                enable_prefix_caching=True, # Generally good for performance
                trust_remote_code=True, # Be aware of security implications
                #max_num_seq=1024,
                #speculative_model="[ngram]",  # [!code highlight]
                #num_speculative_tokens=5,  # [!code highlight]
                #ngram_prompt_lookup_max=4,  # [!code highlight]
                #use_v2_block_manager=True,  # [!code highlight]
            )
        except RuntimeError as e:
            if "CUDA" in str(e) and attempt < 2:
                logger.warning(f"CUDA error during model load, retrying: {e}")
                time.sleep(2)
                return load_aphrodite_model(model_name, attempt + 1)
            else:
                raise # Re-raise if not CUDA error or too many attempts

        # Attempt to create JSONLogitsProcessor ONCE upon successful model load
        logits_processor = None
        try:
            from outlines.serve.vllm import JSONLogitsProcessor
            from pydantic import BaseModel # Define schemas here or import
            # These need to match the schema defined in entity_extractor.py
            class EntityRelationshipItem(BaseModel):
                from_entity_type: str
                from_entity_name: str
                relationship_type: Optional[str] = None
                to_entity_name: Optional[str] = None
                to_entity_type: Optional[str] = None
            class EntityRelationshipList(BaseModel):
                entity_relationship_list: List[EntityRelationshipItem]

            logger.info("Creating JSONLogitsProcessor for extraction")
            if hasattr(llm, 'llm_engine'):
                logits_processor = JSONLogitsProcessor(EntityRelationshipList, llm.llm_engine)
                logger.info("JSONLogitsProcessor created successfully.")
            else:
                logger.warning("Could not access llm.llm_engine for JSONLogitsProcessor. Extraction might fail.")
                logits_processor = None # Ensure it's None if creation failed
        except ImportError:
             logger.warning("Outlines library not installed. JSON forcing for extraction will be disabled.")
             logits_processor = None
        except Exception as e:
            logger.error(f"Failed to create JSONLogitsProcessor: {e}", exc_info=True)
            logits_processor = None # Ensure it's None if creation failed

        global_llm = llm
        global_logits_processor = logits_processor # Store it globally
        logger.info(f"Aphrodite model {model_name} loaded successfully")
        return llm # Return only the llm object
    except Exception as e:
        logger.error(f"Error loading Aphrodite model: {e}", exc_info=True)
        return None

def process_extraction_request(llm, prompts):
    """Process entity extraction request (non-streaming) with dynamic params."""
    global global_logits_processor # Access the globally created processor
    try:
        from aphrodite import SamplingParams
        logger.info(f"Processing extraction request with {len(prompts)} prompts")

        # Create SamplingParams dynamically for extraction
        extraction_params = SamplingParams(
            temperature=CONFIG["aphrodite"]["extraction_temperature"],
            max_tokens=CONFIG["aphrodite"]["extraction_max_new_tokens"],
            logits_processors=[global_logits_processor] if global_logits_processor else [] # Use global processor if available
        )

        if not global_logits_processor:
             logger.warning("Executing extraction without JSONLogitsProcessor. Output format may not be guaranteed.")

        # Use llm.generate for batch processing
        outputs = llm.generate(
            prompts=prompts,
            sampling_params=extraction_params,
            use_tqdm=True # Show progress bar for long batches
        )
        results = []
        for output in outputs:
            if output.outputs:
                results.append(output.outputs[0].text)
            else:
                # Return empty JSON structure on failure/no output
                logger.warning(f"Extraction request for prompt '{output.prompt[:50]}...' produced no output.")
                results.append("{\"entity_relationship_list\": []}")
        return {"status": "success", "results": results}
    except Exception as e:
        logger.error(f"Error processing extraction request: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}

def process_chat_request(llm, prompt):
    """Process chat generation request (non-streaming) with dynamic params."""
    try:
        from aphrodite import SamplingParams
        logger.info(f"Processing non-streaming chat request")

        # Create SamplingParams dynamically for chat
        chat_params = SamplingParams(
            temperature=CONFIG["aphrodite"]["chat_temperature"],
            max_tokens=CONFIG["aphrodite"]["chat_max_new_tokens"],
            top_p=CONFIG["aphrodite"]["top_p"]
            # No logits_processors for chat
        )

        # Use llm.generate to get the full response at once
        # llm.generate expects a list of prompts, even if just one
        outputs = llm.generate(
            prompts=[prompt], # Pass prompt as a list
            sampling_params=chat_params
        )
        if outputs and outputs[0].outputs:
            # Return the complete generated text
            return {"status": "success", "result": outputs[0].outputs[0].text}
        else:
            logger.warning("Chat generation produced no output.")
            return {"status": "error", "error": "No output generated"}
    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}

def aphrodite_worker(request_queue, response_queue):
    """Worker process function for Aphrodite."""
    # Signal handler for graceful shutdown
    def handle_terminate(signum, frame):
        logger.info(f"Received signal {signum}, shutting down worker")
        # No explicit Aphrodite cleanup needed here, process termination handles it
        sys.exit(0) # Use standard exit

    signal.signal(signal.SIGTERM, handle_terminate)
    signal.signal(signal.SIGINT, handle_terminate)

    init_worker_logging()
    logger.info(f"Aphrodite worker process started (PID: {os.getpid()})")

    # Access global variables defined outside
    global global_llm, global_logits_processor

    current_model_name = None # Only track the name

    while True:
        try:
            request = request_queue.get(timeout=1.0) # Use timeout to allow signal handling
        except queue.Empty:
            continue # No request, continue loop

        command = request.get("command", "")
        request_id = request.get("request_id", None) # Get request ID if provided

        if command == "exit":
            logger.info("Received exit command, shutting down worker.")
            response_queue.put({"status": "success", "message": "Worker exiting", "request_id": request_id})
            break

        try:
            if command == "load_model":
                model_name = request.get("model_name")
                # Load only if the requested model is different from the current one
                if current_model_name != model_name:
                    logger.info(f"Loading model: {model_name}")
                    # load_aphrodite_model now handles globals: global_llm, global_logits_processor
                    loaded_llm = load_aphrodite_model(model_name)
                    if loaded_llm:
                        current_model_name = model_name
                        response_queue.put({"status": "success", "model_name": model_name, "request_id": request_id})
                    else:
                        # Failed load, reset state
                        current_model_name = None
                        global_llm = None
                        global_logits_processor = None
                        response_queue.put({"status": "error", "error": f"Failed to load model {model_name}", "request_id": request_id})
                else:
                    logger.info(f"Model {model_name} already loaded.")
                    response_queue.put({"status": "success", "model_name": model_name, "already_loaded": True, "request_id": request_id})

            elif command == "extract_entities":
                if not global_llm: # Check if the global LLM object exists
                    response_queue.put({"status": "error", "error": "No model loaded", "request_id": request_id})
                    continue

                prompts = request.get("prompts", [])
                if not prompts:
                    response_queue.put({"status": "error", "error": "No prompts provided for extraction", "request_id": request_id})
                    continue

                # Pass the global llm object
                result = process_extraction_request(global_llm, prompts)
                result["request_id"] = request_id # Add request_id to response
                response_queue.put(result)

            elif command == "generate_chat":
                if not global_llm: # Check if the global LLM object exists
                    response_queue.put({"status": "error", "error": "No model loaded", "request_id": request_id})
                    continue

                prompt = request.get("prompt", "")
                if not prompt:
                    response_queue.put({"status": "error", "error": "No prompt provided for chat", "request_id": request_id})
                    continue

                # Pass the global llm object
                result = process_chat_request(global_llm, prompt)
                result["request_id"] = request_id # Add request_id to response
                response_queue.put(result)

            elif command == "status":
                status_info = {
                    "status": "success",
                    "pid": os.getpid(),
                    "model_loaded": global_llm is not None,
                    "current_model": current_model_name,
                    # "is_chat_model": is_chat_model_loaded, # Removed
                    "request_id": request_id
                }
                try:
                    if torch.cuda.is_available():
                        status_info["gpu_memory_allocated_gb"] = round(torch.cuda.memory_allocated() / (1024**3), 2)
                        status_info["gpu_memory_reserved_gb"] = round(torch.cuda.memory_reserved() / (1024**3), 2)
                except Exception as gpu_e:
                    status_info["gpu_error"] = str(gpu_e)
                response_queue.put(status_info)

            else:
                logger.warning(f"Unknown command received: {command}")
                response_queue.put({"status": "error", "error": f"Unknown command: {command}", "request_id": request_id})

        except Exception as e:
            logger.error(f"Error processing command '{command}': {e}", exc_info=True)
            try:
                response_queue.put({
                    "status": "error",
                    "error": f"Worker error processing command '{command}': {str(e)}",
                    "traceback": traceback.format_exc(),
                    "request_id": request_id
                })
            except Exception as q_e:
                logger.error(f"Failed to put error message onto response queue: {q_e}")

    logger.info("Worker process shutting down.")
    # No need for sys.exit(0) here, loop exit is sufficient

class AphroditeService:
    """Service class for managing Aphrodite in a separate process."""
    def __init__(self):
        self.process: Optional[Process] = None
        self.request_queue: Optional[Queue] = None
        self.response_queue: Optional[Queue] = None
        self.current_model_info: Dict[str, Any] = {"name": None} # Track only loaded model name
        self._request_counter = 0 # Simple counter for request IDs

    def _generate_request_id(self) -> str:
        """Generate a unique request ID."""
        self._request_counter += 1
        return f"req_{self._request_counter}_{int(time.time()*1000)}"

    def _send_request(self, command: str, data: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
        """Helper to send request and wait for response."""
        if not self.is_running() or self.request_queue is None or self.response_queue is None:
            # Attempt to start if not running, might be needed for first call
            logger.warning(f"Service not running when trying to send command: {command}. Attempting start...")
            if not self.start():
                 logger.error(f"Failed to start service. Cannot process command: {command}")
                 return {"status": "error", "error": "Service not running and failed to start"}
            # If start was successful, continue with the request
            logger.info("Service started successfully, proceeding with request.")


        request_id = self._generate_request_id()
        payload = {"command": command, "request_id": request_id, **data}

        try:
            logger.debug(f"Sending request {request_id}: {command}")
            self.request_queue.put(payload)
            # Wait for the specific response matching the request_id
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    response = self.response_queue.get(timeout=1.0) # Check queue periodically
                    if response.get("request_id") == request_id:
                        logger.debug(f"Received response for {request_id}")
                        return response
                    else:
                        # Put back responses for other requests (or log/handle)
                        logger.warning(f"Received unexpected response for request {response.get('request_id')}, expecting {request_id}. Discarding.")
                        # Avoid requeuing to prevent potential loops
                except queue.Empty:
                    continue # Continue waiting until timeout
            logger.error(f"Timeout waiting for response for request {request_id} ({command})")
            return {"status": "error", "error": f"Timeout waiting for response ({command})", "request_id": request_id}
        except Exception as e:
            logger.error(f"Error sending request {request_id} ({command}): {e}", exc_info=True)
            return {"status": "error", "error": f"IPC error: {str(e)}", "request_id": request_id}

    def get_process_info(self):
        """Get saved process/model info."""
        if not self.is_running():
            return None
        # Use the locally tracked info
        return {
            "pid": self.process.pid if self.process else None,
            "model_name": self.current_model_info["name"],
            # "is_chat_model": self.current_model_info["is_chat"] # Removed
        }

    def is_process_running(self, pid):
        """Check if a process with the given PID is running."""
        if pid is None: return False
        try:
            import psutil
            return psutil.pid_exists(pid)
        except ImportError:
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                return False
        except Exception: # Catch other potential errors like invalid PID format
             return False

    def is_running(self):
        """Check if the managed service process is running."""
        return self.process is not None and self.process.is_alive()

    def start(self, max_retries=1): # Reduce default retries
        """Start the Aphrodite service process."""
        if self.is_running():
            logger.info("Service already running.")
            return True
        for attempt in range(max_retries):
            try:
                logger.info(f"Starting Aphrodite service process (attempt {attempt+1}/{max_retries})...")
                # Use get_context("spawn") for robustness across platforms
                ctx = multiprocessing.get_context("spawn")
                self.request_queue = ctx.Queue()
                self.response_queue = ctx.Queue()
                self.process = ctx.Process(
                    target=aphrodite_worker,
                    args=(self.request_queue, self.response_queue),
                    daemon=True # Ensure it exits if main process crashes
                )
                self.process.start()
                logger.info(f"Aphrodite service process started (PID: {self.process.pid})")
                # Optionally wait for a 'ready' signal or check status immediately
                time.sleep(1) # Brief pause to allow process start
                status = self.get_status(timeout=15) # Quick status check
                if status.get("status") == "success":
                    logger.info("Service started and responded to status check.")
                    return True
                else:
                     logger.warning(f"Service process started but failed initial status check: {status.get('error', 'Unknown')}")
                     self.shutdown() # Clean up failed start
                     if attempt < max_retries - 1: time.sleep(2)
                     else: return False # Failed all attempts

            except Exception as e:
                logger.error(f"Critical error starting Aphrodite service (attempt {attempt+1}): {e}", exc_info=True)
                if self.process and self.process.is_alive(): self.process.terminate()
                self.process = None
                if attempt < max_retries - 1: time.sleep(2)
                else: return False # Failed all attempts
        return False

    def load_model(self, model_name):
        """Load a model in the service process (generically)."""
        if not self.is_running():
            logger.warning("Service not running, attempting to start...")
            if not self.start(): return False

        logger.info(f"Requesting model load: {model_name}")
        response = self._send_request(
            "load_model",
            {"model_name": model_name}, # No is_chat_model flag
            timeout=600 # Increase timeout significantly for model loading
        )
        if response.get("status") == "success":
            logger.info(f"Model {model_name} loaded successfully in service.")
            self.current_model_info = {"name": model_name} # Update local tracker
            return True
        else:
            logger.error(f"Failed to load model in service: {response.get('error')}")
            # Reset local tracker if load failed
            self.current_model_info = {"name": None}
            return False

    def extract_entities(self, prompts):
        """Extract entities (non-streaming). No need for model type check here."""
        # Worker process will handle applying correct parameters
        return self._send_request(
            "extract_entities",
            {"prompts": prompts},
            timeout=600 # Long timeout for potentially large batch extraction
        )

    def generate_chat(self, prompt):
        """Generate chat response (non-streaming). No need for model type check here."""
         # Worker process will handle applying correct parameters
        return self._send_request(
            "generate_chat",
            {"prompt": prompt},
            timeout=120 # Adjust timeout as needed for chat response generation
        )

    def get_status(self, timeout=10):
        """Get service status."""
        if not self.is_running():
            return {"status": "not_running", "model_loaded": False}
        # Ask the worker for its current status
        response = self._send_request("status", {}, timeout=timeout)
        # Update local tracker based on worker status if successful
        if response.get("status") == "success":
             self.current_model_info["name"] = response.get("current_model")
        elif response.get("status") != "error": # Don't clear model if IPC error etc.
             self.current_model_info["name"] = None # Worker might have failed to load
        return response


    def shutdown(self):
        """Shutdown the service process."""
        if not self.process:
            logger.info("Shutdown requested but no process found.")
            return True

        pid = self.process.pid if self.process.pid else "unknown"
        logger.info(f"Attempting graceful shutdown of process {pid}...")

        if self.process.is_alive():
            response = self._send_request("exit", {}, timeout=10) # Send exit command
            if response.get("status") != "success":
                 logger.warning(f"Worker did not acknowledge exit command cleanly: {response.get('error')}")

            # Give the process time to exit after command
            self.process.join(timeout=5)

            # Force terminate if still alive
            if self.process.is_alive():
                logger.warning(f"Process {pid} did not exit gracefully, terminating.")
                self.process.terminate()
                self.process.join(timeout=5) # Wait for termination

            if self.process.is_alive():
                 logger.error(f"Process {pid} failed to terminate, attempting kill.")
                 try:
                     # Use os.kill directly as process.kill() might not exist in spawn context always
                     os.kill(pid, signal.SIGKILL)
                     time.sleep(1) # Short pause after kill
                 except ProcessLookupError:
                      logger.info(f"Process {pid} already terminated before kill attempt.")
                 except Exception as kill_e:
                      logger.error(f"Failed to kill process {pid}: {kill_e}")
        else:
             logger.info(f"Process {pid} was already terminated before shutdown steps.")

        # Reset state
        self.process = None
        self.request_queue = None
        self.response_queue = None
        self.current_model_info = {"name": None}
        logger.info(f"Aphrodite service shutdown complete for PID {pid}.")
        return True

# Singleton instance
_service_instance = None
def get_service():
    """Get the singleton service instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = AphroditeService()
    return _service_instance

# --- END OF REWRITTEN FILE aphrodite_service.py ---