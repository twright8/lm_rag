
"""
Entity and relationship extraction module for Anti-Corruption RAG System.
Uses the active LLM backend (Aphrodite, OpenRouter, or Gemini) for extraction.
Enhanced with metadata extraction, advanced deduplication, and entity type detection.
"""
import sys
import os
from pathlib import Path
import uuid
import yaml
import json
import torch
import gc
import time
import io
import re
from typing import List, Dict, Any, Optional, Tuple, Set, Literal, Type
from pydantic import BaseModel, Field
import pandas as pd
import recordlinkage
from recordlinkage.index import SortedNeighbourhood
import jellyfish
import networkx as nx
from thefuzz import fuzz

# Add project root to path
# Assuming this file is in src/core/extraction/
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path: # Avoid adding duplicates
    sys.path.append(str(ROOT_DIR))

# Import utils
from src.utils.logger import setup_logger
from src.utils.resource_monitor import log_memory_usage
# Import backend managers conditionally based on config
# Ensure app_setup is importable or handle potential ImportError
try:
    # Import backend flags and manager getter
    from src.ui.app_setup import (
        get_active_llm_manager, IS_OPENROUTER_ACTIVE, IS_GEMINI_ACTIVE, LLM_BACKEND
    )
except ImportError as setup_err:
     # Fallback logging if app_setup fails
     import logging
     logging.basicConfig(level=logging.INFO)
     fallback_logger = logging.getLogger(__name__)
     fallback_logger.error(f"Failed to import from src.ui.app_setup: {setup_err}. Check paths and dependencies.")
     # Define dummy values to allow script to load but fail later
     def get_active_llm_manager(): return None
     IS_OPENROUTER_ACTIVE = False
     IS_GEMINI_ACTIVE = False # Add Gemini flag
     LLM_BACKEND = "UNKNOWN"

# Import specific manager classes conditionally
if IS_OPENROUTER_ACTIVE:
    try: from src.utils.openrouter_manager import OpenRouterManager
    except ImportError: OpenRouterManager = None
    AphroditeService = None
    GeminiManager = None # Add Gemini placeholder
elif IS_GEMINI_ACTIVE: # Add Gemini case
    try: from src.utils.gemini_manager import GeminiManager
    except ImportError: GeminiManager = None
    AphroditeService = None
    OpenRouterManager = None
else: # Aphrodite is default or fallback
    try: from src.utils.aphrodite_service import AphroditeService
    except ImportError: AphroditeService = None
    OpenRouterManager = None
    GeminiManager = None

# Initialize logger
logger = setup_logger(__name__)

# Load configuration
CONFIG_PATH = ROOT_DIR / "config.yaml"
try:
    with open(CONFIG_PATH, "r") as f:
        CONFIG = yaml.safe_load(f)
except FileNotFoundError:
    logger.error(f"Configuration file not found at {CONFIG_PATH}. Using default settings.")
    CONFIG = {}
except Exception as e:
    logger.error(f"Error loading configuration: {e}. Using default settings.")
    CONFIG = {}

# --- Deduplication Configuration and Helpers (Unchanged) ---
# Default configuration (can be overridden by config.yaml)
DEFAULT_DEDUPLICATION_CONFIG = {
    "normalization_rules": {
        "PERSON": {
            "lowercase": True,
            "remove_titles": ["mr", "mrs", "ms", "miss", "dr", "prof", "sir", "madam", "lord", "lady", "rev", "hon", "president", "governor", "mayor"],
            "remove_suffixes": ["jr", "sr", "i", "ii", "iii", "iv", "v", "md", "phd", "esq", "dds"],
            "remove_punctuation": r"[.,'\"`’]", # Remove common punctuation
            "normalize_hyphens": True, # Replace hyphens with space
            "strip_whitespace": True,
        },
        "ORGANIZATION": { # Includes NGO, GOV, COMMERCIAL
            "lowercase": True,
            "remove_legal_suffixes": [
                "ltd", "inc", "corp", "llc", "plc", "gmbh", "ag", "bv", "spa", "sarl", "sas", "pte", "co", "corp",
                "limited", "incorporated", "corporation", "company", "associates", "foundation", "trust", "partners",
                "group", "holding", "holdings", "bank", "consulting", "services", "solutions", "international",
                 "global", "ventures", "capital", "industries", "systems", "technologies", "enterprises", "trading"
                ],
            "remove_punctuation": r"[.,'&()\"`’]", # Punctuation commonly found in org names
            "strip_whitespace": True,
        },
         "LOCATION": {
            "lowercase": True,
            "remove_punctuation": r"[.,'\"`’]",
            "strip_whitespace": True,
        },
        "POSITION": {
            "lowercase": True,
            "remove_punctuation": r"[.,'\"`’]",
            "strip_whitespace": True,
        },
        "DEFAULT": { # Fallback for other types or if specific type rules are missing
            "lowercase": True,
            "remove_punctuation": r"[.,'\"`’]",
            "strip_whitespace": True,
        }
    },
    "similarity_thresholds": {
        "PERSON": {
            "token_set_ratio": 88,
            "jaro_winkler": 0.85,
            "initial_match_bonus": True,
            "blocking_window": 7
        },
        "ORGANIZATION": { # Shared thresholds for NGO, GOV, COMMERCIAL
            "token_set_ratio": 92, # Slightly lower to catch variations
            "jaro_winkler": 0.88,
            "blocking_window": 5
        },
        "LOCATION": {
            "token_set_ratio": 95,
            "jaro_winkler": 0.92,
            "blocking_window": 3
        },
         "POSITION": {
            "token_set_ratio": 90,
            "jaro_winkler": 0.88,
            "blocking_window": 5
        },
        "DEFAULT": { # Fallback
            "token_set_ratio": 90,
            "jaro_winkler": 0.85,
            "blocking_window": 5
        }
    }
}
DEDUPLICATION_CONFIG = CONFIG.get("deduplication", DEFAULT_DEDUPLICATION_CONFIG)
def normalize_name(name: str, entity_type: str) -> str:
    """Applies normalization rules based on entity type using DEDUPLICATION_CONFIG."""
    if not isinstance(name, str) or not name: return ""
    config_rules = DEDUPLICATION_CONFIG["normalization_rules"]
    rules = config_rules.get(entity_type)
    if not rules:
        if entity_type in ["NGO", "GOVERNMENT_BODY", "COMMERCIAL_COMPANY"]: rules = config_rules.get("ORGANIZATION", config_rules["DEFAULT"])
        else: rules = config_rules["DEFAULT"]
    normalized_name = name
    if rules.get("lowercase"): normalized_name = normalized_name.lower()
    if 'remove_titles' in rules:
        titles_pattern = r'\b(?:' + '|'.join(re.escape(title) for title in rules['remove_titles']) + r')\.?\b'
        normalized_name = re.sub(titles_pattern, '', normalized_name, flags=re.IGNORECASE).strip()
    suffixes_to_remove = rules.get('remove_suffixes', []) + rules.get('remove_legal_suffixes', [])
    if suffixes_to_remove:
        suffixes_pattern = r'(?:[,\s]+)?(?:' + '|'.join(re.escape(suffix) for suffix in suffixes_to_remove) + r')\.?\b$'
        normalized_name = re.sub(suffixes_pattern, '', normalized_name, flags=re.IGNORECASE).strip()
    if rules.get("normalize_hyphens"): normalized_name = normalized_name.replace('-', ' ')
    if 'remove_punctuation' in rules and rules['remove_punctuation']: normalized_name = re.sub(rules['remove_punctuation'], '', normalized_name)
    if rules.get("strip_whitespace"): normalized_name = ' '.join(normalized_name.split())
    return normalized_name
def check_initial_match(name1: str, name2: str) -> bool:
    """Checks if one name is like 'J. Smith' and the other 'John Smith'."""
    if not name1 or not name2: return False
    parts1 = name1.split(); parts2 = name2.split()
    if len(parts1) < 2 or len(parts2) < 2: return False
    if len(parts1[0]) <= 2 and parts1[0].endswith('.') and len(parts2[0]) > 1 and parts1[-1] == parts2[-1]: return parts1[0][0] == parts2[0][0]
    if len(parts1[0]) == 1 and not parts1[0].endswith('.') and len(parts2[0]) > 1 and parts1[-1] == parts2[-1]: return parts1[0][0] == parts2[0][0]
    if len(parts2[0]) <= 2 and parts2[0].endswith('.') and len(parts1[0]) > 1 and parts1[-1] == parts2[-1]: return parts2[0][0] == parts1[0][0]
    if len(parts2[0]) == 1 and not parts2[0].endswith('.') and len(parts1[0]) > 1 and parts1[-1] == parts2[-1]: return parts2[0][0] == parts1[0][0]
    return False
def compare_entities(entity1: Dict, entity2: Dict, normalized_name1: str, normalized_name2: str) -> bool:
    """Compares two entities using multiple metrics and rules defined in DEDUPLICATION_CONFIG."""
    if entity1['type'] != entity2['type']: return False
    entity_type = entity1['type']
    config_thresholds = DEDUPLICATION_CONFIG["similarity_thresholds"]
    thresholds = config_thresholds.get(entity_type)
    if not thresholds:
        if entity_type in ["NGO", "GOVERNMENT_BODY", "COMMERCIAL_COMPANY"]: thresholds = config_thresholds.get("ORGANIZATION", config_thresholds["DEFAULT"])
        else: thresholds = config_thresholds["DEFAULT"]
    if not normalized_name1 or not normalized_name2: return False
    try:
        token_set = fuzz.token_set_ratio(normalized_name1, normalized_name2)
        jaro_w = jellyfish.jaro_winkler_similarity(normalized_name1, normalized_name2)
    except Exception as sim_err: logger.warning(f"Error calculating similarity for '{normalized_name1}' vs '{normalized_name2}': {sim_err}"); return False
    meets_thresholds = (token_set >= thresholds["token_set_ratio"] and jaro_w >= thresholds["jaro_winkler"])
    if meets_thresholds:
        if len(normalized_name1) < 5 and len(normalized_name2) < 5 and token_set < 100: logger.debug(f"Skipping potential short name match: '{normalized_name1}' vs '{normalized_name2}'"); return False
        return True
    if entity_type == "PERSON" and thresholds.get("initial_match_bonus"):
        if check_initial_match(normalized_name1, normalized_name2):
            parts1 = normalized_name1.split(); parts2 = normalized_name2.split()
            if len(parts1) > 1 and len(parts2) > 1:
                 last_name1 = parts1[-1]; last_name2 = parts2[-1]
                 last_name_jaro = jellyfish.jaro_winkler_similarity(last_name1, last_name2)
                 if last_name_jaro >= 0.90: logger.debug(f"Match based on Initial Rule: '{normalized_name1}' vs '{normalized_name2}'"); return True
                 else: logger.debug(f"Initial Rule rejected due to dissimilar last names: '{last_name1}' vs '{last_name2}'")
    return False
def choose_canonical_entity(entity_list: List[Dict]) -> Dict:
    """Chooses the best representation from a cluster of duplicates and aggregates their context."""
    if not entity_list: return {}
    try: best_entity = max(entity_list, key=lambda e: (len(e.get('name', '')), -ord(e.get('id', 'z'*36)[0])))
    except Exception as max_err: logger.warning(f"Error choosing canonical entity: {max_err}. Entities: {entity_list}"); best_entity = entity_list[0]
    canonical_id = best_entity['id']; canonical_name = best_entity.get('name', 'Unknown'); canonical_type = best_entity.get('type', 'Unknown')
    canonical_source_doc = best_entity.get('source_document', 'Unknown'); canonical_description = best_entity.get('description', '')
    all_chunk_ids = set(); all_doc_ids = set(); all_file_names = set(); all_page_numbers = set(); original_entity_ids = set()
    for entity in entity_list:
        original_entity_ids.add(entity['id']); context = entity.get('context', {})
        if 'chunk_ids' in context and isinstance(context['chunk_ids'], list): all_chunk_ids.update(context['chunk_ids'])
        if context.get('document_id'): all_doc_ids.add(context['document_id'])
        if context.get('file_name'): all_file_names.add(context['file_name'])
        if context.get('page_number') is not None:
            try: all_page_numbers.add(str(context['page_number']))
            except: pass
    canonical_entity = {
        "id": canonical_id, "name": canonical_name, "type": canonical_type, "source_document": canonical_source_doc, "description": canonical_description,
        "context": {
            "chunk_ids": sorted(list(all_chunk_ids)), "document_id": sorted(list(all_doc_ids))[0] if all_doc_ids else None,
            "file_name": sorted(list(all_file_names))[0] if all_file_names else None, "page_number": sorted(list(all_page_numbers))[0] if all_page_numbers else None,
            "original_entity_ids": sorted(list(original_entity_ids)), "is_canonical": True
        }
    }
    return canonical_entity
# --- END: Deduplication Configuration and Helpers ---


# --- Pydantic Models for Fixed Extraction Schema (Unchanged) ---
class EntityRelationshipItem(BaseModel):
    from_entity_type: Optional[Literal[
        "PERSON", "NGO", "GOVERNMENT_BODY", "COMMERCIAL_COMPANY", "LOCATION", "POSITION", "MONEY", "ASSET", "EVENT"]] = Field(default=None)
    from_entity_name: Optional[str] = Field(default=None)
    relationship_type: Optional[Literal["WORKS_FOR", "OWNS", "LOCATED_IN", "CONNECTED_TO", "MET_WITH"]] = Field(default=None)
    relationship_description: Optional[str] = Field(default=None)
    to_entity_name: Optional[str] = Field(default=None)
    to_entity_type: Optional[Literal[
        "PERSON", "NGO", "GOVERNMENT_BODY", "COMMERCIAL_COMPANY", "LOCATION", "POSITION", "MONEY", "ASSET", "EVENT"]] = Field(default=None)

class MetaData(BaseModel):
    summary: Optional[str] = Field(default=None)
    red_flags: Optional[str] = Field(default=None)

class EntityRelationshipList(BaseModel):
    entity_relationship_list: List[EntityRelationshipItem] = Field(...) # Required list
    metadata: Optional[MetaData] = Field(default=None)
# --- END: Pydantic Models ---


class EntityExtractor:
    """
    LLM-based entity and relationship extractor using the active backend.
    """

    def __init__(self, model_name=None, debug=False):
        """
        Initialize entity extractor.
        Args:
            model_name (str, optional): Name of the model to use for extraction (used by all backends).
            debug (bool, optional): Enable debugging output.
        """
        logger.info("Initializing EntityExtractor...")
        self.debug = debug
        self.llm_manager = get_active_llm_manager() # Get the active manager instance

        if self.llm_manager is None:
             logger.error("LLM Manager is None during EntityExtractor initialization! Backend detection will fail.")
             self.is_openrouter = False
             self.is_aphrodite = False
             self.is_gemini = False # Add Gemini flag
        else:
             # Check instance type safely, considering potential import failures
             self.is_openrouter = isinstance(self.llm_manager, OpenRouterManager) if OpenRouterManager else False
             self.is_aphrodite = isinstance(self.llm_manager, AphroditeService) if AphroditeService else False
             self.is_gemini = isinstance(self.llm_manager, GeminiManager) if GeminiManager else False # Add Gemini check

        logger.info(f"EntityExtractor detected backend: {LLM_BACKEND.upper()}")

        # Determine model name based on backend
        if model_name:
            self.model_name = model_name
        elif self.is_openrouter and self.llm_manager:
            self.model_name = self.llm_manager.models.get("extraction", CONFIG.get("openrouter", {}).get("extraction_model", "default/openrouter"))
        elif self.is_gemini and self.llm_manager: # Add Gemini case
            # Use the specific model from the example
            self.model_name = self.llm_manager.models.get("extraction", "gemini-2.5-pro-exp-03-25")
        elif self.is_aphrodite and self.llm_manager:
            # Assuming Aphrodite config structure
            self.model_name = CONFIG.get("models", {}).get("extraction_models", {}).get("text_small", "default/aphrodite")
        else:
            self.model_name = "default/unknown" # Fallback if no manager or backend detected
            logger.error("EntityExtractor could not determine model name due to invalid backend/manager.")

        self.deduplication_config = DEDUPLICATION_CONFIG
        self.extracted_data_path = ROOT_DIR / CONFIG.get("storage", {}).get("extracted_data_path", "data/extracted")
        self.extracted_data_path.mkdir(parents=True, exist_ok=True)

        self.raw_entities: List[Dict] = []
        self.raw_relationships: List[Dict] = []
        self.entities: List[Dict] = []
        self.relationships: List[Dict] = []
        self.modified_chunks: List[Dict] = []
        self.all_chunks: List[Tuple[Dict[str, Any], bool]] = []

        self.entity_types = [
            "PERSON", "NGO", "GOVERNMENT_BODY", "COMMERCIAL_COMPANY",
            "LOCATION", "POSITION", "MONEY", "ASSET", "EVENT"
        ]

        # Store the JSON schema derived from the Pydantic model for OpenRouter
        self.entity_json_schema: Optional[Dict] = None
        if self.is_openrouter and self.llm_manager:
            logger.info("Attempting to generate JSON schema for OpenRouter extraction...")
            self.entity_json_schema = self.llm_manager._pydantic_to_json_schema(EntityRelationshipList)
            if self.entity_json_schema:
                logger.info("Successfully generated JSON schema for OpenRouter.")
            else:
                logger.error("Failed to generate JSON schema for entity extraction from Pydantic model. OpenRouter extraction might fail.")
        # Gemini uses the Pydantic model directly, no pre-conversion needed here.

        logger.info(f"Initialized EntityExtractor with backend: {LLM_BACKEND.upper()}, model={self.model_name}, debug={debug}")
        log_memory_usage(logger)

    def ensure_model_loaded(self) -> bool:
        """Ensure the designated extraction model is ready in the active backend."""
        logger.info("Entering ensure_model_loaded...")
        if self.is_openrouter:
            logger.info("Checking OpenRouter backend readiness...")
            if self.llm_manager and hasattr(self.llm_manager, 'client'):
                 client_ready = self.llm_manager.client is not None
                 logger.info(f"OpenRouter manager found. Client is ready: {client_ready}")
                 if client_ready:
                     logger.info(f"OpenRouter backend ready for extraction (using model: {self.model_name}).")
                     return True
                 else:
                     logger.error("OpenRouter manager client is None. Check API key and initialization.")
                     return False
            else:
                 logger.error("OpenRouter manager or its client attribute not found.")
                 return False
        elif self.is_gemini: # Add Gemini check
             logger.info("Checking Gemini backend readiness...")
             if self.llm_manager and hasattr(self.llm_manager, 'client'):
                  client_ready = self.llm_manager.client is not None
                  logger.info(f"Gemini manager found. Client is ready: {client_ready}")
                  if client_ready:
                      logger.info(f"Gemini backend ready for extraction (using model: {self.model_name}).")
                      return True
                  else:
                      logger.error("Gemini manager client is None. Check API key and initialization.")
                      return False
             else:
                  logger.error("Gemini manager or its client attribute not found.")
                  return False
        elif self.is_aphrodite:
            # Use Aphrodite's existing logic via the manager instance
            logger.info("Checking Aphrodite backend readiness...")
            if not self.llm_manager:
                 logger.error("Aphrodite manager is None in ensure_model_loaded.")
                 return False
            if not self.llm_manager.is_running():
                logger.info("Aphrodite service not running, starting it")
                if not self.llm_manager.start():
                    logger.error("Failed to start Aphrodite service")
                    return False
            status = self.llm_manager.get_status()
            if not status.get("model_loaded", False) or status.get("current_model") != self.model_name:
                logger.info(f"Loading Aphrodite extraction model: {self.model_name}")
                if not self.llm_manager.load_model(self.model_name):
                    logger.error(f"Failed to load Aphrodite extraction model {self.model_name}")
                    return False
                logger.info(f"Aphrodite model {self.model_name} loaded successfully.")
            else:
                logger.info(f"Aphrodite extraction model {self.model_name} already loaded.")
            return True
        else:
            logger.error("No valid LLM backend manager available.")
            return False

    def queue_chunk(self, chunk: Dict[str, Any], is_visual: bool = False):
        """Add a chunk to the collection for later batch processing."""
        chunk_text = chunk.get('text', '')
        if not chunk_text or not chunk_text.strip():
            logger.warning(f"Skipping empty text chunk {chunk.get('chunk_id', 'unknown')}")
            self._add_empty_modified_chunk(chunk)
            return
        self.all_chunks.append((chunk, is_visual))

    def process_chunk(self, chunk: Dict[str, Any], is_visual: bool = False):
        """Process a single chunk (adds to queue, then processes all)."""
        if isinstance(chunk, list):
            visual_ids = list(is_visual) if isinstance(is_visual, (list, set)) else ([c['chunk_id'] for c in chunk] if is_visual else [])
            self.process_chunks(chunk, visual_ids)
            return
        self.queue_chunk(chunk, bool(is_visual))
        self.process_all_chunks()

    def process_chunks(self, chunks: List[Dict[str, Any]], visual_chunks_ids: List[str] = None):
        """Queue multiple chunks for batch processing."""
        if not chunks: return
        visual_chunks_set = set(visual_chunks_ids or [])
        spreadsheet_rows = sum(1 for chunk in chunks if chunk.get('metadata', {}).get('chunk_method') == 'spreadsheet_row')
        logger.info(f"Adding {len(chunks)} chunks for batch processing ({spreadsheet_rows} spreadsheet rows, {len(visual_chunks_ids or [])} visual)")
        for chunk in chunks:
            is_visual = chunk.get('chunk_id', 'unknown') in visual_chunks_set
            self.queue_chunk(chunk, is_visual)
        logger.info(f"Added {len(chunks)} chunks to queue. Processing now.")
        self.process_all_chunks()

    def process_all_chunks(self, progress_callback=None):
        """Process all collected chunks in batches using the active LLM backend."""
        if not self.all_chunks:
            logger.info("No chunks to process")
            return

        total_chunks = len(self.all_chunks)
        logger.info(f"Processing all {total_chunks} chunks using {LLM_BACKEND.upper()} backend...")
        log_memory_usage(logger)

        try:
            logger.info("Ensuring LLM backend model is loaded/ready...")
            model_ready = self.ensure_model_loaded()
            logger.info(f"Model readiness check returned: {model_ready}")
            if not model_ready:
                logger.error(f"LLM backend ({LLM_BACKEND.upper()}) not ready. Adding empty chunks.")
                for chunk, _ in self.all_chunks: self._add_empty_modified_chunk(chunk)
                self.all_chunks = []
                return

            # For now, treat all as text chunks (visual processing not implemented here)
            text_chunks = self.all_chunks
            spreadsheet_chunks = sum(1 for chunk, _ in text_chunks if chunk.get('metadata', {}).get('chunk_method') == 'spreadsheet_row')
            logger.info(f"Processing {len(text_chunks)} chunks ({spreadsheet_chunks} spreadsheet rows)")

            if progress_callback: progress_callback(0.0, f"Preparing to process {total_chunks} chunks")

            if text_chunks:
                start_time = time.time()
                # Batch size primarily relevant for Aphrodite; API calls are sequential per chunk here
                batch_size = CONFIG.get("extraction", {}).get("information_extraction", {}).get("batch_size", 512)

                num_batches = (len(text_chunks) + batch_size - 1) // batch_size
                logger.info(f"Processing in {num_batches} conceptual batches (size ~{batch_size})")

                for i in range(0, len(text_chunks), batch_size):
                    batch = text_chunks[i:min(i + batch_size, len(text_chunks))]
                    batch_num = (i // batch_size) + 1
                    logger.info(f"Processing batch {batch_num}/{num_batches} ({len(batch)} chunks)")
                    if progress_callback: progress_callback(i / total_chunks, f"Processing batch {batch_num}/{num_batches}")

                    # Call the appropriate batch processing method
                    self._process_text_chunks_batch(batch)

                    # Optional: GC between batches
                    # gc.collect()
                    # if torch.cuda.is_available(): torch.cuda.empty_cache()

                elapsed = time.time() - start_time
                logger.info(f"All batch processing completed in {elapsed:.2f}s")

            self.all_chunks = [] # Clear after processing
            if progress_callback: progress_callback(1.0, "All chunks processed")

        except Exception as e:
            logger.error(f"Error processing chunks: {e}", exc_info=True)
            for chunk, _ in self.all_chunks: self._add_empty_modified_chunk(chunk)
            self.all_chunks = []

    def _process_text_chunks_batch(self, text_chunks_batch: List[Tuple[Dict[str, Any], bool]]):
        """Process a single batch of text chunks using the active LLM backend."""
        batch_start_time = time.time()
        processed_count = 0
        error_count = 0

        if self.is_openrouter:
            # --- OpenRouter Path (Sequential Calls per Chunk) ---
            logger.info("Entering OpenRouter processing logic in _process_text_chunks_batch.")
            if not self.entity_json_schema:
                logger.error("JSON schema for entity extraction not available. Cannot process with OpenRouter.")
                for chunk, _ in text_chunks_batch: self._add_empty_modified_chunk(chunk)
                return
            else:
                logger.info("JSON schema found. Proceeding with OpenRouter calls.")

            logger.info(f"Processing {len(text_chunks_batch)} chunks sequentially via OpenRouter...")
            for chunk_idx, (chunk, _) in enumerate(text_chunks_batch):
                chunk_id = chunk.get('chunk_id', f'unknown_batch_chunk_{chunk_idx}')
                chunk_text = chunk.get('text', '')
                if not chunk_text.strip():
                    self._add_empty_modified_chunk(chunk); continue

                # Create OpenAI messages list
                prompt_content = self._create_extraction_prompt(chunk_text)
                messages = [{"role": "system", "content": "You are an AI assistant specialized in extracting structured data."},
                            {"role": "user", "content": prompt_content}]

                try:
                    logger.debug(f"Calling OpenRouterManager.extract_structured for chunk {chunk_id} (Index {chunk_idx})...")
                    response = self.llm_manager.extract_structured(
                        messages=messages,
                        json_schema=self.entity_json_schema,
                        model_name=self.model_name,
                        task_name="extraction"
                    )
                    logger.debug(f"Received response from extract_structured for chunk {chunk_id}: Status={response.get('status')}")
                    if response.get("status") != "success":
                         logger.warning(f"extract_structured failed for chunk {chunk_id}: {response.get('error')}")

                    if response.get("status") == "success":
                        result_dict = response.get("result", {})
                        try:
                            # Use the Pydantic model defined at the top for validation
                            parsed_result = EntityRelationshipList.parse_obj(result_dict)
                            self._process_extraction_result(parsed_result, chunk)
                            processed_count += 1
                        except Exception as pydantic_err:
                            error_count += 1
                            logger.warning(f"Pydantic validation failed for OpenRouter result (chunk {chunk_id}): {pydantic_err}. Raw: {result_dict}")
                            self._add_empty_modified_chunk(chunk)
                    else:
                        error_count += 1
                        self._add_empty_modified_chunk(chunk)

                except Exception as e:
                    error_count += 1
                    logger.error(f"Unexpected error calling OpenRouter for chunk {chunk_id}: {e}", exc_info=True)
                    self._add_empty_modified_chunk(chunk)

        elif self.is_gemini: # Add Gemini Path
            # --- Gemini Path (Sequential Calls per Chunk) ---
            logger.info("Entering Gemini processing logic in _process_text_chunks_batch.")
            if not self.llm_manager or not self.llm_manager.client:
                 logger.error("Gemini manager or client not initialized. Cannot process.")
                 for chunk, _ in text_chunks_batch: self._add_empty_modified_chunk(chunk)
                 return

            logger.info(f"Processing {len(text_chunks_batch)} chunks sequentially via Gemini...")
            for chunk_idx, (chunk, _) in enumerate(text_chunks_batch):
                chunk_id = chunk.get('chunk_id', f'unknown_batch_chunk_{chunk_idx}')
                chunk_text = chunk.get('text', '')
                if not chunk_text.strip():
                    self._add_empty_modified_chunk(chunk); continue

                # Create the single prompt string for Gemini
                prompt_string = self._create_extraction_prompt(chunk_text)

                try:
                    logger.debug(f"Calling GeminiManager.extract_structured for chunk {chunk_id} (Index {chunk_idx})...")
                    # Call Gemini manager, passing the Pydantic model directly
                    response = self.llm_manager.extract_structured(
                        prompt=prompt_string,
                        pydantic_schema=EntityRelationshipList, # Pass the Pydantic class
                        model_name=self.model_name, # Use the specific model name
                        task_name="extraction"
                    )
                    logger.debug(f"Received response from Gemini extract_structured for chunk {chunk_id}: Status={response.get('status')}")
                    if response.get("status") != "success":
                         logger.warning(f"Gemini extract_structured failed for chunk {chunk_id}: {response.get('error')}")

                    if response.get("status") == "success":
                        # Result should already be a parsed dictionary validated by Gemini API
                        result_dict = response.get("result", {})
                        try:
                            # Re-validate with Pydantic locally for consistency (optional but good practice)
                            parsed_result = EntityRelationshipList.parse_obj(result_dict)
                            self._process_extraction_result(parsed_result, chunk)
                            processed_count += 1
                        except Exception as pydantic_err:
                            error_count += 1
                            logger.warning(f"Local Pydantic validation failed for Gemini result (chunk {chunk_id}): {pydantic_err}. Raw: {result_dict}")
                            self._add_empty_modified_chunk(chunk)
                    else:
                        error_count += 1
                        self._add_empty_modified_chunk(chunk)

                except Exception as e:
                    error_count += 1
                    logger.error(f"Unexpected error calling Gemini for chunk {chunk_id}: {e}", exc_info=True)
                    self._add_empty_modified_chunk(chunk)

        elif self.is_aphrodite:
            # --- Aphrodite Path (Batch Call) ---
            logger.info("Entering Aphrodite processing logic in _process_text_chunks_batch.")
            try:
                if not self.llm_manager or not self.llm_manager.is_running():
                    logger.error("Aphrodite service stopped unexpectedly or manager is None.")
                    for chunk, _ in text_chunks_batch: self._add_empty_modified_chunk(chunk)
                    return

                prompts = []
                chunk_map = {} # Map index -> chunk
                for i, (chunk, _) in enumerate(text_chunks_batch):
                    chunk_text = chunk.get('text', '')
                    prompt = self._create_extraction_prompt(chunk_text) # Raw prompt string
                    prompts.append(prompt)
                    chunk_map[i] = chunk

                logger.info(f"Sending batch of {len(prompts)} prompts to Aphrodite service")
                request_start_time = time.time()
                response = self.llm_manager.extract_entities(prompts) # Batch call
                request_elapsed = time.time() - request_start_time
                logger.info(f"Aphrodite service request/response took {request_elapsed:.2f}s")

                if response.get("status") != "success":
                    error_msg = response.get("error", "Unknown error")
                    logger.error(f"Aphrodite service error during extraction: {error_msg}")
                    for chunk, _ in text_chunks_batch: self._add_empty_modified_chunk(chunk)
                    return

                results = response.get("results", [])
                if len(results) != len(text_chunks_batch):
                    logger.warning(f"Aphrodite output count mismatch: Expected {len(text_chunks_batch)}, Got {len(results)}")

                processing_start = time.time()
                missing_count = 0
                for i in range(len(text_chunks_batch)):
                     chunk = chunk_map[i]
                     chunk_id = chunk.get('chunk_id', f'unknown-{i}')
                     if i < len(results):
                         result_text = results[i]
                         try:
                             # Aphrodite returns JSON string, needs parsing
                             result_dict = json.loads(result_text)
                             parsed_result = EntityRelationshipList.parse_obj(result_dict)
                             self._process_extraction_result(parsed_result, chunk)
                             processed_count += 1
                         except Exception as parse_err:
                             error_count += 1
                             logger.warning(f"Error parsing Aphrodite result for chunk {chunk_id}: {parse_err}. Content: '{result_text[:100]}...'")
                             self._add_empty_modified_chunk(chunk)
                     else:
                         missing_count += 1
                         if missing_count % 20 == 1: logger.warning(f"No Aphrodite output for chunk index {i}: {chunk_id}")
                         self._add_empty_modified_chunk(chunk)
                processing_elapsed = time.time() - processing_start
                logger.info(f"Processed Aphrodite batch results in {processing_elapsed:.2f}s.")
                if missing_count > 0: logger.warning(f"Missing Aphrodite outputs: {missing_count}")

            except Exception as e:
                logger.error(f"Error in Aphrodite _process_text_chunks_batch: {e}", exc_info=True)
                for chunk, _ in text_chunks_batch: self._add_empty_modified_chunk(chunk)
        else:
            logger.error("No valid LLM backend configured for batch processing.")
            for chunk, _ in text_chunks_batch: self._add_empty_modified_chunk(chunk)


        total_elapsed = time.time() - batch_start_time
        logger.info(f"Batch processing finished in {total_elapsed:.2f}s. Success: {processed_count}, Errors: {error_count}")


    def _create_extraction_prompt(self, text: str) -> str:
        """Create the extraction prompt content (used by all backends)."""
        # This prompt remains the same as provided previously, suitable for Gemini too.
        prompt = f"""Extract entities, relationships, and free-text metadata from the following text.
Your task is twofold:
1. Identify specific types of entities and their relationships. Ensure all extracted entities and relationships are specific and identifiable (e.g., named individuals, organizations, specific locations); avoid extracting generic pronouns (like 'we', 'they') or vague group descriptions/relationships.
2. Produce an augmented free-text summary of the text along with a section highlighting any potential red flags for corruption.

The JSON output must have two keys: "entity_relationship_list" and "metadata".

-- Entity Extraction --

For each item in the "entity_relationship_list", include the following fields:
- "from_entity_type" (Required): Must be one of the following types: "PERSON", "NGO", "GOVERNMENT_BODY", "COMMERCIAL_COMPANY", "LOCATION", "POSITION", "MONEY", "ASSET", "EVENT".
- "from_entity_name" (Required if an entity is mentioned): The name of the source entity.
- "relationship_type" (Optional): One of the following if applicable: "WORKS_FOR", "OWNS", "LOCATED_IN", "CONNECTED_TO", "MET_WITH".
- "relationship_description" (Optional): A brief description providing extra context for the relationship.
- "to_entity_name" (Optional): The name of the target entity if a relationship exists.
- "to_entity_type" (Optional): Must be one of the allowed entity types if provided.

If an entity is mentioned without a relationship, include it with the relationship fields set to null.
**Important:** Do not output the same entity twice, even if they also appear within a relationship.

-- Metadata Augmentation --

The "metadata" object should include:
- "summary" (Optional but recommended): A concise, natural language summary of the text chunk. This summary should:
  - Capture the overall subject matter and context (for example, whether the text describes a company, financial transactions, regulatory issues, etc.).
  - Mention high-level cues about the types of entities present (e.g., noting that the text involves a person, a commercial company, or a government body) without including specific names or detailed identifiers. This approach—known as entity abstraction or normalization—helps users search for generic entity types.
  - CRITICAL: LIMIT STRICTLY to 100 words.
- "red_flags" (Optional but recommended): A brief free-text section (ideally under 50 words) starting with "Red flags:" that lists any potential indicators or signals related to corruption (e.g., irregular financial practices, suspicious offshore transactions, or unusual relationships).

-- Example Output --

{{
  "entity_relationship_list": [
    {{
      "from_entity_type": "PERSON",
      "from_entity_name": "John Smith",
      "relationship_type": null,
      "relationship_description": null,
      "to_entity_name": null,
      "to_entity_type": null
    }},
    {{
      "from_entity_type": "COMMERCIAL_COMPANY",
      "from_entity_name": "Global Investments Ltd",
      "relationship_type": null,
      "relationship_description": null,
      "to_entity_name": null,
      "to_entity_type": null
    }},
    {{
      "from_entity_type": "PERSON",
      "from_entity_name": "John Smith",
      "relationship_type": "CONNECTED_TO",
      "relationship_description": "Has a business relationship with",
      "to_entity_name": "Global Investments Ltd",
      "to_entity_type": "COMMERCIAL_COMPANY"
    }}
  ],
  "metadata": {{
    "summary": "This document discusses various business activities including financial transactions and partnerships involving a major commercial company and notable individuals. The content is described in abstract terms, referring to a person, a commercial company, and a location, rather than listing specific names. It also implies possible issues with financial transparency.",
    "red_flags": "Red flags: Unusual offshore transactions and inconsistent financial reporting."
  }}
}}

-- Instructions --

1. Process the provided text and extract all relevant entities and their relationships according to the rules above. Do not output the same entity twice, even if they have a relationship.
2. Generate the metadata summary by:
   - Writing a succinct overview of the key content and themes in the text.
   - Refraining from including detailed names. Instead, abstract specific names to their entity types (e.g., use 'a person' rather than 'John Smith') to aid in generalized search.
   - Including a "Red flags:" section at the end if any corruption-related signals are present, or leaving it blank if none are detected.
3. Ensure your output is valid JSON with the exact field names as specified.
4. Use null for any optional field where the information is not applicable.

If no entities or relationships matching the criteria are found in the text, return an empty list [] for the entity_relationship_list

Text to analyze:
{text}
"""
        return prompt

    def _process_extraction_result(self, result: EntityRelationshipList, chunk: Dict[str, Any]):
        """Process extracted entities/relationships and create enhanced chunk text. (Unchanged)"""
        chunk_id = chunk.get('chunk_id', 'unknown')
        document_id = chunk.get('document_id', 'unknown')
        file_name = chunk.get('file_name', 'unknown')
        page_number = chunk.get('page_num', None)
        chunk_text = chunk.get('text', '')
        entity_types_in_chunk = set()

        if result and result.entity_relationship_list:
            for item in result.entity_relationship_list:
                if not item.from_entity_type or not item.from_entity_name:
                    logger.warning(f"Skipping item with missing from_entity details in chunk {chunk_id}: {item}")
                    continue
                from_entity_id = str(uuid.uuid4())
                self.raw_entities.append({
                    "id": from_entity_id, "name": item.from_entity_name.strip(), "type": item.from_entity_type.strip().upper(),
                    "source_document": file_name, "description": f"Found in document: {file_name}",
                    "context": {"chunk_ids": [chunk_id], "document_id": document_id, "file_name": file_name, "page_number": page_number}
                })
                entity_types_in_chunk.add(item.from_entity_type)
                if item.relationship_type and item.to_entity_name and item.to_entity_type:
                    to_entity_id = str(uuid.uuid4())
                    self.raw_entities.append({
                        "id": to_entity_id, "name": item.to_entity_name.strip(), "type": item.to_entity_type.strip().upper(),
                        "source_document": file_name, "description": f"Found in document: {file_name}",
                        "context": {"chunk_ids": [chunk_id], "document_id": document_id, "file_name": file_name, "page_number": page_number}
                    })
                    entity_types_in_chunk.add(item.to_entity_type)
                    rel_id = str(uuid.uuid4())
                    self.raw_relationships.append({
                        "id": rel_id, "source_entity_id": from_entity_id, "target_entity_id": to_entity_id,
                        "relationship_type": item.relationship_type, "type": item.relationship_type,
                        "description": item.relationship_description, "chunk_id": chunk_id, "document_id": document_id,
                        "file_name": file_name, "page_number": page_number
                    })

        summary = "No summary available."; red_flags = "No specific red flags identified."
        if result and hasattr(result, 'metadata') and result.metadata:
            if result.metadata.summary: summary = result.metadata.summary.strip()
            if result.metadata.red_flags: red_flags = result.metadata.red_flags.strip()
        entity_types_list = sorted(list(entity_types_in_chunk))
        entity_types_text = ", ".join(entity_types_list) if entity_types_list else "None detected"
        modified_text = f"""{chunk_text}\n----------\nSummary of chunk:\n{summary}\n\nRed Flags:\n{red_flags}\n\nEntities Included:\n{entity_types_text}\n"""
        modified_chunk_data = chunk.copy(); modified_chunk_data['text'] = modified_text; modified_chunk_data['original_text'] = chunk_text
        if 'metadata' not in modified_chunk_data: modified_chunk_data['metadata'] = {}
        modified_chunk_data['metadata']['extracted_summary'] = summary; modified_chunk_data['metadata']['extracted_red_flags'] = red_flags
        display_entity_types = ["Company" if x == "COMMERCIAL_COMPANY" else x for x in entity_types_list]
        modified_chunk_data['metadata']['extracted_entity_types'] = display_entity_types
        modified_chunk_data['metadata']['file_name'] = file_name; modified_chunk_data['file_name'] = file_name
        self.modified_chunks.append(modified_chunk_data)

    def _add_empty_modified_chunk(self, chunk: Dict[str, Any]):
        """Adds a chunk to modified_chunks with default empty metadata. (Unchanged)"""
        modified_chunk_data = chunk.copy(); chunk_text = chunk.get('text', '')
        modified_text = f"""{chunk_text}\n----------\nSummary of chunk:\nNo summary available. Extraction was not successful or no content was found.\n\nRed Flags:\nNo specific red flags identified.\n\nEntities Included:\nNone detected\n"""
        modified_chunk_data['text'] = modified_text; modified_chunk_data['original_text'] = chunk_text
        if 'metadata' not in modified_chunk_data: modified_chunk_data['metadata'] = {}
        modified_chunk_data['metadata']['extracted_summary'] = "No summary available. Extraction was not successful or no content was found."
        modified_chunk_data['metadata']['extracted_red_flags'] = "No specific red flags identified."
        modified_chunk_data['metadata']['extracted_entity_types'] = []
        file_name = chunk.get('file_name', 'Unknown'); modified_chunk_data['metadata']['file_name'] = file_name; modified_chunk_data['file_name'] = file_name
        self.modified_chunks.append(modified_chunk_data)

    # --- Deduplication Methods (Unchanged) ---
    def deduplicate_entities_enhanced(self):
        """Enhanced entity deduplication using normalization, blocking, multi-metric comparison, and clustering."""
        logger.info(f"Starting enhanced entity deduplication on {len(self.raw_entities)} raw entities...")
        if not self.raw_entities: logger.warning("No raw entities to deduplicate."); self.entities = []; return {}
        try:
            entity_df = pd.DataFrame(self.raw_entities)
            if 'id' not in entity_df.columns: logger.error("Raw entities DataFrame missing 'id'. Adding UUIDs."); entity_df['id'] = [str(uuid.uuid4()) for _ in range(len(entity_df))]
            entity_df['name'] = entity_df['name'].fillna('').astype(str); entity_df['type'] = entity_df['type'].fillna('Unknown').astype(str)
            entity_df['original_index'] = entity_df.index
        except Exception as df_err: logger.error(f"Error creating DataFrame: {df_err}", exc_info=True); self.entities = self.raw_entities; return {e['id']: e['id'] for e in self.entities if 'id' in e}
        logger.info("Normalizing entity names...")
        try:
            entity_df['normalized_name'] = entity_df.apply(lambda row: normalize_name(row.get('name', ''), row.get('type', 'Unknown')), axis=1)
            original_count = len(entity_df); entity_df = entity_df[entity_df['normalized_name'] != '']; removed_count = original_count - len(entity_df)
            if removed_count > 0: logger.warning(f"Removed {removed_count} entities with empty normalized names.")
            if entity_df.empty: logger.warning("No entities remaining after normalization."); self.entities = []; return {}
        except Exception as norm_err: logger.error(f"Error during normalization: {norm_err}", exc_info=True); self.entities = self.raw_entities; return {e['id']: e['id'] for e in self.entities if 'id' in e}
        logger.info("Generating candidate pairs using blocking...")
        candidate_pairs = pd.MultiIndex(levels=[[], []], codes=[[], []]); all_types = entity_df['type'].unique()
        entity_df_for_blocking = entity_df.set_index('original_index', drop=False)
        for entity_type in all_types:
            type_indices = entity_df_for_blocking[entity_df_for_blocking['type'] == entity_type].index
            if len(type_indices) < 2: continue
            type_subset_df = entity_df_for_blocking.loc[type_indices]
            config_thresholds = DEDUPLICATION_CONFIG["similarity_thresholds"]
            thresholds = config_thresholds.get(entity_type) or config_thresholds.get("ORGANIZATION") if entity_type in ["NGO", "GOVERNMENT_BODY", "COMMERCIAL_COMPANY"] else config_thresholds["DEFAULT"]
            window = thresholds.get("blocking_window", 5)
            try:
                indexer = recordlinkage.Index(); indexer.add(SortedNeighbourhood(on='normalized_name', window=window))
                type_pairs = indexer.index(type_subset_df)
                if not type_pairs.empty: candidate_pairs = candidate_pairs.union(type_pairs)
            except Exception as e: logger.warning(f"Blocking failed for type {entity_type} (window={window}): {e}")
        logger.info(f"Generated {len(candidate_pairs)} candidate pairs.")
        if len(candidate_pairs) > 500000: logger.warning(f"Large number of candidate pairs ({len(candidate_pairs)}).")
        entity_df.set_index('original_index', inplace=True, drop=False)
        logger.info("Comparing candidate pairs..."); matches = []; comparison_errors = 0; pair_count = len(candidate_pairs); log_interval = max(1000, pair_count // 20)
        if not candidate_pairs.empty:
            for i, (idx1, idx2) in enumerate(candidate_pairs):
                if i % log_interval == 0 and i > 0: logger.debug(f"Comparing pair {i}/{pair_count}...")
                try:
                    entity1 = entity_df.loc[idx1].to_dict(); entity2 = entity_df.loc[idx2].to_dict()
                    norm_name1 = entity_df.loc[idx1, 'normalized_name']; norm_name2 = entity_df.loc[idx2, 'normalized_name']
                    if compare_entities(entity1, entity2, norm_name1, norm_name2): matches.append(tuple(sorted((entity1['id'], entity2['id']))))
                except KeyError as ke: comparison_errors += 1; logger.warning(f"KeyError comparing pair ({idx1}, {idx2}): {ke}.")
                except Exception as e: comparison_errors += 1; logger.warning(f"Error comparing pair ({idx1}, {idx2}): {e}")
        if comparison_errors > 0: logger.error(f"Encountered {comparison_errors} errors during pair comparison.")
        unique_matches = set(matches); logger.info(f"Found {len(unique_matches)} potential duplicate pairs.")
        merged_map = {}; final_entities = []
        if not unique_matches:
            logger.info("No duplicate pairs found. Keeping all entities.")
            final_entities = entity_df.drop(columns=['original_index', 'normalized_name'], errors='ignore').to_dict('records')
            merged_map = {entity['id']: entity['id'] for entity in final_entities if 'id' in entity}
        else:
            logger.info("Clustering duplicate pairs..."); G = nx.Graph()
            all_entity_ids_in_df = entity_df['id'].tolist(); G.add_nodes_from(all_entity_ids_in_df)
            valid_matches = [(u, v) for u, v in unique_matches if u in all_entity_ids_in_df and v in all_entity_ids_in_df]
            G.add_edges_from(valid_matches); connected_components = list(nx.connected_components(G))
            logger.info(f"Found {len(connected_components)} clusters (including singletons).")
            logger.info("Choosing canonical entities and merging context...")
            entity_lookup_by_id = {row.to_dict().get('id'): row.to_dict() for _, row in entity_df.iterrows() if row.to_dict().get('id')}
            processed_ids = set(); cluster_count = 0
            for component in connected_components:
                cluster_count += 1; #if cluster_count % 500 == 0: logger.debug(f"Processing cluster {cluster_count}/{len(connected_components)}...")
                if not component: continue
                duplicate_entities_list = [entity_lookup_by_id[entity_id] for entity_id in component if entity_id in entity_lookup_by_id]
                if not duplicate_entities_list: logger.warning(f"Cluster component {component} resulted in empty list."); continue
                if len(duplicate_entities_list) == 1:
                    single_entity = duplicate_entities_list[0]; single_entity_id = single_entity['id']
                    context = single_entity.get('context', {}); cleaned_context = {"chunk_ids": sorted(list(set(context.get('chunk_ids', [])))), "document_id": context.get('document_id'), "file_name": context.get('file_name'), "page_number": context.get('page_number'), "original_entity_ids": [single_entity_id], "is_canonical": True}
                    final_entity = {"id": single_entity_id, "name": single_entity.get('name'), "type": single_entity.get('type'), "source_document": single_entity.get('source_document'), "description": single_entity.get('description'), "context": cleaned_context}
                    final_entities.append(final_entity); merged_map[single_entity_id] = single_entity_id; processed_ids.add(single_entity_id)
                else:
                    canonical_entity = choose_canonical_entity(duplicate_entities_list)
                    if canonical_entity and 'id' in canonical_entity:
                        canonical_entity.pop('original_index', None); canonical_entity.pop('normalized_name', None)
                        final_entities.append(canonical_entity)
                        for entity_id in component: merged_map[entity_id] = canonical_entity['id']
                        processed_ids.update(component)
                    else: logger.error(f"Failed to choose canonical entity for component: {component}."); processed_ids.update(component)
            all_original_ids = set(all_entity_ids_in_df)
            if processed_ids != all_original_ids:
                missing_ids = all_original_ids - processed_ids; logger.warning(f"Mismatch: {len(missing_ids)} entity IDs not processed. Example: {list(missing_ids)[:5]}.")
                for missing_id in missing_ids:
                    if missing_id in entity_lookup_by_id:
                        single_entity = entity_lookup_by_id[missing_id]; context = single_entity.get('context', {}); cleaned_context = {"chunk_ids": sorted(list(set(context.get('chunk_ids', [])))), "document_id": context.get('document_id'), "file_name": context.get('file_name'), "page_number": context.get('page_number'), "original_entity_ids": [missing_id], "is_canonical": True}
                        final_entity = {"id": missing_id, "name": single_entity.get('name'), "type": single_entity.get('type'), "source_document": single_entity.get('source_document'), "description": single_entity.get('description'), "context": cleaned_context}
                        final_entities.append(final_entity); merged_map[missing_id] = missing_id
        self.entities = final_entities
        logger.info(f"Enhanced entity deduplication complete. Final entity count: {len(self.entities)}.")
        return merged_map
    def _update_relationships_after_dedup(self, merged_map: Dict[str, str]):
        """Updates relationship source/target IDs based on entity merge map."""
        if not self.raw_relationships: self.relationships = []; return
        logger.info(f"Updating {len(self.raw_relationships)} raw relationships with canonical entity IDs...")
        updated_relationships = []; updated_count = 0; skipped_count = 0
        for rel in self.raw_relationships:
            original_source_id = rel.get('source_entity_id'); original_target_id = rel.get('target_entity_id')
            canonical_source_id = merged_map.get(original_source_id); canonical_target_id = merged_map.get(original_target_id)
            if canonical_source_id and canonical_target_id:
                if canonical_source_id == canonical_target_id: skipped_count += 1; continue
                updated_rel = rel.copy(); updated_rel['source_entity_id'] = canonical_source_id; updated_rel['target_entity_id'] = canonical_target_id
                updated_rel['from_entity_id'] = canonical_source_id; updated_rel['to_entity_id'] = canonical_target_id
                if 'type' not in updated_rel and 'relationship_type' in updated_rel: updated_rel['type'] = updated_rel['relationship_type']
                updated_relationships.append(updated_rel)
                if canonical_source_id != original_source_id or canonical_target_id != original_target_id: updated_count += 1
            else: skipped_count += 1
        self.relationships = updated_relationships
        logger.info(f"Relationship ID update complete. Kept {len(self.relationships)}, Updated IDs in {updated_count}, Skipped {skipped_count}.")
    def deduplicate_relationships(self):
        """Deduplicate relationships based on canonical source, target, and type."""
        logger.info("Starting relationship deduplication...")
        if not self.relationships: logger.info("No relationships to deduplicate."); return
        unique_relationships = []; seen_signatures = set(); duplicate_count = 0
        for rel in self.relationships:
            source_id = rel.get('source_entity_id'); target_id = rel.get('target_entity_id'); rel_type = rel.get('type', rel.get('relationship_type', 'UNKNOWN'))
            if not source_id or not target_id: duplicate_count +=1; continue
            signature = (source_id, target_id, rel_type)
            if signature not in seen_signatures: unique_relationships.append(rel); seen_signatures.add(signature)
            else: duplicate_count += 1
        original_count = len(self.relationships); self.relationships = unique_relationships
        logger.info(f"Relationship deduplication complete. Removed {duplicate_count} duplicates. Final count: {len(self.relationships)}.")
    # --- END: Deduplication Methods ---

    def save_results(self):
        """ Run final deduplication and save extracted entities and relationships. (Unchanged) """
        try:
            logger.info("Starting final processing and saving...")
            if self.all_chunks: logger.info(f"Processing {len(self.all_chunks)} remaining chunks..."); self.process_all_chunks()
            original_entity_count = len(self.raw_entities); original_relationship_count = len(self.raw_relationships)
            logger.info(f"Raw counts before deduplication: {original_entity_count} entities, {original_relationship_count} relationships.")
            if not self.raw_entities and not self.raw_relationships: logger.warning("No raw data to deduplicate or save."); self.entities = []; self.relationships = []
            else:
                logger.info("Running enhanced entity deduplication..."); merge_map = self.deduplicate_entities_enhanced()
                logger.info("Updating relationship IDs..."); self._update_relationships_after_dedup(merge_map)
                logger.info("Running relationship deduplication..."); self.deduplicate_relationships()
            final_entity_count = len(self.entities); final_relationship_count = len(self.relationships)
            logger.info(f"Final counts after deduplication: {final_entity_count} entities, {final_relationship_count} relationships.")
            self.extracted_data_path.mkdir(parents=True, exist_ok=True)
            entities_file = self.extracted_data_path / "entities.json"
            try:
                with open(entities_file, 'w', encoding='utf-8') as f: json.dump(self.entities, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved {len(self.entities)} deduplicated entities to {entities_file}")
            except Exception as save_err: logger.error(f"Failed to save entities.json: {save_err}", exc_info=True)
            relationships_file = self.extracted_data_path / "relationships.json"
            try:
                with open(relationships_file, 'w', encoding='utf-8') as f: json.dump(self.relationships, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved {len(self.relationships)} deduplicated relationships to {relationships_file}")
            except Exception as save_err: logger.error(f"Failed to save relationships.json: {save_err}", exc_info=True)
        except Exception as e: logger.error(f"Critical error during save_results: {e}", exc_info=True)

    def get_modified_chunks(self) -> List[Dict[str, Any]]:
        """Returns the list of chunks with tagged entities. (Unchanged)"""
        if self.all_chunks: logger.warning("Accessing modified chunks before all queued chunks were processed. Processing now."); self.process_all_chunks()
        return self.modified_chunks

    def clear_results(self):
        """Clears extracted data (raw and final) and modified chunks. (Unchanged)"""
        logger.info("Clearing all extraction results and modified chunks.")
        self.raw_entities = []; self.raw_relationships = []; self.entities = []; self.relationships = []
        self.modified_chunks = []; self.all_chunks = []; gc.collect()