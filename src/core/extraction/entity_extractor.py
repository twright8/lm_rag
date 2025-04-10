"""
Entity and relationship extraction module for Anti-Corruption RAG System.
Uses LLM-based extraction with Aphrodite service in a persistent child process.
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
from typing import List, Dict, Any, Optional, Tuple, Set, Literal
from pydantic import BaseModel, Field
import pandas as pd
import recordlinkage
from recordlinkage.index import SortedNeighbourhood
import jellyfish
import networkx as nx
from thefuzz import fuzz

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))

# Import utils
from src.utils.logger import setup_logger
from src.utils.resource_monitor import log_memory_usage
from src.utils.aphrodite_service import get_service

# Initialize logger
logger = setup_logger(__name__)

# Load configuration
CONFIG_PATH = ROOT_DIR / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

# --- START: Deduplication Configuration and Helpers ---

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

# Load or use default config
DEDUPLICATION_CONFIG = CONFIG.get("deduplication", DEFAULT_DEDUPLICATION_CONFIG)


def normalize_name(name: str, entity_type: str) -> str:
    """Applies normalization rules based on entity type using DEDUPLICATION_CONFIG."""
    if not isinstance(name, str) or not name:
        return ""

    config_rules = DEDUPLICATION_CONFIG["normalization_rules"]

    # Determine rule set: Specific type -> ORGANIZATION (if applicable) -> DEFAULT
    rules = config_rules.get(entity_type)
    if not rules:
        if entity_type in ["NGO", "GOVERNMENT_BODY", "COMMERCIAL_COMPANY"]:
            rules = config_rules.get("ORGANIZATION", config_rules["DEFAULT"])
        else:
            rules = config_rules["DEFAULT"]

    normalized_name = name

    # Apply rules
    if rules.get("lowercase"):
        normalized_name = normalized_name.lower()

    # Remove titles (PERSON specific)
    if 'remove_titles' in rules:
        titles_pattern = r'\b(?:' + '|'.join(re.escape(title) for title in rules['remove_titles']) + r')\.?\b'
        normalized_name = re.sub(titles_pattern, '', normalized_name, flags=re.IGNORECASE).strip()

    # Remove suffixes (PERSON or ORGANIZATION)
    suffixes_to_remove = rules.get('remove_suffixes', []) + rules.get('remove_legal_suffixes', [])
    if suffixes_to_remove:
        # Match suffixes at the end of the string, possibly preceded by comma/space
        suffixes_pattern = r'(?:[,\s]+)?(?:' + '|'.join(re.escape(suffix) for suffix in suffixes_to_remove) + r')\.?\b$'
        normalized_name = re.sub(suffixes_pattern, '', normalized_name, flags=re.IGNORECASE).strip()

    # Normalize hyphens before removing other punctuation
    if rules.get("normalize_hyphens"):
        normalized_name = normalized_name.replace('-', ' ')

    # Remove punctuation
    if 'remove_punctuation' in rules and rules['remove_punctuation']:
        normalized_name = re.sub(rules['remove_punctuation'], '', normalized_name)

    # Strip extra whitespace
    if rules.get("strip_whitespace"):
        normalized_name = ' '.join(normalized_name.split())

    return normalized_name

def check_initial_match(name1: str, name2: str) -> bool:
    """Checks if one name is like 'J. Smith' and the other 'John Smith'."""
    if not name1 or not name2: return False
    parts1 = name1.split()
    parts2 = name2.split()
    if len(parts1) < 2 or len(parts2) < 2: return False

    # J. Smith vs John Smith or J Smith vs John Smith
    if len(parts1[0]) <= 2 and parts1[0].endswith('.') and len(parts2[0]) > 1 and parts1[-1] == parts2[-1]:
         # Check first initial, ignore '.'
        return parts1[0][0] == parts2[0][0]
    if len(parts1[0]) == 1 and not parts1[0].endswith('.') and len(parts2[0]) > 1 and parts1[-1] == parts2[-1]:
        return parts1[0][0] == parts2[0][0]

    # John Smith vs J. Smith or John Smith vs J Smith
    if len(parts2[0]) <= 2 and parts2[0].endswith('.') and len(parts1[0]) > 1 and parts1[-1] == parts2[-1]:
        return parts2[0][0] == parts1[0][0]
    if len(parts2[0]) == 1 and not parts2[0].endswith('.') and len(parts1[0]) > 1 and parts1[-1] == parts2[-1]:
        return parts2[0][0] == parts1[0][0]

    return False


def compare_entities(entity1: Dict, entity2: Dict, normalized_name1: str, normalized_name2: str) -> bool:
    """Compares two entities using multiple metrics and rules defined in DEDUPLICATION_CONFIG."""
    if entity1['type'] != entity2['type']:
        return False

    entity_type = entity1['type']
    config_thresholds = DEDUPLICATION_CONFIG["similarity_thresholds"]

    # Determine threshold set: Specific type -> ORGANIZATION -> DEFAULT
    thresholds = config_thresholds.get(entity_type)
    if not thresholds:
        if entity_type in ["NGO", "GOVERNMENT_BODY", "COMMERCIAL_COMPANY"]:
             thresholds = config_thresholds.get("ORGANIZATION", config_thresholds["DEFAULT"])
        else:
             thresholds = config_thresholds["DEFAULT"]

    if not normalized_name1 or not normalized_name2:
        return False # Cannot compare if normalization failed

    # Calculate similarities on NORMALIZED names
    try:
        token_set = fuzz.token_set_ratio(normalized_name1, normalized_name2)
        jaro_w = jellyfish.jaro_winkler_similarity(normalized_name1, normalized_name2)
    except Exception as sim_err:
        logger.warning(f"Error calculating similarity for '{normalized_name1}' vs '{normalized_name2}': {sim_err}")
        return False

    # --- Core Decision Logic ---
    meets_thresholds = (token_set >= thresholds["token_set_ratio"] and
                        jaro_w >= thresholds["jaro_winkler"])

    if meets_thresholds:
        # Optional: Add stricter check for very short names to avoid false positives
        if len(normalized_name1) < 5 and len(normalized_name2) < 5 and token_set < 100:
             logger.debug(f"Skipping potential short name match: '{normalized_name1}' vs '{normalized_name2}' (Scores: TSR={token_set}, JW={jaro_w:.2f})")
             return False
        return True

    # --- Enhancement Rules (e.g., Initial Matching for PERSON) ---
    if entity_type == "PERSON" and thresholds.get("initial_match_bonus"):
        if check_initial_match(normalized_name1, normalized_name2):
            # Add a secondary check - ensure last names are similar enough if initial matches
            parts1 = normalized_name1.split()
            parts2 = normalized_name2.split()
            if len(parts1) > 1 and len(parts2) > 1:
                 last_name1 = parts1[-1]
                 last_name2 = parts2[-1]
                 last_name_jaro = jellyfish.jaro_winkler_similarity(last_name1, last_name2)
                 # Require high similarity for last names if using initial rule
                 if last_name_jaro >= 0.90:
                     logger.debug(f"Match based on Initial Rule: '{normalized_name1}' vs '{normalized_name2}' (Last Name JW: {last_name_jaro:.2f})")
                     return True
                 else:
                      logger.debug(f"Initial Rule rejected due to dissimilar last names: '{last_name1}' vs '{last_name2}' (JW: {last_name_jaro:.2f})")

    return False


def choose_canonical_entity(entity_list: List[Dict]) -> Dict:
    """Chooses the best representation from a cluster of duplicates and aggregates their context."""
    if not entity_list:
        return {}

    # Heuristic: Prefer entity with the longest original name.
    # Break ties using the shortest (lexicographically first) ID for stability.
    try:
        best_entity = max(entity_list, key=lambda e: (len(e.get('name', '')), -ord(e.get('id', 'z'*36)[0])))
    except Exception as max_err:
         logger.warning(f"Error choosing canonical entity (using first entity as fallback): {max_err}. Entities: {entity_list}")
         best_entity = entity_list[0] # Fallback to first entity


    canonical_id = best_entity['id']
    canonical_name = best_entity.get('name', 'Unknown')
    canonical_type = best_entity.get('type', 'Unknown')
    # Preserve original source document and description if available from the chosen best entity
    canonical_source_doc = best_entity.get('source_document', 'Unknown')
    canonical_description = best_entity.get('description', '')

    # Aggregate context
    all_chunk_ids = set()
    all_doc_ids = set()
    all_file_names = set()
    all_page_numbers = set()
    original_entity_ids = set()

    for entity in entity_list:
        original_entity_ids.add(entity['id'])
        context = entity.get('context', {})
        if 'chunk_ids' in context and isinstance(context['chunk_ids'], list):
            all_chunk_ids.update(context['chunk_ids'])
        if context.get('document_id'):
            all_doc_ids.add(context['document_id'])
        if context.get('file_name'):
            all_file_names.add(context['file_name'])
        if context.get('page_number') is not None:
            try: # Ensure page numbers are treated consistently (e.g., as int or str)
                all_page_numbers.add(str(context['page_number']))
            except: pass # Ignore if cannot convert


    # Final canonical entity structure
    canonical_entity = {
        "id": canonical_id,
        "name": canonical_name,
        "type": canonical_type,
        "source_document": canonical_source_doc, # Keep source doc from chosen entity
        "description": canonical_description,   # Keep description from chosen entity
        "context": {
            "chunk_ids": sorted(list(all_chunk_ids)),
            "document_id": sorted(list(all_doc_ids))[0] if all_doc_ids else None, # Take first doc ID for simplicity
            "file_name": sorted(list(all_file_names))[0] if all_file_names else None, # Take first file name
            "page_number": sorted(list(all_page_numbers))[0] if all_page_numbers else None, # Take first page number
            "original_entity_ids": sorted(list(original_entity_ids)), # List of all merged IDs
            "is_canonical": True # Flag indicating this is the chosen representation
        }
    }
    return canonical_entity

# --- END: Deduplication Configuration and Helpers ---


# LLM input/output format models (remain the same)
class EntityRelationshipItem(BaseModel):
    """Item in the entity_relationship_list representing an entity and its relationships."""
    from_entity_type: Optional[Literal[
        "PERSON", "NGO", "GOVERNMENT_BODY", "COMMERCIAL_COMPANY", "LOCATION", "POSITION", "MONETARY_AMOUNT", "ASSET", "EVENT"]] = None
    from_entity_name: Optional[str] = None
    relationship_type: Optional[Literal["WORKS_FOR", "OWNS", "LOCATED_IN", "CONNECTED_TO", "MET_WITH"]] = None
    relationship_description: Optional[str] = None
    to_entity_name: Optional[str] = None
    to_entity_type: Optional[Literal[
        "PERSON", "NGO", "GOVERNMENT_BODY", "COMMERCIAL_COMPANY", "LOCATION", "POSITION", "MONEY", "ASSET", "EVENT"]] = None


class MetaData(BaseModel):
    """Metadata for entity extraction results including summary and red flags."""
    summary: Optional[str] = None
    red_flags: Optional[str] = None


class EntityRelationshipList(BaseModel):
    """Container for the entity_relationship_list."""
    entity_relationship_list: List[EntityRelationshipItem]
    metadata: Optional[MetaData] = None


class EntityExtractor:
    """
    LLM-based entity and relationship extractor with advanced deduplication.
    Uses a persistent Aphrodite service in a child process.
    """

    def __init__(self, model_name=None, debug=False):
        """
        Initialize entity extractor.
        Args:
            model_name (str, optional): Name of the model to use for extraction
            debug (bool, optional): Enable debugging output
        """
        self.debug = debug
        if model_name:
            self.model_name = model_name
        else:
            self.model_name = CONFIG["models"]["extraction_models"]["text_small"]

        # Use the detailed config loaded above
        self.deduplication_config = DEDUPLICATION_CONFIG

        # Storage paths
        self.extracted_data_path = ROOT_DIR / CONFIG["storage"]["extracted_data_path"]
        self.extracted_data_path.mkdir(parents=True, exist_ok=True)

        # Init extraction stores (raw data before deduplication)
        self.raw_entities: List[Dict] = []
        self.raw_relationships: List[Dict] = []

        # Final, deduplicated stores (populated after save_results)
        self.entities: List[Dict] = []
        self.relationships: List[Dict] = []

        self.modified_chunks: List[Dict] = [] # Store chunks modified with tags
        self.all_chunks: List[Tuple[Dict[str, Any], bool]] = [] # Stores tuples of (chunk, is_visual)

        # Entity types for tagging (sync with prompt)
        self.entity_types = [
            "PERSON", "NGO", "GOVERNMENT_BODY", "COMMERCIAL_COMPANY",
            "LOCATION", "POSITION", "MONEY", "ASSET", "EVENT"
        ]

        self.aphrodite_service = get_service()

        logger.info(f"Initialized EntityExtractor with model={self.model_name}, debug={debug}, using advanced deduplication.")
        log_memory_usage(logger)

    def ensure_model_loaded(self):
        """Ensure the designated extraction model is loaded in the service."""
        if not self.aphrodite_service.is_running():
            logger.info("Aphrodite service not running, starting it")
            if not self.aphrodite_service.start():
                logger.error("Failed to start Aphrodite service")
                return False

        status = self.aphrodite_service.get_status()
        if not status.get("model_loaded", False) or status.get("current_model") != self.model_name:
            logger.info(f"Loading designated extraction model: {self.model_name}")
            if not self.aphrodite_service.load_model(self.model_name):
                logger.error(f"Failed to load extraction model {self.model_name}")
                return False
            logger.info(f"Model {self.model_name} loaded successfully for extraction.")
        else:
            logger.info(f"Extraction model {self.model_name} already loaded.")
        return True

    def queue_chunk(self, chunk: Dict[str, Any], is_visual: bool = False):
        """Add a chunk to the collection for later batch processing."""
        chunk_text = chunk.get('text', '')
        if not chunk_text or not chunk_text.strip():
            logger.warning(f"Skipping empty text chunk {chunk.get('chunk_id', 'unknown')}")
            self._add_empty_modified_chunk(chunk) # Add empty chunk to keep counts consistent
            return
        self.all_chunks.append((chunk, is_visual))
        # logger.debug(f"Queued chunk {chunk.get('chunk_id', 'unknown')}. Total queued: {len(self.all_chunks)}") # Reduce logging verbosity

    def process_chunk(self, chunk: Dict[str, Any], is_visual: bool = False):
        """Process a single chunk (adds to queue, then processes all)."""
        if isinstance(chunk, list):
            visual_ids = list(is_visual) if isinstance(is_visual, (list, set)) else ([c['chunk_id'] for c in chunk] if is_visual else [])
            self.process_chunks(chunk, visual_ids)
            return
        self.queue_chunk(chunk, bool(is_visual))
        self.process_all_chunks() # Process immediately

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
        """Process all collected chunks in batches using the persistent LLM service."""
        if not self.all_chunks:
            logger.info("No chunks to process")
            return

        total_chunks = len(self.all_chunks)
        logger.info(f"Processing all {total_chunks} chunks...")
        log_memory_usage(logger)

        try:
            if not self.ensure_model_loaded():
                logger.error(f"Failed to load extraction model ({self.model_name}). Adding empty chunks.")
                for chunk, _ in self.all_chunks: self._add_empty_modified_chunk(chunk)
                self.all_chunks = []
                return

            # For now, treat all as text chunks
            text_chunks = self.all_chunks
            spreadsheet_chunks = sum(1 for chunk, _ in text_chunks if chunk.get('metadata', {}).get('chunk_method') == 'spreadsheet_row')
            logger.info(f"Processing {len(text_chunks)} chunks ({spreadsheet_chunks} spreadsheet rows)")

            if progress_callback: progress_callback(0.0, f"Preparing to process {total_chunks} chunks")

            if text_chunks:
                start_time = time.time()
                batch_size = CONFIG.get("extraction", {}).get("information_extraction", {}).get("batch_size", 512) # Configurable batch size

                num_batches = (len(text_chunks) + batch_size - 1) // batch_size
                logger.info(f"Processing in {num_batches} batches of size {batch_size}")

                for i in range(0, len(text_chunks), batch_size):
                    batch = text_chunks[i:min(i + batch_size, len(text_chunks))]
                    batch_num = (i // batch_size) + 1
                    logger.info(f"Processing batch {batch_num}/{num_batches} ({len(batch)} chunks)")
                    if progress_callback: progress_callback(i / total_chunks, f"Processing batch {batch_num}/{num_batches}")

                    self._process_text_chunks_batch(batch)

                    # Optional: Add short sleep and GC between batches for large datasets
                    # time.sleep(1)
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
        """Process a single batch of text chunks."""
        try:
            batch_start_time = time.time()
            if not self.aphrodite_service.is_running():
                logger.error("Aphrodite service stopped unexpectedly.")
                for chunk, _ in text_chunks_batch: self._add_empty_modified_chunk(chunk)
                return

            prompts = []
            chunk_map = {} # Map index -> chunk
            for i, (chunk, _) in enumerate(text_chunks_batch):
                chunk_text = chunk.get('text', '')
                prompt = self._create_extraction_prompt(chunk_text)
                prompts.append(prompt)
                chunk_map[i] = chunk

            logger.info(f"Sending batch of {len(prompts)} prompts to service")
            request_start_time = time.time()
            try:
                response = self.aphrodite_service.extract_entities(prompts)
                logger.info("Received response from service")
            except Exception as svc_err:
                logger.error(f"Error calling Aphrodite service: {svc_err}", exc_info=True)
                response = {"status": "error", "error": str(svc_err)}
            request_elapsed = time.time() - request_start_time
            logger.info(f"Service request/response took {request_elapsed:.2f}s")

            if response.get("status") != "success":
                error_msg = response.get("error", "Unknown error")
                logger.error(f"Aphrodite service error during extraction: {error_msg}")
                for chunk, _ in text_chunks_batch: self._add_empty_modified_chunk(chunk)
                return

            results = response.get("results", [])
            if len(results) != len(text_chunks_batch):
                logger.warning(f"Output count mismatch: Expected {len(text_chunks_batch)}, Got {len(results)}")

            processing_start = time.time()
            processed_count = 0
            error_count = 0
            missing_count = 0

            for i in range(len(text_chunks_batch)):
                 chunk = chunk_map[i]
                 chunk_id = chunk.get('chunk_id', f'unknown-{i}')

                 if i < len(results):
                     result_text = results[i]
                     parsed_result = None
                     try:
                         # Attempt to parse the JSON output
                         result_dict = json.loads(result_text)
                         parsed_result = EntityRelationshipList.parse_obj(result_dict)
                         processed_count += 1
                         # Process the valid result
                         self._process_extraction_result(parsed_result, chunk)
                     except Exception as parse_err:
                         error_count += 1
                         logger.warning(f"Error parsing result for chunk {chunk_id}: {parse_err}. Content: '{result_text[:100]}...'")
                         self._add_empty_modified_chunk(chunk) # Add empty on parse error

                 else: # Handle missing output for this chunk index
                     missing_count += 1
                     if missing_count % 20 == 1: # Log less frequently for missing outputs
                          logger.warning(f"No output generated for chunk at index {i}: {chunk_id} (and potentially others).")
                     self._add_empty_modified_chunk(chunk)


            processing_elapsed = time.time() - processing_start
            total_elapsed = time.time() - batch_start_time
            logger.info(f"Processed batch results in {processing_elapsed:.2f}s. Total batch time: {total_elapsed:.2f}s")
            logger.info(f"Batch Stats: Success={processed_count}, Parse Errors={error_count}, Missing Outputs={missing_count}")

        except Exception as e:
            logger.error(f"Error in _process_text_chunks_batch: {e}", exc_info=True)
            for chunk, _ in text_chunks_batch: self._add_empty_modified_chunk(chunk)


    def _create_extraction_prompt(self, text: str) -> str:
        """Create the extraction prompt for the LLM."""
        # This prompt remains the same as provided in the original script
        prompt = f"""Extract entities, relationships, and free-text metadata from the following text.
Your task is twofold:
1. Identify specific types of entities and their relationships.
2. Produce an augmented free-text summary of the text along with a section highlighting any potential red flags for corruption.

The JSON output must have two keys: "entity_relationship_list" and "metadata".

-- Entity Extraction --

For each item in the "entity_relationship_list", include the following fields:
- "from_entity_type" (Required): Must be one of the following types: "PERSON", "NGO", "GOVERNMENT_BODY", "COMMERCIAL_COMPANY", "LOCATION", "POSITION", "MONETARY_AMOUNT", "ASSET", "EVENT".
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
  - Be limited to around 100 words.
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

Text to analyze:
{text}
"""
        return prompt

    def _process_extraction_result(self, result: EntityRelationshipList, chunk: Dict[str, Any]):
        """Process extracted entities/relationships and create enhanced chunk text."""
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

                # Add source entity to raw list
                from_entity_id = str(uuid.uuid4()) # Generate temporary ID
                self.raw_entities.append({
                    "id": from_entity_id,
                    "name": item.from_entity_name.strip(),
                    "type": item.from_entity_type.strip().upper(),
                    "source_document": file_name,
                    "description": f"Found in document: {file_name}",
                    "context": {
                        "chunk_ids": [chunk_id],
                        "document_id": document_id,
                        "file_name": file_name,
                        "page_number": page_number,
                    }
                })
                entity_types_in_chunk.add(item.from_entity_type)

                # If relationship exists, add target entity and relationship to raw lists
                if item.relationship_type and item.to_entity_name and item.to_entity_type:
                    to_entity_id = str(uuid.uuid4()) # Generate temporary ID
                    self.raw_entities.append({
                        "id": to_entity_id,
                        "name": item.to_entity_name.strip(),
                        "type": item.to_entity_type.strip().upper(),
                        "source_document": file_name,
                        "description": f"Found in document: {file_name}",
                        "context": {
                            "chunk_ids": [chunk_id],
                            "document_id": document_id,
                            "file_name": file_name,
                            "page_number": page_number,
                        }
                    })
                    entity_types_in_chunk.add(item.to_entity_type)

                    rel_id = str(uuid.uuid4())
                    self.raw_relationships.append({
                        "id": rel_id,
                        "source_entity_id": from_entity_id, # Use temporary IDs for now
                        "target_entity_id": to_entity_id,
                        "relationship_type": item.relationship_type,
                        "type": item.relationship_type, # Compatibility field
                        "description": item.relationship_description,
                        "chunk_id": chunk_id,
                        "document_id": document_id,
                        "file_name": file_name,
                        "page_number": page_number,
                    })

        # Process metadata
        summary = "No summary available."
        red_flags = "No specific red flags identified."
        if result and hasattr(result, 'metadata') and result.metadata:
            if result.metadata.summary: summary = result.metadata.summary.strip()
            if result.metadata.red_flags: red_flags = result.metadata.red_flags.strip()

        entity_types_list = sorted(list(entity_types_in_chunk))
        entity_types_text = ", ".join(entity_types_list) if entity_types_list else "None detected"

        # Create enhanced text for embedding
        modified_text = f"""{chunk_text}
----------
Summary of chunk:
{summary}

Red Flags:
{red_flags}

Entities Included:
{entity_types_text}
"""
        # Store modified chunk data
        modified_chunk_data = chunk.copy()
        modified_chunk_data['text'] = modified_text
        modified_chunk_data['original_text'] = chunk_text

        if 'metadata' not in modified_chunk_data: modified_chunk_data['metadata'] = {}
        modified_chunk_data['metadata']['extracted_summary'] = summary
        modified_chunk_data['metadata']['extracted_red_flags'] = red_flags
        # Clean type names for metadata display if needed
        display_entity_types = ["Company" if x == "COMMERCIAL_COMPANY" else x for x in entity_types_list]
        modified_chunk_data['metadata']['extracted_entity_types'] = display_entity_types
        modified_chunk_data['metadata']['file_name'] = file_name
        modified_chunk_data['file_name'] = file_name # Top level compatibility

        self.modified_chunks.append(modified_chunk_data)
        # logger.debug(f"Processed extraction for chunk {chunk_id}") # Reduce verbosity


    def _add_empty_modified_chunk(self, chunk: Dict[str, Any]):
        """Adds a chunk to modified_chunks with default empty metadata."""
        modified_chunk_data = chunk.copy()
        chunk_text = chunk.get('text', '')
        modified_text = f"""{chunk_text}
----------
Summary of chunk:
No summary available. Extraction was not successful or no content was found.

Red Flags:
No specific red flags identified.

Entities Included:
None detected
"""
        modified_chunk_data['text'] = modified_text
        modified_chunk_data['original_text'] = chunk_text

        if 'metadata' not in modified_chunk_data: modified_chunk_data['metadata'] = {}
        modified_chunk_data['metadata']['extracted_summary'] = "No summary available. Extraction was not successful or no content was found."
        modified_chunk_data['metadata']['extracted_red_flags'] = "No specific red flags identified."
        modified_chunk_data['metadata']['extracted_entity_types'] = []
        file_name = chunk.get('file_name', 'Unknown')
        modified_chunk_data['metadata']['file_name'] = file_name
        modified_chunk_data['file_name'] = file_name

        self.modified_chunks.append(modified_chunk_data)
        # logger.debug(f"Added empty modified chunk for {chunk.get('chunk_id', 'unknown')}") # Reduce verbosity

    # --- START: Enhanced Deduplication Methods ---

    def deduplicate_entities_enhanced(self):
        """Enhanced entity deduplication using normalization, blocking, multi-metric comparison, and clustering."""
        logger.info(f"Starting enhanced entity deduplication on {len(self.raw_entities)} raw entities...")
        if not self.raw_entities:
            logger.warning("No raw entities to deduplicate.")
            self.entities = []  # Ensure final list is empty
            return {}  # Return empty map

        # 1. Prepare DataFrame
        try:
            entity_df = pd.DataFrame(self.raw_entities)
            if 'id' not in entity_df.columns:
                logger.error("Raw entities DataFrame is missing 'id' column. Adding new UUIDs.")
                entity_df['id'] = [str(uuid.uuid4()) for _ in range(len(entity_df))]

            entity_df['name'] = entity_df['name'].fillna('').astype(str)
            entity_df['type'] = entity_df['type'].fillna('Unknown').astype(str)
            entity_df['original_index'] = entity_df.index  # Keep track for mapping back pairs if needed
        except Exception as df_err:
            logger.error(f"Error creating DataFrame from raw entities: {df_err}", exc_info=True)
            self.entities = self.raw_entities
            return {e['id']: e['id'] for e in self.entities if 'id' in e}

        # 2. Normalize Names
        logger.info("Normalizing entity names...")
        try:
            entity_df['normalized_name'] = entity_df.apply(
                lambda row: normalize_name(row.get('name', ''), row.get('type', 'Unknown')), axis=1
            )
            original_count = len(entity_df)
            entity_df = entity_df[entity_df['normalized_name'] != '']
            removed_count = original_count - len(entity_df)
            if removed_count > 0:
                logger.warning(f"Removed {removed_count} entities with empty normalized names.")
            if entity_df.empty:
                logger.warning("No entities remaining after normalization.")
                self.entities = []
                return {}
        except Exception as norm_err:
            logger.error(f"Error during name normalization: {norm_err}", exc_info=True)
            self.entities = self.raw_entities
            return {e['id']: e['id'] for e in self.entities if 'id' in e}

        # 3. Blocking (Generate Candidate Pairs)
        logger.info("Generating candidate pairs using blocking...")
        candidate_pairs = pd.MultiIndex(levels=[[], []], codes=[[], []])
        all_types = entity_df['type'].unique()

        # --- Use original_index from the main DataFrame for blocking ---
        entity_df_for_blocking = entity_df.set_index('original_index', drop=False)  # Set index for RL

        for entity_type in all_types:
            type_indices = entity_df_for_blocking[entity_df_for_blocking['type'] == entity_type].index
            if len(type_indices) < 2: continue

            type_subset_df = entity_df_for_blocking.loc[type_indices]  # Subset using the index

            config_thresholds = DEDUPLICATION_CONFIG["similarity_thresholds"]
            thresholds = config_thresholds.get(entity_type) or config_thresholds.get("ORGANIZATION") if entity_type in [
                "NGO", "GOVERNMENT_BODY", "COMMERCIAL_COMPANY"] else config_thresholds["DEFAULT"]
            window = thresholds.get("blocking_window", 5)

            try:
                indexer = recordlinkage.Index()
                # Use SortedNeighbourhood on the subset DataFrame (already indexed by original_index)
                indexer.add(SortedNeighbourhood(on='normalized_name', window=window))
                type_pairs = indexer.index(type_subset_df)  # Indices are original_index values
                if not type_pairs.empty:
                    candidate_pairs = candidate_pairs.union(type_pairs)
            except Exception as e:
                logger.warning(f"Blocking failed for type {entity_type} (window={window}): {e}")

        logger.info(f"Generated {len(candidate_pairs)} candidate pairs across all types.")
        if len(candidate_pairs) > 500000:
            logger.warning(f"Large number of candidate pairs ({len(candidate_pairs)}). Comparison might be slow.")

        # --- Revert index for easier lookups later if needed ---
        entity_df.set_index('original_index', inplace=True, drop=False)  # Ensure it's indexed by original_index

        # 4. Comparison
        logger.info("Comparing candidate pairs...")
        matches = []
        comparison_errors = 0
        pair_count = len(candidate_pairs)
        log_interval = max(1000, pair_count // 20)  # Adjust logging frequency

        if not candidate_pairs.empty:
            for i, (idx1, idx2) in enumerate(candidate_pairs):
                if i % log_interval == 0 and i > 0:
                    logger.debug(f"Comparing pair {i}/{pair_count}...")
                try:
                    # Use .loc with the original_index values
                    entity1 = entity_df.loc[idx1].to_dict()
                    entity2 = entity_df.loc[idx2].to_dict()
                    norm_name1 = entity_df.loc[idx1, 'normalized_name']
                    norm_name2 = entity_df.loc[idx2, 'normalized_name']

                    if compare_entities(entity1, entity2, norm_name1, norm_name2):
                        # Store match using original IDs for graph building
                        matches.append(tuple(sorted((entity1['id'], entity2['id']))))
                except KeyError as ke:
                    comparison_errors += 1
                    if comparison_errors < 10:
                        logger.warning(
                            f"KeyError comparing pair ({idx1}, {idx2}): {ke}. Check DataFrame indexing or data integrity.")
                except Exception as e:
                    comparison_errors += 1
                    if comparison_errors < 10:
                        logger.warning(f"Error comparing pair ({idx1}, {idx2}): {e}")

        if comparison_errors > 0:
            logger.error(f"Encountered {comparison_errors} errors during pair comparison.")

        unique_matches = set(matches)
        logger.info(f"Found {len(unique_matches)} potential duplicate pairs after comparison.")

        # 5. Clustering using NetworkX (on original IDs)
        merged_map = {}  # old_id -> canonical_id
        final_entities = []

        if not unique_matches:
            logger.info("No duplicate pairs found. Keeping all entities.")
            final_entities = entity_df.drop(columns=['original_index', 'normalized_name'], errors='ignore').to_dict(
                'records')
            merged_map = {entity['id']: entity['id'] for entity in final_entities if 'id' in entity}
        else:
            logger.info("Clustering duplicate pairs...")
            G = nx.Graph()
            all_entity_ids_in_df = entity_df['id'].tolist()  # Get all valid IDs from the DF
            G.add_nodes_from(all_entity_ids_in_df)  # Add nodes first
            valid_matches = [(u, v) for u, v in unique_matches if
                             u in all_entity_ids_in_df and v in all_entity_ids_in_df]
            G.add_edges_from(valid_matches)  # Add only valid edges

            connected_components = list(nx.connected_components(G))
            logger.info(f"Found {len(connected_components)} clusters (including singletons).")

            # 6. Merging and Canonicalization
            logger.info("Choosing canonical entities and merging context...")

            # *** CORRECTED LOOKUP DICTIONARY CREATION ***
            # Create lookup map ensuring 'id' is IN the value dictionary
            entity_lookup_by_id = {}
            for _, row in entity_df.iterrows():  # Iterate over rows
                entity_dict = row.to_dict()
                entity_id = entity_dict.get('id')
                if entity_id:
                    # Store the full dictionary, including 'id'
                    entity_lookup_by_id[entity_id] = entity_dict
            # *******************************************

            processed_ids = set()
            cluster_count = 0
            for component in connected_components:  # component contains original IDs
                cluster_count += 1
                if cluster_count % 500 == 0:
                    logger.debug(f"Processing cluster {cluster_count}/{len(connected_components)}...")

                if not component: continue

                # Retrieve full entity dicts using the CORRECTED lookup
                duplicate_entities_list = [entity_lookup_by_id[entity_id] for entity_id in component if
                                           entity_id in entity_lookup_by_id]

                if not duplicate_entities_list:  # Should not happen if lookup is correct
                    logger.warning(f"Cluster component {component} resulted in empty list after lookup.")
                    continue

                if len(duplicate_entities_list) == 1:
                    # Singleton cluster - retrieve the single entity
                    single_entity = duplicate_entities_list[0]
                    single_entity_id = single_entity['id']

                    # Clean up context and ensure 'is_canonical' is set
                    context = single_entity.get('context', {})
                    cleaned_context = {
                        "chunk_ids": sorted(list(set(context.get('chunk_ids', [])))),
                        "document_id": context.get('document_id'),
                        "file_name": context.get('file_name'),
                        "page_number": context.get('page_number'),
                        "original_entity_ids": [single_entity_id],
                        "is_canonical": True
                    }
                    # Reconstruct the final entity dict, removing temp fields
                    final_entity = {
                        "id": single_entity_id,
                        "name": single_entity.get('name'),
                        "type": single_entity.get('type'),
                        "source_document": single_entity.get('source_document'),
                        "description": single_entity.get('description'),
                        "context": cleaned_context
                    }
                    final_entities.append(final_entity)
                    merged_map[single_entity_id] = single_entity_id
                    processed_ids.add(single_entity_id)

                else:  # len(duplicate_entities_list) > 1
                    # Cluster with duplicates
                    canonical_entity = choose_canonical_entity(duplicate_entities_list)
                    if canonical_entity and 'id' in canonical_entity:  # Ensure canonical entity is valid
                        # Remove temporary fields before adding
                        canonical_entity.pop('original_index', None)
                        canonical_entity.pop('normalized_name', None)

                        final_entities.append(canonical_entity)
                        # Map all original IDs in the component to the canonical ID
                        for entity_id in component:
                            merged_map[entity_id] = canonical_entity['id']
                        processed_ids.update(component)
                    else:
                        logger.error(
                            f"Failed to choose a valid canonical entity for component: {component}. Entities: {duplicate_entities_list}")
                        # As fallback, add first entity from list as singleton? Or skip? Skipping is safer.
                        for entity_id in component:
                            processed_ids.add(entity_id)  # Mark as processed to avoid warning later

            # Verify all original IDs have been processed
            all_original_ids = set(all_entity_ids_in_df)  # Use IDs from the filtered DF
            if processed_ids != all_original_ids:
                missing_ids = all_original_ids - processed_ids
                logger.warning(
                    f"Mismatch: {len(missing_ids)} entity IDs were not processed during clustering/merging. Example: {list(missing_ids)[:5]}. This might indicate an issue.")
                # Add missing entities as singletons only if they exist in lookup
                for missing_id in missing_ids:
                    if missing_id in entity_lookup_by_id:
                        single_entity = entity_lookup_by_id[missing_id]
                        context = single_entity.get('context', {})
                        cleaned_context = {
                            "chunk_ids": sorted(list(set(context.get('chunk_ids', [])))),
                            "document_id": context.get('document_id'),
                            "file_name": context.get('file_name'),
                            "page_number": context.get('page_number'),
                            "original_entity_ids": [missing_id],
                            "is_canonical": True
                        }
                        final_entity = {
                            "id": missing_id, "name": single_entity.get('name'), "type": single_entity.get('type'),
                            "source_document": single_entity.get('source_document'),
                            "description": single_entity.get('description'),
                            "context": cleaned_context
                        }
                        final_entities.append(final_entity)
                        merged_map[missing_id] = missing_id

        # Update the main self.entities list with the deduplicated results
        self.entities = final_entities
        logger.info(f"Enhanced entity deduplication complete. Final entity count: {len(self.entities)}.")
        return merged_map  # Return map for relationship updates

    def _update_relationships_after_dedup(self, merged_map: Dict[str, str]):
        """Updates relationship source/target IDs based on entity merge map."""
        if not self.raw_relationships:
            self.relationships = [] # Ensure final list is empty
            return

        logger.info(f"Updating {len(self.raw_relationships)} raw relationships with canonical entity IDs...")
        updated_relationships = []
        updated_count = 0
        skipped_count = 0

        for rel in self.raw_relationships:
            original_source_id = rel.get('source_entity_id')
            original_target_id = rel.get('target_entity_id')

            # Find the canonical IDs using the map
            canonical_source_id = merged_map.get(original_source_id)
            canonical_target_id = merged_map.get(original_target_id)

            # Only keep relationship if both source and target map to a valid canonical ID
            if canonical_source_id and canonical_target_id:
                # Avoid self-loops potentially created by merging
                if canonical_source_id == canonical_target_id:
                    skipped_count += 1
                    continue

                updated_rel = rel.copy()
                updated_rel['source_entity_id'] = canonical_source_id
                updated_rel['target_entity_id'] = canonical_target_id

                # Ensure compatibility fields are also updated/present
                updated_rel['from_entity_id'] = canonical_source_id
                updated_rel['to_entity_id'] = canonical_target_id
                if 'type' not in updated_rel and 'relationship_type' in updated_rel:
                     updated_rel['type'] = updated_rel['relationship_type'] # Ensure 'type' exists

                updated_relationships.append(updated_rel)

                if canonical_source_id != original_source_id or canonical_target_id != original_target_id:
                    updated_count += 1
            else:
                # Log if a relationship's entity was completely removed (should be rare if map is correct)
                # logger.warning(f"Skipping relationship {rel.get('id')} as source/target ID mapping failed. Original IDs: ({original_source_id}, {original_target_id})")
                skipped_count += 1

        self.relationships = updated_relationships # Store the updated relationships
        logger.info(f"Relationship ID update complete. Kept {len(self.relationships)}, Updated IDs in {updated_count}, Skipped {skipped_count} (self-loops or missing mappings).")


    def deduplicate_relationships(self):
        """Deduplicate relationships based on canonical source, target, and type."""
        logger.info("Starting relationship deduplication...")
        if not self.relationships:
            logger.info("No relationships to deduplicate.")
            return

        unique_relationships = []
        seen_signatures = set()
        duplicate_count = 0

        for rel in self.relationships:
            # Signature uses canonical IDs and the 'type' field preferentially
            source_id = rel.get('source_entity_id')
            target_id = rel.get('target_entity_id')
            rel_type = rel.get('type', rel.get('relationship_type', 'UNKNOWN')) # Use 'type'

            if not source_id or not target_id: # Skip if IDs are missing
                 duplicate_count +=1 # Treat as invalid/duplicate
                 continue

            signature = (source_id, target_id, rel_type)

            if signature not in seen_signatures:
                unique_relationships.append(rel)
                seen_signatures.add(signature)
            else:
                # logger.debug(f"Removing duplicate relationship: {signature}") # Reduce verbosity
                duplicate_count += 1

        original_count = len(self.relationships)
        self.relationships = unique_relationships
        logger.info(f"Relationship deduplication complete. Removed {duplicate_count} duplicates. Final count: {len(self.relationships)}.")

    # --- END: Enhanced Deduplication Methods ---

    def save_results(self):
        """
        Run final deduplication and save extracted entities and relationships.
        """
        try:
            logger.info("Starting final processing and saving...")
            # Process any remaining chunks if needed
            if self.all_chunks:
                logger.info(f"Processing {len(self.all_chunks)} remaining chunks before saving...")
                self.process_all_chunks()

            original_entity_count = len(self.raw_entities)
            original_relationship_count = len(self.raw_relationships)
            logger.info(f"Raw counts before deduplication: {original_entity_count} entities, {original_relationship_count} relationships.")

            if not self.raw_entities and not self.raw_relationships:
                logger.warning("No raw data to deduplicate or save.")
                self.entities = []
                self.relationships = []
            else:
                # 1. Deduplicate Entities (Enhanced) -> Populates self.entities
                logger.info("Running enhanced entity deduplication...")
                merge_map = self.deduplicate_entities_enhanced() # Returns map, updates self.entities

                # 2. Update Relationships using the merge map -> Populates self.relationships
                logger.info("Updating relationship IDs based on entity deduplication map...")
                self._update_relationships_after_dedup(merge_map) # Operates on self.raw_relationships, stores result in self.relationships

                # 3. Deduplicate Relationships (using canonical IDs) -> Updates self.relationships
                logger.info("Running relationship deduplication...")
                self.deduplicate_relationships() # Operates on self.relationships

            # Log final counts
            final_entity_count = len(self.entities)
            final_relationship_count = len(self.relationships)
            logger.info(f"Final counts after deduplication: {final_entity_count} entities, {final_relationship_count} relationships.")

            # 4. Save Final Results
            self.extracted_data_path.mkdir(parents=True, exist_ok=True)

            # Save deduplicated entities
            entities_file = self.extracted_data_path / "entities.json"
            try:
                with open(entities_file, 'w', encoding='utf-8') as f:
                    json.dump(self.entities, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved {len(self.entities)} deduplicated entities to {entities_file}")
            except Exception as save_err:
                 logger.error(f"Failed to save entities.json: {save_err}", exc_info=True)


            # Save deduplicated relationships
            relationships_file = self.extracted_data_path / "relationships.json"
            try:
                with open(relationships_file, 'w', encoding='utf-8') as f:
                    json.dump(self.relationships, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved {len(self.relationships)} deduplicated relationships to {relationships_file}")
            except Exception as save_err:
                 logger.error(f"Failed to save relationships.json: {save_err}", exc_info=True)

            # Optionally save modified chunks if needed downstream (though maybe not necessary anymore)
            # modified_chunks_file = self.extracted_data_path / "modified_chunks.json"
            # try:
            #     with open(modified_chunks_file, 'w', encoding='utf-8') as f:
            #         json.dump(self.modified_chunks, f, indent=2, ensure_ascii=False)
            #     logger.info(f"Saved {len(self.modified_chunks)} modified chunks to {modified_chunks_file}")
            # except Exception as save_err:
            #      logger.error(f"Failed to save modified_chunks.json: {save_err}", exc_info=True)

        except Exception as e:
            logger.error(f"Critical error during save_results: {e}", exc_info=True)


    def get_modified_chunks(self) -> List[Dict[str, Any]]:
        """Returns the list of chunks with tagged entities."""
        # Ensure chunks are processed before returning modified ones
        if self.all_chunks:
             logger.warning("Accessing modified chunks before all queued chunks were processed. Processing now.")
             self.process_all_chunks()
        return self.modified_chunks

    def clear_results(self):
        """Clears extracted data (raw and final) and modified chunks."""
        logger.info("Clearing all extraction results and modified chunks.")
        self.raw_entities = []
        self.raw_relationships = []
        self.entities = []
        self.relationships = []
        self.modified_chunks = []
        self.all_chunks = []
        gc.collect()