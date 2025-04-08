"""
Entity and relationship extraction module for Anti-Corruption RAG System.
Uses LLM-based extraction with Aphrodite service in a persistent child process.
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
from typing import List, Dict, Any, Union, Optional, Tuple, Set
from pydantic import BaseModel, Field
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

# LLM input/output format - what the model generates
class EntityRelationshipItem(BaseModel):
    """Item in the entity_relationship_list representing an entity and its relationships."""
    from_entity_type: str
    from_entity_name: str
    relationship_type: Optional[str] = None
    to_entity_name: Optional[str] = None
    to_entity_type: Optional[str] = None


class EntityRelationshipList(BaseModel):
    """Container for the entity_relationship_list."""
    entity_relationship_list: List[EntityRelationshipItem]


class EntityExtractor:
    """
    LLM-based entity and relationship extractor.
    Uses a persistent Aphrodite service in a child process.
    """

    def __init__(self, model_name=None, debug=False):
        """
        Initialize entity extractor.

        Args:
            model_name (str, optional): Name of the model to use for extraction
            debug (bool, optional): Enable debugging output
        """
        # Set debug mode
        self.debug = debug

        # Set model name
        if model_name:
            self.model_name = model_name
        else:
            # Ensure we select a model suitable for extraction (potentially instruction-tuned)
            # Defaulting to text_small as before, assuming it's suitable
            self.model_name = CONFIG["models"]["extraction_models"]["text_small"]

        # Deduplication configuration
        self.deduplication_threshold = CONFIG["extraction"]["deduplication_threshold"]

        # Storage paths
        self.extracted_data_path = ROOT_DIR / CONFIG["storage"]["extracted_data_path"]
        self.extracted_data_path.mkdir(parents=True, exist_ok=True)

        # Init extraction stores
        self.entities = []
        self.relationships = []
        self.modified_chunks = []  # Store chunks modified with tags

        # Collection for ALL chunks - will process in one go
        self.all_chunks = []  # Stores tuples of (chunk, is_visual)

        # Request tracking
        self.pending_requests = {}  # Maps request_id -> chunk data

        # Entity types for tagging (sync with prompt)
        self.entity_types = [
            "PERSON", "ORGANIZATION", "GOVERNMENT_BODY", "COMMERCIAL_COMPANY",
            "LOCATION", "POSITION", "MONEY", "ASSET", "EVENT"
        ]

        # LLM service reference
        self.aphrodite_service = get_service()

        logger.info(f"Initialized EntityExtractor with model={self.model_name}, debug={debug}")
        log_memory_usage(logger)

    def ensure_model_loaded(self):
        """
        Ensure the designated extraction model is loaded in the service.

        Returns:
            bool: Success status
        """
        # Check if the service is running
        if not self.aphrodite_service.is_running():
            logger.info("Aphrodite service not running, starting it")
            if not self.aphrodite_service.start():
                logger.error("Failed to start Aphrodite service")
                return False

        # Check if the correct model is loaded
        status = self.aphrodite_service.get_status()
        if not status.get("model_loaded", False) or status.get("current_model") != self.model_name:
            logger.info(f"Loading designated extraction model: {self.model_name}")
            # Load model generically, no is_chat flag needed
            if not self.aphrodite_service.load_model(self.model_name):
                logger.error(f"Failed to load extraction model {self.model_name}")
                return False
            logger.info(f"Model {self.model_name} loaded successfully for extraction.")
        else:
             logger.info(f"Extraction model {self.model_name} already loaded.")

        return True

    # We no longer need a shutdown method that releases the model
    # as the service persists beyond document processing

    def queue_chunk(self, chunk: Dict[str, Any], is_visual: bool = False):
        """
        Add a chunk to the collection for later batch processing.

        Args:
            chunk (dict): Chunk data (must contain 'text', optional 'page_img')
            is_visual (bool): Whether to process this as visual content
        """
        # Skip empty chunks
        chunk_text = chunk.get('text', '')
        if not chunk_text.strip():
            logger.warning(f"Skipping empty text chunk {chunk.get('chunk_id', 'unknown')}")
            self._add_empty_modified_chunk(chunk)  # Add empty chunk to keep counts consistent
            return

        # Add to collection
        self.all_chunks.append((chunk, is_visual))
        logger.debug(f"Queued chunk {chunk.get('chunk_id', 'unknown')} for batch processing (current total: {len(self.all_chunks)})")

    def process_chunk(self, chunk: Dict[str, Any], is_visual: bool = False):
        """
        Process a single chunk by adding it to the collection.
        For backward compatibility. Handles list input from app.py.

        Args:
            chunk (dict or list): Chunk data or list of chunks
            is_visual (bool or list): Whether to process as visual or list of visual IDs
        """
        # Check if we got a list of chunks (from app.py) instead of a single chunk
        if isinstance(chunk, list):
            logger.info(f"Received a list of {len(chunk)} chunks. Processing as batch.")
            # Process the list of chunks with the provided visual flag
            # Ensure visual_chunks_ids is a list if is_visual is a list/set, otherwise handle bool
            visual_ids = list(is_visual) if isinstance(is_visual, (list, set)) else ([c['chunk_id'] for c in chunk] if is_visual else [])
            self.process_chunks(chunk, visual_ids)
            return

        # Handle single chunk processing
        self.queue_chunk(chunk, bool(is_visual)) # Ensure is_visual is bool

        # Process right away for single chunks
        self.process_all_chunks()


    def process_chunks(self, chunks: List[Dict[str, Any]], visual_chunks_ids: List[str] = None):
        """
        Queue multiple chunks for batch processing.

        Args:
            chunks (List[Dict]): List of chunks to process
            visual_chunks_ids (List[str], optional): List of chunk IDs to process as visual
        """
        if not chunks:
            logger.warning("No chunks to process")
            return

        # Count spreadsheet rows vs regular chunks
        spreadsheet_rows = sum(1 for chunk in chunks if chunk.get('metadata', {}).get('chunk_method') == 'spreadsheet_row')
        regular_chunks = len(chunks) - spreadsheet_rows
        
        visual_chunks_ids = visual_chunks_ids or []
        visual_chunks_set = set(visual_chunks_ids)

        logger.info(f"Adding {len(chunks)} chunks for batch processing ({spreadsheet_rows} spreadsheet rows, {regular_chunks} regular chunks, {len(visual_chunks_ids)} visual chunks)")

        # Queue all chunks
        chunk_counter = 0
        for chunk in chunks:
            chunk_id = chunk.get('chunk_id', 'unknown')
            is_visual = chunk_id in visual_chunks_set
            is_spreadsheet = chunk.get('metadata', {}).get('chunk_method') == 'spreadsheet_row'
            
            chunk_counter += 1
            if chunk_counter % 10 == 0 or chunk_counter == 1 or chunk_counter == len(chunks):
                logger.debug(f"Queuing chunk {chunk_counter}/{len(chunks)}, ID: {chunk_id}, is_spreadsheet: {is_spreadsheet}, is_visual: {is_visual}")
            
            self.queue_chunk(chunk, is_visual)

        # Process immediately after queuing all
        logger.info(f"Added {len(chunks)} chunks to queue. Processing now.")
        self.process_all_chunks()

    def process_all_chunks(self, progress_callback=None):
        """
        Process all collected chunks in a single batch using the persistent LLM service.

        Args:
            progress_callback: Optional callback function to report progress (0.0-1.0)
        """
        if not self.all_chunks:
            logger.info("No chunks to process")
            return

        total_chunks = len(self.all_chunks)
        logger.info(f"Processing all {total_chunks} chunks at once")
        
        # Log memory usage at start of processing
        logger.info("Memory usage before starting chunk processing:")
        log_memory_usage(logger)

        try:
            # Ensure the designated extraction model is loaded in the service
            if not self.ensure_model_loaded():
                logger.error(f"Failed to load the extraction model ({self.model_name}). Adding empty chunks.")
                for chunk, _ in self.all_chunks:
                    self._add_empty_modified_chunk(chunk)
                self.all_chunks = []
                return

            # Separate visual and text chunks
            text_chunks = []
            visual_chunks = []
            
            # Count spreadsheet chunks for logging
            spreadsheet_chunks = 0

            for chunk, is_visual in self.all_chunks:
                # Check if it's a spreadsheet row
                is_spreadsheet = chunk.get('metadata', {}).get('chunk_method') == 'spreadsheet_row'
                if is_spreadsheet:
                    spreadsheet_chunks += 1
                
                # Note: Actual visual processing logic isn't implemented here yet.
                # If visual processing were added, it might need a different model/prompt.
                # For now, visual chunks are treated like text chunks.
                # if is_visual and chunk.get("page_img"):
                #    visual_chunks.append((chunk, is_visual))
                # else:
                text_chunks.append((chunk, is_visual))

            # Log chunk counts
            logger.info(f"Processing {len(text_chunks)} chunks ({spreadsheet_chunks} spreadsheet rows, {len(text_chunks) - spreadsheet_chunks} regular chunks)")

            # Report initial progress
            if progress_callback:
                progress_callback(0.0, f"Preparing to process {total_chunks} chunks")

            # Process text chunks as a single batch
            if text_chunks:
                start_time = time.time()
                logger.info(f"Processing {len(text_chunks)} text chunks in a single batch")

                if progress_callback:
                    progress_callback(0.1, f"Processing {len(text_chunks)} text chunks in a batch")

                try:
                    # Log initial state before batch processing
                    logger.info("Starting batch processing of text chunks...")
                    
                    # Break into smaller batches if too many chunks
                    batch_size = 2048  # Process in batches of 100 chunks
                    
                    if len(text_chunks) > batch_size:
                        logger.info(f"Breaking processing into {(len(text_chunks) + batch_size - 1) // batch_size} batches of {batch_size} chunks")
                        
                        for i in range(0, len(text_chunks), batch_size):
                            batch = text_chunks[i:i+batch_size]
                            batch_start = time.time()
                            logger.info(f"Processing batch {i//batch_size + 1}/{(len(text_chunks) + batch_size - 1)//batch_size} ({len(batch)} chunks)")
                            
                            # Process this batch
                            self._process_text_chunks_batch(batch)
                            
                            batch_elapsed = time.time() - batch_start
                            logger.info(f"Batch {i//batch_size + 1} completed in {batch_elapsed:.2f}s")
                    else:
                        # Process all chunks in one go if under the batch size
                        self._process_text_chunks_batch(text_chunks)
                    
                    elapsed = time.time() - start_time
                    logger.info(f"All batch text processing completed in {elapsed:.2f}s")
                    
                except Exception as e:
                    logger.error(f"Error during batch processing: {e}", exc_info=True)
                    # Continue with processing to ensure we handle as many chunks as possible

                if progress_callback:
                    progress_callback(1.0, "All chunks processed") # Now goes to 1.0

            # Clear the collection
            self.all_chunks = []

        except Exception as e:
            logger.error(f"Error processing chunks: {e}")
            import traceback
            logger.error(traceback.format_exc())

            # Add empty chunks on error as well
            for chunk, _ in self.all_chunks:
                self._add_empty_modified_chunk(chunk)

            # Clear the collection
            self.all_chunks = []

    def _process_text_chunks_batch(self, text_chunks: List[Tuple[Dict[str, Any], bool]]):
        """
        Process a batch of text chunks using the persistent LLM service.

        Args:
            text_chunks: List of (chunk, is_visual) tuples to process
        """
        try:
            batch_start_time = time.time()
            logger.info(f"Starting to process batch of {len(text_chunks)} text chunks")
            
            # Model should already be loaded by ensure_model_loaded
            if not self.aphrodite_service.is_running():
                 logger.error("Aphrodite service stopped unexpectedly.")
                 for chunk, _ in text_chunks:
                     self._add_empty_modified_chunk(chunk)
                 return

            # Count spreadsheet chunks for special handling
            spreadsheet_chunks = 0
            for chunk, _ in text_chunks:
                if chunk.get('metadata', {}).get('chunk_method') == 'spreadsheet_row':
                    spreadsheet_chunks += 1
            
            if spreadsheet_chunks > 0:
                logger.info(f"Batch contains {spreadsheet_chunks} spreadsheet rows")

            # Prepare prompts for all text chunks
            logger.info("Preparing extraction prompts...")
            
            prompts = []
            chunk_map = {}  # Map index -> chunk for tracking

            prompt_prep_start = time.time()
            for i, (chunk, _) in enumerate(text_chunks):
                chunk_id = chunk.get('chunk_id', f'unknown-{i}')
                chunk_text = chunk.get('text', '')
                is_spreadsheet = chunk.get('metadata', {}).get('chunk_method') == 'spreadsheet_row'
                
                # Log every 10 chunks, plus first and last for tracking progress
                if i % 10 == 0 or i == 0 or i == len(text_chunks) - 1:
                    logger.debug(f"Preparing prompt {i+1}/{len(text_chunks)} for chunk {chunk_id} (spreadsheet: {is_spreadsheet}, text length: {len(chunk_text)})")
                
                prompt = self._create_extraction_prompt(chunk_text)
                prompts.append(prompt)
                chunk_map[i] = chunk
            
            logger.info(f"Prompts prepared in {time.time() - prompt_prep_start:.2f}s")

            # Generate outputs in a single batch through the service
            request_start_time = time.time()
            logger.info(f"Sending batch of {len(prompts)} extraction prompts to Aphrodite service")

            # Send request to the service (extract_entities applies extraction params)
            try:
                response = self.aphrodite_service.extract_entities(prompts)
                logger.info("Successfully received response from Aphrodite service")
            except Exception as svc_err:
                logger.error(f"Error from Aphrodite service: {svc_err}", exc_info=True)
                # Create an error response
                response = {"status": "error", "error": str(svc_err)}

            request_elapsed = time.time() - request_start_time
            logger.info(f"Batch request/response cycle took {request_elapsed:.2f}s")

            # Check response status
            if response.get("status") != "success":
                error_msg = response.get("error", "Unknown error")
                logger.error(f"Error from Aphrodite service during extraction: {error_msg}")
                # Add empty chunks on error
                logger.info(f"Adding {len(text_chunks)} empty modified chunks due to service error")
                for chunk, _ in text_chunks:
                    self._add_empty_modified_chunk(chunk)
                return

            # Get results from response
            results = response.get("results", [])
            logger.info(f"Received {len(results)} results from service")

            # Check if we got the expected number of outputs
            if len(results) != len(text_chunks):
                logger.warning(f"Expected {len(text_chunks)} outputs but got {len(results)} - this may cause issues")

            # Process the results
            processing_start = time.time()
            logger.info("Processing extraction results...")
            
            processed_count = 0
            error_count = 0
            
            for i, result_text in enumerate(results):
                if i >= len(text_chunks):
                    logger.warning(f"Received more results than chunks, skipping result at index {i}")
                    continue

                chunk = chunk_map[i]
                chunk_id = chunk.get('chunk_id', f'unknown-{i}')
                is_spreadsheet = chunk.get('metadata', {}).get('chunk_method') == 'spreadsheet_row'

                # Log every 10 chunks, plus first and last for tracking progress
                if i % 10 == 0 or i == 0 or i == len(results) - 1:
                    logger.debug(f"Processing output for chunk {i+1}/{len(results)}: {chunk_id} (spreadsheet: {is_spreadsheet})")

                if self.debug:
                    # Print the full output to console in debug mode
                    print(f"\n===== APHRODITE OUTPUT FOR CHUNK {chunk_id} =====")
                    print(result_text)
                    print("========================================")

                # Parse the extraction result
                parsed_result = None
                try:
                    # Direct JSON parsing (rely on JSONLogitsProcessor if available)
                    result_dict = json.loads(result_text)
                    parsed_result = EntityRelationshipList.parse_obj(result_dict)
                    processed_count += 1
                except Exception as parse_err:
                    error_count += 1
                    logger.warning(f"Error parsing result for chunk {chunk_id}: {parse_err}. Content: '{result_text[:100]}...'")


                # Check if we got a valid result
                if parsed_result and hasattr(parsed_result, 'entity_relationship_list'):
                    # Log results
                    num_items = len(parsed_result.entity_relationship_list) if parsed_result.entity_relationship_list else 0
                    
                    # Only log detailed info for significant extractions or spreadsheets
                    if num_items > 0 or is_spreadsheet:
                        logger.info(f"Extracted {num_items} items from chunk {chunk_id} (spreadsheet: {is_spreadsheet})")

                    # Process the extraction result
                    self._process_extraction_result(parsed_result, chunk)
                else:
                    logger.warning(f"No valid extraction result obtained for chunk {chunk_id}. Adding empty modified chunk.")
                    self._add_empty_modified_chunk(chunk)

            # Handle any chunks that didn't get results
            missing_count = 0
            for i, (chunk, _) in enumerate(text_chunks):
                if i >= len(results):
                    missing_count += 1
                    chunk_id = chunk.get('chunk_id', f'unknown-{i}')
                    
                    # Log only every 10th missing chunk to avoid log flooding
                    if missing_count % 10 == 1:
                        logger.warning(f"No output generated for chunk at index {i}: {chunk_id} (and potentially others). Adding empty modified chunks.")
                    
                    self._add_empty_modified_chunk(chunk)
                    
            if missing_count > 0:
                logger.warning(f"Total of {missing_count} chunks had no output generated")
            
            # Log processing statistics
            processing_elapsed = time.time() - processing_start
            total_elapsed = time.time() - batch_start_time
            logger.info(f"Processing of results completed in {processing_elapsed:.2f}s")
            logger.info(f"Complete batch processing took {total_elapsed:.2f}s")
            logger.info(f"Statistics: {processed_count} chunks processed successfully, {error_count} parsing errors, {missing_count} missing outputs")

        except Exception as e:
            logger.error(f"Error processing text chunks batch: {e}")
            import traceback
            logger.error(traceback.format_exc())

            # Add empty chunks for all chunks in the batch
            for chunk, _ in text_chunks:
                self._add_empty_modified_chunk(chunk)

    def _create_extraction_prompt(self, text: str) -> str:
        """
        Create the extraction prompt for the LLM.

        Args:
            text (str): Text to extract entities from

        Returns:
            str: Prompt for the LLM
        """
        # This prompt remains the same, instructing the LLM on the task and format.
        prompt = f"""
Extract entities and relationships from the following text.
Your task is to identify specific types of entities and their relationships.

Entities must be one of the following types:
- PERSON: Individual people mentioned
- ORGANIZATION: Generic organizations
- GOVERNMENT_BODY: Government entities or agencies
- COMMERCIAL_COMPANY: Business entities
- LOCATION: Physical locations or geographic areas
- POSITION: Job titles or roles
- MONEY: Monetary amounts or values
- ASSET: Physical or financial assets
- EVENT: Meetings, incidents, or happenings

Output the results as a JSON object with a single key "entity_relationship_list".
Each item in the list should have:
- "from_entity_type": Type of the source entity.
- "from_entity_name": Name of the source entity.
- "relationship_type": (Optional) Type of the relationship (e.g., "WORKS_FOR", "OWNS", "LOCATED_IN").
- "to_entity_name": (Optional) Name of the target entity.
- "to_entity_type": (Optional) Type of the target entity.

If an entity is mentioned without a relationship, include it with null relationship fields.

Example output:
{{
  "entity_relationship_list": [
    {{
      "from_entity_type": "PERSON",
      "from_entity_name": "John Smith",
      "relationship_type": null,
      "to_entity_name": null,
      "to_entity_type": null
    }},
    {{
      "from_entity_type": "COMMERCIAL_COMPANY",
      "from_entity_name": "Global Investments Ltd",
      "relationship_type": null,
      "to_entity_name": null,
      "to_entity_type": null
    }},
    {{
      "from_entity_type": "PERSON",
      "from_entity_name": "John Smith",
      "relationship_type": "ASSOCIATED_WITH",
      "to_entity_name": "Global Investments Ltd",
      "to_entity_type": "COMMERCIAL_COMPANY"
    }}
  ]
}}

Text to analyze:
{text}

JSON Output:
"""
        return prompt

    def _process_extraction_result(self, result: EntityRelationshipList, chunk: Dict[str, Any]):
        """
        Process the extracted entities and relationships. (No changes needed here)

        Args:
            result: Parsed extraction result
            chunk: Original chunk data
        """
        chunk_id = chunk.get('chunk_id', 'unknown')
        document_id = chunk.get('document_id', 'unknown')
        file_name = chunk.get('file_name', 'unknown')
        page_number = chunk.get('page_num', None)
        chunk_text = chunk.get('text', '')  # Keep original text

        entities_in_chunk = {}  # Temp store for entities in this chunk {temp_id: entity_dict}
        relationships_in_chunk = []
        entity_types_in_chunk = set()  # To track what entity types are in this chunk for tagging

        if result and result.entity_relationship_list:
            for item in result.entity_relationship_list:
                # Validate types before processing
                if not item.from_entity_type or not item.from_entity_name:
                    logger.warning(f"Skipping item with missing from_entity details in chunk {chunk_id}: {item}")
                    continue

                # Create/get source entity
                from_entity_id = self._add_or_get_entity(
                    item.from_entity_name,
                    item.from_entity_type,
                    chunk_id, document_id, file_name, page_number,
                    entities_in_chunk
                )
                entity_types_in_chunk.add(item.from_entity_type)

                # If relationship exists, create/get target entity and add relationship
                if item.relationship_type and item.to_entity_name and item.to_entity_type:
                    to_entity_id = self._add_or_get_entity(
                        item.to_entity_name,
                        item.to_entity_type,
                        chunk_id, document_id, file_name, page_number,
                        entities_in_chunk
                    )
                    entity_types_in_chunk.add(item.to_entity_type)

                    rel_id = str(uuid.uuid4())
                    relationship = {
                        "id": rel_id,
                        "source_entity_id": from_entity_id,
                        "target_entity_id": to_entity_id,
                        "from_entity_id": from_entity_id,  # Add this for graph compatibility
                        "to_entity_id": to_entity_id,  # Add this for graph compatibility
                        "relationship_type": item.relationship_type,  # This field name is needed for graph visualization
                        "type": item.relationship_type,  # Keep the original field too
                        "chunk_id": chunk_id,
                        "document_id": document_id,
                        "file_name": file_name,
                        "page_number": page_number,
                    }
                    relationships_in_chunk.append(relationship)

        # Add entities and relationships from this chunk to the main lists
        self.entities.extend(list(entities_in_chunk.values()))
        self.relationships.extend(relationships_in_chunk)

        # Prepend entity types as tags to the chunk text
        tag_text = f"Tags: {list(entity_types_in_chunk)}" if entity_types_in_chunk else "Tags: []"
        modified_text = f"{tag_text}\n\n{chunk_text}"

        # Store the modified chunk
        modified_chunk_data = chunk.copy()
        modified_chunk_data['text'] = modified_text  # Update text with tags
        modified_chunk_data['original_text'] = chunk_text  # Store original text for rendering

        # Ensure all metadata fields are preserved
        if 'metadata' not in modified_chunk_data:
            modified_chunk_data['metadata'] = {}

        # Make sure file_name is included in metadata
        modified_chunk_data['metadata']['file_name'] = file_name
        modified_chunk_data['file_name'] = file_name  # Duplicate at top level for compatibility

        self.modified_chunks.append(modified_chunk_data)

        logger.info(f"Processed {len(entities_in_chunk)} entities and {len(relationships_in_chunk)} relationships for chunk {chunk_id}")


    def _add_or_get_entity(self, name: str, type: str, chunk_id: str, document_id: str, file_name: str, page_number: Optional[int], entities_in_chunk: Dict[str, Dict]) -> str:
        """
        Adds an entity to the chunk's temporary store or returns existing ID.
        (No changes needed here)
        """
        # Normalize name/type
        name = name.strip()
        type = type.strip().upper()

        # Simple check for existing entity in this chunk's temp store
        for entity_id, entity_data in entities_in_chunk.items():
            if entity_data['name'] == name and entity_data['type'] == type:
                # Update context if needed (e.g., add chunk_id to a list)
                if chunk_id not in entity_data['context']['chunk_ids']:
                     entity_data['context']['chunk_ids'].append(chunk_id)
                return entity_id

        # Create new entity
        entity_id = str(uuid.uuid4())
        entity = {
            "id": entity_id,
            "name": name,
            "type": type,
            "source_document": file_name,  # Add this for explorer compatibility
            "description": f"Found in document: {file_name}",  # Add description
            "context": {
                "chunk_ids": [chunk_id],
                "document_id": document_id,
                "file_name": file_name,
                "page_number": page_number,
            }
        }
        entities_in_chunk[entity_id] = entity
        return entity_id

    def _add_empty_modified_chunk(self, chunk: Dict[str, Any]):
        """
        Adds a chunk to modified_chunks without any new entities/rels.
        (No changes needed here)
        """
        # Ensure essential keys are present even if no extraction happened
        modified_chunk_data = chunk.copy()
        chunk_text = chunk.get('text', '')

        # Add empty tags
        modified_chunk_data['text'] = f"Tags: []\n\n{chunk_text}"
        modified_chunk_data['original_text'] = chunk_text  # Store original text for rendering

        # Preserve all metadata
        file_name = chunk.get('file_name', 'Unknown')

        # Ensure metadata exists
        if 'metadata' not in modified_chunk_data:
            modified_chunk_data['metadata'] = {}

        # Make sure file_name is included in metadata
        modified_chunk_data['metadata']['file_name'] = file_name
        modified_chunk_data['file_name'] = file_name  # Duplicate at top level for compatibility

        self.modified_chunks.append(modified_chunk_data)
        logger.debug(f"Added empty modified chunk for {chunk.get('chunk_id', 'unknown')}")

    def deduplicate_entities(self):
        """
        Deduplicate entities based on name and type similarity.
        (No changes needed here)
        """
        logger.info(f"Starting entity deduplication (threshold: {self.deduplication_threshold})")
        if not self.entities:
            logger.info("No entities to deduplicate.")
            return

        unique_entities = []
        merged_map = {}  # Maps old ID -> new ID
        potential_duplicates = {}  # Store entities by (type, initial_char) for faster comparison

        # Group entities for faster comparison
        for i, entity in enumerate(self.entities):
            key = (entity['type'], entity['name'][0].lower() if entity['name'] else '')
            if key not in potential_duplicates:
                potential_duplicates[key] = []
            potential_duplicates[key].append((i, entity))

        processed_indices = set()

        for key in potential_duplicates:
            group = potential_duplicates[key]
            for i, entity1 in group:
                if i in processed_indices:
                    continue

                current_entity = entity1.copy()  # Start with the first entity
                current_entity_id = current_entity['id']
                merged_ids = {current_entity_id}  # IDs merged into this one

                for j, entity2 in group:
                    if i == j or j in processed_indices:
                        continue

                    # Check similarity only if types match (already guaranteed by grouping)
                    # and names are similar enough
                    similarity = fuzz.token_sort_ratio(entity1['name'], entity2['name'])
                    if similarity >= self.deduplication_threshold:
                        logger.debug(f"Merging entity '{entity2['name']}' ({entity2['id']}) into '{current_entity['name']}' ({current_entity_id}) - Score: {similarity}")

                        # Merge context
                        current_entity['context']['chunk_ids'] = list(set(
                            current_entity['context']['chunk_ids'] + entity2['context']['chunk_ids']
                        ))
                        # Keep the first page number encountered? Or a list? For now, keep first.
                        if current_entity['context']['page_number'] is None and entity2['context']['page_number'] is not None:
                            current_entity['context']['page_number'] = entity2['context']['page_number']

                        merged_map[entity2['id']] = current_entity_id
                        merged_ids.add(entity2['id'])
                        processed_indices.add(j)

                unique_entities.append(current_entity)
                processed_indices.add(i)
                # Ensure the primary entity maps to itself if it wasn't merged into another
                if current_entity_id not in merged_map:
                     merged_map[current_entity_id] = current_entity_id

        original_count = len(self.entities)
        self.entities = unique_entities
        logger.info(f"Entity deduplication complete. Reduced {original_count} to {len(self.entities)} entities.")

        # Update relationships
        self._update_relationships_after_dedup(merged_map)

    def _update_relationships_after_dedup(self, merged_map: Dict[str, str]):
        """
        Updates relationship source/target IDs based on entity merge map.
        (No changes needed here)
        """
        if not self.relationships:
            return

        updated_relationships = []
        updated_count = 0
        for rel in self.relationships:
            original_source_id = rel['source_entity_id']
            original_target_id = rel['target_entity_id']

            # Update source and target IDs if they were merged
            rel['source_entity_id'] = merged_map.get(original_source_id, original_source_id)
            rel['target_entity_id'] = merged_map.get(original_target_id, original_target_id)

            # Add 'from_entity_id' and 'to_entity_id' if missing (for graph compatibility)
            if 'from_entity_id' not in rel:
                 rel['from_entity_id'] = rel['source_entity_id']
            if 'to_entity_id' not in rel:
                 rel['to_entity_id'] = rel['target_entity_id']

            if rel['source_entity_id'] != original_source_id or rel['target_entity_id'] != original_target_id:
                updated_count += 1

            # Keep the relationship
            updated_relationships.append(rel)

        self.relationships = updated_relationships
        logger.info(f"Updated {updated_count} relationships with merged entity IDs.")

    def deduplicate_relationships(self):
        """
        Deduplicate relationships based on source, target, and type.
        (No changes needed here)
        """
        logger.info("Starting relationship deduplication")
        if not self.relationships:
            logger.info("No relationships to deduplicate.")
            return

        unique_relationships = []
        seen_signatures = set()

        for rel in self.relationships:
            # Create a signature for the relationship
            source_id = rel['source_entity_id']
            target_id = rel['target_entity_id']
            # Use 'type' field if exists, otherwise fallback to 'relationship_type'
            rel_type = rel.get('type', rel.get('relationship_type', 'UNKNOWN'))


            # Signature based on source, target, and type
            signature = (source_id, target_id, rel_type)

            if signature not in seen_signatures:
                unique_relationships.append(rel)
                seen_signatures.add(signature)
            else:
                logger.debug(f"Removing duplicate relationship: {signature}")

        original_count = len(self.relationships)
        self.relationships = unique_relationships
        logger.info(f"Relationship deduplication complete. Reduced {original_count} to {len(self.relationships)} relationships.")

    def save_results(self):
        """
        Save extracted entities and relationships to files after deduplication.
        (No changes needed here)
        """
        try:
            # Log start of save process
            logger.info("Starting to save extraction results")

            # Process any remaining chunks before saving
            if self.all_chunks:
                logger.info(f"Processing {len(self.all_chunks)} remaining chunks before saving results")
                self.process_all_chunks()

            # Store counts before deduplication
            original_entity_count = len(self.entities)
            original_relationship_count = len(self.relationships)
            logger.info(f"Before deduplication: {original_entity_count} entities, {original_relationship_count} relationships")

            # Deduplicate entities (updates relationships implicitly)
            logger.info(f"Deduplicating entities with threshold {self.deduplication_threshold}...")
            self.deduplicate_entities()

            # Deduplicate relationships
            logger.info("Deduplicating relationships...")
            self.deduplicate_relationships()

            # Log post-deduplication counts
            deduped_entity_count = len(self.entities)
            deduped_relationship_count = len(self.relationships)
            logger.info(f"After deduplication: {deduped_entity_count} entities, {deduped_relationship_count} relationships")

            # Ensure directory exists
            self.extracted_data_path.mkdir(parents=True, exist_ok=True)

            # Save entities
            entities_file = self.extracted_data_path / "entities.json"
            with open(entities_file, 'w', encoding='utf-8') as f:
                json.dump(self.entities, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self.entities)} entities to {entities_file}")

            # Save relationships
            relationships_file = self.extracted_data_path / "relationships.json"
            with open(relationships_file, 'w', encoding='utf-8') as f:
                json.dump(self.relationships, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self.relationships)} relationships to {relationships_file}")

        except Exception as e:
            logger.error(f"Error saving extraction results: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def get_modified_chunks(self) -> List[Dict[str, Any]]:
        """Returns the list of chunks with tagged entities."""
        return self.modified_chunks

    def clear_results(self):
        """Clears extracted data and modified chunks."""
        logger.info("Clearing extraction results and modified chunks.")
        self.entities = []
        self.relationships = []
        self.modified_chunks = []
        self.all_chunks = []  # Also clear queued chunks
        gc.collect()  # Force garbage collection
