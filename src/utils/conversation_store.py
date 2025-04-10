"""
Conversation storage utilities for the Anti-Corruption RAG System.
Manages persistent storage of conversations as JSON files.
"""
import json
import os
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional

from src.utils.logger import setup_logger

# Initialize logger
logger = setup_logger(__name__)

class ConversationStore:
    """Manages persistent storage of conversations as JSON files."""

    def __init__(self, root_dir: Path):
        """Initialize conversation storage directory."""
        self.root_dir = root_dir
        self.conversations_dir = root_dir / "data" / "conversations"
        try:
            self.conversations_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Conversation storage initialized at: {self.conversations_dir}")
        except OSError as e:
            logger.error(f"Failed to create conversation directory {self.conversations_dir}: {e}", exc_info=True)
            # Consider raising an exception or handling this more gracefully
        self.conversations: Dict[str, Dict[str, Any]] = {}
        self.load_conversations() # Load existing conversations on init

    def load_conversations(self) -> int:
        """Load all saved .json conversation files from disk into memory."""
        loaded_count = 0
        logger.info(f"Loading conversations from: {self.conversations_dir}")
        if not self.conversations_dir.exists():
            logger.warning("Conversations directory does not exist. No conversations loaded.")
            return 0
        try:
            conversation_files = list(self.conversations_dir.glob("*.json"))
            for file_path in conversation_files:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        conversation_data = json.load(f)
                        conversation_id = file_path.stem
                        # Basic validation
                        if isinstance(conversation_data, dict) and "id" in conversation_data:
                            self.conversations[conversation_id] = conversation_data
                            loaded_count += 1
                        else:
                            logger.warning(f"Skipping invalid conversation file (not a dict or missing 'id'): {file_path}")
                except json.JSONDecodeError:
                    logger.error(f"Error decoding JSON from conversation file: {file_path}", exc_info=True)
                except Exception as e:
                    logger.error(f"Error loading conversation file {file_path}: {e}", exc_info=True)
            logger.info(f"Successfully loaded {loaded_count} conversations into memory.")
            return loaded_count
        except Exception as e:
            logger.error(f"Error listing or loading conversations from {self.conversations_dir}: {e}", exc_info=True)
            return 0

    def save_conversation(self, conversation_id: str, data: Dict[str, Any]) -> bool:
        """Save a single conversation dictionary to a JSON file."""
        if not isinstance(data, dict):
            logger.error(f"Save failed: Input data for conversation {conversation_id} is not a dictionary.")
            return False
        if not self.conversations_dir.exists():
             logger.error(f"Save failed: Conversations directory {self.conversations_dir} does not exist.")
             return False

        try:
            # Ensure essential keys exist and update timestamp
            data["last_updated"] = time.time()
            data.setdefault("id", conversation_id)
            data.setdefault("title", f"Conversation {conversation_id[:8]}") # Default title if missing
            data.setdefault("created_at", data["last_updated"])
            data.setdefault("messages", []) # Ensure messages list exists

            file_path = self.conversations_dir / f"{conversation_id}.json"
            tmp_file_path = file_path.with_suffix(".json.tmp")

            # Write to temporary file first
            with open(tmp_file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # Atomically replace original file with temporary file
            os.replace(tmp_file_path, file_path)

            # Update in-memory store
            self.conversations[conversation_id] = data
            logger.debug(f"Successfully saved conversation {conversation_id} to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving conversation {conversation_id} to file: {e}", exc_info=True)
            # Clean up temporary file if it exists on error
            if 'tmp_file_path' in locals() and tmp_file_path.exists():
                 try: tmp_file_path.unlink()
                 except OSError as unlink_e: logger.error(f"Error deleting temp file {tmp_file_path}: {unlink_e}")
            return False

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a conversation dictionary from the in-memory store."""
        conv = self.conversations.get(conversation_id)
        if conv:
            logger.debug(f"Retrieved conversation {conversation_id} from memory.")
        else:
            logger.warning(f"Conversation {conversation_id} not found in memory.")
        return conv # Returns None if not found

    def list_conversations(self) -> List[Dict[str, Any]]:
        """Generate a list of conversation metadata for UI display."""
        conversation_list = []
        for conv_id, data in self.conversations.items():
            conversation_list.append({
                "id": conv_id,
                "title": data.get("title", "Untitled Conversation"),
                "created_at": data.get("created_at", 0),
                "last_updated": data.get("last_updated", 0),
                "message_count": len(data.get("messages", [])),
            })
        # Sort by last updated timestamp, newest first
        conversation_list.sort(key=lambda x: x["last_updated"], reverse=True)
        logger.debug(f"Generated list of {len(conversation_list)} conversations for UI.")
        return conversation_list

    def create_conversation(self, title: str = "New Conversation") -> Optional[str]:
        """Create a new conversation entry and save it immediately."""
        conversation_id = str(uuid.uuid4())
        timestamp = time.time()
        conversation_data = {
            "id": conversation_id,
            "title": title.strip() if title else "New Conversation",
            "created_at": timestamp,
            "last_updated": timestamp,
            "messages": [],
        }
        if self.save_conversation(conversation_id, conversation_data):
            logger.info(f"Created and saved new conversation {conversation_id} with title '{conversation_data['title']}'")
            return conversation_id
        else:
            logger.error(f"Failed to save newly created conversation {conversation_id}")
            return None

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation from disk and remove it from the in-memory store."""
        logger.warning(f"Attempting to delete conversation {conversation_id}")
        deleted_from_disk = False
        deleted_from_memory = False
        try:
            file_path = self.conversations_dir / f"{conversation_id}.json"
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted conversation file: {file_path}")
                deleted_from_disk = True
            else:
                logger.warning(f"Conversation file not found for deletion: {file_path}")
                # Still proceed to remove from memory if it exists there

            if conversation_id in self.conversations:
                del self.conversations[conversation_id]
                logger.info(f"Removed conversation {conversation_id} from memory cache.")
                deleted_from_memory = True

            return deleted_from_disk or deleted_from_memory # Return True if either succeeded
        except Exception as e:
            logger.error(f"Error during deletion of conversation {conversation_id}: {e}", exc_info=True)
            return False

    def clear_all_conversations(self) -> bool:
        """Delete all .json files in the conversations directory and clear the in-memory store."""
        logger.warning("Attempting to clear ALL conversations.")
        all_files_cleared = True
        if not self.conversations_dir.exists():
            logger.warning("Conversation directory doesn't exist, nothing to clear.")
            self.conversations = {}
            return True
        try:
            conversation_files = list(self.conversations_dir.glob("*.json"))
            logger.info(f"Found {len(conversation_files)} conversation files to delete.")
            for file_path in conversation_files:
                try:
                    file_path.unlink()
                except Exception as e:
                    logger.error(f"Error deleting conversation file {file_path}: {e}", exc_info=True)
                    all_files_cleared = False # Mark failure if any file deletion fails

            self.conversations = {} # Clear memory regardless of file deletion success/failure
            logger.info("Cleared in-memory conversation store.")
            return all_files_cleared
        except Exception as e:
            logger.error(f"Error clearing conversations directory {self.conversations_dir}: {e}", exc_info=True)
            self.conversations = {} # Attempt to clear memory even on directory error
            return False