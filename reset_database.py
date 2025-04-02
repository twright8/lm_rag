"""
Reset database script for Anti-Corruption RAG System.

This script will:
1. Delete and recreate the Qdrant collection
2. Delete BM25 indices
3. Delete extracted data files
4. Clear any temporary data

Use this script when you want to completely reset the system.
"""
import sys
import os
from pathlib import Path
import yaml
import shutil

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

# Load configuration
CONFIG_PATH = ROOT_DIR / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

def reset_collection():
    """Delete and recreate the Qdrant collection with proper dimensions."""
    print("\n=== Resetting Vector Collection ===\n")
    
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as rest
        
        # Get Qdrant connection info
        qdrant_host = CONFIG["qdrant"]["host"]
        qdrant_port = CONFIG["qdrant"]["port"]
        qdrant_collection = CONFIG["qdrant"]["collection_name"]
        
        print(f"Connecting to Qdrant at {qdrant_host}:{qdrant_port}")
        
        # Connect to Qdrant
        client = QdrantClient(host=qdrant_host, port=qdrant_port)
        
        # Get collections
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if qdrant_collection in collection_names:
            # Delete collection
            print(f"Deleting collection: {qdrant_collection}")
            client.delete_collection(collection_name=qdrant_collection)
            print("✓ Collection deleted successfully")
            
            # Verify deletion
            collections = client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            if qdrant_collection in collection_names:
                print("! Collection still exists, attempting force recreation")
                try:
                    # Recreate as empty collection
                    client.recreate_collection(
                        collection_name=qdrant_collection,
                        vectors_config={"size": 768, "distance": "Cosine"}
                    )
                    # Delete again
                    client.delete_collection(collection_name=qdrant_collection)
                    print("✓ Collection recreated and deleted successfully")
                except Exception as e:
                    print(f"✗ Error recreating collection: {e}")
        else:
            print(f"Collection {qdrant_collection} does not exist, nothing to delete")
            
        # Create new collection with correct dimensions for E5 embedding model
        print(f"\nCreating new collection: {qdrant_collection} with vector size: 768")
        try:
            client.create_collection(
                collection_name=qdrant_collection,
                vectors_config=rest.VectorParams(
                    size=768,  # E5 model's dimension
                    distance=rest.Distance.COSINE
                )
            )
            print("✓ Collection created successfully with vector size: 768")
            
            # Verify creation
            collection_info = client.get_collection(collection_name=qdrant_collection)
            vector_size = collection_info.config.params.vectors.size
            print(f"✓ Verified collection vector size: {vector_size}")
        except Exception as e:
            print(f"✗ Error creating collection: {e}")
            
    except Exception as e:
        print(f"✗ Error deleting collection: {e}")
        import traceback
        print(traceback.format_exc())

def delete_bm25_indices():
    """Delete BM25 indices."""
    print("\n=== Deleting BM25 Indices ===\n")
    
    # Get BM25 path
    bm25_path = ROOT_DIR / CONFIG["storage"]["bm25_index_path"]
    bm25_dir = bm25_path.parent
    
    print(f"BM25 path: {bm25_path}")
    
    try:
        # Delete index file
        if bm25_path.exists():
            os.remove(bm25_path)
            print(f"✓ Deleted BM25 index: {bm25_path}")
        else:
            print(f"BM25 index does not exist: {bm25_path}")
            
        # Delete stopwords file
        stopwords_path = bm25_dir / "stopwords.pkl"
        if stopwords_path.exists():
            os.remove(stopwords_path)
            print(f"✓ Deleted stopwords file: {stopwords_path}")
        else:
            print(f"Stopwords file does not exist: {stopwords_path}")
            
    except Exception as e:
        print(f"✗ Error deleting BM25 indices: {e}")

def delete_extracted_data():
    """Delete extracted entity and relationship data."""
    print("\n=== Deleting Extracted Data ===\n")
    
    # Get extracted data path
    extracted_data_path = ROOT_DIR / CONFIG["storage"]["extracted_data_path"]
    
    print(f"Extracted data path: {extracted_data_path}")
    
    try:
        # Delete entities.json
        entities_path = extracted_data_path / "entities.json"
        if entities_path.exists():
            os.remove(entities_path)
            print(f"✓ Deleted entities file: {entities_path}")
        else:
            print(f"Entities file does not exist: {entities_path}")
            
        # Delete relationships.json
        relationships_path = extracted_data_path / "relationships.json"
        if relationships_path.exists():
            os.remove(relationships_path)
            print(f"✓ Deleted relationships file: {relationships_path}")
        else:
            print(f"Relationships file does not exist: {relationships_path}")
            
    except Exception as e:
        print(f"✗ Error deleting extracted data: {e}")

def clear_temp_files():
    """Clear temporary files."""
    print("\n=== Clearing Temporary Files ===\n")
    
    # Get temp directory
    temp_dir = ROOT_DIR / "temp"
    
    print(f"Temp directory: {temp_dir}")
    
    try:
        if temp_dir.exists():
            # Delete all files in temp directory
            for file_path in temp_dir.iterdir():
                if file_path.is_file():
                    os.remove(file_path)
                    print(f"✓ Deleted temp file: {file_path}")
            print("✓ Cleared temp directory")
        else:
            print("Temp directory does not exist")
            
    except Exception as e:
        print(f"✗ Error clearing temp files: {e}")

def reset_database():
    """Reset the entire database."""
    print("\n" + "="*50)
    print("     ANTI-CORRUPTION RAG SYSTEM RESET")
    print("="*50 + "\n")
    
    print("This will delete all data and reset the system to a clean state.")
    confirm = input("Are you sure you want to continue? (y/n): ")
    
    if confirm.lower() != 'y':
        print("Reset cancelled.")
        return
    
    # Reset Qdrant collection (delete and recreate)
    reset_collection()
    
    # Delete BM25 indices
    delete_bm25_indices()
    
    # Delete extracted data
    delete_extracted_data()
    
    # Clear temp files
    clear_temp_files()
    
    print("\n" + "="*50)
    print("     RESET COMPLETE")
    print("="*50 + "\n")
    
    print("The system has been reset to a clean state.")
    print("You can now restart the application.")

if __name__ == "__main__":
    reset_database()