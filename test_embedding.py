"""
Embedding model test script for Anti-Corruption RAG System.
This script tests the embedding models to ensure they work correctly.
"""
import sys
import os
from pathlib import Path
import yaml
import time
import json
import torch
import numpy as np

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

# Load configuration
CONFIG_PATH = ROOT_DIR / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

def test_embedding_model():
    """
    Test the embedding model to ensure it works correctly.
    """
    print("\n=== Testing Embedding Model ===\n")
    
    # Get model name from config
    embedding_model_name = CONFIG["models"]["embedding_model"]
    print(f"Using embedding model: {embedding_model_name}")
    
    # Test text
    test_sentences = [
        "This is a test sentence.",
        "Another sentence to test embeddings."
    ]
    
    try:
        # Import BatchedInference
        from embed import BatchedInference
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Load embedding model
        print("Loading embedding model...")
        start_time = time.time()
        embed_register = BatchedInference(
            model_id=embedding_model_name,
            engine="torch",
            device=device
        )
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")
        
        # Generate embeddings
        print("\nGenerating embeddings for test sentences...")
        start_time = time.time()
        future = embed_register.embed(
            sentences=test_sentences,
            model_id=embedding_model_name
        )
        
        # Get result
        result = future.result()
        embed_time = time.time() - start_time
        print(f"Embeddings generated in {embed_time:.2f} seconds")
        
        # Analyze the result
        print("\n=== Embedding Result Analysis ===\n")
        
        # Check if result is a tuple (common with infinity embedding)
        if isinstance(result, tuple):
            print(f"Result is a tuple with {len(result)} elements")
            print(f"First element type: {type(result[0])}")
            embeddings = result[0]
            
            # Check if second element is token usage
            if len(result) > 1:
                print(f"Token usage: {result[1]}")
        else:
            print(f"Result is a {type(result)}")
            embeddings = result
        
        # Analyze the embeddings
        if not embeddings:
            print("ERROR: No embeddings returned")
            return
            
        print(f"Got {len(embeddings)} embeddings")
        
        # Check the first embedding
        first_embedding = embeddings[0]
        print(f"First embedding type: {type(first_embedding)}")
        
        if hasattr(first_embedding, 'shape'):
            print(f"Shape: {first_embedding.shape}")
            
        if hasattr(first_embedding, 'tolist'):
            embedding_list = first_embedding.tolist()
            print(f"Length as list: {len(embedding_list)}")
            print(f"First 5 values: {embedding_list[:5]}")
        elif isinstance(first_embedding, list):
            print(f"Length: {len(first_embedding)}")
            print(f"First 5 values: {first_embedding[:5]}")
        else:
            print(f"Unknown format. Value: {str(first_embedding)[:100]}")
        
        # Test vector operations
        print("\n=== Testing Vector Operations ===\n")
        
        # Convert to common format
        def get_vector(emb):
            if hasattr(emb, 'tolist'):
                return emb.tolist()
            return emb if isinstance(emb, list) else None
        
        if len(embeddings) >= 2:
            vec1 = get_vector(embeddings[0])
            vec2 = get_vector(embeddings[1])
            
            if vec1 and vec2 and len(vec1) == len(vec2):
                # Calculate cosine similarity
                from numpy import dot
                from numpy.linalg import norm
                
                # Convert to numpy arrays
                a = np.array(vec1)
                b = np.array(vec2)
                
                # Calculate similarity
                similarity = dot(a, b)/(norm(a)*norm(b))
                print(f"Cosine similarity between test sentences: {similarity:.4f}")
            else:
                print("Could not calculate similarity - vectors invalid or different lengths")
        
        # Success
        print("\n=== TEST PASSED ===")
        print("Embedding model is working correctly!")
        
        # Cleanup
        embed_register.stop()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"\n=== TEST FAILED ===")
        print(f"Error: {e}")
        import traceback
        print(traceback.format_exc())
        
        # Provide troubleshooting advice
        print("\nTroubleshooting advice:")
        print("1. Check if the model name is correct in config.yaml")
        print("2. Ensure you have enough memory (RAM and VRAM)")
        print("3. Check your internet connection if downloading model")
        print("4. Try a smaller model like 'intfloat/multilingual-e5-small'")

if __name__ == "__main__":
    test_embedding_model()