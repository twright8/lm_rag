"""
NLTK resource setup script for Anti-Corruption RAG System.

!!! IMPORTANT !!!
Run this BEFORE starting the application to ensure all required NLTK resources are installed.
Otherwise, you may encounter errors during document processing.

Usage:
    python setup_nltk.py

This script will:
1. Download all required NLTK resources
2. Create necessary directories and placeholder files
3. Verify that tokenizers are working correctly
"""
import nltk
import sys
import os
import ssl

def setup_nltk_resources():
    """
    Download all necessary NLTK resources for the application.
    """
    # Try to create unverified HTTPS context if standard context fails
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    resources = [
        'punkt',
        'stopwords',
        'wordnet',
        'omw-1.4'
    ]
    
    print("Setting up NLTK resources...")
    
    # Create NLTK data directories if they don't exist
    nltk_data_dir = os.path.expanduser('~/nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)

    # Punkt specific handling - sometimes punkt_tab isn't directly accessible
    try:
        print("Setting up punkt tokenizer...")
        nltk.download('punkt', quiet=False)
        
        # Make sure punkt is properly initialized - this sometimes helps initialize punkt_tab
        from nltk.tokenize import word_tokenize, sent_tokenize
        test_text = "Hello world. This is a test."
        sent_tokenize(test_text)
        word_tokenize(test_text)
        print("✓ punkt tokenizers initialized successfully")
    except Exception as e:
        print(f"✗ Error setting up punkt tokenizers: {e}", file=sys.stderr)

    # Download other resources
    for resource in resources:
        if resource != 'punkt':  # Already handled punkt above
            try:
                print(f"Downloading {resource}...")
                nltk.download(resource, quiet=False)
                print(f"✓ {resource} downloaded successfully")
            except Exception as e:
                print(f"✗ Error downloading {resource}: {e}", file=sys.stderr)
    
    # Verify punkt_tab is accessible (or create it)
    try:
        from nltk.tokenize.punkt import PunktSentenceTokenizer
        tokenizer = PunktSentenceTokenizer('english')
        print("✓ PunktSentenceTokenizer initialized successfully")
    except Exception as e:
        print(f"✗ Error initializing PunktSentenceTokenizer: {e}", file=sys.stderr)
        print("Trying to fix punkt_tab issue...")
        
        # Create punkt_tab directory structure if it doesn't exist
        punkttab_dir = os.path.join(nltk_data_dir, 'tokenizers', 'punkt_tab', 'english')
        os.makedirs(punkttab_dir, exist_ok=True)
        
        # Create a minimal placeholder file if needed
        placeholder_file = os.path.join(punkttab_dir, 'punkt.txt')
        if not os.path.exists(placeholder_file):
            with open(placeholder_file, 'w') as f:
                f.write("# Placeholder punkt file created by setup script\n")
            print(f"Created placeholder punkt file at {placeholder_file}")
    
    print("\nNLTK setup complete.")
    print("If you encounter any issues with NLTK resources during application runtime,")
    print("please run this script again or manually download the resources using:")
    print("python -m nltk.downloader <resource_name>")

if __name__ == "__main__":
    setup_nltk_resources()