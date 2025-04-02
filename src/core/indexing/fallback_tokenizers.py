"""
Fallback tokenization implementations for when NLTK resources are unavailable.
"""
import re
import logging

logger = logging.getLogger(__name__)

def simple_sent_tokenize(text):
    """
    Simple sentence tokenizer that splits on common sentence ending punctuation.
    
    Args:
        text (str): Text to tokenize
    
    Returns:
        list: List of sentences
    """
    # Pattern to split sentences (handles common punctuation cases)
    pattern = r'(?<=[.!?])\s+'
    sentences = re.split(pattern, text)
    
    # Filter out empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    logger.info(f"Fallback sentence tokenizer created {len(sentences)} sentences")
    return sentences

def simple_word_tokenize(text):
    """
    Simple word tokenizer that splits on whitespace and punctuation.
    
    Args:
        text (str): Text to tokenize
    
    Returns:
        list: List of tokens
    """
    # Remove punctuation that shouldn't be separated (apostrophes, hyphens in compound words)
    text = re.sub(r'(\w)\'(\w)', r'\1APOSTROPHE\2', text)
    text = re.sub(r'(\w)-(\w)', r'\1HYPHEN\2', text)
    
    # Split on whitespace and punctuation
    tokens = re.findall(r'\w+', text)
    
    # Restore apostrophes and hyphens
    tokens = [token.replace('APOSTROPHE', "'").replace('HYPHEN', '-') for token in tokens]
    
    logger.info(f"Fallback word tokenizer created {len(tokens)} tokens")
    return tokens