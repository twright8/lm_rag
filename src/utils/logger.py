"""
Logging utilities for the Anti-Corruption RAG System.
"""
import os
import sys
import logging
import yaml
from pathlib import Path
from datetime import datetime

# Get the root directory
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Load configuration if available
CONFIG_PATH = ROOT_DIR / "config.yaml"
CONFIG = {}
if CONFIG_PATH.exists():
    try:
        with open(CONFIG_PATH, "r") as f:
            CONFIG = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")

# Fallback logging configuration
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FILE = "anti_corruption_rag.log"

# Get logging settings from config
LOG_LEVEL = CONFIG.get("logging", {}).get("level", DEFAULT_LOG_LEVEL)
LOG_FILE = CONFIG.get("logging", {}).get("log_file", DEFAULT_LOG_FILE)
CONSOLE_LEVEL = CONFIG.get("logging", {}).get("console_level", DEFAULT_LOG_LEVEL)

# Create logs directory if it doesn't exist
LOGS_DIR = ROOT_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True, parents=True)

# Convert log level string to logging level
def _get_log_level(level_str):
    """
    Convert a string log level to a logging level constant.
    
    Args:
        level_str (str): Log level string
        
    Returns:
        int: Logging level constant
    """
    levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    return levels.get(level_str.upper(), logging.INFO)

# Set up logger
def setup_logger(name, level=None):
    """
    Set up a logger with file and console handlers.
    
    Args:
        name (str): Logger name
        level (str, optional): Log level. Defaults to config value.
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Use provided level or fall back to config/default
    log_level = _get_log_level(level) if level else _get_log_level(LOG_LEVEL)
    console_level = _get_log_level(CONSOLE_LEVEL)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Clear any existing handlers
    logger.handlers = []
    
    # Add timestamp to log file name
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file_path = LOGS_DIR / f"{timestamp}_{LOG_FILE}"
    
    # Create file handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(log_level)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_format = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
