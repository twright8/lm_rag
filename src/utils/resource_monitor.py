"""
Resource monitoring utilities for tracking GPU/CPU usage.
"""
import os
import gc
import psutil
from typing import Dict, Any, Optional

def get_process_memory() -> Dict[str, float]:
    """
    Get current process memory usage.
    
    Returns:
        dict: Memory usage information in GB
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return {
        'rss': mem_info.rss / (1024 ** 3),  # Resident Set Size in GB
        'vms': mem_info.vms / (1024 ** 3),  # Virtual Memory Size in GB
    }

def get_gpu_info() -> Optional[Dict[str, Any]]:
    """
    Get GPU information.
    
    Returns:
        dict: GPU information or None if no GPU is available
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        
        gpu_id = 0  # Using the first GPU by default
        
        # Get GPU properties
        props = torch.cuda.get_device_properties(gpu_id)
        
        # Get memory usage
        memory_allocated = torch.cuda.memory_allocated(gpu_id) / (1024 ** 3)  # GB
        memory_reserved = torch.cuda.memory_reserved(gpu_id) / (1024 ** 3)  # GB
        memory_total = props.total_memory / (1024 ** 3)  # GB
        
        return {
            'name': props.name,
            'memory_allocated': memory_allocated,
            'memory_reserved': memory_reserved,
            'memory_total': memory_total,
            'memory_free': memory_total - memory_allocated,
            'compute_capability': f"{props.major}.{props.minor}"
        }
    except ImportError:
        return None
    except Exception as e:
        print(f"Error getting GPU info: {e}")
        return None

def log_memory_usage(logger, unused_string: str = None) -> None:
    """
    Log memory usage.

    Args:
        logger: Logger instance.
        unused_string (str, optional): An optional string argument that is not used
                                       by the function. Defaults to None.
    """
    # Log process memory
    process_memory = get_process_memory()
    logger.info(f"Process memory - RSS: {process_memory['rss']:.2f} GB, VMS: {process_memory['vms']:.2f} GB")

    # Log GPU memory if available
    gpu_info = get_gpu_info()
    if gpu_info:
        logger.info(f"GPU memory - Allocated: {gpu_info['memory_allocated']:.2f} GB, "
                   f"Reserved: {gpu_info['memory_reserved']:.2f} GB, "
                   f"Total: {gpu_info['memory_total']:.2f} GB")

def force_gc() -> None:
    """
    Force garbage collection and CUDA cache clearing.
    """
    # Run Python's garbage collector
    gc.collect()
    
    # Clear CUDA cache if available
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
