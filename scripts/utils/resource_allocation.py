"""
Utility module for dynamic resource allocation.
"""
import psutil
import math
import os

def get_memory_mb(fraction=0.75, min_memory=1000, max_memory=None):
    """
    Calculate available memory in MB based on system state.
    
    Args:
        fraction: Fraction of total memory to allocate
        min_memory: Minimum memory to allocate in MB
        max_memory: Maximum memory to allocate in MB
    
    Returns:
        Available memory in MB
    """
    total_mem = psutil.virtual_memory().total / (1024 * 1024)  # Convert to MB
    available_mem = total_mem * fraction
    
    if max_memory:
        available_mem = min(available_mem, max_memory)
    
    return max(min_memory, int(available_mem))

def get_threads(fraction=0.75, min_threads=1, max_threads=None):
    """
    Calculate available CPU threads based on system state.
    
    Args:
        fraction: Fraction of total CPUs to allocate
        min_threads: Minimum threads to allocate
        max_threads: Maximum threads to allocate
    
    Returns:
        Available threads
    """
    total_cpus = psutil.cpu_count(logical=True)
    available_cpus = math.floor(total_cpus * fraction)
    
    if max_threads:
        available_cpus = min(available_cpus, max_threads)
    
    return max(min_threads, available_cpus) 