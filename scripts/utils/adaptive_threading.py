"""
Utility module for adaptive threading based on system load.
"""
import os
import multiprocessing
import psutil
import time
import math
from .logger import setup_logger

logger = setup_logger(__name__)

def get_optimal_threads(min_threads=1, max_threads=None, target_cpu_percent=75):
    """
    Determine optimal number of threads based on current system load.
    
    Args:
        min_threads: Minimum number of threads to use
        max_threads: Maximum number of threads to use (defaults to number of CPU cores)
        target_cpu_percent: Target CPU utilization percentage
    
    Returns:
        Optimal number of threads to use
    """
    # Get total CPU cores
    total_cpus = multiprocessing.cpu_count()
    
    # Default max_threads to total CPUs if not specified
    if max_threads is None:
        max_threads = total_cpus
    
    # Get current CPU utilization (average across all cores)
    current_cpu_percent = psutil.cpu_percent(interval=0.1)
    
    # Calculate available CPU capacity
    available_percent = max(0, target_cpu_percent - current_cpu_percent)
    
    # Calculate optimal threads
    if current_cpu_percent >= target_cpu_percent:
        # System is already heavily loaded, use minimum
        optimal_threads = min_threads
    else:
        # Allocate threads based on available capacity
        optimal_threads = int((available_percent / 100) * total_cpus)
        
        # Ensure we're within min/max bounds
        optimal_threads = max(min_threads, min(optimal_threads, max_threads))
    
    logger.debug(f"Current CPU: {current_cpu_percent}%, Target: {target_cpu_percent}%, "
                f"Optimal threads: {optimal_threads}")
    
    return optimal_threads

class AdaptiveThreadPool:
    """Thread pool that dynamically adjusts its size based on system load."""
    
    def __init__(self, min_threads=1, max_threads=None, target_cpu_percent=75, 
                 adjustment_interval=30):
        """
        Initialize adaptive thread pool.
        
        Args:
            min_threads: Minimum number of threads to use
            max_threads: Maximum number of threads to use
            target_cpu_percent: Target CPU utilization percentage
            adjustment_interval: How often to adjust thread count (in seconds)
        """
        self.min_threads = min_threads
        self.max_threads = max_threads or multiprocessing.cpu_count()
        self.target_cpu_percent = target_cpu_percent
        self.adjustment_interval = adjustment_interval
        
        # Start with median number of threads
        self.current_threads = (self.min_threads + self.max_threads) // 2
        
        self.pool = multiprocessing.Pool(processes=self.current_threads)
        self.last_adjustment_time = time.time()
        
        logger.info(f"Created adaptive thread pool with {self.current_threads} threads "
                   f"(min: {min_threads}, max: {self.max_threads})")
    
    def map(self, func, iterable, chunksize=1):
        """
        Map function to iterable using thread pool.
        
        Args:
            func: Function to apply to each item
            iterable: Iterable of items to process
            chunksize: Chunk size for multiprocessing
            
        Returns:
            Results of function application
        """
        # Check if we need to adjust thread count
        if time.time() - self.last_adjustment_time > self.adjustment_interval:
            self._adjust_thread_count()
        
        return self.pool.map(func, iterable, chunksize=chunksize)
    
    def _adjust_thread_count(self):
        """Adjust thread count based on current system load."""
        optimal_threads = get_optimal_threads(
            min_threads=self.min_threads,
            max_threads=self.max_threads,
            target_cpu_percent=self.target_cpu_percent
        )
        
        # Only recreate pool if thread count changes significantly (>= 2 threads difference)
        if abs(optimal_threads - self.current_threads) >= 2:
            logger.info(f"Adjusting thread count from {self.current_threads} to {optimal_threads}")
            
            # Recreate pool with new thread count
            self.pool.close()
            self.pool.join()
            self.current_threads = optimal_threads
            self.pool = multiprocessing.Pool(processes=self.current_threads)
        
        self.last_adjustment_time = time.time()
    
    def close(self):
        """Close thread pool."""
        self.pool.close()
        self.pool.join()
        logger.debug("Closed adaptive thread pool") 