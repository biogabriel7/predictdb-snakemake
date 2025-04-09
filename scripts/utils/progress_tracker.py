"""
Utility module for tracking progress of long-running operations.
"""
import time
import threading
import datetime
import os
from .logger import setup_logger

logger = setup_logger(__name__)

class ProgressTracker:
    """Track progress of long-running operations with periodic updates."""
    
    def __init__(self, total_items, description="Processing", update_interval=10):
        """
        Initialize progress tracker.
        
        Args:
            total_items: Total number of items to process
            description: Description of the operation
            update_interval: How often to log progress (in seconds)
        """
        self.total_items = total_items
        self.description = description
        self.update_interval = update_interval
        self.processed_items = 0
        self.start_time = None
        self._stop_event = threading.Event()
        self._thread = None
    
    def start(self):
        """Start tracking progress."""
        self.start_time = time.time()
        self.processed_items = 0
        logger.info(f"Started {self.description}: 0/{self.total_items} (0.0%)")
        
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._update_progress_thread)
        self._thread.daemon = True
        self._thread.start()
    
    def update(self, processed_items=None, increment=None):
        """
        Update progress.
        
        Args:
            processed_items: Total items processed so far (absolute)
            increment: Number of items processed since last update (relative)
        """
        if processed_items is not None:
            self.processed_items = processed_items
        elif increment is not None:
            self.processed_items += increment
    
    def _update_progress_thread(self):
        """Thread that periodically logs progress updates."""
        while not self._stop_event.is_set():
            # Sleep for the update interval
            self._stop_event.wait(self.update_interval)
            
            # Log progress if the thread is still running
            if not self._stop_event.is_set():
                self._log_progress()
    
    def _log_progress(self):
        """Log current progress."""
        if self.processed_items > 0 and self.total_items > 0:
            # Calculate progress percentage
            percentage = (self.processed_items / self.total_items) * 100
            
            # Calculate elapsed time
            elapsed_seconds = time.time() - self.start_time
            elapsed_time = str(datetime.timedelta(seconds=int(elapsed_seconds)))
            
            # Calculate estimated time remaining
            if percentage > 0:
                total_seconds = elapsed_seconds / (percentage / 100)
                remaining_seconds = total_seconds - elapsed_seconds
                remaining_time = str(datetime.timedelta(seconds=int(remaining_seconds)))
            else:
                remaining_time = "unknown"
            
            # Log progress
            logger.info(f"{self.description}: "
                       f"{self.processed_items}/{self.total_items} "
                       f"({percentage:.1f}%) - "
                       f"Elapsed: {elapsed_time}, Remaining: {remaining_time}")
    
    def finish(self):
        """Finish tracking progress."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        
        elapsed_seconds = time.time() - self.start_time
        elapsed_time = str(datetime.timedelta(seconds=int(elapsed_seconds)))
        
        logger.info(f"Finished {self.description}: "
                   f"{self.processed_items}/{self.total_items} "
                   f"({(self.processed_items / self.total_items) * 100:.1f}%) - "
                   f"Total time: {elapsed_time}") 