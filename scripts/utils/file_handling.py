"""
Utility module for file handling operations.
"""
import gzip
import os

def is_gzipped(filename):
    """Check if a file is gzipped by examining its magic number."""
    try:
        with open(filename, 'rb') as f:
            return f.read(2) == b'\x1f\x8b'
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filename}")
    except Exception as e:
        raise Exception(f"Error checking if file is gzipped: {str(e)}")

def open_file(filename, mode='r'):
    """
    Open a file, handling gzipped files if needed.
    
    Args:
        filename: Path to the file to open
        mode: File mode ('r', 'w', 'a', 'rb', 'wb', 'ab')
    
    Returns:
        File handle for reading or writing
    """
    if not os.path.exists(filename) and 'r' in mode:
        raise FileNotFoundError(f"File not found: {filename}")
    
    # Check if the file is gzipped only when reading or appending
    is_gz = False
    if os.path.exists(filename) and ('r' in mode or 'a' in mode):
        is_gz = is_gzipped(filename)
    elif 'w' in mode and filename.endswith('.gz'):
        is_gz = True
    
    if is_gz and 'b' not in mode:
        return gzip.open(filename, mode + 't')
    elif is_gz and 'b' in mode:
        return gzip.open(filename, mode)
    else:
        return open(filename, mode) 