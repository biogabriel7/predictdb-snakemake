#!/usr/bin/env python3
"""
Utility script for formatting and linting Python code.
"""
import subprocess
import sys
import os
import glob

def run_command(cmd, description):
    """
    Run a command and print the output.
    
    Args:
        cmd: Command to run
        description: Description of the command
        
    Returns:
        Boolean indicating if command was successful
    """
    print(f"Running {description}...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"{description} successful!")
            if result.stdout:
                print(result.stdout)
        else:
            print(f"{description} failed:")
            print(result.stderr)
            return False
        return True
    except Exception as e:
        print(f"Error running {description}: {str(e)}")
        return False

def format_python_files(directory):
    """
    Format all Python files in a directory using black, isort, and pylint.
    
    Args:
        directory: Directory containing Python files
        
    Returns:
        Boolean indicating if formatting was successful
    """
    python_files = glob.glob(os.path.join(directory, "**/*.py"), recursive=True)
    
    if not python_files:
        print(f"No Python files found in {directory}")
        return True
    
    print(f"Found {len(python_files)} Python files to format")
    
    # Run isort
    isort_success = run_command(
        ["isort"] + python_files,
        "import sorting (isort)"
    )
    
    # Run black
    black_success = run_command(
        ["black"] + python_files,
        "code formatting (black)"
    )
    
    # Run pylint
    pylint_success = run_command(
        ["pylint"] + python_files,
        "linting (pylint)"
    )
    
    return isort_success and black_success and pylint_success

def main():
    """Main function for command-line execution."""
    # Default directory is scripts/
    if len(sys.argv) < 2:
        directory = "scripts"
    else:
        directory = sys.argv[1]
    
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory")
        return 1
    
    success = format_python_files(directory)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 