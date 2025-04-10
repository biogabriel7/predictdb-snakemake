#!/bin/bash
# PredictDb-Snakemake Container Build Script

# Parse command line arguments
SINGULARITY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --singularity)
            SINGULARITY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--singularity]"
            exit 1
            ;;
    esac
done

# Build Docker container
echo "Building Docker container..."
docker build -t predictdb-snakemake .

# Check if Docker build was successful
if [ $? -ne 0 ]; then
    echo "Docker build failed."
    exit 1
fi

echo "Docker container built successfully: predictdb-snakemake"

# Optionally build Singularity container
if [ "$SINGULARITY" = true ]; then
    echo "Building Singularity container..."
    
    # Check if singularity is installed
    if ! command -v singularity &> /dev/null; then
        echo "Error: Singularity is not installed or not in PATH."
        exit 1
    fi
    
    singularity build predictdb-snakemake.sif docker-daemon://predictdb-snakemake:latest
    
    if [ $? -ne 0 ]; then
        echo "Singularity build failed."
        exit 1
    fi
    
    echo "Singularity container built successfully: predictdb-snakemake.sif"
fi

echo "Container build process complete."

# Make sure you're in the project directory (where environment.yaml is located)
conda env update -f environment.yaml --prune 