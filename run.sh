#!/bin/bash

# PredictDb-Snakemake optimized execution script for MacBook M3 Pro

# Detect available cores (leave 2 for system)
AVAILABLE_CORES=$(sysctl -n hw.ncpu)
USABLE_CORES=$((AVAILABLE_CORES - 2))

# Set maximum number of cores to use
if [ $USABLE_CORES -gt 8 ]; then
    MAX_CORES=8
else
    MAX_CORES=$USABLE_CORES
fi

# Get available memory in GB
TOTAL_MEM_KB=$(sysctl -n hw.memsize | awk '{print $1/1024}')
TOTAL_MEM_GB=$(echo "$TOTAL_MEM_KB/1024/1024" | bc)
# Use 75% of available memory
USABLE_MEM_GB=$(echo "$TOTAL_MEM_GB * 0.75" | bc | awk '{print int($1)}')

echo "System information:"
echo "CPU: $(sysctl -n machdep.cpu.brand_string)"
echo "Cores: $AVAILABLE_CORES (using $MAX_CORES)"
echo "Memory: ${TOTAL_MEM_GB}GB (using ${USABLE_MEM_GB}GB)"
echo

# Execute Snakemake with optimized settings
echo "Starting PredictDb-Snakemake pipeline..."
snakemake \
    --cores $MAX_CORES \
    --resources mem_mb=$(($USABLE_MEM_GB * 1024)) \
    --use-conda \
    --conda-frontend mamba \
    --keep-going \
    --printshellcmds \
    --reason \
    "$@"

# Check execution status
if [ $? -eq 0 ]; then
    echo "Pipeline completed successfully!"
else
    echo "Pipeline execution failed. Check logs for details."
fi