#!/bin/bash

# PredictDb-Snakemake optimized execution script

# Parse command line arguments
USE_CONTAINER=false
CONTAINER_CMD="docker"
ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --container)
            USE_CONTAINER=true
            shift
            ;;
        --container-cmd)
            CONTAINER_CMD="$2"
            shift 2
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

# Check for resume mode
check_resume() {
    if [ -d "results/checkpoints" ] && [ "$(ls -A results/checkpoints)" ]; then
        echo "Found checkpoints from previous run. Resuming pipeline..."
        RESUME_ARG="--rerun-incomplete"
    else
        echo "Starting new pipeline run..."
        mkdir -p results/checkpoints
        RESUME_ARG=""
    fi
    echo $RESUME_ARG
}

RESUME_ARG=$(check_resume)

# Detect available cores (leave 2 for system)
if [ "$(uname)" == "Darwin" ]; then
    # macOS
    AVAILABLE_CORES=$(sysctl -n hw.ncpu)
else
    # Linux
    AVAILABLE_CORES=$(nproc)
fi
USABLE_CORES=$((AVAILABLE_CORES - 2))

# Set maximum number of cores to use
if [ $USABLE_CORES -gt 8 ]; then
    MAX_CORES=8
else
    MAX_CORES=$USABLE_CORES
fi

# Get available memory in GB
if [ "$(uname)" == "Darwin" ]; then
    # macOS
    TOTAL_MEM_KB=$(sysctl -n hw.memsize | awk '{print $1/1024}')
else
    # Linux
    TOTAL_MEM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
fi
TOTAL_MEM_GB=$(echo "$TOTAL_MEM_KB/1024/1024" | bc)
# Use 75% of available memory
USABLE_MEM_GB=$(echo "$TOTAL_MEM_GB * 0.75" | bc | awk '{print int($1)}')

echo "System information:"
if [ "$(uname)" == "Darwin" ]; then
    echo "CPU: $(sysctl -n machdep.cpu.brand_string)"
else
    echo "CPU: $(grep "model name" /proc/cpuinfo | head -n 1 | cut -d':' -f2 | xargs)"
fi
echo "Cores: $AVAILABLE_CORES (using $MAX_CORES)"
echo "Memory: ${TOTAL_MEM_GB}GB (using ${USABLE_MEM_GB}GB)"
echo

# Execute Snakemake - either directly or via container
if [ "$USE_CONTAINER" = true ]; then
    echo "Starting PredictDb-Snakemake pipeline in container..."
    
    # Make sure directories exist for container volumes
    mkdir -p results logs benchmarks
    
    # Determine container runtime commands
    if [ "$CONTAINER_CMD" = "docker" ]; then
        $CONTAINER_CMD run --rm \
            -v $(pwd):/predictdb \
            -w /predictdb \
            --user $(id -u):$(id -g) \
            predictdb-snakemake \
            --cores $MAX_CORES \
            --resources mem_mb=$(($USABLE_MEM_GB * 1024)) \
            --keep-going \
            --printshellcmds \
            --reason \
            $RESUME_ARG \
            "${ARGS[@]}"
    elif [ "$CONTAINER_CMD" = "singularity" ]; then
        $CONTAINER_CMD exec \
            --bind $(pwd):/predictdb \
            predictdb-snakemake.sif \
            snakemake \
            --cores $MAX_CORES \
            --resources mem_mb=$(($USABLE_MEM_GB * 1024)) \
            --keep-going \
            --printshellcmds \
            --reason \
            $RESUME_ARG \
            "${ARGS[@]}"
    else
        echo "Unsupported container command: $CONTAINER_CMD"
        exit 1
    fi
else
    echo "Starting PredictDb-Snakemake pipeline..."
    snakemake \
        --cores $MAX_CORES \
        --resources mem_mb=$(($USABLE_MEM_GB * 1024)) \
        --use-conda \
        --conda-frontend mamba \
        --keep-going \
        --printshellcmds \
        --reason \
        $RESUME_ARG \
        "${ARGS[@]}"
fi

# Check execution status
STATUS=$?
if [ $STATUS -eq 0 ]; then
    echo "Pipeline completed successfully!"
    
    # Generate benchmark visualization if benchmarks exist
    if [ -d "benchmarks" ] && [ "$(ls -A benchmarks)" ]; then
        echo "Generating benchmark visualizations..."
        python scripts/utils/plot_benchmarks.py
    fi
else
    echo "Pipeline execution failed. Check logs for details."
fi

exit $STATUS