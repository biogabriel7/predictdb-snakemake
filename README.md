# PredictDb-Snakemake

A high-performance Python-based implementation of the PredictDb pipeline using Snakemake workflow manager.

## Overview

PredictDb-Snakemake is a computational pipeline for building prediction models of gene expression levels from genotype data. The models can be used with tools like PrediXcan to study gene regulation and its role in complex traits and diseases.

This project is a complete Python reimplementation of the original [PredictDb-nextflow](https://github.com/hakyimlab/PredictDb-nextflow) pipeline, which combined Nextflow, R, and Python. By consolidating to a single language (Python) and using Snakemake as the workflow manager, this implementation offers:

- Simplified installation and dependency management
- Improved maintainability
- Comparable performance with identical results
- Enhanced accessibility for Python-oriented bioinformaticians

## Features

- Performs genotype data preprocessing and splitting by chromosome
- Supports generation of covariates through PCA or PEER factor analysis
- Trains predictive models using elastic net regression
- Performs nested cross-validation for robust model evaluation
- Creates database files for use with downstream analysis tools
- Calculates genetic covariances between selected SNPs
- Fully parallelizable workflow

### New Features

- **Automatic gzipped file handling**: Transparently reads and writes both plain text and gzipped files
- **Dynamic resource allocation**: Automatically adjusts CPU and memory usage based on system load
- **Benchmarking and performance visualization**: Tracks and visualizes resource usage for pipeline optimization
- **Checkpointing and resume capability**: Allows interrupted pipelines to continue from the last completed step
- **Comprehensive error handling**: Validates inputs and provides detailed error messages
- **Email notifications**: Optional email alerts on pipeline completion or failure
- **Progress tracking**: Real-time progress reporting for long-running operations
- **Memory optimization**: Uses memory mapping and chunked processing for large datasets
- **Containerization support**: Run the pipeline in Docker or Singularity containers for reproducibility
- **Advanced logging system**: Detailed logs for monitoring and debugging

## Requirements

- Python 3.7+
- Snakemake 7.0+

### Python Dependencies

- numpy
- pandas
- scikit-learn
- scipy
- matplotlib
- numba
- psutil
- snakemake

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/PredictDb-Snakemake.git
   cd PredictDb-Snakemake
   ```

2. Create and activate a conda environment:
   ```bash
   conda env create -f environment.yaml
   conda activate predictdb-snakemake
   ```

### Docker Installation (Optional)

To use the containerized version:

1. Make sure Docker is installed
2. Build the container:
   ```bash
   ./build_container.sh
   ```

3. For Singularity:
   ```bash
   ./build_container.sh --singularity
   ```

## Usage

### Basic Usage

1. Edit the configuration file at `config/config.yaml` to specify your input files and parameters.
2. Run the pipeline:
   ```bash
   ./run.sh
   ```

### Advanced Usage

#### Container Execution
```bash
# Run with Docker
./run.sh --container

# Run with Singularity
./run.sh --container --container-cmd singularity
```

#### Additional Configuration

The pipeline now uses multiple configuration files:

- `config/config.yaml` - Main pipeline configuration
- `config/resources.yaml` - Resource limits for each rule
- `config/notifications.yaml` - Email notification settings
- `config/report.yaml` - Report customization

## Pipeline Steps

1. **Preprocessing**:
   - Parse gene annotation
   - Split SNP annotation by chromosome
   - Split genotype by chromosome
   - Transpose gene expression

2. **Covariate Calculation** (optional):
   - Generate PEER factors or PCA components
   - Combine with provided covariates if available

3. **Model Training**:
   - Train elastic net models for each gene
   - Perform cross-validation for performance assessment
   - Calculate weights for predictive SNPs

4. **Database Creation**:
   - Aggregate model summaries
   - Collect weight information
   - Create SQLite database
   - Filter models based on performance
   - Calculate and filter covariances

5. **Reporting and Visualization**:
   - Generate HTML report
   - Create benchmark visualizations

## Output Files

- `results/predict_db_*.db`: SQLite database with prediction models
- `results/Covariances.txt`: SNP covariance matrices
- `results/report.html`: Pipeline execution report
- `results/benchmark_plots/`: Performance visualizations
- Various summary files in the results directory
- Log files in the logs directory

## File Structure

```
PredictDb-Snakemake/
├── Snakefile              # Main workflow definition
├── run.sh                 # Execution script
├── build_container.sh     # Container build script
├── Dockerfile             # Container definition
├── config/                # Configuration files
│   ├── config.yaml        # User-editable parameters
│   ├── resources.yaml     # Resource allocation
│   ├── notifications.yaml # Email notification settings
│   └── report.yaml        # Report customization
├── scripts/               # Python scripts
│   ├── train_elastic_net_model.py
│   ├── make_db.py
│   ├── generate_pcs.py
│   └── ...
│   └── utils/             # Utility modules
│       ├── file_handling.py  # Gzipped file utilities
│       ├── logger.py         # Logging utilities
│       ├── resource_allocation.py  # Dynamic resource management
│       └── ...
├── logs/                  # Log files
├── benchmarks/            # Benchmark measurements
├── results/               # Output directory
└── docs/                  # Documentation
```

## Utility Modules

### file_handling.py
Provides utilities for transparently handling both plain text and gzipped files.

### logger.py
Sets up a consistent logging system across all pipeline components.

### resource_allocation.py
Dynamic CPU and memory allocation based on system load.

### progress_tracker.py
Real-time progress reporting for long-running operations.

### input_validation.py
Validates input files to catch errors early.

### adaptive_threading.py
Dynamically adjusts thread pool size based on system load.

### plot_benchmarks.py
Visualizes pipeline performance metrics.

## License

This project is available under the [MIT License](LICENSE).

## Citation

If you use this software in your research, please cite:

```
Your Name et al. PredictDb-Snakemake: A Python implementation of the PredictDb pipeline using Snakemake. GitHub (2023).
```

## Acknowledgments

This project is a reimplementation of the [PredictDb-nextflow](https://github.com/hakyimlab/PredictDb-nextflow) workflow developed by the Im Lab.

## Contact

For questions or issues, please open an issue on GitHub or contact [your email or contact information].
