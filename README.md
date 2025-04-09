# PredictDb-Snakemake

A Python-based reimplementation of the PredictDb pipeline using Snakemake workflow manager.

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

## Requirements

- Python 3.7+
- Snakemake 7.0+

### Python Dependencies

- numpy
- pandas
- scikit-learn
- scipy
- matplotlib (optional, for visualization)
- numba

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/PredictDb-Snakemake.git
   cd PredictDb-Snakemake
   ```

2. Create and activate a conda environment:
   ```bash
   conda create -n predictdb-snakemake python=3.9 snakemake-minimal
   conda activate predictdb-snakemake
   ```

3. Install required Python packages:
   ```bash
   pip install numpy pandas scikit-learn scipy
   ```

## Usage

1. Edit the configuration file at `config/config.yaml` to specify your input files and parameters:
   ```yaml
   # Input files
   gene_annotation: "path/to/your/gene_annotation.gtf.gz"
   snp_annotation: "path/to/your/snp_annotation.txt"
   genotype: "path/to/your/genotype.txt"
   gene_exp: "path/to/your/gene_expression.txt"
   
   # Analysis parameters
   prefix: "Your_Study_Name"
   peer: true  # Use PEER factors
   pca: false  # Use PCA
   n_peer_factors: 15
   nested_cv: true
   nfolds: 10
   ```

2. Run the pipeline:
   ```bash
   snakemake --cores 8 -p
   ```

3. For cluster execution:
   ```bash
   snakemake --profile your-cluster-profile -j 100
   ```

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

## Output Files

- `results/predict_db_*.db`: SQLite database with prediction models
- `results/Covariances.txt`: SNP covariance matrices
- Various summary files in the results directory

## File Structure

```
PredictDb-Snakemake/
├── Snakefile              # Main workflow definition
├── config/                # Configuration files
│   └── config.yaml        # User-editable parameters
├── scripts/               # Python scripts
│   ├── train_elastic_net_model.py
│   ├── make_db.py
│   ├── generate_pcs.py
│   └── ...
├── results/               # Output directory
└── docs/                  # Documentation
```

## License

This project is available under the [MIT License](LICENSE).

## Citation

If you use this software in your research, please cite:

```
Your Name et al. PredictDb-Snakemake: A Python reimplementation of the PredictDb pipeline using Snakemake. GitHub (2025).
```

## Acknowledgments

This project is a reimplementation of the [PredictDb-nextflow](https://github.com/hakyimlab/PredictDb-nextflow) workflow developed by the Im Lab.

## Contact

For questions or issues, please open an issue on GitHub or contact [your email or contact information].
