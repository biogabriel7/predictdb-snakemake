#!/usr/bin/env python3
"""
Test script for partitioned model training.
"""

import os
import sys
import json
import subprocess
import argparse
import pandas as pd
import numpy as np
import time

# Add parent directory to path for importing local modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.utils.logger import setup_logger

# Set up logger
logger = setup_logger(__name__)

def generate_test_data(output_dir, n_genes=100, n_snps=1000, n_samples=50):
    """
    Generate test data for the partitioned training approach.
    
    Args:
        output_dir: Directory to write test data
        n_genes: Number of genes to generate
        n_snps: Number of SNPs to generate
        n_samples: Number of samples to generate
    """
    logger.info(f"Generating test data in {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate gene annotation
    gene_annot = pd.DataFrame({
        'gene_id': [f"ENSG{i:08d}" for i in range(1, n_genes+1)],
        'chr': np.random.choice(list(range(1, 23)), n_genes),
        'start': np.random.randint(1, 1000000, n_genes),
        'end': np.random.randint(1000000, 2000000, n_genes),
        'tss': np.random.randint(500000, 1500000, n_genes)
    })
    
    # Ensure unique gene IDs
    gene_annot = gene_annot.drop_duplicates(subset=['gene_id'])
    
    # Generate SNP annotation
    snp_df = pd.DataFrame({
        'varID': [f"rs{i}" for i in range(1, n_snps+1)],
        'rsid': [f"rs{i}" for i in range(1, n_snps+1)],
        'chr': np.random.choice(list(range(1, 23)), n_snps),
        'pos': np.random.randint(1, 2000000, n_snps),
        'refAllele': np.random.choice(['A', 'C', 'G', 'T'], n_snps),
        'effectAllele': np.random.choice(['A', 'C', 'G', 'T'], n_snps)
    })
    
    # Ensure unique SNP IDs
    snp_df = snp_df.drop_duplicates(subset=['varID'])
    
    # Generate sample IDs
    sample_ids = [f"SAMPLE_{i}" for i in range(1, n_samples+1)]
    
    # Generate genotype matrix
    genotype = np.random.choice([0, 1, 2], size=(len(snp_df), n_samples))
    genotype_df = pd.DataFrame(genotype, columns=sample_ids)
    genotype_df.insert(0, 'ID', snp_df['varID'].values)
    
    # Generate gene expression matrix
    expression = np.random.normal(size=(len(gene_annot), n_samples))
    expression_df = pd.DataFrame(expression, columns=sample_ids)
    expression_df.insert(0, 'gene_id', gene_annot['gene_id'].values)
    
    # Generate covariates
    covariates = np.random.normal(size=(5, n_samples))
    covariates_df = pd.DataFrame(covariates, columns=sample_ids)
    covariates_df.insert(0, 'cov_id', [f"COV_{i}" for i in range(1, 6)])
    
    # Save files
    gene_annot.to_csv(os.path.join(output_dir, 'gene_annotation.txt'), sep='\t', index=False)
    snp_df.to_csv(os.path.join(output_dir, 'snp_annotation.txt'), sep='\t', index=False)
    genotype_df.to_csv(os.path.join(output_dir, 'genotype.txt'), sep='\t', index=False)
    expression_df.to_csv(os.path.join(output_dir, 'gene_expression.txt'), sep='\t', index=False)
    covariates_df.to_csv(os.path.join(output_dir, 'covariates.txt'), sep='\t', index=False)
    
    logger.info(f"Generated {len(gene_annot)} genes, {len(snp_df)} SNPs, {n_samples} samples")
    
    return {
        'gene_annot': gene_annot,
        'snp_df': snp_df,
        'genotype_df': genotype_df,
        'expression_df': expression_df,
        'covariates_df': covariates_df
    }

def create_test_config(data_dir, output_dir, partitions=5):
    """
    Create a test configuration file for Snakemake.
    
    Args:
        data_dir: Directory containing test data
        output_dir: Directory for output files
        partitions: Number of partitions to create
    """
    config = {
        'gene_annotation': os.path.join(data_dir, 'gene_annotation.txt'),
        'snp_annotation': os.path.join(data_dir, 'snp_annotation.txt'),
        'genotype': os.path.join(data_dir, 'genotype.txt'),
        'gene_exp': os.path.join(data_dir, 'gene_expression.txt'),
        'covariates': os.path.join(data_dir, 'covariates.txt'),
        'partitions': partitions,
        'cis_window': 500000,
        'model_r2_threshold': 0.01,
        'nested_cv': False,
        'nfolds': 3,
        'peer': False,
        'pca': True,
        'n_peer_factors': 3,
        'prefix': 'test',
        'keepIntermediate': True
    }
    
    # Create config directory
    os.makedirs(os.path.join(output_dir, 'config'), exist_ok=True)
    
    # Write config to file
    with open(os.path.join(output_dir, 'config', 'test_config.yaml'), 'w') as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    
    logger.info(f"Created test configuration in {os.path.join(output_dir, 'config', 'test_config.yaml')}")
    
    return os.path.join(output_dir, 'config', 'test_config.yaml')

def run_test_workflow(config_file, output_dir, dry_run=True):
    """
    Run the test workflow with Snakemake.
    
    Args:
        config_file: Path to config file
        output_dir: Directory for output files
        dry_run: Whether to do a dry run
    """
    # Create command
    cmd = [
        'snakemake',
        '--configfile', config_file,
        '--directory', output_dir,
        '--cores', '2',
        '--reason'
    ]
    
    if dry_run:
        cmd.append('--dryrun')
    
    # Run command
    logger.info(f"Running command: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    # Log output
    logger.info(f"Snakemake stdout: {stdout.decode('utf-8')}")
    if stderr:
        logger.error(f"Snakemake stderr: {stderr.decode('utf-8')}")
    
    return process.returncode

def verify_partitions(partitions_dir):
    """
    Verify that partitions were created correctly.
    
    Args:
        partitions_dir: Directory containing partition files
    """
    # Check if directory exists
    if not os.path.exists(partitions_dir):
        logger.error(f"Partitions directory {partitions_dir} does not exist")
        return False
    
    # Get partition files
    partition_files = [f for f in os.listdir(partitions_dir) if f.startswith('partition_') and f.endswith('.json')]
    
    if not partition_files:
        logger.error(f"No partition files found in {partitions_dir}")
        return False
    
    logger.info(f"Found {len(partition_files)} partition files")
    
    # Verify each partition file
    total_genes = 0
    for partition_file in partition_files:
        with open(os.path.join(partitions_dir, partition_file), 'r') as f:
            try:
                partition = json.load(f)
                total_genes += len(partition)
                logger.info(f"Partition {partition_file}: {len(partition)} genes")
            except json.JSONDecodeError:
                logger.error(f"Failed to parse partition file {partition_file}")
                return False
    
    logger.info(f"Total genes across all partitions: {total_genes}")
    
    return True

def main():
    """Main function for testing partitioned training."""
    parser = argparse.ArgumentParser(description='Test partitioned training')
    parser.add_argument('--data-dir', default='tests/data', help='Directory for test data')
    parser.add_argument('--output-dir', default='tests/output', help='Directory for test output')
    parser.add_argument('--n-genes', type=int, default=100, help='Number of genes to generate')
    parser.add_argument('--n-snps', type=int, default=1000, help='Number of SNPs to generate')
    parser.add_argument('--n-samples', type=int, default=50, help='Number of samples to generate')
    parser.add_argument('--n-partitions', type=int, default=5, help='Number of partitions to create')
    parser.add_argument('--run-workflow', action='store_true', help='Actually run the workflow (not just dry run)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate test data
    start_time = time.time()
    logger.info("Generating test data...")
    test_data = generate_test_data(
        args.data_dir,
        n_genes=args.n_genes,
        n_snps=args.n_snps,
        n_samples=args.n_samples
    )
    logger.info(f"Test data generation took {time.time() - start_time:.2f} seconds")
    
    # Create test config
    start_time = time.time()
    logger.info("Creating test configuration...")
    config_file = create_test_config(
        args.data_dir,
        args.output_dir,
        partitions=args.n_partitions
    )
    logger.info(f"Configuration creation took {time.time() - start_time:.2f} seconds")
    
    # Run test workflow
    start_time = time.time()
    logger.info("Running test workflow...")
    exit_code = run_test_workflow(
        config_file,
        args.output_dir,
        dry_run=not args.run_workflow
    )
    logger.info(f"Workflow execution took {time.time() - start_time:.2f} seconds")
    
    if exit_code != 0:
        logger.error(f"Workflow failed with exit code {exit_code}")
        return exit_code
    
    if args.run_workflow:
        # Verify partitions
        start_time = time.time()
        logger.info("Verifying partitions...")
        success = verify_partitions(os.path.join(args.output_dir, 'results', 'gene_partitions'))
        logger.info(f"Partition verification took {time.time() - start_time:.2f} seconds")
        
        if not success:
            logger.error("Partition verification failed")
            return 1
    
    logger.info("Test completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 