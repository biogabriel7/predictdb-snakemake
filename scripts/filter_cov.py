#!/usr/bin/env python3

"""
Script to filter covariance data based on filtered database.
Replaces R-based filter_cov.R functionality.
"""

import pandas as pd
import sqlite3
import os
import sys

# Get input and output files from Snakemake
filtered_db = snakemake.input.filtered_db
unfiltered_cov = snakemake.input.all_covs
filtered_cov = snakemake.output.filtered_covs

def get_filtered_genes(db_file):
    """
    Get list of genes in the filtered database.
    
    Args:
        db_file: Path to filtered database file
    
    Returns:
        List of gene IDs
    """
    # Connect to database
    conn = sqlite3.connect(db_file)
    
    # Get genes from extra table
    query = "SELECT gene FROM extra"
    genes = pd.read_sql(query, conn)['gene'].tolist()
    
    # Close connection
    conn.close()
    
    return genes

def filter_covariances(genes, unfiltered_cov_file, filtered_cov_file):
    """
    Filter covariance file to include only genes in the filtered database.
    
    Args:
        genes: List of gene IDs to keep
        unfiltered_cov_file: Path to unfiltered covariance file
        filtered_cov_file: Path to output filtered covariance file
    """
    # Read covariance file
    all_covs = pd.read_csv(unfiltered_cov_file, sep='\t')
    
    # Filter to include only genes in the filtered database
    filtered_covs = all_covs[all_covs['GENE'].isin(genes)]
    
    # Write filtered covariance file
    filtered_covs.to_csv(filtered_cov_file, sep='\t', index=False)
    
    print(f"Filtered covariances from {len(all_covs)} to {len(filtered_covs)} entries")
    print(f"Retained {filtered_covs['GENE'].nunique()} genes out of {all_covs['GENE'].nunique()}")

def main():
    """Main function."""
    # Get genes from filtered database
    filtered_genes = get_filtered_genes(filtered_db)
    
    # Filter covariance file
    filter_covariances(filtered_genes, unfiltered_cov, filtered_cov)
    
    print(f"Filtered covariance file saved to {filtered_cov}")

if __name__ == "__main__":
    main()