#!/usr/bin/env python3

"""
Script to collect and combine covariance matrices from all chromosomes.
Replaces R-based covariance_summary.R functionality.
"""

import pandas as pd
import os
import sys

# Get input and output files from Snakemake
covariance_files = snakemake.input.covariances
output_file = snakemake.output.all_covs

def read_covariance_file(file_path):
    """Read a covariance file."""
    try:
        df = pd.read_csv(file_path, sep='\t')
        print(f"Read {df.shape[0]} covariance entries from {os.path.basename(file_path)}")
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return pd.DataFrame()

def main():
    """Collect and combine covariance matrices."""
    # Initialize empty DataFrame for all covariances
    all_covs = pd.DataFrame()
    
    # Read and combine each covariance file
    for cov_file in covariance_files:
        df = read_covariance_file(cov_file)
        all_covs = pd.concat([all_covs, df], ignore_index=True)
    
    # Check for duplicates
    duplicates = all_covs.duplicated(subset=['GENE', 'RSID1', 'RSID2']).sum()
    if duplicates > 0:
        print(f"WARNING: {duplicates} duplicate covariance entries found. Keeping the first occurrence.")
        all_covs = all_covs.drop_duplicates(subset=['GENE', 'RSID1', 'RSID2'], keep='first')
    
    # Sort by gene and then SNP IDs
    all_covs = all_covs.sort_values(by=['GENE', 'RSID1', 'RSID2'])
    
    # Write combined covariances
    all_covs.to_csv(output_file, sep='\t', index=False)
    
    print(f"Combined {all_covs.shape[0]} covariance entries across {all_covs['GENE'].nunique()} genes")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    main()