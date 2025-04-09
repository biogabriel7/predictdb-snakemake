#!/usr/bin/env python3

"""
Script to collect and combine SNP weight summaries from all chromosomes.
Replaces R-based weight_summary.R functionality.
"""

import pandas as pd
import os
import sys

# Get input and output files from Snakemake
weight_summary_files = snakemake.input.weight_summaries
output_file = snakemake.output.all_weight_sum

def read_weight_file(file_path):
    """Read a weight summary file."""
    try:
        df = pd.read_csv(file_path, sep='\t')
        print(f"Read {df.shape[0]} weight entries from {os.path.basename(file_path)}")
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return pd.DataFrame()

def main():
    """Collect and combine weight summaries."""
    # Initialize empty DataFrame for all weight summaries
    all_weights = pd.DataFrame()
    
    # Read and combine each weight file
    for weight_file in weight_summary_files:
        df = read_weight_file(weight_file)
        all_weights = pd.concat([all_weights, df], ignore_index=True)
    
    # Check for duplicates
    duplicates = all_weights.duplicated(subset=['gene_id', 'rsid']).sum()
    if duplicates > 0:
        print(f"WARNING: {duplicates} duplicate (gene_id, rsid) entries found. Keeping the first occurrence.")
        all_weights = all_weights.drop_duplicates(subset=['gene_id', 'rsid'], keep='first')
    
    # Sort by gene_id and then by absolute weight value (descending)
    if 'gene_id' in all_weights.columns and 'beta' in all_weights.columns:
        all_weights['abs_beta'] = all_weights['beta'].abs()
        all_weights = all_weights.sort_values(by=['gene_id', 'abs_beta'], ascending=[True, False])
        all_weights = all_weights.drop(columns=['abs_beta'])
    
    # Write combined weight summaries
    all_weights.to_csv(output_file, sep='\t', index=False)
    
    print(f"Combined {all_weights.shape[0]} weight entries across {all_weights['gene_id'].nunique()} genes")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    main()