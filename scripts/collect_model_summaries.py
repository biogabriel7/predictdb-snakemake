#!/usr/bin/env python3

"""
Script to collect and combine model summaries from all chromosomes.
Replaces R-based model_summary.R functionality.
"""

import pandas as pd
import os
import sys

# Get input and output files from Snakemake
model_summary_files = snakemake.input.model_summaries
output_file = snakemake.output.all_model_sum

def read_summary_file(file_path):
    """Read a model summary file."""
    try:
        df = pd.read_csv(file_path, sep='\t')
        print(f"Read {df.shape[0]} model summaries from {os.path.basename(file_path)}")
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return pd.DataFrame()

def main():
    """Collect and combine model summaries."""
    # Initialize empty DataFrame for all model summaries
    all_models = pd.DataFrame()
    
    # Read and combine each summary file
    for summary_file in model_summary_files:
        df = read_summary_file(summary_file)
        all_models = pd.concat([all_models, df], ignore_index=True)
    
    # Check for duplicates
    if all_models.duplicated(subset=['gene_id']).any():
        print("WARNING: Duplicate gene_id entries found. Keeping the first occurrence.")
        all_models = all_models.drop_duplicates(subset=['gene_id'], keep='first')
    
    # Sort by performance
    if 'rho_avg_squared' in all_models.columns:
        all_models = all_models.sort_values(by='rho_avg_squared', ascending=False)
    
    # Write combined model summaries
    all_models.to_csv(output_file, sep='\t', index=False)
    
    print(f"Combined {all_models.shape[0]} model summaries and saved to {output_file}")

if __name__ == "__main__":
    main()