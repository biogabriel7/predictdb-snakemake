#!/usr/bin/env python3

"""
Script to collect and combine chromosome summaries.
Replaces R-based chrom_summary.R functionality.
"""

import pandas as pd
import os
import sys

# Get input from Snakemake - assuming we'll collect from model training outputs
# In this case, we'll generate chromosome summaries from the model summaries
model_summary_files = snakemake.input.model_summaries
output_file = snakemake.output.all_chrom_sum

def extract_chrom_summary(file_path):
    """
    Extract chromosome summary information from a model summary file.
    
    Args:
        file_path: Path to model summary file
    
    Returns:
        DataFrame with chromosome summary information
    """
    try:
        # Read model summary file
        df = pd.read_csv(file_path, sep='\t')
        
        if len(df) == 0:
            return pd.DataFrame()
        
        # Extract chromosome number from filename
        chrom = os.path.basename(file_path).split('_')[0].replace('chr', '')
        
        # Get summary statistics for this chromosome
        n_genes = df.shape[0]
        n_samples = df['n_samples'].iloc[0] if 'n_samples' in df.columns else 0
        n_snps = df['n_snps_in_window'].sum() if 'n_snps_in_window' in df.columns else 0
        n_snps_in_model = df['n_snps_in_model'].sum() if 'n_snps_in_model' in df.columns else 0
        cv_seed = df['cv_seed'].iloc[0] if 'cv_seed' in df.columns else 42
        
        # Create chromosome summary
        chrom_summary = pd.DataFrame({
            'chrom': [chrom],
            'n_genes': [n_genes],
            'n_samples': [n_samples],
            'n_snps': [n_snps],
            'n_snps_in_model': [n_snps_in_model],
            'cv_seed': [cv_seed]
        })
        
        print(f"Extracted summary for chromosome {chrom}: {n_genes} genes, {n_snps_in_model} SNPs in models")
        return chrom_summary
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return pd.DataFrame()

def main():
    """Collect and combine chromosome summaries."""
    # Initialize empty DataFrame for all chromosome summaries
    all_chrom_summaries = pd.DataFrame()
    
    # Process each model summary file
    for summary_file in model_summary_files:
        chrom_summary = extract_chrom_summary(summary_file)
        all_chrom_summaries = pd.concat([all_chrom_summaries, chrom_summary], ignore_index=True)
    
    # Sort by chromosome number
    all_chrom_summaries['chrom_num'] = pd.to_numeric(all_chrom_summaries['chrom'], errors='coerce')
    all_chrom_summaries = all_chrom_summaries.sort_values('chrom_num')
    all_chrom_summaries = all_chrom_summaries.drop(columns=['chrom_num'])
    
    # Write combined chromosome summaries
    all_chrom_summaries.to_csv(output_file, sep='\t', index=False)
    
    print(f"Combined summaries for {all_chrom_summaries.shape[0]} chromosomes")
    print(f"Total genes: {all_chrom_summaries['n_genes'].sum()}")
    print(f"Total SNPs in models: {all_chrom_summaries['n_snps_in_model'].sum()}")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    main()