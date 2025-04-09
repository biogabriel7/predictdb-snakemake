#!/usr/bin/env python3

"""
Script to combine covariates from different sources.
Replaces R-based combine_covariates.R functionality.
"""

import pandas as pd
import numpy as np
import os
import sys

# Get inputs from Snakemake
computed_covs_file = snakemake.input.computed_covs  # PEER or PCA covariates
provided_covs_file = snakemake.input.provided_covs  # User-provided covariates
final_covs_file = snakemake.output.final_covs       # Combined output file

def read_covariates(file_path):
    """Read covariates file if it exists."""
    if file_path and os.path.exists(file_path):
        try:
            covs = pd.read_csv(file_path, sep='\t', index_col=0)
            print(f"Loaded covariates from {file_path}: {covs.shape[0]} covariates × {covs.shape[1]} samples")
            return covs
        except Exception as e:
            print(f"Error reading covariates from {file_path}: {str(e)}")
            return None
    return None

def combine_covariates(computed_covs, provided_covs):
    """
    Combine covariates from different sources.
    
    Args:
        computed_covs: DataFrame with PEER or PCA covariates (covariates × samples)
        provided_covs: DataFrame with user-provided covariates (covariates × samples)
    
    Returns:
        Combined covariates DataFrame
    """
    if computed_covs is not None and provided_covs is not None:
        # Check if columns (samples) match
        computed_samples = set(computed_covs.columns)
        provided_samples = set(provided_covs.columns)
        
        if computed_samples != provided_samples:
            common_samples = computed_samples.intersection(provided_samples)
            print(f"WARNING: Sample mismatch between computed and provided covariates.")
            print(f"Using only {len(common_samples)} common samples.")
            
            # Filter to common samples
            computed_covs = computed_covs[list(common_samples)]
            provided_covs = provided_covs[list(common_samples)]
        
        # Combine covariates
        combined = pd.concat([computed_covs, provided_covs])
        
        # Check for duplicate covariate names
        if combined.index.duplicated().any():
            print("WARNING: Duplicate covariate names found. Adding suffixes to make them unique.")
            combined = combined.reset_index()
            combined['index'] = combined['index'] + '_' + combined.groupby('index').cumcount().astype(str)
            combined = combined.set_index('index')
        
        print(f"Combined covariates: {combined.shape[0]} covariates × {combined.shape[1]} samples")
        return combined
    
    elif computed_covs is not None:
        print(f"Using only computed covariates: {computed_covs.shape[0]} covariates × {computed_covs.shape[1]} samples")
        return computed_covs
    
    elif provided_covs is not None:
        print(f"Using only provided covariates: {provided_covs.shape[0]} covariates × {provided_covs.shape[1]} samples")
        return provided_covs
    
    else:
        print("No covariates found. Returning empty DataFrame.")
        return pd.DataFrame()

def main():
    """Main function to combine covariates."""
    # Read covariates
    computed_covs = read_covariates(computed_covs_file)
    provided_covs = read_covariates(provided_covs_file)
    
    # Combine covariates
    combined_covs = combine_covariates(computed_covs, provided_covs)
    
    # Write output
    combined_covs.to_csv(final_covs_file, sep='\t')
    
    print(f"Combined covariates saved to {final_covs_file}")

if __name__ == "__main__":
    main()