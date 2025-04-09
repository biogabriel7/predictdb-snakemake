#!/usr/bin/env python3

"""
Script to generate PEER factors from gene expression data.
Replaces R-based PEER factor generation.

Note: This requires the PyPEER package to be installed:
    pip install peer
"""

import numpy as np
import pandas as pd
import os
import sys
try:
    import peer
except ImportError:
    print("ERROR: PEER package not found. Please install it with:")
    print("pip install git+https://github.com/PMBio/peer.git")
    sys.exit(1)

# Get input and output files from Snakemake
gene_expr_file = snakemake.input.gene_expr
peers_outfile = snakemake.output.peers
n_factors = snakemake.params.n_factors

def read_expression_data(file_path):
    """Read gene expression data from CSV file."""
    try:
        gene_exp = pd.read_csv(file_path, index_col=0)
        print(f"Loaded expression data: {gene_exp.shape[0]} samples × {gene_exp.shape[1]} genes")
        return gene_exp
    except Exception as e:
        print(f"Error reading expression data: {str(e)}")
        sys.exit(1)

def run_peer_analysis(expression_data, n_factors):
    """
    Run PEER factor analysis on gene expression data.
    
    Args:
        expression_data: DataFrame with samples as rows and genes as columns
        n_factors: Number of PEER factors to compute
    
    Returns:
        PEER factors as a DataFrame
    """
    # Convert expression data to numpy array (samples × genes)
    expr_values = expression_data.values
    
    # Initialize PEER model
    model = peer.PEER()
    model.setPhenoMean(expr_values)
    
    # Set number of factors
    model.setNk(n_factors)
    
    # Set prior parameters (default values)
    model.setPriorAlpha(0.001, 0.1)
    model.setPriorEps(0.1, 10.0)
    
    # Run inference
    print(f"Running PEER inference with {n_factors} factors...")
    model.update()
    
    # Get factors (samples × factors)
    factors = model.getX()
    
    # Convert to DataFrame
    factor_names = [f"PEER_{i+1}" for i in range(n_factors)]
    peer_factors = pd.DataFrame(factors, index=expression_data.index, columns=factor_names)
    
    # Get variance explained by each factor
    alpha = model.getAlpha()
    var_explained = 1.0 / alpha
    
    # Print variance explained
    print("Variance explained by PEER factors:")
    for i, var in enumerate(var_explained):
        print(f"  PEER_{i+1}: {var:.4f}")
    
    return peer_factors

def main():
    """Main function to generate PEER factors."""
    # Read expression data
    expression_data = read_expression_data(gene_expr_file)
    
    # Run PEER analysis
    peer_factors = run_peer_analysis(expression_data, n_factors)
    
    # Transpose for output format compatibility (factors × samples)
    peer_factors_transposed = peer_factors.T
    
    # Write output
    peer_factors_transposed.to_csv(peers_outfile, sep='\t')
    
    print(f"Generated {n_factors} PEER factors and saved to {peers_outfile}")

if __name__ == "__main__":
    main()