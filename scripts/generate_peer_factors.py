#!/usr/bin/env python3

"""
Script to generate PEER factors from gene expression data.
Optimized with Numba for Apple M3 chip performance.

Note: This requires the PyPEER package to be installed:
    pip install peer
"""

import numpy as np
import pandas as pd
import os
import sys
import numba as nb
from numba import prange
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
n_threads = snakemake.threads

# Configure NumPy to use all available cores for BLAS operations
os.environ["OMP_NUM_THREADS"] = str(n_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
os.environ["MKL_NUM_THREADS"] = str(n_threads)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)

@nb.njit(parallel=True, fastmath=True)
def scale_matrix_numba(X):
    """
    Numba-optimized matrix scaling (standardization).
    
    Args:
        X: Input matrix to scale (samples × genes)
    
    Returns:
        Scaled matrix
    """
    n, p = X.shape
    X_scaled = np.zeros_like(X)
    
    for j in prange(p):
        # Calculate mean
        col_mean = 0.0
        for i in range(n):
            col_mean += X[i, j]
        col_mean /= n
        
        # Calculate standard deviation
        col_std = 0.0
        for i in range(n):
            diff = X[i, j] - col_mean
            col_std += diff * diff
        col_std = np.sqrt(col_std / (n - 1))
        
        # Scale the column
        if col_std > 0:
            for i in range(n):
                X_scaled[i, j] = (X[i, j] - col_mean) / col_std
        else:
            for i in range(n):
                X_scaled[i, j] = 0.0
    
    return X_scaled

@nb.njit(parallel=True)
def compute_residuals_numba(X, factors, weights):
    """
    Numba-optimized computation of residuals after PEER factor removal.
    
    Args:
        X: Original expression matrix (samples × genes)
        factors: PEER factors matrix (samples × factors)
        weights: Weight matrix (factors × genes)
    
    Returns:
        Residual expression matrix after removing factor effects
    """
    n_samples, n_genes = X.shape
    n_factors = factors.shape[1]
    residuals = np.zeros_like(X)
    
    # Copy original data
    for i in range(n_samples):
        for j in range(n_genes):
            residuals[i, j] = X[i, j]
    
    # Subtract factor effects
    for i in range(n_samples):
        for j in prange(n_genes):
            for k in range(n_factors):
                residuals[i, j] -= factors[i, k] * weights[k, j]
    
    return residuals

def read_expression_data(file_path):
    """Read gene expression data from CSV file with memory optimization."""
    try:
        # Try to read the whole file
        gene_exp = pd.read_csv(file_path, index_col=0)
        print(f"Loaded expression data: {gene_exp.shape[0]} samples × {gene_exp.shape[1]} genes")
        return gene_exp
    except MemoryError:
        # If memory error, try reading in chunks (for very large files)
        chunks = []
        for chunk in pd.read_csv(file_path, index_col=0, chunksize=1000):
            chunks.append(chunk)
        gene_exp = pd.concat(chunks)
        print(f"Loaded expression data in chunks: {gene_exp.shape[0]} samples × {gene_exp.shape[1]} genes")
        return gene_exp

def run_peer_analysis(expression_data, n_factors, max_iterations=1000):
    """
    Run PEER factor analysis on gene expression data with Numba optimizations.
    
    Args:
        expression_data: DataFrame with samples as rows and genes as columns
        n_factors: Number of PEER factors to compute
        max_iterations: Maximum number of iterations for PEER
    
    Returns:
        PEER factors as a DataFrame
    """
    # Convert expression data to numpy array (samples × genes)
    expr_values = expression_data.values
    
    # Apply Numba-optimized scaling for better numerical stability
    print("Scaling expression data with Numba...")
    expr_scaled = scale_matrix_numba(expr_values)
    
    # Initialize PEER model
    print(f"Initializing PEER model with {n_factors} factors...")
    model = peer.PEER()
    model.setPhenoMean(expr_scaled)
    
    # Set number of factors
    model.setNk(n_factors)
    
    # Set prior parameters
    model.setPriorAlpha(0.001, 0.1)
    model.setPriorEps(0.1, 10.0)
    model.setMaxIters(max_iterations)
    
    # Run inference
    print(f"Running PEER inference (this may take a while)...")
    model.update()
    
    # Get factors (samples × factors)
    factors = model.getX()
    
    # Get weights (factors × genes)
    weights = model.getW()
    
    # Calculate variance explained by each factor
    alpha = model.getAlpha()
    var_explained = 1.0 / alpha
    
    # Print variance explained
    print("Variance explained by PEER factors:")
    total_var = np.sum(var_explained)
    for i, var in enumerate(var_explained):
        percent = (var / total_var) * 100
        print(f"  PEER_{i+1}: {var:.4f} ({percent:.2f}%)")
    
    # Create DataFrame with factor values
    factor_names = [f"PEER_{i+1}" for i in range(n_factors)]
    peer_factors = pd.DataFrame(factors, index=expression_data.index, columns=factor_names)
    
    # Calculate residuals using Numba (for diagnostic purposes)
    print("Computing residuals with Numba...")
    residuals = compute_residuals_numba(expr_scaled, factors, weights)
    
    # Assess how much variance remains in residuals
    residual_var = np.var(residuals)
    original_var = np.var(expr_scaled)
    var_explained_total = 1 - (residual_var / original_var)
    print(f"Total variance explained by all factors: {var_explained_total:.4f} ({var_explained_total*100:.2f}%)")
    
    return peer_factors

def main():
    """Main function to generate PEER factors with Numba optimizations."""
    # Read expression data
    print(f"Reading expression data from {gene_expr_file}...")
    expression_data = read_expression_data(gene_expr_file)
    
    # Run PEER analysis with Numba optimization
    peer_factors = run_peer_analysis(expression_data, n_factors)
    
    # Transpose for output format compatibility (factors × samples)
    peer_factors_transposed = peer_factors.T
    
    # Write output
    print(f"Writing {n_factors} PEER factors to {peers_outfile}...")
    peer_factors_transposed.to_csv(peers_outfile, sep='\t')
    
    print(f"PEER factor generation complete.")

if __name__ == "__main__":
    main()