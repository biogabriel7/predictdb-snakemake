#!/usr/bin/env python3

"""
Script to generate principal components from gene expression data.
Optimized with Numba for Apple M3 chip performance.
"""

import numpy as np
import pandas as pd
import os
import sys
import numba as nb
from numba import prange

# Get inputs from Snakemake
gene_expr_file = snakemake.input.gene_expr
covs_outfile = snakemake.output.pcs
n_components = snakemake.params.n_components
n_threads = snakemake.threads

# Configure NumPy to use all available cores for BLAS operations
os.environ["OMP_NUM_THREADS"] = str(n_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
os.environ["MKL_NUM_THREADS"] = str(n_threads)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)

@nb.njit(parallel=True, fastmath=True)
def center_matrix_numba(X):
    """
    Numba-optimized matrix centering.
    
    Args:
        X: Input matrix to center (samples × genes)
    
    Returns:
        Centered matrix
    """
    n, p = X.shape
    X_centered = np.zeros_like(X)
    
    for j in prange(p):
        # Calculate mean
        col_mean = 0.0
        for i in range(n):
            col_mean += X[i, j]
        col_mean /= n
        
        # Center the column
        for i in range(n):
            X_centered[i, j] = X[i, j] - col_mean
    
    return X_centered

@nb.njit(fastmath=True)
def compute_covariance_numba(X_centered):
    """
    Numba-optimized covariance matrix computation.
    
    Args:
        X_centered: Centered data matrix (samples × genes)
    
    Returns:
        Covariance matrix (genes × genes)
    """
    n, p = X_centered.shape
    cov_matrix = np.zeros((p, p), dtype=np.float64)
    
    for i in range(p):
        for j in range(i, p):
            cov_ij = 0.0
            for k in range(n):
                cov_ij += X_centered[k, i] * X_centered[k, j]
            cov_ij /= (n - 1)
            cov_matrix[i, j] = cov_ij
            cov_matrix[j, i] = cov_ij  # Symmetric matrix
    
    return cov_matrix

@nb.njit(fastmath=True)
def compute_pcs_from_eigenvectors(X_centered, eigenvectors, n_components):
    """
    Numba-optimized projection of data onto eigenvectors to get PC scores.
    
    Args:
        X_centered: Centered data matrix (samples × genes)
        eigenvectors: Matrix of eigenvectors (genes × components)
        n_components: Number of principal components to keep
    
    Returns:
        Principal component scores (samples × components)
    """
    n_samples = X_centered.shape[0]
    pcs = np.zeros((n_samples, n_components))
    
    for i in range(n_samples):
        for j in range(n_components):
            pc_val = 0.0
            for k in range(eigenvectors.shape[0]):
                pc_val += X_centered[i, k] * eigenvectors[k, j]
            pcs[i, j] = pc_val
    
    return pcs

def custom_pca_with_numba(X, n_components):
    """
    Perform PCA using Numba-optimized functions.
    
    Args:
        X: Input data matrix (samples × genes)
        n_components: Number of principal components to compute
    
    Returns:
        pc_scores: Principal component scores (samples × components)
        explained_variance_ratio: Proportion of variance explained by each PC
    """
    # Center the data
    X_centered = center_matrix_numba(X)
    
    # Compute covariance matrix
    cov_matrix = compute_covariance_numba(X_centered)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Keep only the top n_components
    eigenvalues = eigenvalues[:n_components]
    eigenvectors = eigenvectors[:, :n_components]
    
    # Project data onto eigenvectors to get PC scores
    pc_scores = compute_pcs_from_eigenvectors(X_centered, eigenvectors, n_components)
    
    # Calculate explained variance ratio
    total_var = np.sum(np.diag(cov_matrix))
    explained_variance_ratio = eigenvalues / total_var
    
    return pc_scores, explained_variance_ratio

def read_expression_data(file_path):
    """Read gene expression data from CSV file with memory optimization."""
    try:
        gene_exp = pd.read_csv(file_path, index_col=0)
        print(f"Loaded expression data: {gene_exp.shape[0]} samples × {gene_exp.shape[1]} genes")
        return gene_exp
    except MemoryError:
        chunks = []
        for chunk in pd.read_csv(file_path, index_col=0, chunksize=1000):
            chunks.append(chunk)
        gene_exp = pd.concat(chunks)
        print(f"Loaded expression data in chunks: {gene_exp.shape[0]} samples × {gene_exp.shape[1]} genes")
        return gene_exp

def main():
    """Main function to generate principal components with Numba optimizations."""
    # Read gene expression data
    print(f"Reading expression data from {gene_expr_file}...")
    gene_exp = read_expression_data(gene_expr_file)
    
    # Convert to numpy array for faster processing
    expr_values = gene_exp.values
    
    # Determine whether to use custom Numba PCA or sklearn based on data size
    use_numba = (expr_values.shape[0] > 50) and (expr_values.shape[1] > 100)
    
    if use_numba:
        print(f"Using Numba-optimized PCA implementation for {expr_values.shape[0]} samples × {expr_values.shape[1]} genes...")
        # Compute PCs using custom Numba-optimized implementation
        pcs, explained_variance_ratio = custom_pca_with_numba(expr_values, n_components)
    else:
        print("Using sklearn's PCA implementation for smaller dataset...")
        # For smaller datasets, sklearn's implementation is more efficient
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        # Standardize the data
        scaler = StandardScaler()
        expr_scaled = scaler.fit_transform(expr_values)
        
        # Compute the PCs
        pca = PCA(n_components=n_components)
        pcs = pca.fit_transform(expr_scaled)
        explained_variance_ratio = pca.explained_variance_ratio_
    
    # Convert to dataframe
    pc_columns = [f'PC{i+1}' for i in range(n_components)]
    pcs_df = pd.DataFrame(pcs, columns=pc_columns, index=gene_exp.index)
    
    # Transpose the PCs for output format compatibility (PCs × samples)
    pcs_transposed = pcs_df.T
    
    # Write the covariates as a tab-delimited file
    pcs_transposed.to_csv(covs_outfile, sep='\t')
    
    # Print variance explained
    print("Variance explained by principal components:")
    total_explained = 0
    for i in range(n_components):
        percent = explained_variance_ratio[i] * 100
        total_explained += percent
        print(f"  PC{i+1}: {percent:.2f}%")
    
    print(f"Total variance explained by {n_components} PCs: {total_explained:.2f}%")
    print(f"Generated {n_components} principal components and saved to {covs_outfile}")

if __name__ == "__main__":
    main()