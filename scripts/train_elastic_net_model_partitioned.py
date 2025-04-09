#!/usr/bin/env python3

"""
Script to train Elastic Net models for predicting gene expression.
This version uses balanced partitions instead of chromosomes for better computational efficiency.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold, GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
import os
import sys
import json
import multiprocessing as mp
from functools import partial
import warnings
import numba as nb
from numba import prange

# Import utility modules
from utils.file_handling import open_file
from utils.logger import setup_logger
from utils.progress_tracker import ProgressTracker
from utils.input_validation import validate_inputs
from utils.adaptive_threading import AdaptiveThreadPool
from utils.data_partitioning import read_partitions_file, filter_snps_for_partition, filter_genes_for_partition

warnings.filterwarnings('ignore')

# Set up logger
log_file = snakemake.log[0] if hasattr(snakemake, 'log') else None
logger = setup_logger(__name__, log_file)

# Get inputs from Snakemake
gene_annot_file = snakemake.input.gene_annot
snp_file = snakemake.input.snp_file
genotype_file = snakemake.input.genotype_file
expression_file = snakemake.input.gene_expr
partitions_file = snakemake.input.partitions
partition_id = int(snakemake.params.partition_id)
nested_cv = snakemake.params.nested_cv
n_folds = snakemake.params.n_folds
n_threads = snakemake.threads

# Check if covariates file exists in input
has_covariates = False
if hasattr(snakemake.input, 'covariates'):
    covariates_file = snakemake.input.covariates
    has_covariates = True

# Output files
model_summary_file = snakemake.output.model_summary
weight_summary_file = snakemake.output.weight_summary
covariance_file = snakemake.output.covariance

# Configure NumPy to use all available cores for BLAS operations
os.environ["OMP_NUM_THREADS"] = str(n_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
os.environ["MKL_NUM_THREADS"] = str(n_threads)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)

# Numba optimized functions
@nb.njit(parallel=True, fastmath=True)
def pearson_correlation_numba(x, y):
    """
    Numba-optimized Pearson correlation calculation.
    
    Args:
        x: First array
        y: Second array
    
    Returns:
        Correlation coefficient
    """
    n = len(x)
    mean_x = 0.0
    mean_y = 0.0
    
    # Calculate means
    for i in range(n):
        mean_x += x[i]
        mean_y += y[i]
    
    mean_x /= n
    mean_y /= n
    
    # Calculate covariance and variances
    cov_xy = 0.0
    var_x = 0.0
    var_y = 0.0
    
    for i in range(n):
        x_diff = x[i] - mean_x
        y_diff = y[i] - mean_y
        cov_xy += x_diff * y_diff
        var_x += x_diff * x_diff
        var_y += y_diff * y_diff
    
    # Calculate correlation
    if var_x > 0 and var_y > 0:
        return cov_xy / np.sqrt(var_x * var_y)
    else:
        return 0.0

@nb.njit(fastmath=True)
def z_score_numba(corr, n):
    """
    Numba-optimized Z-score calculation for correlation.
    
    Args:
        corr: Correlation coefficient
        n: Sample size
    
    Returns:
        Z-score
    """
    # Ensure correlation is within valid range to avoid NaN
    corr = max(min(corr, 0.9999), -0.9999)
    return 0.5 * np.log((1 + corr) / (1 - corr)) * np.sqrt(n - 3)

@nb.njit(parallel=True, fastmath=True)
def calculate_covariance_numba(X_scaled):
    """
    Numba-optimized covariance calculation.
    
    Args:
        X_scaled: Scaled feature matrix
    
    Returns:
        Covariance matrix
    """
    n, p = X_scaled.shape
    cov = np.zeros((p, p))
    
    for i in prange(p):
        for j in range(i, p):
            cov_ij = 0.0
            for k in range(n):
                cov_ij += X_scaled[k, i] * X_scaled[k, j]
            cov_ij /= n
            cov[i, j] = cov_ij
            cov[j, i] = cov_ij  # Symmetric matrix
    
    return cov

@nb.njit(fastmath=True)
def scale_matrix_numba(X):
    """
    Numba-optimized matrix scaling (standardization).
    
    Args:
        X: Input matrix to scale
    
    Returns:
        Scaled matrix
    """
    n, p = X.shape
    X_scaled = np.zeros_like(X)
    
    for j in range(p):
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

def get_gene_annotation(gene_annot_file):
    """Parse gene annotation file"""
    try:
        with open_file(gene_annot_file, 'r') as f:
            gene_df = pd.read_csv(f, sep='\t')
        logger.info(f"Loaded gene annotation data: {gene_df.shape[0]} genes")
        return gene_df
    except Exception as e:
        logger.error(f"Error loading gene annotation file: {str(e)}")
        sys.exit(1)

def get_snp_annotation(snp_file):
    """Parse SNP annotation file"""
    try:
        with open_file(snp_file, 'r') as f:
            snp_df = pd.read_csv(f, sep='\t')
        logger.info(f"Loaded SNP annotation data: {snp_df.shape[0]} SNPs")
        return snp_df
    except Exception as e:
        logger.error(f"Error loading SNP annotation file: {str(e)}")
        sys.exit(1)

def get_genotype_data(genotype_file, snp_ids=None):
    """Load genotype data with memory optimization, filtered by SNP IDs if provided"""
    try:
        # For very large files, we'll read in chunks and filter
        logger.info(f"Loading genotype data from {genotype_file}")
        file_size = os.path.getsize(genotype_file)
        
        if snp_ids is not None:
            logger.info(f"Filtering to {len(snp_ids)} SNPs in partition")
            snp_ids = set(snp_ids)  # Convert to set for faster lookups
        
        if file_size > 1e9:  # If file is larger than 1GB
            logger.info("Large genotype file detected, using chunked loading")
            chunks = []
            chunk_iter = pd.read_csv(open_file(genotype_file, 'r'), sep='\t', chunksize=10000)
            
            # Use progress tracker for chunk loading
            chunk_count = max(1, int(file_size / 1e7))  # Estimate chunk count
            progress = ProgressTracker(chunk_count, "Loading genotype chunks")
            progress.start()
            
            for i, chunk in enumerate(chunk_iter):
                # Filter to SNPs in this partition if IDs are provided
                if snp_ids is not None:
                    chunk_filtered = chunk[chunk['ID'].isin(snp_ids)]
                    if not chunk_filtered.empty:
                        chunks.append(chunk_filtered)
                else:
                    chunks.append(chunk)
                progress.update(processed_items=i+1)
            
            progress.finish()
            
            if not chunks:
                logger.warning("No genotype data found for the specified SNPs")
                return pd.DataFrame()
                
            return pd.concat(chunks)
        else:
            # For smaller files, read entire file
            with open_file(genotype_file, 'r') as f:
                geno_df = pd.read_csv(f, sep='\t')
            
            # Filter to SNPs in this partition if IDs are provided
            if snp_ids is not None:
                geno_df = geno_df[geno_df['ID'].isin(snp_ids)]
                
            logger.info(f"Loaded genotype data: {geno_df.shape[0]} variants, {geno_df.shape[1]-1} samples")
            return geno_df
    except MemoryError:
        logger.error("Memory error while loading genotype data. Try reducing batch size.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading genotype data: {str(e)}")
        sys.exit(1)

def get_gene_expression(expression_file):
    """Load gene expression data"""
    try:
        with open_file(expression_file, 'r') as f:
            expr_df = pd.read_csv(f, sep='\t', index_col=0)
        logger.info(f"Loaded gene expression data: {expr_df.shape[0]} genes, {expr_df.shape[1]} samples")
        return expr_df
    except Exception as e:
        logger.error(f"Error loading gene expression file: {str(e)}")
        sys.exit(1)

def get_covariates(covariates_file):
    """Load covariates data"""
    try:
        if covariates_file:
            with open_file(covariates_file, 'r') as f:
                cov_df = pd.read_csv(f, sep='\t', index_col=0)
            logger.info(f"Loaded covariates data: {cov_df.shape[0]} covariates, {cov_df.shape[1]} samples")
            return cov_df
        return None
    except Exception as e:
        logger.error(f"Error loading covariates file: {str(e)}")
        sys.exit(1)

def filter_by_gene(gene_id, snp_df, geno_df, expr_df, gene_annot_df, window=1000000):
    """Filter SNPs within window size of gene"""
    try:
        # Get gene info
        gene_info = gene_annot_df[gene_annot_df['gene_id'] == gene_id]
        if gene_info.empty:
            logger.warning(f"Gene {gene_id} not found in gene annotation file")
            return None, None, None
        
        gene_info = gene_info.iloc[0]
        chr = gene_info['chr']
        start = max(0, gene_info['start'] - window)
        end = gene_info['end'] + window
        
        # Filter SNPs by position
        snp_filtered = snp_df[(snp_df['chr'] == chr) & 
                             (snp_df['pos'] >= start) & 
                             (snp_df['pos'] <= end)]
        
        if snp_filtered.empty:
            logger.warning(f"No SNPs found for gene {gene_id} within window")
            return None, None, None
        
        # Get genotypes for filtered SNPs
        snp_ids = set(snp_filtered['varID'])
        geno_filtered = geno_df[geno_df['ID'].isin(snp_ids)]
        
        if geno_filtered.empty:
            logger.warning(f"No genotype data found for SNPs near gene {gene_id}")
            return None, None, None
        
        # Get expression for gene
        if gene_id not in expr_df.columns:
            logger.warning(f"Gene {gene_id} not found in expression data")
            return None, None, None
        
        gene_expr = expr_df[gene_id]
        
        return snp_filtered, geno_filtered, gene_expr
    except Exception as e:
        logger.error(f"Error filtering data for gene {gene_id}: {str(e)}")
        return None, None, None

def calculate_correlation(y_true, y_pred):
    """Calculate correlation between true and predicted values"""
    try:
        # Use numba optimized function for larger arrays
        if len(y_true) > 1000:
            corr = pearson_correlation_numba(y_true, y_pred)
        else:
            corr, _ = stats.pearsonr(y_true, y_pred)
        
        return corr
    except Exception as e:
        logger.error(f"Error calculating correlation: {str(e)}")
        return 0.0

def train_elastic_net(X, y, covariates=None, n_folds=10, nested=False, alphas=None, l1_ratios=None):
    """Train elastic net model with cross-validation"""
    try:
        # Default hyperparameters if not provided
        if alphas is None:
            alphas = np.logspace(-3, 0, 20)
        if l1_ratios is None:
            l1_ratios = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1]
        
        n_samples = X.shape[0]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Add covariates if provided
        if covariates is not None and covariates.shape[0] > 0:
            covariates_scaled = scaler.fit_transform(covariates)
            X_with_covs = np.hstack((X_scaled, covariates_scaled))
        else:
            X_with_covs = X_scaled
        
        # Configure cross-validation
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # Train Elastic Net model
        enet = ElasticNetCV(
            alphas=alphas,
            l1_ratio=l1_ratios,
            cv=cv,
            max_iter=1000,
            tol=1e-4,
            selection='random',
            random_state=42
        )
        
        # Fit model
        if nested:
            # For nested CV, use cross_val_predict
            group_cv = GroupKFold(n_splits=n_folds)
            groups = np.arange(n_samples) % n_folds  # Simple grouping
            y_pred = cross_val_predict(enet, X_with_covs, y, cv=group_cv, groups=groups)
            
            # Fit final model on all data
            enet.fit(X_with_covs, y)
        else:
            # Regular CV within ElasticNetCV
            enet.fit(X_with_covs, y)
            y_pred = enet.predict(X_with_covs)
        
        # Evaluate model
        r2 = enet.score(X_with_covs, y)
        correlation = calculate_correlation(y, y_pred)
        zscore = calculate_z_score(correlation, n_samples)
        pval = 2 * (1 - stats.norm.cdf(abs(zscore)))
        
        # Calculate weights for SNPs only (excluding covariates)
        if covariates is not None and covariates.shape[0] > 0:
            snp_coef = enet.coef_[:X.shape[1]]
        else:
            snp_coef = enet.coef_
        
        # Calculate covariance matrix
        if np.count_nonzero(snp_coef) > 0:
            non_zero_idx = np.nonzero(snp_coef)[0]
            if len(non_zero_idx) > 1:
                X_subset = X_scaled[:, non_zero_idx]
                cov_matrix = calculate_covariance(X_subset, snp_coef[non_zero_idx])
            else:
                cov_matrix = np.array([[1.0]])
        else:
            cov_matrix = np.array([[]])
        
        return {
            'model': enet,
            'performance': {
                'r2': r2,
                'correlation': correlation,
                'zscore': zscore,
                'pval': pval,
                'n_snps_in_model': np.count_nonzero(snp_coef),
                'alpha': enet.alpha_,
                'l1_ratio': enet.l1_ratio_
            },
            'weights': snp_coef,
            'covariance': cov_matrix,
            'non_zero_idx': np.nonzero(snp_coef)[0] if np.count_nonzero(snp_coef) > 0 else []
        }
    except Exception as e:
        logger.error(f"Error training elastic net model: {str(e)}")
        return {
            'model': None,
            'performance': {
                'r2': 0,
                'correlation': 0,
                'zscore': 0,
                'pval': 1,
                'n_snps_in_model': 0,
                'alpha': 0,
                'l1_ratio': 0
            },
            'weights': np.array([]),
            'covariance': np.array([[]]),
            'non_zero_idx': []
        }

def calculate_z_score(corr, n):
    """Calculate Z-score from correlation coefficient using Fisher transformation"""
    # Use numba optimized function for larger sample sizes
    if n > 1000:
        return z_score_numba(corr, n)
    else:
        # Fisher z-transformation
        z = 0.5 * np.log((1 + corr) / (1 - corr)) if abs(corr) < 1 else 0
        z *= np.sqrt(n - 3)
        return z

def calculate_covariance(X, weights):
    """Calculate covariance matrix for SNPs with non-zero weights"""
    try:
        # Use numba optimized function for larger matrices
        if X.shape[0] * X.shape[1] > 10000:
            return calculate_covariance_numba(X)
        else:
            return np.cov(X, rowvar=False)
    except Exception as e:
        logger.error(f"Error calculating covariance: {str(e)}")
        return np.array([[]])

def create_memory_mapped_matrix(data, prefix, dtype=np.float64):
    """
    Create a memory-mapped matrix from data.
    
    Args:
        data: Input data (numpy array or DataFrame)
        prefix: Prefix for temporary file name
        dtype: Data type for memory-mapped array
    
    Returns:
        Memory-mapped array and file path
    """
    # Create temporary file path
    temp_dir = 'results/temp'
    os.makedirs(temp_dir, exist_ok=True)
    
    temp_file = os.path.join(temp_dir, f"{prefix}_{os.getpid()}.mmap")
    
    # Convert data to numpy array if necessary
    if hasattr(data, 'values'):
        data_array = data.values
    else:
        data_array = data
    
    # Create memory-mapped array
    shape = data_array.shape
    mmap_array = np.memmap(temp_file, dtype=dtype, mode='w+', shape=shape)
    
    # Copy data to memory-mapped array
    mmap_array[:] = data_array[:]
    mmap_array.flush()
    
    logger.debug(f"Created memory-mapped matrix of shape {shape} at {temp_file}")
    
    return mmap_array, temp_file

def process_gene(gene_id, gene_annot_df, snp_df, geno_df, expr_df, covariates_df=None, nested_cv=False, n_folds=10):
    """Process a single gene to train and evaluate model"""
    try:
        # Filter data for this gene
        snp_filtered, geno_filtered, gene_expr = filter_by_gene(
            gene_id, snp_df, geno_df, expr_df, gene_annot_df
        )
        
        if snp_filtered is None or geno_filtered is None or gene_expr is None:
            return None, [], []
        
        # Prepare feature matrix (X) and target (y)
        snp_ids = geno_filtered['ID'].values
        sample_ids = geno_df.columns[1:]
        X = geno_filtered.iloc[:, 1:].T.values
        y = gene_expr.values
        
        # Use memory mapping for large matrices
        if X.shape[0] * X.shape[1] > 1000000:  # More than ~8MB assuming 8-byte floats
            logger.debug(f"Using memory mapping for gene {gene_id} (X shape: {X.shape})")
            X_mmap, x_temp_file = create_memory_mapped_matrix(X, f"X_{gene_id}")
            X = X_mmap
        
        # Prepare covariates if available
        covs = None
        if covariates_df is not None and not covariates_df.empty:
            covs = covariates_df.loc[:, sample_ids].values.T
            
            # Use memory mapping for large covariate matrices
            if covs.shape[0] * covs.shape[1] > 1000000:
                logger.debug(f"Using memory mapping for covariates (shape: {covs.shape})")
                covs_mmap, covs_temp_file = create_memory_mapped_matrix(covs, f"covs_{gene_id}")
                covs = covs_mmap
        
        # Train model
        result = train_elastic_net(X, y, covs, n_folds, nested_cv)
        
        # Prepare model summary
        model_summary = {
            'gene': gene_id,
            'chromosome': gene_annot_df[gene_annot_df['gene_id'] == gene_id]['chr'].values[0],
            'n_snps': snp_filtered.shape[0],
            'n_snps_in_window': snp_filtered.shape[0],
            'n_snps_in_model': result['performance']['n_snps_in_model'],
            'best_alpha': result['performance']['alpha'],
            'best_l1_ratio': result['performance']['l1_ratio'],
            'cv_r2': result['performance']['r2'],
            'cv_corr': result['performance']['correlation'],
            'zscore': result['performance']['zscore'],
            'pval': result['performance']['pval'],
            'pred_perf_pval': result['performance']['pval']
        }
        
        # Prepare weight summaries
        weight_summaries = []
        non_zero_idx = result['non_zero_idx']
        weights = result['weights']
        
        # Only create weight entries for non-zero weights
        for i in non_zero_idx:
            if i < len(snp_ids):  # Ensure index is valid
                snp_id = snp_ids[i]
                snp_info = snp_filtered[snp_filtered['varID'] == snp_id].iloc[0]
                
                weight_summaries.append({
                    'gene': gene_id,
                    'rsid': snp_info['rsid'],
                    'varID': snp_id,
                    'ref_allele': snp_info['refAllele'],
                    'effect_allele': snp_info['effectAllele'],
                    'weight': weights[i],
                    'chromosome': snp_info['chr'],
                    'position': snp_info['pos']
                })
        
        # Prepare covariance data
        covariance_data = []
        if len(non_zero_idx) > 1:
            cov_matrix = result['covariance']
            for i in range(len(non_zero_idx)):
                for j in range(i, len(non_zero_idx)):
                    if i < len(snp_ids) and j < len(snp_ids):  # Ensure indices are valid
                        snp_id_i = snp_ids[non_zero_idx[i]]
                        snp_id_j = snp_ids[non_zero_idx[j]]
                        
                        covariance_data.append({
                            'gene': gene_id,
                            'rsid1': snp_filtered[snp_filtered['varID'] == snp_id_i]['rsid'].values[0],
                            'rsid2': snp_filtered[snp_filtered['varID'] == snp_id_j]['rsid'].values[0],
                            'varID1': snp_id_i,
                            'varID2': snp_id_j,
                            'covariance': cov_matrix[i, j]
                        })
        
        # Clean up memory-mapped files
        if 'X_mmap' in locals() and os.path.exists(x_temp_file):
            del X_mmap
            os.remove(x_temp_file)
            
        if 'covs_mmap' in locals() and os.path.exists(covs_temp_file):
            del covs_mmap
            os.remove(covs_temp_file)
        
        return model_summary, weight_summaries, covariance_data
    
    except Exception as e:
        logger.error(f"Error processing gene {gene_id}: {str(e)}")
        return None, [], []

def main():
    """Main function to train models for genes in the partition"""
    # Load partitions file
    partitions = read_partitions_file(partitions_file)
    
    # Get current partition
    if partition_id >= len(partitions):
        logger.error(f"Partition ID {partition_id} is out of range. Only {len(partitions)} partitions exist.")
        sys.exit(1)
        
    current_partition = partitions[partition_id]
    chrom = current_partition['chrom']
    start = current_partition['start']
    end = current_partition['end']
    genes_to_process = current_partition['gene_ids']
    
    logger.info(f"Processing partition {partition_id}: chromosome {chrom}, region {start}-{end}, {len(genes_to_process)} genes")
    
    # Validate inputs
    if not validate_inputs(gene_annot_file, snp_file, genotype_file, expression_file, 
                         covariates_file if has_covariates else None):
        logger.error("Input validation failed, aborting")
        sys.exit(1)
    
    # Load data
    logger.info("Loading input data...")
    gene_annot_df = get_gene_annotation(gene_annot_file)
    
    # Filter gene annotations to this partition
    gene_annot_filtered = filter_genes_for_partition(gene_annot_df, current_partition)
    
    # Load and filter SNP annotations to this partition
    full_snp_df = get_snp_annotation(snp_file)
    snp_df = full_snp_df[(full_snp_df['chr'] == chrom) & 
                        (full_snp_df['pos'] >= start) & 
                        (full_snp_df['pos'] <= end)]
    
    logger.info(f"Filtered to {snp_df.shape[0]} SNPs in partition region")
    
    # Get the SNP IDs to filter genotype data
    snp_ids = snp_df['varID'].tolist()
    
    # Load filtered genotype data
    geno_df = get_genotype_data(genotype_file, snp_ids=snp_ids)
    
    # Load expression data
    expr_df = get_gene_expression(expression_file)
    
    # Load covariates if available
    covariates_df = None
    if has_covariates:
        covariates_df = get_covariates(covariates_file)
    
    # Create adaptive thread pool
    logger.info(f"Using up to {n_threads} threads for processing")
    thread_pool = AdaptiveThreadPool(
        min_threads=1,
        max_threads=n_threads,
        target_cpu_percent=80
    )
    
    # Track progress
    progress_tracker = ProgressTracker(
        total_items=len(genes_to_process),
        description=f"Training models for partition {partition_id}",
        update_interval=30  # Update every 30 seconds
    )
    progress_tracker.start()
    
    # Process genes using adaptive thread pool
    try:
        results = thread_pool.map(
            lambda gene_id: process_gene(
                gene_id, gene_annot_df, snp_df, geno_df, expr_df, 
                covariates_df, nested_cv, n_folds
            ),
            genes_to_process
        )
    finally:
        thread_pool.close()
    
    # Collect results
    model_summaries = []
    weight_summaries = []
    covariance_data = []
    
    for i, (model_summary, weights, covariances) in enumerate(results):
        if model_summary is not None:
            model_summaries.append(model_summary)
            weight_summaries.extend(weights)
            covariance_data.extend(covariances)
        
        # Update progress after processing each gene
        progress_tracker.update(processed_items=i+1)
    
    # Finish progress tracking
    progress_tracker.finish()
    
    # Convert results to DataFrames
    model_df = pd.DataFrame(model_summaries)
    weight_df = pd.DataFrame(weight_summaries)
    cov_df = pd.DataFrame(covariance_data)
    
    # Write results to files
    logger.info(f"Writing {len(model_summaries)} model summaries to {model_summary_file}")
    model_df.to_csv(model_summary_file, sep='\t', index=False)
    
    logger.info(f"Writing {len(weight_summaries)} weight entries to {weight_summary_file}")
    weight_df.to_csv(weight_summary_file, sep='\t', index=False)
    
    logger.info(f"Writing {len(covariance_data)} covariance entries to {covariance_file}")
    cov_df.to_csv(covariance_file, sep='\t', index=False)
    
    logger.info(f"Completed training for partition {partition_id}")

if __name__ == "__main__":
    main() 