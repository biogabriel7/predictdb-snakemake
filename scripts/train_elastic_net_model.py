#!/usr/bin/env python3

"""
Script to train Elastic Net models for predicting gene expression.
Optimized for Apple M3 chip performance.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold, GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
import os
import sys
from scipy.stats import pearsonr
import warnings
import threading
import multiprocessing as mp
from functools import partial
warnings.filterwarnings('ignore')

# Get inputs from Snakemake
gene_annot_file = snakemake.input.gene_annot
snp_file = snakemake.input.snp_file
genotype_file = snakemake.input.genotype_file
expression_file = snakemake.input.gene_expr
chrom = snakemake.params.chrom
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
# This is particularly effective on Apple Silicon
os.environ["OMP_NUM_THREADS"] = str(n_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
os.environ["MKL_NUM_THREADS"] = str(n_threads)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)

def get_gene_annotation(gene_annot_file):
    """Parse gene annotation file"""
    gene_df = pd.read_csv(gene_annot_file, sep='\t')
    return gene_df

def get_snp_annotation(snp_file):
    """Parse SNP annotation file"""
    snp_df = pd.read_csv(snp_file, sep='\t')
    return snp_df

def get_genotype_data(genotype_file):
    """Load genotype data"""
    # For larger files, use chunking to reduce memory usage
    geno_df = pd.read_csv(genotype_file, sep='\t')
    return geno_df

def get_gene_expression(expression_file):
    """Load gene expression data"""
    expr_df = pd.read_csv(expression_file, sep='\t', index_col=0)
    return expr_df

def get_covariates(covariates_file):
    """Load covariates data if provided"""
    if covariates_file and os.path.exists(covariates_file):
        cov_df = pd.read_csv(covariates_file, sep='\t', index_col=0)
        return cov_df
    return None

def filter_by_gene(gene_id, snp_df, geno_df, expr_df, gene_annot_df, window=1000000):
    """Filter SNPs within window size of gene"""
    gene_info = gene_annot_df[gene_annot_df['gene_id'] == gene_id].iloc[0]
    chr = gene_info['chr']
    start = max(0, gene_info['start'] - window)
    end = gene_info['end'] + window
    
    # Filter SNPs by position
    snp_filtered = snp_df[(snp_df['chr'] == chr) & 
                           (snp_df['pos'] >= start) & 
                           (snp_df['pos'] <= end)]
    
    # Get genotypes for filtered SNPs
    snp_ids = snp_filtered['varID'].tolist()
    geno_filtered = geno_df[geno_df['ID'].isin(snp_ids)]
    
    # Get expression for gene
    gene_expr = expr_df[gene_id] if gene_id in expr_df.columns else None
    
    return snp_filtered, geno_filtered, gene_expr

def calculate_correlation(y_true, y_pred):
    """Calculate Pearson correlation and p-value"""
    correlation, p_value = pearsonr(y_true, y_pred)
    return correlation, p_value

def train_elastic_net(X, y, covariates=None, n_folds=10, nested=False, alphas=None, l1_ratios=None):
    """
    Train elastic net model with or without nested CV.
    Optimized for performance on Apple M3.
    """
    # Set default hyperparameter grids if not provided
    if alphas is None:
        alphas = np.logspace(-3, 0, 10)
    if l1_ratios is None:
        l1_ratios = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1]
    
    # Combine features with covariates if provided
    if covariates is not None:
        X_combined = np.hstack((X, covariates))
    else:
        X_combined = X
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)
    
    # Set up cross-validation
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    if nested:
        # Nested cross-validation
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Outer loop for performance estimation
        fold_predictions = []
        fold_alphas = []
        fold_l1_ratios = []
        fold_models = []
        
        for train_idx, test_idx in outer_cv.split(X_scaled):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Inner loop for hyperparameter tuning
            model = ElasticNetCV(
                l1_ratio=l1_ratios,
                alphas=alphas,
                cv=inner_cv,
                max_iter=1000,
                tol=1e-4,
                random_state=42,
                n_jobs=n_threads  # Use multi-threading
            )
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            fold_predictions.append((y_test, y_pred))
            fold_alphas.append(model.alpha_)
            fold_l1_ratios.append(model.l1_ratio_)
            fold_models.append(model)
        
        # Train final model on all data with best hyperparameters
        best_alpha = np.median(fold_alphas)
        best_l1_ratio = np.median(fold_l1_ratios)
        
        final_model = ElasticNetCV(
            l1_ratio=[best_l1_ratio],
            alphas=[best_alpha],
            cv=cv,
            max_iter=1000,
            tol=1e-4,
            random_state=42,
            n_jobs=n_threads  # Use multi-threading
        )
        final_model.fit(X_scaled, y)
        
        # Calculate performance metrics
        all_y_true = []
        all_y_pred = []
        for y_true, y_pred in fold_predictions:
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)
        
        r2_avg, p_value = calculate_correlation(all_y_true, all_y_pred)
        
        return final_model, r2_avg, p_value, best_alpha, best_l1_ratio
    else:
        # Standard cross-validation with optimized settings
        model = ElasticNetCV(
            l1_ratio=l1_ratios,
            alphas=alphas,
            cv=cv,
            max_iter=1000,
            tol=1e-4,
            random_state=42,
            n_jobs=n_threads  # Use multi-threading for Apple M3
        )
        
        # Pre-allocate memory for efficiency
        model.fit(X_scaled, y)
        
        # Calculate performance with cross-validation
        y_pred = cross_val_predict(
            model, X_scaled, y, cv=cv, n_jobs=n_threads
        )
        
        r2_avg, p_value = calculate_correlation(y, y_pred)
        
        return model, r2_avg, p_value, model.alpha_, model.l1_ratio_

def calculate_z_score(corr, n):
    """Calculate Z-score for correlation"""
    z = 0.5 * np.log((1 + corr) / (1 - corr)) * np.sqrt(n - 3)
    return z

def calculate_covariance(X, weights):
    """Calculate genetic covariance"""
    # Scale genotypes
    X_scaled = StandardScaler().fit_transform(X)
    
    # Calculate covariance: X'X
    covariance = np.dot(X_scaled.T, X_scaled) / X_scaled.shape[0]
    
    return covariance

def process_gene(gene_id, gene_annot_df, snp_df, geno_df, expr_df, covariates_df=None, nested_cv=False, n_folds=10):
    """Process a single gene - used for parallelization"""
    print(f"Processing gene: {gene_id}")
    
    try:
        # Filter data for this gene
        snp_filtered, geno_filtered, gene_expr = filter_by_gene(gene_id, snp_df, geno_df, expr_df, gene_annot_df)
        
        if gene_expr is None or len(snp_filtered) == 0 or len(geno_filtered) == 0:
            print(f"Skipping gene {gene_id}: insufficient data")
            return None, None, None
        
        # Prepare features and target
        X = geno_filtered.iloc[:, 1:].values.T  # Transpose to samples x features
        y = gene_expr.values
        
        # Get covariates if available
        cov_matrix = None
        if covariates_df is not None:
            cov_matrix = covariates_df.values.T  # Transpose to samples x covariates
        
        # Train model
        model, r2_avg, p_value, alpha, l1_ratio = train_elastic_net(
            X, y, covariates=cov_matrix, n_folds=n_folds, nested=nested_cv
        )
        
        # Get model info
        n_samples = len(y)
        n_snps_in_model = np.sum(model.coef_ != 0)
        z_score = calculate_z_score(r2_avg, n_samples)
        
        # Record model summary
        gene_name = gene_annot_df[gene_annot_df['gene_id'] == gene_id]['gene_name'].values[0]
        model_summary = {
            'gene_id': gene_id,
            'gene_name': gene_name,
            'cv_seed': 42,
            'n_samples': n_samples,
            'chrom': chrom,
            'n_snps_in_window': len(snp_filtered),
            'n_snps_in_model': n_snps_in_model,
            'alpha': alpha,
            'lambda': alpha * l1_ratio,
            'rho_avg_squared': r2_avg**2,
            'zscore': z_score,
            'zscore_pval': p_value
        }
        
        # Record weights for SNPs with non-zero coefficients
        weight_summaries = []
        nonzero_indices = np.where(model.coef_ != 0)[0]
        for idx in nonzero_indices:
            snp_id = snp_filtered.iloc[idx]['varID']
            rsid = snp_filtered.iloc[idx]['rsid']
            ref = snp_filtered.iloc[idx]['refAllele']
            alt = snp_filtered.iloc[idx]['effectAllele']
            weight = model.coef_[idx]
            
            weight_summary = {
                'gene_id': gene_id,
                'rsid': rsid,
                'varID': snp_id,
                'ref': ref,
                'alt': alt,
                'beta': weight
            }
            weight_summaries.append(weight_summary)
        
        # Calculate and record covariance
        covariance_data = []
        if n_snps_in_model > 0:
            # Filter X to include only SNPs in the model
            X_model = X[:, nonzero_indices]
            weights = model.coef_[nonzero_indices]
            
            # Calculate covariance
            cov_matrix = calculate_covariance(X_model, weights)
            
            # Record covariance entries
            for i in range(len(nonzero_indices)):
                snp_i = snp_filtered.iloc[nonzero_indices[i]]['varID']
                for j in range(i, len(nonzero_indices)):
                    snp_j = snp_filtered.iloc[nonzero_indices[j]]['varID']
                    cov_value = cov_matrix[i, j]
                    
                    cov_entry = {
                        'GENE': gene_id,
                        'RSID1': snp_i,
                        'RSID2': snp_j,
                        'VALUE': cov_value
                    }
                    covariance_data.append(cov_entry)
        
        return model_summary, weight_summaries, covariance_data
    
    except Exception as e:
        print(f"Error processing gene {gene_id}: {str(e)}")
        return None, None, None

def main():
    """Main function with parallel processing for Apple M3."""
    # Load data
    gene_annot_df = get_gene_annotation(gene_annot_file)
    snp_df = get_snp_annotation(snp_file)
    geno_df = get_genotype_data(genotype_file)
    expr_df = get_gene_expression(expression_file)
    covariates_df = get_covariates(covariates_file) if has_covariates else None
    
    # Filter genes by chromosome
    chrom_genes = gene_annot_df[gene_annot_df['chr'] == chrom]['gene_id'].tolist()
    print(f"Found {len(chrom_genes)} genes on chromosome {chrom}")
    
    # Initialize result collections
    model_summaries = []
    weight_summaries = []
    covariance_data = []
    
    # Process each gene (potential for parallelization)
    # For small datasets, process serially
    if len(chrom_genes) <= 10:
        for gene_id in chrom_genes:
            model_summary, weights, covariances = process_gene(
                gene_id, gene_annot_df, snp_df, geno_df, expr_df, 
                covariates_df, nested_cv, n_folds
            )
            
            if model_summary is not None:
                model_summaries.append(model_summary)
                weight_summaries.extend(weights)
                covariance_data.extend(covariances)
    else:
        # For larger datasets, use multiprocessing
        # Use max of 75% of the specified threads to avoid thread contention
        n_parallel = max(1, min(4, int(n_threads * 0.75)))
        print(f"Using {n_parallel} parallel processes for gene processing")
        
        # Create partial function with fixed parameters
        process_gene_partial = partial(
            process_gene,
            gene_annot_df=gene_annot_df,
            snp_df=snp_df,
            geno_df=geno_df,
            expr_df=expr_df,
            covariates_df=covariates_df,
            nested_cv=nested_cv,
            n_folds=n_folds
        )
        
        # Process genes in parallel
        with mp.Pool(processes=n_parallel) as pool:
            results = pool.map(process_gene_partial, chrom_genes)
        
        # Collect results
        for model_summary, weights, covariances in results:
            if model_summary is not None:
                model_summaries.append(model_summary)
                weight_summaries.extend(weights)
                covariance_data.extend(covariances)
    
    # Save results to files
    pd.DataFrame(model_summaries).to_csv(model_summary_file, sep='\t', index=False)
    pd.DataFrame(weight_summaries).to_csv(weight_summary_file, sep='\t', index=False)
    pd.DataFrame(covariance_data).to_csv(covariance_file, sep='\t', index=False)
    
    print(f"Completed processing chromosome {chrom}")
    print(f"Generated models for {len(model_summaries)} genes")
    print(f"Saved {len(weight_summaries)} weights and {len(covariance_data)} covariance entries")

if __name__ == "__main__":
    main()