#!/usr/bin/env python3

"""
Script to train Elastic Net models for predicting gene expression.
Replaces R-based elasticnet.R and nested_cv_elasticnet.R functionality.
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
warnings.filterwarnings('ignore')

# Get inputs from Snakemake
gene_annot_file = snakemake.input.gene_annot
snp_file = snakemake.input.snp_file
genotype_file = snakemake.input.genotype_file
expression_file = snakemake.input.gene_expr
chrom = snakemake.params.chrom
nested_cv = snakemake.params.nested_cv
n_folds = snakemake.params.n_folds

# Check if covariates file exists in input
has_covariates = False
if hasattr(snakemake.input, 'covariates'):
    covariates_file = snakemake.input.covariates
    has_covariates = True

# Output files
model_summary_file = snakemake.output.model_summary
weight_summary_file = snakemake.output.weight_summary
covariance_file = snakemake.output.covariance

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
    geno_df = pd.read_csv(genotype_file, sep='\t')
    return geno_df

def get_gene_expression(expression_file):
    """Load gene expression data"""
    expr_df = pd.read_csv(expression_file, sep='\t')
    return expr_df

def get_covariates(covariates_file):
    """Load covariates data if provided"""
    if covariates_file and os.path.exists(covariates_file):
        cov_df = pd.read_csv(covariates_file, sep='\t')
        return cov_df
    return None

def filter_by_gene(gene_id, snp_df, geno_df, expr_df, window=1000000):
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

def train_elastic_net(X, y, covariates=None, n_folds=10, nested=False):
    """Train elastic net model with or without nested CV"""
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
                l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1],
                alphas=np.logspace(-3, 0, 10),
                cv=inner_cv,
                max_iter=1000,
                tol=1e-4,
                random_state=42
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
            random_state=42
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
        # Standard cross-validation
        model = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1],
            alphas=np.logspace(-3, 0, 10),
            cv=cv,
            max_iter=1000,
            tol=1e-4,
            random_state=42
        )
        
        model.fit(X_scaled, y)
        
        # Calculate performance with cross-validation
        y_pred = cross_val_predict(
            model, X_scaled, y, cv=cv
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

# Load data
gene_annot_df = get_gene_annotation(gene_annot_file)
snp_df = get_snp_annotation(snp_file)
geno_df = get_genotype_data(genotype_file)
expr_df = get_gene_expression(expression_file)
covariates_df = get_covariates(covariates_file) if has_covariates else None

# Filter genes by chromosome
chrom_genes = gene_annot_df[gene_annot_df['chr'] == chrom]['gene_id'].tolist()

# Initialize result dataframes
model_summaries = []
weight_summaries = []
covariance_data = []

# Train models for each gene
for gene_id in chrom_genes:
    print(f"Processing gene: {gene_id}")
    
    # Filter data for this gene
    snp_filtered, geno_filtered, gene_expr = filter_by_gene(gene_id, snp_df, geno_df, expr_df)
    
    if gene_expr is None or len(snp_filtered) == 0 or len(geno_filtered) == 0:
        continue
    
    # Prepare features and target
    X = geno_filtered.iloc[:, 1:].values.T  # Transpose to samples x features
    y = gene_expr.values
    
    # Get covariates if available
    cov_matrix = None
    if covariates_df is not None:
        cov_matrix = covariates_df.values.T  # Transpose to samples x covariates
    
    # Train model
    try:
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
        model_summaries.append(model_summary)
        
        # Record weights for SNPs with non-zero coefficients
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
    
    except Exception as e:
        print(f"Error processing gene {gene_id}: {str(e)}")
        continue

# Save results to files
pd.DataFrame(model_summaries).to_csv(model_summary_file, sep='\t', index=False)
pd.DataFrame(weight_summaries).to_csv(weight_summary_file, sep='\t', index=False)
pd.DataFrame(covariance_data).to_csv(covariance_file, sep='\t', index=False)

print(f"Completed processing chromosome {chrom}")