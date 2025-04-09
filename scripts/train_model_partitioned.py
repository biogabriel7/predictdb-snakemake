#!/usr/bin/env python3
"""
Train predictive models for gene expression using a specific partition of genes.
This allows for efficient parallel training across multiple partitions.
"""

import argparse
import json
import logging
import numpy as np
import os
import pandas as pd
import pickle
import sys
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(snakemake.log[0]),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def preprocess_data(genotype_file, expression_file, gene_list):
    """
    Load and preprocess genotype and expression data for the specified genes.
    
    Args:
        genotype_file: Path to genotype data file
        expression_file: Path to expression data file
        gene_list: List of gene IDs to process
        
    Returns:
        tuple: (genotype_data, expression_data) as pandas DataFrames
    """
    logger.info(f"Loading expression data from {expression_file}")
    expression_data = pd.read_csv(expression_file, sep='\t', index_col=0)
    
    # Filter expression data to include only the genes in the partition
    expression_data = expression_data.loc[expression_data.index.isin(gene_list)]
    
    logger.info(f"Loading genotype data from {genotype_file}")
    genotype_data = pd.read_csv(genotype_file, sep='\t', index_col=0)
    
    # Ensure samples match between genotype and expression data
    common_samples = list(set(genotype_data.columns).intersection(set(expression_data.columns)))
    logger.info(f"Found {len(common_samples)} common samples between genotype and expression data")
    
    genotype_data = genotype_data[common_samples]
    expression_data = expression_data[common_samples]
    
    return genotype_data, expression_data

def get_cis_snps(gene_id, gene_annot, genotype_data, cis_window=1000000):
    """
    Get cis-SNPs for a specific gene.
    
    Args:
        gene_id: Gene ID to get cis-SNPs for
        gene_annot: Gene annotation DataFrame
        genotype_data: Genotype data DataFrame
        cis_window: Size of cis-window around TSS (default: 1Mb)
    
    Returns:
        pandas.DataFrame: Genotype data for cis-SNPs
    """
    # Get gene information
    gene_info = gene_annot[gene_annot['gene_id'] == gene_id]
    
    if len(gene_info) == 0:
        logger.warning(f"Gene {gene_id} not found in annotation file")
        return None
    
    gene_info = gene_info.iloc[0]
    chrom = gene_info['chromosome']
    tss = gene_info['tss']
    
    # Extract chromosome and position from SNP IDs (assuming format like chr1_12345)
    snp_info = pd.DataFrame(index=genotype_data.index)
    snp_info['chromosome'] = snp_info.index.str.split('_').str[0]
    snp_info['position'] = snp_info.index.str.split('_').str[1].astype(int)
    
    # Filter SNPs in cis-window
    cis_snps = snp_info[
        (snp_info['chromosome'] == chrom) & 
        (snp_info['position'] >= tss - cis_window) & 
        (snp_info['position'] <= tss + cis_window)
    ]
    
    # Get genotype data for cis-SNPs
    cis_genotype = genotype_data.loc[cis_snps.index]
    
    logger.info(f"Found {len(cis_genotype)} cis-SNPs for gene {gene_id}")
    
    return cis_genotype

def train_model(X, y, hyperparameters, cv_folds=5):
    """
    Train an ElasticNet model with cross-validation for a single gene.
    
    Args:
        X: Features (SNP genotypes)
        y: Target (gene expression)
        hyperparameters: Dictionary of hyperparameters or grid of hyperparameters for search
        cv_folds: Number of cross-validation folds
        
    Returns:
        tuple: (trained_model, model_performance)
    """
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Check if we need to do grid search or just use provided hyperparameters
    if isinstance(hyperparameters, dict) and all(not isinstance(v, list) for v in hyperparameters.values()):
        # Use provided hyperparameters
        model = ElasticNet(
            alpha=hyperparameters.get('alpha', 0.01),
            l1_ratio=hyperparameters.get('l1_ratio', 0.5),
            max_iter=hyperparameters.get('max_iter', 1000),
            tol=hyperparameters.get('tol', 1e-4),
            random_state=42
        )
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring='neg_mean_squared_error')
        cv_mse = -np.mean(cv_scores)
        cv_r2 = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring='r2')
        cv_r2_mean = np.mean(cv_r2)
        
        # Train final model on all data
        model.fit(X_scaled, y)
        
        # Get model performance
        y_pred = model.predict(X_scaled)
        train_mse = mean_squared_error(y, y_pred)
        train_r2 = r2_score(y, y_pred)
        
        # Count non-zero coefficients
        n_snps_in_model = np.sum(model.coef_ != 0)
        
        model_performance = {
            'cv_mse': cv_mse,
            'cv_r2': cv_r2_mean,
            'train_mse': train_mse,
            'train_r2': train_r2,
            'n_snps_in_model': n_snps_in_model,
            'n_snps_total': X.shape[1],
            'converged': model.n_iter_ < model.max_iter
        }
        
    else:
        # Perform grid search for hyperparameter tuning
        param_grid = hyperparameters
        model = GridSearchCV(
            ElasticNet(random_state=42),
            param_grid=param_grid,
            cv=cv_folds,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        # Fit grid search
        model.fit(X_scaled, y)
        
        # Get best model
        best_model = model.best_estimator_
        
        # Get model performance
        y_pred = best_model.predict(X_scaled)
        train_mse = mean_squared_error(y, y_pred)
        train_r2 = r2_score(y, y_pred)
        
        # Count non-zero coefficients
        n_snps_in_model = np.sum(best_model.coef_ != 0)
        
        model_performance = {
            'cv_mse': -model.best_score_,
            'cv_r2': cross_val_score(best_model, X_scaled, y, cv=cv_folds, scoring='r2').mean(),
            'train_mse': train_mse,
            'train_r2': train_r2,
            'n_snps_in_model': n_snps_in_model,
            'n_snps_total': X.shape[1],
            'best_params': model.best_params_,
            'converged': best_model.n_iter_ < best_model.get_params()['max_iter']
        }
        
        # Update model to the best model
        model = best_model
    
    # Save scaler for prediction
    model_data = {
        'model': model,
        'scaler': scaler,
        'performance': model_performance
    }
    
    return model_data

def main():
    """Main function to train models for genes in a partition."""
    # Get input files from Snakemake
    genotype_file = snakemake.input.genotypes
    expression_file = snakemake.input.expression
    gene_annot_file = snakemake.input.gene_annot
    partition_file = snakemake.input.partition
    
    # Get output files from Snakemake
    model_dir = snakemake.output.model_dir
    summary_file = snakemake.output.summary
    
    # Get parameters from Snakemake
    min_r2 = snakemake.params.get('min_r2', 0.01)
    hyperparameters = snakemake.params.get('hyperparameters', {
        'alpha': 0.01,
        'l1_ratio': 0.5,
        'max_iter': 1000,
        'tol': 1e-4
    })
    
    # Create output directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Load gene annotation
    logger.info(f"Loading gene annotation from {gene_annot_file}")
    gene_annot = pd.read_csv(gene_annot_file, sep='\t')
    
    # Load partition
    logger.info(f"Loading partition from {partition_file}")
    with open(partition_file, 'r') as f:
        gene_list = json.load(f)
    
    logger.info(f"Partition contains {len(gene_list)} genes")
    
    # Preprocess data for this partition
    genotype_data, expression_data = preprocess_data(genotype_file, expression_file, gene_list)
    
    # Summary dataframe to store model performance
    summary_data = []
    
    # Train models for each gene in the partition
    for gene_id in gene_list:
        logger.info(f"Training model for gene {gene_id}")
        
        # Get gene expression
        if gene_id not in expression_data.index:
            logger.warning(f"Gene {gene_id} not found in expression data, skipping")
            continue
        
        gene_expression = expression_data.loc[gene_id]
        
        # Get cis-SNPs for this gene
        cis_genotype = get_cis_snps(gene_id, gene_annot, genotype_data)
        
        if cis_genotype is None or cis_genotype.shape[0] == 0:
            logger.warning(f"No cis-SNPs found for gene {gene_id}, skipping")
            continue
        
        # Transpose genotype data for model training (samples as rows)
        X = cis_genotype.T
        y = gene_expression
        
        # Train model
        model_data = train_model(X, y, hyperparameters)
        
        # Save model if it meets minimum R2 threshold
        if model_data['performance']['cv_r2'] >= min_r2:
            model_file = os.path.join(model_dir, f"{gene_id}.pkl")
            with open(model_file, 'wb') as f:
                pickle.dump(model_data, f)
            
            # Add performance to summary
            perf = model_data['performance']
            summary_data.append({
                'gene_id': gene_id,
                'cv_r2': perf['cv_r2'],
                'cv_mse': perf['cv_mse'],
                'train_r2': perf['train_r2'],
                'train_mse': perf['train_mse'],
                'n_snps_in_model': perf['n_snps_in_model'],
                'n_snps_total': perf['n_snps_total'],
                'converged': perf['converged']
            })
        else:
            logger.info(f"Model for gene {gene_id} did not meet minimum R2 threshold ({model_data['performance']['cv_r2']} < {min_r2})")
    
    # Save summary
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_file, sep='\t', index=False)
    
    logger.info(f"Trained {len(summary_data)} models that met minimum R2 threshold")
    logger.info(f"Average CV R2: {summary_df['cv_r2'].mean():.4f}")

if __name__ == "__main__":
    main() 