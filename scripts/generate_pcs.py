#!/usr/bin/env python3

"""
Script to generate principal components from gene expression data.
Replaces R-based generate_pcs.R functionality.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Get inputs from Snakemake
gene_expr_file = snakemake.input.gene_expr
covs_outfile = snakemake.output.pcs
n_components = snakemake.params.n_components

# Read gene expression data
gene_exp_transpose = pd.read_csv(gene_expr_file)

# Standardize the data
scaler = StandardScaler()
gene_exp_scaled = scaler.fit_transform(gene_exp_transpose)

# Compute the PCs
pca = PCA(n_components=n_components)
pcs = pca.fit_transform(gene_exp_scaled)

# Convert to dataframe
pc_columns = [f'PC{i+1}' for i in range(n_components)]
pcs_df = pd.DataFrame(pcs, columns=pc_columns, index=gene_exp_transpose.index)

# Transpose the PCs
pcs_transposed = pcs_df.T

# Write the covariates as a tab-delimited file
pcs_transposed.to_csv(covs_outfile, sep='\t')

print(f"Generated {n_components} principal components")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")