#!/usr/bin/env python3

"""
Script to transpose gene expression data for use in modeling.
Replaces R-based transpose_gene_expr.R functionality.
"""

import pandas as pd
import os
import sys

# Get input and output files from Snakemake
infile = snakemake.input.gene_expr
outfile = snakemake.output.tr_expr
outfile_peer = snakemake.output.tr_expr_peer

def read_expression_file(file_path):
    """Read gene expression file in a flexible manner based on extension."""
    if file_path.endswith('.txt') or file_path.endswith('.tsv'):
        gene_exp = pd.read_csv(file_path, sep='\t')
    elif file_path.endswith('.csv'):
        gene_exp = pd.read_csv(file_path)
    else:
        raise ValueError(f"Invalid file format for {file_path}, check the extension (should be .txt, .tsv, or .csv)")
    
    return gene_exp

def transpose_expression(gene_exp):
    """
    Transpose the gene expression matrix.
    
    Args:
        gene_exp: DataFrame with gene expression data, where the first column is gene IDs
                  and remaining columns are sample expression values
    
    Returns:
        Transposed DataFrame with samples as rows and genes as columns
    """
    # Drop columns we don't need in the gene expression dataframe if they exist
    drop_cols = ['Chr', 'TargetID', 'Coord']
    for col in drop_cols:
        if col in gene_exp.columns:
            gene_exp = gene_exp.drop(columns=[col])
    
    # Get the gene identifiers from the first column
    gene_column_name = gene_exp.columns[0]
    gene_ids = gene_exp[gene_column_name].values
    
    # Transpose the expression values
    gene_exp_transpose = gene_exp.drop(columns=[gene_column_name]).T
    gene_exp_transpose.columns = gene_ids
    
    return gene_exp_transpose

def main():
    """Main function to read, process, and write gene expression data."""
    # Read input file
    gene_exp = read_expression_file(infile)
    
    # Transpose gene expression matrix
    gene_exp_transpose = transpose_expression(gene_exp)
    
    # Write out the transposed file in TSV format for downstream analysis
    gene_exp_transpose.to_csv(outfile, sep='\t')
    
    # Write out the transposed file in CSV format for PEER factors
    gene_exp_transpose.to_csv(outfile_peer, sep=',')
    
    print(f"Gene expression data transposed: {gene_exp_transpose.shape[0]} samples × {gene_exp_transpose.shape[1]} genes")

if __name__ == "__main__":
    main()