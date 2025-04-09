#!/usr/bin/env python3

"""
Script to filter the PredictDb database based on model performance.
Replaces R-based filter_db.R functionality.
"""

import pandas as pd
import sqlite3
import os
import sys

# Get input and output files from Snakemake
db_file = snakemake.input.whole_db
filtered_db_file = snakemake.output.filtered_db
r2_threshold = snakemake.params.r2_threshold
pval_threshold = snakemake.params.pval_threshold

def filter_database(db_file, filtered_db_file, r2_threshold, pval_threshold):
    """
    Filter database based on model performance thresholds.
    
    Args:
        db_file: Path to input database file
        filtered_db_file: Path to output filtered database file
        r2_threshold: Minimum R² threshold for models
        pval_threshold: Maximum p-value threshold for models
    """
    print(f"Filtering database with thresholds: R² > {r2_threshold}, p-value < {pval_threshold}")
    
    # Connect to input database
    in_conn = sqlite3.connect(db_file)
    
    # Get model summaries
    model_summaries = pd.read_sql("SELECT * FROM extra", in_conn)
    
    # Apply filters
    filtered_models = model_summaries[
        (model_summaries['pred.perf.R2'] > r2_threshold) & 
        (model_summaries['pred.perf.pval'] < pval_threshold)
    ]
    
    # Get list of genes that pass filters
    filtered_genes = filtered_models['gene'].tolist()
    
    print(f"Found {len(filtered_genes)} genes passing thresholds out of {len(model_summaries)}")
    
    # Create new database with filtered models
    os.system(f"cp {db_file} {filtered_db_file}")
    out_conn = sqlite3.connect(filtered_db_file)
    
    # Replace extra table with filtered models
    filtered_models.to_sql('extra', out_conn, if_exists='replace', index=False)
    out_conn.execute("CREATE INDEX gene_model_summary ON extra (gene)")
    
    # Filter weights table to keep only weights for filtered genes
    weights = pd.read_sql("SELECT * FROM weights", in_conn)
    filtered_weights = weights[weights['gene'].isin(filtered_genes)]
    
    # Replace weights table
    filtered_weights.to_sql('weights', out_conn, if_exists='replace', index=False)
    out_conn.execute("CREATE INDEX weights_rsid ON weights (rsid)")
    out_conn.execute("CREATE INDEX weights_gene ON weights (gene)")
    out_conn.execute("CREATE INDEX weights_rsid_gene ON weights (rsid, gene)")
    
    # Close connections
    in_conn.close()
    out_conn.close()
    
    print(f"Filtered database saved to {filtered_db_file}")
    print(f"Retained {len(filtered_weights)} weights out of {len(weights)}")

def main():
    """Main function."""
    filter_database(db_file, filtered_db_file, r2_threshold, pval_threshold)

if __name__ == "__main__":
    main()