#!/usr/bin/env python3
"""
Create balanced partitions of genes for parallelized model training.
This helps distribute computational load evenly across jobs.
"""

import argparse
import json
import logging
import numpy as np
import os
import pandas as pd
from collections import defaultdict

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

def get_gene_snp_counts(gene_annot_file, snp_annot_file, cis_window=1000000):
    """
    Calculate the number of cis-SNPs for each gene.
    
    Args:
        gene_annot_file: Path to gene annotation file
        snp_annot_file: Path to SNP annotation file
        cis_window: Size of cis-window around TSS (default: 1Mb)
        
    Returns:
        pandas.DataFrame: DataFrame with gene_id and snp_count columns
    """
    logger.info(f"Loading gene annotation from {gene_annot_file}")
    gene_annot = pd.read_csv(gene_annot_file, sep='\t')
    
    logger.info(f"Loading SNP annotation from {snp_annot_file}")
    snp_annot = pd.read_csv(snp_annot_file, sep='\t')
    
    # Create dictionary to store SNP counts for each gene
    gene_snp_counts = defaultdict(int)
    chromosome_snp_counts = defaultdict(int)
    
    # Count SNPs per chromosome for reporting
    for chromosome in snp_annot['chromosome'].unique():
        chromosome_snps = snp_annot[snp_annot['chromosome'] == chromosome]
        chromosome_snp_counts[chromosome] = len(chromosome_snps)
        
    logger.info(f"SNP counts per chromosome: {dict(chromosome_snp_counts)}")
    
    # For each gene, count SNPs in cis-window
    for i, gene in gene_annot.iterrows():
        gene_id = gene['gene_id']
        chromosome = gene['chromosome']
        tss = gene['tss']
        
        # Get SNPs on same chromosome and within cis-window
        cis_snps = snp_annot[
            (snp_annot['chromosome'] == chromosome) &
            (snp_annot['position'] >= tss - cis_window) &
            (snp_annot['position'] <= tss + cis_window)
        ]
        
        gene_snp_counts[gene_id] = len(cis_snps)
        
        if (i + 1) % 1000 == 0:
            logger.info(f"Processed {i+1}/{len(gene_annot)} genes")
    
    # Convert to DataFrame
    counts_df = pd.DataFrame({
        'gene_id': list(gene_snp_counts.keys()),
        'snp_count': list(gene_snp_counts.values())
    })
    
    # Filter genes with no SNPs
    counts_df = counts_df[counts_df['snp_count'] > 0]
    
    logger.info(f"Found {len(counts_df)} genes with at least one cis-SNP")
    logger.info(f"Average SNP count per gene: {counts_df['snp_count'].mean():.2f}")
    logger.info(f"Total SNP count across all genes: {counts_df['snp_count'].sum()}")
    
    return counts_df, chromosome_snp_counts

def create_balanced_partitions(gene_snp_counts, chromosome_snp_counts, 
                               n_partitions=10, min_snps_per_partition=None, 
                               max_snps_per_partition=None, min_genes_per_partition=None):
    """
    Create balanced partitions of genes based on SNP counts.
    
    Args:
        gene_snp_counts: DataFrame with gene_id and snp_count columns
        chromosome_snp_counts: Dictionary with SNP counts per chromosome
        n_partitions: Number of partitions to create
        min_snps_per_partition: Minimum number of SNPs per partition
        max_snps_per_partition: Maximum number of SNPs per partition
        min_genes_per_partition: Minimum number of genes per partition
        
    Returns:
        list: List of lists, where each inner list contains gene IDs for a partition
    """
    # Sort genes by SNP count (descending)
    sorted_genes = gene_snp_counts.sort_values('snp_count', ascending=False).copy()
    
    # Initialize partitions
    partitions = [[] for _ in range(n_partitions)]
    partition_snp_counts = [0] * n_partitions
    
    # Calculate constraints if not provided
    total_snps = sorted_genes['snp_count'].sum()
    total_genes = len(sorted_genes)
    
    if min_snps_per_partition is None:
        min_snps_per_partition = total_snps // (n_partitions * 2)  # At least half the average
    
    if max_snps_per_partition is None:
        max_snps_per_partition = total_snps // (n_partitions // 2)  # At most twice the average
    
    if min_genes_per_partition is None:
        min_genes_per_partition = total_genes // (n_partitions * 2)  # At least half the average
    
    logger.info(f"Creating {n_partitions} partitions with constraints:")
    logger.info(f"  Min SNPs per partition: {min_snps_per_partition}")
    logger.info(f"  Max SNPs per partition: {max_snps_per_partition}")
    logger.info(f"  Min genes per partition: {min_genes_per_partition}")
    
    # Add genes to partitions, starting from the ones with the most SNPs
    for i, row in sorted_genes.iterrows():
        gene_id = row['gene_id']
        snp_count = row['snp_count']
        
        # Find partition with the lowest SNP count
        min_idx = np.argmin(partition_snp_counts)
        
        # Check if adding this gene would exceed max_snps_per_partition
        if partition_snp_counts[min_idx] + snp_count > max_snps_per_partition:
            # Try finding another partition that can accommodate this gene
            valid_partitions = [
                idx for idx, count in enumerate(partition_snp_counts)
                if count + snp_count <= max_snps_per_partition
            ]
            
            if valid_partitions:
                # Find partition with lowest SNP count among valid ones
                min_idx = valid_partitions[np.argmin([partition_snp_counts[idx] for idx in valid_partitions])]
            else:
                # No partition can accommodate this gene, try next gene
                logger.warning(f"Gene {gene_id} with {snp_count} SNPs could not be assigned to any partition")
                continue
        
        # Add gene to partition
        partitions[min_idx].append(gene_id)
        partition_snp_counts[min_idx] += snp_count
    
    # Check constraints
    for i, partition in enumerate(partitions):
        partition_snps = partition_snp_counts[i]
        partition_genes = len(partition)
        
        logger.info(f"Partition {i+1}: {partition_genes} genes, {partition_snps} SNPs")
        
        if partition_snps < min_snps_per_partition:
            logger.warning(f"Partition {i+1} has fewer SNPs than minimum ({partition_snps} < {min_snps_per_partition})")
        
        if partition_genes < min_genes_per_partition:
            logger.warning(f"Partition {i+1} has fewer genes than minimum ({partition_genes} < {min_genes_per_partition})")
    
    return partitions

def main():
    """Main function to create balanced partitions of genes."""
    # Get input files from Snakemake
    gene_annot_file = snakemake.input.gene_annot
    snp_annot_file = snakemake.input.snp_annot
    
    # Get output file from Snakemake
    partitions_dir = snakemake.output.partitions_dir
    
    # Get parameters from Snakemake
    n_partitions = snakemake.params.get('n_partitions', 10)
    cis_window = snakemake.params.get('cis_window', 1000000)
    min_snps_per_partition = snakemake.params.get('min_snps_per_partition', None)
    max_snps_per_partition = snakemake.params.get('max_snps_per_partition', None)
    min_genes_per_partition = snakemake.params.get('min_genes_per_partition', None)
    
    # Create output directory
    os.makedirs(partitions_dir, exist_ok=True)
    
    # Get gene SNP counts
    gene_snp_counts, chromosome_snp_counts = get_gene_snp_counts(
        gene_annot_file, snp_annot_file, cis_window
    )
    
    # Create balanced partitions
    partitions = create_balanced_partitions(
        gene_snp_counts, chromosome_snp_counts,
        n_partitions, min_snps_per_partition,
        max_snps_per_partition, min_genes_per_partition
    )
    
    # Save partitions to files
    for i, partition in enumerate(partitions):
        partition_file = os.path.join(partitions_dir, f"partition_{i+1}.json")
        with open(partition_file, 'w') as f:
            json.dump(partition, f, indent=2)
        
        logger.info(f"Saved partition {i+1} with {len(partition)} genes to {partition_file}")
    
    # Create a file with all partitions
    all_partitions_file = os.path.join(partitions_dir, "all_partitions.json")
    with open(all_partitions_file, 'w') as f:
        json.dump(partitions, f, indent=2)
    
    logger.info(f"Saved all partitions to {all_partitions_file}")

if __name__ == "__main__":
    main() 