"""
Utility module for creating balanced data partitions for more efficient processing.
"""
import os
import json
import pandas as pd
import numpy as np
from .logger import setup_logger

logger = setup_logger(__name__)

def create_balanced_partitions(gene_df, snp_df, target_partition_count=100):
    """
    Create partitions with similar computational load based on gene and SNP density.
    
    Args:
        gene_df: DataFrame with gene annotations
        snp_df: DataFrame with SNP annotations
        target_partition_count: Target number of partitions to create
        
    Returns:
        List of partition dictionaries with chromosome, region, and gene information
    """
    logger.info(f"Creating balanced partitions with target count of {target_partition_count}")
    partitions = []
    
    # Get gene counts per chromosome
    chrom_gene_counts = gene_df['chr'].value_counts().to_dict()
    logger.info(f"Gene distribution by chromosome: {chrom_gene_counts}")
    
    # Get SNP counts per chromosome
    chrom_snp_counts = snp_df['chr'].value_counts().to_dict()
    
    # Calculate computational load estimate per chromosome
    # (based on genes × SNPs as a proxy for computational complexity)
    chrom_load = {}
    for chrom in set(chrom_gene_counts.keys()) | set(chrom_snp_counts.keys()):
        gene_count = chrom_gene_counts.get(chrom, 0)
        snp_count = chrom_snp_counts.get(chrom, 0)
        chrom_load[chrom] = gene_count * snp_count if gene_count > 0 and snp_count > 0 else 0
    
    # Average load per partition
    total_load = sum(chrom_load.values())
    target_load_per_partition = total_load / target_partition_count if total_load > 0 else 0
    
    logger.info(f"Total computational load: {total_load}")
    logger.info(f"Target load per partition: {target_load_per_partition}")
    
    # For large chromosomes, split into regions by position
    for chrom, load in chrom_load.items():
        if load <= 0:
            continue
            
        if load > target_load_per_partition * 1.5:
            # This chromosome needs to be split into multiple partitions
            genes_in_chrom = gene_df[gene_df['chr'] == chrom]
            
            if len(genes_in_chrom) == 0:
                continue
                
            # Sort genes by position
            genes_sorted = genes_in_chrom.sort_values('start')
            
            # Calculate number of partitions needed for this chromosome
            n_partitions = max(1, int(round(load / target_load_per_partition)))
            
            logger.info(f"Splitting chromosome {chrom} into {n_partitions} partitions (load: {load})")
            
            # Split into approximately equal-sized partitions
            genes_per_partition = max(1, len(genes_sorted) // n_partitions)
            
            for i in range(n_partitions):
                start_idx = i * genes_per_partition
                end_idx = min((i + 1) * genes_per_partition, len(genes_sorted))
                
                if start_idx == end_idx:
                    continue
                    
                partition_genes = genes_sorted.iloc[start_idx:end_idx]
                region_start = partition_genes['start'].min()
                region_end = partition_genes['end'].max()
                
                # Add window around the region for SNPs in cis
                window_size = 1000000  # 1Mb window
                region_start = max(0, region_start - window_size)
                region_end = region_end + window_size
                
                partition = {
                    'chrom': chrom,
                    'start': int(region_start),
                    'end': int(region_end),
                    'n_genes': len(partition_genes),
                    'gene_ids': partition_genes['gene_id'].tolist(),
                    'partition_type': 'region'
                }
                
                partitions.append(partition)
        else:
            # Small chromosome, keep as one partition
            genes_in_chrom = gene_df[gene_df['chr'] == chrom]
            
            if len(genes_in_chrom) == 0:
                continue
            
            partition = {
                'chrom': chrom,
                'start': int(genes_in_chrom['start'].min()),
                'end': int(genes_in_chrom['end'].max()),
                'n_genes': len(genes_in_chrom),
                'gene_ids': genes_in_chrom['gene_id'].tolist(),
                'partition_type': 'chromosome'
            }
            
            partitions.append(partition)
    
    # Check if we have partitions
    if not partitions:
        logger.warning("No partitions created! Check if gene and SNP data is valid.")
        # Create fallback partitions based on chromosomes
        for chrom in sorted(set(gene_df['chr'].unique())):
            genes_in_chrom = gene_df[gene_df['chr'] == chrom]
            if len(genes_in_chrom) == 0:
                continue
                
            partition = {
                'chrom': chrom,
                'start': 0,
                'end': int(1e9),  # Large number to cover the whole chromosome
                'n_genes': len(genes_in_chrom),
                'gene_ids': genes_in_chrom['gene_id'].tolist(),
                'partition_type': 'chromosome_fallback'
            }
            partitions.append(partition)
    
    # Add partition IDs
    for i, partition in enumerate(partitions):
        partition['id'] = i
    
    logger.info(f"Created {len(partitions)} balanced partitions")
    
    # Log some stats about the partitions
    genes_per_partition = [p['n_genes'] for p in partitions]
    logger.info(f"Genes per partition - Min: {min(genes_per_partition)}, "
               f"Max: {max(genes_per_partition)}, "
               f"Avg: {sum(genes_per_partition)/len(partitions):.1f}")
    
    return partitions

def write_partitions_file(partitions, output_file):
    """
    Write partitions to a JSON file.
    
    Args:
        partitions: List of partition dictionaries
        output_file: Path to output JSON file
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(partitions, f, indent=2)
    
    logger.info(f"Wrote {len(partitions)} partitions to {output_file}")

def read_partitions_file(partitions_file):
    """
    Read partitions from a JSON file.
    
    Args:
        partitions_file: Path to JSON file with partitions
        
    Returns:
        List of partition dictionaries
    """
    with open(partitions_file, 'r') as f:
        partitions = json.load(f)
    
    logger.info(f"Read {len(partitions)} partitions from {partitions_file}")
    return partitions

def filter_snps_for_partition(snp_df, partition):
    """
    Filter SNP DataFrame to only include SNPs in the specified partition.
    
    Args:
        snp_df: DataFrame with SNP annotations
        partition: Partition dictionary with chromosome and region information
        
    Returns:
        Filtered SNP DataFrame
    """
    chrom = partition['chrom']
    start = partition['start']
    end = partition['end']
    
    # Filter SNPs by position
    filtered = snp_df[(snp_df['chr'] == chrom) & 
                     (snp_df['pos'] >= start) & 
                     (snp_df['pos'] <= end)]
    
    return filtered

def filter_genes_for_partition(gene_df, partition):
    """
    Filter gene DataFrame to only include genes in the specified partition.
    
    Args:
        gene_df: DataFrame with gene annotations
        partition: Partition dictionary with gene IDs
        
    Returns:
        Filtered gene DataFrame
    """
    gene_ids = partition['gene_ids']
    return gene_df[gene_df['gene_id'].isin(gene_ids)] 