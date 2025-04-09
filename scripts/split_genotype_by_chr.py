#!/usr/bin/env python3

"""
Script to split a genotype file into multiple files by chromosome.
Optimized with Numba for Apple M3 chip performance.
"""

import os
import sys
import numpy as np
import numba as nb
from numba import prange
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

# Get input and output files from Snakemake
genotype_file = snakemake.input.genotype
output_files = snakemake.output.genotype_files
n_threads = snakemake.threads if hasattr(snakemake, 'threads') else 4

# Define SNP complement dictionary for filtering
SNP_COMPLEMENT = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}

@nb.njit
def is_ambiguous_strand(ref, alt):
    """
    Numba-optimized function to check if SNP has ambiguous strands.
    
    Args:
        ref: Reference allele
        alt: Alternative allele
    
    Returns:
        True if ambiguous (A/T or C/G), False otherwise
    """
    if ref == 'A' and alt == 'T':
        return True
    if ref == 'T' and alt == 'A':
        return True
    if ref == 'C' and alt == 'G':
        return True
    if ref == 'G' and alt == 'C':
        return True
    return False

@nb.njit
def is_single_letter(allele):
    """
    Numba-optimized function to check if allele is a single letter.
    
    Args:
        allele: Allele string
    
    Returns:
        True if single letter, False otherwise
    """
    return len(allele) == 1

def process_genotype_chunk(chunk_data):
    """
    Process a chunk of genotype file and filter SNPs.
    
    Args:
        chunk_data: Tuple of (chunk_id, lines, header)
    
    Returns:
        Dictionary mapping chromosome number to filtered lines
    """
    chunk_id, lines, header = chunk_data
    
    # Dictionary to store filtered lines by chromosome
    filtered_by_chr = {i: [] for i in range(1, 23)}
    seen_varids = set()
    
    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue
        
        # Parse the variant ID
        parts = line.split('\t')
        variant_parts = parts[0].split('_')
        
        # Skip if we don't have enough parts in the variant ID
        if len(variant_parts) < 4:
            continue
        
        # Extract chromosome and alleles
        chrom = variant_parts[0].replace('chr', '')
        ref_allele = variant_parts[2]
        alt_allele = variant_parts[3]
        
        # Skip non-numeric chromosomes or chromosomes outside 1-22
        try:
            chrom_num = int(chrom)
            if chrom_num < 1 or chrom_num > 22:
                continue
        except ValueError:
            continue
        
        # Skip non-single letter polymorphisms
        if len(ref_allele) > 1 or len(alt_allele) > 1:
            continue
        
        # Skip ambiguous strands
        if is_ambiguous_strand(ref_allele, alt_allele):
            continue
        
        # Skip duplicates
        var_id = '_'.join(variant_parts)
        if var_id in seen_varids:
            continue
        
        # Add to filtered lines for this chromosome
        seen_varids.add(var_id)
        filtered_by_chr[chrom_num].append(line)
    
    return filtered_by_chr

def split_genotype_parallel(geno_file, out_files, n_workers=4):
    """
    Split a genotype file into multiple files by chromosome using parallel processing.
    
    Args:
        geno_file: Path to input genotype file
        out_files: List of output file paths
        n_workers: Number of worker processes
    """
    print(f"Splitting genotype file: {geno_file}")
    print(f"Using {n_workers} worker processes")
    
    # Estimate file size to determine chunk size
    file_size = os.path.getsize(geno_file)
    
    # Determine chunk size based on file size (aim for ~100MB chunks)
    chunk_size = max(100000, min(1000000, file_size // (100 * n_workers)))
    print(f"File size: {file_size / (1024*1024):.2f} MB")
    print(f"Chunk size: {chunk_size} lines")
    
    # Create output file connections
    out_handles = {}
    for i in range(1, 23):
        out_file = next((f for f in out_files if f'chr{i}' in f), None)
        if out_file:
            out_handles[i] = open(out_file, 'w')
    
    # Read and process header
    with open(geno_file, 'r') as f:
        header = f.readline()
        
        # Write header to all output files
        for handle in out_handles.values():
            handle.write(header)
        
        # Process file in chunks
        chunk_id = 0
        chunks_to_process = []
        current_chunk = []
        lines_in_chunk = 0
        
        print("Reading and splitting file into chunks...")
        for line in f:
            current_chunk.append(line)
            lines_in_chunk += 1
            
            if lines_in_chunk >= chunk_size:
                chunks_to_process.append((chunk_id, current_chunk, header))
                chunk_id += 1
                current_chunk = []
                lines_in_chunk = 0
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks_to_process.append((chunk_id, current_chunk, header))
    
    print(f"Created {len(chunks_to_process)} chunks to process")
    
    # Process chunks in parallel
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(process_genotype_chunk, chunk) for chunk in chunks_to_process]
        
        # Collect results as they complete
        for i, future in enumerate(futures):
            result = future.result()
            results.append(result)
            print(f"Processed chunk {i+1}/{len(chunks_to_process)}")
    
    # Combine results and write to output files
    print("Writing filtered SNPs to chromosome files...")
    snp_counts = {i: 0 for i in range(1, 23)}
    
    for result in results:
        for chrom, lines in result.items():
            if chrom in out_handles:
                for line in lines:
                    out_handles[chrom].write(line)
                    snp_counts[chrom] += 1
    
    # Close output files
    for handle in out_handles.values():
        handle.close()
    
    # Print summary
    total_snps = sum(snp_counts.values())
    print(f"Split genotype file into {len(out_files)} chromosome files")
    print(f"Total SNPs processed: {total_snps}")
    for chrom, count in snp_counts.items():
        print(f"  Chromosome {chrom}: {count} SNPs")

def main():
    """Main function to split genotype file with Numba and parallel processing."""
    # Get output directory and filename prefix
    out_dir = os.path.dirname(output_files[0])
    out_prefix = os.path.join(out_dir, "genotype")
    
    # Create output file paths
    out_files = [f"{out_prefix}.chr{i}.txt" for i in range(1, 23)]
    
    # Split genotype file with parallel processing
    split_genotype_parallel(genotype_file, out_files, n_workers=min(n_threads, 8))

if __name__ == "__main__":
    main()