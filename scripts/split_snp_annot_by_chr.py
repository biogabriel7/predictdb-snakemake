#!/usr/bin/env python3

"""
Script to split a SNP annotation file into multiple files by chromosome.
Optimized with Numba for Apple M3 chip performance.
"""

import sys
import os
import numpy as np
import numba as nb
from numba import prange
from concurrent.futures import ProcessPoolExecutor
import gzip

# Get input and output files from Snakemake
snp_annot_file = snakemake.input.snp_annotation
output_files = snakemake.output.snp_files
n_threads = snakemake.threads if hasattr(snakemake, 'threads') else 4

# Define SNP complement dictionary for filtering
SNP_COMPLEMENT = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
HEADER_FIELDS = ['chr', 'pos', 'varID', 'refAllele', 'effectAllele', 'rsid']

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

def is_gzipped(filename):
    """Check if a file is gzipped by examining its magic number."""
    with open(filename, 'rb') as f:
        return f.read(2) == b'\x1f\x8b'

def open_file(filename, mode='r'):
    """Open a file, handling gzipped files if needed."""
    if is_gzipped(filename) and 'b' not in mode:
        return gzip.open(filename, mode + 't')
    elif is_gzipped(filename) and 'b' in mode:
        return gzip.open(filename, mode)
    else:
        return open(filename, mode)

def process_snp_chunk(chunk_data):
    """
    Process a chunk of SNP annotation file and filter SNPs.
    
    Args:
        chunk_data: Tuple of (chunk_id, lines)
    
    Returns:
        Dictionary mapping chromosome number to filtered lines
    """
    chunk_id, lines = chunk_data
    
    # Dictionary to store filtered lines by chromosome
    filtered_by_chr = {i: [] for i in range(1, 23)}
    
    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue
        
        # Parse the line
        attrs = line.split('\t')
        
        # Skip if line doesn't have enough fields
        if len(attrs) < 7:
            continue
        
        # Extract fields
        chrom = attrs[0].replace('chr', '')
        pos = attrs[1]
        var_id = attrs[2]
        ref_allele = attrs[3]
        effect_allele = attrs[4]
        rsid = attrs[6]
        
        # Skip non-numeric chromosomes or chromosomes outside 1-22
        try:
            chrom_num = int(chrom)
            if chrom_num < 1 or chrom_num > 22:
                continue
        except ValueError:
            continue
        
        # Skip non-single letter polymorphisms
        if len(ref_allele) > 1 or len(effect_allele) > 1:
            continue
        
        # Skip ambiguous strands
        if ref_allele not in SNP_COMPLEMENT or is_ambiguous_strand(ref_allele, effect_allele):
            continue
        
        # Skip missing rsids
        if rsid == '.':
            continue
        
        # Create output row
        row = '\t'.join([chrom, pos, var_id, ref_allele, effect_allele, rsid]) + '\n'
        
        # Add to filtered lines for this chromosome
        filtered_by_chr[chrom_num].append(row)
    
    return filtered_by_chr

def split_snp_annot_parallel(annot_file, out_files, n_workers=4):
    """
    Split a SNP annotation file into multiple files by chromosome using parallel processing.
    
    Args:
        annot_file: Path to input SNP annotation file
        out_files: List of output file paths
        n_workers: Number of worker processes
    """
    print(f"Splitting SNP annotation file: {annot_file}")
    print(f"Using {n_workers} worker processes")
    
    # Create output file connections
    out_handles = {}
    for i in range(1, 23):
        out_file = next((f for f in out_files if f'chr{i}' in f), None)
        if out_file:
            out_handles[i] = open(out_file, 'w')
    
    # Write header to all output files
    header = '\t'.join(HEADER_FIELDS) + '\n'
    for handle in out_handles.values():
        handle.write(header)
    
    # Estimate file size to determine chunk size
    if is_gzipped(annot_file):
        # For gzipped files, use a fixed chunk size
        chunk_size = 100000
    else:
        file_size = os.path.getsize(annot_file)
        # Determine chunk size based on file size (aim for ~100MB chunks)
        chunk_size = max(100000, min(1000000, file_size // (100 * n_workers)))
    
    print(f"Chunk size: {chunk_size} lines")
    
    # Process file in chunks
    chunks_to_process = []
    with open_file(annot_file, 'r') as f:
        # Skip header
        f.readline()
        
        chunk_id = 0
        current_chunk = []
        lines_in_chunk = 0
        
        print("Reading and splitting file into chunks...")
        for line in f:
            current_chunk.append(line)
            lines_in_chunk += 1
            
            if lines_in_chunk >= chunk_size:
                chunks_to_process.append((chunk_id, current_chunk))
                chunk_id += 1
                current_chunk = []
                lines_in_chunk = 0
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks_to_process.append((chunk_id, current_chunk))
    
    print(f"Created {len(chunks_to_process)} chunks to process")
    
    # Process chunks in parallel
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(process_snp_chunk, chunk) for chunk in chunks_to_process]
        
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
    print(f"Split SNP annotation file into {len(out_files)} chromosome files")
    print(f"Total SNPs processed: {total_snps}")
    for chrom, count in snp_counts.items():
        print(f"  Chromosome {chrom}: {count} SNPs")

def main():
    """Main function to split SNP annotation file with parallel processing."""
    # Get output directory and filename prefix
    out_dir = os.path.dirname(output_files[0])
    out_prefix = os.path.join(out_dir, "snp_annot")
    
    # Create output file paths
    out_files = [f"{out_prefix}.chr{i}.txt" for i in range(1, 23)]
    
    # Split SNP annotation file with parallel processing
    split_snp_annot_parallel(snp_annot_file, out_files, n_workers=min(n_threads, 8))

if __name__ == "__main__":
    main()