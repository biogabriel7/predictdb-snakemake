#!/usr/bin/env python3

"""
Script to parse GTF gene annotation files.
Extracts gene information and writes it to a tab-delimited output file.
"""

import re
import gzip
import os
import sys

# Get input and output files from Snakemake
gtf_file = snakemake.input.gtf
out_file = snakemake.output.parsed_annot

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

def parse_gtf_attribute(attribute_string):
    """Parse the attribute string from a GTF file line."""
    attributes = {}
    # Remove trailing semicolon if present
    if attribute_string.endswith(';'):
        attribute_string = attribute_string[:-1]
    
    # Split by semicolon and process each attribute
    for attribute in attribute_string.split(';'):
        attribute = attribute.strip()
        if not attribute:
            continue
        
        # Extract key and value
        match = re.match(r'(\S+)\s+"?([^"]+)"?', attribute)
        if match:
            key, value = match.groups()
            attributes[key] = value.strip('"')
    
    return attributes

def extract_gene_info(gtf_file, out_file):
    """Extract gene information from GTF file."""
    gene_info = {}
    
    # Process GTF file
    with open_file(gtf_file) as f:
        for line in f:
            # Skip comments
            if line.startswith('#'):
                continue
            
            # Parse the line
            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue
            
            chrom, source, feature, start, end, score, strand, frame, attribute_string = fields
            
            # Only process gene features
            if feature.lower() != 'gene':
                continue
            
            # Parse attributes
            attributes = parse_gtf_attribute(attribute_string)
            
            # Extract required information
            gene_id = attributes.get('gene_id', '')
            gene_name = attributes.get('gene_name', gene_id)
            
            # Skip if gene_id is missing
            if not gene_id:
                continue
            
            # Store gene information
            gene_info[gene_id] = {
                'chr': chrom,
                'gene_id': gene_id,
                'gene_name': gene_name,
                'start': int(start),
                'end': int(end),
                'strand': strand
            }
    
    # Write output file
    with open(out_file, 'w') as out:
        # Write header
        header = ['chr', 'gene_id', 'gene_name', 'start', 'end', 'strand']
        out.write('\t'.join(header) + '\n')
        
        # Write gene information
        for gene_id, info in gene_info.items():
            values = [str(info[field]) for field in header]
            out.write('\t'.join(values) + '\n')
    
    print(f"Extracted information for {len(gene_info)} genes")

if __name__ == "__main__":
    extract_gene_info(gtf_file, out_file)