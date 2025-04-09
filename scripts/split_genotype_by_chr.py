#!/usr/bin/env python3

"""
Script to split a genotype file into multiple files by chromosome.
Adapted from the original split_genotype_by_chr.py script.
"""

import os
import sys

# Get input and output files from Snakemake
genotype_file = snakemake.input.genotype
out_prefix = os.path.join(os.path.dirname(snakemake.output.genotype_files[0]), "genotype")

# Define SNP complement dictionary for filtering
SNP_COMPLEMENT = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}

def split_genotype(geno_file, out_prefix):
    """
    Split a genotype file into multiple files by chromosome.
    
    Args:
        geno_file: Path to input genotype file
        out_prefix: Prefix for output files
    """
    # Make output file names from prefix
    geno_by_chr_fns = [f"{out_prefix}.chr{i}.txt" for i in range(1, 23)]
    
    # Open connection to each output file
    geno_by_chr = [open(f, 'w') for f in geno_by_chr_fns]
    
    with open(geno_file, 'r') as geno:
        # Write header in each file
        header = geno.readline()
        snps = set()
        for f in geno_by_chr:
            f.write(header)
        
        # Process each line
        for line in geno:
            # First attribute of line is chr_pos_refAllele_effAllele_build
            # Extract this attribute and parse into list
            varID_list = line.split()[0].split('_')
            chr = varID_list[0]
            
            # Skip if we don't have at least 4 elements in the variant ID
            if len(varID_list) < 4:
                continue
                
            refAllele = varID_list[2]
            effectAllele = varID_list[3]
            
            # Skip non-single letter polymorphisms
            if len(refAllele) > 1 or len(effectAllele) > 1:
                continue
                
            # Skip ambiguous strands
            if refAllele in SNP_COMPLEMENT and SNP_COMPLEMENT[refAllele] == effectAllele:
                continue
                
            varID = '_'.join(varID_list)
            
            # Some SNPs have 2 rows for some reason. Attributes are nearly
            # identical. Only keep the first one found.
            if varID in snps:
                continue
                
            snps.add(varID)
            
            # Write line to appropriate file
            if "chr" in chr:
                chr = chr.replace("chr", "")
                
            try:
                index = int(chr) - 1
                if 0 <= index < 22:  # Valid chromosome index (1-22)
                    geno_by_chr[index].write(line)
            except (ValueError, IndexError):
                # Skip if chromosome number is invalid
                continue
    
    # Close all output files
    for f in geno_by_chr:
        f.close()
    
    print(f"Split genotype file into {len(geno_by_chr_fns)} chromosome files")
    print(f"Processed {len(snps)} unique SNPs")

def main():
    """Main function."""
    split_genotype(genotype_file, out_prefix)

if __name__ == "__main__":
    main()