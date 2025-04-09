"""
Utility module for validating input files.
"""
import os
import pandas as pd
import numpy as np
from .logger import setup_logger

logger = setup_logger(__name__)

def validate_file_exists(file_path, description="File"):
    """
    Check if a file exists and is not empty.
    
    Args:
        file_path: Path to the file
        description: Description of the file for logging
        
    Returns:
        Boolean indicating if file exists and is not empty
    """
    if not file_path:
        logger.error(f"{description} path is empty")
        return False
    
    if not os.path.exists(file_path):
        logger.error(f"{description} not found: {file_path}")
        return False
    
    if os.path.getsize(file_path) == 0:
        logger.error(f"{description} is empty: {file_path}")
        return False
    
    return True

def validate_gene_annotation_file(file_path):
    """
    Validate gene annotation file format.
    
    Args:
        file_path: Path to the gene annotation file
        
    Returns:
        Boolean indicating if file is valid
    """
    if not validate_file_exists(file_path, "Gene annotation file"):
        return False
    
    try:
        df = pd.read_csv(file_path, sep='\t', nrows=5)
        
        required_columns = ['chr', 'gene_id', 'gene_name', 'start', 'end', 'strand']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Gene annotation file missing required columns: {', '.join(missing_columns)}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error validating gene annotation file: {str(e)}")
        return False

def validate_snp_annotation_file(file_path):
    """
    Validate SNP annotation file format.
    
    Args:
        file_path: Path to the SNP annotation file
        
    Returns:
        Boolean indicating if file is valid
    """
    if not validate_file_exists(file_path, "SNP annotation file"):
        return False
    
    try:
        df = pd.read_csv(file_path, sep='\t', nrows=5)
        
        required_columns = ['chr', 'pos', 'varID', 'refAllele', 'effectAllele', 'rsid']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"SNP annotation file missing required columns: {', '.join(missing_columns)}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error validating SNP annotation file: {str(e)}")
        return False

def validate_genotype_file(file_path, sample_count=None):
    """
    Validate genotype file format.
    
    Args:
        file_path: Path to the genotype file
        sample_count: Expected number of samples
        
    Returns:
        Boolean indicating if file is valid
    """
    if not validate_file_exists(file_path, "Genotype file"):
        return False
    
    try:
        # Read just the header and first few rows
        from .file_handling import open_file
        with open_file(file_path, 'r') as f:
            header = f.readline().strip().split('\t')
            first_row = f.readline().strip().split('\t')
        
        if len(header) < 2:
            logger.error(f"Genotype file has insufficient columns: {len(header)}")
            return False
        
        if header[0] != 'ID':
            logger.warning(f"Genotype file first column is not 'ID': {header[0]}")
        
        sample_ids = header[1:]
        if sample_count and len(sample_ids) != sample_count:
            logger.warning(f"Genotype file has {len(sample_ids)} samples, expected {sample_count}")
        
        # Check that all values in first row (after ID) are numeric
        try:
            values = [float(x) for x in first_row[1:]]
            if not all(0 <= x <= 2 for x in values):
                logger.warning("Genotype values outside expected range [0, 2]")
        except ValueError:
            logger.error("Genotype file contains non-numeric values")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error validating genotype file: {str(e)}")
        return False

def validate_expression_file(file_path, sample_count=None):
    """
    Validate expression file format.
    
    Args:
        file_path: Path to the expression file
        sample_count: Expected number of samples
        
    Returns:
        Boolean indicating if file is valid
    """
    if not validate_file_exists(file_path, "Expression file"):
        return False
    
    try:
        # Read just the header to check columns
        df = pd.read_csv(file_path, sep='\t', nrows=0)
        
        if len(df.columns) < 2:
            logger.error(f"Expression file has insufficient columns: {len(df.columns)}")
            return False
        
        # Check sample count if provided
        if sample_count and len(df.columns) - 1 != sample_count:
            logger.warning(f"Expression file has {len(df.columns) - 1} samples, expected {sample_count}")
        
        return True
    except Exception as e:
        logger.error(f"Error validating expression file: {str(e)}")
        return False

def validate_covariates_file(file_path, sample_count=None):
    """
    Validate covariates file format.
    
    Args:
        file_path: Path to the covariates file
        sample_count: Expected number of samples
        
    Returns:
        Boolean indicating if file is valid
    """
    if not file_path:
        logger.info("No covariates file provided")
        return True
    
    if not validate_file_exists(file_path, "Covariates file"):
        return False
    
    try:
        df = pd.read_csv(file_path, sep='\t', index_col=0)
        
        if df.shape[1] == 0:
            logger.error("Covariates file has no sample columns")
            return False
        
        # Check sample count if provided
        if sample_count and df.shape[1] != sample_count:
            logger.warning(f"Covariates file has {df.shape[1]} samples, expected {sample_count}")
        
        # Check for non-numeric values
        if not df.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all():
            logger.error("Covariates file contains non-numeric values")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error validating covariates file: {str(e)}")
        return False

def validate_inputs(gene_annot_file, snp_file, genotype_file, expression_file, covariates_file=None):
    """
    Validate all input files for a model training run.
    
    Args:
        gene_annot_file: Path to gene annotation file
        snp_file: Path to SNP annotation file
        genotype_file: Path to genotype file
        expression_file: Path to expression file
        covariates_file: Path to covariates file (optional)
        
    Returns:
        Boolean indicating if all inputs are valid
    """
    valid = True
    
    # Validate gene annotation file
    if not validate_gene_annotation_file(gene_annot_file):
        valid = False
    
    # Validate SNP annotation file
    if not validate_snp_annotation_file(snp_file):
        valid = False
    
    # Validate genotype file
    if not validate_genotype_file(genotype_file):
        valid = False
    
    # Validate expression file
    if not validate_expression_file(expression_file):
        valid = False
    
    # Validate covariates file if provided
    if covariates_file and not validate_covariates_file(covariates_file):
        valid = False
    
    return valid 