#!/usr/bin/env python3
"""
Utility module for plotting benchmark results.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
import sys
from .logger import setup_logger

logger = setup_logger(__name__)

def load_benchmark_files(benchmark_dir):
    """
    Load all benchmark files from directory.
    
    Args:
        benchmark_dir: Directory containing benchmark files
        
    Returns:
        DataFrame with combined benchmark data
    """
    benchmark_files = glob.glob(os.path.join(benchmark_dir, "*.txt"))
    
    results = []
    for file_path in benchmark_files:
        try:
            # Extract rule name from filename
            rule_name = os.path.basename(file_path).replace('.txt', '')
            
            # Read benchmark file
            df = pd.read_csv(file_path, sep='\t')
            
            # Add rule name column
            df['rule'] = rule_name
            
            results.append(df)
        except Exception as e:
            logger.error(f"Error reading {file_path}: {str(e)}")
    
    # Combine all benchmark results
    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()

def plot_benchmarks(benchmark_df, output_dir):
    """
    Generate benchmark visualizations.
    
    Args:
        benchmark_df: DataFrame with benchmark data
        output_dir: Directory to save visualizations
        
    Returns:
        DataFrame with benchmark summary
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if benchmark_df.empty:
        logger.warning("No benchmark data to plot")
        return pd.DataFrame()
    
    # Plot execution time by rule
    plt.figure(figsize=(12, 8))
    benchmark_summary = benchmark_df.groupby('rule')['s'].agg(['mean', 'min', 'max'])
    benchmark_summary.sort_values('mean', ascending=False).plot(kind='barh')
    plt.title('Execution Time by Rule')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Rule')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'execution_time.png'))
    
    # Plot memory usage by rule
    plt.figure(figsize=(12, 8))
    benchmark_summary = benchmark_df.groupby('rule')['max_rss'].agg(['mean', 'min', 'max']) / 1024  # Convert to MB
    benchmark_summary.sort_values('mean', ascending=False).plot(kind='barh')
    plt.title('Memory Usage by Rule')
    plt.xlabel('Memory (MB)')
    plt.ylabel('Rule')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'memory_usage.png'))
    
    # Plot CPU load by rule
    plt.figure(figsize=(12, 8))
    if 'mean_load' in benchmark_df.columns:
        benchmark_summary = benchmark_df.groupby('rule')['mean_load'].agg(['mean', 'min', 'max'])
        benchmark_summary.sort_values('mean', ascending=False).plot(kind='barh')
        plt.title('CPU Load by Rule')
        plt.xlabel('CPU Load')
        plt.ylabel('Rule')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cpu_load.png'))
    
    # Generate CSV summary
    benchmark_summary = benchmark_df.groupby('rule').agg({
        's': ['mean', 'min', 'max'],
        'h:m:s': ['count'],
        'max_rss': ['mean', 'max'],
        'max_vms': ['mean', 'max'] if 'max_vms' in benchmark_df.columns else None,
        'max_uss': ['mean', 'max'] if 'max_uss' in benchmark_df.columns else None,
        'max_pss': ['mean', 'max'] if 'max_pss' in benchmark_df.columns else None,
        'io_in': ['sum'] if 'io_in' in benchmark_df.columns else None,
        'io_out': ['sum'] if 'io_out' in benchmark_df.columns else None,
        'mean_load': ['mean', 'max'] if 'mean_load' in benchmark_df.columns else None
    })
    
    # Drop any None columns
    benchmark_summary = benchmark_summary.dropna(axis=1, how='all')
    
    # Convert memory metrics to MB
    for col in ['max_rss', 'max_vms', 'max_uss', 'max_pss']:
        for stat in ['mean', 'max']:
            if (col, stat) in benchmark_summary.columns:
                benchmark_summary[(col, stat)] /= 1024  # Convert to MB
    
    # Save summary to CSV
    benchmark_summary.to_csv(os.path.join(output_dir, 'benchmark_summary.csv'))
    
    logger.info(f"Benchmark visualizations saved to {output_dir}")
    
    return benchmark_summary

def main():
    """Main function for command-line execution."""
    if len(sys.argv) < 2:
        benchmark_dir = "benchmarks"
    else:
        benchmark_dir = sys.argv[1]
    
    output_dir = "results/benchmark_plots"
    
    logger.info(f"Loading benchmark data from {benchmark_dir}")
    
    # Load benchmark data
    benchmark_df = load_benchmark_files(benchmark_dir)
    
    if benchmark_df.empty:
        logger.error(f"No benchmark files found in {benchmark_dir}")
        return 1
    
    # Generate plots and summary
    benchmark_summary = plot_benchmarks(benchmark_df, output_dir)
    
    # Print summary
    logger.info("Benchmark Summary:")
    logger.info(f"\n{benchmark_summary}")
    
    logger.info(f"Benchmark visualizations saved to {output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 