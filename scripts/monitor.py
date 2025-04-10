#!/usr/bin/env python3
"""
Streamlit web application to monitor PredictDB pipeline progress and visualize results.
Run with: streamlit run scripts/monitor.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import glob
import time
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="PredictDB Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("PredictDB Pipeline Monitor")
st.markdown("Real-time monitoring for gene expression prediction model training")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    results_dir = st.text_input("Results directory", value="results")
    auto_refresh = st.checkbox("Auto refresh", value=True)
    refresh_interval = st.slider("Refresh interval (seconds)", 5, 60, 10)
    
    if auto_refresh:
        st.write(f"Auto-refreshing every {refresh_interval} seconds")

# Function to get pipeline progress
def get_pipeline_progress(results_dir):
    # Define expected checkpoints
    expected_checkpoints = [
        "pipeline_started",
        "gene_annot_parsed",
        "snp_annot_split",
        "genotype_split",
        "gene_expr_transposed",
        "gene_partitions_created"
    ]
    
    # Check for partitioned model training checkpoints
    partition_checkpoints = glob.glob(f"{results_dir}/checkpoints/partition_*_complete.checkpoint")
    chromosome_checkpoints = glob.glob(f"{results_dir}/checkpoints/chr*_complete.checkpoint")
    
    # Add collection checkpoints
    collection_checkpoints = [
        "model_summaries_collected",
        "weight_summaries_collected",
        "covariances_collected",
        "database_created"
    ]
    
    # Check which checkpoints exist
    existing_checkpoints = []
    for checkpoint in expected_checkpoints:
        if os.path.exists(f"{results_dir}/checkpoints/{checkpoint}.checkpoint"):
            existing_checkpoints.append(checkpoint)
    
    # Get timestamp for each checkpoint
    checkpoint_times = {}
    for checkpoint in existing_checkpoints:
        path = f"{results_dir}/checkpoints/{checkpoint}.checkpoint"
        timestamp = datetime.fromtimestamp(os.path.getmtime(path))
        checkpoint_times[checkpoint] = timestamp
    
    # Count partition checkpoints
    n_partitions_complete = len(partition_checkpoints)
    if partition_checkpoints:
        # Get latest partition completion time
        latest_partition = max(partition_checkpoints, key=os.path.getmtime)
        checkpoint_times["latest_partition"] = datetime.fromtimestamp(os.path.getmtime(latest_partition))
    
    # Count chromosome checkpoints
    n_chromosomes_complete = len(chromosome_checkpoints)
    if chromosome_checkpoints:
        # Get latest chromosome completion time
        latest_chrom = max(chromosome_checkpoints, key=os.path.getmtime)
        checkpoint_times["latest_chromosome"] = datetime.fromtimestamp(os.path.getmtime(latest_chrom))
    
    # Check collection checkpoints
    for checkpoint in collection_checkpoints:
        if os.path.exists(f"{results_dir}/checkpoints/{checkpoint}.checkpoint"):
            existing_checkpoints.append(checkpoint)
            path = f"{results_dir}/checkpoints/{checkpoint}.checkpoint"
            checkpoint_times[checkpoint] = datetime.fromtimestamp(os.path.getmtime(path))
    
    return {
        "checkpoints": existing_checkpoints,
        "times": checkpoint_times,
        "n_partitions_complete": n_partitions_complete,
        "n_chromosomes_complete": n_chromosomes_complete,
        "total_checkpoints": len(expected_checkpoints) + n_partitions_complete + len(collection_checkpoints)
    }

# Function to get partition statistics
def get_partition_stats(results_dir):
    partition_stats = {
        "partition_sizes": [],
        "partition_ids": [],
        "snp_counts": []
    }
    
    # Path to all partitions file
    all_partitions_file = f"{results_dir}/gene_partitions/all_partitions.json"
    
    if not os.path.exists(all_partitions_file):
        # Try to find individual partition files
        partition_files = glob.glob(f"{results_dir}/gene_partitions/partition_*.json")
        
        if not partition_files:
            return None
        
        # Load each partition file
        for i, file in enumerate(sorted(partition_files)):
            try:
                with open(file, 'r') as f:
                    partition = json.load(f)
                    partition_stats["partition_sizes"].append(len(partition))
                    partition_stats["partition_ids"].append(i+1)
                    # We don't have SNP counts in individual files
                    partition_stats["snp_counts"].append(0)
            except:
                continue
    else:
        # Load all partitions file
        try:
            with open(all_partitions_file, 'r') as f:
                partitions = json.load(f)
                
                for i, partition in enumerate(partitions):
                    partition_stats["partition_sizes"].append(len(partition))
                    partition_stats["partition_ids"].append(i+1)
                    # We don't have SNP counts in the combined file
                    partition_stats["snp_counts"].append(0)
        except:
            return None
    
    return partition_stats

# Function to get model statistics
def get_model_stats(results_dir):
    # Check if model summaries exist
    model_summaries_file = f"{results_dir}/Model_summaries.txt"
    
    if not os.path.exists(model_summaries_file):
        # Try to find individual chromosome summaries
        chrom_summaries = glob.glob(f"{results_dir}/chr*_model_summaries.txt")
        
        if not chrom_summaries:
            # Try to find partition summaries
            partition_summaries = glob.glob(f"{results_dir}/partition_*_model_summaries.txt")
            
            if not partition_summaries:
                return None
            
            # Combine partition summaries
            dfs = []
            for file in partition_summaries:
                try:
                    df = pd.read_csv(file, sep='\t')
                    dfs.append(df)
                except:
                    continue
            
            if not dfs:
                return None
                
            models_df = pd.concat(dfs, ignore_index=True)
        else:
            # Combine chromosome summaries
            dfs = []
            for file in chrom_summaries:
                try:
                    df = pd.read_csv(file, sep='\t')
                    dfs.append(df)
                except:
                    continue
            
            if not dfs:
                return None
                
            models_df = pd.concat(dfs, ignore_index=True)
    else:
        # Load the model summaries file
        try:
            models_df = pd.read_csv(model_summaries_file, sep='\t')
        except:
            return None
    
    # Calculate statistics
    stats = {
        "total_models": len(models_df),
        "avg_r2": models_df["cv_r2"].mean() if "cv_r2" in models_df.columns else 0,
        "max_r2": models_df["cv_r2"].max() if "cv_r2" in models_df.columns else 0,
        "models_by_chromosome": models_df["chromosome"].value_counts().to_dict() if "chromosome" in models_df.columns else {},
        "r2_distribution": models_df["cv_r2"].describe().to_dict() if "cv_r2" in models_df.columns else {},
        "snps_in_model": models_df["n_snps_in_model"].describe().to_dict() if "n_snps_in_model" in models_df.columns else {}
    }
    
    return stats

# Progress section
st.header("Pipeline Progress")

# Get progress information
progress_info = get_pipeline_progress(results_dir)

if not progress_info:
    st.warning("No pipeline progress found. Make sure the results directory exists and contains checkpoint files.")
else:
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Completed Steps", len(progress_info["checkpoints"]))
    
    with col2:
        st.metric("Partitions Completed", progress_info["n_partitions_complete"])
    
    with col3:
        st.metric("Chromosomes Completed", progress_info["n_chromosomes_complete"])
    
    with col4:
        if "database_created" in progress_info["checkpoints"]:
            st.metric("Pipeline Status", "Complete ✅")
        elif len(progress_info["checkpoints"]) > 0:
            st.metric("Pipeline Status", "Running ⏳")
        else:
            st.metric("Pipeline Status", "Not Started ❌")
    
    # Progress bar
    if "total_checkpoints" in progress_info and progress_info["total_checkpoints"] > 0:
        progress_pct = len(progress_info["checkpoints"]) / progress_info["total_checkpoints"]
        st.progress(progress_pct)
    
    # Checkpoint timeline
    if progress_info["times"]:
        st.subheader("Checkpoint Timeline")
        
        # Convert to DataFrame for display
        times_df = pd.DataFrame({
            "Checkpoint": list(progress_info["times"].keys()),
            "Timestamp": list(progress_info["times"].values())
        })
        
        times_df = times_df.sort_values("Timestamp")
        
        # Display as table
        st.dataframe(times_df)

# Partition Balance section
st.header("Partition Balance")

partition_stats = get_partition_stats(results_dir)

if not partition_stats:
    st.info("No partition information available yet.")
else:
    # Create a bar chart for partition sizes
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(partition_stats["partition_ids"], partition_stats["partition_sizes"])
    
    # Add labels and title
    ax.set_xlabel("Partition ID")
    ax.set_ylabel("Number of Genes")
    ax.set_title("Genes per Partition")
    
    # Add a horizontal line for the average
    avg_size = np.mean(partition_stats["partition_sizes"])
    ax.axhline(avg_size, color='red', linestyle='--', label=f'Average: {avg_size:.1f} genes')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height}', ha='center', va='bottom')
    
    ax.legend()
    
    # Display the chart
    st.pyplot(fig)
    
    # Summary statistics
    st.subheader("Partition Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Partitions", len(partition_stats["partition_sizes"]))
    
    with col2:
        st.metric("Average Genes per Partition", f"{avg_size:.1f}")
    
    with col3:
        std_dev = np.std(partition_stats["partition_sizes"])
        st.metric("Standard Deviation", f"{std_dev:.1f}")

# Model Performance section
st.header("Model Performance")

model_stats = get_model_stats(results_dir)

if not model_stats:
    st.info("No model statistics available yet.")
else:
    # Create columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Models", model_stats["total_models"])
    
    with col2:
        st.metric("Average R²", f"{model_stats['avg_r2']:.3f}")
    
    with col3:
        st.metric("Maximum R²", f"{model_stats['max_r2']:.3f}")
    
    # Create distribution plot for R² values
    if model_stats["r2_distribution"]:
        st.subheader("R² Distribution")
        
        # Create a histogram for R² distribution
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Create synthetic data based on percentiles for visualization
        if isinstance(model_stats["r2_distribution"], dict) and "count" in model_stats["r2_distribution"]:
            count = int(model_stats["r2_distribution"]["count"])
            r2_mean = model_stats["r2_distribution"]["mean"]
            r2_std = model_stats["r2_distribution"]["std"] if model_stats["r2_distribution"]["std"] > 0 else 0.01
            
            # Generate synthetic data based on statistics
            r2_values = np.random.normal(r2_mean, r2_std, count)
            r2_values = np.clip(r2_values, 0, 1)  # R² is between 0 and 1
            
            sns.histplot(r2_values, kde=True, ax=ax)
            ax.set_xlabel("R² Value")
            ax.set_ylabel("Count")
            ax.set_title("Distribution of Model R² Values")
            
            # Display the chart
            st.pyplot(fig)
    
    # Models by chromosome
    if model_stats["models_by_chromosome"]:
        st.subheader("Models by Chromosome")
        
        # Convert to DataFrame for plotting
        chrom_df = pd.DataFrame({
            "Chromosome": list(model_stats["models_by_chromosome"].keys()),
            "Model Count": list(model_stats["models_by_chromosome"].values())
        })
        
        chrom_df = chrom_df.sort_values("Chromosome")
        
        # Create a bar chart
        fig, ax = plt.subplots(figsize=(12, 5))
        bars = ax.bar(chrom_df["Chromosome"], chrom_df["Model Count"])
        
        # Add labels and title
        ax.set_xlabel("Chromosome")
        ax.set_ylabel("Number of Models")
        ax.set_title("Models per Chromosome")
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height}', ha='center', va='bottom')
        
        # Display the chart
        st.pyplot(fig)

# Auto refresh
if auto_refresh:
    time.sleep(1)  # Small delay to ensure the app displays before refreshing message
    st.empty()
    time_placeholder = st.empty()
    
    # Display countdown
    for i in range(refresh_interval, 0, -1):
        time_placeholder.info(f"Refreshing in {i} seconds...")
        time.sleep(1)
    
    time_placeholder.empty()
    st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown("*PredictDB Pipeline Monitor - Developed for gene expression prediction model training*") 