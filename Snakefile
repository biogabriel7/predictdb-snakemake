# PredictDb Snakemake Workflow
# Optimized for MacBook M3 Pro

import os
import sys
import datetime
from email.mime.text import MIMEText
import smtplib

# Add scripts directory to Python path for importing utils
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))

# Configuration files
configfile: "config/config.yaml"
configfile: "config/resources.yaml"
configfile: "config/notifications.yaml"
configfile: "config/report.yaml"

# Import resource allocation utilities
from scripts.utils.resource_allocation import get_memory_mb, get_threads

# Create logs directory
os.makedirs("logs", exist_ok=True)
os.makedirs("results/checkpoints", exist_ok=True)
os.makedirs("benchmarks", exist_ok=True)

# Chromosome list
CHROMOSOMES = list(range(1, 23))

# Final output target
rule all:
    input:
        "results/predict_db_{config[prefix]}.db",
        "results/Covariances.txt",
        "results/report.html"

# Pipeline start checkpoint for resuming
checkpoint status:
    output:
        touch("results/checkpoints/pipeline_started.checkpoint")
    run:
        with open(output[0], "w") as f:
            f.write(f"Pipeline started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Preprocessing rules
rule parse_gene_annot:
    input:
        gtf = config["gene_annotation"],
        checkpoint = "results/checkpoints/pipeline_started.checkpoint"
    output:
        parsed_annot = "results/gene_annot.parsed.txt",
        checkpoint = touch("results/checkpoints/gene_annot_parsed.checkpoint")
    log:
        "logs/parse_gene_annot.log"
    benchmark:
        "benchmarks/parse_gene_annot.txt"
    resources:
        mem_mb = lambda wildcards, attempt: get_memory_mb(max_memory=resources["parse_gtf"]["memory_mb"]) * attempt,
        threads = lambda wildcards, attempt: min(get_threads(), resources["parse_gtf"]["threads"])
    threads: lambda wildcards, attempt: min(get_threads(), resources["parse_gtf"]["threads"])
    script:
        "scripts/parse_gtf.py"

rule split_snp_annot:
    input:
        snp_annot = config["snp_annotation"],
        checkpoint = "results/checkpoints/pipeline_started.checkpoint"
    output:
        snp_files = expand("results/snp_annot.chr{chrom}.txt", chrom=CHROMOSOMES),
        checkpoint = touch("results/checkpoints/snp_annot_split.checkpoint")
    log:
        "logs/split_snp_annot.log"
    benchmark:
        "benchmarks/split_snp_annot.txt"
    resources:
        mem_mb = lambda wildcards, attempt: get_memory_mb(max_memory=4000) * attempt,
        threads = lambda wildcards, attempt: min(get_threads(), 2)
    threads: lambda wildcards, attempt: min(get_threads(), 2)
    script:
        "scripts/split_snp_annot_by_chr.py"

rule split_genotype:
    input:
        genotype = config["genotype"],
        checkpoint = "results/checkpoints/pipeline_started.checkpoint"
    output:
        genotype_files = expand("results/genotype.chr{chrom}.txt", chrom=CHROMOSOMES),
        checkpoint = touch("results/checkpoints/genotype_split.checkpoint")
    log:
        "logs/split_genotype.log"
    benchmark:
        "benchmarks/split_genotype.txt"
    resources:
        mem_mb = lambda wildcards, attempt: get_memory_mb(max_memory=resources["split_genotype"]["memory_mb"]) * attempt,
        threads = lambda wildcards, attempt: min(get_threads(), resources["split_genotype"]["threads"])
    threads: lambda wildcards, attempt: min(get_threads(), resources["split_genotype"]["threads"])
    script:
        "scripts/split_genotype_by_chr.py"

rule transpose_gene_expr:
    input:
        gene_expr = config["gene_exp"],
        checkpoint = "results/checkpoints/pipeline_started.checkpoint"
    output:
        tr_expr = "results/gene_expression.txt",
        tr_expr_peer = "results/gene_expression_for_peer.csv",
        checkpoint = touch("results/checkpoints/gene_expr_transposed.checkpoint")
    log:
        "logs/transpose_gene_expr.log"
    benchmark:
        "benchmarks/transpose_gene_expr.txt"
    resources:
        mem_mb = lambda wildcards, attempt: get_memory_mb(max_memory=resources["transpose_gene_expr"]["memory_mb"]) * attempt,
        threads = lambda wildcards, attempt: min(get_threads(), resources["transpose_gene_expr"]["threads"])
    threads: lambda wildcards, attempt: min(get_threads(), resources["transpose_gene_expr"]["threads"])
    script:
        "scripts/transpose_gene_expr.py"

# Covariate calculations
rule generate_peer_factors:
    input:
        gene_expr = "results/gene_expression_for_peer.csv",
        checkpoint = "results/checkpoints/gene_expr_transposed.checkpoint"
    output:
        peers = "results/PEER_covariates.txt",
        checkpoint = touch("results/checkpoints/peer_factors_generated.checkpoint")
    log:
        "logs/generate_peer_factors.log"
    benchmark:
        "benchmarks/generate_peer_factors.txt"
    params:
        n_factors = config["n_peer_factors"]
    resources:
        mem_mb = lambda wildcards, attempt: get_memory_mb(max_memory=resources["generate_peer_factors"]["memory_mb"]) * attempt,
        threads = lambda wildcards, attempt: min(get_threads(), resources["generate_peer_factors"]["threads"])
    threads: lambda wildcards, attempt: min(get_threads(), resources["generate_peer_factors"]["threads"])
    script:
        "scripts/generate_peer_factors.py"

rule generate_pcs:
    input:
        gene_expr = "results/gene_expression_for_peer.csv",
        checkpoint = "results/checkpoints/gene_expr_transposed.checkpoint"
    output:
        pcs = "results/PCA_covariates.txt",
        checkpoint = touch("results/checkpoints/pcs_generated.checkpoint")
    log:
        "logs/generate_pcs.log"
    benchmark:
        "benchmarks/generate_pcs.txt"
    params:
        n_components = 10  # Default to 10 PCs
    resources:
        mem_mb = lambda wildcards, attempt: get_memory_mb(max_memory=resources["generate_pcs"]["memory_mb"]) * attempt,
        threads = lambda wildcards, attempt: min(get_threads(), resources["generate_pcs"]["threads"])
    threads: lambda wildcards, attempt: min(get_threads(), resources["generate_pcs"]["threads"])
    script:
        "scripts/generate_pcs.py"

rule combine_covariates:
    input:
        computed_covs = lambda wildcards: "results/PEER_covariates.txt" if config["peer"] else "results/PCA_covariates.txt",
        provided_covs = config["covariates"] if config["covariates"] else [],
        peer_checkpoint = "results/checkpoints/peer_factors_generated.checkpoint" if config["peer"] else [],
        pca_checkpoint = "results/checkpoints/pcs_generated.checkpoint" if config["pca"] else []
    output:
        final_covs = "results/covariates.txt",
        checkpoint = touch("results/checkpoints/covariates_combined.checkpoint")
    log:
        "logs/combine_covariates.log"
    benchmark:
        "benchmarks/combine_covariates.txt"
    resources:
        mem_mb = lambda wildcards, attempt: get_memory_mb(max_memory=2000) * attempt,
        threads = 1
    threads: 1
    script:
        "scripts/combine_covariates.py"

# Model training per chromosome - this is the most resource-intensive step
rule create_gene_partitions:
    input:
        gene_annot = "results/gene_annot.parsed.txt",
        snp_annot = expand("results/snp_annot.chr{chrom}.txt", chrom=CHROMOSOMES),
        gene_annot_checkpoint = "results/checkpoints/gene_annot_parsed.checkpoint",
        snp_checkpoint = "results/checkpoints/snp_annot_split.checkpoint"
    output:
        partitions_dir = directory("results/gene_partitions"),
        checkpoint = touch("results/checkpoints/gene_partitions_created.checkpoint")
    log:
        "logs/create_gene_partitions.log"
    benchmark:
        "benchmarks/create_gene_partitions.txt"
    params:
        n_partitions = config.get("partitions", 10),
        cis_window = config.get("cis_window", 1000000)
    resources:
        mem_mb = lambda wildcards, attempt: get_memory_mb(max_memory=resources["parse_gtf"]["memory_mb"]) * attempt,
        threads = lambda wildcards, attempt: min(get_threads(), resources["parse_gtf"]["threads"])
    threads: lambda wildcards, attempt: min(get_threads(), resources["parse_gtf"]["threads"])
    script:
        "scripts/create_balanced_partitions.py"

# Create a checkpoint to generate partition list 
checkpoint get_gene_partitions:
    input:
        partitions_dir = "results/gene_partitions",
        checkpoint = "results/checkpoints/gene_partitions_created.checkpoint"
    output:
        partition_list = "results/gene_partitions/partition_list.txt"
    run:
        import glob
        import os
        
        # Find all partition files in the directory
        partition_files = glob.glob(os.path.join(input.partitions_dir, "partition_*.json"))
        partition_ids = [os.path.basename(f).replace("partition_", "").replace(".json", "") for f in partition_files]
        
        # Write partition IDs to output file
        with open(output.partition_list, "w") as f:
            for partition_id in sorted(partition_ids):
                f.write(f"{partition_id}\n")

# Train models for each partition
rule train_model_partitioned:
    input:
        gene_annot = "results/gene_annot.parsed.txt",
        snp_files = expand("results/snp_annot.chr{chrom}.txt", chrom=CHROMOSOMES),
        genotype_files = expand("results/genotype.chr{chrom}.txt", chrom=CHROMOSOMES),
        gene_expr = "results/gene_expression.txt",
        covariates = "results/covariates.txt" if config["peer"] or config["pca"] or config["covariates"] else [],
        partition_file = "results/gene_partitions/partition_{partition_id}.json",
        gene_annot_checkpoint = "results/checkpoints/gene_annot_parsed.checkpoint",
        partitions_checkpoint = "results/checkpoints/gene_partitions_created.checkpoint",
        expr_checkpoint = "results/checkpoints/gene_expr_transposed.checkpoint",
        cov_checkpoint = "results/checkpoints/covariates_combined.checkpoint" if config["peer"] or config["pca"] or config["covariates"] else []
    output:
        model_summary = "results/partition_{partition_id}_model_summaries.txt",
        weight_summary = "results/partition_{partition_id}_weight_summaries.txt",
        covariance = "results/partition_{partition_id}_covariance.txt",
        checkpoint = touch("results/checkpoints/partition_{partition_id}_complete.checkpoint")
    log:
        "logs/train_model_partition_{partition_id}.log"
    benchmark:
        "benchmarks/train_model_partition_{partition_id}.txt"
    params:
        partition_id = "{partition_id}",
        nested_cv = config["nested_cv"],
        n_folds = config["nfolds"],
        r2_threshold = config.get("model_r2_threshold", 0.01),
        cis_window = config.get("cis_window", 1000000)
    resources:
        mem_mb = lambda wildcards, attempt: get_memory_mb(max_memory=resources["train_model"]["memory_mb"]) * attempt,
        threads = lambda wildcards, attempt: min(get_threads(), resources["train_model"]["threads"])
    threads: lambda wildcards, attempt: min(get_threads(), resources["train_model"]["threads"])
    script:
        "scripts/train_model_partitioned.py"

# Replace original train_model rule with a rule to collect partition results
rule collect_partition_results:
    input:
        partition_files = lambda wildcards: expand("results/partition_{partition_id}_model_summaries.txt", 
                                                partition_id=get_partition_ids()),
        checkpoints = lambda wildcards: expand("results/checkpoints/partition_{partition_id}_complete.checkpoint", 
                                            partition_id=get_partition_ids())
    output:
        model_summary = "results/chr{chrom}_model_summaries.txt",
        weight_summary = "results/chr{chrom}_weight_summaries.txt",
        covariance = "results/chr{chrom}_covariance.txt",
        checkpoint = touch("results/checkpoints/chr{chrom}_complete.checkpoint")
    log:
        "logs/collect_partition_results_chr{chrom}.log"
    params:
        chrom = "{chrom}"
    resources:
        mem_mb = 2000,
        threads = 1
    run:
        import pandas as pd
        import os

        # Set up logging
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log[0]), logging.StreamHandler()]
        )
        logger = logging.getLogger("collect_partitions")

        # Collect model summaries
        model_dfs = []
        weight_dfs = []
        cov_dfs = []
        
        for partition_file in input.partition_files:
            partition_id = os.path.basename(partition_file).split("_")[1]
            
            # Read files for this partition
            model_file = f"results/partition_{partition_id}_model_summaries.txt"
            weight_file = f"results/partition_{partition_id}_weight_summaries.txt"
            cov_file = f"results/partition_{partition_id}_covariance.txt"
            
            if os.path.exists(model_file) and os.path.getsize(model_file) > 0:
                try:
                    model_df = pd.read_csv(model_file, sep='\t')
                    # Filter for models on this chromosome
                    chrom_models = model_df[model_df['chromosome'] == params.chrom]
                    if not chrom_models.empty:
                        model_dfs.append(chrom_models)
                except Exception as e:
                    logger.warning(f"Error reading {model_file}: {str(e)}")
            
            if os.path.exists(weight_file) and os.path.getsize(weight_file) > 0:
                try:
                    weight_df = pd.read_csv(weight_file, sep='\t')
                    # Filter for weights from genes on this chromosome
                    if 'chromosome' in weight_df.columns:
                        chrom_weights = weight_df[weight_df['chromosome'] == params.chrom]
                        if not chrom_weights.empty:
                            weight_dfs.append(chrom_weights)
                except Exception as e:
                    logger.warning(f"Error reading {weight_file}: {str(e)}")
            
            if os.path.exists(cov_file) and os.path.getsize(cov_file) > 0:
                try:
                    cov_df = pd.read_csv(cov_file, sep='\t')
                    cov_dfs.append(cov_df)
                except Exception as e:
                    logger.warning(f"Error reading {cov_file}: {str(e)}")
        
        # Combine results
        if model_dfs:
            combined_models = pd.concat(model_dfs, ignore_index=True)
            combined_models.to_csv(output.model_summary, sep='\t', index=False)
            logger.info(f"Wrote {len(combined_models)} model summaries to {output.model_summary}")
        else:
            # Create empty file
            pd.DataFrame().to_csv(output.model_summary, sep='\t', index=False)
            logger.warning(f"No model summaries found for chromosome {params.chrom}")
        
        if weight_dfs:
            combined_weights = pd.concat(weight_dfs, ignore_index=True)
            combined_weights.to_csv(output.weight_summary, sep='\t', index=False)
            logger.info(f"Wrote {len(combined_weights)} weight entries to {output.weight_summary}")
        else:
            # Create empty file
            pd.DataFrame().to_csv(output.weight_summary, sep='\t', index=False)
            logger.warning(f"No weight entries found for chromosome {params.chrom}")
        
        if cov_dfs:
            combined_covs = pd.concat(cov_dfs, ignore_index=True)
            combined_covs.to_csv(output.covariance, sep='\t', index=False)
            logger.info(f"Wrote {len(combined_covs)} covariance entries to {output.covariance}")
        else:
            # Create empty file
            pd.DataFrame().to_csv(output.covariance, sep='\t', index=False)
            logger.warning(f"No covariance entries found for chromosome {params.chrom}")

# Function to get partition IDs from checkpoint
def get_partition_ids():
    checkpoint_output = checkpoints.get_gene_partitions.get()
    
    with open(checkpoint_output.partition_list) as f:
        partition_ids = [line.strip() for line in f]
    
    return partition_ids

# Results aggregation - these are lightweight operations
rule collect_model_summaries:
    input:
        model_summaries = expand("results/chr{chrom}_model_summaries.txt", chrom=CHROMOSOMES),
        checkpoints = expand("results/checkpoints/chr{chrom}_complete.checkpoint", chrom=CHROMOSOMES)
    output:
        all_model_sum = "results/Model_summaries.txt",
        checkpoint = touch("results/checkpoints/model_summaries_collected.checkpoint")
    log:
        "logs/collect_model_summaries.log"
    benchmark:
        "benchmarks/collect_model_summaries.txt"
    resources:
        mem_mb = lambda wildcards, attempt: get_memory_mb(max_memory=2000) * attempt,
        threads = 1
    threads: 1
    script:
        "scripts/collect_model_summaries.py"

rule collect_weight_summaries:
    input:
        weight_summaries = expand("results/chr{chrom}_weight_summaries.txt", chrom=CHROMOSOMES),
        checkpoints = expand("results/checkpoints/chr{chrom}_complete.checkpoint", chrom=CHROMOSOMES)
    output:
        all_weight_sum = "results/Weight_summaries.txt",
        checkpoint = touch("results/checkpoints/weight_summaries_collected.checkpoint")
    log:
        "logs/collect_weight_summaries.log"
    benchmark:
        "benchmarks/collect_weight_summaries.txt"
    resources:
        mem_mb = lambda wildcards, attempt: get_memory_mb(max_memory=2000) * attempt,
        threads = 1
    threads: 1
    script:
        "scripts/collect_weight_summaries.py"

rule collect_covariances:
    input:
        covariances = expand("results/chr{chrom}_covariance.txt", chrom=CHROMOSOMES),
        checkpoints = expand("results/checkpoints/chr{chrom}_complete.checkpoint", chrom=CHROMOSOMES)
    output:
        all_covs = "results/Covariances.txt",
        checkpoint = touch("results/checkpoints/covariances_collected.checkpoint")
    log:
        "logs/collect_covariances.log"
    benchmark:
        "benchmarks/collect_covariances.txt"
    resources:
        mem_mb = lambda wildcards, attempt: get_memory_mb(max_memory=4000) * attempt,
        threads = 1
    threads: 1
    script:
        "scripts/collect_covariances.py"

rule collect_chrom_summaries:
    input:
        model_summaries = expand("results/chr{chrom}_model_summaries.txt", chrom=CHROMOSOMES),
        checkpoints = expand("results/checkpoints/chr{chrom}_complete.checkpoint", chrom=CHROMOSOMES)
    output:
        all_chrom_sum = "results/Chromosome_summary.txt",
        checkpoint = touch("results/checkpoints/chrom_summaries_collected.checkpoint")
    log:
        "logs/collect_chrom_summaries.log"
    benchmark:
        "benchmarks/collect_chrom_summaries.txt"
    resources:
        mem_mb = lambda wildcards, attempt: get_memory_mb(max_memory=2000) * attempt,
        threads = 1
    threads: 1
    script:
        "scripts/collect_chrom_summaries.py"

# Database creation
rule make_database:
    input:
        model_sum = "results/Model_summaries.txt",
        weight_sum = "results/Weight_summaries.txt",
        chrom_sum = "results/Chromosome_summary.txt",
        model_checkpoint = "results/checkpoints/model_summaries_collected.checkpoint",
        weight_checkpoint = "results/checkpoints/weight_summaries_collected.checkpoint",
        chrom_checkpoint = "results/checkpoints/chrom_summaries_collected.checkpoint"
    output:
        whole_db = "results/predict_db_{config[prefix]}.db",
        checkpoint = touch("results/checkpoints/database_created.checkpoint")
    log:
        "logs/make_database.log"
    benchmark:
        "benchmarks/make_database.txt"
    params:
        population = config["prefix"]
    resources:
        mem_mb = lambda wildcards, attempt: get_memory_mb(max_memory=resources["make_database"]["memory_mb"]) * attempt,
        threads = lambda wildcards, attempt: min(get_threads(), resources["make_database"]["threads"])
    threads: lambda wildcards, attempt: min(get_threads(), resources["make_database"]["threads"])
    script:
        "scripts/make_db.py"

rule filter_database:
    input:
        whole_db = "results/predict_db_{config[prefix]}.db",
        checkpoint = "results/checkpoints/database_created.checkpoint"
    output:
        filtered_db = "results/predict_db_{config[prefix]}.filtered.db",
        checkpoint = touch("results/checkpoints/database_filtered.checkpoint")
    log:
        "logs/filter_database.log"
    benchmark:
        "benchmarks/filter_database.txt"
    params:
        r2_threshold = 0.01,
        pval_threshold = 0.05
    resources:
        mem_mb = lambda wildcards, attempt: get_memory_mb(max_memory=2000) * attempt,
        threads = 1
    threads: 1
    script:
        "scripts/filter_db.py"

rule filter_covariance:
    input:
        filtered_db = "results/predict_db_{config[prefix]}.filtered.db",
        all_covs = "results/Covariances.txt",
        db_checkpoint = "results/checkpoints/database_filtered.checkpoint",
        cov_checkpoint = "results/checkpoints/covariances_collected.checkpoint"
    output:
        filtered_covs = "results/Covariances.filtered.txt",
        checkpoint = touch("results/checkpoints/covariances_filtered.checkpoint")
    log:
        "logs/filter_covariance.log"
    benchmark:
        "benchmarks/filter_covariance.txt"
    resources:
        mem_mb = lambda wildcards, attempt: get_memory_mb(max_memory=2000) * attempt,
        threads = 1
    threads: 1
    script:
        "scripts/filter_cov.py"

# Generate HTML report
rule generate_report:
    input:
        db = "results/predict_db_{config[prefix]}.db",
        covs = "results/Covariances.txt",
        db_checkpoint = "results/checkpoints/database_created.checkpoint",
        cov_checkpoint = "results/checkpoints/covariances_collected.checkpoint"
    output:
        report = "results/report.html"
    params:
        title = config["report_title"] if hasattr(config, "report_title") else "PredictDb Pipeline Report",
        description = config["report_description"] if hasattr(config, "report_description") else "Summary of execution"
    log:
        "logs/generate_report.log"
    benchmark:
        "benchmarks/generate_report.txt"
    resources:
        mem_mb = 2000,
        threads = 1
    threads: 1
    run:
        shell("snakemake --report {output.report} --report-stylesheet default --configfile config/report.yaml")

# Cleanup rule for temporary files if keepIntermediate is False
if not config["keepIntermediate"]:
    onsuccess:
        shell("rm -f results/temp_*")

# Define onerror and onsuccess handlers for email notifications
onerror:
    if config["email"]["enabled"] and config["email"]["on_failure"]:
        try:
            msg = MIMEText(f"Pipeline failed. Check logs in {os.getcwd()}")
            msg['Subject'] = 'PredictDb Pipeline Failure'
            msg['From'] = config["email"]["smtp_username"]
            msg['To'] = config["email"]["to"]
            
            s = smtplib.SMTP(config["email"]["smtp_server"], config["email"]["smtp_port"])
            s.login(config["email"]["smtp_username"], config["email"]["smtp_password"])
            s.send_message(msg)
            s.quit()
            print("Failure notification email sent.")
        except Exception as e:
            print(f"Failed to send email notification: {str(e)}")

onsuccess:
    # Generate benchmark plots if benchmarks exist
    if os.path.exists("benchmarks") and len(os.listdir("benchmarks")) > 0:
        shell("python scripts/utils/plot_benchmarks.py")
        
    # Send success email if enabled
    if config["email"]["enabled"] and config["email"]["on_success"]:
        try:
            msg = MIMEText("PredictDb pipeline completed successfully.")
            msg['Subject'] = 'PredictDb Pipeline Success'
            msg['From'] = config["email"]["smtp_username"]
            msg['To'] = config["email"]["to"]
            
            s = smtplib.SMTP(config["email"]["smtp_server"], config["email"]["smtp_port"])
            s.login(config["email"]["smtp_username"], config["email"]["smtp_password"])
            s.send_message(msg)
            s.quit()
            print("Success notification email sent.")
        except Exception as e:
            print(f"Failed to send email notification: {str(e)}")