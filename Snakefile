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
rule train_model:
    input:
        gene_annot = "results/gene_annot.parsed.txt",
        snp_file = "results/snp_annot.chr{chrom}.txt",
        genotype_file = "results/genotype.chr{chrom}.txt",
        gene_expr = "results/gene_expression.txt",
        covariates = "results/covariates.txt" if config["peer"] or config["pca"] or config["covariates"] else [],
        gene_annot_checkpoint = "results/checkpoints/gene_annot_parsed.checkpoint",
        snp_checkpoint = "results/checkpoints/snp_annot_split.checkpoint",
        genotype_checkpoint = "results/checkpoints/genotype_split.checkpoint",
        expr_checkpoint = "results/checkpoints/gene_expr_transposed.checkpoint",
        cov_checkpoint = "results/checkpoints/covariates_combined.checkpoint" if config["peer"] or config["pca"] or config["covariates"] else []
    output:
        model_summary = "results/chr{chrom}_model_summaries.txt",
        weight_summary = "results/chr{chrom}_weight_summaries.txt",
        covariance = "results/chr{chrom}_covariance.txt",
        checkpoint = touch("results/checkpoints/chr{chrom}_complete.checkpoint")
    log:
        "logs/train_model_chr{chrom}.log"
    benchmark:
        "benchmarks/train_model_chr{chrom}.txt"
    params:
        chrom = "{chrom}",
        nested_cv = config["nested_cv"],
        n_folds = config["nfolds"]
    resources:
        mem_mb = lambda wildcards, attempt: get_memory_mb(max_memory=resources["train_model"]["memory_mb"]) * attempt,
        threads = lambda wildcards, attempt: min(get_threads(), resources["train_model"]["threads"])
    threads: lambda wildcards, attempt: min(get_threads(), resources["train_model"]["threads"])
    report: "Model training for chromosome {wildcards.chrom} completed, with genome-wide expression prediction models."
    script:
        "scripts/train_elastic_net_model.py"

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