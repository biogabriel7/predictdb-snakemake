# PredictDb Snakemake Workflow
# Optimized for MacBook M3 Pro

import os

# Configuration
configfile: "config/config.yaml"
# Resource configuration
configfile: "config/resources.yaml"

# Chromosome list
CHROMOSOMES = list(range(1, 23))

# Final output target
rule all:
    input:
        "results/predict_db_{config[prefix]}.db",
        "results/Covariances.txt"

# Preprocessing rules
rule parse_gene_annot:
    input:
        gtf = config["gene_annotation"]
    output:
        parsed_annot = "results/gene_annot.parsed.txt"
    resources:
        mem_mb = resources["parse_gtf"]["memory_mb"],
        threads = resources["parse_gtf"]["threads"]
    threads: resources["parse_gtf"]["threads"]
    script:
        "scripts/parse_gtf.py"

rule split_snp_annot:
    input:
        snp_annot = config["snp_annotation"]
    output:
        snp_files = expand("results/snp_annot.chr{chrom}.txt", chrom=CHROMOSOMES)
    resources:
        mem_mb = 4000,
        threads = 2
    threads: 2
    script:
        "scripts/split_snp_annot_by_chr.py"

rule split_genotype:
    input:
        genotype = config["genotype"]
    output:
        genotype_files = expand("results/genotype.chr{chrom}.txt", chrom=CHROMOSOMES)
    resources:
        mem_mb = resources["split_genotype"]["memory_mb"],
        threads = resources["split_genotype"]["threads"]
    threads: resources["split_genotype"]["threads"]
    script:
        "scripts/split_genotype_by_chr.py"

rule transpose_gene_expr:
    input:
        gene_expr = config["gene_exp"]
    output:
        tr_expr = "results/gene_expression.txt",
        tr_expr_peer = "results/gene_expression_for_peer.csv"
    resources:
        mem_mb = resources["transpose_gene_expr"]["memory_mb"],
        threads = resources["transpose_gene_expr"]["threads"]
    threads: resources["transpose_gene_expr"]["threads"]
    script:
        "scripts/transpose_gene_expr.py"

# Covariate calculations
rule generate_peer_factors:
    input:
        gene_expr = "results/gene_expression_for_peer.csv"
    output:
        peers = "results/PEER_covariates.txt"
    params:
        n_factors = config["n_peer_factors"]
    resources:
        mem_mb = resources["generate_peer_factors"]["memory_mb"],
        threads = resources["generate_peer_factors"]["threads"]
    threads: resources["generate_peer_factors"]["threads"]
    script:
        "scripts/generate_peer_factors.py"

rule generate_pcs:
    input:
        gene_expr = "results/gene_expression_for_peer.csv"
    output:
        pcs = "results/PCA_covariates.txt"
    params:
        n_components = 10  # Default to 10 PCs
    resources:
        mem_mb = resources["generate_pcs"]["memory_mb"],
        threads = resources["generate_pcs"]["threads"]
    threads: resources["generate_pcs"]["threads"]
    script:
        "scripts/generate_pcs.py"

rule combine_covariates:
    input:
        computed_covs = lambda wildcards: "results/PEER_covariates.txt" if config["peer"] else "results/PCA_covariates.txt",
        provided_covs = config["covariates"] if config["covariates"] else []
    output:
        final_covs = "results/covariates.txt"
    resources:
        mem_mb = 2000,
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
        covariates = "results/covariates.txt" if config["peer"] or config["pca"] or config["covariates"] else []
    output:
        model_summary = "results/chr{chrom}_model_summaries.txt",
        weight_summary = "results/chr{chrom}_weight_summaries.txt",
        covariance = "results/chr{chrom}_covariance.txt"
    params:
        chrom = "{chrom}",
        nested_cv = config["nested_cv"],
        n_folds = config["nfolds"]
    resources:
        mem_mb = resources["train_model"]["memory_mb"],
        threads = resources["train_model"]["threads"]
    threads: resources["train_model"]["threads"]
    script:
        "scripts/train_elastic_net_model.py"

# Results aggregation - these are lightweight operations
rule collect_model_summaries:
    input:
        model_summaries = expand("results/chr{chrom}_model_summaries.txt", chrom=CHROMOSOMES)
    output:
        all_model_sum = "results/Model_summaries.txt"
    resources:
        mem_mb = 2000,
        threads = 1
    threads: 1
    script:
        "scripts/collect_model_summaries.py"

rule collect_weight_summaries:
    input:
        weight_summaries = expand("results/chr{chrom}_weight_summaries.txt", chrom=CHROMOSOMES)
    output:
        all_weight_sum = "results/Weight_summaries.txt"
    resources:
        mem_mb = 2000,
        threads = 1
    threads: 1
    script:
        "scripts/collect_weight_summaries.py"

rule collect_covariances:
    input:
        covariances = expand("results/chr{chrom}_covariance.txt", chrom=CHROMOSOMES)
    output:
        all_covs = "results/Covariances.txt"
    resources:
        mem_mb = 4000,
        threads = 1
    threads: 1
    script:
        "scripts/collect_covariances.py"

rule collect_chrom_summaries:
    input:
        model_summaries = expand("results/chr{chrom}_model_summaries.txt", chrom=CHROMOSOMES)
    output:
        all_chrom_sum = "results/Chromosome_summary.txt"
    resources:
        mem_mb = 2000,
        threads = 1
    threads: 1
    script:
        "scripts/collect_chrom_summaries.py"

# Database creation
rule make_database:
    input:
        model_sum = "results/Model_summaries.txt",
        weight_sum = "results/Weight_summaries.txt",
        chrom_sum = "results/Chromosome_summary.txt"
    output:
        whole_db = "results/predict_db_{config[prefix]}.db"
    params:
        population = config["prefix"]
    resources:
        mem_mb = resources["make_database"]["memory_mb"],
        threads = resources["make_database"]["threads"]
    threads: resources["make_database"]["threads"]
    script:
        "scripts/make_db.py"

rule filter_database:
    input:
        whole_db = "results/predict_db_{config[prefix]}.db"
    output:
        filtered_db = "results/predict_db_{config[prefix]}.filtered.db"
    params:
        r2_threshold = 0.01,
        pval_threshold = 0.05
    resources:
        mem_mb = 2000,
        threads = 1
    threads: 1
    script:
        "scripts/filter_db.py"

rule filter_covariance:
    input:
        filtered_db = "results/predict_db_{config[prefix]}.filtered.db",
        all_covs = "results/Covariances.txt"
    output:
        filtered_covs = "results/Covariances.filtered.txt"
    resources:
        mem_mb = 2000,
        threads = 1
    threads: 1
    script:
        "scripts/filter_cov.py"