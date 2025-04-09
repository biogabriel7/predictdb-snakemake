# PredictDb Snakemake Workflow
# Converted from Nextflow pipeline

import os

# Configuration
configfile: "config/config.yaml"

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
    script:
        "scripts/parse_gtf.py"

rule split_snp_annot:
    input:
        snp_annot = config["snp_annotation"]
    output:
        snp_files = expand("results/snp_annot.chr{chrom}.txt", chrom=CHROMOSOMES)
    script:
        "scripts/split_snp_annot_by_chr.py"

rule split_genotype:
    input:
        genotype = config["genotype"]
    output:
        genotype_files = expand("results/genotype.chr{chrom}.txt", chrom=CHROMOSOMES)
    script:
        "scripts/split_genotype_by_chr.py"

rule transpose_gene_expr:
    input:
        gene_expr = config["gene_exp"]
    output:
        tr_expr = "results/gene_expression.txt",
        tr_expr_peer = "results/gene_expression_for_peer.csv"
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
    script:
        "scripts/generate_peer_factors.py"

rule generate_pcs:
    input:
        gene_expr = "results/gene_expression_for_peer.csv"
    output:
        pcs = "results/PCA_covariates.txt"
    params:
        n_components = 10  # Default to 10 PCs
    script:
        "scripts/generate_pcs.py"

rule combine_covariates:
    input:
        computed_covs = lambda wildcards: "results/PEER_covariates.txt" if config["peer"] else "results/PCA_covariates.txt",
        provided_covs = config["covariates"] if config["covariates"] else []
    output:
        final_covs = "results/covariates.txt"
    script:
        "scripts/combine_covariates.py"

# Model training per chromosome
rule train_model_with_covs:
    input:
        gene_annot = "results/gene_annot.parsed.txt",
        snp_file = "results/snp_annot.chr{chrom}.txt",
        genotype_file = "results/genotype.chr{chrom}.txt",
        gene_expr = "results/gene_expression.txt",
        covariates = "results/covariates.txt"
    output:
        model_summary = "results/chr{chrom}_model_summaries.txt",
        weight_summary = "results/chr{chrom}_weight_summaries.txt",
        covariance = "results/chr{chrom}_covariance.txt"
    params:
        chrom = "{chrom}",
        nested_cv = config["nested_cv"],
        n_folds = config["nfolds"]
    script:
        "scripts/train_elastic_net_model.py"

rule train_model_without_covs:
    input:
        gene_annot = "results/gene_annot.parsed.txt",
        snp_file = "results/snp_annot.chr{chrom}.txt",
        genotype_file = "results/genotype.chr{chrom}.txt",
        gene_expr = "results/gene_expression.txt"
    output:
        model_summary = "results/chr{chrom}_model_summaries.txt",
        weight_summary = "results/chr{chrom}_weight_summaries.txt",
        covariance = "results/chr{chrom}_covariance.txt"
    params:
        chrom = "{chrom}",
        nested_cv = config["nested_cv"],
        n_folds = config["nfolds"]
    script:
        "scripts/train_elastic_net_model.py"

# Results aggregation
rule collect_model_summaries:
    input:
        model_summaries = expand("results/chr{chrom}_model_summaries.txt", chrom=CHROMOSOMES)
    output:
        all_model_sum = "results/Model_summaries.txt"
    script:
        "scripts/collect_model_summaries.py"

rule collect_weight_summaries:
    input:
        weight_summaries = expand("results/chr{chrom}_weight_summaries.txt", chrom=CHROMOSOMES)
    output:
        all_weight_sum = "results/Weight_summaries.txt"
    script:
        "scripts/collect_weight_summaries.py"

rule collect_covariances:
    input:
        covariances = expand("results/chr{chrom}_covariance.txt", chrom=CHROMOSOMES)
    output:
        all_covs = "results/Covariances.txt"
    script:
        "scripts/collect_covariances.py"

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
    script:
        "scripts/filter_db.py"

rule filter_covariance:
    input:
        filtered_db = "results/predict_db_{config[prefix]}.filtered.db",
        all_covs = "results/Covariances.txt"
    output:
        filtered_covs = "results/Covariances.filtered.txt"
    script:
        "scripts/filter_cov.py"