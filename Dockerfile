FROM continuumio/miniconda3:latest

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy environment file
COPY environment.yaml /tmp/environment.yaml

# Create conda environment
RUN conda env create -f /tmp/environment.yaml && \
    conda clean -afy

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "predictdb-snakemake", "/bin/bash", "-c"]

# Set working directory
WORKDIR /predictdb

# Copy pipeline files
COPY . /predictdb/

# Set default command
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "predictdb-snakemake", "snakemake"]
CMD ["--help"] 