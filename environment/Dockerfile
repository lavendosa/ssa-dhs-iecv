# Dockerfile for 100% Containerized Reproducibility
# Sub-Saharan African DHS Pediatric Healthcare Seeking Study

FROM python:3.11.8-slim-bookworm

LABEL maintainer="Research Engineering Team"
LABEL description="Reproducible container for 38-country DHS pediatric machine learning study"
LABEL zenodo_doi="10.5281/zenodo.10892345"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 \
    OPENBLAS_NUM_THREADS=8 \
    DHS_RAW_DATA_DIR="/workspace/data/raw_dhs"

# Install system dependencies (build-essential, git, libgomp1 for LightGBM/XGBoost OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy dependency requirements
COPY environment/requirements.txt /workspace/environment/requirements.txt

# Install python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /workspace/environment/requirements.txt

# Copy repository code, documentation, and artifacts
COPY . /workspace/

# Expose Jupyter port
EXPOSE 8888

# Default entrypoint
CMD ["python", "run_pipeline.py", "--help"]
