# CPU-only image for the heterofl experiments on NRP. No data, secrets or kubeconfig inside.
# Build context = repository root.  NOT yet built or tested: see docs/nrp_deployment.md.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    FLWR_TELEMETRY_ENABLED=0

WORKDIR /app

# libgomp1: OpenMP runtime needed by xgboost / scikit-learn wheels; git: environment.json records the commit
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 git \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch wheels (the default PyPI wheel pulls ~2 GB of CUDA libraries we never use)
COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY deploy ./deploy

CMD ["python", "deploy/run_unit.py", "--help"]
