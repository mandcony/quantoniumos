# Reproducible Build Guide

## Environment
- OS: Ubuntu 22.04 (Jammy)
- Tooling: Python 3.10+, nasm, build-essential

## One-command container build
```bash
docker build -t mandcony/quantoniumos:quick .
docker run --rm -e QOS_TEST_SEED=deadbeef mandcony/quantoniumos:quick
```

## Native host build
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e .[dev]
make -C algorithms/rft/kernels all || true
export RFT_KERNEL_LIB=$(find algorithms/rft/kernels -name 'libquantum_symbolic.so' | head -n1)
pytest -m "not slow" --cov=algorithms --cov=core --cov=quantonium_os_src --cov-report=xml
```

## Dependencies
- Locked base (example): see `requirements-lock.txt` (regenerate with pip-compile)

## Data
- List corpora and SHA-256 in `data/README.md`. Provide a `tools/data/fetch.py` script to download/regenerate deterministically.

## Tags and artifacts
- Tag the paper-aligned commit: `v0.4.0-48rft`.
- Attach `wheel`, `libquantum_symbolic.so`, and `coverage.xml` with SHA256.
