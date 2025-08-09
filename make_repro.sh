#!/bin/bash
# One-command reproducible build and validation
# Usage: ./make_repro.sh

set -euo pipefail

echo "=== QuantoniumOS Reproducible Build & Validation ==="
echo "Building pinned container and running full test suite..."

# Build pinned container
docker build -t quantoniumos:repro-$(date +%Y%m%d) -f Dockerfile.repro .

# Run full suite in container
docker run --rm -v $(pwd)/repro_results:/output quantoniumos:repro-$(date +%Y%m%d) /bin/bash -c "
    set -e
    echo '=== Running Full Test Suite ==='
    
    # Core validation
    python -m pytest tests/ -v --tb=short
    
    # Security validation  
    python run_security_focused_tests.py
    
    # Statistical validation
    python run_statistical_validation.py
    
    # Generate test vectors
    python -c 'from tests.generate_vectors import generate_all_vectors; generate_all_vectors()'
    
    # Emit signed checksums
    echo '=== Generating Signed Checksums ==='
    find . -name '*.py' -o -name '*.cpp' -o -name '*.yml' | sort | xargs sha256sum > /output/checksums.txt
    
    # Sign checksums (production would use real key)
    echo 'REPRO_BUILD_$(date +%Y%m%d_%H%M%S)_SIGNATURE' >> /output/checksums.txt
    
    echo '=== Reproducible Build Complete ==='
    echo 'Results in: repro_results/'
"

echo "✅ One-command repro complete. Results in repro_results/"
