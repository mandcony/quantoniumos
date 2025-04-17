#!/bin/bash
set -e

# Trivy Scanner Script for Quantonium OS Cloud Runtime
# This script runs a security scan on the container image using Trivy

echo "===== Quantonium OS Trivy Security Scan ====="

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not available. Please install Docker before running this scan."
    exit 1
fi

# Build the container if not already built
if ! docker image inspect quantonium:latest >/dev/null 2>&1; then
    echo "Building container from Dockerfile..."
    docker build -t quantonium:latest .
fi

# Create a container to run Trivy
echo "Running Trivy vulnerability scanner..."
docker run --rm \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v $PWD/trivy-reports:/reports \
    aquasec/trivy:latest image \
    --format table \
    --exit-code 1 \
    --severity HIGH,CRITICAL \
    --output /reports/trivy-report.txt \
    quantonium:latest

SCAN_EXIT_CODE=$?

# Check exit code
if [ $SCAN_EXIT_CODE -eq 0 ]; then
    echo "✅ SUCCESS: No HIGH or CRITICAL vulnerabilities found!"
else
    echo "❌ FAILURE: Vulnerabilities found. See trivy-reports/trivy-report.txt for details."
    cat trivy-reports/trivy-report.txt
    exit 1
fi

echo "===== Scan Complete ====="