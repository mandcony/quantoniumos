#!/bin/bash
set -e

# Cosign Verification Script for Quantonium OS Cloud Runtime
# This script verifies the container signature using cosign

echo "===== Quantonium OS Signature Verification ====="

# Check if cosign is available
if ! command -v cosign &> /dev/null; then
    echo "ERROR: Cosign is not available."
    echo "Please install cosign: https://docs.sigstore.dev/cosign/installation/"
    exit 1
fi

# Check if image name is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <image-name>"
    echo "Example: $0 ghcr.io/quantonium/quantonium:latest"
    exit 1
fi

IMAGE_NAME="$1"

# Verify image signature
echo "Verifying signature for $IMAGE_NAME..."

# For GitHub Actions keyless signing
cosign verify \
    --certificate-identity-regexp=https://github.com/quantonium \
    --certificate-oidc-issuer=https://token.actions.githubusercontent.com \
    "$IMAGE_NAME"

VERIFY_EXIT_CODE=$?

# Check exit code
if [ $VERIFY_EXIT_CODE -eq 0 ]; then
    echo "✅ SUCCESS: Signature verified for $IMAGE_NAME!"
else
    echo "❌ FAILURE: Signature verification failed for $IMAGE_NAME."
    exit 1
fi

echo "===== Verification Complete ====="