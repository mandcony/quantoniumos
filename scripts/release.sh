#!/bin/bash
# Quantonium OS - Release Script
# This script builds, signs, and pushes a tagged container image

set -e

# Version is taken from environment variable or from pyproject.toml
if [ -z "$VERSION" ]; then
    VERSION=$(grep -m 1 'version = ' pyproject.toml | cut -d '"' -f 2)
fi

# Registry settings
REGISTRY="ghcr.io"
REPO="quantonium"
IMAGE_NAME="quantonium"
FULL_IMAGE="${REGISTRY}/${REPO}/${IMAGE_NAME}"

# Function to check required tools
check_tools() {
    echo "Checking required tools..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker is not installed"
        exit 1
    fi
    
    # Check cosign
    if ! command -v cosign &> /dev/null; then
        echo "❌ Cosign is not installed"
        echo "Please install from: https://github.com/sigstore/cosign"
        exit 1
    fi
    
    # Check if COSIGN_KEY is set (for GitHub Actions)
    if [ -z "$COSIGN_KEY" ] && [ -z "$COSIGN_PASSWORD" ]; then
        echo "⚠️  Warning: COSIGN_KEY environment variable not set"
        echo "Image will be pushed but not signed"
    fi
    
    echo "✅ All required tools are available"
}

# Function to build the image
build_image() {
    echo "📦 Building ${FULL_IMAGE}:${VERSION}..."
    
    docker build \
        --build-arg VERSION="${VERSION}" \
        --tag "${FULL_IMAGE}:${VERSION}" \
        --tag "${FULL_IMAGE}:latest" \
        .
        
    echo "✅ Build completed"
}

# Function to push the image
push_image() {
    echo "📤 Pushing ${FULL_IMAGE}:${VERSION}..."
    
    # Check if we're logged in to the registry
    if ! docker info | grep -q "${REGISTRY}"; then
        echo "⚠️  Not logged in to ${REGISTRY}"
        echo "Please run: docker login ${REGISTRY}"
        exit 1
    fi
    
    docker push "${FULL_IMAGE}:${VERSION}"
    docker push "${FULL_IMAGE}:latest"
    
    echo "✅ Push completed"
}

# Function to sign the image
sign_image() {
    if [ -z "$COSIGN_KEY" ] && [ -z "$COSIGN_PASSWORD" ]; then
        echo "⚠️  Skipping image signing (no COSIGN_KEY)"
        return
    fi
    
    echo "🔐 Signing ${FULL_IMAGE}:${VERSION}..."
    
    # Sign the image with cosign
    cosign sign --key env://COSIGN_KEY "${FULL_IMAGE}:${VERSION}"
    
    # Also sign the latest tag
    cosign sign --key env://COSIGN_KEY "${FULL_IMAGE}:latest"
    
    echo "✅ Signing completed"
    
    # Verify the signature
    echo "🔍 Verifying signature..."
    cosign verify --key env://COSIGN_KEY "${FULL_IMAGE}:${VERSION}"
}

# Main execution
echo "🚀 Releasing Quantonium OS ${VERSION}"
echo "==================================="

check_tools
build_image
push_image
sign_image

echo "==================================="
echo "✅ Release process completed successfully!"
echo ""
echo "Image: ${FULL_IMAGE}:${VERSION}"
echo "Verify with: cosign verify --key YOUR_PUBLIC_KEY ${FULL_IMAGE}:${VERSION}"
echo ""
echo "Add to docker-compose.yml:"
echo "  image: ${FULL_IMAGE}:${VERSION}"