#!/bin/bash
set -e

# Container validation script for Quantonium OS Cloud Runtime
# This script verifies that the container functions correctly in read-only mode
# and performs security checks to validate Phase 4 implementation.

echo "===== Quantonium OS Container Validation ====="
echo "Running validation for security phase 4..."

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not available. Please install Docker before running this validation."
    exit 1
fi

# Build the container
echo "Building container from Dockerfile..."
docker build -t quantonium:validation .

# Run with --read-only flag
echo "Testing container with read-only filesystem..."
CONTAINER_ID=$(docker run -d --read-only \
    -p 5001:5000 \
    -v /tmp:/tmp/logs:rw \
    -e QUANTONIUM_API_KEY=test_key \
    -e SESSION_SECRET=test_secret \
    --name quantonium-validation \
    quantonium:validation)

# Wait for container to start
echo "Waiting for container to initialize..."
sleep 5

# Test /api/health endpoint
echo "Testing API health endpoint..."
HEALTH_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5001/api/health)

if [ "$HEALTH_STATUS" = "200" ]; then
    echo "✅ SUCCESS: Health endpoint returned 200 OK"
else
    echo "❌ FAILURE: Health endpoint returned $HEALTH_STATUS (expected 200)"
    docker logs quantonium-validation
    docker rm -f quantonium-validation
    exit 1
fi

# Test API status endpoint
echo "Testing API status endpoint..."
STATUS_RESPONSE=$(curl -s http://localhost:5001/status)
if [[ "$STATUS_RESPONSE" == *"operational"* ]]; then
    echo "✅ SUCCESS: Status endpoint working properly"
else
    echo "❌ FAILURE: Status endpoint not working properly"
    echo "Response: $STATUS_RESPONSE"
    docker logs quantonium-validation
    docker rm -f quantonium-validation
    exit 1
fi

# Check if we're running as non-root
echo "Checking container runs as non-root user..."
USER_ID=$(docker exec quantonium-validation id -u)
if [ "$USER_ID" = "1001" ]; then
    echo "✅ SUCCESS: Container running as non-root user (UID 1001)"
else
    echo "❌ FAILURE: Container running as UID $USER_ID, expected 1001"
    docker rm -f quantonium-validation
    exit 1
fi

# Test log writing to mounted volume
echo "Testing log writing to mounted volume..."
docker exec quantonium-validation curl -s http://localhost:5000/api/health > /dev/null
if docker exec quantonium-validation ls -la /tmp/logs/quantonium_api.log > /dev/null 2>&1; then
    echo "✅ SUCCESS: Logs written successfully to mounted volume"
else
    echo "❌ FAILURE: Could not write logs to mounted volume"
    docker rm -f quantonium-validation
    exit 1
fi

# Clean up
echo "Cleaning up..."
docker rm -f quantonium-validation

echo "===== Validation Complete ====="
echo "All tests passed! The container works correctly with read-only filesystem and security measures."