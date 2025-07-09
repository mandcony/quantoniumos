#!/bin/bash
# Trivy Security Scanner for Quantonium OS
# This script runs Trivy to scan the quantonium container for vulnerabilities
# Fails on HIGH and CRITICAL severity issues

set -e

CONTAINER_NAME="quantonium:latest"

echo "🔍 Running Trivy vulnerability scan on $CONTAINER_NAME"
echo "======================================================"

# Ensure container is built
if [[ "$(docker images -q $CONTAINER_NAME 2> /dev/null)" == "" ]]; then
  echo "Error: Container $CONTAINER_NAME not found"
  echo "Run 'docker-compose build' first"
  exit 1
fi

# Create the report directory if it doesn't exist
mkdir -p ./security-reports

# Run Trivy with JSON output and fail on HIGH and CRITICAL severity issues
docker run --rm \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$(pwd)/security-reports:/tmp/security-reports" \
  aquasec/trivy:latest \
  image \
  --exit-code 1 \
  --severity HIGH,CRITICAL \
  --format json \
  --output /tmp/security-reports/trivy-report.json \
  $CONTAINER_NAME

RESULT=$?

# Check the result
if [ $RESULT -eq 0 ]; then
  echo "✅ Trivy scan passed successfully!"
else
  echo "❌ Trivy scan failed with exit code $RESULT"
  echo "Please check the report at ./security-reports/trivy-report.json"
  # Extract the issues to present them in the terminal
  echo "Top issues found:"
  grep -o '"VulnerabilityID":"[^"]*' ./security-reports/trivy-report.json | cut -d'"' -f4 | sort | uniq | head -10
fi

exit $RESULT