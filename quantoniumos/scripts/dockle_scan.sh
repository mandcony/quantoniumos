#!/bin/bash
# Dockle Security Scanner for Quantonium OS
# This script runs Dockle to scan the quantonium container for security issues
# Fails on FATAL and WARN level issues

set -e

CONTAINER_NAME="quantonium:latest"

echo "🔒 Running Dockle security scan on $CONTAINER_NAME"
echo "======================================================"

# Ensure container is built
if [[ "$(docker images -q $CONTAINER_NAME 2> /dev/null)" == "" ]]; then
  echo "Error: Container $CONTAINER_NAME not found"
  echo "Run 'docker-compose build' first"
  exit 1
fi

# Create the report directory if it doesn't exist
mkdir -p ./security-reports

# Run Dockle with JSON output and fail on FATAL and WARN level issues
docker run --rm \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$(pwd)/security-reports:/tmp/security-reports" \
  goodwithtech/dockle:latest \
  --exit-code 1 \
  --exit-level WARN \
  --format json \
  --output /tmp/security-reports/dockle-report.json \
  $CONTAINER_NAME

RESULT=$?

# Check the result
if [ $RESULT -eq 0 ]; then
  echo "✅ Dockle scan passed successfully!"
else
  echo "❌ Dockle scan failed with exit code $RESULT"
  echo "Please check the report at ./security-reports/dockle-report.json"
  # Extract the issues to present them in the terminal
  echo "Issues found:"
  grep -o '"description":"[^"]*' ./security-reports/dockle-report.json | cut -d'"' -f4 | sort | uniq
fi

exit $RESULT