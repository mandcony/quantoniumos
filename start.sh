#!/bin/bash

echo "Starting Quantonium OS Cloud Runtime..."
echo "Initializing symbolic modules..."

# Check if Python is installed
if command -v python3 &>/dev/null; then
    echo "Using Python3 interpreter..."
    python3 main.py
elif command -v python &>/dev/null; then
    echo "Using Python interpreter..."
    python main.py
else
    echo "Error: Python interpreter not found!"
    exit 1
fi
