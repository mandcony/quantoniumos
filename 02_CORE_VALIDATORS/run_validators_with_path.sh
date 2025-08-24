#!/bin/bash
# Script to run validators with correct Python path

# Set the Python path to include the root directory
export PYTHONPATH=/workspaces/quantoniumos:$PYTHONPATH

# Run the validators
python run_all_validators.py "$@"

# Return the exit code from the validators
exit $?
