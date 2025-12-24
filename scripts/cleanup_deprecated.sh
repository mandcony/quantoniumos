#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
set -e

echo "Starting cleanup of deprecated files..."

# List of deprecated files to remove
FILES_TO_REMOVE=(
    "algorithms/rft/core/closed_form_rft.py"
    "algorithms/rft/core/rft_optimized.py"
    "algorithms/rft/transform_core/phi_phase_fft.py"
    "algorithms/rft/core/phi_phase_fft.py"
)

for file in "${FILES_TO_REMOVE[@]}"; do
    if [ -f "$file" ]; then
        echo "Removing deprecated file: $file"
        rm "$file"
    else
        echo "File not found (already removed?): $file"
    fi
done

echo "Cleanup complete."
