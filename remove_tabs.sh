#!/bin/bash

# Array of files to modify
files=(
    "static/container-operations.html"
    "static/quantum-entropy.html"
    "static/quantum-grid.html"
    "static/quantum-benchmark.html"
)

# Remove the tab divs
for file in "${files[@]}"; do
    sed -i '/<div class="tabs">/,/<\/div>/ c\    <!-- Tabs removed as requested -->' "$file"
    
    # Remove the tab CSS classes
    sed -i '/.tabs {/,/.tab:hover:not(.active) {/c\        /* Tab styling removed as requested */' "$file"
done

echo "Tabs removed from all files"