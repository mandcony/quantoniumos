#!/bin/bash

# Array of files to modify
files=(
    "static/resonance-transform.html"
    "static/container-operations.html"
    "static/quantum-entropy.html"
    "static/quantum-grid.html"
    "static/quantum-benchmark.html"
)

# Add CSS for hiding header when embedded
for file in "${files[@]}"; do
    # Add the CSS rule for body.embedded header
    sed -i '/header {/a \
        \/* Hide header when embedded in the OS interface *\/\
        body.embedded header {\
            display: none;\
        }' "$file"
    
    # Add the JavaScript to check for embedded parameter
    sed -i '/<script>/a \
        \/\/ Check if page is embedded in the OS interface\
        function checkEmbedded() {\
            const urlParams = new URLSearchParams(window.location.search);\
            if (urlParams.get('\''embedded'\'') === '\''true'\'') {\
                document.body.classList.add('\''embedded'\'');\
            }\
        }\
        \
        \/\/ Run on page load\
        window.addEventListener('\''DOMContentLoaded'\'', checkEmbedded);' "$file"
done

echo "Embedded header fixes applied to all files"