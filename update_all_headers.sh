#!/bin/bash

# Get list of all app HTML files
app_files=(
  "static/container-operations.html"
  "static/quantum-benchmark.html"
  "static/quantum-browser.html"
  "static/quantum-container.html"
  "static/quantum-encryption.html"
  "static/quantum-entropy.html" 
  "static/quantum-grid.html"
  "static/quantum-mail.html"
  "static/quantum-notes.html"
  "static/quantum-rft.html"
  "static/quantum_benchmark.html"
  "static/quantum_container.html"
  "static/quantum_encryption.html"
  "static/quantum_entropy.html"
  "static/quantum_rft.html"
  "static/64-benchmark.html"
)

# Process each file
for file in "${app_files[@]}"; do
  if [ -f "$file" ]; then
    echo "Processing $file..."
    
    # Check if the file has a header tag
    if grep -q "<header" "$file"; then
      # Check if embedded CSS already exists
      if ! grep -q "body.embedded header" "$file"; then
        # Add the CSS rule for body.embedded header
        sed -i '/header {/a \
            \/* Hide header when embedded in the OS interface *\/\
            body.embedded header {\
                display: none;\
            }' "$file"
      fi
      
      # Check if the embedded detection JavaScript already exists
      if ! grep -q "checkEmbedded" "$file"; then
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
      fi
    else
      echo "No header tag found in $file, skipping..."
    fi
  else
    echo "File $file does not exist, skipping..."
  fi
done

echo "Embedded header fixes applied to all app files"