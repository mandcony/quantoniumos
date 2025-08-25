#!/usr/bin/env python3
"""
QuantoniumOS Import Fix Script

This script updates imports in Python files to ensure they can correctly
find modules that have been moved to new locations.
"""

import os
import re
import sys

# Root directory
ROOT_DIR = "/workspaces/quantoniumos"

# File organization mapping (from original name to new path)
FILE_MAPPING = {
    "quantoniumos.py": "core/quantoniumos.py",
    "quantonium_os_unified.py": "core/quantonium_os_unified.py",
    "launch_quantoniumos.py": "core/launch_quantoniumos.py",
    "main.py": "core/main.py",
    "app.py": "core/app.py",
    "launch_gui.py": "core/launch_gui.py",
    "bulletproof_quantum_kernel.py": "05_QUANTUM_ENGINES/bulletproof_quantum_kernel.py",
    "topological_quantum_kernel.py": "05_QUANTUM_ENGINES/topological_quantum_kernel.py",
    "working_quantum_kernel.py": "05_QUANTUM_ENGINES/working_quantum_kernel.py",
    "paper_compliant_rft_fixed.py": "04_RFT_ALGORITHMS/paper_compliant_rft_fixed.py",
    "canonical_true_rft.py": "04_RFT_ALGORITHMS/canonical_true_rft.py",
    "build_crypto_engine.py": "06_CRYPTOGRAPHY/build_crypto_engine.py",
    "quantonium_design_system.py": "11_QUANTONIUMOS/quantonium_design_system.py",
    "quantonium_hpc_pipeline.py": "quantonium_hpc_pipeline/quantonium_hpc_pipeline.py",
}

# Module name mapping (from original module name to new import path)
MODULE_MAPPING = {
    "quantoniumos": "11_QUANTONIUMOS.core.quantoniumos",
    "quantonium_os_unified": "11_QUANTONIUMOS.quantonium_os_unified",
    "launch_quantoniumos": "11_QUANTONIUMOS.core.launch_quantoniumos",
    "main": "03_RUNNING_SYSTEMS.main",
    "app": "03_RUNNING_SYSTEMS.app",
    "launch_gui": "11_QUANTONIUMOS.core.launch_gui",
    "bulletproof_quantum_kernel": "05_QUANTUM_ENGINES.bulletproof_quantum_kernel",
    "topological_quantum_kernel": "05_QUANTUM_ENGINES.topological_quantum_kernel",
    "working_quantum_kernel": "05_QUANTUM_ENGINES.working_quantum_kernel",
    "paper_compliant_rft_fixed": "04_RFT_ALGORITHMS.paper_compliant_rft_fixed",
    "canonical_true_rft": "04_RFT_ALGORITHMS.canonical_true_rft",
    "build_crypto_engine": "06_CRYPTOGRAPHY.build_crypto_engine",
    "quantonium_design_system": "11_QUANTONIUMOS.quantonium_design_system",
    "quantonium_hpc_pipeline": "11_QUANTONIUMOS.quantonium_hpc_pipeline",
}

# Log file
LOG_FILE = os.path.join(ROOT_DIR, "import_fix_log.txt")

def fix_imports_in_file(file_path):
    """Fix imports in a single file"""
    if not os.path.exists(file_path) or not file_path.endswith('.py'):
        return [], f"Skipped non-Python file: {file_path}"
    
    with open(file_path, 'r', errors='replace') as f:
        content = f.read()
    
    original_content = content
    changes = []
    
    # Fix import statements
    for module, new_path in MODULE_MAPPING.items():
        # Pattern for "from module import something"
        pattern1 = fr"from\s+{module}\s+import"
        replacement1 = f"from {new_path} import"
        if re.search(pattern1, content):
            content = re.sub(pattern1, replacement1, content)
            changes.append(f"- Changed '{pattern1}' to '{replacement1}'")
        
        # Pattern for "import module"
        pattern2 = fr"import\s+{module}(?:\s|$)"
        replacement2 = f"import {new_path} as {module}"
        if re.search(pattern2, content) and not re.search(fr"import\s+{new_path}", content):
            content = re.sub(pattern2, replacement2, content)
            changes.append(f"- Changed '{pattern2}' to '{replacement2}'")
    
    # Only write file if changes were made
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        return changes, f"Updated imports in: {file_path}"
    else:
        return [], f"No changes needed in: {file_path}"

def find_python_files(directory):
    """Find all Python files in directory and subdirectories"""
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def fix_imports():
    """Fix imports in all Python files"""
    log_entries = []
    log_entries.append("QuantoniumOS Import Fix Log")
    log_entries.append("=" * 80)
    log_entries.append("")
    
    # Find all Python files
    python_files = find_python_files(ROOT_DIR)
    log_entries.append(f"Found {len(python_files)} Python files")
    log_entries.append("")
    
    # Fix imports in each file
    files_updated = 0
    for file_path in python_files:
        changes, message = fix_imports_in_file(file_path)
        
        if changes:
            files_updated += 1
            log_entries.append(message)
            for change in changes:
                log_entries.append(f"  {change}")
            log_entries.append("")
    
    # Summary
    log_entries.append(f"Updated imports in {files_updated} files")
    
    # Write log file
    with open(LOG_FILE, 'w') as f:
        f.write('\n'.join(log_entries))
    
    return log_entries

def main():
    """Main function"""
    print("Starting QuantoniumOS import fix...")
    log_entries = fix_imports()
    
    # Print summary to console
    for entry in log_entries:
        print(entry)
    
    print(f"\nImport fix complete. Log written to {LOG_FILE}")

if __name__ == "__main__":
    main()
