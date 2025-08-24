#!/usr/bin/env python3
"""
QuantoniumOS File Organization Script

This script organizes files from the root directory into appropriate subdirectories
based on their purpose and functionality.
"""

import os
import shutil
import json
from datetime import datetime

# Define the organization structure
ORGANIZATION = {
    # Core system files
    "core": [
        "quantoniumos.py",
        "quantonium_os_unified.py",
        "launch_quantoniumos.py",
        "main.py",
        "app.py",
        "launch_gui.py",
        "__init__.py",
    ],
    
    # Quantum kernel and algorithms
    "05_QUANTUM_ENGINES": [
        "bulletproof_quantum_kernel.py",
        "topological_quantum_kernel.py",
        "working_quantum_kernel.py",
    ],
    
    # RFT algorithms and cryptography
    "04_RFT_ALGORITHMS": [
        "paper_compliant_rft_fixed.py",
        "canonical_true_rft.py",
    ],
    
    # Cryptography-related files
    "06_CRYPTOGRAPHY": [
        "build_crypto_engine.py",
    ],
    
    # Testing files
    "07_TESTS_BENCHMARKS": [
        "test_quantum_aspects.py",
        "test_vertex_capacity.py",
        "test_encryption.py",
        "test_avalanche.py",
        "test_key_avalanche.py",
        "test_performance.py",
        "test_rft_basic.py",
        "test_rft_implementation.py",
        "test_rft_unitarity.py",
        "run_comprehensive_quantum_tests.py",
        "run_quantoniumos_test.py",
    ],
    
    # Validators
    "02_CORE_VALIDATORS": [
        "run_all_validators.py",
        "run_validators_with_path.sh",
        "wire_validators.py",
    ],
    
    # System design and documentation
    "13_DOCUMENTATION": [
        "README.md",
        "LICENSE.md",
        "CONFIG_APPROACH.md",
        "CONFIGURATION_GUIDE.md",
        "ENCRYPTION_INTEGRATION_REPORT.md",
        "VALIDATION_REPORT.md",
        "VALIDATION_FIX_REPORT.md",
        "QUANTONIUM_DESIGN_MANUAL.py",
    ],
    
    # Configuration
    "14_CONFIGURATION": [
        ".gitignore",
        ".pre-commit-config.yaml",
    ],
    
    # Pipeline and HPC-related files
    "quantonium_hpc_pipeline": [
        "quantonium_hpc_pipeline.py",
    ],
    
    # Design system
    "11_QUANTONIUMOS": [
        "quantonium_design_system.py",
    ],
    
    # Utilities
    "10_UTILITIES": [
        "fix_app_paths.py",
        "fix_null_bytes.py",
        "consolidate_similar_code.py",
        "remove_cleanup_files.py",
    ],
    
    # Validation results
    "validation_results": [
        "basic_validation_report.json",
        "comprehensive_claim_validation_results.json",
        "definitive_quantum_validation_results.json",
        "quantoniumos_validation_report.json",
    ],
    
    # Files to be cleaned up
    "cleanup": [
        "cleanup_files_removal_log.txt",
        "paper_compliant_rft_fixed.py.fixed",
        "run_all_validators.py.new",
    ]
}

# Root directory
ROOT_DIR = "/workspaces/quantoniumos"

# Log file
LOG_FILE = os.path.join(ROOT_DIR, "organization_log.txt")

def ensure_directory(directory):
    """Ensure directory exists, create if it doesn't"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        return f"Created directory: {directory}"
    return f"Directory already exists: {directory}"

def move_file(src, dest):
    """Move a file from src to dest"""
    if not os.path.exists(src):
        return f"Source file does not exist: {src}"
    
    # Create directory if it doesn't exist
    dest_dir = os.path.dirname(dest)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # Check if destination file already exists
    if os.path.exists(dest) and os.path.samefile(src, dest):
        return f"Source and destination are the same file: {src}"
    
    if os.path.exists(dest):
        # Get file stats
        src_stat = os.stat(src)
        dest_stat = os.stat(dest)
        
        # Compare modification times and sizes
        if src_stat.st_mtime > dest_stat.st_mtime or src_stat.st_size != dest_stat.st_size:
            # Backup existing file
            backup_dest = f"{dest}.bak"
            shutil.copy2(dest, backup_dest)
            # Copy source to destination
            shutil.copy2(src, dest)
            return f"Updated file (original backed up): {dest}"
        else:
            return f"Destination file is same or newer: {dest}"
    else:
        # Copy file to new location
        shutil.copy2(src, dest)
        return f"Copied file to new location: {dest}"

def organize_files():
    """Organize files according to the defined structure"""
    log_entries = []
    log_entries.append(f"QuantoniumOS File Organization Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_entries.append("=" * 80)
    log_entries.append("")
    
    # Process each directory and its files
    for directory, files in ORGANIZATION.items():
        # Create full directory path
        dir_path = os.path.join(ROOT_DIR, directory)
        
        # Ensure directory exists
        log_entries.append(ensure_directory(dir_path))
        
        # Process each file for this directory
        for filename in files:
            src_path = os.path.join(ROOT_DIR, filename)
            dest_path = os.path.join(dir_path, filename)
            
            # Move the file
            result = move_file(src_path, dest_path)
            log_entries.append(f"  {result}")
        
        log_entries.append("")
    
    # Write log file
    with open(LOG_FILE, 'w') as f:
        f.write('\n'.join(log_entries))
    
    return log_entries

def main():
    """Main function"""
    print("Starting QuantoniumOS file organization...")
    log_entries = organize_files()
    
    # Print summary to console
    for entry in log_entries:
        print(entry)
    
    print(f"\nOrganization complete. Log written to {LOG_FILE}")

if __name__ == "__main__":
    main()
