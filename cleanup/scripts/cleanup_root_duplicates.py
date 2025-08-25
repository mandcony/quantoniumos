#!/usr/bin/env python3
"""
QuantoniumOS Root Directory Cleanup Script

This script removes duplicated files from the root directory that have already been
organized into their proper subdirectories according to the organization structure.
"""

import os
import shutil
from datetime import datetime

# Define the organization structure (same as in organize_files.py)
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

# Additional files that should not be deleted from root (keep these)
KEEP_IN_ROOT = [
    "organize_files.py",
    "cleanup_root_duplicates.py",
    "fix_imports.py",
    "organization_log.txt",
    "__pycache__"
]

# Root directory
ROOT_DIR = "/workspaces/quantoniumos"

# Log file
LOG_FILE = os.path.join(ROOT_DIR, "root_cleanup_log.txt")

def cleanup_root_directory():
    """Remove duplicated files from the root directory"""
    log_entries = []
    log_entries.append(f"QuantoniumOS Root Directory Cleanup Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_entries.append("=" * 80)
    log_entries.append("")
    
    # Collect all files that should be in directories
    files_to_check = []
    for directory, files in ORGANIZATION.items():
        for filename in files:
            # Check if file exists in both root and destination
            root_path = os.path.join(ROOT_DIR, filename)
            dest_path = os.path.join(ROOT_DIR, directory, filename)
            
            if os.path.exists(root_path) and os.path.exists(dest_path):
                files_to_check.append((filename, root_path, dest_path))
    
    # Remove duplicated files if they exist in proper directories
    removed_count = 0
    for filename, root_path, dest_path in files_to_check:
        # Skip files in KEEP_IN_ROOT list
        if filename in KEEP_IN_ROOT:
            log_entries.append(f"Keeping file in root (explicitly listed): {filename}")
            continue
            
        # Compare files to ensure they're the same
        try:
            with open(root_path, 'rb') as f1, open(dest_path, 'rb') as f2:
                content1 = f1.read()
                content2 = f2.read()
                
                if content1 == content2:
                    # Files are identical, safe to remove the root copy
                    os.remove(root_path)
                    log_entries.append(f"Removed duplicate file from root: {filename}")
                    removed_count += 1
                else:
                    log_entries.append(f"Warning: Files differ, keeping both: {filename}")
        except Exception as e:
            log_entries.append(f"Error comparing {filename}: {str(e)}")
    
    log_entries.append("")
    log_entries.append(f"Total files removed: {removed_count}")
    
    # Write log file
    with open(LOG_FILE, 'w') as f:
        f.write('\n'.join(log_entries))
    
    return log_entries

def main():
    """Main function"""
    print("Starting QuantoniumOS root directory cleanup...")
    log_entries = cleanup_root_directory()
    
    # Print summary to console
    for entry in log_entries:
        print(entry)
    
    print(f"\nCleanup complete. Log written to {LOG_FILE}")

if __name__ == "__main__":
    main()
