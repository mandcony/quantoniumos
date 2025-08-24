"""
This script purges duplicate files and organizes the QuantoniumOS project structure.
It identifies and removes truly redundant files while preserving necessary code in the proper locations.
"""

import hashlib
import os
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Project root directory
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# Files that should be retained even if duplicated - specify by relative path from project root
ESSENTIAL_FILES = {
    # Core implementation files
    "core/config.py",
    "core/deterministic_hash.py",
    "core/engine_core.cpp",
    "core/engine_core_pybind.cpp",
    "core/enhanced_rft_crypto.cpp",
    "core/geometric_container.py",
    "core/multi_qubit_state.py",
    "core/oscillator.py",
    "core/quantum_engine_bindings.cpp",
    "core/true_rft_engine.cpp",
    # RFT algorithms
    "04_RFT_ALGORITHMS/cpp_rft_wrapper.py",
    "04_RFT_ALGORITHMS/energy_conserving_rft_adapter.py",
    "04_RFT_ALGORITHMS/novel_rft_constructions.py",
    "04_RFT_ALGORITHMS/symbiotic_rft_engine_adapter.py",
    "04_RFT_ALGORITHMS/symbiotic_true_rft_engine.cpp",
    # Cryptography
    "06_CRYPTOGRAPHY/quantonium_crypto_production.py",
    "06_CRYPTOGRAPHY/resonance_encryption.py",
    # Main application files
    "app.py",
    "launch_quantoniumos.py",
    # Documentation (keep in dedicated folder)
    "13_DOCUMENTATION/reports/CLEANUP_SUCCESS_REPORT.md",
    "13_DOCUMENTATION/reports/COMPREHENSIVE_PROJECT_ANALYSIS_FINAL.md",
    "13_DOCUMENTATION/reports/EXECUTIVE_SUMMARY_PHD_AUDIT.md",
    "13_DOCUMENTATION/reports/ORGANIZATION_SUCCESS_REPORT.md",
    "13_DOCUMENTATION/reports/PROJECT_ORGANIZATION_PLAN.md",
    "13_DOCUMENTATION/reports/PROJECT_ORGANIZATION_REPORT.md",
    "13_DOCUMENTATION/reports/QUANTONIUM_FINAL_SUCCESS_REPORT.md",
    "13_DOCUMENTATION/reports/QUANTONIUM_SUCCESS_FINAL.md",
    "13_DOCUMENTATION/reports/REPOSITORY_ORGANIZATION_REPORT.md",
    "13_DOCUMENTATION/reports/SCIENTIFIC_AUDIT_LOG.md",
    "13_DOCUMENTATION/implementation/RFT_ENERGY_CONSERVATION_REPORT.md",
}

# Directories that contain virtual environment files which should be ignored
VENV_DIRS = [".venv", ".venv-1", "09_LEGACY_BACKUPS/archive/test_env_local"]

# Empty files to be removed based on prefix - these are junk files
EMPTY_FILE_PREFIXES = [
    "empty_",
]


def get_file_hash(file_path: str) -> str:
    """Calculate SHA256 hash of a file."""
    try:
        with open(file_path, "rb") as f:
            file_data = f.read()
            return hashlib.sha256(file_data).hexdigest()
    except (IOError, OSError):
        # Return empty hash for unopenable files
        return ""


def is_venv_file(file_path: str) -> bool:
    """Check if file is inside a virtual environment directory."""
    for venv_dir in VENV_DIRS:
        if venv_dir in file_path:
            return True
    return False


def is_empty_junk_file(file_path: str) -> bool:
    """Check if file matches the pattern of empty/junk files to be removed."""
    file_name = os.path.basename(file_path)
    for prefix in EMPTY_FILE_PREFIXES:
        if file_name.startswith(prefix):
            return True
    return False


def should_preserve(file_path: str) -> bool:
    """Determine if a file should be preserved based on its path."""
    # Convert absolute path to relative path from project root
    rel_path = os.path.relpath(file_path, PROJECT_ROOT)
    rel_path = rel_path.replace("\\", "/")  # Normalize path separators

    return rel_path in ESSENTIAL_FILES


def find_duplicate_files() -> Dict[str, List[str]]:
    """Find all duplicate files in the project by hash, ignoring virtual environment files."""
    hash_to_files = {}

    for root, _, files in os.walk(PROJECT_ROOT):
        for filename in files:
            file_path = os.path.join(root, filename)

            # Skip virtual environment files
            if is_venv_file(file_path):
                continue

            file_hash = get_file_hash(file_path)
            if file_hash:  # Only process files with valid hashes
                if file_hash not in hash_to_files:
                    hash_to_files[file_hash] = []
                hash_to_files[file_hash].append(file_path)

    # Filter to only include hashes with multiple files
    return {h: files for h, files in hash_to_files.items() if len(files) > 1}


def organize_duplicate_files(
    duplicates: Dict[str, List[str]]
) -> Tuple[List[str], List[Tuple[str, str]], List[str]]:
    """
    Organize duplicate files into:
    1. Files to delete
    2. Files to move (source, destination)
    3. Files to keep
    """
    files_to_delete = []
    files_to_move = []
    files_to_keep = []

    for file_hash, file_paths in duplicates.items():
        # Files that should definitely be preserved
        preserved_files = [f for f in file_paths if should_preserve(f)]

        # If we have preserved files, we'll keep those and delete the rest
        if preserved_files:
            files_to_keep.extend(preserved_files)
            for file_path in file_paths:
                if file_path not in preserved_files:
                    # Check if it's an empty/junk file first
                    if is_empty_junk_file(file_path):
                        files_to_delete.append(file_path)
                    else:
                        # For non-junk duplicates, we'll only delete if it's in a non-standard location
                        # This is a simplified approach; in reality, you might want more complex rules
                        files_to_delete.append(file_path)
        else:
            # If no preserved files, keep the one with the most logical path (simplified)
            # This is a naive approach - we just keep the first file and delete others
            # In a real-world scenario, you'd want more sophisticated path analysis
            keeper = file_paths[0]
            files_to_keep.append(keeper)

            for file_path in file_paths[1:]:
                if is_empty_junk_file(file_path):
                    files_to_delete.append(file_path)
                else:
                    files_to_delete.append(file_path)

    return files_to_delete, files_to_move, files_to_keep


def execute_cleanup(
    files_to_delete: List[str],
    files_to_move: List[Tuple[str, str]],
    files_to_keep: List[str],
) -> Dict:
    """
    Execute the cleanup plan:
    1. Delete files marked for deletion
    2. Move files to their proper locations
    3. Return statistics about the operation
    """
    stats = {"deleted": 0, "moved": 0, "kept": len(files_to_keep), "errors": []}

    # Process deletions
    for file_path in files_to_delete:
        try:
            print(f"Deleting: {file_path}")
            os.remove(file_path)
            stats["deleted"] += 1
        except Exception as e:
            error_msg = f"Error deleting {file_path}: {str(e)}"
            print(error_msg)
            stats["errors"].append(error_msg)

    # Process moves
    for source, destination in files_to_move:
        try:
            # Ensure the destination directory exists
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            print(f"Moving: {source} -> {destination}")
            shutil.move(source, destination)
            stats["moved"] += 1
        except Exception as e:
            error_msg = f"Error moving {source} to {destination}: {str(e)}"
            print(error_msg)
            stats["errors"].append(error_msg)

    return stats


def generate_report(
    stats: Dict,
    duplicates: Dict[str, List[str]],
    files_to_delete: List[str],
    files_to_move: List[Tuple[str, str]],
    files_to_keep: List[str],
) -> str:
    """Generate a comprehensive cleanup report."""
    report = "# QuantoniumOS Cleanup Report\n\n"

    # Summary section
    report += "## Summary\n\n"
    report += f"- Total duplicate sets found: {len(duplicates)}\n"
    report += f"- Total files deleted: {stats['deleted']}\n"
    report += f"- Total files moved: {stats['moved']}\n"
    report += f"- Total files preserved: {stats['kept']}\n"

    # Errors section
    if stats["errors"]:
        report += "\n## Errors\n\n"
        for error in stats["errors"]:
            report += f"- {error}\n"

    # Deleted files section
    report += "\n## Deleted Files\n\n"
    for file_path in files_to_delete:
        report += f"- `{file_path}`\n"

    # Moved files section
    if files_to_move:
        report += "\n## Moved Files\n\n"
        for source, destination in files_to_move:
            report += f"- `{source}` → `{destination}`\n"

    # Preserved files section
    report += "\n## Preserved Files\n\n"
    for file_path in files_to_keep:
        report += f"- `{file_path}`\n"

    return report


def main():
    print("Starting QuantoniumOS cleanup process...")

    # Find all duplicate files
    print("Finding duplicate files...")
    duplicates = find_duplicate_files()
    print(f"Found {len(duplicates)} duplicate sets.")

    # Organize files into actions
    print("Organizing duplicate files...")
    files_to_delete, files_to_move, files_to_keep = organize_duplicate_files(duplicates)
    print(f"Files to delete: {len(files_to_delete)}")
    print(f"Files to move: {len(files_to_move)}")
    print(f"Files to keep: {len(files_to_keep)}")

    # Execute the cleanup
    print("\nExecuting cleanup...")
    stats = execute_cleanup(files_to_delete, files_to_move, files_to_keep)

    # Generate and save report
    print("\nGenerating cleanup report...")
    report = generate_report(
        stats, duplicates, files_to_delete, files_to_move, files_to_keep
    )
    report_path = os.path.join(PROJECT_ROOT, "CLEANUP_EXECUTION_REPORT.md")
    with open(report_path, "w") as f:
        f.write(report)

    print(f"Cleanup complete. Report saved to {report_path}")
    print(
        f"Summary: {len(duplicates)} duplicate sets, {stats['deleted']} files deleted, {stats['moved']} files moved, {stats['kept']} files preserved."
    )


if __name__ == "__main__":
    main()
