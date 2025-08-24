#!/usr/bin/env python3
"""
Script to remove cleanup and organization files that are not directly wired into QuantoniumOS.
These files have already served their purpose and are no longer needed.
"""

import os
import shutil

# Files to remove - these are cleanup/organization scripts that are not directly used by the app
files_to_remove = [
    # Root cleanup files
    "/workspaces/quantoniumos/clean_unnecessary_files.py",
    "/workspaces/quantoniumos/cleanup_duplicates.py",
    "/workspaces/quantoniumos/cleanup_master.py",
    "/workspaces/quantoniumos/clean_frontend_backend.py",
    "/workspaces/quantoniumos/create_organization_plan.py",
    "/workspaces/quantoniumos/create_organization_summary.py",
    
    # Utility directory cleanup files
    "/workspaces/quantoniumos/10_UTILITIES/fast_cleanup.py",
    "/workspaces/quantoniumos/10_UTILITIES/ultra_fast_cleanup.py",
    "/workspaces/quantoniumos/10_UTILITIES/cleanup_duplicates.py",
    "/workspaces/quantoniumos/10_UTILITIES/phd_level_project_cleaner.py",
    "/workspaces/quantoniumos/10_UTILITIES/create_organization_plan.py",
    "/workspaces/quantoniumos/10_UTILITIES/create_organization_summary.py",
    "/workspaces/quantoniumos/10_UTILITIES/implement_organization_structure.py",
    
    # Cleanup summary files (already served their purpose)
    "/workspaces/quantoniumos/CLEANUP_SUMMARY.md",
    "/workspaces/quantoniumos/MASTER_CLEANUP_REPORT.md"
]

# Log file for the removal process
log_file = "/workspaces/quantoniumos/cleanup_files_removal_log.txt"

def remove_files():
    """Remove the specified files and log the results"""
    with open(log_file, "w") as log:
        log.write("QuantoniumOS Cleanup Files Removal Log\n")
        log.write("=====================================\n\n")
        log.write("Files removed:\n\n")
        
        removed_count = 0
        not_found_count = 0
        
        for file_path in files_to_remove:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    log.write(f"✅ Removed: {file_path}\n")
                    print(f"Removed: {file_path}")
                    removed_count += 1
                except Exception as e:
                    log.write(f"❌ Error removing {file_path}: {str(e)}\n")
                    print(f"Error removing {file_path}: {str(e)}")
            else:
                log.write(f"⚠️ Not found: {file_path}\n")
                print(f"Not found: {file_path}")
                not_found_count += 1
        
        log.write("\nSummary:\n")
        log.write(f"- Files removed: {removed_count}\n")
        log.write(f"- Files not found: {not_found_count}\n")
        log.write(f"- Total files processed: {len(files_to_remove)}\n")
    
    print(f"\nSummary:")
    print(f"- Files removed: {removed_count}")
    print(f"- Files not found: {not_found_count}")
    print(f"- Total files processed: {len(files_to_remove)}")
    print(f"- Log written to: {log_file}")

if __name__ == "__main__":
    print("Starting cleanup files removal process...")
    remove_files()
    print("Cleanup files removal process completed.")
