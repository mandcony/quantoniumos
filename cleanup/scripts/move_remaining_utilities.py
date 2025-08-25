#!/usr/bin/env python3
"""
Move remaining utility scripts to the utilities folder
"""

import os
import shutil
from datetime import datetime

# Root directory
ROOT_DIR = "/workspaces/quantoniumos"

# Utilities directory
UTILITIES_DIR = os.path.join(ROOT_DIR, "10_UTILITIES")

# Files to move
FILES_TO_MOVE = [
    "cleanup_root_duplicates.py",
    "fix_imports.py",
    "organize_files.py"
]

# Log file
LOG_FILE = os.path.join(ROOT_DIR, "final_cleanup_log.txt")

def move_file(src, dest):
    """Move a file from src to dest"""
    if not os.path.exists(src):
        return f"Source file does not exist: {src}"
    
    # Create directory if it doesn't exist
    dest_dir = os.path.dirname(dest)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # Copy file to new location
    shutil.copy2(src, dest)
    os.remove(src)  # Remove original after copying
    return f"Moved file: {src} -> {dest}"

def main():
    """Main function"""
    print("Moving remaining utility scripts to utilities folder...")
    log_entries = []
    log_entries.append(f"QuantoniumOS Final Cleanup Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_entries.append("=" * 80)
    log_entries.append("")
    
    # Ensure utilities directory exists
    if not os.path.exists(UTILITIES_DIR):
        os.makedirs(UTILITIES_DIR)
        log_entries.append(f"Created directory: {UTILITIES_DIR}")
    
    # Move each file
    for filename in FILES_TO_MOVE:
        src_path = os.path.join(ROOT_DIR, filename)
        dest_path = os.path.join(UTILITIES_DIR, filename)
        
        if os.path.exists(src_path):
            result = move_file(src_path, dest_path)
            log_entries.append(result)
    
    # Write log file
    with open(LOG_FILE, 'w') as f:
        f.write('\n'.join(log_entries))
    
    # Print summary
    for entry in log_entries:
        print(entry)
    
    print(f"\nFinal cleanup complete. Log written to {LOG_FILE}")

if __name__ == "__main__":
    main()
