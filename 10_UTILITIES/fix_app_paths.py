#!/usr/bin/env python3
"""
Fix validator paths for app launcher files
"""

import os
import sys
import json

# Paths to fix in the validators
paths_to_fix = {
    '/workspaces/quantoniumos/apps/launch_rft_visualizer.py': '/workspaces/quantoniumos/workspaces/quantoniumos/apps/launch_rft_visualizer.py',
    '/workspaces/quantoniumos/apps/launch_quantum_simulator.py': '/workspaces/quantoniumos/workspaces/quantoniumos/apps/launch_quantum_simulator.py',
    '/workspaces/quantoniumos/apps/launch_q_notes.py': '/workspaces/quantoniumos/workspaces/quantoniumos/apps/launch_q_notes.py',
    '/workspaces/quantoniumos/apps/launch_q_vault.py': '/workspaces/quantoniumos/workspaces/quantoniumos/apps/launch_q_vault.py',
    '/workspaces/quantoniumos/apps/launch_q_mail.py': '/workspaces/quantoniumos/workspaces/quantoniumos/apps/launch_q_mail.py',
    '/workspaces/workspaces/quantoniumos/apps/launch_rft_visualizer.py': '/workspaces/quantoniumos/workspaces/quantoniumos/apps/launch_rft_visualizer.py',
    '/workspaces/workspaces/quantoniumos/apps/launch_quantum_simulator.py': '/workspaces/quantoniumos/workspaces/quantoniumos/apps/launch_quantum_simulator.py',
    '/workspaces/workspaces/quantoniumos/apps/launch_q_notes.py': '/workspaces/quantoniumos/workspaces/quantoniumos/apps/launch_q_notes.py',
    '/workspaces/workspaces/quantoniumos/apps/launch_q_vault.py': '/workspaces/quantoniumos/workspaces/quantoniumos/apps/launch_q_vault.py',
    '/workspaces/workspaces/quantoniumos/apps/launch_q_mail.py': '/workspaces/quantoniumos/workspaces/quantoniumos/apps/launch_q_mail.py'
}

# Files to modify
files_to_check = [
    '02_CORE_VALIDATORS/validate_system.py',
    'verify_system.py'
]

def fix_paths_in_file(file_path):
    """Fix paths in a Python file"""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False
        
    with open(file_path, 'r') as f:
        content = f.read()
        
    modified = False
    for old_path, new_path in paths_to_fix.items():
        if old_path in content:
            content = content.replace(old_path, new_path)
            modified = True
            
    if modified:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Updated paths in {file_path}")
    else:
        print(f"No paths to update in {file_path}")
        
    return modified

def create_app_symlinks():
    """Create symbolic links from /workspaces/apps to /workspaces/quantoniumos/apps"""
    source_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'apps')
    target_dir = '/workspaces/apps'
    
    # Create target directory if it doesn't exist
    if not os.path.exists(target_dir):
        try:
            os.makedirs(target_dir, exist_ok=True)
            print(f"Created directory: {target_dir}")
        except PermissionError:
            print(f"Permission denied: Cannot create directory {target_dir}")
            return 0
    
    # Create symbolic links for all app launcher files
    count = 0
    for launcher in os.listdir(source_dir):
        if launcher.startswith('launch_') and launcher.endswith('.py'):
            source_file = os.path.join(source_dir, launcher)
            target_file = os.path.join(target_dir, launcher)
            
            if not os.path.exists(target_file):
                try:
                    # Create a symlink
                    os.symlink(source_file, target_file)
                    print(f"Created symlink: {source_file} -> {target_file}")
                    count += 1
                except PermissionError:
                    print(f"Permission denied: Cannot create symlink to {target_file}")
                except OSError as e:
                    print(f"Error creating symlink: {e}")
    
    print(f"Created {count} symlinks for app launcher files")
    return count

def main():
    """Main function"""
    root_dir = os.path.dirname(os.path.abspath(__file__))
    modified_files = 0
    
    # First create symlinks to fix access to app files
    created_symlinks = create_app_symlinks()
    
    # Then fix paths in validation files
    for file_name in files_to_check:
        file_path = os.path.join(root_dir, file_name)
        if fix_paths_in_file(file_path):
            modified_files += 1
            
    print(f"Updated {modified_files} files")
    
    # Also search recursively for any other files that might have these paths
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.py') and file not in [os.path.basename(f) for f in files_to_check]:
                file_path = os.path.join(root, file)
                if fix_paths_in_file(file_path):
                    modified_files += 1
                    
    print(f"Total updated files: {modified_files}")
    print(f"Total created symlinks: {created_symlinks}")
    
if __name__ == "__main__":
    main()
