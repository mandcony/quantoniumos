"""
QuantoniumOS Project Final Organization Script

This script implements the complete organization structure as described in
ORGANIZATION_COMPLETE_SUCCESS.md to ensure the project meets professional
research-grade standards with proper directory structure.
"""

import glob
import os
import re
import shutil
import sys
from pathlib import Path

# Project root directory
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# Directory structure to create
NUMBERED_DIRS = [
    "01_START_HERE",
    "02_CORE_VALIDATORS",
    "03_RUNNING_SYSTEMS",
    "04_RFT_ALGORITHMS",
    "05_QUANTUM_ENGINES",
    "06_CRYPTOGRAPHY",
    "07_TESTS_BENCHMARKS",
    "08_RESEARCH_ANALYSIS",
    "09_LEGACY_BACKUPS",
    "10_UTILITIES",
    "11_QUANTONIUMOS",
    "12_TEST_RESULTS",
    "13_DOCUMENTATION",
    "14_CONFIGURATION",
    "15_DEPLOYMENT",
    "16_EXPERIMENTAL",
    "17_BUILD_ARTIFACTS",
    "18_DEBUG_TOOLS",
]

# Subdirectories to create within main directories
SUBDIRS = {
    "12_TEST_RESULTS": [
        "validation_reports",
        "rft_validation",
        "benchmark_results",
        "test_logs",
    ],
    "13_DOCUMENTATION": [
        "research_papers",
        "reports",
        "implementation",
        "guides",
        "legal",
    ],
    "14_CONFIGURATION": ["build_configs", "requirements", "ci_cd", "environment"],
    "15_DEPLOYMENT": ["launchers", "installers", "production"],
    "16_EXPERIMENTAL": ["prototypes", "research_data", "analysis"],
    "17_BUILD_ARTIFACTS": ["compiled", "binaries", "cache"],
    "18_DEBUG_TOOLS": ["validators", "fixers", "cleaners", "debug_scripts"],
}

# Files to keep in root directory
ROOT_FILES = ["__init__.py", "README.md", "launch_quantoniumos.py"]

# Mapping of file patterns to target directories
FILE_PATTERNS = [
    # Core source files
    (r".*quantum.*\.py", "11_QUANTONIUMOS"),
    (r".*quantonium.*\.py", "11_QUANTONIUMOS"),
    (r".*rft.*engine.*\.cpp", "05_QUANTUM_ENGINES"),
    (r".*true_rft.*\.cpp", "05_QUANTUM_ENGINES"),
    (r".*crypto.*\.py", "06_CRYPTOGRAPHY"),
    (r".*encrypt.*\.py", "06_CRYPTOGRAPHY"),
    # Test files
    (r"test_.*\.py", "07_TESTS_BENCHMARKS"),
    (r".*_test\.py", "07_TESTS_BENCHMARKS"),
    (r".*_test_suite\.py", "07_TESTS_BENCHMARKS"),
    # Validator files
    (r".*validator.*\.py", "02_CORE_VALIDATORS"),
    (r"validate_.*\.py", "02_CORE_VALIDATORS"),
    (r"verify_.*\.py", "02_CORE_VALIDATORS"),
    # Build and utility files
    (r"build_.*\.py", "10_UTILITIES"),
    (r".*_build\.py", "10_UTILITIES"),
    # RFT algorithm files
    (r".*rft.*\.py", "04_RFT_ALGORITHMS"),
    (r".*resonance.*\.py", "04_RFT_ALGORITHMS"),
    # Documentation and reports
    (r".*\.md", "13_DOCUMENTATION/reports"),
    # Debug tools
    (r"debug_.*\.py", "18_DEBUG_TOOLS/debug_scripts"),
    (r".*_debug\.py", "18_DEBUG_TOOLS/debug_scripts"),
    (r"fix_.*\.py", "18_DEBUG_TOOLS/fixers"),
    (r"patch_.*\.py", "18_DEBUG_TOOLS/fixers"),
    # Running systems
    (r"app\.py", "03_RUNNING_SYSTEMS"),
    (r"launch_.*\.py", "15_DEPLOYMENT/launchers"),
    # Configuration files
    (r".*\.toml", "14_CONFIGURATION/build_configs"),
    (r".*requirements.*\.txt", "14_CONFIGURATION/requirements"),
    (r".*\.json", "14_CONFIGURATION/environment"),
    # Experimental and research
    (r".*_hpc_.*\.py", "16_EXPERIMENTAL/prototypes"),
    (r".*analysis.*\.py", "08_RESEARCH_ANALYSIS"),
]

# Special files with exact destinations
SPECIAL_FILES = {
    "app.py": "03_RUNNING_SYSTEMS/app.py",
    "quantoniumos.py": "11_QUANTONIUMOS/quantoniumos.py",
    "launch_quantoniumos.py": "launch_quantoniumos.py",  # Keep in root
    "README.md": "README.md",  # Keep in root
    "__init__.py": "__init__.py",  # Keep in root
}


def create_directory_structure():
    """Create the directory structure for the project."""
    print("Creating directory structure...")

    # Create main numbered directories
    for directory in NUMBERED_DIRS:
        dir_path = os.path.join(PROJECT_ROOT, directory)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {directory}")

    # Create subdirectories
    for parent, subdirs in SUBDIRS.items():
        parent_path = os.path.join(PROJECT_ROOT, parent)
        for subdir in subdirs:
            subdir_path = os.path.join(parent_path, subdir)
            if not os.path.exists(subdir_path):
                os.makedirs(subdir_path)
                print(f"Created subdirectory: {parent}/{subdir}")

    # Create a navigation guide in 01_START_HERE
    create_navigation_guide()


def create_navigation_guide():
    """Create a navigation guide in the 01_START_HERE directory."""
    guide_path = os.path.join(PROJECT_ROOT, "01_START_HERE", "NAVIGATION_GUIDE.md")

    content = """# QuantoniumOS Navigation Guide

## Welcome to QuantoniumOS!

This guide will help you navigate the newly organized project structure.

## Directory Structure

### Numbered Directories
"""

    # Add numbered directories
    for directory in NUMBERED_DIRS:
        dir_name = directory.split("_", 1)[1]
        content += f"- **{directory}** - {dir_name.title().replace('_', ' ')}\n"

    content += """
### Infrastructure Directories
- **apps/** - Quantum applications
- **core/** - Core libraries
- **src/** - Source code
- **third_party/** - External dependencies

## Getting Started

1. Run the main application:
   ```
   python launch_quantoniumos.py
   ```

2. Explore the documentation:
   ```
   13_DOCUMENTATION/guides/
   ```

3. Run tests and validators:
   ```
   python 07_TESTS_BENCHMARKS/run_test_suite.py
   python 02_CORE_VALIDATORS/validate_system.py
   ```

## Directory Contents
"""

    # Add details about key directories
    for directory in NUMBERED_DIRS:
        content += f"\n### {directory}\n"

        # Add subdirectories if they exist
        if directory in SUBDIRS:
            for subdir in SUBDIRS[directory]:
                content += f"- **{subdir}/** - {subdir.replace('_', ' ').title()}\n"

    content += """
## Need Help?

If you need assistance navigating the project, please refer to the complete documentation in `13_DOCUMENTATION/`.

Happy quantum computing!
"""

    with open(guide_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Created navigation guide: 01_START_HERE/NAVIGATION_GUIDE.md")


def copy_file(source, target):
    """Copy a file, creating directories as needed."""
    target_dir = os.path.dirname(target)
    if target_dir and not os.path.exists(target_dir):
        os.makedirs(target_dir)

    try:
        # Copy the file
        shutil.copy2(source, target)
        print(
            f"Copied: {os.path.relpath(source, PROJECT_ROOT)} -> {os.path.relpath(target, PROJECT_ROOT)}"
        )
        return True
    except Exception as e:
        print(f"Error copying {source} to {target}: {e}")
        return False


def move_file(source, target):
    """Move a file, creating directories as needed."""
    # Skip if source doesn't exist
    if not os.path.exists(source):
        print(f"Source file does not exist: {source}")
        return False

    # Skip if source and target are the same
    if os.path.normpath(source) == os.path.normpath(target):
        print(f"Source and target are the same, skipping: {source}")
        return True

    # Create target directory if it doesn't exist
    target_dir = os.path.dirname(target)
    if target_dir and not os.path.exists(target_dir):
        os.makedirs(target_dir)

    try:
        # Move the file
        shutil.move(source, target)
        print(
            f"Moved: {os.path.relpath(source, PROJECT_ROOT)} -> {os.path.relpath(target, PROJECT_ROOT)}"
        )
        return True
    except Exception as e:
        print(f"Error moving {source} to {target}: {e}")

        # Try copying instead
        try:
            copy_file(source, target)
            os.remove(source)
            return True
        except:
            return False


def organize_files():
    """Organize files based on patterns."""
    print("\nOrganizing files...")

    # Get all files in the root directory
    root_files = []
    for item in os.listdir(PROJECT_ROOT):
        item_path = os.path.join(PROJECT_ROOT, item)
        if os.path.isfile(item_path):
            root_files.append(item)

    # Process special files first
    for file_name, target_path in SPECIAL_FILES.items():
        source_path = os.path.join(PROJECT_ROOT, file_name)
        target_full_path = os.path.join(PROJECT_ROOT, target_path)

        if os.path.exists(source_path) and file_name not in ROOT_FILES:
            move_file(source_path, target_full_path)

    # Handle remaining files in root
    for file_name in root_files:
        # Skip root files that should stay and already processed special files
        if file_name in ROOT_FILES or file_name in SPECIAL_FILES:
            continue

        source_path = os.path.join(PROJECT_ROOT, file_name)

        # Skip directories, .git, and hidden files
        if os.path.isdir(source_path) or file_name.startswith("."):
            continue

        # Try to match file to a pattern
        matched = False
        for pattern, target_dir in FILE_PATTERNS:
            if re.match(pattern, file_name, re.IGNORECASE):
                target_path = os.path.join(PROJECT_ROOT, target_dir, file_name)
                move_file(source_path, target_path)
                matched = True
                break

        # If no pattern matched, move to appropriate directory based on extension
        if not matched:
            if file_name.endswith(".py"):
                # Python files go to utilities by default
                target_path = os.path.join(PROJECT_ROOT, "10_UTILITIES", file_name)
            elif file_name.endswith(".md"):
                # Markdown files go to documentation
                target_path = os.path.join(
                    PROJECT_ROOT, "13_DOCUMENTATION", "reports", file_name
                )
            elif file_name.endswith(".cpp") or file_name.endswith(".h"):
                # C++ files go to quantum engines
                target_path = os.path.join(
                    PROJECT_ROOT, "05_QUANTUM_ENGINES", file_name
                )
            else:
                # Other files go to legacy backups
                target_path = os.path.join(PROJECT_ROOT, "09_LEGACY_BACKUPS", file_name)

            move_file(source_path, target_path)


def organize_tests():
    """Organize test files into appropriate directories."""
    print("\nOrganizing test files...")

    # Find all test files in the project
    test_files = []
    for root, _, files in os.walk(PROJECT_ROOT):
        for file in files:
            if (
                file.startswith("test_")
                or file.endswith("_test.py")
                or "test" in file.lower()
                and file.endswith(".py")
            ):
                test_files.append(os.path.join(root, file))

    # Move test files to test directory
    for test_file in test_files:
        # Skip if already in test directory
        if "07_TESTS_BENCHMARKS" in test_file:
            continue

        file_name = os.path.basename(test_file)
        target_path = os.path.join(PROJECT_ROOT, "07_TESTS_BENCHMARKS", file_name)

        # Copy instead of moving to avoid breaking imports
        copy_file(test_file, target_path)


def organize_documentation():
    """Organize documentation files."""
    print("\nOrganizing documentation...")

    # Find all markdown files
    md_files = []
    for root, _, files in os.walk(PROJECT_ROOT):
        for file in files:
            if file.endswith(".md") and not any(d in root for d in NUMBERED_DIRS):
                md_files.append(os.path.join(root, file))

    # Categorize documentation
    for md_file in md_files:
        file_name = os.path.basename(md_file)

        # Skip README.md in root
        if file_name == "README.md" and os.path.dirname(md_file) == PROJECT_ROOT:
            continue

        # Categorize based on content and name
        if "report" in file_name.lower():
            target_path = os.path.join(
                PROJECT_ROOT, "13_DOCUMENTATION", "reports", file_name
            )
        elif "guide" in file_name.lower() or "howto" in file_name.lower():
            target_path = os.path.join(
                PROJECT_ROOT, "13_DOCUMENTATION", "guides", file_name
            )
        elif "paper" in file_name.lower() or "research" in file_name.lower():
            target_path = os.path.join(
                PROJECT_ROOT, "13_DOCUMENTATION", "research_papers", file_name
            )
        elif "license" in file_name.lower() or "legal" in file_name.lower():
            target_path = os.path.join(
                PROJECT_ROOT, "13_DOCUMENTATION", "legal", file_name
            )
        else:
            target_path = os.path.join(
                PROJECT_ROOT, "13_DOCUMENTATION", "reports", file_name
            )

        # Copy the file
        copy_file(md_file, target_path)


def create_entry_point():
    """Create or update the main entry point file."""
    entry_point_path = os.path.join(PROJECT_ROOT, "launch_quantoniumos.py")

    content = '''#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
QuantoniumOS Main Entry Point

This is the main entry point for the QuantoniumOS system.
Execute this file to launch the complete quantum operating system.
"""

import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def main():
    """Main entry point for QuantoniumOS."""
    print("\\n" + "=" * 80)
    print(" QuantoniumOS Quantum Operating System ".center(80, "="))
    print("=" * 80 + "\\n")
    
    print("Initializing QuantoniumOS...")
    
    try:
        # Import the main application
        from _11_QUANTONIUMOS import quantoniumos
        
        # Run the system
        quantoniumos.launch()
        
        print("\\nQuantoniumOS launched successfully!")
        return 0
    except ImportError as e:
        print(f"Error importing QuantoniumOS modules: {e}")
        print("\\nPlease ensure the project is properly installed and organized.")
        return 1
    except Exception as e:
        print(f"Error launching QuantoniumOS: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
'''

    with open(entry_point_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Created main entry point: launch_quantoniumos.py")


def create_readme():
    """Create or update the main README.md file."""
    readme_path = os.path.join(PROJECT_ROOT, "README.md")

    content = """# QuantoniumOS

## Professional Quantum Computing Framework

QuantoniumOS is a comprehensive quantum computing operating system designed for research, development, and production applications.

## Getting Started

1. **Run the main application**:
   ```
   python launch_quantoniumos.py
   ```

2. **Explore the project**:
   ```
   python -m 01_START_HERE.explore
   ```

3. **Run tests**:
   ```
   python -m 07_TESTS_BENCHMARKS.run_test_suite
   ```

## Project Structure

QuantoniumOS follows a professional research-grade directory structure:

### Numbered Directories
- **01_START_HERE** - Navigation & documentation
- **02_CORE_VALIDATORS** - Validation scripts
- **03_RUNNING_SYSTEMS** - Production applications
- **04_RFT_ALGORITHMS** - RFT implementations
- **05_QUANTUM_ENGINES** - Quantum processing
- **06_CRYPTOGRAPHY** - Cryptographic systems
- **07_TESTS_BENCHMARKS** - Test infrastructure
- **08_RESEARCH_ANALYSIS** - Research tools
- **09_LEGACY_BACKUPS** - Backup storage
- **10_UTILITIES** - Build & development tools
- **11_QUANTONIUMOS** - Main operating system
- **12_TEST_RESULTS** - Centralized test data
- **13_DOCUMENTATION** - Complete documentation
- **14_CONFIGURATION** - Configuration management
- **15_DEPLOYMENT** - Production deployment
- **16_EXPERIMENTAL** - Research & prototypes
- **17_BUILD_ARTIFACTS** - Build outputs
- **18_DEBUG_TOOLS** - Debugging & validation

### Infrastructure Directories
- **apps/** - Quantum applications
- **core/** - Core libraries
- **src/** - Source code
- **third_party/** - External dependencies

## Documentation

Complete documentation is available in the `13_DOCUMENTATION` directory:

- **guides/** - User and developer guides
- **reports/** - Project status reports
- **research_papers/** - Academic papers and proofs
- **implementation/** - Implementation documentation
- **legal/** - Licenses and legal documents

## License

Copyright © 2025 QuantoniumOS Project. All rights reserved.
"""

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Created main README.md")


def create_init_files():
    """Create __init__.py files in all directories to make them proper packages."""
    print("\nCreating __init__.py files...")

    # Create in numbered directories
    for directory in NUMBERED_DIRS:
        dir_path = os.path.join(PROJECT_ROOT, directory)
        init_path = os.path.join(dir_path, "__init__.py")

        if not os.path.exists(init_path):
            with open(init_path, "w", encoding="utf-8") as f:
                f.write(f'"""\n{directory} package for QuantoniumOS.\n"""\n')
            print(f"Created: {directory}/__init__.py")

        # Create in subdirectories
        if directory in SUBDIRS:
            for subdir in SUBDIRS[directory]:
                subdir_path = os.path.join(dir_path, subdir)
                sub_init_path = os.path.join(subdir_path, "__init__.py")

                if not os.path.exists(sub_init_path):
                    with open(sub_init_path, "w", encoding="utf-8") as f:
                        f.write(
                            f'"""\n{directory}/{subdir} package for QuantoniumOS.\n"""\n'
                        )
                    print(f"Created: {directory}/{subdir}/__init__.py")


def fix_imports():
    """Fix imports in Python files to use the new directory structure."""
    print("\nFixing imports in Python files...")

    # Mapping of old paths to new paths
    path_mapping = {}

    # Build the mapping
    for root, _, files in os.walk(PROJECT_ROOT):
        for file in files:
            if file.endswith(".py"):
                rel_path = os.path.relpath(os.path.join(root, file), PROJECT_ROOT)
                module_name = os.path.splitext(file)[0]
                path_mapping[module_name] = rel_path

    # Fix imports in all Python files
    for root, _, files in os.walk(PROJECT_ROOT):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)

                # Skip virtual environment files
                if ".venv" in file_path:
                    continue

                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()

                    # Fix direct imports
                    modified = False
                    for module_name, new_path in path_mapping.items():
                        if (
                            f"import {module_name}" in content
                            or f"from {module_name} import" in content
                        ):
                            # Convert path to import format
                            import_path = os.path.dirname(new_path).replace(os.sep, ".")
                            if import_path:
                                # Replace the import
                                new_import = f"from {import_path} import {module_name}"
                                pattern = rf"(import\s+{module_name}|from\s+{module_name}\s+import)"
                                if re.search(pattern, content):
                                    content = re.sub(pattern, new_import, content)
                                    modified = True

                    if modified:
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(content)
                        print(
                            f"Fixed imports in: {os.path.relpath(file_path, PROJECT_ROOT)}"
                        )
                except Exception as e:
                    print(f"Error fixing imports in {file_path}: {e}")


def main():
    print("\n" + "=" * 80)
    print(" QuantoniumOS Project Organization ".center(80, "="))
    print("=" * 80 + "\n")

    print("Starting complete project organization...")

    # Create the directory structure
    create_directory_structure()

    # Create important files
    create_entry_point()
    create_readme()

    # Organize files
    organize_files()
    organize_tests()
    organize_documentation()

    # Create package structure
    create_init_files()

    # Fix imports (commented out for safety - can be complex)
    # fix_imports()

    print("\n" + "=" * 80)
    print(" Organization Complete! ".center(80, "="))
    print("=" * 80 + "\n")

    print(
        "The QuantoniumOS project has been organized according to professional research-grade standards."
    )
    print("Main entry point: launch_quantoniumos.py")
    print("Project navigation: 01_START_HERE/NAVIGATION_GUIDE.md")
    print("\nREADY FOR ADVANCED USE!\n")


if __name__ == "__main__":
    main()
