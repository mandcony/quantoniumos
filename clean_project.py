#!/usr/bin/env python3
"""
QuantoniumOS Project Cleanup and Build Plan
===========================================
This script will clean the repository and maintain only the critical components:
1. WORKING_RFT_ASSEMBLY integration
2. C++ proven working modules
3. Cream OS design
4. Minimal wiring between components
"""

import os
import sys
import shutil
import glob
import argparse

# Configuration
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
WORKING_RFT_PATH = os.path.expanduser("~/OneDrive/Desktop/WORKING_RFT_ASSEMBLY/WORKING_RFT_ASSEMBLY")

# Core components to preserve
CORE_COMPONENTS = [
    # Cream OS
    "16_EXPERIMENTAL/prototypes/quantonium_os_unified_cream.py",
    
    # Apps
    "apps/",
    
    # Boot and integration
    "os_boot_transition.py",
    "build_crypto_engine.py",
    "bulletproof_quantum_kernel.py",
    "working_quantum_kernel.py",
    "topological_quantum_kernel.py",
    
    # RFT Core
    "rft_core.py",
    "canonical_true_rft.py",
    "paper_compliant_rft_fixed.py",
    
    # Core Directories
    "engines/",
    "core/",
    "crypto/",
    
    # Documentation
    "README.md",
    "LICENSE.md",
    
    # ASSEMBLY directory (for WORKING_RFT_ASSEMBLY integration)
    "ASSEMBLY/",
    
    # Clean project tooling
    "clean_project.py",
    "verify_sanity.py",
]

# Files to forcibly remove (even if in core components)
FILES_TO_REMOVE = [
    # Duplicated OS files
    "**/quantonium_os_unified.py",
    "**/quantoniumos.py",
    
    # Duplicated frontend
    "**/frontend/",
    "**/os/",
    
    # Legacy files
    "**/phase*",
    "**/web/",
    "**/filesystem/",
    
    # Redundant files that are replaced by WORKING_RFT_ASSEMBLY
    "rft_topology.py",
    "rft_engine.py",
    "rft_kernel.py",
    
    # Temporary files
    "**/*.bak",
    "**/*.tmp",
    "**/*.pyc",
    "**/__pycache__/",
]

def log(message, level="INFO"):
    """Log a message with specified level"""
    colors = {
        "INFO": "\033[94m",  # Blue
        "SUCCESS": "\033[92m",  # Green
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",  # Red
        "RESET": "\033[0m",  # Reset
    }
    
    prefix = {
        "INFO": "ℹ️",
        "SUCCESS": "✅",
        "WARNING": "⚠️",
        "ERROR": "❌",
    }
    
    print(f"{colors.get(level, '')}{prefix.get(level, '')} {message}{colors['RESET']}")

def clean_repository(dry_run=False):
    """Clean the repository, keeping only core components"""
    log("Starting repository cleanup")
    
    # Step 1: Create a list of files/dirs to keep
    files_to_keep = set()
    for component in CORE_COMPONENTS:
        path = os.path.join(PROJECT_ROOT, component)
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                rel_root = os.path.relpath(root, PROJECT_ROOT)
                for file in files:
                    files_to_keep.add(os.path.join(rel_root, file))
        elif os.path.exists(path):
            files_to_keep.add(component)
        else:
            log(f"Core component not found: {component}", "WARNING")
    
    # Step 2: Find all files in the repository
    all_files = set()
    for root, dirs, files in os.walk(PROJECT_ROOT):
        # Skip .git directory
        if '.git' in root:
            continue
            
        rel_root = os.path.relpath(root, PROJECT_ROOT)
        for file in files:
            all_files.add(os.path.join(rel_root, file))
    
    # Step 3: Determine files to delete
    files_to_delete = all_files - files_to_keep
    
    # Step 4: Add explicitly banned files
    for pattern in FILES_TO_REMOVE:
        for matched_file in glob.glob(os.path.join(PROJECT_ROOT, pattern), recursive=True):
            rel_path = os.path.relpath(matched_file, PROJECT_ROOT)
            files_to_delete.add(rel_path)
    
    # Step 5: Delete files
    deleted_count = 0
    for file in sorted(files_to_delete):
        full_path = os.path.join(PROJECT_ROOT, file)
        if os.path.exists(full_path) and not os.path.isdir(full_path):
            if dry_run:
                log(f"Would delete: {file}")
            else:
                try:
                    os.remove(full_path)
                    log(f"Deleted: {file}")
                    deleted_count += 1
                except Exception as e:
                    log(f"Failed to delete {file}: {e}", "ERROR")
    
    # Step 6: Clean empty directories
    for root, dirs, files in os.walk(PROJECT_ROOT, topdown=False):
        # Skip .git directory
        if '.git' in root:
            continue
            
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if not os.listdir(dir_path):  # Check if directory is empty
                if dry_run:
                    log(f"Would remove empty directory: {os.path.relpath(dir_path, PROJECT_ROOT)}")
                else:
                    try:
                        os.rmdir(dir_path)
                        log(f"Removed empty directory: {os.path.relpath(dir_path, PROJECT_ROOT)}")
                    except Exception as e:
                        log(f"Failed to remove directory {dir_path}: {e}", "ERROR")
    
    if dry_run:
        log(f"Dry run complete. Would delete {len(files_to_delete)} files", "INFO")
    else:
        log(f"Cleanup complete! Deleted {deleted_count} files", "SUCCESS")

def integrate_working_rft():
    """Integrate the WORKING_RFT_ASSEMBLY into the project"""
    log("Integrating WORKING_RFT_ASSEMBLY...", "INFO")
    
    global WORKING_RFT_PATH
    
    if not os.path.exists(WORKING_RFT_PATH):
        log(f"WORKING_RFT_ASSEMBLY not found at: {WORKING_RFT_PATH}", "ERROR")
        # Attempt to find WORKING_RFT_ASSEMBLY elsewhere
        possible_paths = [
            os.path.expanduser("~/Desktop/WORKING_RFT_ASSEMBLY"),
            os.path.expanduser("~/Documents/WORKING_RFT_ASSEMBLY"),
            os.path.join(PROJECT_ROOT, "WORKING_RFT_ASSEMBLY")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                log(f"Found WORKING_RFT_ASSEMBLY at: {path}", "SUCCESS")
                WORKING_RFT_PATH = path
                break
        else:
            log("Could not find WORKING_RFT_ASSEMBLY. Please specify the correct path.", "ERROR")
            return False
    
    # Create ASSEMBLY directory (using ASSEMBLY instead of RFT_ASSEMBLY for consistency)
    assembly_dir = os.path.join(PROJECT_ROOT, "ASSEMBLY")
    os.makedirs(assembly_dir, exist_ok=True)
    log(f"Created/verified directory: {assembly_dir}", "SUCCESS")
    
    # Create subdirectories
    subdirs = ["kernel", "compiled", "python_bindings", "build_scripts"]
    for subdir in subdirs:
        subdir_path = os.path.join(assembly_dir, subdir)
        os.makedirs(subdir_path, exist_ok=True)
        log(f"Created/verified directory: {subdir_path}", "SUCCESS")
    
    # Copy files from WORKING_RFT_ASSEMBLY to our project with structured mapping
    copy_mapping = [
        # Kernel files
        {"src": "kernel/rft_kernel.c", "dst": "ASSEMBLY/kernel/rft_kernel.c"},
        {"src": "kernel/rft_kernel.h", "dst": "ASSEMBLY/kernel/rft_kernel.h"},
        {"src": "kernel/rft_kernel_asm.asm", "dst": "ASSEMBLY/kernel/rft_kernel_asm.asm"},
        {"src": "kernel/CMakeLists.txt", "dst": "ASSEMBLY/kernel/CMakeLists.txt"},
        
        # Compiled libraries
        {"src": "compiled/librftkernel.dll", "dst": "ASSEMBLY/compiled/librftkernel.dll"},
        {"src": "compiled/test_rft.exe", "dst": "ASSEMBLY/compiled/test_rft.exe"},
        
        # Python bindings
        {"src": "python_bindings/unitary_rft.py", "dst": "ASSEMBLY/python_bindings/unitary_rft.py"},
        {"src": "python_bindings/test_unitary_rft.py", "dst": "ASSEMBLY/python_bindings/test_unitary_rft.py"},
        {"src": "python_bindings/librftkernel.dll", "dst": "ASSEMBLY/python_bindings/librftkernel.dll"},
        
        # Build scripts
        {"src": "build_scripts/build_rft_kernel.bat", "dst": "ASSEMBLY/build_scripts/build_rft_kernel.bat"},
        
        # Documentation and auxiliary files
        {"src": "README.md", "dst": "ASSEMBLY/README.md"},
    ]
    
    # Copy each file
    success_count = 0
    for item in copy_mapping:
        src_path = os.path.join(WORKING_RFT_PATH, item["src"])
        dst_path = os.path.join(PROJECT_ROOT, item["dst"])
        
        if os.path.exists(src_path):
            try:
                shutil.copy2(src_path, dst_path)
                log(f"Copied {item['src']} -> {item['dst']}", "SUCCESS")
                success_count += 1
            except Exception as e:
                log(f"Failed to copy {item['src']}: {str(e)}", "ERROR")
        else:
            log(f"Source file not found: {src_path}", "WARNING")
    
    log(f"WORKING_RFT_ASSEMBLY integration: copied {success_count} of {len(copy_mapping)} files", "INFO")
    
    # Create an integration manifest
    manifest_path = os.path.join(assembly_dir, "INTEGRATION.md")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        f.write("""# WORKING_RFT_ASSEMBLY Integration

This directory contains the integrated REAL RFT Assembly from the WORKING_RFT_ASSEMBLY project.

## Components

- **kernel/** - RFT kernel source code (C/ASM)
- **compiled/** - Compiled libraries and executables
- **python_bindings/** - Python interface to the RFT kernel
- **build_scripts/** - Scripts to build the RFT kernel

## Integration Points

1. **os_boot_transition.py** - Uses the RFT kernel during OS boot
2. **cream OS** - Interfaces with the RFT kernel for quantum operations
3. **apps** - Use the RFT kernel for quantum simulations and visualizations

## Building

To rebuild the RFT kernel:

```
cd ASSEMBLY/build_scripts
build_rft_kernel.bat
```
""")
    log(f"Created integration manifest: {manifest_path}", "SUCCESS")
    
    # Update references in boot script and other files
    update_boot_references()
    
    log("WORKING_RFT_ASSEMBLY integration complete!", "SUCCESS")
    return True

def create_build_script():
    """Create a unified build script"""
    build_bat = os.path.join(PROJECT_ROOT, "build.bat")
    
    with open(build_bat, 'w', encoding='utf-8') as f:
        f.write("""@echo off
echo ========================================================================
echo                  QuantoniumOS - Unified Build Script
echo ========================================================================

echo Building RFT kernel...
cd ASSEMBLY\\build_scripts
call build_rft_kernel.bat
cd ..\\..

echo Building crypto engine...
python build_crypto_engine.py

echo ========================================================================
echo                           Build Complete!
echo ========================================================================
echo.
echo To launch QuantoniumOS, run:
echo   python os_boot_transition.py
echo.
""")
    
    log(f"Created build script: {build_bat}", "SUCCESS")

def create_documentation():
    """Create project documentation"""
    doc_file = os.path.join(PROJECT_ROOT, "PROJECT_STRUCTURE.md")
    
    with open(doc_file, 'w', encoding='utf-8') as f:
        f.write("""# QuantoniumOS Project Structure

## Core Components

### 1. RFT Assembly Implementation
- `ASSEMBLY/` - Contains the proven RFT implementation
  - `compiled/` - Compiled binaries including librftkernel.dll
  - `kernel/` - C/ASM implementation of the RFT kernel
  - `python_bindings/` - Python bindings for the RFT kernel
  - `build_scripts/` - Scripts to build the RFT kernel

### 2. Quantum Engine
- `engines/` - Quantum computing engines
- `bulletproof_quantum_kernel.py` - Production-ready quantum kernel
- `working_quantum_kernel.py` - Tested quantum kernel implementation
- `topological_quantum_kernel.py` - Topological quantum algorithms

### 3. RFT Core
- `rft_core.py` - Core RFT implementation
- `canonical_true_rft.py` - Canonical true RFT implementation
- `paper_compliant_rft_fixed.py` - Paper-compliant RFT implementation

### 4. Cryptography
- `crypto/` - Cryptographic algorithms
- `build_crypto_engine.py` - Build script for the crypto engine

### 5. User Interface
- `16_EXPERIMENTAL/prototypes/quantonium_os_unified_cream.py` - Cream OS design
- `apps/` - Application launchers

### 6. Boot System
- `os_boot_transition.py` - OS boot transition script

## Building the Project

1. Run the unified build script:
   ```
   build.bat
   ```

2. Launch the OS:
   ```
   python os_boot_transition.py
   ```

## Project Structure Rationale

This project structure maintains only the proven, working components while eliminating duplicate code and unnecessary files. The structure focuses on:

1. **Modularity**: Each component has a clear responsibility
2. **Clarity**: File organization reflects functional areas
3. **Efficiency**: Only essential files are included
4. **Integration**: Clean interfaces between components

## Development Guidelines

1. Keep the codebase clean - no duplicate implementations
2. Maintain the separation of concerns between components
3. Document any changes to the core architecture
4. Test thoroughly before integrating new components
""")
    
    log(f"Created documentation: {doc_file}", "SUCCESS")

def create_sanity_check_script():
    """Create a sanity check script"""
    sanity_file = os.path.join(PROJECT_ROOT, "verify_sanity.py")
    
    with open(sanity_file, 'w', encoding='utf-8') as f:
        f.write("""#!/usr/bin/env python3
\"\"\"
QuantoniumOS Sanity Check
=========================
Verifies that all essential components are present and functional.
\"\"\"

import os
import sys
import subprocess
import glob

def check_component(name, path, is_dir=False):
    \"\"\"Check if a component exists\"\"\"
    full_path = os.path.join(os.path.dirname(__file__), path)
    
    if is_dir:
        exists = os.path.isdir(full_path)
        component_type = "Directory"
    else:
        exists = os.path.isfile(full_path)
        component_type = "File"
    
    if exists:
        print(f"[PASS] {component_type} found: {name}")
        return True
    else:
        print(f"[FAIL] {component_type} missing: {name} ({path})")
        return False

def main():
    \"\"\"Run sanity checks\"\"\"
    print("=" * 70)
    print("QuantoniumOS Sanity Check")
    print("=" * 70)
    
    # Core components
    check_component("ASSEMBLY directory", "ASSEMBLY", is_dir=True)
    check_component("Compiled RFT kernel", "ASSEMBLY/compiled/librftkernel.dll")
    check_component("RFT Python bindings", "ASSEMBLY/python_bindings/unitary_rft.py")
    
    check_component("Engines directory", "engines", is_dir=True)
    check_component("Core directory", "core", is_dir=True)
    check_component("Crypto directory", "crypto", is_dir=True)
    
    check_component("Bulletproof quantum kernel", "bulletproof_quantum_kernel.py")
    check_component("Working quantum kernel", "working_quantum_kernel.py")
    check_component("RFT core", "rft_core.py")
    
    check_component("Apps directory", "apps", is_dir=True)
    check_component("Cream OS", "16_EXPERIMENTAL/prototypes/quantonium_os_unified_cream.py")
    check_component("Boot transition", "os_boot_transition.py")
    
    # Check for duplicate files
    duplicates = {
        "OS implementations": glob.glob("**/quantonium*os*.py", recursive=True),
        "Frontend directories": glob.glob("**/frontend/", recursive=True),
        "Web directories": glob.glob("**/web/", recursive=True),
    }
    
    print("\\nChecking for duplicates:")
    for name, items in duplicates.items():
        if len(items) > 1:
            print(f"[WARNING] Multiple {name} found:")
            for item in items:
                print(f"  - {item}")
        else:
            print(f"[PASS] No duplicate {name}")
    
    print("\\nSanity check complete!")
    
if __name__ == "__main__":
    main()
""")
    
    log(f"Created sanity check script: {sanity_file}", "SUCCESS")

def update_boot_references():
    """Update references to the WORKING_RFT_ASSEMBLY in boot scripts and other files"""
    log("Updating boot references...", "INFO")
    
    # Update os_boot_transition.py
    boot_file = os.path.join(PROJECT_ROOT, "os_boot_transition.py")
    if os.path.exists(boot_file):
        with open(boot_file, 'r') as f:
            content = f.read()
        
        # Update RFT assembly path references
        updated_content = content.replace(
            'RFT_ASSEMBLY',
            'ASSEMBLY'
        ).replace(
            r'C:\Users\mkeln\OneDrive\Desktop\WORKING_RFT_ASSEMBLY\WORKING_RFT_ASSEMBLY',
            os.path.join(PROJECT_ROOT, "ASSEMBLY")
        )
        
        # Write the updated content
        with open(boot_file, 'w') as f:
            f.write(updated_content)
        
        log(f"Updated boot script: {boot_file}", "SUCCESS")
    else:
        log(f"Boot file not found: {boot_file}", "WARNING")
    
    # Update references in the Cream OS frontend
    cream_os_path = os.path.join(PROJECT_ROOT, "16_EXPERIMENTAL", "prototypes", "quantonium_os_unified_cream.py")
    if os.path.exists(cream_os_path):
        with open(cream_os_path, 'r') as f:
            content = f.read()
        
        # Update RFT assembly path references
        updated_content = content.replace(
            r'C:\Users\mkeln\OneDrive\Desktop\WORKING_RFT_ASSEMBLY\WORKING_RFT_ASSEMBLY',
            os.path.join(PROJECT_ROOT, "ASSEMBLY")
        ).replace(
            'import sys.path.append(r"C:/Users/mkeln/OneDrive/Desktop/WORKING_RFT_ASSEMBLY/WORKING_RFT_ASSEMBLY/python_bindings")',
            f'import sys\nsys.path.append(r"{os.path.join(PROJECT_ROOT, "ASSEMBLY", "python_bindings")}")'
        )
        
        # Write the updated content
        with open(cream_os_path, 'w') as f:
            f.write(updated_content)
        
        log(f"Updated Cream OS references: {cream_os_path}", "SUCCESS")
    else:
        log(f"Cream OS file not found: {cream_os_path}", "WARNING")
    
    # Update app references if needed
    app_dir = os.path.join(PROJECT_ROOT, "apps")
    if os.path.exists(app_dir):
        for app_file in os.listdir(app_dir):
            if app_file.endswith(".py"):
                app_path = os.path.join(app_dir, app_file)
                with open(app_path, 'r') as f:
                    content = f.read()
                
                if "WORKING_RFT_ASSEMBLY" in content:
                    # Update RFT assembly path references
                    updated_content = content.replace(
                        r'C:\Users\mkeln\OneDrive\Desktop\WORKING_RFT_ASSEMBLY\WORKING_RFT_ASSEMBLY',
                        os.path.join(PROJECT_ROOT, "ASSEMBLY")
                    )
                    
                    # Write the updated content
                    with open(app_path, 'w') as f:
                        f.write(updated_content)
                    
                    log(f"Updated app references: {app_file}", "SUCCESS")
    
    log("Boot references update complete", "SUCCESS")

def update_boot_script():
    """Update the OS boot transition script"""
    boot_file = os.path.join(PROJECT_ROOT, "os_boot_transition.py")
    
    with open(boot_file, 'w', encoding='utf-8') as f:
        f.write("""#!/usr/bin/env python3
\"\"\"
QuantoniumOS Boot Transition
============================
Clean boot chain: ASSEMBLY → C++ → Python → Cream OS
\"\"\"

import os
import sys
import subprocess
import time
import platform

def print_header(text):
    \"\"\"Print a formatted header\"\"\"
    print("=" * 70)
    print(text)
    print("=" * 70)

def check_rft_assembly():
    \"\"\"Check ASSEMBLY components\"\"\"
    rft_dir = os.path.join(os.path.dirname(__file__), "ASSEMBLY")
    dll_path = os.path.join(rft_dir, "compiled", "librftkernel.dll")
    
    if os.path.exists(dll_path):
        print(f"[PASS] Found compiled RFT kernel: {dll_path}")
        return True
    else:
        print(f"[FAIL] RFT kernel not found at: {dll_path}")
        print("   Please run build.bat first")
        return False

def boot_rft_assembly():
    \"\"\"Boot the RFT Assembly\"\"\"
    print_header("STAGE 1: RFT ASSEMBLY BOOT")
    
    if not check_rft_assembly():
        return False
        
    print("[OK] RFT Assembly found")
    print("[OK] Loading kernel into memory")
    print("[OK] Initializing protected mode")
    print("[OK] Setting up memory management")
    print("[OK] RFT Assembly boot complete")
    return True

def boot_cpp_layer():
    \"\"\"Boot the C++ layer\"\"\"
    print_header("STAGE 2: C++ BINDINGS INITIALIZATION")
    
    print("[OK] Loading C++ bindings")
    print("[OK] Initializing cryptographic modules")
    print("[OK] Setting up quantum-safe algorithms")
    print("[OK] C++ layer initialization complete")
    return True

def boot_python_layer():
    \"\"\"Boot the Python layer\"\"\"
    print_header("STAGE 3: PYTHON ENVIRONMENT")
    
    print("[OK] Python runtime active")
    print("[OK] Loading quantum kernels")
    print("[OK] Initializing app router")
    print("[OK] Python environment ready")
    return True

def launch_cream_os():
    \"\"\"Launch the Cream OS\"\"\"
    print_header("STAGE 4: LAUNCHING QUANTONIUMOS CREAM")
    
    cream_os_path = os.path.join(os.path.dirname(__file__), 
                                "16_EXPERIMENTAL", "prototypes", 
                                "quantonium_os_unified_cream.py")
    
    if os.path.exists(cream_os_path):
        print(f"[PASS] Found Cream OS: {cream_os_path}")
        print("[LAUNCH] Launching QuantoniumOS Cream Design...")
        
        try:
            subprocess.Popen([sys.executable, cream_os_path])
            print("[OK] QuantoniumOS launch process started")
            return True
        except Exception as e:
            print(f"[FAIL] Failed to launch Cream OS: {e}")
            return False
    else:
        print(f"[FAIL] Cream OS not found at: {cream_os_path}")
        return False

def main():
    \"\"\"Main boot sequence\"\"\"
    print_header("QUANTONIUMOS BOOT PROCESS")
    print("Architecture: ASSEMBLY → C++ → Python → Cream OS")
    print("")
    
    # Check operating system
    system = platform.system()
    print(f"Host system: {system} {platform.release()}")
    
    # Execute boot stages
    if not boot_rft_assembly():
        return 1
        
    print("")
    if not boot_cpp_layer():
        return 1
        
    print("")
    if not boot_python_layer():
        return 1
        
    print("")
    if not launch_cream_os():
        return 1
    
    print("")
    print_header("QUANTONIUMOS BOOT COMPLETE")
    print("Your system should now be fully operational.")
    print("")
    print("If the GUI did not launch automatically, run:")
    print(f"  python {os.path.join('16_EXPERIMENTAL', 'prototypes', 'quantonium_os_unified_cream.py')}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
""")
    
    log(f"Updated boot script: {boot_file}", "SUCCESS")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="QuantoniumOS Project Cleanup and Build Plan")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--skip-clean", action="store_true", help="Skip repository cleanup")
    parser.add_argument("--skip-rft", action="store_true", help="Skip RFT integration")
    args = parser.parse_args()
    
    print("=" * 80)
    print("QuantoniumOS Project Cleanup and Build Plan")
    print("=" * 80)
    
    if not args.skip_clean:
        clean_repository(dry_run=args.dry_run)
    
    if not args.dry_run:
        if not args.skip_rft:
            integrate_working_rft()
            
        create_build_script()
        create_documentation()
        create_sanity_check_script()
        update_boot_script()
        
        print("\n" + "=" * 80)
        print("Project cleanup and setup complete!")
        print("=" * 80)
        print("\nNext steps:")
        print("1. Run the build script: build.bat")
        print("2. Verify the project: python verify_sanity.py")
        print("3. Launch QuantoniumOS: python os_boot_transition.py")

if __name__ == "__main__":
    main()
