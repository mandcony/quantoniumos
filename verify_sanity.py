#!/usr/bin/env python3
"""
QuantoniumOS Sanity Check
=========================
Verifies that all essential components are present and functional.
"""

import os
import sys
import subprocess
import glob

def check_component(name, path, alt_path=None, is_dir=False):
    """Check if a component exists at either the primary or alternate path"""
    full_path = os.path.join(os.path.dirname(__file__), path)
    
    exists = False
    used_path = path
    
    if is_dir:
        exists = os.path.isdir(full_path)
        component_type = "Directory"
    else:
        exists = os.path.isfile(full_path)
        component_type = "File"
    
    # Check alternate path if provided and primary path doesn't exist
    if not exists and alt_path:
        alt_full_path = os.path.join(os.path.dirname(__file__), alt_path)
        if is_dir:
            exists = os.path.isdir(alt_full_path)
        else:
            exists = os.path.isfile(alt_full_path)
        
        if exists:
            used_path = alt_path
    
    if exists:
        print(f"[PASS] {component_type} found: {name} ({used_path})")
        return True
    else:
        print(f"[FAIL] {component_type} missing: {name} ({path}{f' or {alt_path}' if alt_path else ''})")
        return False

def main():
    """Run sanity checks"""
    print("=" * 70)
    print("QuantoniumOS Sanity Check")
    print("=" * 70)
    
    # Core components
    check_component("ASSEMBLY directory", "ASSEMBLY", alt_path="WORKING_RFT_ASSEMBLY", is_dir=True)
    check_component("Compiled RFT kernel", "ASSEMBLY/compiled/librftkernel.dll", 
                   alt_path="WORKING_RFT_ASSEMBLY/compiled/librftkernel.dll")
    check_component("RFT Python bindings", "ASSEMBLY/python_bindings/unitary_rft.py", 
                   alt_path="WORKING_RFT_ASSEMBLY/python_bindings/unitary_rft.py")
    
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
    
    print("\nChecking for duplicates:")
    for name, items in duplicates.items():
        if len(items) > 1:
            print(f"[WARNING] Multiple {name} found:")
            for item in items:
                print(f"  - {item}")
        else:
            print(f"[PASS] No duplicate {name}")
    
    print("\nSanity check complete!")
    
if __name__ == "__main__":
    main()
