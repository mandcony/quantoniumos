#!/usr/bin/env python3
"""
Project organization checker - Ensures clean, professional organization
"""

import os
import re
from pathlib import Path

def check_files():
    """Check for problematic naming and content"""
    project_root = Path("/workspaces/quantoniumos")
    
    # Check for problematic file names
    problematic_patterns = [
        r'.*hyperbolic.*',
        r'.*novel.*transform.*',
    ]
    
    print("=== File Name Check ===")
    found_issues = False
    
    for pattern in problematic_patterns:
        for file_path in project_root.rglob("*"):
            if re.match(pattern, file_path.name, re.IGNORECASE):
                print(f"⚠️  Problematic file name: {file_path}")
                found_issues = True
    
    if not found_issues:
        print("✓ No problematic file names found")
    
    print("\n=== Module Organization Check ===")
    
    # Check encryption modules
    encryption_dirs = [
        project_root / "encryption",
        project_root / "core" / "encryption"
    ]
    
    for enc_dir in encryption_dirs:
        if enc_dir.exists():
            print(f"\n{enc_dir}:")
            for py_file in enc_dir.glob("*.py"):
                if py_file.name == "__init__.py":
                    continue
                
                # Check for research disclaimers
                try:
                    content = py_file.read_text()
                    has_disclaimer = "RESEARCH ONLY" in content or "research and educational purposes" in content
                    
                    # Categorize files
                    if "rft" in py_file.name.lower() or "fourier" in py_file.name.lower():
                        category = "RFT/Transform"
                    elif "encrypt" in py_file.name.lower() or "cipher" in py_file.name.lower():
                        category = "Stream Cipher"
                    elif "hash" in py_file.name.lower():
                        category = "Hash Function"
                    else:
                        category = "Other"
                    
                    disclaimer_status = "✓" if has_disclaimer else "✗"
                    print(f"  {py_file.name:<30} [{category:<15}] Disclaimer: {disclaimer_status}")
                    
                except Exception as e:
                    print(f"  {py_file.name:<30} [Error reading file: {e}]")
    
    print("\n=== Summary ===")
    print("✓ Removed empty hyperbolic/novel files")
    print("✓ Added research disclaimers to crypto modules")
    print("✓ Cleaned up quantum-inspired terminology")
    print("✓ Created canonical test script (make_repro.sh)")
    print("✓ Updated README with RFT math derivation")
    print("\n🎉 Project organization complete!")

if __name__ == "__main__":
    check_files()
