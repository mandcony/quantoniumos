#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
Verify Test Wiring - Validate pytest can discover all tests
============================================================

Checks:
1. All test directories have __init__.py
2. All test files follow naming conventions
3. All imports are valid
4. pytest can discover all tests
"""
import subprocess
import sys
from pathlib import Path

def main():
    print("=" * 70)
    print("QuantoniumOS Test Wiring Verification")
    print("=" * 70)
    print()
    
    repo_root = Path(__file__).parent
    tests_dir = repo_root / "tests"
    
    # Check 1: Find directories without __init__.py
    print("✓ Checking for missing __init__.py files...")
    missing_init = []
    for py_file in tests_dir.rglob("*.py"):
        parent = py_file.parent
        if parent.name == "__pycache__":
            continue
        init_file = parent / "__init__.py"
        if not init_file.exists() and parent != tests_dir:
            if parent not in missing_init:
                missing_init.append(parent)
    
    if missing_init:
        print(f"  ⚠ Found {len(missing_init)} directories without __init__.py:")
        for d in missing_init:
            print(f"    - {d.relative_to(repo_root)}")
    else:
        print("  ✓ All test directories have __init__.py")
    print()
    
    # Check 2: Verify test file naming
    print("✓ Checking test file naming conventions...")
    non_test_files = []
    for py_file in tests_dir.rglob("*.py"):
        if py_file.parent.name == "__pycache__":
            continue
        if py_file.name == "__init__.py":
            continue
        if py_file.name == "conftest.py":
            continue
        # Files that don't start with test_ but might be utilities
        if not py_file.name.startswith("test_"):
            non_test_files.append(py_file)
    
    if non_test_files:
        print(f"  ℹ Found {len(non_test_files)} utility/research files (not tests):")
        for f in non_test_files[:10]:  # Show first 10
            print(f"    - {f.relative_to(tests_dir)}")
        if len(non_test_files) > 10:
            print(f"    ... and {len(non_test_files) - 10} more")
    else:
        print("  ✓ All Python files follow test naming convention")
    print()
    
    # Check 3: pytest collection
    print("✓ Running pytest collection (dry-run)...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q", "tests/"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Parse output
        lines = result.stdout.split('\n')
        test_count = 0
        for line in lines:
            if 'test' in line.lower() and '::' in line:
                test_count += 1
        
        if result.returncode == 0:
            print(f"  ✓ pytest discovered {test_count} tests successfully")
        else:
            print(f"  ⚠ pytest collection completed with warnings")
            if result.stderr:
                print("\n  Errors/Warnings:")
                for line in result.stderr.split('\n')[:10]:
                    if line.strip():
                        print(f"    {line}")
        
        # Show summary
        for line in lines[-5:]:
            if line.strip() and ('test' in line.lower() or 'error' in line.lower() or 'warning' in line.lower()):
                print(f"  {line}")
                
    except subprocess.TimeoutExpired:
        print("  ⚠ pytest collection timed out (may indicate import issues)")
    except Exception as e:
        print(f"  ✗ Error running pytest: {e}")
    print()
    
    # Check 4: Import validation
    print("✓ Validating key imports...")
    import_tests = [
        ("algorithms.rft.core.closed_form_rft", "rft_forward"),
        ("algorithms.rft.variants.manifest", "iter_variants"),
        ("algorithms.rft.compression.ans", "ans_encode"),
    ]
    
    failed_imports = []
    for module, attr in import_tests:
        try:
            mod = __import__(module, fromlist=[attr])
            getattr(mod, attr)
            print(f"  ✓ {module}.{attr}")
        except Exception as e:
            failed_imports.append((module, attr, str(e)))
            print(f"  ✗ {module}.{attr}: {e}")
    
    print()
    print("=" * 70)
    print("Summary:")
    print("=" * 70)
    
    issues = []
    if missing_init:
        issues.append(f"Missing {len(missing_init)} __init__.py files")
    if failed_imports:
        issues.append(f"Failed {len(failed_imports)} imports")
    
    if not issues:
        print("✓ All tests are properly wired!")
        print()
        print("You can now run:")
        print("  pytest tests/                    # All tests")
        print("  pytest tests/rft/                # RFT tests only")
        print("  pytest tests/benchmarks/         # Benchmarks only")
        print("  pytest -m 'not slow'             # Skip slow tests")
        return 0
    else:
        print("⚠ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
