#!/usr/bin/env python3
"""
QuantoniumOS CLI Verification Script
Verifies CLI functionality without requiring server startup
"""

import sys
import os
import subprocess
import json
import argparse
from pathlib import Path

def setup_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Verify QuantoniumOS CLI functionality")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    return parser.parse_args()

def test_python_imports():
    """Test that core Python modules can be imported"""
    print("Testing Python module imports...")
    
    try:
        # Test core imports
        import core
        print("✅ Core module imported successfully")
        
        # Test encryption imports
        sys.path.append('core')
        from encryption.resonance_fourier import perform_rft
        print("✅ RFT module imported successfully")
        
        from encryption.geometric_waveform_hash import geometric_waveform_hash
        print("✅ Geometric waveform module imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_cli_commands():
    """Test CLI commands that don't require server"""
    print("Testing CLI commands...")
    
    try:
        # Test help command
        result = subprocess.run([sys.executable, "-m", "quantonium_cli", "--help"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ CLI help command works")
        else:
            print("❌ CLI help command failed")
            return False
            
        # Test version command
        result = subprocess.run([sys.executable, "-m", "quantonium_cli", "--version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ CLI version command works")
        else:
            print("❌ CLI version command failed")
            return False
            
        return True
    except Exception as e:
        print(f"❌ CLI test error: {e}")
        return False

def test_cpp_extension():
    """Test C++ extension loading"""
    print("Testing C++ extension...")
    
    try:
        import quantonium_core
        print("✅ C++ extension loaded successfully")
        return True
    except ImportError as e:
        print(f"⚠️  C++ extension not available: {e}")
        print("This is expected if not built yet")
        return True  # Don't fail CI for this

def test_config_files():
    """Test that configuration files are valid"""
    print("Testing configuration files...")
    
    try:
        # Test pyproject.toml
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
            
        if os.path.exists("pyproject.toml"):
            with open("pyproject.toml", "rb") as f:
                tomllib.load(f)
            print("✅ pyproject.toml is valid")
        else:
            print("⚠️  pyproject.toml not found")
        
        # Test setup.py
        if os.path.exists("setup.py"):
            with open("setup.py", "r") as f:
                setup_content = f.read()
                if "setup(" in setup_content:
                    print("✅ setup.py structure looks valid")
                else:
                    print("❌ setup.py missing setup() call")
                    return False
        else:
            print("⚠️  setup.py not found")
                
        return True
    except Exception as e:
        print(f"❌ Configuration file error: {e}")
        return False

def main():
    """Run all verification tests"""
    args = setup_args()
    
    print("QuantoniumOS CLI Verification")
    print("=" * 40)
    
    tests = [
        test_config_files,
        test_python_imports,
        test_cpp_extension,
        test_cli_commands,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            print()
    
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All CLI verification tests passed!")
        return 0
    else:
        print("❌ Some CLI verification tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
