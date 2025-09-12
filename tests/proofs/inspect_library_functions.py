#!/usr/bin/env python3
"""
LIBRARY FUNCTION INSPECTOR
=========================
Check what functions are actually available in libquantum_symbolic.dll
"""

import os
import ctypes
from ctypes import cdll
import subprocess

# Load the library
script_dir = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(script_dir, "..", "..", "src", "assembly", "compiled", "libquantum_symbolic.dll")

print(f"Inspecting library: {lib_path}")
print(f"Library exists: {os.path.exists(lib_path)}")

if os.path.exists(lib_path):
    try:
        # Load the library
        lib = cdll.LoadLibrary(lib_path)
        print("✅ Library loaded successfully!")
        
        # Try to use objdump or dumpbin to list exported functions
        try:
            result = subprocess.run(['dumpbin', '/exports', lib_path], 
                                  capture_output=True, text=True, shell=True)
            if result.returncode == 0:
                print("\nExported functions from dumpbin:")
                print(result.stdout)
            else:
                print("dumpbin not available")
        except:
            print("Could not run dumpbin")
        
        # Try common function names that might be in your library
        common_names = [
            'rft_init', 'rft_forward', 'rft_inverse', 'rft_cleanup',
            'quantum_init', 'quantum_forward', 'quantum_inverse', 'quantum_cleanup',
            'symbolic_init', 'symbolic_forward', 'symbolic_inverse', 'symbolic_cleanup',
            'transform_init', 'transform_forward', 'transform_inverse', 'transform_cleanup',
            'unitary_init', 'unitary_forward', 'unitary_inverse', 'unitary_cleanup',
            'rft_create', 'rft_transform', 'rft_destroy',
            'create_rft', 'apply_rft', 'destroy_rft',
            'init', 'forward', 'inverse', 'cleanup'
        ]
        
        available_functions = []
        for name in common_names:
            try:
                func = getattr(lib, name)
                available_functions.append(name)
                print(f"✅ Found function: {name}")
            except AttributeError:
                pass
        
        print(f"\nTotal functions found: {len(available_functions)}")
        if available_functions:
            print("Available functions:", available_functions)
        else:
            print("❌ No standard RFT functions found")
            
        # Try to access lib attributes directly
        print(f"\nLibrary object: {lib}")
        print(f"Library handle: {lib._handle}")
        
    except Exception as e:
        print(f"❌ Library inspection failed: {e}")
else:
    print("❌ Library not found")
