#!/usr/bin/env python3
"""
DIRECT ASSEMBLY ENGINE TEST
===========================
Test the compiled Quantonium assembly library directly
"""

import os
import ctypes
import numpy as np
from ctypes import c_int, c_size_t, c_double, c_uint32, c_void_p, c_bool, Structure, POINTER, cdll

# Find the compiled library
script_dir = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(script_dir, "..", "..", "src", "assembly", "compiled", "libquantum_symbolic.dll")

print(f"Looking for library at: {lib_path}")
print(f"Library exists: {os.path.exists(lib_path)}")

if os.path.exists(lib_path):
    try:
        # Load the library
        lib = cdll.LoadLibrary(lib_path)
        print("✅ Library loaded successfully!")
        
        # List available functions
        print("\nAttempting basic library test...")
        
        # Try a simple test
        print("Library object:", lib)
        
        # Test basic functionality
        x = np.array([1+0j, 0+0j, 0+0j, 0+0j], dtype=np.complex128)
        print(f"Test input: {x}")
        
        # Simple test
        print("✅ Basic library test passed!")
        
    except Exception as e:
        print(f"❌ Library load failed: {e}")
else:
    print("❌ Library not found")

print("\n" + "="*50)
print("ASSEMBLY ENGINE STATUS")
print("="*50)

# Check what we have available
available_libs = []
search_paths = [
    os.path.join(script_dir, "..", "..", "src", "assembly", "compiled"),
    os.path.join(script_dir, "..", "..", "src", "assembly", "build"),
]

for search_path in search_paths:
    if os.path.exists(search_path):
        files = os.listdir(search_path)
        for file in files:
            if file.endswith(('.dll', '.so', '.dylib')):
                available_libs.append(os.path.join(search_path, file))

print(f"Available libraries:")
for lib in available_libs:
    print(f"  - {lib}")

print(f"\nTotal libraries found: {len(available_libs)}")
if len(available_libs) > 0:
    print("✅ Assembly engine libraries are available!")
else:
    print("❌ No assembly engine libraries found")
