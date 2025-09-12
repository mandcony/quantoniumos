#!/usr/bin/env python3
"""
DEEP ASSEMBLY LIBRARY DIAGNOSTIC
"""

import sys
import os
import ctypes

print("🔧 DEEP ASSEMBLY LIBRARY DIAGNOSTIC")
print("=" * 50)

# Add path to assembly bindings
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'assembly', 'python_bindings'))

# Check the library loading process step by step
script_dir = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(script_dir, "src", "assembly", "compiled", "libquantum_symbolic.dll")

print(f"1. Checking library path: {lib_path}")
if os.path.exists(lib_path):
    print("✓ Library file exists")
else:
    print("✗ Library file NOT found")
    exit(1)

print("\n2. Loading library...")
try:
    lib = ctypes.cdll.LoadLibrary(lib_path)
    print(f"✓ Library loaded: {lib}")
except Exception as e:
    print(f"✗ Library loading failed: {e}")
    exit(1)

print("\n3. Checking available functions...")
# Try to list all available functions
print("Available functions in library:")
try:
    # Use dir() to see what's available
    print(f"Library attributes: {dir(lib)}")
except:
    pass

print("\n4. Testing specific function access...")
functions_to_test = [
    "rft_init",
    "rft_cleanup", 
    "rft_forward",
    "rft_inverse",
    "rft_quantum_basis"
]

for func_name in functions_to_test:
    try:
        func = getattr(lib, func_name)
        print(f"✓ {func_name}: {func} (type: {type(func)})")
    except AttributeError:
        print(f"✗ {func_name}: NOT FOUND")
    except Exception as e:
        print(f"✗ {func_name}: ERROR - {e}")

print("\n5. Testing function call (rft_init)...")
try:
    # Import the structures we need
    from unitary_rft import RFTEngine, c_size_t, c_uint32, POINTER, c_int
    
    # Try to set up rft_init signature
    lib.rft_init.argtypes = [POINTER(RFTEngine), c_size_t, c_uint32]
    lib.rft_init.restype = c_int
    print("✓ rft_init signature set successfully")
    
    # Try to call rft_init
    engine = RFTEngine()
    result = lib.rft_init(engine, 32, 0x0000000C)  # RFT_FLAG_QUANTUM_SAFE | RFT_FLAG_UNITARY
    print(f"✓ rft_init call result: {result}")
    
except Exception as e:
    print(f"✗ rft_init test failed: {e}")
    import traceback
    traceback.print_exc()

print("\nDIAGNOSTIC COMPLETE")
