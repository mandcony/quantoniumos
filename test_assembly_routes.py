#!/usr/bin/env python3
"""
ASSEMBLY ROUTES DIAGNOSTIC - CHECK IF RFT BINDINGS ARE BROKEN
"""

import sys
import os

print("🔧 ASSEMBLY ROUTES DIAGNOSTIC")
print("=" * 50)

# Add the path to the assembly bindings
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'assembly', 'python_bindings'))

print("1. Testing basic imports...")
try:
    import numpy as np
    print("✓ numpy import: OK")
except Exception as e:
    print(f"✗ numpy import: FAILED - {e}")

print("\n2. Testing RFT binding imports...")
try:
    from unitary_rft import UnitaryRFT, RFT_FLAG_QUANTUM_SAFE, RFT_FLAG_USE_RESONANCE
    print("✓ RFT imports: OK")
except Exception as e:
    print(f"✗ RFT imports: FAILED - {e}")
    print("ASSEMBLY ROUTES ARE BROKEN!")
    sys.exit(1)

print("\n3. Testing UnitaryRFT instantiation...")
try:
    rft_engine = UnitaryRFT(32, RFT_FLAG_QUANTUM_SAFE | RFT_FLAG_USE_RESONANCE)
    print("✓ UnitaryRFT creation: OK")
except Exception as e:
    print(f"✗ UnitaryRFT creation: FAILED - {e}")
    print("RFT ENGINE INSTANTIATION BROKEN!")

print("\n4. Testing RFT methods availability...")
try:
    print(f"✓ RFT engine object: {rft_engine}")
    print(f"✓ RFT size: {rft_engine.size}")
    
    # Check if forward method exists and is callable
    if hasattr(rft_engine, 'forward'):
        print(f"✓ forward method exists: {rft_engine.forward}")
        if callable(rft_engine.forward):
            print("✓ forward method is callable")
        else:
            print("✗ forward method is NOT callable!")
            print(f"Type: {type(rft_engine.forward)}")
    else:
        print("✗ forward method MISSING!")
        
except Exception as e:
    print(f"✗ RFT method check: FAILED - {e}")

print("\n5. Testing RFT library loading...")
try:
    if hasattr(rft_engine, 'lib'):
        print(f"✓ RFT lib loaded: {rft_engine.lib}")
        if hasattr(rft_engine.lib, 'rft_forward'):
            print(f"✓ rft_forward function: {rft_engine.lib.rft_forward}")
        else:
            print("✗ rft_forward function MISSING from library!")
    else:
        print("✗ RFT lib NOT loaded!")
except Exception as e:
    print(f"✗ RFT library check: FAILED - {e}")

print("\n6. Testing actual forward transform...")
try:
    test_state = np.zeros(32, dtype=complex)
    test_state[0] = 1.0
    
    result = rft_engine.forward(test_state)
    print(f"✓ Forward transform: SUCCESS - {len(result)} elements")
    print(f"  Result type: {type(result)}")
    print(f"  First few values: {result[:3]}")
except Exception as e:
    print(f"✗ Forward transform: FAILED - {e}")
    print("ASSEMBLY ROUTES COMPLETELY BROKEN!")

print("\n7. Checking compiled libraries...")
script_dir = os.path.dirname(os.path.abspath(__file__))
lib_paths = [
    os.path.join(script_dir, "src", "assembly", "compiled", "libquantum_symbolic.dll"),
    os.path.join(script_dir, "src", "assembly", "compiled", "rftkernel.dll"),
    os.path.join(script_dir, "src", "assembly", "compiled", "librftkernel.dll"),
]

for lib_path in lib_paths:
    if os.path.exists(lib_path):
        print(f"✓ Found library: {lib_path}")
    else:
        print(f"✗ Missing library: {lib_path}")

print("\nDIAGNOSTIC COMPLETE")
