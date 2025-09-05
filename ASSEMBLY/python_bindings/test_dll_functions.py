#!/usr/bin/env python3
"""
Test if new functions are available in the compiled DLL
"""

import ctypes
import os

def test_dll_functions():
    """Test if the new mathematical functions are available."""
    print("🔍 Testing DLL Function Availability")
    print("=" * 40)
    
    dll_path = "../compiled/rftkernel.dll"
    if not os.path.exists(dll_path):
        print(f"❌ DLL not found: {dll_path}")
        return False
    
    try:
        lib = ctypes.cdll.LoadLibrary(dll_path)
        print(f"✅ DLL loaded: {dll_path}")
        
        # Test basic functions
        basic_functions = [
            'rft_init',
            'rft_forward', 
            'rft_inverse',
            'rft_cleanup'
        ]
        
        print(f"\n📋 Basic Functions:")
        for func_name in basic_functions:
            try:
                func = getattr(lib, func_name)
                print(f"   ✅ {func_name}: Available")
            except AttributeError:
                print(f"   ❌ {func_name}: Missing")
        
        # Test new mathematical functions
        new_functions = [
            'rft_von_neumann_entropy',
            'rft_entanglement_measure',
            'rft_validate_bell_state',
            'rft_validate_golden_ratio_properties'
        ]
        
        print(f"\n🧮 New Mathematical Functions:")
        missing_functions = []
        for func_name in new_functions:
            try:
                func = getattr(lib, func_name)
                print(f"   ✅ {func_name}: Available")
            except AttributeError:
                print(f"   ❌ {func_name}: Missing")
                missing_functions.append(func_name)
        
        if missing_functions:
            print(f"\n⚠️  Missing Functions: {len(missing_functions)}")
            print(f"   This means the DLL wasn't rebuilt with the new code.")
            return False
        else:
            print(f"\n🎉 All functions available!")
            return True
            
    except Exception as e:
        print(f"❌ Error loading DLL: {e}")
        return False

if __name__ == "__main__":
    test_dll_functions()
