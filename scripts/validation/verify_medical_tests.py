#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""Quick verification that medical tests can be imported and run."""

import sys
import traceback
import argparse

def verify_imports():
    """Verify all medical test modules can be imported."""
    print("Verifying medical test imports...")
    
    try:
        from tests.medical import test_imaging_reconstruction
        print("  ✓ test_imaging_reconstruction")
    except Exception as e:
        print(f"  ✗ test_imaging_reconstruction: {e}")
        return False
        
    try:
        from tests.medical import test_biosignal_compression
        print("  ✓ test_biosignal_compression")
    except Exception as e:
        print(f"  ✗ test_biosignal_compression: {e}")
        return False
        
    try:
        from tests.medical import test_genomics_transforms
        print("  ✓ test_genomics_transforms")
    except Exception as e:
        print(f"  ✗ test_genomics_transforms: {e}")
        return False
        
    try:
        from tests.medical import test_medical_security
        print("  ✓ test_medical_security")
    except Exception as e:
        print(f"  ✗ test_medical_security: {e}")
        return False
        
    try:
        from tests.medical import test_edge_wearable
        print("  ✓ test_edge_wearable")
    except Exception as e:
        print(f"  ✗ test_edge_wearable: {e}")
        return False
        
    try:
        from tests.medical import run_medical_benchmarks
        print("  ✓ run_medical_benchmarks")
    except Exception as e:
        print(f"  ✗ run_medical_benchmarks: {e}")
        return False
    
    return True

def verify_key_functions():
    """Verify key functions exist and are correctly named."""
    print("\nVerifying key functions...")
    
    from tests.medical.test_medical_security import check_collision_resistance
    print(f"  ✓ check_collision_resistance function exists")
    
    from tests.medical.test_medical_security import compute_avalanche_effect
    print(f"  ✓ compute_avalanche_effect function exists")
    
    from tests.medical.test_imaging_reconstruction import rft_denoise_2d
    print(f"  ✓ rft_denoise_2d function exists")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Quick verification that medical tests can be imported and run'
    )
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Increase output verbosity')
    args = parser.parse_args()
    
    success = True
    
    if not verify_imports():
        success = False
        
    if success and not verify_key_functions():
        success = False
    
    if success:
        print("\n✓ All medical test modules verified successfully!")
        sys.exit(0)
    else:
        print("\n✗ Some verifications failed")
        sys.exit(1)
