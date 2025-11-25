#!/usr/bin/env python3
"""Comprehensive import validation for all QuantoniumOS paths."""

import sys
import traceback

def test_import(module_path, description):
    """Test a single import and report results."""
    try:
        __import__(module_path)
        print(f"✅ {description}: {module_path}")
        return True
    except ImportError as e:
        print(f"❌ {description}: {module_path}")
        print(f"   Error: {e}")
        return False
    except Exception as e:
        print(f"⚠️  {description}: {module_path}")
        print(f"   Unexpected error: {e}")
        return False

def main():
    """Run all import tests."""
    print("="*70)
    print("QuantoniumOS Import Path Validation")
    print("="*70)
    
    tests = [
        # Core RFT modules
        ("algorithms.rft.core.canonical_true_rft", "Canonical RFT"),
        ("algorithms.rft.core.closed_form_rft", "Closed-Form RFT"),
        
        # Crypto modules
        ("algorithms.rft.crypto.enhanced_cipher", "Enhanced Cipher"),
        ("algorithms.rft.crypto.primitives.quantum_prng", "Quantum PRNG"),
        
        # Compression modules
        ("algorithms.rft.compression.rft_vertex_codec", "Vertex Codec"),
        ("algorithms.rft.compression.entropy", "Entropy Module"),
        ("algorithms.rft.compression.lossless.rans_stream", "RANS Stream"),
        
        # Hybrid modules
        ("algorithms.rft.hybrids.rft_hybrid_codec", "Hybrid Codec"),
        ("algorithms.rft.hybrids.legacy_mca", "Legacy MCA"),
        
        # Quantum modules
        ("algorithms.rft.quantum.geometric_waveform_hash", "Geometric Waveform Hash"),
        ("algorithms.rft.quantum.quantum_gates", "Quantum Gates"),
        
        # Variants and theorems
        ("algorithms.rft.variants.registry", "Variants Registry"),
        
        # Kernel bindings
        ("algorithms.rft.kernels.python_bindings.unitary_rft", "Unitary RFT Bindings"),
        
        # Hybrid basis (shim module)
        ("algorithms.rft.hybrid_basis", "Hybrid Basis Shim"),
        
        # Top-level package
        ("quantoniumos", "QuantoniumOS Package"),
    ]
    
    print("\nTesting imports...\n")
    
    passed = 0
    failed = 0
    
    for module_path, description in tests:
        if test_import(module_path, description):
            passed += 1
        else:
            failed += 1
        print()
    
    print("="*70)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*70)
    
    if failed > 0:
        print("\n⚠️  Some imports failed. Check the error messages above.")
        sys.exit(1)
    else:
        print("\n✅ All imports successful! Repository is correctly configured.")
        sys.exit(0)

if __name__ == "__main__":
    main()
