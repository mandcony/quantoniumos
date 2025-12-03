#!/usr/bin/env python3
"""
Test the C/ASM RFT implementation safely.
"""
import sys
import os
import numpy as np
import platform
import pytest

# Prefer native Windows build locations for the assembly DLL when available
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))

if platform.system().lower() == "windows":
    candidates = [
        os.path.join(REPO_ROOT, "src", "assembly", "compiled", "libquantum_symbolic.dll"),
        os.path.join(REPO_ROOT, "src", "assembly", "compiled", "rftkernel.dll"),
        os.path.join(REPO_ROOT, "src", "assembly", "compiled", "librftkernel.dll"),
        os.path.join(REPO_ROOT, "system", "assembly", "assembly", "compiled", "libquantum_symbolic.dll"),
    ]
    for p in candidates:
        if os.path.exists(p):
            os.environ.setdefault("RFT_KERNEL_LIB", p)
            break

def _roundtrip_ok(N: int) -> float:
    from unitary_rft import UnitaryRFT, RFT_FLAG_QUANTUM_SAFE
    rft = UnitaryRFT(N, RFT_FLAG_QUANTUM_SAFE)
    if getattr(rft, "_is_mock", True):
        pytest.skip("Assembly DLL not available; using mock implementation")
    x = np.zeros(N, dtype=np.complex128)
    x[0] = 1.0
    y = rft.forward(x)
    z = rft.inverse(y)
    # Norm preservation
    in_norm = float(np.linalg.norm(x))
    out_norm = float(np.linalg.norm(y))
    assert abs(in_norm - out_norm) < 1e-12, f"Norm mismatch: {in_norm} vs {out_norm}"
    # Reconstruction error
    err = float(np.max(np.abs(x - z)))
    assert err < 1e-10, f"Reconstruction error too high: {err}"
    return err


def test_c_rft():
    """Test the C RFT implementation with safety checks."""
    try:
        from unitary_rft import UnitaryRFT, RFT_FLAG_QUANTUM_SAFE
    except ImportError as e:
        pytest.skip(f"Native RFT library not available: {e}")
    
    try:
        print("✅ UnitaryRFT imported successfully")
        
        # Test with a very small size first
        print("Testing with size 4...")
        rft = UnitaryRFT(4, RFT_FLAG_QUANTUM_SAFE)
        print(f"Mock mode: {rft._is_mock}")
        
        if rft._is_mock:
            print("❌ Using mock implementation - C library not working")
            pytest.skip("Assembly DLL not available; using mock implementation")
        # N=4 baseline
        err4 = _roundtrip_ok(4)
        print(f"N=4 ok, err={err4}")
        # N=16
        err16 = _roundtrip_ok(16)
        print(f"N=16 ok, err={err16}")
        # N=64
        err64 = _roundtrip_ok(64)
        print(f"N=64 ok, err={err64}")
        # All passed
        assert True

    except Exception as e:
        print(f"❌ C/ASM RFT test failed: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"C/ASM RFT test failed: {e}")

if __name__ == "__main__":
    success = test_c_rft()
    if success:
        print("\n🎉 C/ASM RFT kernels are ready for quantum entanglement!")
    else:
        print("\n💥 C/ASM RFT kernels need debugging")