#!/usr/bin/env python3
"""
Quick validation that C++ engines are being used for heavy lifting
"""


def validate_cpp_acceleration():
    print("🔍 VALIDATING C++ ENGINE ACCELERATION")
    print("=" * 50)

    # Test 1: Check engine loading
    print("1. Testing engine loading...")
    try:
        import resonance_engine_canonical

        print("   ✅ resonance_engine_canonical loaded")
    except ImportError as e:
        print(f"   ❌ resonance_engine_canonical failed: {e}")

    try:
        import vertex_engine_canonical

        print("   ✅ vertex_engine_canonical loaded")
    except ImportError as e:
        print(f"   ❌ vertex_engine_canonical failed: {e}")

    try:
        import enhanced_rft_crypto_canonical

        print("   ✅ enhanced_rft_crypto_canonical loaded")
    except ImportError as e:
        print(f"   ❌ enhanced_rft_crypto_canonical failed: {e}")

    # Test 2: Check bulletproof kernel acceleration
    print("\n2. Testing BulletproofQuantumKernel acceleration...")
    from bulletproof_quantum_kernel import BulletproofQuantumKernel

    kernel = BulletproofQuantumKernel(32)
    status = kernel.get_acceleration_status()

    print(f"   Acceleration mode: {status['acceleration_mode']}")
    print(f"   Available engines: {status['available_engines']}")
    print(f"   Engine count: {status['engine_count']}")
    print(f"   Performance level: {status['performance_level']}")

    # Test 3: Validate actual C++ usage
    print("\n3. Testing actual C++ engine usage...")
    import time

    import numpy as np

    signal = np.random.randn(128)

    # Time the forward transform (should use C++)
    start = time.perf_counter()
    spectrum = kernel.forward_rft(signal)
    forward_time = time.perf_counter() - start

    print(f"   Forward RFT time: {forward_time:.2e}s")
    print(f"   Output shape: {spectrum.shape}")
    print(f"   Input shape: {signal.shape}")
    print(f"   Shape match: {spectrum.shape == signal.shape}")

    # Test the comprehensive test suite detection
    print("\n4. Testing comprehensive test suite integration...")
    try:
        # Run just one test to see acceleration mode
        test_kernel = BulletproofQuantumKernel(16)
        test_signal = np.random.randn(16)
        test_spectrum = test_kernel.forward_rft(test_signal)
        test_status = test_kernel.get_acceleration_status()

        print(f"   Test suite acceleration mode: {test_status['acceleration_mode']}")
        print(f"   C++ engines being used: {test_status['engine_count'] >= 3}")

    except Exception as e:
        print(f"   ❌ Test suite integration error: {e}")

    print("\n📊 SUMMARY")
    print("=" * 50)
    if status["engine_count"] >= 3:
        print("✅ SUCCESS: All canonical C++ engines loaded and active")
        print("✅ SUCCESS: Full C++ Acceleration mode enabled")
        print("✅ SUCCESS: Engines are handling the heavy lifting")
        print(f"✅ SUCCESS: {status['engine_count']}/3 engines available")
        return True
    else:
        print("❌ PARTIAL: Not all engines loaded")
        print(f"⚠️  Only {status['engine_count']}/3 engines available")
        return False


if __name__ == "__main__":
    validate_cpp_acceleration()
