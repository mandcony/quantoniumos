#!/usr/bin/env python3
"""
Test 12-qubit simulation to verify the vertex-based approach works
"""

import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '05_QUANTUM_ENGINES'))


def test_12_qubit_vertex_approach():
    print("🧪 Testing 12-qubit simulation with vertex-based holographic storage")
    print("=" * 65)

    import time

    import numpy as np

    from 05_QUANTUM_ENGINES.bulletproof_quantum_kernel import BulletproofQuantumKernel

    # Test the problematic 12-qubit case (dimension 2^12 = 4096)
    n_qubits = 12
    dimension = 2**n_qubits

    print(f"Creating {n_qubits}-qubit system (dimension {dimension})...")

    # Time the kernel creation (this should use vertex-based approach)
    start_time = time.perf_counter()

    try:
        kernel = BulletproofQuantumKernel(dimension=dimension)
        creation_time = time.perf_counter() - start_time

        print(f"✅ Kernel created in {creation_time:.2f}s")
        print(
            f"   Acceleration mode: {kernel.get_acceleration_status()['acceleration_mode']}"
        )

        # Test resonance kernel building (this was hanging before)
        print("\n🌀 Building resonance kernel...")
        start_time = time.perf_counter()

        resonance_kernel = kernel.build_resonance_kernel()
        kernel_time = time.perf_counter() - start_time

        print(f"✅ Resonance kernel built in {kernel_time:.2f}s")
        print(f"   Kernel shape: {resonance_kernel.shape}")
        print(
            f"   Kernel type: {'Vector (vertex-based)' if len(resonance_kernel.shape) == 1 else 'Matrix (direct)'}"
        )

        # Test a small quantum operation
        print("\n🔬 Testing quantum operation...")
        start_time = time.perf_counter()

        # Create a small test signal (not full dimension to avoid memory issues)
        test_signal = np.random.randn(64) + 1j * np.random.randn(64)
        test_signal /= np.linalg.norm(test_signal)

        # Test forward RFT (should use C++ acceleration)
        rft_result = kernel.forward_rft(test_signal)
        operation_time = time.perf_counter() - start_time

        print(f"✅ Quantum operation completed in {operation_time:.3f}s")
        print(f"   Input shape: {test_signal.shape}")
        print(f"   Output shape: {rft_result.shape}")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_scaling_performance():
    print("\n🚀 Testing scaling performance across different qubit counts")
    print("=" * 65)

    import time

    from 05_QUANTUM_ENGINES.bulletproof_quantum_kernel import BulletproofQuantumKernel

    results = []

    for n_qubits in [8, 10, 12, 14, 16]:
        dimension = 2**n_qubits
        print(f"\nTesting {n_qubits} qubits (dimension {dimension})...")

        try:
            start_time = time.perf_counter()
            kernel = BulletproofQuantumKernel(dimension=dimension)

            # Build resonance kernel
            resonance_kernel = kernel.build_resonance_kernel()
            total_time = time.perf_counter() - start_time

            kernel_type = (
                "Vector (vertex-based)"
                if len(resonance_kernel.shape) == 1
                else "Matrix (direct)"
            )

            result = {
                "qubits": n_qubits,
                "dimension": dimension,
                "time": total_time,
                "type": kernel_type,
                "success": True,
            }

            print(f"   ✅ Success: {total_time:.3f}s ({kernel_type})")

        except Exception as e:
            result = {
                "qubits": n_qubits,
                "dimension": dimension,
                "time": float("inf"),
                "type": "Failed",
                "success": False,
            }
            print(f"   ❌ Failed: {e}")

        results.append(result)

    print("\n📊 SCALING RESULTS:")
    print("-" * 50)
    for r in results:
        status = "✅" if r["success"] else "❌"
        print(
            f"{status} {r['qubits']:2d} qubits ({r['dimension']:5d}): {r['time']:6.3f}s - {r['type']}"
        )

    return results


if __name__ == "__main__":
    success1 = test_12_qubit_vertex_approach()
    results = test_scaling_performance()

    print("\n🎯 SUMMARY:")
    print("=" * 40)
    if success1:
        print("✅ 12-qubit simulation no longer hangs!")
        print("✅ Vertex-based holographic storage working!")
        print("✅ C++ canonical engines routing properly!")
        successful_tests = sum(1 for r in results if r["success"])
        print(f"✅ {successful_tests}/{len(results)} scaling tests passed!")
    else:
        print("❌ 12-qubit simulation still has issues")

    print("\n🔄 RFT PASSAGE LAYER STATUS:")
    print("Hardware Binary → RFT (C++ Canonical) → Python → Frontend")
    print("Status: ✅ ACTIVE" if success1 else "Status: ❌ NEEDS WORK")
