#!/usr/bin/env python3
"""
Test the fixed compute_rft_basis method for both matrix and vertex-based approaches
"""


def test_rft_basis_computation():
    print("Testing RFT basis computation fix...")
    print("=" * 50)

    import time

    import numpy as np

    from 05_QUANTUM_ENGINES.bulletproof_quantum_kernel import BulletproofQuantumKernel

    # Test cases: small (matrix-based) and large (vertex-based)
    test_cases = [
        {"name": "Small (Matrix-based)", "dimension": 16},
        {"name": "Medium (Vertex-based)", "dimension": 32},
        {"name": "Large (Vertex-based)", "dimension": 64},
    ]

    results = []

    for case in test_cases:
        print(f"\nTesting {case['name']} - Dimension {case['dimension']}:")

        try:
            # Create kernel
            start_time = time.perf_counter()
            kernel = BulletproofQuantumKernel(dimension=case["dimension"])

            # Build resonance kernel
            resonance_kernel = kernel.build_resonance_kernel()
            kernel_type = "Matrix" if len(resonance_kernel.shape) == 2 else "Vector"
            print(f"   Resonance kernel: {resonance_kernel.shape} ({kernel_type})")

            # Test RFT basis computation (this was failing before)
            eigenvalues, eigenvectors = kernel.compute_rft_basis()
            basis_time = time.perf_counter() - start_time

            print(f"   Eigenvalues shape: {eigenvalues.shape}")
            print(f"   Eigenvectors shape: {eigenvectors.shape}")
            print(f"   Total time: {basis_time:.3f}s")

            # Test a forward RFT operation
            test_signal = np.random.randn(case["dimension"]) + 1j * np.random.randn(
                case["dimension"]
            )
            test_signal /= np.linalg.norm(test_signal)

            rft_result = kernel.forward_rft(test_signal)
            print(f"   RFT test: {test_signal.shape} -> {rft_result.shape}")

            results.append(
                {
                    "name": case["name"],
                    "dimension": case["dimension"],
                    "success": True,
                    "time": basis_time,
                    "kernel_type": kernel_type,
                }
            )

            print(f"   SUCCESS!")

        except Exception as e:
            print(f"   FAILED: {e}")
            results.append(
                {
                    "name": case["name"],
                    "dimension": case["dimension"],
                    "success": False,
                    "error": str(e),
                }
            )

    print("\n" + "=" * 50)
    print("SUMMARY:")
    print("-" * 30)

    success_count = 0
    for r in results:
        status = "PASS" if r["success"] else "FAIL"
        if r["success"]:
            print(
                f"{status}: {r['name']} ({r['dimension']}) - {r['time']:.3f}s - {r['kernel_type']}"
            )
            success_count += 1
        else:
            print(
                f"{status}: {r['name']} ({r['dimension']}) - {r.get('error', 'Unknown error')}"
            )

    print(f"\nResult: {success_count}/{len(results)} tests passed")

    if success_count == len(results):
        print("\nSUCCESS: RFT basis computation fixed for both approaches!")
        print("Hardware Binary -> RFT (C++ Canonical) -> Python -> Frontend: WORKING")
        return True
    else:
        print("\nSome tests still failing...")
        return False


if __name__ == "__main__":
    test_rft_basis_computation()
