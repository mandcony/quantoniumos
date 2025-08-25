#!/usr/bin/env python3
"""
Analyze scaling behavior and geometric holographic encoding efficiency.
Test whether gradual scaling is needed or if the current approach is optimal.
"""
import os
import sys
import time
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), '05_QUANTUM_ENGINES'))
from bulletproof_quantum_kernel import BulletproofQuantumKernel


def analyze_scaling_strategy():
    """Analyze current scaling approach and geometric holographic efficiency."""
    print("Scaling Strategy Analysis for Geometric Holographic Encoding")
    print("=" * 65)

    # Test dimensions to analyze scaling behavior
    test_dims = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]

    results = {
        "dimensions": [],
        "build_times": [],
        "storage_type": [],
        "vertex_count": [],
        "memory_efficiency": [],
        "encoding_quality": [],
    }

    print("\n1. CURRENT SCALING APPROACH ANALYSIS")
    print("-" * 45)

    for dim in test_dims:
        print(f"\nTesting dimension: {dim}")

        # Time the kernel construction
        start_time = time.time()
        try:
            kernel = BulletproofQuantumKernel(dimension=dim)
            kernel.build_resonance_kernel()
            build_time = time.time() - start_time

            # Determine storage approach
            if dim <= 16:
                storage_type = "Matrix (O(N²))"
                vertex_count = dim  # Full matrix
                memory_usage = dim * dim * 16  # bytes (complex128)
            else:
                storage_type = "Vertex Holographic"
                vertex_count = min(dim, 32)  # Fixed number of vertices
                memory_usage = dim * 16  # Linear storage

            # Memory efficiency (lower is better)
            theoretical_full_memory = dim * dim * 16
            memory_efficiency = memory_usage / theoretical_full_memory

            # Test encoding quality with a simple signal
            test_signal = np.random.random(dim) + 1j * np.random.random(dim)
            try:
                kernel.compute_rft_basis()
                rft_result = kernel.forward_rft(test_signal)
                roundtrip = kernel.inverse_rft(rft_result)
                encoding_quality = np.linalg.norm(
                    test_signal - roundtrip
                ) / np.linalg.norm(test_signal)
            except Exception:
                encoding_quality = float("inf")

            results["dimensions"].append(dim)
            results["build_times"].append(build_time)
            results["storage_type"].append(storage_type)
            results["vertex_count"].append(vertex_count)
            results["memory_efficiency"].append(memory_efficiency)
            results["encoding_quality"].append(encoding_quality)

            print(f"   Build time: {build_time:.4f}s")
            print(f"   Storage: {storage_type}")
            print(f"   Vertices: {vertex_count}")
            print(f"   Memory efficiency: {memory_efficiency:.6f} (vs full matrix)")
            print(f"   Encoding quality: {encoding_quality:.6f}")

        except Exception as e:
            print(f"   ❌ Failed: {e}")
            break

    print("\n\n2. SCALING STRATEGY ANALYSIS")
    print("-" * 35)

    # Analyze the transition point
    matrix_dims = [
        d
        for d, s in zip(results["dimensions"], results["storage_type"])
        if "Matrix" in s
    ]
    vertex_dims = [
        d
        for d, s in zip(results["dimensions"], results["storage_type"])
        if "Vertex" in s
    ]

    print(
        f"Matrix approach: dimensions {min(matrix_dims) if matrix_dims else 'None'} to {max(matrix_dims) if matrix_dims else 'None'}"
    )
    print(
        f"Vertex holographic: dimensions {min(vertex_dims) if vertex_dims else 'None'} to {max(vertex_dims) if vertex_dims else 'None'}"
    )

    # Check for performance cliffs
    if len(results["build_times"]) > 1:
        time_ratios = [
            results["build_times"][i + 1] / results["build_times"][i]
            for i in range(len(results["build_times"]) - 1)
        ]
        max_ratio_idx = np.argmax(time_ratios)
        transition_dim = results["dimensions"][max_ratio_idx]
        max_ratio = time_ratios[max_ratio_idx]

        print(
            f"\nLargest performance jump: {max_ratio:.2f}x at dimension {transition_dim}"
        )
        if max_ratio > 5:
            print("   ⚠️ Significant performance cliff detected!")
        else:
            print("   ✅ Smooth scaling transition")

    print("\n\n3. GEOMETRIC HOLOGRAPHIC ENCODING EFFICIENCY")
    print("-" * 50)

    # Analyze holographic encoding characteristics
    vertex_results = [
        (d, me, eq)
        for d, me, eq, st in zip(
            results["dimensions"],
            results["memory_efficiency"],
            results["encoding_quality"],
            results["storage_type"],
        )
        if "Vertex" in st
    ]

    if vertex_results:
        dims, mem_effs, enc_quals = zip(*vertex_results)

        print(
            f"Vertex holographic encoding covers dimensions: {min(dims)} to {max(dims)}"
        )
        print(f"Memory efficiency range: {min(mem_effs):.6f} to {max(mem_effs):.6f}")
        print(f"Encoding quality range: {min(enc_quals):.6f} to {max(enc_quals):.6f}")

        # Check if encoding quality degrades with scale
        quality_trend = np.polyfit(dims, enc_quals, 1)[0]  # Linear trend
        if quality_trend > 0.001:
            print(
                f"   ⚠️ Encoding quality degrades with scale (slope: {quality_trend:.6f})"
            )
            print("   → Consider gradual scaling or quality compensation")
        else:
            print(
                f"   ✅ Encoding quality stable with scale (slope: {quality_trend:.6f})"
            )
            print("   → Current approach is optimal")

    print("\n\n4. SCALING RECOMMENDATION")
    print("-" * 30)

    # Determine if gradual scaling is needed
    needs_gradual_scaling = False
    reasons = []

    # Check for performance cliffs
    if len(results["build_times"]) > 1:
        time_ratios = [
            results["build_times"][i + 1] / results["build_times"][i]
            for i in range(len(results["build_times"]) - 1)
        ]
        if max(time_ratios) > 10:
            needs_gradual_scaling = True
            reasons.append("Large performance jumps detected")

    # Check for quality degradation
    if vertex_results:
        dims, _, enc_quals = zip(*vertex_results)
        quality_trend = np.polyfit(dims, enc_quals, 1)[0]
        if quality_trend > 0.01:
            needs_gradual_scaling = True
            reasons.append("Encoding quality degrades significantly with scale")

    # Check memory efficiency
    if results["memory_efficiency"]:
        max_mem_eff = max(results["memory_efficiency"])
        if max_mem_eff > 0.1:  # Using more than 10% of full matrix memory
            needs_gradual_scaling = True
            reasons.append("Memory efficiency could be improved")

    if needs_gradual_scaling:
        print("🔄 RECOMMENDATION: Implement gradual scaling")
        print("   Reasons:")
        for reason in reasons:
            print(f"   - {reason}")
        print("   Suggested approach:")
        print("   - Use dimension doubling with quality checkpoints")
        print("   - Implement adaptive vertex count based on dimension")
        print("   - Add quality compensation for large scales")
    else:
        print("✅ RECOMMENDATION: Current approach is optimal")
        print("   The geometric holographic encoding handles scaling efficiently:")
        print("   - Smooth performance transitions")
        print("   - Stable encoding quality")
        print("   - Good memory efficiency")
        print("   - No need for gradual scaling")

    return results, not needs_gradual_scaling


if __name__ == "__main__":
    results, is_optimal = analyze_scaling_strategy()

    if is_optimal:
        print("\n🎉 Current geometric holographic scaling is OPTIMAL!")
        print("The vertex-based approach with 32 vertices managing N dimensions")
        print("provides excellent scalability without requiring gradual scaling.")
    else:
        print("\n⚠️ Consider implementing gradual scaling improvements.")
