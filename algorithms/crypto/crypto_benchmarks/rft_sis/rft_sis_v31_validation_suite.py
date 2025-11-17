# QuantoniumOS — RFT–SIS Hybrid Geometric Hash (v3.1)
# Copyright (C) 2025 Luis M. Minier
# License: AGPL-3.0-or-later
# Patent Notice: Practices claims of U.S. Patent Application No. 19/169,399.
# Commercial use of RFT requires a paid license. See PATENT_NOTICE.md / COMMERCIAL_LICENSE.md.
# Security Status: Research code. No formal security proof. No warranty.

#!/usr/bin/env python3
"""
FULL VALIDATION REPORT: RFT-SIS Hash v3.1
"""

import sys
sys.path.insert(0, '/home/claude')
from rft_sis_hash_v31 import RFTSISHashV31, Point2D, Point3D, CanonicalTrueRFT

import numpy as np
import time
import json
from datetime import datetime

def comprehensive_validation():
    print("=" * 70)
    print("RFT-SIS HASH v3.1 - FULL VALIDATION REPORT")
    print(f"Generated: {datetime.now().isoformat()}")
    print("=" * 70)
    
    results = {}
    
    # ==========================================================================
    # TEST 1: RFT MATRIX UNITARITY
    # ==========================================================================
    print("\n[1] RFT MATRIX UNITARITY")
    print("-" * 70)
    
    unitarity_results = []
    for N in [64, 128, 256, 512]:
        rft = CanonicalTrueRFT(N=N)
        Psi = rft._rft_matrix
        
        # Unitarity check
        product = Psi.conj().T @ Psi
        frobenius_err = np.linalg.norm(product - np.eye(N), 'fro')
        
        # Energy preservation
        x = np.random.randn(N) + 1j * np.random.randn(N)
        energy_ratio = np.linalg.norm(Psi @ x) / np.linalg.norm(x)
        
        # Roundtrip
        x_rt = Psi.conj().T @ (Psi @ x)
        rt_err = np.linalg.norm(x - x_rt) / np.linalg.norm(x)
        
        unitarity_results.append({
            'N': N,
            'frobenius_error': float(frobenius_err),
            'energy_ratio': float(energy_ratio),
            'roundtrip_error': float(rt_err)
        })
        
        print(f"N={N:4d}: ||Ψ†Ψ-I||_F = {frobenius_err:.2e}, "
              f"Energy = {energy_ratio:.15f}, RT err = {rt_err:.2e}")
    
    results['unitarity'] = unitarity_results
    
    # ==========================================================================
    # TEST 2: AVALANCHE EFFECT (Statistical)
    # ==========================================================================
    print("\n[2] AVALANCHE EFFECT (100 trials per delta)")
    print("-" * 70)
    
    hasher = RFTSISHashV31()
    avalanche_results = {}
    
    for delta in [1e-3, 1e-6, 1e-9, 1e-12, 1e-15]:
        bit_diffs = []
        
        for _ in range(100):
            x = np.random.uniform(-10, 10)
            y = np.random.uniform(-10, 10)
            
            h1 = hasher.hash_point(Point2D(x, y))
            h2 = hasher.hash_point(Point2D(x + delta, y))
            
            diff = sum(bin(a ^ b).count('1') for a, b in zip(h1, h2))
            bit_diffs.append(diff / 256)
        
        mean_d = np.mean(bit_diffs)
        std_d = np.std(bit_diffs)
        
        avalanche_results[str(delta)] = {
            'mean': float(mean_d),
            'std': float(std_d),
            'min': float(np.min(bit_diffs)),
            'max': float(np.max(bit_diffs))
        }
        
        status = "✓" if 0.40 <= mean_d <= 0.60 else "✗"
        print(f"Δ={delta:.0e}: {mean_d*100:.2f}% ± {std_d*100:.2f}% "
              f"[{np.min(bit_diffs)*100:.1f}%, {np.max(bit_diffs)*100:.1f}%] {status}")
    
    results['avalanche'] = avalanche_results
    
    # ==========================================================================
    # TEST 3: COLLISION RESISTANCE
    # ==========================================================================
    print("\n[3] COLLISION RESISTANCE")
    print("-" * 70)
    
    num_tests = 10000
    hashes = set()
    collisions = 0
    
    np.random.seed(12345)
    for _ in range(num_tests):
        x = np.random.uniform(-100, 100)
        y = np.random.uniform(-100, 100)
        h = hasher.hash_point(Point2D(x, y))
        if h in hashes:
            collisions += 1
        else:
            hashes.add(h)
    
    collision_rate = collisions / num_tests
    
    results['collision_resistance'] = {
        'num_points': num_tests,
        'collisions': collisions,
        'collision_rate': float(collision_rate),
        'unique_hashes': len(hashes)
    }
    
    print(f"Points tested: {num_tests}")
    print(f"Collisions: {collisions}")
    print(f"Collision rate: {collision_rate*100:.4f}%")
    print(f"Status: {'✓ PASS' if collisions == 0 else '✗ FAIL'}")
    
    # ==========================================================================
    # TEST 4: BIT DISTRIBUTION
    # ==========================================================================
    print("\n[4] BIT DISTRIBUTION UNIFORMITY")
    print("-" * 70)
    
    num_samples = 1000
    bit_counts = np.zeros(256)
    
    np.random.seed(54321)
    for _ in range(num_samples):
        x = np.random.uniform(-100, 100)
        y = np.random.uniform(-100, 100)
        h = hasher.hash_point(Point2D(x, y))
        
        for byte_idx, byte in enumerate(h):
            for bit_idx in range(8):
                if byte & (1 << bit_idx):
                    bit_counts[byte_idx * 8 + bit_idx] += 1
    
    bit_freq = bit_counts / num_samples
    mean_freq = np.mean(bit_freq)
    std_freq = np.std(bit_freq)
    
    results['bit_distribution'] = {
        'mean_frequency': float(mean_freq),
        'std_frequency': float(std_freq),
        'min_frequency': float(np.min(bit_freq)),
        'max_frequency': float(np.max(bit_freq)),
        'ideal': 0.5
    }
    
    print(f"Mean bit frequency: {mean_freq:.4f} (ideal: 0.5000)")
    print(f"Std deviation: {std_freq:.4f}")
    print(f"Range: [{np.min(bit_freq):.4f}, {np.max(bit_freq):.4f}]")
    print(f"Status: {'✓ PASS' if abs(mean_freq - 0.5) < 0.02 else '✗ FAIL'}")
    
    # ==========================================================================
    # TEST 5: PERFORMANCE
    # ==========================================================================
    print("\n[5] PERFORMANCE BENCHMARKS")
    print("-" * 70)
    
    # Warm up
    for _ in range(100):
        hasher.hash_point(Point2D(1.0, 2.0))
    
    # Benchmark
    num_ops = 1000
    start = time.perf_counter()
    for i in range(num_ops):
        hasher.hash_point(Point2D(float(i), float(i + 1)))
    elapsed = time.perf_counter() - start
    
    ops_per_sec = num_ops / elapsed
    ms_per_op = (elapsed / num_ops) * 1000
    
    results['performance'] = {
        'operations': num_ops,
        'total_time_sec': float(elapsed),
        'hashes_per_sec': float(ops_per_sec),
        'ms_per_hash': float(ms_per_op)
    }
    
    print(f"Throughput: {ops_per_sec:.1f} hashes/second")
    print(f"Latency: {ms_per_op:.3f} ms/hash")
    print(f"Parameters: n={hasher.sis_n}, m={hasher.sis_m}, q={hasher.sis_q}")
    
    # ==========================================================================
    # TEST 6: 3D HASHING
    # ==========================================================================
    print("\n[6] 3D POINT HASHING")
    print("-" * 70)
    
    # Note: v3.1 supports 3D natively
    base_3d = Point3D(1.0, 2.0, 3.0)
    h_base = hasher.hash_point(base_3d)
    
    diff_x = sum(bin(a ^ b).count('1') for a, b in zip(
        h_base, hasher.hash_point(Point3D(1.001, 2.0, 3.0)))) / 256
    diff_y = sum(bin(a ^ b).count('1') for a, b in zip(
        h_base, hasher.hash_point(Point3D(1.0, 2.001, 3.0)))) / 256
    diff_z = sum(bin(a ^ b).count('1') for a, b in zip(
        h_base, hasher.hash_point(Point3D(1.0, 2.0, 3.001)))) / 256
    
    results['3d_hashing'] = {
        'diff_x': float(diff_x),
        'diff_y': float(diff_y),
        'diff_z': float(diff_z)
    }
    
    print(f"Delta in X: {diff_x*100:.2f}% bit difference")
    print(f"Delta in Y: {diff_y*100:.2f}% bit difference")
    print(f"Delta in Z: {diff_z*100:.2f}% bit difference")
    all_ok = all(0.40 <= d <= 0.60 for d in [diff_x, diff_y, diff_z])
    print(f"Status: {'✓ PASS' if all_ok else '✗ FAIL'}")
    
    # ==========================================================================
    # TEST 7: SIS SECURITY PARAMETERS
    # ==========================================================================
    print("\n[7] SECURITY PARAMETER ANALYSIS")
    print("-" * 70)
    
    n = hasher.sis_n
    m = hasher.sis_m
    q = hasher.sis_q
    beta = hasher.sis_beta
    
    # Approximate security (simplified estimate)
    security_bits = n * np.log2(q / beta)
    delta_hermite = (q / beta) ** (1 / n)
    
    results['security'] = {
        'n': n,
        'm': m,
        'q': q,
        'beta': beta,
        'estimated_security_bits': float(security_bits),
        'root_hermite_delta': float(delta_hermite),
        'nist_level_1': security_bits >= 128
    }
    
    print(f"Parameters: n={n}, m={m}, q={q}, β={beta}")
    print(f"Estimated security: {security_bits:.0f} bits")
    print(f"Root Hermite δ: {delta_hermite:.6f}")
    print(f"NIST Level 1 (128-bit): {'✓ YES' if security_bits >= 128 else '✗ NO'}")
    
    # ==========================================================================
    # TEST 8: SIS VECTOR BOUNDS
    # ==========================================================================
    print("\n[8] SIS VECTOR BOUND ENFORCEMENT")
    print("-" * 70)
    
    violations = 0
    max_observed = 0
    
    np.random.seed(99999)
    for _ in range(1000):
        x = np.random.uniform(-100, 100)
        y = np.random.uniform(-100, 100)
        
        coords = np.array([x, y], dtype=np.float64)
        expanded = hasher._expand_coordinates_cryptographically(coords)
        rft_out = hasher._rft_preprocess(expanded)
        s = hasher._quantize_to_sis(rft_out)
        
        max_s = np.max(np.abs(s))
        max_observed = max(max_observed, max_s)
        
        if max_s > beta:
            violations += 1
    
    results['sis_bounds'] = {
        'beta': beta,
        'max_observed': int(max_observed),
        'violations': violations
    }
    
    print(f"SIS bound β: {beta}")
    print(f"Max observed |s|: {max_observed}")
    print(f"Violations: {violations}/1000")
    print(f"Status: {'✓ PASS' if violations == 0 else '✗ FAIL'}")
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    tests_passed = 0
    total_tests = 8
    
    # Check each test
    if all(r['frobenius_error'] < 1e-10 for r in unitarity_results):
        tests_passed += 1
        print("✓ [1] Unitarity: PASS")
    else:
        print("✗ [1] Unitarity: FAIL")
    
    if all(0.40 <= avalanche_results[k]['mean'] <= 0.60 
           for k in avalanche_results):
        tests_passed += 1
        print("✓ [2] Avalanche: PASS")
    else:
        print("✗ [2] Avalanche: FAIL")
    
    if collisions == 0:
        tests_passed += 1
        print("✓ [3] Collision Resistance: PASS")
    else:
        print("✗ [3] Collision Resistance: FAIL")
    
    if abs(mean_freq - 0.5) < 0.02:
        tests_passed += 1
        print("✓ [4] Bit Distribution: PASS")
    else:
        print("✗ [4] Bit Distribution: FAIL")
    
    if ops_per_sec > 100:  # At least 100 hashes/sec
        tests_passed += 1
        print("✓ [5] Performance: PASS")
    else:
        print("✗ [5] Performance: FAIL")
    
    if all_ok:
        tests_passed += 1
        print("✓ [6] 3D Hashing: PASS")
    else:
        print("✗ [6] 3D Hashing: FAIL")
    
    if security_bits >= 128:
        tests_passed += 1
        print("✓ [7] Security Parameters: PASS")
    else:
        print("✗ [7] Security Parameters: FAIL")
    
    if violations == 0:
        tests_passed += 1
        print("✓ [8] SIS Bounds: PASS")
    else:
        print("✗ [8] SIS Bounds: FAIL")
    
    print(f"\nTotal: {tests_passed}/{total_tests} tests passed")
    print("=" * 70)
    
    results['summary'] = {
        'tests_passed': tests_passed,
        'total_tests': total_tests,
        'all_pass': tests_passed == total_tests
    }
    
    return results


if __name__ == "__main__":
    results = comprehensive_validation()
    
    # Save to JSON for records
    with open('/home/claude/rft_sis_v31_validation_report.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nFull results saved to: rft_sis_v31_validation_report.json")
