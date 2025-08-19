||

# LEGACY RFT IMPLEMENTATION - REPLACE WITH CANONICAL # from canonical_true_rft
import forward_true_rft, inverse_true_rft #!/usr/bin/env python3
""""""
Comprehensive RFT Non-Equivalence and Sigma-Tightening Validation Suite This module implements the complete test plan for demonstrating: 1. RFT != scaled/permuted DFT (non-equivalence tests) 2. Hash avalanche variance sigma <= 2 (sigma-tightening validation) Designed for publication-grade cryptographic validation.
"""
"""

import json
import numpy as np from statistics
import mean, pstdev from numpy.linalg
import norm, inv
import logging

# Configure logging logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') logger = logging.getLogger(__name__)
def run_non_equivalence_tests():
"""
"""
        Run all non-equivalence tests and
        return results.
"""
        """ logger.info("Running RFT non-equivalence test suite...")
        try:
from tests.test_rft_non_equivalence import ( test_rft_not_scaled_permuted_dft, test_rft_not_diagonalizing_shift, test_rft_moduli_not_scalable_to_uniform, build_rft_matrix, unitary_dft, greedy_match_columns, best_diagonal_right_and_left, cyclic_shift, diagonality_measure, sinkhorn_flatten_moduli ) results = {}

        # Test A: Column-wise scaled matching logger.info("Running Test A: Column-wise scaled matching...")
        try:

        # Use C++ engine for forward RFT
def forward_true_rft(x):
        try:
import quantonium_core rft = quantonium_core.ResonanceFourierTransform(x.tolist()) result = rft.forward_transform()
        return np.array(result, dtype=complex)
        except: from canonical_true_rft
import forward_true_rft, inverse_true_rft

        # Legacy wrapper maintained for: perform_rft_list result = perform_rft_list(x.tolist())
        return np.array([complex(freq, amp) for freq, amp in result])
        for n in (8, 12, 16): Psi_builder = build_rft_matrix(forward_true_rft) Psi = Psi_builder(n) F = unitary_dft(n) match = greedy_match_columns(Psi, F) D1, D2 = best_diagonal_right_and_left(Psi, F, match) approx = D1 @ F[:, match] @ D2 R = norm(Psi - approx, ord='fro') / max(1.0, norm(Psi, ord='fro')) results[f"nonequiv_residual_R_n{n}"] = float(R) logger.info(f" n={n}: Residual R = {R:.6e}") test_rft_not_scaled_permuted_dft() results["test_a_column_matching"] = "PASS" except Exception as e: logger.error(f"Test A failed: {e}") results["test_a_column_matching"] = f"FAIL: {e}"

        # Test B: Shift-operator diagonalization logger.info("Running Test B: Shift-operator diagonalization...")
        try:
def forward_true_rft(x):
        try:
import quantonium_core rft = quantonium_core.ResonanceFourierTransform(x.tolist()) result = rft.forward_transform()
        return np.array(result, dtype=complex)
        except: from canonical_true_rft
import forward_true_rft, inverse_true_rft

        # Legacy wrapper maintained for: perform_rft_list result = perform_rft_list(x.tolist())
        return np.array([complex(freq, amp) for freq, amp in result])
        for n in (8, 12, 16): Psi = build_rft_matrix(forward_true_rft)(n) S = cyclic_shift(n) M = inv(Psi) @ S @ Psi off = diagonality_measure(M) results[f"shift_offdiag_norm_n{n}"] = float(off) logger.info(f" n={n}: Off-diagonal norm = {off:.6e}") test_rft_not_diagonalizing_shift() results["test_b_shift_diagonalization"] = "PASS" except Exception as e: logger.error(f"Test B failed: {e}") results["test_b_shift_diagonalization"] = f"FAIL: {e}"

        # Test C: Uniform-modulus scalability logger.info("Running Test C: Uniform-modulus scalability...")
        try:
def forward_true_rft(x):
        try:
import quantonium_core rft = quantonium_core.ResonanceFourierTransform(x.tolist()) result = rft.forward_transform()
        return np.array(result, dtype=complex)
        except: from canonical_true_rft
import forward_true_rft, inverse_true_rft

        # Legacy wrapper maintained for: perform_rft_list result = perform_rft_list(x.tolist())
        return np.array([complex(freq, amp) for freq, amp in result])
        for n in (8, 12, 16): Psi = build_rft_matrix(forward_true_rft)(n) ok, _, _ = sinkhorn_flatten_moduli(Psi, iters=2000, tol=1e-8) results[f"sinkhorn_uniform_moduli_success_n{n}"] = bool(ok) logger.info(f" n={n}: Sinkhorn convergence = {ok}") test_rft_moduli_not_scalable_to_uniform() results["test_c_uniform_moduli"] = "PASS" except Exception as e: logger.error(f"Test C failed: {e}") results["test_c_uniform_moduli"] = f"FAIL: {e}"
        return results except ImportError as e: logger.error(f"Import error in non-equivalence tests: {e}")
        return {"error": f"Import error: {e}"}
def run_sigma_tightening_tests(): """"""
        Run sigma-tightening validation tests and
        return results.
"""
        """ logger.info("Running sigma-tightening validation suite...")
        try:
from tests.test_hash_sigma_tightening import ( test_hash_avalanche_sigma_tightened, bit_avalanche_rate ) from encryption.geometric_waveform_hash
import geometric_waveform_hash_bytes results = {} key = b'RFT-key-0123456789ABCDEF' rng = np.random.default_rng(42) N = 200 logger.info(f"Testing avalanche properties with {N} samples...")

        # Test avalanche properties rates = []
        for i in range(N):
        if i % 50 == 0: logger.info(f" Progress: {i}/{N}") m = rng.bytes(256) h1 = geometric_waveform_hash_bytes(m, key, rounds=2)

        # Flip 1 random bit b = bytearray(m) bit_pos = rng.integers(0, len(b)) bit_mask = 1 << rng.integers(0, 8) b[bit_pos] ^= bit_mask h2 = geometric_waveform_hash_bytes(bytes(b), key, rounds=2) rates.append(bit_avalanche_rate(h1, h2)) mu = mean(rates) sigma = pstdev(rates) results["hash_avalanche_mean"] = float(mu) results["hash_avalanche_sigma"] = float(sigma) logger.info(f"Avalanche mean: {mu:.2f}%") logger.info(f"Avalanche sigma: {sigma:.2f}%")

        # Run the formal test
        try: test_hash_avalanche_sigma_tightened() results["sigma_tightening_test"] = "PASS" logger.info("sigma-tightening test: PASS") except AssertionError as e: results["sigma_tightening_test"] = f"FAIL: {e}" logger.warning(f"sigma-tightening test: FAIL - {e}")
        return results except ImportError as e: logger.error(f"Import error in sigma-tightening tests: {e}")
        return {"error": f"Import error: {e}"}
def run_comprehensive_validation(): """"""
        Run complete validation suite and save results.
"""
        """ logger.info("Starting comprehensive RFT validation suite...")

        # Initialize results structure
import datetime results = { "test_suite": "RFT Non-Equivalence and Sigma-Tightening Validation", "version": "1.0", "timestamp": datetime.datetime.now().isoformat(), "non_equivalence_tests": {}, "sigma_tightening_tests": {}, "summary": {} }

        # Run non-equivalence tests non_equiv_results = run_non_equivalence_tests() results["non_equivalence_tests"] = non_equiv_results

        # Run sigma-tightening tests sigma_results = run_sigma_tightening_tests() results["sigma_tightening_tests"] = sigma_results

        # Generate summary non_equiv_pass = sum(1 for k, v in non_equiv_results.items()
        if k.startswith("test_") and v == "PASS") sigma_pass = sum(1 for k, v in sigma_results.items()
        if k.endswith("_test") and v == "PASS") results["summary"] = { "non_equivalence_tests_passed": non_equiv_pass, "sigma_tightening_tests_passed": sigma_pass, "overall_status": "PASS" if (non_equiv_pass >= 2 and sigma_pass >= 1) else "PARTIAL/FAIL", "validation_complete": True }

        # Save results output_file = "comprehensive_non_equivalence_and_sigma_validation.json" with open(output_file, 'w') as f: json.dump(results, f, indent=2) logger.info(f"Results saved to {output_file}") logger.info(f"Overall status: {results['summary']['overall_status']}")
        return results

if __name__ == "__main__": results = run_comprehensive_validation()

# Print summary
print("||n" + "="*60)
print("COMPREHENSIVE RFT VALIDATION SUMMARY")
print("="*60)
print(f"Non-equivalence tests passed: {results['summary']['non_equivalence_tests_passed']}")
print(f"Sigma-tightening tests passed: {results['summary']['sigma_tightening_tests_passed']}")
print(f"Overall status: {results['summary']['overall_status']}")
print("="*60)

# Print key metrics
if available if "hash_avalanche_sigma" in results["sigma_tightening_tests"]: sigma = results["sigma_tightening_tests"]["hash_avalanche_sigma"]
print(f"Hash avalanche sigma: {sigma:.3f} ({'✓ <= 2.0'
if sigma <= 2.0 else '✗ > 2.0'})")

# Print residual norms for key, value in results["non_equivalence_tests"].items():
if key.startswith("nonequiv_residual_R_"): n = key.split("_n")[1]
print(f"Non-equivalence residual (n={n}): {value:.2e} ({'✓ ≫ 1e-3'
if value > 1e-3 else '✗ <= 1e-3'})")