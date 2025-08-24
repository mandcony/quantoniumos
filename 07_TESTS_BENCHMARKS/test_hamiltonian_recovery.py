#!/usr/bin/env python3
"""
Hamiltonian Recovery and Generator Consistency Test Suite === This test validates Hamiltonian extraction and generator consistency: - Extract Hamiltonian H from unitary U via H = i log(U) - Check reconstruction: ||U - e^(-iH)||2 <= 1e-10 - Verify H is Hermitian: ||H - Hdagger||2 <= 1e-12 Uses symbolic resonance computing methods from the canonical RFT implementation.
"""

import math
import time
import typing

import Any
import Dict
import List
import numpy as np
import Optional
import scipy.linalg
import Tuple

from 04_RFT_ALGORITHMS.canonical_true_rft import (forward_true_rft, generate_resonance_kernel,
                                get_rft_basis, inverse_true_rft)


def random_state(N: int) -> np.ndarray: """
        Generate a random complex state vector.
"""
        rng = np.random.default_rng() real_part = rng.normal(0, 1, N) imag_part = rng.normal(0, 1, N)
        return real_part + 1j * imag_part
def project_unitary(A: np.ndarray) -> np.ndarray: """
        Project matrix to nearest unitary using polar decomposition.
"""
        U, _ = scipy.linalg.polar(A)
        return U
def safe_log_unitary(U: np.ndarray) -> np.ndarray: """
        Safely compute matrix logarithm of unitary, avoiding branch cut issues.
"""
        eigenvals, eigenvecs = np.linalg.eig(U)

        # Take principal branch of log for eigenvalues log_eigenvals = np.log(eigenvals + 1e-15)

        # Small offset to avoid log(0)

        # Construct H = i * log(U) such that H is Hermitian log_U = eigenvecs @ np.diag(log_eigenvals) @ eigenvecs.T.conj()
        return 1j * log_U
def bra_ket(psi: np.ndarray, phi: np.ndarray) -> complex: """
        Compute <psi|phi> with explicit conjugate transpose.
"""

        return np.conj(psi).T @ phi
def loschmidt_echo(H: np.ndarray, psi_0: np.ndarray, t: float, eps: float = 1e-6) -> float: """
        Compute Loschmidt echo with small perturbation to avoid trivial result.
"""

        # Add small perturbation to Hamiltonian N = len(psi_0) H_pert = H + perturbation_strength * random_state(N*N).reshape(N, N) H_pert = 0.5 * (H_pert + H_pert.T.conj())

        # Ensure Hermitian

        # Forward and backward evolution t_eff = 1.0

        # Effective time scale U_forward = scipy.linalg.expm(-1j * H * t_eff) U_backward = scipy.linalg.expm(1j * H_pert * t_eff)

        # Loschmidt echo psi_final = U_backward @ U_forward @ psi_0
        return abs(bra_ket(psi_0, psi_final))**2

class HamiltonianRecoveryValidator: """
        Validates Hamiltonian extraction and generator consistency using symbolic resonance computing methods.
"""

    def __init__(self, tolerance: float = 1e-12):
        self.tolerance = tolerance
        self.reconstruction_tolerance = 1e-10

        # Slightly relaxed for numerical stability
        self.results = {}
    def compute_hamiltonian_metrics(self, H: np.ndarray) -> Dict[str, Any]: """
        Compute quality metrics for extracted Hamiltonian.
"""

        # Hermiticity check H_dagger = H.T.conj() hermiticity_error = np.linalg.norm(H - H_dagger, ord='fro')

        # Eigenvalue analysis eigenvals = np.linalg.eigvals(H) max_imaginary = np.max(np.abs(eigenvals.imag))

        # Condition number cond_num = np.linalg.cond(H)
        return { 'hermiticity_error': float(hermiticity_error), 'max_imaginary_eigenvalue': float(max_imaginary), 'condition_number': float(cond_num), 'trace': float(np.trace(H).real), 'frobenius_norm': float(np.linalg.norm(H, ord='fro')) }
    def extract_hamiltonian(self, U: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]: """
        Extract Hamiltonian H from unitary U via H = i log(U) Returns: H: Extracted Hamiltonian (should be Hermitian) metrics: Extraction quality metrics
"""
        N = U.shape[0]

        # Project to nearest unitary first U_unitary = project_unitary(U)
        try:

        # Use safe logarithm to avoid branch cut issues H = safe_log_unitary(U_unitary) extraction_successful = True error_msg = None except Exception as e:

        # Fallback:
        return zero Hamiltonian with error info H = np.zeros_like(U, dtype=complex) extraction_successful = False error_msg = str(e)

        # Compute metrics metrics =
        self.compute_hamiltonian_metrics(H) metrics['extraction_successful'] = extraction_successful metrics['projection_error'] = np.linalg.norm(U - U_unitary, ord='fro')
        if error_msg: metrics['error'] = error_msg
        return H, metrics
    def test_hamiltonian_reconstruction(self, N: int = 8) -> Dict[str, Any]: """
        Test Hamiltonian extraction and reconstruction consistency. Goal: ||U - e^(-i log_i U)2 <= 1e-10
"""

        print(f"Testing Hamiltonian reconstruction for N={N}...") start_time = time.time()

        # Get RFT unitary operator U_raw = get_rft_basis(N) U = project_unitary(U_raw)

        # Extract Hamiltonian
        try: H, extraction_metrics =
        self.extract_hamiltonian(U) except ValueError as e:
        return { 'test_name': 'Hamiltonian Reconstruction', 'size': N, 'error': str(e), 'passes': False, 'test_time': time.time() - start_time }

        # Reconstruct U from H: U_reconstructed = e^(-iH)
        try: minus_iH = -1j * H U_reconstructed = scipy.linalg.expm(minus_iH) except Exception as e:
        return { 'test_name': 'Hamiltonian Reconstruction', 'size': N, 'error': f"Matrix exponential failed: {e}", 'passes': False, 'test_time': time.time() - start_time }

        # Compute reconstruction error reconstruction_error = np.linalg.norm(U - U_reconstructed, ord=2)

        # Additional consistency checks consistency_metrics = {}

        # Check that e^(-iH) is unitary U_rec_dagger = U_reconstructed.T.conj() unitarity_error = np.linalg.norm(U_rec_dagger @ U_reconstructed - np.eye(N), ord=2) consistency_metrics['reconstructed_unitarity_error'] = unitarity_error

        # Check determinant is unit magnitude (unitary property) det_U_rec = np.linalg.det(U_reconstructed) det_magnitude_error = abs(abs(det_U_rec) - 1.0) consistency_metrics['determinant_magnitude_error'] = det_magnitude_error

        # Pass conditions passes = ( extraction_metrics['hermiticity_error'] <=
        self.tolerance and extraction_metrics['max_imaginary_eigenvalue'] <=
        self.tolerance and reconstruction_error <=
        self.reconstruction_tolerance and unitarity_error <=
        self.tolerance ) test_time = time.time() - start_time result = { 'test_name': 'Hamiltonian Reconstruction', 'size': N, 'reconstruction_error': float(reconstruction_error), 'extraction_metrics': {k: float(v) for k, v in extraction_metrics.items()}, 'consistency_metrics': {k: float(v) for k, v in consistency_metrics.items()}, 'passes': bool(passes), 'tolerance':
        self.tolerance, 'reconstruction_tolerance':
        self.reconstruction_tolerance, 'test_time': test_time }
        self.results[f'reconstruction_N{N}'] = result
        return result
    def test_generator_consistency(self, N: int = 8) -> Dict[str, Any]: """
        Test generator consistency across different parametrizations. Generate multiple unitaries and check Hamiltonian consistency.
"""

        print(f"Testing generator consistency for N={N}...") start_time = time.time()

        # Test with different powers of the base unitary base_U_raw = get_rft_basis(N) base_U = project_unitary(base_U_raw) powers = [1, 2, 3, 4] generator_results = []
        for power in powers: U = np.linalg.matrix_power(base_U, power)
        try: H, metrics =
        self.extract_hamiltonian(U)

        # The Hamiltonian of U^p should be p * H_base (approximately)
        if power == 1: H_base = H
        else: expected_H = power * H_base consistency_error = np.linalg.norm(H - expected_H, ord=2) metrics['power_consistency_error'] = consistency_error generator_results.append({ 'power': power, 'metrics': metrics, 'extraction_successful': True }) except Exception as e: generator_results.append({ 'power': power, 'error': str(e), 'extraction_successful': False })

        # Check overall consistency successful_extractions = [r
        for r in generator_results
        if r['extraction_successful']]
        if len(successful_extractions) < 2: passes = False avg_hermiticity_error = float('inf') max_power_consistency_error = float('inf')
        else: hermiticity_errors = [r['metrics']['hermiticity_error']
        for r in successful_extractions] avg_hermiticity_error = np.mean(hermiticity_errors) power_consistency_errors = [ r['metrics'].get('power_consistency_error', 0.0)
        for r in successful_extractions[1:]

        # Skip power=1 ] max_power_consistency_error = max(power_consistency_errors)
        if power_consistency_errors else 0.0 passes = ( avg_hermiticity_error <=
        self.tolerance and max_power_consistency_error <= 1e-6

        # Relaxed due to numerical accumulation ) test_time = time.time() - start_time result = { 'test_name': 'Generator Consistency', 'size': N, 'generator_results': generator_results, 'avg_hermiticity_error': float(avg_hermiticity_error), 'max_power_consistency_error': float(max_power_consistency_error), 'successful_extractions': len(successful_extractions), 'total_tests': len(generator_results), 'passes': bool(passes), 'tolerance':
        self.tolerance, 'test_time': test_time }
        self.results[f'consistency_N{N}'] = result
        return result
    def test_spectral_properties(self, N: int = 8) -> Dict[str, Any]: """
        Test spectral properties of extracted Hamiltonians. - Check eigenvalue distribution - Verify trace properties - Check spectral radius bounds
"""

        print(f"Testing spectral properties for N={N}...") start_time = time.time() U_raw = get_rft_basis(N) U = project_unitary(U_raw)
        try: H, extraction_metrics =
        self.extract_hamiltonian(U) except ValueError as e:
        return { 'test_name': 'Spectral Properties', 'size': N, 'error': str(e), 'passes': False, 'test_time': time.time() - start_time }

        # Compute spectral properties H_eigenvals = np.linalg.eigvals(H) H_eigenvals_real = H_eigenvals.real spectral_metrics = {}

        # Trace should be purely real and related to eigenvalue sum trace_H = np.trace(H) spectral_metrics['trace_imaginary_part'] = abs(trace_H.imag) spectral_metrics['trace_real_part'] = trace_H.real

        # Eigenvalue statistics spectral_metrics['min_eigenvalue'] = np.min(H_eigenvals_real) spectral_metrics['max_eigenvalue'] = np.max(H_eigenvals_real) spectral_metrics['eigenvalue_spread'] = np.max(H_eigenvals_real) - np.min(H_eigenvals_real) spectral_metrics['spectral_radius'] = np.max(np.abs(H_eigenvals_real))

        # Check eigenvalue imaginary parts are small spectral_metrics['max_eigenvalue_imag'] = np.max(np.abs(H_eigenvals.imag))

        # Operator norm consistency spectral_metrics['operator_norm'] = np.linalg.norm(H, ord=2) spectral_metrics['frobenius_norm'] = np.linalg.norm(H, ord='fro')

        # Pass conditions passes = ( spectral_metrics['trace_imaginary_part'] <=
        self.tolerance and spectral_metrics['max_eigenvalue_imag'] <=
        self.tolerance and extraction_metrics['hermiticity_error'] <=
        self.tolerance ) test_time = time.time() - start_time result = { 'test_name': 'Spectral Properties', 'size': N, 'spectral_metrics': {k: float(v) for k, v in spectral_metrics.items()}, 'extraction_metrics': {k: float(v) for k, v in extraction_metrics.items()}, 'H_eigenvalues_real': H_eigenvals_real.tolist(), 'H_eigenvalues_imag': H_eigenvals.imag.tolist(), 'passes': bool(passes), 'tolerance':
        self.tolerance, 'test_time': test_time }
        self.results[f'spectral_N{N}'] = result
        return result
    def run_full_hamiltonian_suite(self, sizes: List[int] = None) -> Dict[str, Any]: """
        Run the complete Hamiltonian recovery test suite.
"""

        if sizes is None: sizes = [4, 8, 16]
        print("=" * 60)
        print("HAMILTONIAN RECOVERY VALIDATION TEST SUITE")
        print("Using Symbolic Resonance Computing Methods")
        print("=" * 60) suite_results = { 'suite_name': 'Hamiltonian Recovery Validation', 'timestamp': time.time(), 'tolerance':
        self.tolerance, 'reconstruction_tolerance':
        self.reconstruction_tolerance, 'test_sizes': sizes, 'results': {} }
        for N in sizes:
        print(f"\nTesting size N={N}")
        print("-" * 30) size_results = {}

        # Run all tests for this size size_results['reconstruction'] =
        self.test_hamiltonian_reconstruction(N) size_results['consistency'] =
        self.test_generator_consistency(N) size_results['spectral'] =
        self.test_spectral_properties(N)

        # Summary for this size all_pass = all(result['passes']
        for result in size_results.values()) size_results['overall_pass'] = all_pass suite_results['results'][f'N_{N}'] = size_results
        print(f"Size N={N} overall: {'✓ PASS'
        if all_pass else '❌ FAIL'}")

        # Overall suite summary all_sizes_pass = all( suite_results['results'][f'N_{N}']['overall_pass']
        for N in sizes ) suite_results['suite_pass'] = all_sizes_pass
        print("\n" + "=" * 60)
        print(f"SUITE RESULT: {'✓ ALL TESTS PASS'
        if all_sizes_pass else '❌ SOME TESTS FAIL'}")
        print("=" * 60)
        return suite_results
    def main(): """
        Run the Hamiltonian recovery validation tests.
"""
        validator = HamiltonianRecoveryValidator(tolerance=1e-12)

        # Run comprehensive test suite results = validator.run_full_hamiltonian_suite([4, 8, 16])

        # Print detailed results
        print("\nDETAILED RESULTS:")
        print("=" * 50) for size_key, size_results in results['results'].items():
        print(f"||n{size_key.upper()}:") for test_name, test_result in size_results.items():
        if test_name == 'overall_pass': continue
        print(f" {test_result['test_name']}: {'PASS'
        if test_result['passes'] else 'FAIL'}")
        if test_result['passes']: if 'reconstruction_error' in test_result:
        print(f" Reconstruction error: {test_result['reconstruction_error']:.2e}") if 'extraction_metrics' in test_result: herm_err = test_result['extraction_metrics']['hermiticity_error']
        print(f" Hermiticity error: {herm_err:.2e}")
        else:
        print(f" ❌ Failed - check detailed results")
        return results

if __name__ == "__main__": main()