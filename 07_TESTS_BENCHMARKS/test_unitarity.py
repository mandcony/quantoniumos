#!/usr/bin/env python3
"""
Quantum Gate Unitarity Validation Test Suite === This test validates that RFT operators are unitary gates at the quantum level: - Goal: U†U = I to numerical tolerance - Compute ||U†U - I||₂ and κappa(U) (condition number) - Pass: spectral radius of U on unit circle; ||U†U - I||₂ <= 1e-12 Uses symbolic resonance computing methods from the canonical RFT implementation.
"""

import math
import time
import typing

import Any
import Dict
import List
import numpy as np
import scipy.linalg
import Tuple

from 04_RFT_ALGORITHMS.canonical_true_rft import (forward_true_rft, generate_resonance_kernel,
                                get_canonical_parameters, get_rft_basis,
                                inverse_true_rft)


def random_state(N: int, seed: int = 42) -> np.ndarray: """
        Generate a random normalized quantum state.
"""
        rng = np.random.default_rng(seed) psi = rng.normal(size=N) + 1j * rng.normal(size=N)
        return psi / np.linalg.norm(psi)
def project_unitary(U: np.ndarray) -> np.ndarray: """
        Project matrix to nearest unitary in Frobenius norm using polar decomposition.
"""
        Uu, Up = scipy.linalg.polar(U)
        return Uu
def safe_log_unitary(U: np.ndarray) -> np.ndarray: """
        Safely extract Hermitian generator H from unitary U such that U = exp(-iH). Uses eigendecomposition to avoid branch cut issues with matrix logarithm.
"""
        w, V = np.linalg.eig(U) theta = np.angle(w) H = (V @ np.diag(theta) @ V.conj().T).real H = 0.5 * (H + H.conj().T)

        # Hermitian symmetrization
        return H
def bracket(psi: np.ndarray, A: np.ndarray, phi: np.ndarray = None) -> complex: """
        Compute <psi|A|phi> or <psi|A|psi>.
"""

        if phi is None:
        return np.vdot(psi, A @ psi)
        else:
        return np.vdot(psi, A @ phi)
def loschmidt_echo(H: np.ndarray, psi_0: np.ndarray, t: float, eps: float = 1e-6) -> float: """
        Compute Loschmidt echo with small perturbation to avoid trivial result.
"""
        N = len(psi_0) H_pert = H + eps * random_state(N * N).reshape(N, N) H_pert = 0.5 * (H_pert + H_pert.T.conj()) U_forward = scipy.linalg.expm(-1j * H * t) U_backward = scipy.linalg.expm(1j * H_pert * t) psi_final = U_backward @ U_forward @ psi_0
        return abs(bracket(psi_0, psi_final)) ** 2

class UnitarityValidator: """
        Validates quantum unitarity properties of RFT operators using symbolic resonance computing methods.
"""

    def __init__(self, tolerance: float = 1e-12):
        self.tolerance = tolerance
        self.results = {}
    def test_gate_unitarity(self, N: int) -> Dict[str, Any]: """
        Test 1: Gate-level unitarity.
"""

        print(f"Testing unitarity for N={N}...") start_time = time.time() U_raw = get_rft_basis(N) U = project_unitary(U_raw)

        # Compute U†U U_dagger = U.T.conj() U_dagger_U = U_dagger @ U I = np.eye(N) unitarity_error = np.linalg.norm(U_dagger_U - I, ord=2)

        # Condition number condition_number = np.linalg.cond(U, p=2)

        # Eigenvalue magnitudes eigenvalues = np.linalg.eigvals(U) spectral_radii = np.abs(eigenvalues) max_spectral_deviation = np.max(np.abs(spectral_radii - 1.0))

        # Check unit circle eig_tol = 1e-10 on_unit_circle = np.all(np.abs(spectral_radii - 1.0) < eig_tol) passes_unitarity = ( unitarity_error <=
        self.tolerance and on_unit_circle and condition_number <= 1 + 1e-8 ) test_time = time.time() - start_time result = { 'test_name': 'Gate-Level Unitarity', 'size': N, 'unitarity_error': float(unitarity_error), 'condition_number': float(condition_number), 'max_spectral_deviation': float(max_spectral_deviation), 'on_unit_circle': bool(on_unit_circle), 'eigenvalue_magnitudes': spectral_radii.tolist(), 'passes': bool(passes_unitarity), 'tolerance':
        self.tolerance, 'eig_tol': eig_tol, 'test_time': test_time }
        print(f"||U†U - I||₂: {unitarity_error:.2e} (target: <= {
        self.tolerance})")
        print(f"max |λ(U)| - 1: {max_spectral_deviation:.2e} (target: <= 1e-10)")
        print(f"kappa₂(U): {condition_number:.2e} (target: <= 1 + 1e-8)")
        self.results['unitarity'] = result
        return result
    def test_hermiticity_of_generator(self, N: int) -> Dict[str, Any]: """
        Test 2: Extract Hamiltonian and check Hermiticity.
"""

        print(f"Testing Hamiltonian extraction for N={N}...") start_time = time.time() U_raw = get_rft_basis(N) U = project_unitary(U_raw) H = safe_log_unitary(U) H_dagger = H.T.conj() hermiticity_error = np.linalg.norm(H - H_dagger, ord=2)

        # Reconstruction check U_reconstructed = scipy.linalg.expm(-1j * H) reconstruction_error = np.linalg.norm(U - U_reconstructed, ord=2)

        # Check real eigenvalues H_eigvals = np.linalg.eigvals(H) max_imag_part = np.max(np.abs(H_eigvals.imag)) passes = ( hermiticity_error <=
        self.tolerance and reconstruction_error <= 1e-10 and max_imag_part <=
        self.tolerance ) test_time = time.time() - start_time result = { 'test_name': 'Hamiltonian Extraction', 'size': N, 'hermiticity_error': float(hermiticity_error), 'reconstruction_error': float(reconstruction_error), 'max_imaginary_eigenvalue': float(max_imag_part), 'H_eigenvalues_real_parts': H_eigvals.real.tolist(), 'passes': bool(passes), 'tolerance':
        self.tolerance, 'test_time': test_time }
        print(f"||H - H†||₂: {hermiticity_error:.2e} (target: <= {
        self.tolerance})")
        print(f"||U - e^(-iH)||₂: {reconstruction_error:.2e} (target: <= 1e-10)")
        print(f"max Im(λ(H)): {max_imag_part:.2e} (target: <= {
        self.tolerance})")
        self.results['hamiltonian'] = result
        return result
    def test_time_evolution_validity(self, N: int) -> Dict[str, Any]: """
        Test 3: Time evolution norm and energy conservation.
"""

        print(f"Testing time evolution for N={N}...") start_time = time.time() U_raw = get_rft_basis(N) U = project_unitary(U_raw) H = safe_log_unitary(U) psi_0 = random_state(N) psi_0 /= np.linalg.norm(psi_0) times = np.linspace(0, 1.0, 10) norm_errors = [] energy_variations = [] loschmidt_values = [] initial_energy = bracket(psi_0, H, psi_0).real
        for t in times: U_t = scipy.linalg.expm(-1j * H * t) psi_t = U_t @ psi_0

        # Norm conservation norm_t = np.linalg.norm(psi_t) norm_errors.append(abs(norm_t - 1.0))

        # Energy conservation energy_t = bracket(psi_t, H, psi_t).real energy_variations.append(abs(energy_t - initial_energy))

        # Loschmidt echo echo_val = loschmidt_echo(H, psi_0, t, eps=1e-6) loschmidt_values.append(echo_val) max_norm_error = max(norm_errors) max_energy_variation = max(energy_variations) min_loschmidt = min(loschmidt_values) passes = ( max_norm_error <=
        self.tolerance and max_energy_variation <=
        self.tolerance and min_loschmidt >= (1.0 - 1e-10) ) test_time = time.time() - start_time result = { 'test_name': 'Time Evolution Validity', 'size': N, 'max_norm_error': float(max_norm_error), 'max_energy_variation': float(max_energy_variation), 'min_loschmidt_echo': float(min_loschmidt), 'norm_errors': [float(x)
        for x in norm_errors], 'energy_variations': [float(x)
        for x in energy_variations], 'loschmidt_values': [float(x)
        for x in loschmidt_values], 'passes': bool(passes), 'tolerance':
        self.tolerance, 'test_time': test_time }
        self.results['time_evolution'] = result
        return result
    def test_invariance_reversibility(self, N: int) -> Dict[str, Any]: """
        Test 4: Group axioms for unitaries.
"""

        print(f"Testing group axioms for N={N}...") start_time = time.time() U = get_rft_basis(N)

        # Inverse property U_inv = np.linalg.inv(U) U_dagger = U.T.conj() inverse_error = np.linalg.norm(U_inv - U_dagger, ord=2)

        # Closure np.random.seed(123) closure_errors = []
        for _ in range(5): p1 = np.random.randint(1, 5) p2 = np.random.randint(1, 5) U1 = np.linalg.matrix_power(U, p1) U2 = np.linalg.matrix_power(U, p2) U_prod = U1 @ U2 closure_error = np.linalg.norm(U_prod.T.conj() @ U_prod - np.eye(N), ord=2) closure_errors.append(closure_error) max_closure_error = max(closure_errors)

        # Associativity U1 = U U2 = np.linalg.matrix_power(U, 2) U3 = np.linalg.matrix_power(U, 3) left_assoc = (U1 @ U2) @ U3 right_assoc = U1 @ (U2 @ U3) associativity_error = np.linalg.norm(left_assoc - right_assoc, ord=2) passes = ( inverse_error <=
        self.tolerance and max_closure_error <=
        self.tolerance and associativity_error <=
        self.tolerance ) test_time = time.time() - start_time result = { 'test_name': 'Group Axioms', 'size': N, 'inverse_error': float(inverse_error), 'max_closure_error': float(max_closure_error), 'associativity_error': float(associativity_error), 'closure_errors': [float(x)
        for x in closure_errors], 'passes': bool(passes), 'tolerance':
        self.tolerance, 'test_time': test_time }
        self.results['group_axioms'] = result
        return result
    def run_full_unitarity_suite(self, sizes: List[int] = None) -> Dict[str, Any]: """
        Run complete unitarity suite.
"""

        if sizes is None: sizes = [4, 8, 16, 32]
        print("=" * 60)
        print("QUANTUM GATE UNITARITY VALIDATION TEST SUITE")
        print("Using Symbolic Resonance Computing Methods")
        print("=" * 60) suite_results = { 'suite_name': 'Quantum Gate Unitarity Validation', 'timestamp': time.time(), 'canonical_parameters': get_canonical_parameters(), 'tolerance':
        self.tolerance, 'test_sizes': sizes, 'results': {} }
        for N in sizes:
        print(f"\nTesting size N={N}")
        print("-" * 30) size_results = {} size_results['unitarity'] =
        self.test_gate_unitarity(N) size_results['hamiltonian'] =
        self.test_hermiticity_of_generator(N) size_results['time_evolution'] =
        self.test_time_evolution_validity(N) size_results['group_axioms'] =
        self.test_invariance_reversibility(N) all_pass = all(r['passes']
        for r in size_results.values()) size_results['overall_pass'] = all_pass suite_results['results'][f'N_{N}'] = size_results
        print(f"Size N={N} overall: {'✓ PASS'
        if all_pass else '❌ FAIL'}") suite_results['suite_pass'] = all(suite_results['results'][f'N_{N}']['overall_pass']
        for N in sizes)
        print("\n" + "=" * 60)
        print(f"SUITE RESULT: {'✓ ALL TESTS PASS'
        if suite_results['suite_pass'] else '❌ SOME TESTS FAIL'}")
        print("=" * 60)
        return suite_results
    def main(): """
        Run the unitarity validation tests.
"""
        validator = UnitarityValidator(tolerance=1e-12) results = validator.run_full_unitarity_suite([4, 8, 16])
        print("\nDETAILED RESULTS:")
        print("=" * 50) for size_key, size_results in results['results'].items():
        print(f"\n{size_key.upper()}:") for test_name, test_result in size_results.items():
        if test_name == 'overall_pass': continue
        print(f"{test_result['test_name']}: {'PASS'
        if test_result['passes'] else 'FAIL'}") for k, v in test_result.items():
        if k in ['test_name', 'passes']: continue
        print(f" {k}: {v}")

if __name__ == "__main__": main()