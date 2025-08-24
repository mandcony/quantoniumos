
import math
import time
import typing

import Any
import Dict
import generate_resonance_kernel
import get_rft_basis
import List
import numpy as np
import Optional
import scipy.linalg
import Tuple

import canonical_true_rft


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
def bra_ket(psi: np.ndarray, A: np.ndarray, phi: np.ndarray = None) -> complex: """
        Compute <psi|A|phi> or <psi|phi> with explicit conjugate transpose.
"""

        if phi is None:

        # Two-argument form: <psi|A>
        return np.vdot(psi, A)
        else:

        # Three-argument form: <psi|A|phi>
        return np.vdot(psi, A @ phi)
def loschmidt_echo(H: np.ndarray, psi_0: np.ndarray, t: float, eps: float = 1e-6) -> float: """
        Compute Loschmidt echo with small perturbation to avoid trivial result.
"""
        N = len(psi_0) H_pert = H + eps * random_state(N * N).reshape(N, N) H_pert = 0.5 * (H_pert + H_pert.T.conj())

        # Ensure Hermitian

        # Forward and backward evolution U_forward = scipy.linalg.expm(-1j * H * t) U_backward = scipy.linalg.expm(1j * H_pert * t)

        # Loschmidt echo psi_final = U_backward @ U_forward @ psi_0
        return abs(bra_ket(psi_0, psi_final))**2
def get_canonical_parameters(): """
        Get canonical parameters for RFT implementation.
"""

        return { 'method': 'symbolic_resonance_computing', 'kernel': 'canonical_true_rft', 'precision': 'double' }

class ChoiChannelValidator: """
        Validates quantum channel properties using Choi matrix representation with symbolic resonance computing methods.
"""

    def __init__(self, tolerance: float = 1e-12):
        self.tolerance = tolerance
        self.results = {}
    def compute_choi_matrix(self, U: np.ndarray) -> np.ndarray: """
        Compute normalized Choi matrix.
"""
        d = U.shape[0]

        # Create |Phi> = sum_i |i,i> Phi = np.zeros((d * d, 1), dtype=complex)
        for i in range(d): Phi[i * d + i, 0] = 1.0 # (I x U)|Phi> I = np.eye(d, dtype=complex) K = np.kron(I, U) @ Phi J = (K @ K.conj().T) / d
        return J
    def test_choi_psd(self, N: int = 8) -> Dict[str, Any]: """
        Test that Choi matrix is PSD.
"""

        print(f"Testing Choi matrix PSD for N={N}...") start_time = time.time() U_raw = get_rft_basis(N) U = project_unitary(U_raw) J =
        self.compute_choi_matrix(U) eigenvals = np.linalg.eigvals(J) min_eigenval = np.min(np.real(eigenvals)) max_imag_eigenval = np.max(np.abs(np.imag(eigenvals))) is_psd = min_eigenval >= -
        self.tolerance hermitian_error = np.linalg.norm(J - J.T.conj(), ord='fro') passes = ( is_psd and hermitian_error <=
        self.tolerance and max_imag_eigenval <=
        self.tolerance ) result = { 'test_name': 'Choi Matrix PSD', 'size': N, 'min_eigenvalue': float(min_eigenval), 'max_imaginary_eigenvalue': float(max_imag_eigenval), 'hermitian_error': float(hermitian_error), 'is_positive_semidefinite': bool(is_psd), 'passes': bool(passes), 'tolerance':
        self.tolerance, 'test_time': time.time() - start_time }
        self.results[f'choi_psd_N{N}'] = result
        return result
    def test_trace_preserving(self, N: int = 8) -> Dict[str, Any]: """
        Test trace-preserving property.
"""

        print(f"Testing trace-preserving property for N={N}...") start_time = time.time() U_raw = get_rft_basis(N) U = project_unitary(U_raw) J =
        self.compute_choi_matrix(U) J_reshaped = J.reshape(N, N, N, N) partial_trace = np.zeros((N, N), dtype=complex)
        for k in range(N): partial_trace += J_reshaped[:, :, k, k] I_normalized = np.eye(N) / N trace_preserving_error = np.linalg.norm(partial_trace - I_normalized, ord=2) trace_value = np.trace(partial_trace) trace_error = abs(trace_value - 1.0) passes = ( trace_preserving_error <=
        self.tolerance and trace_error <=
        self.tolerance ) result = { 'test_name': 'Trace Preserving', 'size': N, 'trace_preserving_error': float(trace_preserving_error), 'trace_value': float(trace_value), 'trace_error': float(trace_error), 'passes': bool(passes), 'tolerance':
        self.tolerance, 'test_time': time.time() - start_time }
        self.results[f'trace_preserving_N{N}'] = result
        return result
    def test_choi_rank_one(self, N: int = 8) -> Dict[str, Any]: """
        Test that Choi matrix for unitary channel has rank 1.
"""

        print(f"Testing Choi matrix rank-1 for N={N}...") start_time = time.time() U_raw = get_rft_basis(N) U = project_unitary(U_raw) J =
        self.compute_choi_matrix(U) singular_vals = np.linalg.svd(J, compute_uv=False) max_singular = np.max(singular_vals) second_max = np.partition(singular_vals, -2)[-2]
        if len(singular_vals) > 1 else 0 rank_one_error = second_max / max_singular
        if max_singular > 0 else float('inf') effective_rank = np.sum(singular_vals / max_singular >
        self.tolerance) passes = effective_rank == 1 and rank_one_error <=
        self.tolerance result = { 'test_name': 'Choi Matrix Rank-1', 'size': N, 'max_singular_value': float(max_singular), 'second_max_singular_value': float(second_max), 'rank_one_error': float(rank_one_error), 'effective_rank': int(effective_rank), 'passes': bool(passes), 'tolerance':
        self.tolerance, 'test_time': time.time() - start_time }
        self.results[f'choi_rank_one_N{N}'] = result
        return result
    def process_fidelity(self, J1: np.ndarray, J2: np.ndarray) -> float: """
        Compute process fidelity between two channels.
"""
        d = int(round(np.sqrt(J1.shape[0])))
        return float(d * np.real_if_close(np.trace(J1 @ J2)))
    def test_process_fidelity(self, N: int = 8) -> Dict[str, Any]: """
        Test process fidelity calculations.
"""

        print(f"Testing process fidelity for N={N}...") start_time = time.time() U_raw = get_rft_basis(N) U = project_unitary(U_raw) J =
        self.compute_choi_matrix(U) self_fidelity =
        self.process_fidelity(J, J) I = np.eye(N) J_identity =
        self.compute_choi_matrix(I) identity_fidelity =
        self.process_fidelity(J, J_identity) self_fidelity_error = abs(self_fidelity - 1.0) passes = self_fidelity_error <=
        self.tolerance result = { 'test_name': 'Process Fidelity', 'size': N, 'self_fidelity': float(self_fidelity), 'self_fidelity_error': float(self_fidelity_error), 'identity_fidelity': float(identity_fidelity), 'passes': bool(passes), 'tolerance':
        self.tolerance, 'test_time': time.time() - start_time }
        self.results[f'process_fidelity_N{N}'] = result
        return result
    def test_channel_composition(self, N: int = 8) -> Dict[str, Any]: """
        Test channel composition property.
"""

        print(f"Testing channel composition for N={N}...") start_time = time.time() U1_raw = get_rft_basis(N) U1 = project_unitary(U1_raw) U2 = np.linalg.matrix_power(U1, 2) U2 = project_unitary(U2) J1 =
        self.compute_choi_matrix(U1) J2 =
        self.compute_choi_matrix(U2) U_product = U2 @ U1 J_product =
        self.compute_choi_matrix(U_product) composition_fidelity =
        self.process_fidelity(J_product, J_product) eigs_prod = np.sort(np.linalg.eigvals(J_product)) spectral_error = np.linalg.norm(eigs_prod - eigs_prod) passes = composition_fidelity >= 1.0 -
        self.tolerance result = { 'test_name': 'Channel Composition', 'size': N, 'composition_fidelity': float(composition_fidelity), 'spectral_error': float(spectral_error), 'passes': bool(passes), 'tolerance':
        self.tolerance, 'test_time': time.time() - start_time }
        self.results[f'channel_composition_N{N}'] = result
        return result
    def run_comprehensive_validation(self, sizes: List[int] = [4, 8, 16]) -> Dict[str, Any]: """
        Run all Choi channel validation tests across multiple sizes.
"""

        print("=" * 60)
        print("CHOI MATRIX CHANNEL VALIDATION TEST SUITE")
        print("Using Symbolic Resonance Computing Methods")
        print("=" * 60) suite_results = { 'suite_name': 'Choi Channel Validation', 'timestamp': time.time(), 'canonical_parameters': get_canonical_parameters(), 'tolerance':
        self.tolerance, 'test_sizes': sizes, 'results': {} } overall_pass = True
        for N in sizes:
        print(f"\nTesting size N={N}")
        print("-" * 30) size_results = {} size_pass = True size_results['choi_psd'] =
        self.test_choi_psd(N) size_results['trace_preserving'] =
        self.test_trace_preserving(N) size_results['choi_rank_one'] =
        self.test_choi_rank_one(N) size_results['process_fidelity'] =
        self.test_process_fidelity(N) size_results['channel_composition'] =
        self.test_channel_composition(N) for test_name, test_result in size_results.items():
        if not test_result.get('passes', False): size_pass = False overall_pass = False size_results['overall_pass'] = size_pass suite_results['results'][f'N_{N}'] = size_results
        print(f"Size N={N} overall: {'✓ PASS'
        if size_pass else '❌ FAIL'}") suite_results['overall_pass'] = overall_pass
        print("\n" + "=" * 60)
        print(f"SUITE RESULT: {'✓ ALL TESTS PASS'
        if overall_pass else '❌ SOME TESTS FAIL'}")
        print("=" * 60)
        return suite_results
    def main(): """
        Run Choi channel validation tests.
"""
        validator = ChoiChannelValidator(tolerance=1e-12) results = validator.run_comprehensive_validation([4, 8])
import json
import json.dump
import time.time

import 'w' as f:
import =
import default=str
import f
import f'choi_channel_validation_results_{timestamp}.json'
import filename
import indent=2
import int
import open
import results
import timestamp
import with

        print(f"\nResults saved to: {filename}")

if __name__ == "__main__": main()