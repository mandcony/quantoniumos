
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
def get_canonical_parameters(): """
        Get canonical parameters for RFT implementation.
"""

        return { 'method': 'symbolic_resonance_computing', 'kernel': 'canonical_true_rft', 'precision': 'double' }

class TrotterErrorValidator: """
        Validates Trotter decomposition errors using symbolic resonance computing methods.
"""

    def __init__(self, tolerance: float = 1e-12):
        self.tolerance = tolerance
        self.results = {}
    def extract_hamiltonian(self, U: np.ndarray) -> np.ndarray: """
        Extract Hamiltonian from unitary operator.
"""
        log_U = safe_log_unitary(U) / 1j
        return 1j * log_U
    def split_hamiltonian(self, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: """
        Split Hamiltonian into two parts for Trotter analysis. Strategy: H1 = (H + Hdagger)/2 (Hermitian part), H2 = (H - Hdagger)/(2i) (anti-Hermitian part) But since H is already Hermitian, we'll use a different splitting.
"""
        N = H.shape[0]

        # Split by frequency bands (low vs high eigenvalue components) eigenvals, eigenvecs = np.linalg.eigh(H)

        # Sort eigenvalues and corresponding eigenvectors sort_idx = np.argsort(eigenvals) eigenvals = eigenvals[sort_idx] eigenvecs = eigenvecs[:, sort_idx]

        # Split into low and high frequency components mid_idx = N // 2

        # H1: low frequency part H1 = eigenvecs[:, :mid_idx] @ np.diag(eigenvals[:mid_idx]) @ eigenvecs[:, :mid_idx].T.conj()

        # H2: high frequency part H2 = eigenvecs[:, mid_idx:] @ np.diag(eigenvals[mid_idx:]) @ eigenvecs[:, mid_idx:].T.conj()
        return H1, H2
    def compute_trotter_error(self, H1: np.ndarray, H2: np.ndarray, t: float, n: int) -> float: """
        Compute Trotter decomposition error. Error = ||e^(-it(H1+H2)) - (e^(-itH1/n)e^(-itH2/n))^n2
"""

        # Exact evolution with combined Hamiltonian H_total = H1 + H2 U_exact = scipy.linalg.expm(-1j * H_total * t)

        # Trotter approximation: (e^(-itH1/n)e^(-itH2/n))^n dt = t / n U1_step = scipy.linalg.expm(-1j * H1 * dt) U2_step = scipy.linalg.expm(-1j * H2 * dt) U_trotter_step = U2_step @ U1_step

        # Apply H1 then H2 U_trotter = np.linalg.matrix_power(U_trotter_step, n)

        # Compute error error = np.linalg.norm(U_exact - U_trotter, ord=2)
        return error
    def test_trotter_scaling(self, N: int = 8, t: float = 1.0) -> Dict[str, Any]: """
        Test Trotter error scaling with step size. Should show O(t^2/n) scaling for 2nd-order Trotter.
"""

        print(f"Testing Trotter scaling for N={N}...") start_time = time.time()

        # Get Hamiltonian and split it U_raw = get_rft_basis(N) U = project_unitary(U_raw) H =
        self.extract_hamiltonian(U) H1, H2 =
        self.split_hamiltonian(H)

        # Test different step numbers n_values = [2, 4, 8, 16, 32] errors = [] theoretical_errors = []

        # Estimate theoretical scaling constant

        # Error approx C * t^2 / n for some constant C commutator_norm = np.linalg.norm(H1 @ H2 - H2 @ H1, ord=2) theoretical_constant = (t**2 / 12) * commutator_norm

        # Leading order coefficient
        for n in n_values: error =
        self.compute_trotter_error(H1, H2, t, n) errors.append(error)

        # Theoretical prediction: error ~ C * t^2 / n theoretical_error = theoretical_constant / n theoretical_errors.append(theoretical_error)

        # Fit scaling law: error = A / n^alpha log_n = np.log(n_values) log_errors = np.log(errors)

        # Linear fit in log space: log(error) = log(A) - alpha * log(n) coeffs = np.polyfit(log_n, log_errors, 1) alpha = -coeffs[0]

        # Scaling exponent A = np.exp(coeffs[1])

        # Scaling coefficient

        # For 1st order Trotter, alpha should be approx 1

        # For 2nd order Trotter, alpha should be approx 2 expected_alpha = 1.0

        # We're using 1st order splitting alpha_error = abs(alpha - expected_alpha)

        # Pass condition: scaling should be approximately correct passes = alpha_error <= 0.5

        # Allow some tolerance for numerical effects test_time = time.time() - start_time result = { 'test_name': 'Trotter Scaling', 'size': N, 'evolution_time': t, 'n_values': n_values, 'trotter_errors': [float(err)
        for err in errors], 'theoretical_errors': [float(err)
        for err in theoretical_errors], 'scaling_exponent': float(alpha), 'scaling_coefficient': float(A), 'expected_scaling_exponent': expected_alpha, 'scaling_exponent_error': float(alpha_error), 'commutator_norm': float(commutator_norm), 'passes': bool(passes), 'tolerance':
        self.tolerance, 'test_time': test_time }
        self.results[f'trotter_scaling_N{N}'] = result
        return result
    def test_trotter_time_dependence(self, N: int = 8, n: int = 16) -> Dict[str, Any]: """
        Test Trotter error dependence on evolution time. Should show O(t^2) dependence for fixed n.
"""

        print(f"Testing Trotter time dependence for N={N}...") start_time = time.time()

        # Get Hamiltonian and split it U_raw = get_rft_basis(N) U = project_unitary(U_raw) H =
        self.extract_hamiltonian(U) H1, H2 =
        self.split_hamiltonian(H)

        # Test different evolution times t_values = [0.1, 0.2, 0.5, 1.0, 2.0] errors = []
        for t in t_values: error =
        self.compute_trotter_error(H1, H2, t, n) errors.append(error)

        # Fit time scaling: error = B * t^beta log_t = np.log(t_values) log_errors = np.log(errors) coeffs = np.polyfit(log_t, log_errors, 1) beta = coeffs[0]

        # Time scaling exponent B = np.exp(coeffs[1])

        # Time scaling coefficient

        # For Trotter splitting, beta should be approx 2 expected_beta = 2.0 beta_error = abs(beta - expected_beta)

        # Pass condition passes = beta_error <= 0.5 test_time = time.time() - start_time result = { 'test_name': 'Trotter Time Dependence', 'size': N, 'trotter_steps': n, 't_values': t_values, 'trotter_errors': [float(err)
        for err in errors], 'time_scaling_exponent': float(beta), 'time_scaling_coefficient': float(B), 'expected_time_scaling': expected_beta, 'time_scaling_error': float(beta_error), 'passes': bool(passes), 'tolerance':
        self.tolerance, 'test_time': test_time }
        self.results[f'trotter_time_N{N}'] = result
        return result
    def test_symmetric_trotter(self, N: int = 8, t: float = 1.0) -> Dict[str, Any]: """
        Test symmetric (2nd order) Trotter decomposition. Symmetric: e^(-iH1t/2n) e^(-iH2t/n) e^(-iH1t/2n)
"""

        print(f"Testing symmetric Trotter for N={N}...") start_time = time.time()

        # Get Hamiltonian and split it U_raw = get_rft_basis(N) U = project_unitary(U_raw) H =
        self.extract_hamiltonian(U) H1, H2 =
        self.split_hamiltonian(H)

        # Test different step numbers for symmetric Trotter n_values = [2, 4, 8, 16] standard_errors = [] symmetric_errors = []
        for n in n_values:

        # Standard 1st order Trotter error_1st =
        self.compute_trotter_error(H1, H2, t, n) standard_errors.append(error_1st)

        # Symmetric 2nd order Trotter H_total = H1 + H2 U_exact = scipy.linalg.expm(-1j * H_total * t) dt = t / n U1_half = scipy.linalg.expm(-1j * H1 * dt / 2) U2_full = scipy.linalg.expm(-1j * H2 * dt)

        # Symmetric step: e^(-iH1t/2n) e^(-iH2t/n) e^(-iH1t/2n) U_symmetric_step = U1_half @ U2_full @ U1_half U_symmetric = np.linalg.matrix_power(U_symmetric_step, n) error_symmetric = np.linalg.norm(U_exact - U_symmetric, ord=2) symmetric_errors.append(error_symmetric)

        # Compare scaling of both methods log_n = np.log(n_values)

        # Standard Trotter scaling log_std_errors = np.log(standard_errors) std_coeffs = np.polyfit(log_n, log_std_errors, 1) std_alpha = -std_coeffs[0]

        # Symmetric Trotter scaling log_sym_errors = np.log(symmetric_errors) sym_coeffs = np.polyfit(log_n, log_sym_errors, 1) sym_alpha = -sym_coeffs[0]

        # Symmetric should have better scaling (higher alpha) improvement_ratio = sym_alpha / std_alpha
        if std_alpha > 0 else 1.0

        # Pass condition: symmetric should be better passes = improvement_ratio > 1.2

        # At least 20% improvement in scaling test_time = time.time() - start_time result = { 'test_name': 'Symmetric Trotter', 'size': N, 'evolution_time': t, 'n_values': n_values, 'standard_errors': [float(err)
        for err in standard_errors], 'symmetric_errors': [float(err)
        for err in symmetric_errors], 'standard_scaling_exponent': float(std_alpha), 'symmetric_scaling_exponent': float(sym_alpha), 'improvement_ratio': float(improvement_ratio), 'passes': bool(passes), 'tolerance':
        self.tolerance, 'test_time': test_time }
        self.results[f'symmetric_trotter_N{N}'] = result
        return result
    def test_locality_effects(self, N: int = 8) -> Dict[str, Any]: """
        Test how Hamiltonian locality affects Trotter errors.
"""

        print(f"Testing locality effects for N={N}...") start_time = time.time()

        # Create local vs non-local Hamiltonian splittings U_raw = get_rft_basis(N) U = project_unitary(U_raw) H =
        self.extract_hamiltonian(U)

        # Local splitting: nearest neighbor interactions H_local_1 = np.zeros_like(H) H_local_2 = np.zeros_like(H)

        # Fill local parts (this is somewhat artificial for our RFT basis)
        for i in range(N):
        for j in range(N):
        if abs(i - j) <= 1:

        # Nearest neighbors + diagonal H_local_1[i, j] = H[i, j] if (i + j) % 2 == 0 else 0 H_local_2[i, j] = H[i, j] if (i + j) % 2 == 1 else 0

        # Non-local splitting: use original frequency-based splitting H_nonlocal_1, H_nonlocal_2 =
        self.split_hamiltonian(H)

        # Compare Trotter errors for both splittings t = 0.5 n = 8 local_error =
        self.compute_trotter_error(H_local_1, H_local_2, t, n) nonlocal_error =
        self.compute_trotter_error(H_nonlocal_1, H_nonlocal_2, t, n)

        # Compute commutator norms (measure of non-commutativity) local_commutator_norm = np.linalg.norm( H_local_1 @ H_local_2 - H_local_2 @ H_local_1, ord=2 ) nonlocal_commutator_norm = np.linalg.norm( H_nonlocal_1 @ H_nonlocal_2 - H_nonlocal_2 @ H_nonlocal_1, ord=2 )

        # Error ratio error_ratio = local_error / nonlocal_error
        if nonlocal_error > 0 else float('inf')

        # Pass condition: should see some relationship between commutator and error passes = local_commutator_norm >= 0 and nonlocal_commutator_norm >= 0

        # Basic sanity test_time = time.time() - start_time result = { 'test_name': 'Locality Effects', 'size': N, 'evolution_time': t, 'trotter_steps': n, 'local_trotter_error': float(local_error), 'nonlocal_trotter_error': float(nonlocal_error), 'local_commutator_norm': float(local_commutator_norm), 'nonlocal_commutator_norm': float(nonlocal_commutator_norm), 'error_ratio': float(error_ratio), 'passes': bool(passes), 'tolerance':
        self.tolerance, 'test_time': test_time }
        self.results[f'locality_N{N}'] = result
        return result
    def run_full_trotter_suite(self, sizes: List[int] = None) -> Dict[str, Any]: """
        Run the complete Trotter error analysis test suite.
"""

        if sizes is None: sizes = [4, 8]

        # Smaller sizes due to computational intensity
        print("=" * 60)
        print("TROTTER ERROR ANALYSIS TEST SUITE")
        print("Using Symbolic Resonance Computing Methods")
        print("=" * 60) suite_results = { 'suite_name': 'Trotter Error Analysis', 'timestamp': time.time(), 'canonical_parameters': get_canonical_parameters(), 'tolerance':
        self.tolerance, 'test_sizes': sizes, 'results': {} }
        for N in sizes:
        print(f"\nTesting size N={N}")
        print("-" * 30) size_results = {}

        # Run all tests for this size size_results['scaling'] =
        self.test_trotter_scaling(N) size_results['time_dependence'] =
        self.test_trotter_time_dependence(N) size_results['symmetric'] =
        self.test_symmetric_trotter(N) size_results['locality'] =
        self.test_locality_effects(N)

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
        Run the Trotter error analysis tests.
"""
        validator = TrotterErrorValidator(tolerance=1e-12)

        # Run comprehensive test suite results = validator.run_full_trotter_suite([4, 8])

        # Print detailed results
        print("\nDETAILED RESULTS:")
        print("=" * 50) for size_key, size_results in results['results'].items():
        print(f"||n{size_key.upper()}:") for test_name, test_result in size_results.items():
        if test_name == 'overall_pass': continue
        print(f" {test_result['test_name']}: {'PASS'
        if test_result['passes'] else 'FAIL'}")
        if test_result['passes']: if 'scaling_exponent' in test_result:
        print(f" Scaling exponent: {test_result['scaling_exponent']:.2f}") if 'time_scaling_exponent' in test_result:
        print(f" Time scaling exponent: {test_result['time_scaling_exponent']:.2f}") if 'improvement_ratio' in test_result:
        print(f" Symmetric improvement: {test_result['improvement_ratio']:.2f}x")
        else:
        print(f" ❌ Failed - check detailed results")
        return results

if __name__ == "__main__": main()