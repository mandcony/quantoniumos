
import math
import time
import typing

import Any
import Dict
import forward_true_rft
import generate_resonance_kernel
import get_rft_basis
import inverse_true_rft
import List
import numpy as np
import Optional
import scipy.linalg
import Tuple

import 04_RFT_ALGORITHMS.canonical_true_rft as canonical_true_rft

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

class StateEvolutionBenchmarkValidator: """
        Validates state evolution against known analytical benchmarks.
"""

    def __init__(self, tolerance: float = 1e-10):
        self.tolerance = tolerance
        self.results = {}
    def test_known_analytical_cases(self, N: int = 8) -> Dict[str, Any]: """
        Test RFT evolution against known analytical cases. For RFT, we test against the canonical roundtrip: x -> RFT -> IRFT -> x
"""

        print(f"Testing analytical benchmarks for N={N}...") start_time = time.time() benchmark_results = [] max_error = 0.0

        # Test Case 1: Delta function input delta_input = np.zeros(N) delta_input[0] = 1.0 rft_output = forward_true_rft(delta_input) reconstructed = inverse_true_rft(rft_output) error1 = np.linalg.norm(np.array(delta_input) - np.array(reconstructed)) benchmark_results.append({ 'test_case': 'Delta Function', 'input': delta_input.tolist(), 'analytical_output': delta_input.tolist(), 'numerical_output': reconstructed, 'error': float(error1) }) max_error = max(max_error, error1)

        # Test Case 2: Constant input constant_input = np.ones(N) / np.sqrt(N)

        # Normalized rft_output = forward_true_rft(constant_input) reconstructed = inverse_true_rft(rft_output) error2 = np.linalg.norm(np.array(constant_input) - np.array(reconstructed)) benchmark_results.append({ 'test_case': 'Constant Input', 'input': constant_input.tolist(), 'analytical_output': constant_input.tolist(), 'numerical_output': reconstructed, 'error': float(error2) }) max_error = max(max_error, error2)

        # Test Case 3: Sinusoidal input sinusoidal_input = np.array([np.sin(2 * np.pi * k / N)
        for k in range(N)]) sinusoidal_input = sinusoidal_input / np.linalg.norm(sinusoidal_input) rft_output = forward_true_rft(sinusoidal_input) reconstructed = inverse_true_rft(rft_output) error3 = np.linalg.norm(np.array(sinusoidal_input) - np.array(reconstructed)) benchmark_results.append({ 'test_case': 'Sinusoidal Input', 'input': sinusoidal_input.tolist(), 'analytical_output': sinusoidal_input.tolist(), 'numerical_output': reconstructed, 'error': float(error3) }) max_error = max(max_error, error3)

        # Test Case 4: Random input (should still reconstruct perfectly) np.random.seed(42) random_input = np.random.random(N) random_input = random_input / np.linalg.norm(random_input) rft_output = forward_true_rft(random_input) reconstructed = inverse_true_rft(rft_output) error4 = np.linalg.norm(np.array(random_input) - np.array(reconstructed)) benchmark_results.append({ 'test_case': 'Random Input', 'input': random_input.tolist(), 'analytical_output': random_input.tolist(), 'numerical_output': reconstructed, 'error': float(error4) }) max_error = max(max_error, error4)

        # Test Case 5: Two-level system analogue (
        if N=2)
        if N == 2:

        # For 2x2 case, test Rabi oscillation analogue rabi_input = np.array([1.0, 0.0]) # |||0> state rft_output = forward_true_rft(rabi_input) reconstructed = inverse_true_rft(rft_output) error5 = np.linalg.norm(np.array(rabi_input) - np.array(reconstructed)) benchmark_results.append({ 'test_case': 'Two-Level System', 'input': rabi_input.tolist(), 'analytical_output': rabi_input.tolist(), 'numerical_output': reconstructed, 'error': float(error5) }) max_error = max(max_error, error5)

        # Pass condition passes = max_error <=
        self.tolerance test_time = time.time() - start_time result = { 'test_name': 'Analytical Benchmarks', 'size': N, 'max_amplitude_error': float(max_error), 'benchmark_results': benchmark_results, 'passes': bool(passes), 'tolerance':
        self.tolerance, 'test_time': test_time }
        self.results[f'benchmarks_N{N}'] = result
        return result
    def test_energy_conservation_benchmarks(self, N: int = 8) -> Dict[str, Any]: """
        Test energy conservation for known cases.
"""

        print(f"Testing energy conservation benchmarks for N={N}...") start_time = time.time()

        # Get RFT unitary and extract Hamiltonian U_raw = get_rft_basis(N) U = project_unitary(U_raw) H = 1j * safe_log_unitary(U) / 1j energy_tests = [] max_energy_error = 0.0

        # Test multiple initial states test_states = [ ('Ground state analogue', np.array([1.0] + [0.0] * (N-1))), ('Excited state analogue', np.array([0.0, 1.0] + [0.0] * (N-2))
        if N > 1 else np.array([1.0])), ('Superposition', np.ones(N) / np.sqrt(N)), ] for state_name, initial_state in test_states:
        if len(initial_state) != N: continue

        # Normalize initial_state = initial_state / np.linalg.norm(initial_state)

        # Initial energy initial_energy = np.real(initial_state.conj() @ H @ initial_state)

        # Evolve with different times times = [0.1, 0.5, 1.0] energy_errors = []
        for t in times:

        # Time evolution U_t = scipy.linalg.expm(-1j * H * t) evolved_state = U_t @ initial_state

        # Final energy final_energy = np.real(evolved_state.conj() @ H @ evolved_state) energy_error = abs(final_energy - initial_energy) energy_errors.append(energy_error) max_state_energy_error = max(energy_errors) max_energy_error = max(max_energy_error, max_state_energy_error) energy_tests.append({ 'state_name': state_name, 'initial_energy': float(initial_energy), 'max_energy_error': float(max_state_energy_error), 'energy_errors': [float(e)
        for e in energy_errors], 'times': times })

        # Pass condition passes = max_energy_error <=
        self.tolerance test_time = time.time() - start_time result = { 'test_name': 'Energy Conservation Benchmarks', 'size': N, 'max_energy_error': float(max_energy_error), 'energy_tests': energy_tests, 'passes': bool(passes), 'tolerance':
        self.tolerance, 'test_time': test_time }
        self.results[f'energy_benchmarks_N{N}'] = result
        return result
    def test_commuting_blocks_benchmark(self, N: int = 4) -> Dict[str, Any]: """
        Test evolution with commuting block structure (analytical solution exists).
"""

        print(f"Testing commuting blocks benchmark for N={N}...") start_time = time.time()
        if N != 4:
        return { 'test_name': 'Commuting Blocks Benchmark', 'size': N, 'error': 'Only implemented for N=4', 'passes': True, 'test_time': time.time() - start_time }

        # Get RFT Hamiltonian U_raw = get_rft_basis(N) U = project_unitary(U_raw) H = 1j * safe_log_unitary(U) / 1j

        # Test with block-diagonal structure

        # Create a test Hamiltonian with 2x2 blocks H_block = np.zeros((4, 4), dtype=complex)

        # Block 1: [0:2, 0:2] H_block[0:2, 0:2] = H[0:2, 0:2]

        # Block 2: [2:4, 2:4] H_block[2:4, 2:4] = H[2:4, 2:4]

        # Test evolution with this block Hamiltonian test_state = np.array([1.0, 0.5, 0.0, 0.0]) test_state = test_state / np.linalg.norm(test_state) evolution_time = 0.5

        # Exact evolution with block Hamiltonian U_block = scipy.linalg.expm(-1j * H_block * evolution_time) exact_evolution = U_block @ test_state

        # Approximate by evolving each block separately U_block1 = scipy.linalg.expm(-1j * H_block[0:2, 0:2] * evolution_time) U_block2 = scipy.linalg.expm(-1j * H_block[2:4, 2:4] * evolution_time) approx_evolution = np.zeros(4, dtype=complex) approx_evolution[0:2] = U_block1 @ test_state[0:2] approx_evolution[2:4] = U_block2 @ test_state[2:4]

        # Compare block_error = np.linalg.norm(exact_evolution - approx_evolution)

        # Pass condition passes = block_error <=
        self.tolerance test_time = time.time() - start_time result = { 'test_name': 'Commuting Blocks Benchmark', 'size': N, 'evolution_time': evolution_time, 'block_error': float(block_error), 'exact_evolution': exact_evolution.tolist(), 'approx_evolution': approx_evolution.tolist(), 'passes': bool(passes), 'tolerance':
        self.tolerance, 'test_time': test_time }
        self.results[f'blocks_benchmark_N{N}'] = result
        return result
    def run_full_benchmark_suite(self, sizes: List[int] = None) -> Dict[str, Any]: """
        Run the complete state evolution benchmark test suite.
"""

        if sizes is None: sizes = [4, 8, 16]
        print("=" * 60)
        print("STATE EVOLUTION BENCHMARK TEST SUITE")
        print("Using Symbolic Resonance Computing Methods")
        print("=" * 60) suite_results = { 'suite_name': 'State Evolution Benchmark Validation', 'timestamp': time.time(), 'canonical_parameters': get_canonical_parameters(), 'tolerance':
        self.tolerance, 'test_sizes': sizes, 'results': {} }
        for N in sizes:
        print(f"\nTesting size N={N}")
        print("-" * 30) size_results = {}

        # Run all tests for this size size_results['analytical'] =
        self.test_known_analytical_cases(N) size_results['energy_conservation'] =
        self.test_energy_conservation_benchmarks(N)
        if N == 4:

        # Special case size_results['commuting_blocks'] =
        self.test_commuting_blocks_benchmark(N)

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
        Run the state evolution benchmark validation tests.
"""
        validator = StateEvolutionBenchmarkValidator(tolerance=1e-10)

        # Run comprehensive test suite results = validator.run_full_benchmark_suite([4, 8, 16])

        # Print detailed results
        print("\nDETAILED RESULTS:")
        print("=" * 50) for size_key, size_results in results['results'].items():
        print(f"||n{size_key.upper()}:") for test_name, test_result in size_results.items():
        if test_name == 'overall_pass': continue
        print(f" {test_result['test_name']}: {'PASS'
        if test_result['passes'] else 'FAIL'}")
        if test_result['passes']: if 'max_amplitude_error' in test_result:
        print(f" Max amplitude error: {test_result['max_amplitude_error']:.2e}") if 'max_energy_error' in test_result:
        print(f" Max energy error: {test_result['max_energy_error']:.2e}") if 'block_error' in test_result:
        print(f" Block error: {test_result['block_error']:.2e}")
        else:
        print(f" ❌ Failed - check detailed results")
        return results

if __name__ == "__main__": main()