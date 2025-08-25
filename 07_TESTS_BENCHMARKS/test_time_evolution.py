#!/usr/bin/env python3
"""
Time Evolution Validation Test Suite === This test validates time evolution consistency and conservation laws: - Norm conservation: ||psi(t)||2 constant to machine precision - Energy conservation: <H>ₜ constant (
if H time-invariant) - Loschmidt echo: L(t) = |<psi0|Udagger(t)U(t)|psi0>| approx 1 Uses symbolic resonance computing methods from the canonical RFT implementation.
"""

import math
import time
import typing
import Any
import Dict
import generate_resonance_kernel
import get_rft_basis
import List
import numpy as np
import scipy.linalg
import Tuple
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), '04_RFT_ALGORITHMS'))
import canonical_true_rft as canonical_true_rft

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
import time
import typing

import Any
import Dict
import List
import matplotlib.pyplot as plt
import Optional
import Tuple

from 04_RFT_ALGORITHMS.canonical_true_rft import (generate_resonance_kernel,
                                get_canonical_parameters, get_rft_basis)


class TimeEvolutionValidator: """
        Validates time evolution properties using symbolic resonance computing methods.
"""

    def __init__(self, tolerance: float = 1e-12):
        self.tolerance = tolerance
        self.results = {}
    def extract_hamiltonian(self, U: np.ndarray) -> np.ndarray: """
        Extract Hamiltonian from unitary operator.
"""
        log_U = safe_log_unitary(U) / 1j
        return 1j * log_U
    def time_evolve_state(self, H: np.ndarray, psi_0: np.ndarray, times: np.ndarray) -> List[np.ndarray]: """
        Time evolve state using Hamiltonian. psi(t) = e^(-iHt) psi0
"""
        evolved_states = []
        for t in times: U_t = scipy.linalg.expm(-1j * H * t) psi_t = U_t @ psi_0 evolved_states.append(psi_t)
        return evolved_states
    def test_norm_conservation(self, N: int = 8, max_time: float = 2.0, num_points: int = 20) -> Dict[str, Any]: """
        Test norm conservation during time evolution. For unitary evolution, ||psi(t)||2 should remain constant.
"""

        print(f"Testing norm conservation for N={N}...") start_time = time.time()

        # Get Hamiltonian U_raw = get_rft_basis(N) U = project_unitary(U_raw) H =
        self.extract_hamiltonian(U)

        # Create normalized random initial state np.random.seed(42)

        # Reproducible psi_0 = random_state(N) + 1j * np.random.random(N) psi_0 = psi_0 / np.linalg.norm(psi_0)

        # Time evolution times = np.linspace(0, max_time, num_points) evolved_states =
        self.time_evolve_state(H, psi_0, times)

        # Check norm conservation norms = [np.linalg.norm(psi)
        for psi in evolved_states] norm_deviations = [abs(norm - 1.0)
        for norm in norms] max_norm_deviation = max(norm_deviations) avg_norm_deviation = np.mean(norm_deviations)

        # Pass condition passes = max_norm_deviation <=
        self.tolerance test_time = time.time() - start_time result = { 'test_name': 'Norm Conservation', 'size': N, 'max_time': max_time, 'num_time_points': num_points, 'max_norm_deviation': float(max_norm_deviation), 'avg_norm_deviation': float(avg_norm_deviation), 'norms': [float(n)
        for n in norms], 'times': times.tolist(), 'passes': bool(passes), 'tolerance':
        self.tolerance, 'test_time': test_time }
        self.results[f'norm_conservation_N{N}'] = result
        return result
    def test_energy_conservation(self, N: int = 8, max_time: float = 2.0, num_points: int = 20) -> Dict[str, Any]: """
        Test energy conservation during time evolution. For time-independent H, <H>ₜ = <psi(t)|H|psi(t)> should remain constant.
"""

        print(f"Testing energy conservation for N={N}...") start_time = time.time()

        # Get Hamiltonian U_raw = get_rft_basis(N) U = project_unitary(U_raw) H =
        self.extract_hamiltonian(U)

        # Create normalized random initial state np.random.seed(43)

        # Different seed for different test psi_0 = random_state(N) + 1j * np.random.random(N) psi_0 = psi_0 / np.linalg.norm(psi_0)

        # Initial energy initial_energy = np.real(psi_0.conj() @ H @ psi_0)

        # Time evolution times = np.linspace(0, max_time, num_points) evolved_states =
        self.time_evolve_state(H, psi_0, times)

        # Check energy conservation energies = []
        for psi_t in evolved_states: energy_t = np.real(psi_t.conj() @ H @ psi_t) energies.append(energy_t) energy_deviations = [abs(E - initial_energy)
        for E in energies] max_energy_deviation = max(energy_deviations) avg_energy_deviation = np.mean(energy_deviations)

        # Pass condition passes = max_energy_deviation <=
        self.tolerance test_time = time.time() - start_time result = { 'test_name': 'Energy Conservation', 'size': N, 'max_time': max_time, 'num_time_points': num_points, 'initial_energy': float(initial_energy), 'max_energy_deviation': float(max_energy_deviation), 'avg_energy_deviation': float(avg_energy_deviation), 'energies': [float(E)
        for E in energies], 'times': times.tolist(), 'passes': bool(passes), 'tolerance':
        self.tolerance, 'test_time': test_time }
        self.results[f'energy_conservation_N{N}'] = result
        return result
    def test_loschmidt_echo(self, N: int = 8, max_time: float = 1.0, num_points: int = 15) -> Dict[str, Any]: """
        Test Loschmidt echo during time evolution. L(t) = |<psi0|Udagger(t)U(t)|psi0>| = |<psi0|psi0>| = 1 (since UdaggerU = I)
"""

        print(f"Testing Loschmidt echo for N={N}...") start_time = time.time()

        # Get Hamiltonian U_raw = get_rft_basis(N) U = project_unitary(U_raw) H =
        self.extract_hamiltonian(U)

        # Create normalized random initial state np.random.seed(44) psi_0 = random_state(N) + 1j * np.random.random(N) psi_0 = psi_0 / np.linalg.norm(psi_0)

        # Time evolution and Loschmidt echo calculation times = np.linspace(0, max_time, num_points) loschmidt_echoes = []
        for t in times: U_t = scipy.linalg.expm(-1j * H * t) U_t_dagger = U_t.T.conj()

        # L(t) = |<psi0|Udagger(t)U(t)|psi0>|

        # Since U is unitary, Udagger(t)U(t) = I, so L(t) = |<psi0|psi0> = 1 echo = loschmidt_echo(psi_0, U_t, H) loschmidt_echoes.append(echo) echo_deviations = [abs(echo - 1.0)
        for echo in loschmidt_echoes] max_echo_deviation = max(echo_deviations) avg_echo_deviation = np.mean(echo_deviations) min_echo = min(loschmidt_echoes)

        # Pass condition passes = max_echo_deviation <=
        self.tolerance and min_echo >= (1.0 -
        self.tolerance) test_time = time.time() - start_time result = { 'test_name': 'Loschmidt Echo', 'size': N, 'max_time': max_time, 'num_time_points': num_points, 'max_echo_deviation': float(max_echo_deviation), 'avg_echo_deviation': float(avg_echo_deviation), 'min_echo': float(min_echo), 'loschmidt_echoes': [float(echo)
        for echo in loschmidt_echoes], 'times': times.tolist(), 'passes': bool(passes), 'tolerance':
        self.tolerance, 'test_time': test_time }
        self.results[f'loschmidt_echo_N{N}'] = result
        return result
    def test_reversibility(self, N: int = 8, evolution_time: float = 1.0) -> Dict[str, Any]: """
        Test time evolution reversibility. psi(0) -> psi(t) -> psi(0) should recover the original state.
"""

        print(f"Testing time reversibility for N={N}...") start_time = time.time()

        # Get Hamiltonian U_raw = get_rft_basis(N) U = project_unitary(U_raw) H =
        self.extract_hamiltonian(U)

        # Create normalized random initial state np.random.seed(45) psi_0 = random_state(N) + 1j * np.random.random(N) psi_0 = psi_0 / np.linalg.norm(psi_0)

        # Forward evolution: psi(t) = e^(-iHt) psi0 U_forward = scipy.linalg.expm(-1j * H * evolution_time) psi_t = U_forward @ psi_0

        # Backward evolution: psi(0) = e^(+iHt) psi(t) U_backward = scipy.linalg.expm(1j * H * evolution_time) psi_recovered = U_backward @ psi_t

        # Compute recovery error recovery_error = np.linalg.norm(psi_0 - psi_recovered)

        # Check that forward and backward are inverses identity_error = np.linalg.norm(U_backward @ U_forward - np.eye(N))

        # Pass condition passes = ( recovery_error <=
        self.tolerance and identity_error <=
        self.tolerance ) test_time = time.time() - start_time result = { 'test_name': 'Time Reversibility', 'size': N, 'evolution_time': evolution_time, 'recovery_error': float(recovery_error), 'identity_error': float(identity_error), 'passes': bool(passes), 'tolerance':
        self.tolerance, 'test_time': test_time }
        self.results[f'reversibility_N{N}'] = result
        return result
    def test_quantum_dynamics_consistency(self, N: int = 8) -> Dict[str, Any]: """
        Test consistency of quantum dynamics across different initial states.
"""

        print(f"Testing quantum dynamics consistency for N={N}...") start_time = time.time()

        # Get Hamiltonian U_raw = get_rft_basis(N) U = project_unitary(U_raw) H =
        self.extract_hamiltonian(U)

        # Test with multiple random initial states num_states = 5 evolution_time = 0.5 consistency_results = []
        for i in range(num_states): np.random.seed(50 + i)

        # Different seeds for different states psi_0 = random_state(N) + 1j * np.random.random(N) psi_0 = psi_0 / np.linalg.norm(psi_0)

        # Evolve state times = [0.0, evolution_time/2, evolution_time] evolved_states =
        self.time_evolve_state(H, psi_0, np.array(times))

        # Check properties
        for this initial state state_results = {}

        # Norm conservation norms = [np.linalg.norm(psi)
        for psi in evolved_states] state_results['max_norm_deviation'] = max([abs(n - 1.0)
        for n in norms])

        # Energy conservation energies = [np.real(psi.conj() @ H @ psi)
        for psi in evolved_states] initial_energy = energies[0] state_results['max_energy_deviation'] = max([abs(E - initial_energy)
        for E in energies]) consistency_results.append(state_results)

        # Overall consistency metrics max_norm_deviations = [r['max_norm_deviation']
        for r in consistency_results] max_energy_deviations = [r['max_energy_deviation']
        for r in consistency_results] overall_max_norm_deviation = max(max_norm_deviations) overall_max_energy_deviation = max(max_energy_deviations)

        # Pass condition passes = ( overall_max_norm_deviation <=
        self.tolerance and overall_max_energy_deviation <=
        self.tolerance ) test_time = time.time() - start_time result = { 'test_name': 'Quantum Dynamics Consistency', 'size': N, 'num_test_states': num_states, 'evolution_time': evolution_time, 'overall_max_norm_deviation': float(overall_max_norm_deviation), 'overall_max_energy_deviation': float(overall_max_energy_deviation), 'individual_results': consistency_results, 'passes': bool(passes), 'tolerance':
        self.tolerance, 'test_time': test_time }
        self.results[f'dynamics_consistency_N{N}'] = result
        return result
    def run_full_time_evolution_suite(self, sizes: List[int] = None) -> Dict[str, Any]: """
        Run the complete time evolution validation test suite.
"""

        if sizes is None: sizes = [4, 8, 16]
        print("=" * 60)
        print("TIME EVOLUTION VALIDATION TEST SUITE")
        print("Using Symbolic Resonance Computing Methods")
        print("=" * 60) suite_results = { 'suite_name': 'Time Evolution Validation', 'timestamp': time.time(), 'canonical_parameters': get_canonical_parameters(), 'tolerance':
        self.tolerance, 'test_sizes': sizes, 'results': {} }
        for N in sizes:
        print(f"\nTesting size N={N}")
        print("-" * 30) size_results = {}

        # Run all tests for this size size_results['norm_conservation'] =
        self.test_norm_conservation(N) size_results['energy_conservation'] =
        self.test_energy_conservation(N) size_results['loschmidt_echo'] =
        self.test_loschmidt_echo(N) size_results['reversibility'] =
        self.test_reversibility(N) size_results['dynamics_consistency'] =
        self.test_quantum_dynamics_consistency(N)

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
        Run the time evolution validation tests.
"""
        validator = TimeEvolutionValidator(tolerance=1e-12)

        # Run comprehensive test suite results = validator.run_full_time_evolution_suite([4, 8, 16])

        # Print detailed results
        print("\nDETAILED RESULTS:")
        print("=" * 50) for size_key, size_results in results['results'].items():
        print(f"||n{size_key.upper()}:") for test_name, test_result in size_results.items():
        if test_name == 'overall_pass': continue
        print(f" {test_result['test_name']}: {'PASS'
        if test_result['passes'] else 'FAIL'}")
        if test_result['passes']: if 'max_norm_deviation' in test_result:
        print(f" Max norm deviation: {test_result['max_norm_deviation']:.2e}") if 'max_energy_deviation' in test_result:
        print(f" Max energy deviation: {test_result['max_energy_deviation']:.2e}") if 'recovery_error' in test_result:
        print(f" Recovery error: {test_result['recovery_error']:.2e}")
        else:
        print(f" ❌ Failed - check detailed results")
        return results

if __name__ == "__main__": main()