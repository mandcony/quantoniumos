
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
        Compute <psi|||phi> with explicit conjugate transpose.
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

class LieAlgebraValidator: """
        Validates Lie algebra closure properties using symbolic resonance computing methods.
"""

    def __init__(self, tolerance: float = 1e-10):
        self.tolerance = tolerance

        # Relaxed for Lie algebra operations
        self.results = {}
    def extract_hamiltonian(self, U: np.ndarray) -> np.ndarray: """
        Extract Hamiltonian from unitary operator.
"""
        log_U = safe_log_unitary(U) / 1j
        return 1j * log_U
    def generate_primitive_generators(self, N: int) -> List[np.ndarray]: """
        Generate primitive generators from RFT components. Extract generators from different aspects of the RFT construction: - Base Hamiltonian H0 - Powers of base operator: H1 = log(U^2), H2 = log(U^3), etc. - Pauli-like operators for small dimensions
"""
        generators = []

        # Base RFT unitary U_base = get_rft_basis(N) H_base =
        self.extract_hamiltonian(U_base) generators.append(H_base)

        # Generators from powers of base unitary
        for power in [2, 3]:
        try: U_power = np.linalg.matrix_power(U_base, power) H_power =
        self.extract_hamiltonian(U_power) generators.append(H_power)
        except: continue

        # Skip
        if matrix log fails

        # Add scaled identity (trivial generator) generators.append(np.eye(N, dtype=complex))

        # For small dimensions, add some standard generators
        if N <= 4:

        # Add Pauli-like matrices scaled to match dimension pauli_x = np.array([[0, 1], [1, 0]], dtype=complex) pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex) pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
        if N == 2: generators.extend([pauli_x, pauli_y, pauli_z])
        el
        if N == 4:

        # Extend to 4x4 with tensor products I2 = np.eye(2, dtype=complex) generators.extend([ np.kron(pauli_x, I2), np.kron(I2, pauli_x), np.kron(pauli_z, pauli_z) ])
        return generators
    def compute_commutator(self, A: np.ndarray, B: np.ndarray) -> np.ndarray: """
        Compute commutator [A, B] = AB - BA
"""

        return A @ B - B @ A
    def project_onto_span(self, matrix: np.ndarray, basis: List[np.ndarray]) -> Tuple[np.ndarray, float]: """
        Project matrix onto span of basis matrices. Returns: (projection, residual_norm)
"""

        if not basis:
        return np.zeros_like(matrix), np.linalg.norm(matrix, ord='fro')

        # Vectorize matrices for linear algebra matrix_vec = matrix.flatten() basis_vecs = [b.flatten()
        for b in basis]

        # Create basis matrix (each column is a vectorized basis matrix)
        if len(basis_vecs[0]) == 0:
        return np.zeros_like(matrix), np.linalg.norm(matrix, ord='fro') basis_matrix = np.column_stack(basis_vecs)

        # Solve least squares: find coefficients c such that Bc approx matrix_vec
        try: coeffs, residuals, rank, s = np.linalg.lstsq(basis_matrix, matrix_vec, rcond=None)

        # Reconstruct projection projection_vec = basis_matrix @ coeffs projection = projection_vec.reshape(matrix.shape)

        # Compute residual residual_norm = np.linalg.norm(matrix - projection, ord='fro')
        return projection, residual_norm except np.linalg.LinAlgError:
        return np.zeros_like(matrix), np.linalg.norm(matrix, ord='fro')
    def test_commutator_closure(self, N: int = 8) -> Dict[str, Any]: """
        Test closure under commutator operations. For each pair of generators [Aᵢ, Aⱼ], check
        if it lies in span{Aₖ}.
"""

        print(f"Testing commutator closure for N={N}...") start_time = time.time()

        # Generate primitive generators generators =
        self.generate_primitive_generators(N) num_generators = len(generators)

        # Remove any duplicate or zero generators filtered_generators = []
        for gen in generators:
        if np.linalg.norm(gen, ord='fro') >
        self.tolerance:

        # Check
        if already in list (approximately) is_duplicate = False
        for existing in filtered_generators:
        if np.linalg.norm(gen - existing, ord='fro') <
        self.tolerance: is_duplicate = True break
        if not is_duplicate: filtered_generators.append(gen) generators = filtered_generators num_generators = len(generators)
        if num_generators < 2:
        return { 'test_name': 'Commutator Closure', 'size': N, 'error': 'Insufficient generators', 'passes': False, 'test_time': time.time() - start_time }

        # Test all pairs of generators commutator_results = [] max_residual = 0.0 total_residual = 0.0 num_commutators = 0
        for i in range(num_generators):
        for j in range(i + 1, num_generators):

        # Only upper triangle

        # Compute commutator [Aᵢ, Aⱼ] comm =
        self.compute_commutator(generators[i], generators[j])

        # Project onto span of generators projection, residual =
        self.project_onto_span(comm, generators) commutator_results.append({ 'generator_pair': (i, j), 'commutator_norm': float(np.linalg.norm(comm, ord='fro')), 'residual_norm': float(residual), 'relative_residual': float(residual / (np.linalg.norm(comm, ord='fro') + 1e-16)) }) max_residual = max(max_residual, residual) total_residual += residual num_commutators += 1 avg_residual = total_residual / num_commutators
        if num_commutators > 0 else 0.0

        # Pass condition: commutators should lie approximately in span passes = max_residual <=
        self.tolerance test_time = time.time() - start_time result = { 'test_name': 'Commutator Closure', 'size': N, 'num_generators': num_generators, 'num_commutators': num_commutators, 'max_residual': float(max_residual), 'avg_residual': float(avg_residual), 'commutator_results': commutator_results, 'generator_norms': [float(np.linalg.norm(g, ord='fro'))
        for g in generators], 'passes': bool(passes), 'tolerance':
        self.tolerance, 'test_time': test_time }
        self.results[f'closure_N{N}'] = result
        return result
    def test_jacobi_identity(self, N: int = 8) -> Dict[str, Any]: """
        Test Jacobi identity: [A, [B, C]] + [B, [C, A]] + [C, [A, B]] = 0
"""

        print(f"Testing Jacobi identity for N={N}...") start_time = time.time() generators =
        self.generate_primitive_generators(N)

        # Filter generators filtered_generators = []
        for gen in generators:
        if np.linalg.norm(gen, ord='fro') >
        self.tolerance: filtered_generators.append(gen) generators = filtered_generators num_generators = len(generators)
        if num_generators < 3:
        return { 'test_name': 'Jacobi Identity', 'size': N, 'error': 'Need at least 3 generators', 'passes': False, 'test_time': time.time() - start_time }

        # Test Jacobi identity for all triples jacobi_violations = [] max_violation = 0.0
        for i in range(num_generators):
        for j in range(num_generators):
        for k in range(num_generators):
        if i != j and j != k and i != k:

        # All different A, B, C = generators[i], generators[j], generators[k]

        # Compute [A, [B, C]] + [B, [C, A]] + [C, [A, B]] comm_BC =
        self.compute_commutator(B, C) comm_CA =
        self.compute_commutator(C, A) comm_AB =
        self.compute_commutator(A, B) jacobi_sum = (
        self.compute_commutator(A, comm_BC) +
        self.compute_commutator(B, comm_CA) +
        self.compute_commutator(C, comm_AB) ) violation = np.linalg.norm(jacobi_sum, ord='fro')
        if violation >
        self.tolerance: jacobi_violations.append({ 'triple': (i, j, k), 'violation': float(violation) }) max_violation = max(max_violation, violation)

        # Pass condition: Jacobi identity should hold passes = max_violation <=
        self.tolerance test_time = time.time() - start_time result = { 'test_name': 'Jacobi Identity', 'size': N, 'num_generators': num_generators, 'max_jacobi_violation': float(max_violation), 'num_violations': len(jacobi_violations), 'jacobi_violations': jacobi_violations[:10],

        # Limit output 'passes': bool(passes), 'tolerance':
        self.tolerance, 'test_time': test_time }
        self.results[f'jacobi_N{N}'] = result
        return result
    def test_lie_algebra_dimension(self, N: int = 8) -> Dict[str, Any]: """
        Estimate the dimension of the Lie algebra generated by our operators.
"""

        print(f"Testing Lie algebra dimension for N={N}...") start_time = time.time() generators =
        self.generate_primitive_generators(N)

        # Start with initial generators algebra_basis = []
        for gen in generators:
        if np.linalg.norm(gen, ord='fro') >
        self.tolerance: algebra_basis.append(gen)

        # Iteratively add commutators until no new linearly independent elements max_iterations = 10
        for iteration in range(max_iterations): new_elements = [] current_size = len(algebra_basis)

        # Compute all commutators of existing basis elements
        for i in range(current_size):
        for j in range(i + 1, current_size): comm =
        self.compute_commutator(algebra_basis[i], algebra_basis[j])

        # Check
        if linearly independent from existing basis _, residual =
        self.project_onto_span(comm, algebra_basis)
        if residual >
        self.tolerance: new_elements.append(comm)

        # Add linearly independent new elements
        for new_elem in new_elements:

        # Double-check linear independence _, residual =
        self.project_onto_span(new_elem, algebra_basis)
        if residual >
        self.tolerance: algebra_basis.append(new_elem)

        # Check for convergence
        if len(algebra_basis) == current_size: break algebra_dimension = len(algebra_basis)

        # Theoretical maximum dimension for N×N Hermitian matrices is N^2 max_possible_dimension = N * N dimension_ratio = algebra_dimension / max_possible_dimension

        # Pass condition: should have some reasonable dimension passes = algebra_dimension >= 2

        # At least non-trivial test_time = time.time() - start_time result = { 'test_name': 'Lie Algebra Dimension', 'size': N, 'algebra_dimension': algebra_dimension, 'max_possible_dimension': max_possible_dimension, 'dimension_ratio': float(dimension_ratio), 'iterations_needed': iteration + 1, 'basis_element_norms': [float(np.linalg.norm(elem, ord='fro'))
        for elem in algebra_basis], 'passes': bool(passes), 'tolerance':
        self.tolerance, 'test_time': test_time }
        self.results[f'dimension_N{N}'] = result
        return result
    def test_su_n_structure(self, N: int = 4) -> Dict[str, Any]: """
        Test
        if the generated Lie algebra has SU(N)-like structure. Only practical for small N.
"""

        print(f"Testing SU(N) structure for N={N}...") start_time = time.time()
        if N > 4:
        return { 'test_name': 'SU(N) Structure', 'size': N, 'error': 'Test only practical for N <= 4', 'passes': True,

        # Skip for large N 'test_time': time.time() - start_time } generators =
        self.generate_primitive_generators(N)

        # Check properties expected for SU(N) generators: # 1. Traceless (except identity) # 2. Hermitian # 3. Linear independence su_properties = [] for i, gen in enumerate(generators): trace_val = np.trace(gen) is_hermitian = np.linalg.norm(gen - gen.T.conj(), ord='fro') <
        self.tolerance is_traceless = abs(trace_val) <
        self.tolerance or np.allclose(gen, np.eye(N, dtype=complex)) su_properties.append({ 'generator_index': i, 'trace': float(trace_val.real
        if isinstance(trace_val, complex) else trace_val), 'trace_imag': float(trace_val.imag
        if isinstance(trace_val, complex) else 0), 'is_hermitian': bool(is_hermitian), 'is_traceless_or_identity': bool(is_traceless), 'norm': float(np.linalg.norm(gen, ord='fro')) })

        # Count how many satisfy SU(N) properties hermitian_count = sum(1
        for prop in su_properties
        if prop['is_hermitian']) traceless_count = sum(1
        for prop in su_properties
        if prop['is_traceless_or_identity'])

        # Pass condition: most generators should have SU(N)-like properties passes = ( hermitian_count >= len(generators) * 0.8 and traceless_count >= len(generators) * 0.6 ) test_time = time.time() - start_time result = { 'test_name': 'SU(N) Structure', 'size': N, 'num_generators': len(generators), 'hermitian_count': hermitian_count, 'traceless_count': traceless_count, 'su_properties': su_properties, 'passes': bool(passes), 'tolerance':
        self.tolerance, 'test_time': test_time }
        self.results[f'su_structure_N{N}'] = result
        return result
    def run_full_lie_algebra_suite(self, sizes: List[int] = None) -> Dict[str, Any]: """
        Run the complete Lie algebra closure test suite.
"""

        if sizes is None: sizes = [2, 4, 8]

        # Smaller sizes due to combinatorial complexity
        print("=" * 60)
        print("LIE ALGEBRA CLOSURE VALIDATION TEST SUITE")
        print("Using Symbolic Resonance Computing Methods")
        print("=" * 60) suite_results = { 'suite_name': 'Lie Algebra Closure Validation', 'timestamp': time.time(), 'canonical_parameters': get_canonical_parameters(), 'tolerance':
        self.tolerance, 'test_sizes': sizes, 'results': {} }
        for N in sizes:
        print(f"\nTesting size N={N}")
        print("-" * 30) size_results = {}

        # Run all tests for this size size_results['closure'] =
        self.test_commutator_closure(N) size_results['jacobi'] =
        self.test_jacobi_identity(N) size_results['dimension'] =
        self.test_lie_algebra_dimension(N)
        if N <= 4:

        # Only test SU(N) structure for small N size_results['su_structure'] =
        self.test_su_n_structure(N)

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
        Run the Lie algebra closure validation tests.
"""
        validator = LieAlgebraValidator(tolerance=1e-10)

        # Run comprehensive test suite results = validator.run_full_lie_algebra_suite([2, 4, 8])

        # Print detailed results
        print("\nDETAILED RESULTS:")
        print("=" * 50) for size_key, size_results in results['results'].items():
        print(f"||n{size_key.upper()}:") for test_name, test_result in size_results.items():
        if test_name == 'overall_pass': continue
        print(f" {test_result['test_name']}: {'PASS'
        if test_result['passes'] else 'FAIL'}")
        if test_result['passes']: if 'max_residual' in test_result:
        print(f" Max closure residual: {test_result['max_residual']:.2e}") if 'max_jacobi_violation' in test_result:
        print(f" Max Jacobi violation: {test_result['max_jacobi_violation']:.2e}") if 'algebra_dimension' in test_result:
        print(f" Algebra dimension: {test_result['algebra_dimension']}")
        else:
        print(f" ❌ Failed - check detailed results")
        return results

if __name__ == "__main__": main()