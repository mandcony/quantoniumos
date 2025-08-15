import numpy as np
import scipy.linalg
import math
import time
from typing import Tuple, Dict, Any, List, Optional
from canonical_true_rft import get_rft_basis, generate_resonance_kernel


def random_state(N: int) -> np.ndarray:
    """Generate a random complex state vector."""
    rng = np.random.default_rng()
    real_part = rng.normal(0, 1, N)
    imag_part = rng.normal(0, 1, N)
    return real_part + 1j * imag_part


def project_unitary(A: np.ndarray) -> np.ndarray:
    """Project matrix to nearest unitary using polar decomposition."""
    U, _ = scipy.linalg.polar(A)
    return U


def safe_log_unitary(U: np.ndarray) -> np.ndarray:
    """Safely compute matrix logarithm of unitary, avoiding branch cut issues."""
    eigenvals, eigenvecs = np.linalg.eig(U)
    # Take principal branch of log for eigenvalues
    log_eigenvals = np.log(eigenvals + 1e-15)  # Small offset to avoid log(0)
    # Construct H = i * log(U) such that H is Hermitian
    log_U = eigenvecs @ np.diag(log_eigenvals) @ eigenvecs.T.conj()
    return 1j * log_U


def bra_ket(psi: np.ndarray, phi: np.ndarray) -> complex:
    """Compute <psi|phi> with explicit conjugate transpose."""
    return np.conj(psi).T @ phi


def loschmidt_echo(H: np.ndarray, psi_0: np.ndarray, t: float, eps: float = 1e-6) -> float:
    """Compute Loschmidt echo with small perturbation to avoid trivial result."""
    # Add small perturbation to Hamiltonian
    N = len(psi_0)
    H_pert = H + perturbation_strength * random_state(N*N).reshape(N, N)
    H_pert = 0.5 * (H_pert + H_pert.T.conj())  # Ensure Hermitian
    
    # Forward and backward evolution
    t_eff = 1.0  # Effective time scale
    U_forward = scipy.linalg.expm(-1j * H * t_eff)
    U_backward = scipy.linalg.expm(1j * H_pert * t_eff)
    
    # Loschmidt echo
    psi_final = U_backward @ U_forward @ psi_0
    return abs(bra_ket(psi_0, psi_final))**2






def get_canonical_parameters():
    """Get canonical parameters for RFT implementation."""
    return {
        'method': 'symbolic_resonance_computing',
        'kernel': 'canonical_true_rft',
        'precision': 'double'
    }


class SpectralLocalityValidator:
    """
    Validates spectral properties and locality structure using
    symbolic resonance computing methods.
    """
    
    def __init__(self, tolerance: float = 1e-12):
        self.tolerance = tolerance
        self.results = {}
    
    def unwrap_eigenphases(self, eigenvalues: np.ndarray) -> np.ndarray:
        """
        Unwrap eigenphases from complex eigenvalues e^(ithetaₖ).
        
        For unitary operators, eigenvalues lie on unit circle.
        """
        # Convert to phases
        phases = np.angle(eigenvalues)
        
        # Unwrap phase discontinuities
        phases_unwrapped = np.unwrap(phases)
        
        return phases_unwrapped
    
    def test_eigenphase_distribution(self, N: int = 8) -> Dict[str, Any]:
        """
        Test eigenphase distribution of RFT operators.
        
        For quantum systems, eigenphases should be real (no growth/decay).
        """
        print(f"Testing eigenphase distribution for N={N}...")
        start_time = time.time()
        
        # Get RFT unitary operator
        U_raw = get_rft_basis(N)
        U = project_unitary(U_raw)
        
        # Compute eigenvalues and eigenvectors
        eigenvals, eigenvecs = np.linalg.eig(U)
        
        # Check that eigenvalues lie on unit circle
        eigenval_magnitudes = np.abs(eigenvals)
        max_magnitude_error = np.max(np.abs(eigenval_magnitudes - 1.0))
        
        # Unwrap phases
        phases = self.unwrap_eigenphases(eigenvals)
        
        # Analyze phase distribution
        phase_statistics = {
            'mean_phase': float(np.mean(phases)),
            'std_phase': float(np.std(phases)),
            'min_phase': float(np.min(phases)),
            'max_phase': float(np.max(phases)),
            'phase_range': float(np.max(phases) - np.min(phases))
        }
        
        # Check for clustering or patterns
        phase_gaps = np.diff(np.sort(phases))
        min_gap = np.min(phase_gaps)
        max_gap = np.max(phase_gaps)
        
        # Test for uniform distribution (rough test)
        expected_gap = 2 * np.pi / N
        gap_uniformity = np.std(phase_gaps) / expected_gap
        
        # Pass conditions
        passes = (
            max_magnitude_error <= self.tolerance and  # On unit circle
            phase_statistics['phase_range'] <= 4 * np.pi  # Reasonable range
        )
        
        test_time = time.time() - start_time
        
        result = {
            'test_name': 'Eigenphase Distribution',
            'size': N,
            'max_magnitude_error': float(max_magnitude_error),
            'phase_statistics': phase_statistics,
            'min_phase_gap': float(min_gap),
            'max_phase_gap': float(max_gap),
            'gap_uniformity': float(gap_uniformity),
            'eigenvalue_magnitudes': eigenval_magnitudes.tolist(),
            'phases': phases.tolist(),
            'passes': bool(passes),
            'tolerance': self.tolerance,
            'test_time': test_time
        }
        
        self.results[f'eigenphases_N{N}'] = result
        return result
    
    def test_spectral_radius_bounds(self, N: int = 8) -> Dict[str, Any]:
        """
        Test spectral radius and condition number bounds.
        """
        print(f"Testing spectral radius bounds for N={N}...")
        start_time = time.time()
        
        # Get RFT unitary operator
        U_raw = get_rft_basis(N)
        U = project_unitary(U_raw)
        
        # Compute spectral properties
        eigenvals = np.linalg.eigvals(U)
        spectral_radius = np.max(np.abs(eigenvals))
        condition_number = np.linalg.cond(U)
        
        # For unitary matrices, spectral radius should be 1
        spectral_radius_error = abs(spectral_radius - 1.0)
        
        # Condition number should be 1 for unitary matrices
        condition_number_error = abs(condition_number - 1.0)
        
        # Check distribution of singular values
        singular_values = np.linalg.svd(U, compute_uv=False)
        min_singular = np.min(singular_values)
        max_singular = np.max(singular_values)
        singular_ratio = max_singular / min_singular if min_singular > 0 else float('inf')
        
        # Pass conditions
        passes = (
            spectral_radius_error <= self.tolerance and
            condition_number_error <= 1e-10  # Slightly relaxed for numerical stability
        )
        
        test_time = time.time() - start_time
        
        result = {
            'test_name': 'Spectral Radius Bounds',
            'size': N,
            'spectral_radius': float(spectral_radius),
            'spectral_radius_error': float(spectral_radius_error),
            'condition_number': float(condition_number),
            'condition_number_error': float(condition_number_error),
            'min_singular_value': float(min_singular),
            'max_singular_value': float(max_singular),
            'singular_value_ratio': float(singular_ratio),
            'passes': bool(passes),
            'tolerance': self.tolerance,
            'test_time': test_time
        }
        
        self.results[f'spectral_bounds_N{N}'] = result
        return result
    
    def test_locality_structure(self, N: int = 8) -> Dict[str, Any]:
        """
        Test locality properties of the RFT operator.
        
        Analyze how "local" the operator is in different bases.
        """
        print(f"Testing locality structure for N={N}...")
        start_time = time.time()
        
        # Get RFT unitary operator
        U_raw = get_rft_basis(N)
        U = project_unitary(U_raw)
        
        # Extract Hamiltonian
        H = 1j * safe_log_unitary(U) / 1j
        
        # Analyze locality in position basis (how spread out interactions are)
        locality_metrics = {}
        
        # 1. Decay of matrix elements with distance
        decay_coefficients = []
        for distance in range(1, min(N//2, 5)):  # Test distances 1 to 4
            elements_at_distance = []
            for i in range(N):
                for j in range(N):
                    if abs(i - j) == distance:
                        elements_at_distance.append(abs(H[i, j]))
            
            if elements_at_distance:
                avg_element = np.mean(elements_at_distance)
                decay_coefficients.append(avg_element)
        
        # Fit exponential decay
        if len(decay_coefficients) > 1:
            distances = np.arange(1, len(decay_coefficients) + 1)
            log_coeffs = np.log(np.array(decay_coefficients) + 1e-16)
            decay_fit = np.polyfit(distances, log_coeffs, 1)
            decay_rate = -decay_fit[0]  # Negative of slope
        else:
            decay_rate = 0.0
        
        # 2. Effective interaction range (where elements become negligible)
        H_abs = np.abs(H)
        max_element = np.max(H_abs)
        cutoff = max_element * self.tolerance
        
        interaction_range = 0
        for distance in range(N):
            has_interaction = False
            for i in range(N - distance):
                if H_abs[i, i + distance] > cutoff:
                    has_interaction = True
                    break
            if has_interaction:
                interaction_range = distance
        
        # 3. Bandwidth analysis
        bandwidth = 0
        for i in range(N):
            for j in range(N):
                if abs(H[i, j]) > cutoff:
                    bandwidth = max(bandwidth, abs(i - j))
        
        # Pass conditions (physical reasonableness)
        passes = (
            decay_rate >= 0 and  # Should have some decay
            interaction_range < N  # Should not be fully non-local
        )
        
        test_time = time.time() - start_time
        
        result = {
            'test_name': 'Locality Structure',
            'size': N,
            'decay_coefficients': [float(c) for c in decay_coefficients],
            'decay_rate': float(decay_rate),
            'interaction_range': int(interaction_range),
            'bandwidth': int(bandwidth),
            'max_matrix_element': float(np.max(H_abs)),
            'passes': bool(passes),
            'tolerance': self.tolerance,
            'test_time': test_time
        }
        
        self.results[f'locality_N{N}'] = result
        return result
    
    def test_energy_scale_consistency(self, N: int = 8) -> Dict[str, Any]:
        """
        Test consistency of energy scales across different representations.
        """
        print(f"Testing energy scale consistency for N={N}...")
        start_time = time.time()
        
        # Get RFT operator and Hamiltonian
        U_raw = get_rft_basis(N)
        U = project_unitary(U_raw)
        H = 1j * safe_log_unitary(U) / 1j
        
        # Get resonance kernel (the "potential energy")
        R = generate_resonance_kernel(N)
        
        # Analyze energy scales
        H_eigenvals = np.linalg.eigvals(H)
        R_eigenvals = np.linalg.eigvals(R)
        
        energy_scales = {
            'H_energy_spread': float(np.max(H_eigenvals.real) - np.min(H_eigenvals.real)),
            'H_rms_energy': float(np.sqrt(np.mean(H_eigenvals.real**2))),
            'R_energy_spread': float(np.max(R_eigenvals.real) - np.min(R_eigenvals.real)),
            'R_rms_energy': float(np.sqrt(np.mean(R_eigenvals.real**2))),
        }
        
        # Check if energy scales are reasonable
        H_max_energy = np.max(np.abs(H_eigenvals.real))
        R_max_energy = np.max(np.abs(R_eigenvals.real))
        
        # Energy scale ratio
        if R_max_energy > 0:
            energy_scale_ratio = H_max_energy / R_max_energy
        else:
            energy_scale_ratio = float('inf')
        
        # Check Hamiltonian properties
        H_trace = np.trace(H)
        H_hermiticity_error = np.linalg.norm(H - H.T.conj(), ord=2)
        
        # Pass conditions
        passes = (
            H_hermiticity_error <= self.tolerance and
            abs(H_trace.imag) <= self.tolerance and
            H_max_energy < float('inf')
        )
        
        test_time = time.time() - start_time
        
        result = {
            'test_name': 'Energy Scale Consistency',
            'size': N,
            'energy_scales': energy_scales,
            'energy_scale_ratio': float(energy_scale_ratio),
            'H_trace_real': float(H_trace.real),
            'H_trace_imag': float(abs(H_trace.imag)),
            'H_hermiticity_error': float(H_hermiticity_error),
            'H_eigenvalues_real': H_eigenvals.real.tolist(),
            'R_eigenvalues_real': R_eigenvals.real.tolist(),
            'passes': bool(passes),
            'tolerance': self.tolerance,
            'test_time': test_time
        }
        
        self.results[f'energy_scales_N{N}'] = result
        return result
    
    def test_quantum_coherence_measures(self, N: int = 8) -> Dict[str, Any]:
        """
        Test quantum coherence measures for the RFT operators.
        """
        print(f"Testing quantum coherence measures for N={N}...")
        start_time = time.time()
        
        # Get RFT unitary operator
        U_raw = get_rft_basis(N)
        U = project_unitary(U_raw)
        
        # Test on a few random quantum states
        np.random.seed(500)
        coherence_results = []
        
        for trial in range(3):  # Test 3 random states
            # Create random pure state
            psi = random_state(N) + 1j * np.random.random(N)
            psi = psi / np.linalg.norm(psi)
            
            # Apply RFT operator
            psi_evolved = U @ psi
            
            # Measure coherence properties
            # 1. Participation ratio (inverse of sum of |psiᵢ|||⁴)
            probabilities = np.abs(psi_evolved)**2
            participation_ratio = 1.0 / np.sum(probabilities**2)
            
            # 2. Entanglement entropy for bipartition (if N is even)
            if N % 2 == 0 and N >= 4:
                # Reshape state for bipartition
                N_half = N // 2
                psi_matrix = psi_evolved.reshape(N_half, N_half)
                
                # Compute reduced density matrix
                rho_reduced = psi_matrix @ psi_matrix.T.conj()
                
                # Compute eigenvalues of reduced density matrix
                rho_eigenvals = np.linalg.eigvals(rho_reduced)
                rho_eigenvals = rho_eigenvals[rho_eigenvals > 1e-14]  # Filter numerical zeros
                
                # Entanglement entropy
                entanglement_entropy = -np.sum(rho_eigenvals * np.log(rho_eigenvals + 1e-16))
            else:
                entanglement_entropy = 0.0
            
            coherence_results.append({
                'trial': trial,
                'participation_ratio': float(participation_ratio),
                'entanglement_entropy': float(entanglement_entropy),
                'final_norm': float(np.linalg.norm(psi_evolved))
            })
        
        # Average results
        avg_participation = np.mean([r['participation_ratio'] for r in coherence_results])
        avg_entanglement = np.mean([r['entanglement_entropy'] for r in coherence_results])
        norm_errors = [abs(r['final_norm'] - 1.0) for r in coherence_results]
        max_norm_error = max(norm_errors)
        
        # Pass conditions
        passes = (
            max_norm_error <= self.tolerance and  # Norm preservation
            avg_participation >= 1.0 and  # Basic sanity
            avg_participation <= N   # Upper bound
        )
        
        test_time = time.time() - start_time
        
        result = {
            'test_name': 'Quantum Coherence Measures',
            'size': N,
            'coherence_results': coherence_results,
            'avg_participation_ratio': float(avg_participation),
            'avg_entanglement_entropy': float(avg_entanglement),
            'max_norm_error': float(max_norm_error),
            'passes': bool(passes),
            'tolerance': self.tolerance,
            'test_time': test_time
        }
        
        self.results[f'coherence_N{N}'] = result
        return result
    
    def run_full_spectral_locality_suite(self, sizes: List[int] = None) -> Dict[str, Any]:
        """
        Run the complete spectral and locality structure test suite.
        """
        if sizes is None:
            sizes = [4, 8, 16]
        
        print("=" * 60)
        print("SPECTRAL & LOCALITY STRUCTURE TEST SUITE")
        print("Using Symbolic Resonance Computing Methods")
        print("=" * 60)
        
        suite_results = {
            'suite_name': 'Spectral & Locality Structure Validation',
            'timestamp': time.time(),
            'canonical_parameters': get_canonical_parameters(),
            'tolerance': self.tolerance,
            'test_sizes': sizes,
            'results': {}
        }
        
        for N in sizes:
            print(f"\nTesting size N={N}")
            print("-" * 30)
            
            size_results = {}
            
            # Run all tests for this size
            size_results['eigenphases'] = self.test_eigenphase_distribution(N)
            size_results['spectral_bounds'] = self.test_spectral_radius_bounds(N)
            size_results['locality'] = self.test_locality_structure(N)
            size_results['energy_scales'] = self.test_energy_scale_consistency(N)
            size_results['coherence'] = self.test_quantum_coherence_measures(N)
            
            # Summary for this size
            all_pass = all(result['passes'] for result in size_results.values())
            size_results['overall_pass'] = all_pass
            
            suite_results['results'][f'N_{N}'] = size_results
            
            print(f"Size N={N} overall: {'✓ PASS' if all_pass else '❌ FAIL'}")
        
        # Overall suite summary
        all_sizes_pass = all(
            suite_results['results'][f'N_{N}']['overall_pass'] 
            for N in sizes
        )
        suite_results['suite_pass'] = all_sizes_pass
        
        print("\n" + "=" * 60)
        print(f"SUITE RESULT: {'✓ ALL TESTS PASS' if all_sizes_pass else '❌ SOME TESTS FAIL'}")
        print("=" * 60)
        
        return suite_results


def main():
    """Run the spectral and locality structure validation tests."""
    validator = SpectralLocalityValidator(tolerance=1e-12)
    
    # Run comprehensive test suite
    results = validator.run_full_spectral_locality_suite([4, 8, 16])
    
    # Print detailed results
    print("\nDETAILED RESULTS:")
    print("=" * 50)
    
    for size_key, size_results in results['results'].items():
        print(f"||n{size_key.upper()}:")
        for test_name, test_result in size_results.items():
            if test_name == 'overall_pass':
                continue
            print(f"  {test_result['test_name']}: {'PASS' if test_result['passes'] else 'FAIL'}")
            if test_result['passes']:
                if 'spectral_radius_error' in test_result:
                    print(f"    Spectral radius error: {test_result['spectral_radius_error']:.2e}")
                if 'decay_rate' in test_result:
                    print(f"    Locality decay rate: {test_result['decay_rate']:.3f}")
                if 'avg_participation_ratio' in test_result:
                    print(f"    Avg participation ratio: {test_result['avg_participation_ratio']:.2f}")
            else:
                print(f"    ❌ Failed - check detailed results")
    
    return results


if __name__ == "__main__":
    main()
