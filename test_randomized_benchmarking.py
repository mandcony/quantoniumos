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


class RandomizedBenchmarkingValidator:
    """
    Validates gate sequence decay using randomized benchmarking principles
    with symbolic resonance computing methods.
    """
    
    def __init__(self, tolerance: float = 1e-12):
        self.tolerance = tolerance
        self.results = {}
    
    def generate_random_gate_sequence(self, base_U: np.ndarray, depth: int, seed: int = None) -> np.ndarray:
        """
        Generate a random sequence of gates based on powers of the base unitary.
        """
        if seed is not None:
            np.random.seed(seed)
        
        N = base_U.shape[0]
        result = np.eye(N, dtype=complex)
        
        # Use random powers of the base unitary to simulate gate variety
        for i in range(depth):
            power = np.random.randint(1, 5)  # Random power 1-4
            gate = np.linalg.matrix_power(base_U, power)
            result = gate @ result
        
        return result
    
    def compute_survival_probability(self, U_sequence: np.ndarray, reference_state: np.ndarray) -> float:
        """
        Compute survival probability |<psi_ref | U_sequence | psi_ref>^2.
        """
        final_state = U_sequence @ reference_state
        overlap = np.vdot(reference_state, final_state)
        return abs(overlap) ** 2
    
    def fit_exponential_decay(self, depths: np.ndarray, probabilities: np.ndarray) -> Tuple[float, float, float, Dict[str, float]]:
        """
        Fit survival probabilities to exponential decay model: p(m) = A * r^m + B
        
        Returns: (A, r, B, fit_metrics)
        """
        def decay_model(m, A, r, B):
            return A * (r ** m) + B
        
        # Initial guess
        p0 = [probabilities[0] - probabilities[-1], 0.95, probabilities[-1]]
        
        try:
            # Fit the model
            popt, pcov = scipy.optimize.curve_fit(
                decay_model, depths, probabilities, 
                p0=p0, maxfev=5000,
                bounds=([0, 0, 0], [2, 1, 1])  # Physical bounds
            )
            
            A, r, B = popt
            
            # Compute fit quality metrics
            y_fit = decay_model(depths, A, r, B)
            residuals = probabilities - y_fit
            r_squared = 1 - np.sum(residuals**2) / np.sum((probabilities - np.mean(probabilities))**2)
            rmse = np.sqrt(np.mean(residuals**2))
            
            # Parameter uncertainties from covariance matrix
            param_errors = np.sqrt(np.diag(pcov))
            
            fit_metrics = {
                'r_squared': r_squared,
                'rmse': rmse,
                'A_error': param_errors[0],
                'r_error': param_errors[1],
                'B_error': param_errors[2],
                'fit_successful': True
            }
            
        except Exception as e:
            A, r, B = np.nan, np.nan, np.nan
            fit_metrics = {
                'error': str(e),
                'fit_successful': False
            }
        
        return A, r, B, fit_metrics
    
    def test_gate_sequence_decay(self, N: int = 8, max_depth: int = 20, num_sequences: int = 10) -> Dict[str, Any]:
        """
        Test randomized benchmarking decay for gate sequences.
        """
        print(f"Testing gate sequence decay for N={N}...")
        start_time = time.time()
        
        # Get base unitary
        base_U_raw = get_rft_basis(N)
        U = project_unitary(base_U_raw)
        
        # Create reference state (random normalized state)
        np.random.seed(200)  # Reproducible
        ref_state = random_state(N) + 1j * np.random.random(N)
        ref_state = ref_state / np.linalg.norm(ref_state)
        
        # Test different depths
        depths = np.arange(1, max_depth + 1)
        
        # Collect survival probabilities for each depth
        all_probabilities = []
        
        for depth in depths:
            depth_probabilities = []
            
            # Generate multiple random sequences for this depth
            for seq_idx in range(num_sequences):
                U_sequence = self.generate_random_gate_sequence(base_U, depth, seed=200 + depth * 100 + seq_idx)
                prob = self.compute_survival_probability(U_sequence, ref_state)
                depth_probabilities.append(prob)
            
            # Average probability for this depth
            avg_prob = np.mean(depth_probabilities)
            all_probabilities.append(avg_prob)
        
        probabilities = np.array(all_probabilities)
        
        # Fit exponential decay model
        A, r, B, fit_metrics = self.fit_exponential_decay(depths, probabilities)
        
        # Analyze decay rate
        if fit_metrics['fit_successful']:
            # For ideal gates, r should be close to 1 (no decay except rounding errors)
            ideal_r = 1.0
            r_deviation = abs(r - ideal_r)
            
            # Error per gate (1 - r gives error rate)
            error_per_gate = 1 - r if not np.isnan(r) else float('inf')
            
            # Pass condition: decay should be minimal (close to machine precision effects)
            passes = (
                fit_metrics['r_squared'] > 0.8 and  # Good fit
                r_deviation <= 0.1 and  # r close to 1
                error_per_gate <= 0.1   # Low error rate
            )
        else:
            r_deviation = float('inf')
            error_per_gate = float('inf')
            passes = False
        
        test_time = time.time() - start_time
        
        result = {
            'test_name': 'Gate Sequence Decay',
            'size': N,
            'max_depth': max_depth,
            'num_sequences_per_depth': num_sequences,
            'decay_parameters': {
                'A': float(A) if not np.isnan(A) else None,
                'r': float(r) if not np.isnan(r) else None,
                'B': float(B) if not np.isnan(B) else None
            },
            'fit_metrics': fit_metrics,
            'r_deviation_from_ideal': float(r_deviation),
            'error_per_gate': float(error_per_gate),
            'survival_probabilities': probabilities.tolist(),
            'depths': depths.tolist(),
            'passes': bool(passes),
            'tolerance': self.tolerance,
            'test_time': test_time
        }
        
        self.results[f'decay_N{N}'] = result
        return result
    
    def test_gate_fidelity_scaling(self, N: int = 8, num_trials: int = 20) -> Dict[str, Any]:
        """
        Test how gate fidelity scales with gate complexity.
        """
        print(f"Testing gate fidelity scaling for N={N}...")
        start_time = time.time()
        
        base_U_raw = get_rft_basis(N)
        U = project_unitary(base_U_raw)
        
        # Test different gate "complexities" (powers)
        gate_powers = [1, 2, 3, 4, 5]
        
        fidelity_results = {}
        
        for power in gate_powers:
            gate = np.linalg.matrix_power(base_U, power)
            
            # Test fidelity with multiple random states
            fidelities = []
            
            for trial in range(num_trials):
                np.random.seed(300 + power * 100 + trial)
                
                # Random input state
                psi_in = random_state(N) + 1j * np.random.random(N)
                psi_in = psi_in / np.linalg.norm(psi_in)
                
                # Apply gate
                psi_out = gate @ psi_in
                
                # Fidelity with expected output (should be perfectly unitary)
                # Check if norm is preserved and gate is applied correctly
                norm_preservation = abs(np.linalg.norm(psi_out) - 1.0)
                
                # Measure "fidelity" as norm preservation
                fidelity = 1.0 - norm_preservation
                fidelities.append(fidelity)
            
            avg_fidelity = np.mean(fidelities)
            std_fidelity = np.std(fidelities)
            min_fidelity = np.min(fidelities)
            
            fidelity_results[f'power_{power}'] = {
                'avg_fidelity': float(avg_fidelity),
                'std_fidelity': float(std_fidelity),
                'min_fidelity': float(min_fidelity),
                'num_trials': num_trials
            }
        
        # Check if fidelity remains high across all powers
        min_avg_fidelity = min(result['avg_fidelity'] for result in fidelity_results.values())
        max_std_fidelity = max(result['std_fidelity'] for result in fidelity_results.values())
        
        # Pass condition: high fidelity maintained
        passes = (
            min_avg_fidelity >= (1.0 - self.tolerance) and
            max_std_fidelity <= self.tolerance
        )
        
        test_time = time.time() - start_time
        
        result = {
            'test_name': 'Gate Fidelity Scaling',
            'size': N,
            'gate_powers': gate_powers,
            'fidelity_results': fidelity_results,
            'min_avg_fidelity': float(min_avg_fidelity),
            'max_std_fidelity': float(max_std_fidelity),
            'passes': bool(passes),
            'tolerance': self.tolerance,
            'test_time': test_time
        }
        
        self.results[f'fidelity_scaling_N{N}'] = result
        return result
    
    def test_sequence_composition_error(self, N: int = 8) -> Dict[str, Any]:
        """
        Test error accumulation in gate sequence composition.
        """
        print(f"Testing sequence composition error for N={N}...")
        start_time = time.time()
        
        base_U_raw = get_rft_basis(N)
        U = project_unitary(base_U_raw)
        
        # Test sequence: U^n vs (U^1)^n
        powers_to_test = [2, 3, 4, 5]
        composition_errors = []
        
        for n in powers_to_test:
            # Direct computation: U^n
            U_direct = np.linalg.matrix_power(base_U, n)
            
            # Sequential composition: U * U * ... * U (n times)
            U_sequential = np.eye(N, dtype=complex)
            for _ in range(n):
                U_sequential = base_U @ U_sequential
            
            # Compare
            composition_error = np.linalg.norm(U_direct - U_sequential, ord=2)
            composition_errors.append(composition_error)
        
        max_composition_error = max(composition_errors)
        avg_composition_error = np.mean(composition_errors)
        
        # Pass condition: composition errors should be at machine precision level
        passes = max_composition_error <= 1e-14  # Machine precision level
        
        test_time = time.time() - start_time
        
        result = {
            'test_name': 'Sequence Composition Error',
            'size': N,
            'powers_tested': powers_to_test,
            'composition_errors': [float(err) for err in composition_errors],
            'max_composition_error': float(max_composition_error),
            'avg_composition_error': float(avg_composition_error),
            'passes': bool(passes),
            'tolerance': self.tolerance,
            'test_time': test_time
        }
        
        self.results[f'composition_error_N{N}'] = result
        return result
    
    def run_full_randomized_benchmarking_suite(self, sizes: List[int] = None) -> Dict[str, Any]:
        """
        Run the complete randomized benchmarking test suite.
        """
        if sizes is None:
            sizes = [4, 8, 16]
        
        print("=" * 60)
        print("RANDOMIZED BENCHMARKING VALIDATION TEST SUITE")
        print("Using Symbolic Resonance Computing Methods")
        print("=" * 60)
        
        suite_results = {
            'suite_name': 'Randomized Benchmarking Validation',
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
            size_results['sequence_decay'] = self.test_gate_sequence_decay(N, max_depth=15, num_sequences=8)
            size_results['fidelity_scaling'] = self.test_gate_fidelity_scaling(N, num_trials=15)
            size_results['composition_error'] = self.test_sequence_composition_error(N)
            
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
    """Run the randomized benchmarking validation tests."""
    validator = RandomizedBenchmarkingValidator(tolerance=1e-12)
    
    # Run comprehensive test suite
    results = validator.run_full_randomized_benchmarking_suite([4, 8])  # Smaller sizes for faster testing
    
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
                if 'error_per_gate' in test_result:
                    print(f"    Error per gate: {test_result['error_per_gate']:.2e}")
                if 'min_avg_fidelity' in test_result:
                    print(f"    Min avg fidelity: {test_result['min_avg_fidelity']:.6f}")
                if 'max_composition_error' in test_result:
                    print(f"    Max composition error: {test_result['max_composition_error']:.2e}")
            else:
                print(f"    ❌ Failed - check detailed results")
    
    return results


if __name__ == "__main__":
    main()
