#!/usr/bin/env python3
"""
RFT Mathematical Stability and Fast Algorithm Analysis === This module provides rigorous mathematical analysis of: 1. Fast RFT algorithm (comparable to FFT) 2. Stability analysis under perturbations 3. Computational complexity bounds 4. Numerical precision requirements These analyses are essential for establishing RFT as a practical transform family.
"""

import numpy as np
import scipy.linalg
import math
import time from typing
import Dict, Any, List, Tuple, Optional from 04_RFT_ALGORITHMS.canonical_true_rft import get_rft_basis, generate_phi_sequence PHI = (1.0 + math.sqrt(5.0)) / 2.0

class FastRFTAlgorithm: """
    Fast RFT algorithm with O(N log N) complexity - comparable to FFT
"""

    def __init__(self, N: int):
        self.N = N
        self.phi_sequence = generate_phi_sequence(N)
        self._precompute_twiddle_factors()
    def _precompute_twiddle_factors(self): """
        Precompute twiddle factors for fast algorithm
"""

        self.twiddle_factors = {}

        # Golden ratio twiddle factors
        for stage in range(int(np.log2(
        self.N)) + 1): stage_size = 2 ** stage twiddles = np.zeros(stage_size, dtype=np.complex128)
        for k in range(stage_size): phase = 2 * np.pi *
        self.phi_sequence[k % len(
        self.phi_sequence)] * k / stage_size twiddles[k] = np.exp(1j * phase)
        self.twiddle_factors[stage] = twiddles
    def fast_forward_rft(self, signal: np.ndarray) -> np.ndarray: """
        Fast forward RFT - O(N log N) algorithm This decomposition leverages the golden ratio structure to achieve FFT-like efficiency
        while maintaining RFT's unique properties.
"""
        N = len(signal)
        if N <= 1:
        return signal

        # Bit-reversal permutation (like FFT) signal_br =
        self._bit_reverse_permutation(signal)

        # Cooley-Tukey style decomposition with golden ratio twiddles result = signal_br.copy() num_stages = int(np.log2(N))
        for stage in range(num_stages): step_size = 2 ** (stage + 1) half_step = step_size // 2
        for start in range(0, N, step_size):
        for k in range(half_step):

        # Golden ratio twiddle factor phi_index = k % len(
        self.phi_sequence) twiddle = np.exp(1j * 2 * np.pi *
        self.phi_sequence[phi_index] * k / step_size)

        # Butterfly operation u = result[start + k] v = result[start + k + half_step] * twiddle result[start + k] = u + v result[start + k + half_step] = u - v

        # Apply Gaussian weighting and normalization gaussian_weights =
        self._compute_gaussian_weights(N) result = result * gaussian_weights
        return result / np.sqrt(N)
    def fast_inverse_rft(self, spectrum: np.ndarray) -> np.ndarray: """
        Fast inverse RFT - O(N log N) algorithm
"""

        # Conjugate, apply forward, conjugate again (like IFFT) conjugated = np.conj(spectrum) transformed =
        self.fast_forward_rft(conjugated)
        return np.conj(transformed)
    def _bit_reverse_permutation(self, data: np.ndarray) -> np.ndarray: """
        Bit-reverse permutation for fast algorithm
"""
        N = len(data) result = data.copy() j = 0
        for i in range(1, N): bit = N >> 1
        while j & bit: j ^= bit bit >>= 1 j ^= bit
        if i < j: result[i], result[j] = result[j], result[i]
        return result
    def _compute_gaussian_weights(self, N: int) -> np.ndarray: """
        Compute Gaussian weights for RFT normalization
"""
        x = np.linspace(-3, 3, N) weights = np.exp(-0.5 * x**2)
        return weights / np.linalg.norm(weights)
    def benchmark_fast_algorithm(self, sizes: List[int]) -> Dict[str, Any]: """
        Benchmark fast RFT against direct computation
"""
        results = {}
        for N in sizes:
        if N & (N - 1) != 0:

        # Skip non-power-of-2 sizes for now continue
        print(f"Benchmarking N={N}...")

        # Generate test signal test_signal = (np.random.randn(N) + 1j * np.random.randn(N))

        # Time direct computation start_time = time.time() direct_result =
        self._direct_rft(test_signal) direct_time = time.time() - start_time

        # Time fast algorithm fast_rft = FastRFTAlgorithm(N) start_time = time.time() fast_result = fast_rft.fast_forward_rft(test_signal) fast_time = time.time() - start_time

        # Accuracy comparison accuracy_error = np.linalg.norm(direct_result - fast_result) results[N] = { 'direct_time': direct_time, 'fast_time': fast_time, 'speedup': direct_time / fast_time
        if fast_time > 0 else float('inf'), 'accuracy_error': accuracy_error, 'accurate': accuracy_error < 1e-10 }
        return results
    def _direct_rft(self, signal: np.ndarray) -> np.ndarray: """
        Direct O(N^2) RFT computation for comparison
"""
        N = len(signal) basis = get_rft_basis(N)
        return basis.conj().T @ signal

class RFTStabilityAnalysis: """
        Comprehensive stability analysis for RFT
"""

    def __init__(self, N: int):
        self.N = N
        self.basis = get_rft_basis(N)
    def analyze_numerical_stability(self) -> Dict[str, Any]: """
        Comprehensive numerical stability analysis
"""
        results = {}

        # Condition number analysis results['condition_analysis'] =
        self._analyze_condition_numbers()

        # Perturbation sensitivity results['perturbation_analysis'] =
        self._analyze_perturbation_sensitivity()

        # Round-off error propagation results['roundoff_analysis'] =
        self._analyze_roundoff_errors()

        # Stability under finite precision results['precision_analysis'] =
        self._analyze_finite_precision()
        return results
    def _analyze_condition_numbers(self) -> Dict[str, Any]: """
        Analyze condition numbers and their implications
"""
        # 2-norm condition number cond_2 = np.linalg.cond(
        self.basis, 2)

        # Frobenius norm condition number cond_fro = np.linalg.cond(
        self.basis, 'fro')

        # Condition number of Gram matrix gram_matrix =
        self.basis.conj().T @
        self.basis gram_cond = np.linalg.cond(gram_matrix)

        # Spectral analysis eigenvals = np.linalg.eigvals(gram_matrix) spectral_radius = np.max(np.abs(eigenvals)) spectral_gap = np.max(np.real(eigenvals)) - np.min(np.real(eigenvals))
        return { 'condition_2_norm': cond_2, 'condition_frobenius': cond_fro, 'gram_condition': gram_cond, 'spectral_radius': spectral_radius, 'spectral_gap': spectral_gap, 'well_conditioned': cond_2 < 1e12, 'stability_assessment': 'stable'
        if cond_2 < 1e6 else 'potentially_unstable' }
    def _analyze_perturbation_sensitivity(self) -> Dict[str, Any]: """
        Analyze sensitivity to input perturbations
"""
        perturbation_levels = np.logspace(-15, -3, 13) sensitivity_results = []

        # Test signal test_signal = np.random.randn(
        self.N) + 1j * np.random.randn(
        self.N) test_signal = test_signal / np.linalg.norm(test_signal)

        # Original transform original_transform =
        self.basis.conj().T @ test_signal
        for eps in perturbation_levels:

        # Add perturbation perturbation = eps * (np.random.randn(
        self.N) + 1j * np.random.randn(
        self.N)) perturbed_signal = test_signal + perturbation

        # Transform perturbed signal perturbed_transform =
        self.basis.conj().T @ perturbed_signal

        # Compute amplification factor input_perturbation_norm = np.linalg.norm(perturbation) output_perturbation_norm = np.linalg.norm(perturbed_transform - original_transform) amplification_factor = output_perturbation_norm / input_perturbation_norm sensitivity_results.append({ 'input_perturbation': eps, 'amplification_factor': amplification_factor })

        # Compute average amplification avg_amplification = np.mean([r['amplification_factor']
        for r in sensitivity_results]) max_amplification = np.max([r['amplification_factor']
        for r in sensitivity_results])
        return { 'sensitivity_results': sensitivity_results, 'average_amplification': avg_amplification, 'maximum_amplification': max_amplification, 'stable': max_amplification < 10.0, 'theoretical_bound': np.linalg.cond(
        self.basis) }
    def _analyze_roundoff_errors(self) -> Dict[str, Any]: """
        Analyze accumulation of round-off errors
"""

        # Simulate different precision levels precision_levels = [np.float16, np.float32, np.float64] roundoff_results = {} test_signal = np.random.randn(
        self.N) + 1j * np.random.randn(
        self.N)

        # Reference computation (highest precision) reference_result =
        self.basis.conj().T @ test_signal.astype(np.complex128)
        for dtype in precision_levels:

        # Convert to lower precision low_precision_signal = test_signal.astype(dtype).astype(np.complex128) low_precision_basis =
        self.basis.astype(dtype).astype(np.complex128)

        # Compute with reduced precision low_precision_result = low_precision_basis.conj().T @ low_precision_signal

        # Measure error roundoff_error = np.linalg.norm(reference_result - low_precision_result) relative_error = roundoff_error / np.linalg.norm(reference_result) roundoff_results[str(dtype)] = { 'absolute_error': roundoff_error, 'relative_error': relative_error, 'acceptable': relative_error < 1e-6 }
        return roundoff_results
    def _analyze_finite_precision(self) -> Dict[str, Any]: """
        Analyze behavior under finite precision arithmetic
"""

        # Test unitarity preservation under finite precision precision_tests = {}

        # Single precision test basis_single =
        self.basis.astype(np.complex64) unitarity_error_single = np.linalg.norm( basis_single @ basis_single.conj().T - np.eye(
        self.N), 'fro' )

        # Double precision test basis_double =
        self.basis.astype(np.complex128) unitarity_error_double = np.linalg.norm( basis_double @ basis_double.conj().T - np.eye(
        self.N), 'fro' ) precision_tests['single_precision'] = { 'unitarity_error': unitarity_error_single, 'acceptable': unitarity_error_single < 1e-5 } precision_tests['double_precision'] = { 'unitarity_error': unitarity_error_double, 'acceptable': unitarity_error_double < 1e-12 }
        return precision_tests

class RFTComplexityAnalysis: """
        Computational complexity analysis for RFT
"""

    def __init__(self): pass
    def analyze_computational_complexity(self) -> Dict[str, Any]: """
        Comprehensive complexity analysis
"""

        return { 'direct_complexity':
        self._analyze_direct_complexity(), 'fast_complexity':
        self._analyze_fast_complexity(), 'memory_complexity':
        self._analyze_memory_complexity(), 'parallel_complexity':
        self._analyze_parallel_complexity() }
    def _analyze_direct_complexity(self) -> Dict[str, Any]: """
        Analyze direct O(N^2) computation complexity
"""

        # Time complexity measurements sizes = [8, 16, 32, 64, 128, 256] times = []
        for N in sizes:
        if N > 128:

        # Skip large sizes for direct computation continue test_signal = np.random.randn(N) + 1j * np.random.randn(N) basis = get_rft_basis(N) start_time = time.time() result = basis.conj().T @ test_signal elapsed_time = time.time() - start_time times.append(elapsed_time)

        # Fit to O(N^2) model
        if len(times) >= 3: valid_sizes = sizes[:len(times)] coeffs = np.polyfit([N**2
        for N in valid_sizes], times, 1) complexity_constant = coeffs[0]
        else: complexity_constant = 0
        return { 'sizes_tested': sizes[:len(times)], 'execution_times': times, 'complexity_constant': complexity_constant, 'theoretical_complexity': 'O(N^2)', 'operations_count': lambda N: N**2

        # Complex multiplications }
    def _analyze_fast_complexity(self) -> Dict[str, Any]: """
        Analyze fast O(N log N) algorithm complexity
"""
        sizes = [8, 16, 32, 64, 128, 256, 512] fast_times = []
        for N in sizes:
        if N & (N - 1) != 0:

        # Skip non-power-of-2 continue test_signal = np.random.randn(N) + 1j * np.random.randn(N) fast_rft = FastRFTAlgorithm(N) start_time = time.time() result = fast_rft.fast_forward_rft(test_signal) elapsed_time = time.time() - start_time fast_times.append(elapsed_time)

        # Fit to O(N log N) model
        if len(fast_times) >= 3: valid_sizes = [s
        for s in sizes
        if s & (s - 1) == 0][:len(fast_times)] log_terms = [N * np.log2(N)
        for N in valid_sizes] coeffs = np.polyfit(log_terms, fast_times, 1) fast_complexity_constant = coeffs[0]
        else: fast_complexity_constant = 0
        return { 'sizes_tested': [s
        for s in sizes
        if s & (s - 1) == 0][:len(fast_times)], 'execution_times': fast_times, 'complexity_constant': fast_complexity_constant, 'theoretical_complexity': 'O(N log N)', 'operations_count': lambda N: N * int(np.log2(N)) }
    def _analyze_memory_complexity(self) -> Dict[str, Any]: """
        Analyze memory requirements
"""
        sizes = [8, 16, 32, 64, 128, 256] memory_requirements = {}
        for N in sizes:

        # Direct method memory basis_memory = N * N * 16

        # Complex128 = 16 bytes signal_memory = N * 16 result_memory = N * 16 direct_total = basis_memory + signal_memory + result_memory

        # Fast method memory (no need to store full basis) twiddle_memory = N * np.log2(N) * 16
        if N & (N - 1) == 0 else N * N * 16 temp_memory = N * 16 * 2

        # Working arrays fast_total = twiddle_memory + temp_memory + signal_memory + result_memory memory_requirements[N] = { 'direct_method_bytes': direct_total, 'fast_method_bytes': fast_total, 'memory_savings': (direct_total - fast_total) / direct_total
        if direct_total > 0 else 0 }
        return memory_requirements
    def _analyze_parallel_complexity(self) -> Dict[str, Any]: """
        Analyze parallelization potential
"""

        return { 'direct_method': { 'parallel_complexity': 'O(N)',

        # N parallel dot products 'parallelization_efficiency': 'High', 'memory_bandwidth_bound': True }, 'fast_method': { 'parallel_complexity': 'O(log N)', # log N stages, each parallelizable 'parallelization_efficiency': 'Moderate', 'synchronization_overhead': 'Significant' }, 'recommended_approach': 'Hybrid: parallel direct for small N, fast for large N' }
    def generate_comprehensive_stability_report() -> Dict[str, Any]: """
        Generate comprehensive stability and algorithm analysis
"""

        print("RFT Stability and Fast Algorithm Analysis")
        print("=" * 50) results = {} test_sizes = [16, 32, 64, 128]
        for N in test_sizes:
        print(f"\nAnalyzing N={N}...")

        # Stability analysis stability_analyzer = RFTStabilityAnalysis(N) stability_results = stability_analyzer.analyze_numerical_stability() results[f'stability_N{N}'] = stability_results

        # Fast algorithm benchmarking
        print("\nBenchmarking fast algorithm...") fast_rft = FastRFTAlgorithm(64)

        # Representative size benchmark_results = fast_rft.benchmark_fast_algorithm([16, 32, 64, 128, 256]) results['fast_algorithm_benchmark'] = benchmark_results

        # Complexity analysis
        print("\nAnalyzing computational complexity...") complexity_analyzer = RFTComplexityAnalysis() complexity_results = complexity_analyzer.analyze_computational_complexity() results['complexity_analysis'] = complexity_results

        # Overall assessment all_stable = True fast_algorithm_viable = True for size_key, stability_data in results.items(): if 'stability_N' in size_key:
        if not stability_data['condition_analysis']['well_conditioned']: all_stable = False
        if not stability_data['perturbation_analysis']['stable']: all_stable = False for size, benchmark_data in benchmark_results.items():
        if not benchmark_data['accurate']: fast_algorithm_viable = False results['overall_assessment'] = { 'numerically_stable': all_stable, 'fast_algorithm_viable': fast_algorithm_viable, 'production_ready': all_stable and fast_algorithm_viable, 'comparable_to_fft': fast_algorithm_viable, 'timestamp': time.time() }
        return results

if __name__ == "__main__":

# Generate comprehensive analysis analysis_results = generate_comprehensive_stability_report()

# Print summary assessment = analysis_results['overall_assessment']
print(f"\n" + "="*60)
print(f"FINAL ASSESSMENT:")
print(f"Numerically Stable: {assessment['numerically_stable']}")
print(f"Fast Algorithm Viable: {assessment['fast_algorithm_viable']}")
print(f"Production Ready: {assessment['production_ready']}")
print(f"Comparable to FFT: {assessment['comparable_to_fft']}")
if assessment['production_ready']:
print("\n✅ SUCCESS: RFT has proven stability and fast algorithms!")
print("✅ O(N log N) complexity achieved")
print("✅ Numerical stability verified")
print("✅ Ready for practical implementation")
else:
print("\n⚠️ Some stability or algorithm issues detected")

# Save results
import json
def serialize_for_json(obj):
        if isinstance(obj, np.ndarray):
        return obj.tolist()
        el
        if isinstance(obj, np.integer):
        return int(obj)
        el
        if isinstance(obj, np.floating):
        return float(obj)
        el
        if isinstance(obj, np.bool_):
        return bool(obj)
        el
        if isinstance(obj, np.complex128):
        return {'real': float(obj.real), 'imag': float(obj.imag)}
        el
        if callable(obj):
        return str(obj)
        el
        if isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
        el
        if isinstance(obj, list):
        return [serialize_for_json(item)
        for item in obj]
        else:
        return obj serializable_results = serialize_for_json(analysis_results) with open('rft_stability_analysis.json', 'w') as f: json.dump(serializable_results, f, indent=2)
        print(f"\nDetailed analysis saved to 'rft_stability_analysis.json'")