#!/usr/bin/env python3
"""
RFT Parameter Space Exploration for Enhanced Distinctness === This module explores variations in RFT construction parameters to maximize distinctness from existing transforms and establish novel properties.
"""

import numpy as np
import math
import time
import json from typing
import Dict, Any, List, Tuple, Optional from 04_RFT_ALGORITHMS.canonical_true_rft import generate_phi_sequence PHI = (1.0 + math.sqrt(5.0)) / 2.0

class RFTParameterExploration: """
    Explore RFT parameter space for enhanced transform family properties
"""

    def __init__(self, N: int = 64):
        self.N = N
    def explore_parameter_space(self) -> Dict[str, Any]: """
        Comprehensive exploration of RFT parameter variations
"""

        print(f"RFT Parameter Space Exploration for Enhanced Distinctness")
        print(f"=" * 65) exploration_results = {} # 1. Golden ratio variations
        print("\n1. Exploring golden ratio parameter variations...") exploration_results['phi_variations'] =
        self._explore_phi_variations() # 2. Gaussian kernel parameter variations
        print("\n2. Exploring Gaussian kernel parameters...") exploration_results['gaussian_variations'] =
        self._explore_gaussian_parameters() # 3. LDS sequence modifications
        print("\n3. Exploring LDS sequence modifications...") exploration_results['lds_variations'] =
        self._explore_lds_variations() # 4. Multi-scale constructions
        print("\n4. Exploring multi-scale constructions...") exploration_results['multiscale_variations'] =
        self._explore_multiscale_constructions() # 5. Hybrid parameterizations
        print("\n5. Exploring hybrid parameterizations...") exploration_results['hybrid_variations'] =
        self._explore_hybrid_parameters()

        # Analyze best configurations best_config =
        self._find_optimal_configuration(exploration_results) exploration_results['optimal_configuration'] = best_config
        return exploration_results
    def _explore_phi_variations(self) -> Dict[str, Any]: """
        Explore variations of golden ratio parameterization
"""
        phi_variations = {}

        # Base golden ratio powers phi_powers = [PHI, PHI**2, PHI**0.5, PHI**(-1), PHI**3] for i, phi_val in enumerate(phi_powers):
        print(f" Testing φ^{[1, 2, 0.5, -1, 3][i]} = {phi_val:.6f}") config = { 'phi_base': phi_val, 'sequence_type': 'powers', 'gaussian_sigma': 1.0, 'beta': 2.0 } analysis =
        self._analyze_parameter_configuration(config) phi_variations[f'phi_power_{[1, 2, 0.5, -1, 3][i]}'] = analysis

        # Fibonacci-based variations fibonacci_bases = [ 1.618033988749895,

        # Standard φ 2.618033988749895, # φ + 1 0.618033988749895, # φ - 1 1.324717957244746, # ∛φ ] for i, fib_base in enumerate(fibonacci_bases):
        print(f" Testing Fibonacci variation {i+1}: {fib_base:.6f}") config = { 'phi_base': fib_base, 'sequence_type': 'fibonacci_variant', 'gaussian_sigma': 1.0, 'beta': 2.0 } analysis =
        self._analyze_parameter_configuration(config) phi_variations[f'fibonacci_variant_{i+1}'] = analysis

        # Metallic means metallic_means = [ (1 + math.sqrt(5)) / 2,

        # Golden ratio (1 + math.sqrt(8)) / 2,

        # Silver ratio (1 + math.sqrt(13)) / 2,

        # Bronze ratio ] for i, metal in enumerate(metallic_means): metal_name = ['golden', 'silver', 'bronze'][i]
        print(f" Testing {metal_name} ratio: {metal:.6f}") config = { 'phi_base': metal, 'sequence_type': 'metallic_mean', 'gaussian_sigma': 1.0, 'beta': 2.0 } analysis =
        self._analyze_parameter_configuration(config) phi_variations[f'{metal_name}_ratio'] = analysis
        return phi_variations
    def _explore_gaussian_parameters(self) -> Dict[str, Any]: """
        Explore Gaussian kernel parameter variations
"""
        gaussian_variations = {}

        # Sigma variations sigma_values = [0.5, 1.0, 2.0, 4.0,
        self.N/8,
        self.N/4]
        for sigma in sigma_values:
        print(f" Testing σ = {sigma}") config = { 'phi_base': PHI, 'sequence_type': 'powers', 'gaussian_sigma': sigma, 'beta': 2.0 } analysis =
        self._analyze_parameter_configuration(config) gaussian_variations[f'sigma_{sigma}'] = analysis

        # Different kernel types kernel_types = [ ('gaussian', lambda x, sigma: np.exp(-0.5 * (x/sigma)**2)), ('exponential', lambda x, sigma: np.exp(-abs(x)/sigma)), ('cauchy', lambda x, sigma: 1 / (1 + (x/sigma)**2)), ('sinc', lambda x, sigma: np.sinc(x/sigma)), ] for kernel_name, kernel_func in kernel_types:
        print(f" Testing {kernel_name} kernel") config = { 'phi_base': PHI, 'sequence_type': 'powers', 'gaussian_sigma': 1.0, 'beta': 2.0, 'kernel_type': kernel_name, 'kernel_func': kernel_func } analysis =
        self._analyze_parameter_configuration(config) gaussian_variations[f'{kernel_name}_kernel'] = analysis
        return gaussian_variations
    def _explore_lds_variations(self) -> Dict[str, Any]: """
        Explore LDS sequence modifications
"""
        lds_variations = {}

        # Different sequence generation methods sequence_methods = [ 'powers', # φ^n 'fibonacci',

        # F_n 'lucas',

        # L_n 'tribonacci',

        # T_n 'padovan',

        # P_n ]
        for method in sequence_methods:
        print(f" Testing {method} sequence") config = { 'phi_base': PHI, 'sequence_type': method, 'gaussian_sigma': 1.0, 'beta': 2.0 } analysis =
        self._analyze_parameter_configuration(config) lds_variations[f'{method}_sequence'] = analysis

        # Modulated sequences modulation_types = [ ('none', lambda seq: seq), ('sine', lambda seq: seq * np.sin(np.pi * np.arange(len(seq)) / len(seq))), ('cosine', lambda seq: seq * np.cos(np.pi * np.arange(len(seq)) / len(seq))), ('chirp', lambda seq: seq * np.sin(2 * np.pi * np.arange(len(seq))**2 / len(seq)**2)), ] for mod_name, mod_func in modulation_types:
        if mod_name == 'none': continue
        print(f" Testing {mod_name} modulated sequence") config = { 'phi_base': PHI, 'sequence_type': 'powers', 'gaussian_sigma': 1.0, 'beta': 2.0, 'modulation': mod_name, 'modulation_func': mod_func } analysis =
        self._analyze_parameter_configuration(config) lds_variations[f'{mod_name}_modulated'] = analysis
        return lds_variations
    def _explore_multiscale_constructions(self) -> Dict[str, Any]: """
        Explore multi-scale construction approaches
"""
        multiscale_variations = {}

        # Multi-resolution approaches scales = [ [1],

        # Single scale [1, 2],

        # Two scales [1, 2, 4],

        # Three scales [1, 3, 9],

        # Powers of 3 ] for i, scale_set in enumerate(scales):
        if len(scale_set) == 1: continue
        print(f" Testing multi-scale {scale_set}") config = { 'phi_base': PHI, 'sequence_type': 'powers', 'gaussian_sigma': 1.0, 'beta': 2.0, 'multiscale': True, 'scales': scale_set } analysis =
        self._analyze_parameter_configuration(config) multiscale_variations[f'scales_{scale_set}'] = analysis

        # Wavelet-inspired constructions wavelet_types = [ 'daubechies', 'biorthogonal', 'coiflets', ]
        for wavelet in wavelet_types:
        print(f" Testing {wavelet}-inspired construction") config = { 'phi_base': PHI, 'sequence_type': 'powers', 'gaussian_sigma': 1.0, 'beta': 2.0, 'wavelet_inspired': True, 'wavelet_type': wavelet } analysis =
        self._analyze_parameter_configuration(config) multiscale_variations[f'{wavelet}_inspired'] = analysis
        return multiscale_variations
    def _explore_hybrid_parameters(self) -> Dict[str, Any]: """
        Explore hybrid parameterization approaches
"""
        hybrid_variations = {}

        # Combine multiple mathematical constants constant_combinations = [ ('phi_e', [PHI, math.e]), ('phi_pi', [PHI, math.pi]), ('phi_sqrt2', [PHI, math.sqrt(2)]), ('all_constants', [PHI, math.e, math.pi, math.sqrt(2)]), ] for name, constants in constant_combinations:
        print(f" Testing {name} combination") config = { 'phi_base': PHI, 'sequence_type': 'hybrid', 'constants': constants, 'gaussian_sigma': 1.0, 'beta': 2.0 } analysis =
        self._analyze_parameter_configuration(config) hybrid_variations[name] = analysis

        # Adaptive parameter schemes adaptive_schemes = [ 'dimension_scaled',

        # Parameters scale with N 'prime_based',

        # Use prime numbers 'random_stable',

        # Controlled randomness ]
        for scheme in adaptive_schemes:
        print(f" Testing {scheme} adaptive scheme") config = { 'phi_base': PHI, 'sequence_type': 'powers', 'gaussian_sigma': 1.0, 'beta': 2.0, 'adaptive_scheme': scheme } analysis =
        self._analyze_parameter_configuration(config) hybrid_variations[f'{scheme}_adaptive'] = analysis
        return hybrid_variations
    def _analyze_parameter_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]: """
        Analyze a specific parameter configuration
"""

        # Generate RFT basis with this configuration
        try: rft_basis =
        self._generate_rft_with_config(config)

        # Analyze properties analysis = { 'construction_successful': True, 'unitarity':
        self._check_unitarity(rft_basis), 'distinctness':
        self._compute_distinctness(rft_basis), 'stability':
        self._check_stability(rft_basis), 'novelty_metrics':
        self._compute_novelty_metrics(rft_basis, config), 'overall_score': 0.0 }

        # Compute overall score analysis['overall_score'] =
        self._compute_configuration_score(analysis) except Exception as e: analysis = { 'construction_successful': False, 'error': str(e), 'overall_score': 0.0 }
        return analysis
    def _generate_rft_with_config(self, config: Dict[str, Any]) -> np.ndarray: """
        Generate RFT basis with specific configuration
"""

        # Generate phase sequence based on configuration phi_sequence =
        self._generate_sequence_with_config(config)

        # Create LDS matrix lds_matrix = np.zeros((
        self.N,
        self.N), dtype=np.complex128) sigma = config.get('gaussian_sigma', 1.0) beta = config.get('beta', 2.0) kernel_func = config.get('kernel_func', lambda x, s: np.exp(-0.5 * (x/s)**2))
        for i in range(
        self.N):
        for j in range(
        self.N):

        # Phase from sequence phase_idx = (i + j) % len(phi_sequence) phase = 2 * np.pi * phi_sequence[phase_idx]

        # Kernel function
        if config.get('multiscale', False):

        # Multi-scale kernel kernel_val = 0 scales = config.get('scales', [1])
        for scale in scales: kernel_val += kernel_func((i - j) / scale, sigma) / len(scales)
        else:

        # Single-scale kernel kernel_val = kernel_func(i - j, sigma)

        # Apply modulation
        if specified if 'modulation_func' in config: mod_val = config['modulation_func'](np.array([phase]))[0] phase = phase * mod_val

        # LDS element lds_matrix[i, j] = kernel_val * np.exp(1j * phase) / beta

        # QR decomposition Q, R = np.linalg.qr(lds_matrix)
        return Q
    def _generate_sequence_with_config(self, config: Dict[str, Any]) -> np.ndarray: """
        Generate phase sequence with specific configuration
"""
        sequence_type = config.get('sequence_type', 'powers') phi_base = config.get('phi_base', PHI)
        if sequence_type == 'powers': sequence = np.array([phi_base**i
        for i in range(
        self.N)])
        el
        if sequence_type == 'fibonacci': sequence =
        self._generate_fibonacci_sequence(
        self.N)
        el
        if sequence_type == 'fibonacci_variant': sequence =
        self._generate_fibonacci_variant(
        self.N, phi_base)
        el
        if sequence_type == 'lucas': sequence =
        self._generate_lucas_sequence(
        self.N)
        el
        if sequence_type == 'tribonacci': sequence =
        self._generate_tribonacci_sequence(
        self.N)
        el
        if sequence_type == 'padovan': sequence =
        self._generate_padovan_sequence(
        self.N)
        el
        if sequence_type == 'hybrid': constants = config.get('constants', [PHI]) sequence =
        self._generate_hybrid_sequence(
        self.N, constants)
        else:

        # Default to golden ratio powers sequence = np.array([phi_base**i
        for i in range(
        self.N)])

        # Apply adaptive scheme
        if specified if 'adaptive_scheme' in config: sequence =
        self._apply_adaptive_scheme(sequence, config['adaptive_scheme'])
        return sequence
    def _generate_fibonacci_sequence(self, n: int) -> np.ndarray: """
        Generate Fibonacci sequence
"""
        fib = np.zeros(n)
        if n > 0: fib[0] = 1
        if n > 1: fib[1] = 1
        for i in range(2, n): fib[i] = fib[i-1] + fib[i-2]
        return fib
    def _generate_fibonacci_variant(self, n: int, base: float) -> np.ndarray: """
        Generate Fibonacci-like sequence with different base
"""

        # Use generalized Fibonacci with custom ratio ratio = base fib = np.zeros(n)
        if n > 0: fib[0] = 1
        if n > 1: fib[1] = ratio
        for i in range(2, n): fib[i] = ratio * fib[i-1] + fib[i-2]
        return fib
    def _generate_lucas_sequence(self, n: int) -> np.ndarray: """
        Generate Lucas sequence
"""
        lucas = np.zeros(n)
        if n > 0: lucas[0] = 2
        if n > 1: lucas[1] = 1
        for i in range(2, n): lucas[i] = lucas[i-1] + lucas[i-2]
        return lucas
    def _generate_tribonacci_sequence(self, n: int) -> np.ndarray: """
        Generate Tribonacci sequence
"""
        trib = np.zeros(n)
        if n > 0: trib[0] = 1
        if n > 1: trib[1] = 1
        if n > 2: trib[2] = 2
        for i in range(3, n): trib[i] = trib[i-1] + trib[i-2] + trib[i-3]
        return trib
    def _generate_padovan_sequence(self, n: int) -> np.ndarray: """
        Generate Padovan sequence
"""
        pad = np.zeros(n)
        if n > 0: pad[0] = 1
        if n > 1: pad[1] = 1
        if n > 2: pad[2] = 1
        for i in range(3, n): pad[i] = pad[i-2] + pad[i-3]
        return pad
    def _generate_hybrid_sequence(self, n: int, constants: List[float]) -> np.ndarray: """
        Generate hybrid sequence combining multiple constants
"""
        sequence = np.zeros(n)
        for i in range(n): val = 0 for j, const in enumerate(constants): val += const**(i + j) / len(constants) sequence[i] = val
        return sequence
    def _apply_adaptive_scheme(self, sequence: np.ndarray, scheme: str) -> np.ndarray: """
        Apply adaptive parameter scheme
"""

        if scheme == 'dimension_scaled':

        # Scale with dimension scale_factor = math.log(
        self.N) / math.log(64)

        # Normalize to N=64
        return sequence * scale_factor
        el
        if scheme == 'prime_based':

        # Modulate with prime numbers primes =
        self._get_primes(len(sequence))
        return sequence * np.array(primes[:len(sequence)])
        el
        if scheme == 'random_stable':

        # Add controlled randomness np.random.seed(42)

        # Stable randomness noise = 0.1 * np.random.randn(len(sequence))
        return sequence * (1 + noise)
        return sequence
    def _get_primes(self, n: int) -> List[int]: """
        Get first n prime numbers
"""
        primes = [] candidate = 2
        while len(primes) < n: is_prime = True
        for p in primes:
        if p * p > candidate: break
        if candidate % p == 0: is_prime = False break
        if is_prime: primes.append(candidate) candidate += 1
        return primes
    def _check_unitarity(self, basis: np.ndarray) -> Dict[str, Any]: """
        Check unitarity of basis
"""
        gram = basis.conj().T @ basis identity_error = np.linalg.norm(gram - np.eye(
        self.N), 'fro')
        return { 'identity_error': identity_error, 'is_unitary': identity_error < 1e-10 }
    def _compute_distinctness(self, basis: np.ndarray) -> Dict[str, Any]: """
        Compute distinctness from standard transforms
"""

        # DFT comparison dft_matrix = np.fft.fft(np.eye(
        self.N)) / np.sqrt(
        self.N) dft_corr =
        self._max_correlation(basis, dft_matrix)

        # DCT comparison dct_matrix =
        self._generate_dct_matrix() dct_corr =
        self._max_correlation(basis, dct_matrix) distinctness_score = (2.0 - dft_corr - dct_corr) / 2.0
        return { 'dft_correlation': dft_corr, 'dct_correlation': dct_corr, 'distinctness_score': distinctness_score, 'is_distinct': distinctness_score > 0.2 }
    def _max_correlation(self, basis1: np.ndarray, basis2: np.ndarray) -> float: """
        Compute maximum correlation between bases
"""
        max_corr = 0.0 sample_size = min(16, basis1.shape[1])

        # Sample
        for efficiency indices = np.random.choice(basis1.shape[1], size=sample_size, replace=False)
        for i in indices:
        for j in indices: corr = abs(np.vdot(basis1[:, i], basis2[:, j])) max_corr = max(max_corr, corr)
        return max_corr
    def _generate_dct_matrix(self) -> np.ndarray: """
        Generate DCT matrix
"""
        dct_matrix = np.zeros((
        self.N,
        self.N))
        for k in range(
        self.N):
        for n in range(
        self.N):
        if k == 0: dct_matrix[k, n] = 1.0 / np.sqrt(
        self.N)
        else: dct_matrix[k, n] = np.sqrt(2.0/
        self.N) * np.cos(np.pi * k * (2*n + 1) / (2*
        self.N))
        return dct_matrix.astype(np.complex128)
    def _check_stability(self, basis: np.ndarray) -> Dict[str, Any]: """
        Check numerical stability
"""
        condition_number = np.linalg.cond(basis)
        return { 'condition_number': condition_number, 'is_stable': condition_number < 1e6 }
    def _compute_novelty_metrics(self, basis: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]: """
        Compute metrics for novelty assessment
"""

        # Spectral entropy eigenvals = np.linalg.eigvals(basis @ basis.conj().T) eigenvals_real = np.real(eigenvals) eigenvals_normalized = eigenvals_real / np.sum(eigenvals_real) spectral_entropy = -np.sum(eigenvals_normalized * np.log(eigenvals_normalized + 1e-15))

        # Parameter complexity param_complexity = len([k
        for k in config.keys()
        if not k.endswith('_func')])

        # Golden ratio content phi_base = config.get('phi_base', PHI) phi_deviation = abs(phi_base - PHI)
        return { 'spectral_entropy': spectral_entropy, 'parameter_complexity': param_complexity, 'phi_deviation': phi_deviation, 'novelty_score': spectral_entropy * math.log(param_complexity + 1) * (1 + phi_deviation) }
    def _compute_configuration_score(self, analysis: Dict[str, Any]) -> float: """
        Compute overall score for configuration
"""

        if not analysis['construction_successful']:
        return 0.0

        # Weight different factors weights = { 'unitarity': 0.3, 'distinctness': 0.4, 'stability': 0.2, 'novelty': 0.1 } scores = {} scores['unitarity'] = 1.0
        if analysis['unitarity']['is_unitary'] else 0.0 scores['distinctness'] = analysis['distinctness']['distinctness_score'] scores['stability'] = 1.0
        if analysis['stability']['is_stable'] else 0.0 scores['novelty'] = min(analysis['novelty_metrics']['novelty_score'] / 10.0, 1.0) overall_score = sum(weights[k] * scores[k]
        for k in weights.keys())
        return overall_score
    def _find_optimal_configuration(self, exploration_results: Dict[str, Any]) -> Dict[str, Any]: """
        Find the optimal parameter configuration
"""
        best_score = 0.0 best_config = None best_category = None for category, variations in exploration_results.items():
        if category == 'optimal_configuration': continue for config_name, analysis in variations.items():
        if analysis.get('overall_score', 0.0) > best_score: best_score = analysis['overall_score'] best_config = config_name best_category = category
        return { 'best_category': best_category, 'best_configuration': best_config, 'best_score': best_score, 'recommendation': f"Use {best_config} from {best_category} category for enhanced distinctness" }
    def run_parameter_exploration(): """
        Run comprehensive parameter space exploration
"""

        # Test with different dimensions test_dimensions = [32, 64, 128] all_results = {}
        for N in test_dimensions:
        print(f"\n{'='*80}")
        print(f"PARAMETER EXPLORATION FOR N={N}")
        print(f"{'='*80}") explorer = RFTParameterExploration(N) results = explorer.explore_parameter_space() all_results[f'N_{N}'] = results

        # Print optimal configuration for this size optimal = results['optimal_configuration']
        print(f"\n🏆 OPTIMAL CONFIGURATION FOR N={N}:")
        print(f" Category: {optimal['best_category']}")
        print(f" Configuration: {optimal['best_configuration']}")
        print(f" Score: {optimal['best_score']:.4f}")

        # Find overall best approach global_best_score = 0.0 global_best_config = None for size_key, size_results in all_results.items(): optimal = size_results['optimal_configuration']
        if optimal['best_score'] > global_best_score: global_best_score = optimal['best_score'] global_best_config = (size_key, optimal)
        print(f"\n{'='*80}")
        print(f"GLOBAL OPTIMAL CONFIGURATION")
        print(f"{'='*80}")
        if global_best_config: size, config = global_best_config
        print(f"Dimension: {size}")
        print(f"Category: {config['best_category']}")
        print(f"Configuration: {config['best_configuration']}")
        print(f"Score: {config['best_score']:.4f}")
        print(f"Recommendation: {config['recommendation']}")

        # Save results with open('rft_parameter_exploration.json', 'w') as f:
    def serialize_numpy(obj):
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
        if isinstance(obj, dict):
        return {k: serialize_numpy(v) for k, v in obj.items()}
        el
        if isinstance(obj, list):
        return [serialize_numpy(item)
        for item in obj]
        else:
        return obj

        # Remove function objects before serialization clean_results = {} for size_key, size_results in all_results.items(): clean_size_results = {} for category, variations in size_results.items():
        if category == 'optimal_configuration': clean_size_results[category] = variations
        else: clean_variations = {} for config_name, analysis in variations.items():

        # Remove function objects clean_analysis = {k: v for k, v in analysis.items()
        if not callable(v) and not k.endswith('_func')} clean_variations[config_name] = clean_analysis clean_size_results[category] = clean_variations clean_results[size_key] = clean_size_results serializable_results = serialize_numpy(clean_results) json.dump(serializable_results, f, indent=2)
        print(f"\n📄 Detailed results saved to 'rft_parameter_exploration.json'")
        return all_results

if __name__ == "__main__": run_parameter_exploration()