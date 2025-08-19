||#!/usr/bin/env python3
"""
RFT Ablation Study - Isolating the Mathematical Contributions This implements the ablation controls requested: 1. DFT limit (M=1, phi===1) 2. No-resonance ablation (C_sigma -> I) 3. Parameter sensitivity analysis 4. Bootstrap confidence intervals Scientific goal: Understand what components of RFT actually contribute to performance.
"""
"""

import numpy as np from scipy
import stats from sklearn.utils
import resample
import warnings warnings.filterwarnings('ignore') from canonical_true_rft
import forward_true_rft, inverse_true_rft

# Legacy wrapper maintained for: forward_true_rft, inverse_true_rft

class RFTAblationVariants:
"""
"""
    Different RFT variants for ablation study
"""
"""
    @staticmethod
    def vanilla_dft(x):
"""
"""
        Pure DFT baseline (what RFT reduces to with M=1, phi===0)
"""
"""

        return np.fft.fft(x, norm='ortho') @staticmethod
    def rft_no_resonance(x):
"""
"""
        RFT without resonance operators (C_sigma -> I)
"""
"""

        # This is a simplified version that skips the core resonance step

        # In practice, this would be DFT with just phase shifts n = len(x) phases = np.exp(1j * np.random.uniform(0, 2*np.pi, n))

        # Random phases X_dft = np.fft.fft(x, norm='ortho')
        return phases * X_dft

        # Phase modulation only @staticmethod
    def rft_fixed_phases(x, phi_value=np.pi/4):
"""
"""
        RFT with fixed phases (no adaptive phi)
"""
"""

        # This tests
        if adaptive phase selection matters n = len(x) phases = np.exp(1j * phi_value * np.ones(n)) X_dft = np.fft.fft(x, norm='ortho')

        # Apply some resonance-like operation but with fixed phases resonance_weights = np.exp(-0.1 * np.arange(n))
        return phases * resonance_weights * X_dft @staticmethod
    def rft_random_weights(x, seed=42):
"""
"""
        RFT with randomized resonance weights (test
        if structure matters)
"""
"""
        np.random.seed(seed) n = len(x) X_dft = np.fft.fft(x, norm='ortho')

        # Random weights instead of structured resonance random_weights = np.random.exponential(1.0, n) + 0.1 random_weights = random_weights / np.linalg.norm(random_weights) * np.sqrt(n)
        return random_weights * X_dft
    def generate_parameter_sweep_signals(t, resonance_strength=1.0, coupling_factor=0.5):
"""
"""
        Generate test signals with controllable resonance parameters
"""
"""
        f_base = 20 f_mod = 2.5

        # Controllable resonance structure amplitude_env = 1.0 + resonance_strength * 0.6 * np.sin(2 * np.pi * f_mod * t) phase_coupling = coupling_factor * 0.8 * np.sin(2 * np.pi * f_mod * t) signal_clean = amplitude_env * np.sin(2 * np.pi * f_base * t + phase_coupling)

        # Add noise noise = np.random.normal(0, 0.2 * np.std(signal_clean), len(t))
        return signal_clean + noise, signal_clean
    def bootstrap_confidence_interval(data, n_bootstrap=1000, confidence=0.95):
"""
"""
        Compute bootstrap confidence intervals
"""
"""
        bootstrap_means = []
        for _ in range(n_bootstrap): sample = resample(data, replace=True) bootstrap_means.append(np.mean(sample)) alpha = 1 - confidence lower = np.percentile(bootstrap_means, 100 * alpha/2) upper = np.percentile(bootstrap_means, 100 * (1 - alpha/2))
        return np.mean(data), lower, upper
    def compute_sparsity_simple(coeffs):
"""
"""
        Simple sparsity measure for ablation study
"""
"""
        coeffs_abs = np.abs(coeffs) coeffs_norm = coeffs_abs / (np.linalg.norm(coeffs_abs) + 1e-12)

        # L1/L2 ratio l1 = np.sum(coeffs_norm) l2 = np.linalg.norm(coeffs_norm) sparsity = l1 / (np.sqrt(len(coeffs_norm)) * l2)
        if l2 > 0 else 0
        return sparsity
    def run_ablation_study():
"""
"""
        Comprehensive ablation study
"""
"""
        print("=" * 80)
        print("RFT ABLATION STUDY - ISOLATING MATHEMATICAL CONTRIBUTIONS")
        print("=" * 80)
        print()
        print("🔬 TESTING COMPONENTS:")
        print(" 1. Full RFT (all operators)")
        print(" 2. DFT limit (M=1, phi===0)")
        print(" 3. No-resonance RFT (C_sigma -> I)")
        print(" 4. Fixed-phase RFT (non-adaptive phi)")
        print(" 5. Random-weight RFT (unstructured)")
        print()

        # Test setup n_trials = 25 signal_length = 256 fs = 1000 t = np.linspace(0, signal_length/fs, signal_length)

        # Storage for results variants = { 'Full RFT': [], 'DFT Limit': [], 'No Resonance': [], 'Fixed Phase': [], 'Random Weights': [] } reconstruction_quality = {key: []
        for key in variants.keys()}
        print("🧪 Running ablation trials...")
        for trial in range(n_trials):

        # Generate test signal (moderate resonance structure) test_signal, _ = generate_parameter_sweep_signals(t, resonance_strength=1.0, coupling_factor=0.7) # 1. Full RFT rft_full = forward_true_rft(test_signal) rft_recon = inverse_true_rft(rft_full) reconstruction_quality['Full RFT'].append( np.linalg.norm(test_signal - rft_recon) ) variants['Full RFT'].append(compute_sparsity_simple(rft_full)) # 2. DFT Limit dft_coeffs = RFTAblationVariants.vanilla_dft(test_signal) variants['DFT Limit'].append(compute_sparsity_simple(dft_coeffs)) reconstruction_quality['DFT Limit'].append( np.linalg.norm(test_signal - np.fft.ifft(dft_coeffs, norm='ortho').real) ) # 3. No-resonance RFT no_res_coeffs = RFTAblationVariants.rft_no_resonance(test_signal) variants['No Resonance'].append(compute_sparsity_simple(no_res_coeffs)) reconstruction_quality['No Resonance'].append(

        # Approximate reconstruction (this variant isn't perfectly invertible) np.linalg.norm(test_signal - np.fft.ifft(no_res_coeffs, norm='ortho').real) ) # 4. Fixed-phase RFT fixed_coeffs = RFTAblationVariants.rft_fixed_phases(test_signal) variants['Fixed Phase'].append(compute_sparsity_simple(fixed_coeffs)) reconstruction_quality['Fixed Phase'].append( np.linalg.norm(test_signal - np.fft.ifft(fixed_coeffs, norm='ortho').real) ) # 5. Random-weight RFT random_coeffs = RFTAblationVariants.rft_random_weights(test_signal, seed=trial) variants['Random Weights'].append(compute_sparsity_simple(random_coeffs)) reconstruction_quality['Random Weights'].append( np.linalg.norm(test_signal - np.fft.ifft(random_coeffs, norm='ortho').real) )
        if trial < 3:
        print(f" Trial {trial+1}: Full RFT sparsity = {variants['Full RFT'][-1]:.3f}")
        print()
        print(" ABLATION RESULTS:")
        print()

        # Compare sparsity with confidence intervals
        print("1. SPARSITY COMPARISON (with 95% CI):") sparsity_ranking = []
        for variant_name in variants.keys(): data = variants[variant_name] mean_val, ci_lower, ci_upper = bootstrap_confidence_interval(data) sparsity_ranking.append((variant_name, mean_val))
        print(f" • {variant_name:12s}: {mean_val:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")

        # Rank by performance sparsity_ranking.sort(key=lambda x: x[1], reverse=True)
        print(f"||n Ranking: {' > '.join([name for name, _ in sparsity_ranking])}")
        print()

        # Statistical significance tests
        print("2. STATISTICAL SIGNIFICANCE (vs Full RFT):") rft_full_sparsity = variants['Full RFT'] for variant_name, data in variants.items():
        if variant_name != 'Full RFT':
        try: _, p_value = stats.ttest_ind(rft_full_sparsity, data, alternative='greater') significance = "***"
        if p_value < 0.001 else "**"
        if p_value < 0.01 else "*"
        if p_value < 0.05 else "ns"
        print(f" • vs {variant_name:12s}: p = {p_value:.3e} {significance}")
        except:
        print(f" • vs {variant_name:12s}: test failed")
        print()

        # Reconstruction quality
        print("3. RECONSTRUCTION QUALITY:")
        for variant_name in reconstruction_quality.keys(): data = reconstruction_quality[variant_name] mean_error = np.mean(data)
        print(f" • {variant_name:12s}: {mean_error:.2e}")
        print()

        # Parameter sensitivity analysis
        print("4. PARAMETER SENSITIVITY:") resonance_strengths = [0.1, 0.5, 1.0, 2.0] coupling_factors = [0.1, 0.5, 1.0] sensitivity_results = {}
        for res_strength in resonance_strengths:
        for coupling in coupling_factors: key = f"R={res_strength:.1f}, C={coupling:.1f}" sparsity_values = []
        for trial in range(10):

        # Smaller sample for parameter sweep test_sig, _ = generate_parameter_sweep_signals( t, resonance_strength=res_strength, coupling_factor=coupling ) rft_coeffs = forward_true_rft(test_sig) sparsity_values.append(compute_sparsity_simple(rft_coeffs)) sensitivity_results[key] = np.mean(sparsity_values)
        print(" RFT Sparsity vs Parameters:") for params, sparsity in sensitivity_results.items():
        print(f" {params}: {sparsity:.3f}")
        print()

        # Final scientific conclusions
        print("🔬 SCIENTIFIC CONCLUSIONS:")
        print() rft_rank = [i for i, (name, _) in enumerate(sparsity_ranking)
        if name == 'Full RFT'][0]
        if rft_rank == 0:
        print("✅ FULL RFT SHOWS GENUINE ADVANTAGES:")
        print(" • Outperforms all ablated variants")
        print(" • All components contribute to performance")

        # Identify key contributions dft_improvement = variants['Full RFT'][0] / np.mean(variants['DFT Limit'])
        if variants['DFT Limit'] else 1.0
        print(f" • {dft_improvement:.2f}x improvement over pure DFT")
        el
        if rft_rank <= 1:
        print("⚠️ FULL RFT SHOWS MODEST ADVANTAGES:")
        print(f" • Ranks #{rft_rank+1} among variants")
        print(" • Some components may be redundant")
        else:
        print("❌ FULL RFT UNDERPERFORMS SIMPLER VARIANTS:")
        print(f" • Ranks #{rft_rank+1} among variants")
        print(" • Complex structure may be counterproductive")
        print(" • Consider simpler resonance models")

        # Component analysis full_rft_mean = np.mean(variants['Full RFT']) dft_mean = np.mean(variants['DFT Limit']) no_res_mean = np.mean(variants['No Resonance']) resonance_contribution = full_rft_mean - dft_mean
        print(f" • Resonance operators contribute: {resonance_contribution:+.3f}") if 'Fixed Phase' in variants: fixed_mean = np.mean(variants['Fixed Phase']) adaptivity_contribution = full_rft_mean - fixed_mean
        print(f" • Adaptive phases contribute: {adaptivity_contribution:+.3f}")
        print()
        return variants, sensitivity_results

if __name__ == "__main__": variants, sensitivity = run_ablation_study()
print("=" * 80)
print("ABLATION STUDY COMPLETE")
print("This isolates the mathematical contributions of each RFT component")
print("=" * 80)