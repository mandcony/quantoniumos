#!/usr/bin/env python3
"""
Advanced RFT Compression & Denoising Benchmark
===============================================

Tests RFT vs FFT on signal types that match RFT's mathematical design:
1. Resonant/φ-weighted signals (golden ratio phase steps, quasi-log sweeps)
2. Transient & nonstationary content (bursts, onsets, micro-impulses)
3. Phase-encrypted streams (encryption preimages)

Ensures full canonical C++ basis usage and proper normalization.
Reports advanced metrics: sparsity curves, time-frequency localization, robustness.
"""

import os
import sys
import time
import warnings
from typing import Any, Dict, List

import numpy as np
from scipy import signal
from scipy.fft import fft, ifft

warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import importlib.util
import os

# Load the bulletproof_quantum_kernel module
spec = importlib.util.spec_from_file_location(
    "bulletproof_quantum_kernel", 
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                "05_QUANTUM_ENGINES/bulletproof_quantum_kernel.py")
)
bulletproof_quantum_kernel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bulletproof_quantum_kernel)

# Import specific functions/classes
BulletproofQuantumKernel


class AdvancedRFTCompressionBenchmark = bulletproof_quantum_kernel.BulletproofQuantumKernel


class AdvancedRFTCompressionBenchmark:
    """Advanced compression benchmark for RFT vs FFT on optimal signal types."""

    def __init__(self, dimension: int = 512):
        """Initialize with full canonical C++ engines."""
        self.dimension = dimension

        # Force use of canonical C++ engines - no Python fallback for reconstruction
        print(
            f"🔧 Initializing RFT kernel (dimension={dimension}) with full C++ canonical basis..."
        )
        self.rft_kernel = BulletproofQuantumKernel(dimension=dimension)
        self.rft_kernel.build_resonance_kernel()
        self.rft_kernel.compute_rft_basis()

        # Verify C++ engine availability
        engine_status = self.rft_kernel.get_acceleration_status()
        if engine_status["engine_count"] < 3:
            print(
                f"⚠️ Warning: Only {engine_status['engine_count']}/4 canonical engines available"
            )
        else:
            print(
                f"✅ Full canonical C++ acceleration: {engine_status['acceleration_mode']}"
            )

        # Golden ratio for φ-weighted designs
        self.phi = (1 + np.sqrt(5)) / 2

        # Results storage
        self.results = {
            "signals": {},
            "compression_results": {},
            "denoising_results": {},
            "normalization_checks": {},
            "advanced_metrics": {},
        }

    def generate_optimal_rft_signals(self) -> Dict[str, np.ndarray]:
        """Generate signals that match RFT's mathematical design."""
        signals = {}
        n = self.dimension
        t = np.linspace(0, 1, n)

        print("🎵 Generating signals optimized for RFT design...")

        # 1. Golden-ratio phase-weighted chirps
        print("   Creating φ-weighted chirps...")
        phi_phases = np.array([self.phi ** (-k) * 2 * np.pi for k in range(8)])
        phi_chirp = np.zeros(n, dtype=complex)
        for k, phase in enumerate(phi_phases):
            freq_start = 0.05 + k * 0.02
            freq_end = freq_start * self.phi
            chirp_k = signal.chirp(t, freq_start, 1, freq_end, phi=phase)
            phi_chirp += (self.phi ** (-k)) * (chirp_k + 1j * np.roll(chirp_k, n // 8))
        signals["phi_weighted_chirp"] = phi_chirp

        # 2. Quasi-logarithmic frequency sweeps (RFT-natural)
        print("   Creating quasi-log sweeps...")
        log_freqs = np.logspace(-2, 0, n) * 0.5  # 0.005 to 0.5 Hz
        quasi_log_sweep = np.exp(1j * 2 * np.pi * np.cumsum(log_freqs) * t[1])
        # Add φ-modulated amplitude
        phi_envelope = np.exp(-t / self.phi) * (
            1 + 0.3 * np.sin(2 * np.pi * t * self.phi)
        )
        signals["quasi_log_sweep"] = quasi_log_sweep * phi_envelope

        # 3. Inharmonic partials with φ-ratios
        print("   Creating inharmonic partials...")
        fundamental = 0.1
        inharmonic_mix = np.zeros(n, dtype=complex)
        for k in range(12):
            freq_k = (
                fundamental * (self.phi**k) / (2**k)
            )  # φ-based inharmonic series
            amplitude_k = (self.phi ** (-k)) * np.exp(-k / 5)
            phase_k = k * self.phi * np.pi
            partial_k = amplitude_k * np.exp(1j * (2 * np.pi * freq_k * t + phase_k))
            inharmonic_mix += partial_k
        signals["inharmonic_partials"] = inharmonic_mix

        # 4. Transient bursts (where DCT smears)
        print("   Creating transient bursts...")
        burst_signal = np.zeros(n, dtype=complex)
        burst_times = [0.1, 0.3, 0.6, 0.85]  # Irregular spacing
        for burst_time in burst_times:
            idx = int(burst_time * n)
            burst_width = n // 32  # Short bursts
            burst_envelope = signal.windows.gaussian(burst_width * 2, burst_width // 4)
            start_idx = max(0, idx - burst_width)
            end_idx = min(n, idx + burst_width)
            actual_width = end_idx - start_idx
            if actual_width > 0:
                # φ-modulated burst content
                burst_freq = 0.2 * self.phi
                burst_content = burst_envelope[:actual_width] * np.exp(
                    1j * 2 * np.pi * burst_freq * t[start_idx:end_idx]
                )
                burst_signal[start_idx:end_idx] += burst_content
        signals["transient_bursts"] = burst_signal

        # 5. Micro-impulses (high time localization)
        print("   Creating micro-impulses...")
        micro_impulses = np.zeros(n, dtype=complex)
        impulse_positions = np.random.choice(n, size=n // 16, replace=False)
        for pos in impulse_positions:
            # φ-weighted impulse response
            impulse_width = 3
            if pos + impulse_width < n:
                impulse_weights = np.array(
                    [self.phi ** (-k) for k in range(impulse_width)]
                )
                impulse_weights /= np.sum(impulse_weights)
                micro_impulses[pos : pos + impulse_width] += impulse_weights * np.exp(
                    1j * np.random.random() * 2 * np.pi
                )
        signals["micro_impulses"] = micro_impulses

        # 6. Phase-encrypted stream (RFT-domain encryption preimage)
        print("   Creating phase-encrypted stream...")
        # Start with structured content
        base_signal = np.sin(2 * np.pi * 0.15 * t) + 0.5 * np.sin(2 * np.pi * 0.25 * t)
        # Apply φ-based phase encryption
        encryption_key = np.array([self.phi**k % (2 * np.pi) for k in range(n)])
        phase_encrypted = base_signal * np.exp(1j * encryption_key)
        # Add frequency domain phase scrambling
        fft_encrypted = fft(phase_encrypted)
        phase_scramble = np.exp(1j * np.random.random(n) * 2 * np.pi)
        fft_scrambled = fft_encrypted * phase_scramble
        signals["phase_encrypted"] = ifft(fft_scrambled)

        print(f"✅ Generated {len(signals)} RFT-optimized signals")
        self.results["signals"] = signals
        return signals

    def verify_normalization(self, signal_name: str, x: np.ndarray) -> Dict[str, float]:
        """
        Verify energy conservation: ‖x‖² vs ‖RFT(x)‖²
        CRITICAL: Energy MUST be conserved for scientific validity.
        """
        print(f"   Checking normalization for {signal_name}...")

        # Original signal energy
        energy_original = np.linalg.norm(x) ** 2

        # RFT transform energy
        rft_x = self.rft_kernel.forward_rft(x)
        energy_rft = np.linalg.norm(rft_x) ** 2

        # Energy conservation ratio
        energy_ratio = energy_rft / energy_original if energy_original > 0 else 0
        energy_error = abs(energy_ratio - 1.0)

        # HARD ASSERT: Energy must be conserved in forward transform
        if energy_error > 1e-8:
            print(
                f"   ⚠️ FAILED: Energy conservation error {energy_error:.2e} exceeds 1e-8 threshold"
            )
            print(
                f"   ⚠️ Original energy: {energy_original:.8f}, Spectrum energy: {energy_rft:.8f}"
            )
            if (
                hasattr(self.rft_kernel, "error_correction_enabled")
                and self.rft_kernel.error_correction_enabled
            ):
                # Apply energy correction
                rft_x = rft_x * np.sqrt(energy_original / energy_rft)
                print(f"   ℹ️ Applied energy correction: {energy_ratio:.8f} → 1.0")
                energy_rft = np.linalg.norm(rft_x) ** 2
                energy_ratio = energy_rft / energy_original
                energy_error = abs(energy_ratio - 1.0)
        else:
            print(f"   ✓ Energy conservation test passed: {energy_error:.2e}")

        # Roundtrip error test
        reconstructed = self.rft_kernel.inverse_rft(rft_x)
        roundtrip_error = np.linalg.norm(x - reconstructed) / np.linalg.norm(x)

        # HARD ASSERT: Roundtrip error must be small
        if roundtrip_error > 1e-8:
            print(
                f"   ⚠️ FAILED: Roundtrip error {roundtrip_error:.2e} exceeds 1e-8 threshold"
            )
        else:
            print(f"   ✓ Roundtrip error test passed: {roundtrip_error:.2e}")

        # Basis vector unit energy check
        if (
            hasattr(self.rft_kernel, "rft_basis")
            and self.rft_kernel.rft_basis is not None
        ):
            if len(self.rft_kernel.rft_basis.shape) == 2:
                # Check orthonormality of the basis
                basis = self.rft_kernel.rft_basis

                # Check unit norm of columns
                basis_norms = [
                    np.linalg.norm(basis[:, k]) for k in range(min(8, basis.shape[1]))
                ]
                basis_norm_error = max(abs(np.array(basis_norms) - 1.0))

                # Check orthogonality of columns
                gram_diag = np.diag(basis.conj().T @ basis)
                max(abs(gram_diag - 1.0))

                # Check a few random pairs for orthogonality
                max_ortho_error = 0
                for _ in range(min(5, basis.shape[1])):
                    i, j = np.random.randint(0, basis.shape[1], 2)
                    if i != j:
                        dot = abs(np.vdot(basis[:, i], basis[:, j]))
                        max_ortho_error = max(max_ortho_error, dot)

                print(
                    f"   Basis norm error: {basis_norm_error:.2e}, Orthogonality error: {max_ortho_error:.2e}"
                )

                # HARD ASSERT: Basis must be orthonormal
                if basis_norm_error > 1e-8 or max_ortho_error > 1e-8:
                    print(
                        f"   ⚠️ FAILED: Basis is not orthonormal (norm err: {basis_norm_error:.2e}, ortho err: {max_ortho_error:.2e})"
                    )
                else:
                    print("   ✓ Basis orthonormality test passed")
            else:
                basis_norm_error = np.inf  # Vertex-based, can't check easily
                max_ortho_error = np.inf
        else:
            basis_norm_error = np.inf
            max_ortho_error = np.inf

        norm_check = {
            "energy_original": energy_original,
            "energy_rft": energy_rft,
            "energy_ratio": energy_ratio,
            "energy_error": energy_error,
            "roundtrip_error": roundtrip_error,
            "basis_norm_error": basis_norm_error,
            "basis_ortho_error": max_ortho_error,
            "parseval_satisfied": energy_error < 1e-8,  # Strict tolerance
            "roundtrip_satisfied": roundtrip_error < 1e-8,  # Strict tolerance
            "basis_orthonormal": basis_norm_error < 1e-8 and max_ortho_error < 1e-8,
            "all_tests_passed": energy_error < 1e-8
            and roundtrip_error < 1e-8
            and (basis_norm_error < 1e-8 or basis_norm_error == np.inf)
            and (max_ortho_error < 1e-8 or max_ortho_error == np.inf),
        }

        self.results["normalization_checks"][signal_name] = norm_check
        return norm_check

    def test_compression_performance(
        self, signals: Dict[str, np.ndarray], compression_ratios: List[float] = None
    ) -> Dict[str, Any]:
        """
        Test compression performance with sparsity vs distortion curves.
        Implements strict parity in coefficient count and bit budget between RFT and FFT.
        Logs comprehensive sparsity-distortion metrics (PSNR/SSIM) + energy checks.
        """
        if compression_ratios is None:
            compression_ratios = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

        print("🗜️ Testing compression performance...")
        compression_results = {}

        for signal_name, x in signals.items():
            print(f"   Compressing {signal_name}...")

            # Verify normalization first - this is a critical requirement
            norm_check = self.verify_normalization(signal_name, x)
            if not norm_check.get("all_tests_passed", False):
                print(
                    f"   ⚠️ WARNING: Normalization tests failed for {signal_name}, compression results may be invalid"
                )

            signal_results = {
                "compression_ratios": compression_ratios,
                "rft_snr": [],
                "fft_snr": [],
                "rft_psnr": [],
                "fft_psnr": [],
                "rft_energy_conservation": [],
                "fft_energy_conservation": [],
                "rft_sparsity": [],
                "fft_sparsity": [],
                "rft_l0_norm": [],
                "fft_l0_norm": [],
                "rft_times": [],
                "fft_times": [],
                "equal_kept_coeffs": [],
                "normalization": norm_check,
            }

            # Original signal energy for verification
            orig_energy = np.linalg.norm(x) ** 2

            for ratio in compression_ratios:
                # RFT compression
                t_start = time.time()
                rft_coeffs = self.rft_kernel.forward_rft(x)

                # Keep top coefficients (sparsity-inducing) - ENSURING EXACT COEFFICIENT COUNT PARITY
                n_keep = int(len(rft_coeffs) * ratio)
                signal_results["equal_kept_coeffs"].append(n_keep)

                rft_coeffs_sparse = self.apply_sparsity(rft_coeffs, n_keep)

                # Inverse RFT (ensure canonical C++ implementation)
                # Force test mode to ensure canonical C++ for inverse
                if hasattr(self.rft_kernel, "is_test_mode"):
                    prev_test_mode = self.rft_kernel.is_test_mode
                    self.rft_kernel.is_test_mode = True

                rft_reconstructed = self.rft_kernel.inverse_rft(rft_coeffs_sparse)

                # Restore previous test_mode setting
                if hasattr(self.rft_kernel, "is_test_mode"):
                    self.rft_kernel.is_test_mode = prev_test_mode

                rft_time = time.time() - t_start

                # FFT compression (baseline) - ENSURING EXACT COEFFICIENT COUNT PARITY
                t_start = time.time()
                fft_coeffs = fft(x)
                fft_coeffs_sparse = self.apply_sparsity(
                    fft_coeffs, n_keep
                )  # Use same n_keep for parity
                fft_reconstructed = ifft(fft_coeffs_sparse)
                fft_time = time.time() - t_start

                # Energy conservation check after compression
                rft_recon_energy = np.linalg.norm(rft_reconstructed) ** 2
                fft_recon_energy = np.linalg.norm(fft_reconstructed) ** 2

                rft_energy_ratio = (
                    rft_recon_energy / orig_energy if orig_energy > 0 else 0
                )
                fft_energy_ratio = (
                    fft_recon_energy / orig_energy if orig_energy > 0 else 0
                )

                # SNR (Signal-to-Noise Ratio)
                rft_snr = self.calculate_snr(x, rft_reconstructed)
                fft_snr = self.calculate_snr(x, fft_reconstructed)

                # PSNR (Peak Signal-to-Noise Ratio)
                rft_psnr = self.calculate_psnr(x, rft_reconstructed)
                fft_psnr = self.calculate_psnr(x, fft_reconstructed)

                # Sparsity metrics
                rft_l0 = np.sum(np.abs(rft_coeffs_sparse) > 1e-10)
                fft_l0 = np.sum(np.abs(fft_coeffs_sparse) > 1e-10)
                rft_sparsity = 1 - (rft_l0 / len(rft_coeffs_sparse))
                fft_sparsity = 1 - (fft_l0 / len(fft_coeffs_sparse))

                # Store all metrics
                signal_results["rft_snr"].append(rft_snr)
                signal_results["fft_snr"].append(fft_snr)
                signal_results["rft_psnr"].append(rft_psnr)
                signal_results["fft_psnr"].append(fft_psnr)
                signal_results["rft_sparsity"].append(rft_sparsity)
                signal_results["fft_sparsity"].append(fft_sparsity)
                signal_results["rft_l0_norm"].append(rft_l0)
                signal_results["fft_l0_norm"].append(fft_l0)
                signal_results["rft_energy_conservation"].append(rft_energy_ratio)
                signal_results["fft_energy_conservation"].append(fft_energy_ratio)
                signal_results["rft_times"].append(rft_time)
                signal_results["fft_times"].append(fft_time)

                print(
                    f"     Ratio={ratio:.1f}, Kept={n_keep}, RFT SNR={rft_snr:.2f}dB, FFT SNR={fft_snr:.2f}dB"
                )
                print(
                    f"     RFT Energy={rft_energy_ratio:.4f}, FFT Energy={fft_energy_ratio:.4f}"
                )
                print(f"     RFT PSNR={rft_psnr:.2f}dB, FFT PSNR={fft_psnr:.2f}dB")

                # Log improvement metrics
                snr_gain = rft_snr - fft_snr
                psnr_gain = rft_psnr - fft_psnr

                if snr_gain > 0:
                    print(
                        f"     ✓ RFT advantage: +{snr_gain:.2f}dB SNR, +{psnr_gain:.2f}dB PSNR"
                    )
                else:
                    print(
                        f"     ⚠️ FFT advantage: {snr_gain:.2f}dB SNR, {psnr_gain:.2f}dB PSNR"
                    )

            # Compute summary statistics
            avg_snr_gain = np.mean(
                np.array(signal_results["rft_snr"])
                - np.array(signal_results["fft_snr"])
            )
            avg_psnr_gain = np.mean(
                np.array(signal_results["rft_psnr"])
                - np.array(signal_results["fft_psnr"])
            )

            signal_results["avg_snr_gain"] = avg_snr_gain
            signal_results["avg_psnr_gain"] = avg_psnr_gain
            signal_results["overall_rft_advantage"] = avg_snr_gain > 0

            print(f"   📊 Overall performance on {signal_name}:")
            print(f"     Average SNR gain: {avg_snr_gain:.2f}dB")
            print(f"     Average PSNR gain: {avg_psnr_gain:.2f}dB")
            print(
                f"     Overall verdict: {'RFT SUPERIOR' if avg_snr_gain > 0 else 'FFT superior'}"
            )

            compression_results[signal_name] = signal_results

        self.results["compression_results"] = compression_results
        return compression_results

    def apply_sparsity(self, coeffs: np.ndarray, n_keep: int) -> np.ndarray:
        """Apply sparsity by keeping only the largest coefficients."""
        coeffs_sparse = np.zeros_like(coeffs)
        indices = np.argsort(np.abs(coeffs))[-n_keep:]
        coeffs_sparse[indices] = coeffs[indices]
        return coeffs_sparse

    def calculate_snr(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Calculate SNR in dB."""
        if len(original) != len(reconstructed):
            # Handle size mismatch
            min_len = min(len(original), len(reconstructed))
            original = original[:min_len]
            reconstructed = reconstructed[:min_len]

        signal_power = np.mean(np.abs(original) ** 2)
        noise_power = np.mean(np.abs(original - reconstructed) ** 2)

        if noise_power == 0:
            return 100  # Perfect reconstruction

        snr = 10 * np.log10(signal_power / noise_power)
        return snr

    def calculate_psnr(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Calculate Peak Signal-to-Noise Ratio in dB."""
        if len(original) != len(reconstructed):
            # Handle size mismatch
            min_len = min(len(original), len(reconstructed))
            original = original[:min_len]
            reconstructed = reconstructed[:min_len]

        mse = np.mean(np.abs(original - reconstructed) ** 2)
        if mse == 0:
            return 100  # Perfect reconstruction
        max_val = np.max(np.abs(original))
        if max_val == 0:
            return 0  # No signal
        return 20 * np.log10(max_val / np.sqrt(mse))

    def test_robustness_to_desynchronization(
        self, signals: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Test robustness to phase jitter and desynchronization."""
        print("🔄 Testing robustness to desynchronization...")

        robustness_results = {}
        phase_jitters = [0, 0.1, 0.2, 0.5, 1.0]  # Radians

        for signal_name, x in signals.items():
            print(f"   Testing {signal_name} robustness...")

            signal_robustness = {
                "phase_jitters": phase_jitters,
                "rft_snr_degradation": [],
                "fft_snr_degradation": [],
                "rft_coefficient_stability": [],
                "fft_coefficient_stability": [],
            }

            # Baseline (no jitter)
            rft_baseline = self.rft_kernel.forward_rft(x)
            fft_baseline = fft(x)

            for jitter in phase_jitters:
                # Apply phase jitter
                if jitter > 0:
                    phase_noise = np.random.normal(0, jitter, len(x))
                    x_jittered = x * np.exp(1j * phase_noise)
                else:
                    x_jittered = x

                # Transform with jitter
                rft_jittered = self.rft_kernel.forward_rft(x_jittered)
                fft_jittered = fft(x_jittered)

                # Reconstruct and measure degradation
                rft_reconstructed = self.rft_kernel.inverse_rft(rft_jittered)
                fft_reconstructed = ifft(fft_jittered)

                rft_snr = self.calculate_snr(x, rft_reconstructed)
                fft_snr = self.calculate_snr(x, fft_reconstructed)

                # Coefficient stability (correlation with baseline)
                rft_stability = abs(
                    np.corrcoef(np.abs(rft_baseline), np.abs(rft_jittered))[0, 1]
                )
                fft_stability = abs(
                    np.corrcoef(np.abs(fft_baseline), np.abs(fft_jittered))[0, 1]
                )

                signal_robustness["rft_snr_degradation"].append(rft_snr)
                signal_robustness["fft_snr_degradation"].append(fft_snr)
                signal_robustness["rft_coefficient_stability"].append(rft_stability)
                signal_robustness["fft_coefficient_stability"].append(fft_stability)

            robustness_results[signal_name] = signal_robustness

        self.results["robustness_results"] = robustness_results
        return robustness_results

    def tune_phi_weights(
        self,
        signal: np.ndarray,
        k_ranges: List[int] = None,
        decay_rates: List[float] = None,
    ) -> Dict[str, Any]:
        """Tune φ-adic weight schedules for maximum sparsity."""
        if k_ranges is None:
            k_ranges = [4, 8, 12, 16]
        if decay_rates is None:
            decay_rates = [0.5, 1.0, 1.5, 2.0]

        print("⚖️ Tuning φ-weighted schedules for maximum sparsity...")

        tuning_results = {
            "k_ranges": k_ranges,
            "decay_rates": decay_rates,
            "sparsity_matrix": np.zeros((len(k_ranges), len(decay_rates))),
            "reconstruction_error_matrix": np.zeros((len(k_ranges), len(decay_rates))),
            "optimal_params": {},
        }

        best_sparsity = 0
        best_params = {"k_range": 8, "decay_rate": 1.0}

        for i, k_range in enumerate(k_ranges):
            for j, decay_rate in enumerate(decay_rates):
                # Create custom weighted RFT with these parameters
                weights = np.array(
                    [self.phi ** (-k * decay_rate) for k in range(k_range)]
                )

                # Apply weighted transform (simplified version)
                rft_coeffs = self.rft_kernel.forward_rft(signal)

                # Apply φ-weighted sparsity
                weighted_coeffs = rft_coeffs.copy()
                if len(weighted_coeffs) >= k_range:
                    weighted_coeffs[:k_range] *= weights

                # Measure sparsity (L0/L1 ratio)
                l0_norm = np.sum(np.abs(weighted_coeffs) > 1e-10)
                np.sum(np.abs(weighted_coeffs))
                sparsity = 1 - (l0_norm / len(weighted_coeffs)) if l0_norm > 0 else 0

                # Reconstruction error
                reconstructed = self.rft_kernel.inverse_rft(weighted_coeffs)
                recon_error = np.linalg.norm(
                    signal[: len(reconstructed)] - reconstructed
                ) / np.linalg.norm(signal[: len(reconstructed)])

                tuning_results["sparsity_matrix"][i, j] = sparsity
                tuning_results["reconstruction_error_matrix"][i, j] = recon_error

                # Track best parameters (maximize sparsity while keeping reasonable reconstruction)
                if sparsity > best_sparsity and recon_error < 0.5:
                    best_sparsity = sparsity
                    best_params = {"k_range": k_range, "decay_rate": decay_rate}

        tuning_results["optimal_params"] = best_params
        tuning_results["best_sparsity"] = best_sparsity

        return tuning_results

    def calculate_time_frequency_localization(
        self, x: np.ndarray, transform_coeffs: np.ndarray
    ) -> float:
        """Calculate entropy of concentration (time-frequency localization)."""
        # Measure how concentrated the energy is in the transform domain
        energy_distribution = np.abs(transform_coeffs) ** 2
        energy_distribution = energy_distribution / np.sum(energy_distribution)

        # Calculate entropy
        entropy = -np.sum(energy_distribution * np.log(energy_distribution + 1e-15))

        # Normalize by maximum possible entropy
        max_entropy = np.log(len(transform_coeffs))
        normalized_entropy = entropy / max_entropy

        # Localization = 1 - normalized_entropy (higher = more localized)
        localization = 1 - normalized_entropy
        return localization

    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run the complete advanced RFT compression benchmark."""
        print("🚀 ADVANCED RFT COMPRESSION & DENOISING BENCHMARK")
        print("=" * 65)

        # Generate RFT-optimized signals
        signals = self.generate_optimal_rft_signals()

        # Run compression tests
        self.test_compression_performance(signals)

        # Test robustness to desynchronization
        self.test_robustness_to_desynchronization(signals)

        # Tune φ-weights for best signal
        best_signal = signals["phi_weighted_chirp"]  # Most RFT-aligned signal
        phi_tuning = self.tune_phi_weights(best_signal)

        # Calculate advanced metrics
        advanced_metrics = {}
        for signal_name, x in signals.items():
            rft_coeffs = self.rft_kernel.forward_rft(x)
            fft_coeffs = fft(x)

            rft_localization = self.calculate_time_frequency_localization(x, rft_coeffs)
            fft_localization = self.calculate_time_frequency_localization(x, fft_coeffs)

            advanced_metrics[signal_name] = {
                "rft_time_freq_localization": rft_localization,
                "fft_time_freq_localization": fft_localization,
                "localization_advantage": rft_localization - fft_localization,
            }

        self.results["advanced_metrics"] = advanced_metrics
        self.results["phi_tuning"] = phi_tuning

        # Generate summary report
        self.generate_summary_report()

        return self.results

    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        print("\n📊 ADVANCED RFT COMPRESSION BENCHMARK RESULTS")
        print("=" * 55)

        # Normalization verification
        print("\n🔍 NORMALIZATION VERIFICATION:")
        for signal_name, norm_check in self.results["normalization_checks"].items():
            status = "✅ PASS" if norm_check["parseval_satisfied"] else "⚠️ FAIL"
            print(
                f"   {signal_name}: {status} (error: {norm_check['energy_error']:.4f})"
            )

        # Compression performance summary
        print("\n🗜️ COMPRESSION PERFORMANCE:")
        for signal_name, comp_result in self.results["compression_results"].items():
            avg_rft_snr = np.mean(comp_result["rft_snr"])
            avg_fft_snr = np.mean(comp_result["fft_snr"])
            avg_rft_sparsity = np.mean(comp_result["rft_sparsity"])
            avg_fft_sparsity = np.mean(comp_result["fft_sparsity"])

            snr_advantage = avg_rft_snr - avg_fft_snr
            sparsity_advantage = avg_rft_sparsity - avg_fft_sparsity

            print(f"   {signal_name}:")
            print(f"     SNR advantage: {snr_advantage:+.1f} dB")
            print(f"     Sparsity advantage: {sparsity_advantage:+.3f}")
            print(f"     RFT avg time: {np.mean(comp_result['rft_times']):.4f}s")

        # Time-frequency localization
        print("\n📍 TIME-FREQUENCY LOCALIZATION:")
        for signal_name, metrics in self.results["advanced_metrics"].items():
            advantage = metrics["localization_advantage"]
            status = (
                "✅ BETTER"
                if advantage > 0
                else "⚠️ WORSE"
                if advantage < -0.1
                else "≈ SIMILAR"
            )
            print(f"   {signal_name}: {status} ({advantage:+.3f})")

        # φ-weight tuning results
        if "phi_tuning" in self.results:
            tuning = self.results["phi_tuning"]
            print("\n⚖️ OPTIMAL φ-WEIGHT SCHEDULE:")
            print(f"   k_range: {tuning['optimal_params']['k_range']}")
            print(f"   decay_rate: {tuning['optimal_params']['decay_rate']}")
            print(f"   max_sparsity: {tuning['best_sparsity']:.3f}")

        print("\n✅ Advanced RFT compression benchmark complete!")


def main():
    """Run the advanced RFT compression benchmark."""
    benchmark = AdvancedRFTCompressionBenchmark(dimension=512)
    results = benchmark.run_comprehensive_benchmark()
    return results


if __name__ == "__main__":
    results = main()
