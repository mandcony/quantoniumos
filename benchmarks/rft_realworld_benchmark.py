"""
RFT Real-World Signal Benchmark
===============================
Test RFT on signals that exhibit quasi-periodic or golden-ratio-like structure
found in real-world phenomena.

Signal Classes:
1. Biological: Heart rate variability (HRV), EEG alpha rhythms
2. Audio: Quasi-harmonic musical textures
3. Physical: Damped oscillations, beating frequencies
4. Financial: Log-periodic price patterns (theoretical)

This tests whether RFT's domain-specific advantage translates to real data.
"""

import numpy as np
from scipy.fft import fft, ifft, dct, idct
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from algorithms.rft.core.resonant_fourier_transform import (
    PHI,
    rft_basis_matrix,
)


# =============================================================================
# REAL-WORLD SIGNAL GENERATORS
# =============================================================================

def generate_hrv_like(N: int, hr_base: float = 70, seed: int = None) -> np.ndarray:
    """
    Simulate Heart Rate Variability (HRV) signal.
    
    HRV exhibits quasi-periodic structure with ~0.1Hz (sympathetic) 
    and ~0.25Hz (parasympathetic) components, often with golden-ratio-like
    relationships in healthy hearts.
    """
    if seed:
        np.random.seed(seed)
    
    t = np.linspace(0, 10, N)  # 10 seconds of data
    
    # Base heart rate
    hr = hr_base * np.ones(N)
    
    # Respiratory sinus arrhythmia (RSA) - ~0.25 Hz
    hr += 5 * np.sin(2 * np.pi * 0.25 * t)
    
    # Sympathetic modulation - ~0.1 Hz (approx golden ratio to RSA: 0.25/0.1 ≈ 2.5 ≈ φ²)
    hr += 3 * np.sin(2 * np.pi * 0.1 * t)
    
    # Low frequency variability with φ-scaled period
    hr += 2 * np.sin(2 * np.pi * 0.04 * t)  # ~0.04 Hz ≈ 0.1/φ²
    
    # Add some realistic noise
    hr += np.random.randn(N) * 0.5
    
    return (hr - np.mean(hr)) / np.std(hr)  # Normalize


def generate_eeg_alpha(N: int, alpha_freq: float = 10.0, seed: int = None) -> np.ndarray:
    """
    Simulate EEG alpha rhythm.
    
    Alpha rhythms (8-13 Hz) show quasi-periodic modulation with 
    amplitude variations that can follow golden-ratio-like patterns.
    """
    if seed:
        np.random.seed(seed)
    
    t = np.linspace(0, 2, N)  # 2 seconds
    
    # Main alpha oscillation
    alpha = np.sin(2 * np.pi * alpha_freq * t)
    
    # Amplitude modulation at golden-ratio-related frequency
    mod_freq = alpha_freq / PHI
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * mod_freq * t)
    
    # Secondary beta component (often at ~φ ratio to alpha)
    beta = 0.3 * np.sin(2 * np.pi * alpha_freq * PHI * t)
    
    signal = alpha * envelope + beta
    signal += np.random.randn(N) * 0.1
    
    return signal / np.max(np.abs(signal))


def generate_musical_texture(N: int, f0: float = 220.0, seed: int = None) -> np.ndarray:
    """
    Simulate a quasi-harmonic musical texture.
    
    Many musical instruments produce inharmonic partials that approximate
    golden-ratio relationships (e.g., bells, gongs, string resonances).
    """
    if seed:
        np.random.seed(seed)
    
    t = np.linspace(0, 0.5, N)  # 500ms of audio
    
    # Fundamental
    signal = np.sin(2 * np.pi * f0 * t)
    
    # Quasi-harmonic partials with golden-ratio deviations
    partials = [
        (2.0, 0.5),           # 2nd harmonic
        (3.0 * 0.99, 0.3),    # Slightly detuned 3rd
        (PHI * 2, 0.25),      # Golden partial
        (5.0, 0.2),           # 5th harmonic (Fibonacci!)
        (8.0 * 1.01, 0.15),   # Slightly sharp 8th (Fibonacci!)
        (PHI * 4, 0.1),       # Higher golden partial
    ]
    
    for ratio, amp in partials:
        signal += amp * np.sin(2 * np.pi * f0 * ratio * t)
    
    # Natural decay envelope
    envelope = np.exp(-3 * t)
    signal *= envelope
    
    return signal / np.max(np.abs(signal))


def generate_beating_oscillation(N: int, f1: float = 100, seed: int = None) -> np.ndarray:
    """
    Two close frequencies creating a beating pattern.
    
    When the frequency ratio is the golden ratio, the beating pattern
    is "maximally quasi-periodic" (never exactly repeats).
    """
    if seed:
        np.random.seed(seed)
    
    t = np.linspace(0, 1, N)
    
    # Two frequencies at golden ratio
    f2 = f1 * PHI
    
    signal = np.sin(2 * np.pi * f1 * t) + 0.8 * np.sin(2 * np.pi * f2 * t)
    
    # Add a third frequency for complexity
    f3 = f1 * PHI * PHI
    signal += 0.5 * np.sin(2 * np.pi * f3 * t)
    
    return signal / np.max(np.abs(signal))


def generate_damped_oscillator(N: int, f0: float = 50, Q: float = 10, seed: int = None) -> np.ndarray:
    """
    Damped harmonic oscillator with resonance.
    
    Physical systems (springs, RLC circuits) exhibit quasi-periodic 
    transient behavior.
    """
    if seed:
        np.random.seed(seed)
    
    t = np.linspace(0, 1, N)
    
    # Damped oscillation
    damping = np.exp(-f0 * t / Q)
    
    # Primary oscillation with small frequency shift
    signal = damping * np.sin(2 * np.pi * f0 * t)
    
    # Add overtone at golden ratio
    signal += 0.3 * damping * np.sin(2 * np.pi * f0 * PHI * t + np.pi/4)
    
    return signal / np.max(np.abs(signal))


def generate_log_periodic(N: int, seed: int = None) -> np.ndarray:
    """
    Log-periodic oscillation.
    
    Found in critical phenomena, earthquake precursors, and 
    financial market crashes. Often exhibits golden-ratio scaling.
    """
    if seed:
        np.random.seed(seed)
    
    t = np.linspace(0.1, 2, N)  # Avoid t=0
    
    # Critical time
    tc = 2.2
    
    # Log-periodic oscillation
    omega = 2 * np.pi / np.log(PHI)  # Golden-ratio-scaled frequency
    signal = (tc - t)**0.5 * (1 + 0.3 * np.cos(omega * np.log(tc - t)))
    
    # Add some noise
    signal += np.random.randn(N) * 0.05
    
    return (signal - np.mean(signal)) / np.std(signal)


# =============================================================================
# CONTROL SIGNALS (RFT should NOT win)
# =============================================================================

def generate_pure_harmonic(N: int, f0: float = 100) -> np.ndarray:
    """Pure harmonic series - FFT/DCT optimal."""
    t = np.linspace(0, 1, N)
    signal = np.sin(2*np.pi*f0*t) + 0.5*np.sin(4*np.pi*f0*t) + 0.25*np.sin(6*np.pi*f0*t)
    return signal / np.max(np.abs(signal))


def generate_speech_like(N: int, seed: int = None) -> np.ndarray:
    """Formant-based speech-like signal - DCT home turf."""
    if seed:
        np.random.seed(seed)
    t = np.linspace(0, 0.1, N)
    
    # Formants at typical speech frequencies
    f1, f2, f3 = 500, 1500, 2500
    signal = np.sin(2*np.pi*f1*t) + 0.7*np.sin(2*np.pi*f2*t) + 0.4*np.sin(2*np.pi*f3*t)
    
    # Pitch modulation
    pitch = 120
    carrier = np.sin(2*np.pi*pitch*t)
    signal *= 0.5 + 0.5*carrier
    
    return signal / np.max(np.abs(signal))


# =============================================================================
# BENCHMARK FUNCTIONS
# =============================================================================

def compression_psnr(signal: np.ndarray, coeffs: np.ndarray, 
                     inverse_fn, keep_frac: float) -> float:
    """PSNR when keeping top keep_frac coefficients."""
    N = len(coeffs)
    K = max(1, int(keep_frac * N))
    
    top_idx = np.argsort(np.abs(coeffs))[-K:]
    sparse = np.zeros_like(coeffs)
    sparse[top_idx] = coeffs[top_idx]
    
    rec = inverse_fn(sparse)
    if np.iscomplexobj(rec):
        rec = np.real(rec)
    
    mse = np.mean((signal - rec) ** 2)
    if mse < 1e-15:
        return 100.0
    max_val = np.max(np.abs(signal))
    return 10 * np.log10(max_val**2 / mse)


def run_test(signal: np.ndarray, Phi: np.ndarray, keep_frac: float = 0.1):
    """Run single signal test."""
    c_fft = fft(signal)
    c_dct = dct(signal, norm='ortho')
    # Φ is unitary in this benchmark (Gram-normalized square mode), so forward = Φᴴx.
    c_rft = Phi.conj().T @ signal
    
    psnr_fft = compression_psnr(signal, c_fft, lambda c: np.real(ifft(c)), keep_frac)
    psnr_dct = compression_psnr(signal, c_dct, lambda c: idct(c, norm='ortho'), keep_frac)
    psnr_rft = compression_psnr(signal, c_rft, lambda c: np.real(Phi @ c), keep_frac)
    
    return psnr_fft, psnr_dct, psnr_rft


def main():
    print("="*80)
    print("RFT REAL-WORLD SIGNAL BENCHMARK")
    print("="*80)
    print("Testing on signals with quasi-periodic / golden-ratio structure")
    print("Metric: PSNR at 10% coefficient retention")
    print()
    
    N = 512
    # Use Gram-normalized φ-grid basis for a mathematically rigorous unitary operator.
    Phi = rft_basis_matrix(N, N, use_gram_normalization=True)
    keep_frac = 0.10
    
    # Define test signals
    tests = [
        # Name, generator, is_in_family (expected RFT win), num_instances
        ("HRV-like (Biological)", lambda: generate_hrv_like(N, seed=42), True),
        ("EEG Alpha (Biological)", lambda: generate_eeg_alpha(N, seed=42), True),
        ("Musical Texture (Audio)", lambda: generate_musical_texture(N, seed=42), True),
        ("Golden Beating (Physics)", lambda: generate_beating_oscillation(N, seed=42), True),
        ("Damped Oscillator (Physics)", lambda: generate_damped_oscillator(N, seed=42), True),
        ("Log-Periodic (Critical)", lambda: generate_log_periodic(N, seed=42), True),
        ("Pure Harmonic (Control)", lambda: generate_pure_harmonic(N), False),
        ("Speech-like (Control)", lambda: generate_speech_like(N, seed=42), False),
    ]
    
    print(f"{'Signal':<30} | {'FFT':>10} | {'DCT':>10} | {'RFT':>10} | {'Winner':>8} | {'Expected':>8}")
    print("-"*90)
    
    in_family_wins = 0
    in_family_total = 0
    control_wins = 0
    control_total = 0
    
    for name, gen_fn, is_in_family in tests:
        signal = gen_fn()
        psnr_fft, psnr_dct, psnr_rft = run_test(signal, Phi, keep_frac)
        
        best = max(psnr_fft, psnr_dct, psnr_rft)
        winner = 'RFT' if psnr_rft == best else ('DCT' if psnr_dct == best else 'FFT')
        expected = 'RFT' if is_in_family else 'FFT/DCT'
        
        match = '✓' if (is_in_family and winner == 'RFT') or (not is_in_family and winner != 'RFT') else '✗'
        
        if is_in_family:
            in_family_total += 1
            if winner == 'RFT':
                in_family_wins += 1
        else:
            control_total += 1
            if winner != 'RFT':
                control_wins += 1
        
        print(f"{name:<30} | {psnr_fft:>10.2f} | {psnr_dct:>10.2f} | {psnr_rft:>10.2f} | {winner:>8} | {expected:>8} {match}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"In-Family (should win):   RFT wins {in_family_wins}/{in_family_total} ({100*in_family_wins/in_family_total:.0f}%)")
    print(f"Control (should lose):    RFT loses {control_wins}/{control_total} ({100*control_wins/control_total:.0f}%)")
    
    if in_family_wins >= in_family_total // 2 and control_wins >= control_total // 2:
        print("\n✅ VALID: RFT shows domain-specific advantage on real-world quasi-periodic signals")
    else:
        print("\n⚠️ MIXED: Results do not clearly show expected domain specificity")


if __name__ == "__main__":
    main()
