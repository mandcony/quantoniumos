"""
Signal Classifier for Adaptive Transform Routing
=================================================

This module provides a HEURISTIC classifier to approximate the best
transform choice per signal segment using cheap features.

CALIBRATION STATUS (December 2025):
- Thresholds tuned via exp_classifier_calibration.py
- Accuracy: ~64% (improved from 27% uncalibrated baseline)
- Optimal thresholds found via grid search

USAGE:
- For research experiments comparing router vs oracle
- Use benchmark_classifier() to measure actual accuracy

NEXT STEPS:
1. Add new patent variants as routing targets
2. Re-calibrate with expanded variant set
3. Add tests that pin expected behavior for key signal types

December 2025: Part of the "Next Level RFT" initiative.
"""

import numpy as np
from scipy.fft import fft, dct
from scipy.signal import correlate
from typing import Tuple, Dict, List
from dataclasses import dataclass
from enum import Enum


# Golden ratio
PHI = (1 + np.sqrt(5)) / 2


class TransformType(Enum):
    """Available transforms."""
    FFT = "fft"
    DCT = "dct"
    RFT_GOLDEN = "rft_golden"
    RFT_FIBONACCI = "rft_fibonacci"
    RFT_HARMONIC = "rft_harmonic"
    RFT_GEOMETRIC = "rft_geometric"
    RFT_BEATING = "rft_beating"
    RFT_PHYLLOTAXIS = "rft_phyllotaxis"
    ARFT = "arft"  # Adaptive (signal-specific)
    # Patent variants (USPTO 19/169,399 - Top Performers)
    RFT_MANIFOLD_PROJECTION = "rft_manifold_projection"  # Best for torus/spiral
    RFT_EULER_SPHERE = "rft_euler_sphere"                # Best for phyllotaxis
    RFT_PHASE_COHERENT = "rft_phase_coherent"            # Best for chirp
    RFT_ENTROPY_MODULATED = "rft_entropy_modulated"      # Best for noise
    RFT_LOXODROME = "rft_loxodrome"                      # Good for pure tones


@dataclass
class SignalFeatures:
    """Extracted features for classification."""
    spectral_centroid: float
    spectral_spread: float
    zero_crossing_rate: float
    autocorr_peak_ratio: float
    golden_ratio_score: float
    fibonacci_score: float
    harmonic_score: float
    noise_score: float
    periodicity_score: float


def extract_features(x: np.ndarray) -> SignalFeatures:
    """
    Extract classification features from signal.
    
    Features are designed to detect:
    1. Golden-ratio quasi-periodicity
    2. Fibonacci-modulated structure
    3. Natural harmonic content
    4. Noise vs. structured signal
    5. Periodicity type (integer vs. irrational)
    """
    n = len(x)
    x_norm = x / (np.std(x) + 1e-10)  # Normalize
    
    # =========================================
    # 1. Spectral Features
    # =========================================
    X = np.abs(fft(x_norm))[:n//2]
    freqs = np.arange(n//2)
    
    total_power = np.sum(X**2) + 1e-10
    
    # Spectral centroid: center of mass of spectrum
    spectral_centroid = np.sum(freqs * X**2) / total_power
    
    # Spectral spread: standard deviation around centroid
    spectral_spread = np.sqrt(np.sum((freqs - spectral_centroid)**2 * X**2) / total_power)
    
    # =========================================
    # 2. Zero-Crossing Rate
    # =========================================
    zero_crossings = np.sum(np.abs(np.diff(np.sign(x_norm))) > 0)
    zero_crossing_rate = zero_crossings / n
    
    # =========================================
    # 3. Autocorrelation Features
    # =========================================
    autocorr = correlate(x_norm, x_norm, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / (autocorr[0] + 1e-10)  # Normalize
    
    # Peak ratio: second peak / first peak
    # High ratio = periodic, low ratio = noise-like
    peaks = []
    for i in range(1, len(autocorr)-1):
        if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
            if autocorr[i] > 0.1:  # Threshold
                peaks.append((i, autocorr[i]))
    
    if len(peaks) >= 2:
        autocorr_peak_ratio = peaks[1][1] / peaks[0][1]
    else:
        autocorr_peak_ratio = 0.0
    
    # =========================================
    # 4. Golden Ratio Detection
    # =========================================
    # Check for peaks at golden-ratio-related lags
    golden_lags = [
        int(n / PHI),
        int(n / PHI**2),
        int(n / PHI**3),
        int(n * (PHI - 1)),  # ~0.618 * n
    ]
    
    golden_score = 0.0
    for lag in golden_lags:
        if 0 < lag < len(autocorr):
            golden_score += abs(autocorr[lag])
    golden_score /= len(golden_lags)
    
    # =========================================
    # 5. Fibonacci Pattern Detection
    # =========================================
    fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
    fib_lags = [f for f in fib if f < len(autocorr)]
    
    fibonacci_score = 0.0
    if fib_lags:
        for lag in fib_lags:
            fibonacci_score += abs(autocorr[lag])
        fibonacci_score /= len(fib_lags)
    
    # =========================================
    # 6. Harmonic Series Detection
    # =========================================
    # Check for integer harmonic relationships in spectrum
    if len(peaks) >= 3:
        peak_freqs = [p[0] for p in peaks[:5]]
        
        # Check if peak frequencies are integer multiples
        f0 = peak_freqs[0] if peak_freqs[0] > 0 else 1
        ratios = [f / f0 for f in peak_freqs]
        
        # How close to integers?
        integer_deviation = sum(abs(r - round(r)) for r in ratios) / len(ratios)
        harmonic_score = 1.0 - min(integer_deviation * 5, 1.0)
    else:
        harmonic_score = 0.0
    
    # =========================================
    # 7. Noise Score
    # =========================================
    # Flat spectrum = noise, peaked spectrum = signal
    spectrum_flatness = np.exp(np.mean(np.log(X + 1e-10))) / (np.mean(X) + 1e-10)
    noise_score = spectrum_flatness
    
    # =========================================
    # 8. Periodicity Score
    # =========================================
    # High autocorrelation beyond lag 0 = periodic
    periodicity_score = np.mean(np.abs(autocorr[1:n//4]))
    
    return SignalFeatures(
        spectral_centroid=spectral_centroid,
        spectral_spread=spectral_spread,
        zero_crossing_rate=zero_crossing_rate,
        autocorr_peak_ratio=autocorr_peak_ratio,
        golden_ratio_score=golden_score,
        fibonacci_score=fibonacci_score,
        harmonic_score=harmonic_score,
        noise_score=noise_score,
        periodicity_score=periodicity_score,
    )


def classify_signal(x: np.ndarray) -> Tuple[TransformType, float]:
    """
    Classify signal to select transform.
    
    WARNING: This is an UNCALIBRATED heuristic.
    - Current accuracy: ~40% on test signals
    - Thresholds are initial guesses, not optimized
    - First-match rule ordering can cause errors
    
    Use exp_router_vs_oracle.py to measure actual vs oracle performance.
    
    Returns:
        (predicted_transform, confidence)
        
    Note: 'confidence' is also uncalibrated - treat as relative, not absolute.
    """
    features = extract_features(x)
    
    # =========================================================================
    # CALIBRATED THRESHOLDS (December 2025)
    # Found via exp_classifier_calibration.py grid search
    # Accuracy: 64% (up from 27% with initial guesses)
    # =========================================================================
    PERIODICITY_THRESHOLD = 0.250   # Was 0.1
    GOLDEN_THRESHOLD = 0.080        # Was 0.3
    FIBONACCI_THRESHOLD = 0.280     # Was 0.3 (using sparsity threshold)
    
    # =========================================
    # Rule 1: Noise → DCT (default for unstructured)
    # =========================================
    if features.noise_score > 0.8:
        return TransformType.DCT, 0.6
    
    # =========================================
    # Rule 2: Low periodicity → DCT
    # =========================================
    if features.periodicity_score < PERIODICITY_THRESHOLD:
        return TransformType.DCT, 0.7
    
    # =========================================
    # Rule 3: Strong golden-ratio structure → RFT-Golden
    # =========================================
    if features.golden_ratio_score > GOLDEN_THRESHOLD:
        return TransformType.RFT_GOLDEN, 0.8 + 0.2 * min(features.golden_ratio_score, 1.0)
    
    # =========================================
    # Rule 4: Strong Fibonacci structure → RFT-Fibonacci
    # =========================================
    if features.fibonacci_score > FIBONACCI_THRESHOLD:
        return TransformType.RFT_FIBONACCI, 0.75
    
    # =========================================
    # Rule 5: Strong harmonic structure → RFT-Harmonic or FFT
    # =========================================
    if features.harmonic_score > 0.7:
        # Pure integer harmonics → FFT wins
        if features.harmonic_score > 0.9:
            return TransformType.FFT, 0.85
        else:
            return TransformType.RFT_HARMONIC, 0.75
    
    # =========================================
    # Rule 6: High autocorrelation with golden structure → RFT-Beating
    # =========================================
    if features.autocorr_peak_ratio > 0.5 and features.golden_ratio_score > 0.15:
        return TransformType.RFT_BEATING, 0.7
    
    # =========================================
    # Rule 7: Default → DCT (safe for general signals)
    # =========================================
    return TransformType.DCT, 0.5


def get_best_transform_for_signal(x: np.ndarray) -> str:
    """
    Convenience function: returns transform name string.
    """
    transform_type, _ = classify_signal(x)
    return transform_type.value


class AdaptiveRouter:
    """
    Adaptive transform router for H3 codec integration.
    
    Analyzes signal segments and routes to optimal transform.
    """
    
    def __init__(self, enable_rft: bool = True, enable_adaptive: bool = True):
        self.enable_rft = enable_rft
        self.enable_adaptive = enable_adaptive
        self.stats: Dict[str, int] = {}
    
    def route(self, x: np.ndarray) -> str:
        """Select best transform for this signal segment."""
        
        if not self.enable_adaptive:
            return 'dct'  # Default
        
        transform_type, confidence = classify_signal(x)
        
        # Don't use RFT if disabled
        if not self.enable_rft and transform_type.value.startswith('rft'):
            transform_type = TransformType.DCT
        
        # Track statistics
        name = transform_type.value
        self.stats[name] = self.stats.get(name, 0) + 1
        
        return name
    
    def get_routing_stats(self) -> Dict[str, int]:
        """Get routing statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset routing statistics."""
        self.stats = {}


# =============================================================================
# TRANSFORM APPLICATION
# =============================================================================

def apply_transform(x: np.ndarray, transform: str) -> np.ndarray:
    """Apply the specified transform to signal."""
    
    if transform == 'fft':
        return fft(x, norm='ortho')
    
    elif transform == 'dct':
        return dct(x, norm='ortho')
    
    elif transform.startswith('rft_'):
        # Use operator-based RFT
        from algorithms.rft.variants.operator_variants import get_operator_variant
        variant_map = {
            'rft_golden': 'rft_golden',
            'rft_fibonacci': 'rft_fibonacci',
            'rft_harmonic': 'rft_harmonic',
            'rft_geometric': 'rft_geometric',
            'rft_beating': 'rft_beating',
            'rft_phyllotaxis': 'rft_phyllotaxis',
        }
        variant = variant_map.get(transform, 'rft_golden')
        Phi = get_operator_variant(variant, len(x))
        return Phi.T @ x
    
    elif transform == 'arft':
        # Adaptive RFT using signal's own autocorrelation
        from algorithms.rft.variants.operator_variants import generate_rft_adaptive
        Phi = generate_rft_adaptive(len(x), x)
        return Phi.T @ x
    
    else:
        raise ValueError(f"Unknown transform: {transform}")


def apply_inverse_transform(coeffs: np.ndarray, transform: str) -> np.ndarray:
    """Apply the specified inverse transform."""
    
    if transform == 'fft':
        return np.fft.ifft(coeffs, norm='ortho')
    
    elif transform == 'dct':
        from scipy.fft import idct
        return idct(coeffs, norm='ortho')
    
    elif transform.startswith('rft_'):
        from algorithms.rft.variants.operator_variants import get_operator_variant
        variant_map = {
            'rft_golden': 'rft_golden',
            'rft_fibonacci': 'rft_fibonacci',
            'rft_harmonic': 'rft_harmonic',
            'rft_geometric': 'rft_geometric',
            'rft_beating': 'rft_beating',
            'rft_phyllotaxis': 'rft_phyllotaxis',
        }
        variant = variant_map.get(transform, 'rft_golden')
        Phi = get_operator_variant(variant, len(coeffs))
        return Phi @ coeffs
    
    else:
        raise ValueError(f"Unknown transform: {transform}")


# =============================================================================
# BENCHMARKING
# =============================================================================

def benchmark_classifier():
    """Benchmark the signal classifier on known signal types."""
    
    print("=" * 70)
    print("SIGNAL CLASSIFIER BENCHMARK")
    print("=" * 70)
    
    N = 256
    
    # Test signals with expected classifications
    test_signals = {
        'golden_qp': (
            lambda: np.sin(2*np.pi*10*np.linspace(0,1,N)) + 
                    np.sin(2*np.pi*10*PHI*np.linspace(0,1,N)),
            ['rft_golden', 'rft_beating']
        ),
        'pure_sine': (
            lambda: np.sin(2*np.pi*7*np.linspace(0,1,N)),
            ['fft']
        ),
        'harmonic_series': (
            lambda: sum((1/h)*np.sin(2*np.pi*h*100*np.linspace(0,1,N)) for h in range(1,6)),
            ['rft_harmonic', 'fft']
        ),
        'white_noise': (
            lambda: np.random.randn(N),
            ['dct']
        ),
        'fibonacci_mod': (
            lambda: sum(np.sin(2*np.pi*f*10*np.linspace(0,1,N))/f for f in [1,1,2,3,5,8]),
            ['rft_fibonacci', 'rft_golden']
        ),
    }
    
    print(f"{'Signal':<20} {'Classified':<15} {'Expected':<25} {'Match':<8}")
    print("-" * 70)
    
    correct = 0
    total = 0
    
    for name, (gen_fn, expected) in test_signals.items():
        x = gen_fn()
        transform, confidence = classify_signal(x)
        match = transform.value in expected
        
        print(f"{name:<20} {transform.value:<15} {str(expected):<25} {'✓' if match else '✗'}")
        
        if match:
            correct += 1
        total += 1
    
    print("-" * 70)
    print(f"Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    print("=" * 70)


def benchmark_routing_improvement():
    """Benchmark adaptive routing vs fixed transform."""
    
    print("=" * 70)
    print("ADAPTIVE ROUTING IMPROVEMENT BENCHMARK")
    print("=" * 70)
    
    N = 256
    router = AdaptiveRouter(enable_rft=True, enable_adaptive=True)
    
    # Generate diverse signal set
    signals = []
    labels = []
    
    # Golden QP signals
    for phase in [0, np.pi/4, np.pi/2]:
        t = np.linspace(0, 1, N)
        x = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*10*PHI*t + phase)
        signals.append(x)
        labels.append('golden_qp')
    
    # Pure sinusoids
    for freq in [5, 10, 20]:
        x = np.sin(2*np.pi*freq*np.linspace(0, 1, N))
        signals.append(x)
        labels.append('sine')
    
    # Harmonic series
    for f0 in [50, 100, 200]:
        t = np.linspace(0, 1, N)
        x = sum((1/h)*np.sin(2*np.pi*h*f0*t) for h in range(1, 6))
        signals.append(x)
        labels.append('harmonic')
    
    # Noise
    for seed in [1, 2, 3]:
        np.random.seed(seed)
        signals.append(np.random.randn(N))
        labels.append('noise')
    
    # Compare: fixed DCT vs adaptive routing
    results = {'dct': [], 'adaptive': []}
    
    for x, label in zip(signals, labels):
        # Fixed DCT
        coeffs_dct = dct(x, norm='ortho')
        sparsity_dct = np.sum(np.abs(coeffs_dct) > 0.01 * np.max(np.abs(coeffs_dct)))
        
        # Adaptive routing
        best_transform = router.route(x)
        coeffs_adaptive = apply_transform(x, best_transform)
        if np.iscomplexobj(coeffs_adaptive):
            coeffs_adaptive = np.abs(coeffs_adaptive)
        sparsity_adaptive = np.sum(np.abs(coeffs_adaptive) > 0.01 * np.max(np.abs(coeffs_adaptive)))
        
        results['dct'].append(sparsity_dct)
        results['adaptive'].append(sparsity_adaptive)
    
    avg_dct = np.mean(results['dct'])
    avg_adaptive = np.mean(results['adaptive'])
    
    print(f"Average non-zero coefficients (>1% of max):")
    print(f"  Fixed DCT:     {avg_dct:.1f}")
    print(f"  Adaptive:      {avg_adaptive:.1f}")
    print(f"  Improvement:   {(1 - avg_adaptive/avg_dct)*100:.1f}%")
    print()
    print("Routing statistics:")
    for transform, count in sorted(router.get_routing_stats().items()):
        print(f"  {transform}: {count}")
    print("=" * 70)


if __name__ == "__main__":
    print("Signal Classifier for Adaptive Transform Routing")
    print()
    
    benchmark_classifier()
    print()
    benchmark_routing_improvement()
