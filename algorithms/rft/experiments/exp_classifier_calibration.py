"""
Experiment 3: Classifier Calibration
=====================================

Goal: Calibrate the signal classifier thresholds using oracle data.

Currently the classifier uses hardcoded thresholds (0.3, 0.7, etc.)
that are NOT derived from any training data. This experiment:

1. Runs ALL transforms on ALL test signals
2. Identifies the oracle-best transform for each signal
3. Extracts features for each signal
4. Finds optimal thresholds via grid search

This produces calibrated thresholds that can replace the guesses.
"""

import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from itertools import product

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scipy.fft import fft, ifft, dct, idct
from algorithms.rft.variants.operator_variants import get_operator_variant, PHI
from algorithms.rft.routing.signal_classifier import (
    extract_features, SignalFeatures, TransformType
)


# =============================================================================
# SIGNAL GENERATORS
# =============================================================================

def gen_golden_qp(N, f0=10.0, phase=0.0):
    t = np.linspace(0, 1, N)
    return np.sin(2*np.pi*f0*t) + np.sin(2*np.pi*f0*PHI*t + phase)


def gen_fibonacci_mod(N, f0=10.0, depth=5):
    t = np.linspace(0, 1, N)
    fib = [1, 1]
    for _ in range(depth):
        fib.append(fib[-1] + fib[-2])
    x = np.zeros(N)
    for i, f in enumerate(fib):
        x += (1.0/f) * np.sin(2*np.pi*f0*(PHI**i)*t)
    return x / (np.max(np.abs(x)) + 1e-10)


def gen_harmonic(N, f0=100.0):
    t = np.linspace(0, 1, N)
    x = np.zeros(N)
    for h in range(1, 8):
        x += (1.0/h) * np.sin(2*np.pi*h*f0*t)
    return x / (np.max(np.abs(x)) + 1e-10)


def gen_pure_sine(N, freq=7.0):
    t = np.linspace(0, 1, N)
    return np.sin(2*np.pi*freq*t)


def gen_chirp(N):
    t = np.linspace(0, 1, N)
    return np.sin(2*np.pi*(5 + 20*t)*t)


def gen_square(N, freq=4.0):
    t = np.linspace(0, 1, N)
    return np.sign(np.sin(2*np.pi*freq*t))


def gen_noise(N, seed=42):
    np.random.seed(seed)
    return np.random.randn(N)


def gen_transient(N, pos=0.3, width=0.05):
    """Localized transient (DCT optimal)."""
    t = np.linspace(0, 1, N)
    return np.exp(-((t - pos)/width)**2)


def gen_damped_sine(N, freq=10.0, decay=5.0):
    """Damped sinusoid."""
    t = np.linspace(0, 1, N)
    return np.sin(2*np.pi*freq*t) * np.exp(-decay*t)


def gen_am_modulated(N, carrier=50.0, mod=5.0):
    """AM modulated signal."""
    t = np.linspace(0, 1, N)
    return (1 + 0.5*np.sin(2*np.pi*mod*t)) * np.sin(2*np.pi*carrier*t)


# =============================================================================
# TRANSFORM & PSNR
# =============================================================================

def compute_psnr(x: np.ndarray, transform: str, keep_frac: float = 0.1) -> float:
    """Compute PSNR for given transform at fixed compression ratio."""
    n = len(x)
    
    if transform == 'fft':
        coeffs = fft(x, norm='ortho')
        inv_fn = lambda c: ifft(c, norm='ortho').real
    elif transform == 'dct':
        coeffs = dct(x, norm='ortho')
        inv_fn = lambda c: idct(c, norm='ortho')
    elif transform.startswith('rft_'):
        Phi = get_operator_variant(transform, n)
        coeffs = Phi.T @ x
        inv_fn = lambda c: Phi @ c
    else:
        return -np.inf
    
    # Keep top-k coefficients
    k = max(1, int(n * keep_frac))
    magnitudes = np.abs(coeffs)
    indices = np.argsort(magnitudes)[::-1][:k]
    
    sparse = np.zeros_like(coeffs)
    sparse[indices] = coeffs[indices]
    
    x_rec = inv_fn(sparse)
    if np.iscomplexobj(x_rec):
        x_rec = x_rec.real
    
    mse = np.mean((x - x_rec)**2)
    if mse < 1e-15:
        return 100.0
    
    max_val = np.max(np.abs(x))
    return 10 * np.log10(max_val**2 / mse)


# =============================================================================
# CALIBRATION
# =============================================================================

@dataclass
class CalibrationSample:
    """One sample for calibration."""
    signal_name: str
    features: SignalFeatures
    oracle_transform: str
    psnrs: Dict[str, float]


def collect_samples(N: int = 256) -> List[CalibrationSample]:
    """Generate all calibration samples."""
    
    transforms = ['fft', 'dct', 'rft_golden', 'rft_fibonacci', 
                  'rft_harmonic', 'rft_geometric']
    
    # Expanded signal set for better calibration
    signals = {
        # RFT-optimal signals
        'golden_qp_0': gen_golden_qp(N, phase=0),
        'golden_qp_45': gen_golden_qp(N, phase=np.pi/4),
        'golden_qp_90': gen_golden_qp(N, phase=np.pi/2),
        'fibonacci_d3': gen_fibonacci_mod(N, depth=3),
        'fibonacci_d5': gen_fibonacci_mod(N, depth=5),
        'fibonacci_d7': gen_fibonacci_mod(N, depth=7),
        
        # FFT-optimal signals
        'pure_sine_5': gen_pure_sine(N, freq=5),
        'pure_sine_10': gen_pure_sine(N, freq=10),
        'pure_sine_20': gen_pure_sine(N, freq=20),
        'pure_sine_50': gen_pure_sine(N, freq=50),
        
        # DCT-optimal signals  
        'transient_03': gen_transient(N, pos=0.3),
        'transient_05': gen_transient(N, pos=0.5),
        'transient_07': gen_transient(N, pos=0.7),
        'square_2': gen_square(N, freq=2),
        'square_4': gen_square(N, freq=4),
        
        # Mixed/challenging signals
        'chirp': gen_chirp(N),
        'harmonic': gen_harmonic(N),
        'damped_sine': gen_damped_sine(N),
        'am_mod': gen_am_modulated(N),
        
        # Noise (should use DCT or fail gracefully)
        'noise_1': gen_noise(N, seed=1),
        'noise_2': gen_noise(N, seed=2),
        'noise_3': gen_noise(N, seed=3),
    }
    
    samples = []
    
    for sig_name, x in signals.items():
        psnrs = {}
        for t in transforms:
            try:
                psnrs[t] = compute_psnr(x, t)
            except:
                psnrs[t] = -np.inf
        
        oracle = max(psnrs, key=psnrs.get)
        features = extract_features(x)
        
        samples.append(CalibrationSample(
            signal_name=sig_name,
            features=features,
            oracle_transform=oracle,
            psnrs=psnrs,
        ))
    
    return samples


def simple_rule_accuracy(samples: List[CalibrationSample],
                         periodicity_thresh: float,
                         sparsity_thresh: float,
                         golden_thresh: float) -> Tuple[float, Dict]:
    """
    Evaluate accuracy of simple threshold-based rules.
    
    Rules:
    - If spectral_sparsity < sparsity_thresh AND golden_coherence > golden_thresh: RFT
    - If spectral_sparsity < periodicity_thresh: FFT
    - Else: DCT
    """
    correct = 0
    predictions = {}
    
    for s in samples:
        f = s.features
        
        # Decision logic (using actual SignalFeatures attributes)
        # spectral_spread = how spread the spectrum is (low = sparse = periodic)
        # golden_ratio_score = autocorr at golden-ratio lags
        if f.spectral_spread < sparsity_thresh * 50 and f.golden_ratio_score > golden_thresh:
            pred = 'rft'  # Any RFT variant
        elif f.periodicity_score > periodicity_thresh:
            pred = 'fft'
        else:
            pred = 'dct'
        
        # Check if correct (allow any RFT variant to match)
        oracle = s.oracle_transform
        if pred == 'rft' and oracle.startswith('rft_'):
            correct += 1
        elif pred == oracle:
            correct += 1
        
        predictions[s.signal_name] = (pred, oracle)
    
    return correct / len(samples), predictions


def grid_search_thresholds(samples: List[CalibrationSample]) -> Dict[str, float]:
    """Find optimal thresholds via grid search."""
    
    best_acc = 0
    best_params = None
    
    # Coarse grid
    periodicity_range = np.arange(0.05, 0.5, 0.05)
    sparsity_range = np.arange(0.05, 0.5, 0.05)
    golden_range = np.arange(0.1, 0.8, 0.1)
    
    for p, s, g in product(periodicity_range, sparsity_range, golden_range):
        acc, _ = simple_rule_accuracy(samples, p, s, g)
        if acc > best_acc:
            best_acc = acc
            best_params = {'periodicity': p, 'sparsity': s, 'golden': g}
    
    # Fine grid around best
    if best_params:
        p0, s0, g0 = best_params['periodicity'], best_params['sparsity'], best_params['golden']
        
        for dp, ds, dg in product([-0.02, 0, 0.02], repeat=3):
            p, s, g = p0 + dp, s0 + ds, g0 + dg
            if p > 0 and s > 0 and g > 0:
                acc, _ = simple_rule_accuracy(samples, p, s, g)
                if acc > best_acc:
                    best_acc = acc
                    best_params = {'periodicity': p, 'sparsity': s, 'golden': g}
    
    return best_params, best_acc


# =============================================================================
# MAIN
# =============================================================================

def run_experiment():
    """Run classifier calibration experiment."""
    
    print("=" * 80)
    print("EXPERIMENT 3: CLASSIFIER CALIBRATION")
    print("=" * 80)
    print()
    
    # Collect samples
    print("Collecting calibration samples...")
    samples = collect_samples(N=256)
    print(f"  Generated {len(samples)} samples")
    print()
    
    # Show oracle distribution
    print("Oracle transform distribution:")
    oracle_counts = {}
    for s in samples:
        oracle = s.oracle_transform
        oracle_counts[oracle] = oracle_counts.get(oracle, 0) + 1
    for t, c in sorted(oracle_counts.items()):
        print(f"  {t}: {c}")
    print()
    
    # Current classifier accuracy (uncalibrated)
    print("Current (uncalibrated) classifier:")
    current_acc, current_preds = simple_rule_accuracy(
        samples, 
        periodicity_thresh=0.3,  # Current hardcoded
        sparsity_thresh=0.3,     # Current hardcoded  
        golden_thresh=0.5        # Current hardcoded
    )
    print(f"  Accuracy: {100*current_acc:.1f}%")
    print()
    
    # Grid search for optimal thresholds
    print("Running grid search for optimal thresholds...")
    best_params, best_acc = grid_search_thresholds(samples)
    print(f"  Best accuracy: {100*best_acc:.1f}%")
    print(f"  Optimal thresholds:")
    print(f"    periodicity_thresh: {best_params['periodicity']:.3f}")
    print(f"    sparsity_thresh: {best_params['sparsity']:.3f}")
    print(f"    golden_thresh: {best_params['golden']:.3f}")
    print()
    
    # Show predictions with optimal thresholds
    print("Detailed predictions (optimal thresholds):")
    print("-" * 60)
    print(f"{'Signal':<20} {'Pred':<10} {'Oracle':<15} {'Match'}")
    print("-" * 60)
    
    _, opt_preds = simple_rule_accuracy(
        samples,
        best_params['periodicity'],
        best_params['sparsity'],
        best_params['golden']
    )
    
    for s in samples:
        pred, oracle = opt_preds[s.signal_name]
        is_match = (pred == oracle) or (pred == 'rft' and oracle.startswith('rft_'))
        print(f"{s.signal_name:<20} {pred:<10} {oracle:<15} {'✓' if is_match else '✗'}")
    
    print("-" * 60)
    print()
    
    # Feature analysis
    print("Feature analysis by oracle class:")
    print("-" * 80)
    
    feature_stats = {}  # class -> list of features
    for s in samples:
        # Simplify oracle class
        if s.oracle_transform.startswith('rft_'):
            cls = 'rft'
        else:
            cls = s.oracle_transform
        
        if cls not in feature_stats:
            feature_stats[cls] = []
        feature_stats[cls].append(s.features)
    
    for cls in ['fft', 'dct', 'rft']:
        if cls in feature_stats:
            feats = feature_stats[cls]
            print(f"\n{cls.upper()} (n={len(feats)}):")
            print(f"  spectral_spread:     {np.mean([f.spectral_spread for f in feats]):.3f} ± {np.std([f.spectral_spread for f in feats]):.3f}")
            print(f"  golden_ratio_score:  {np.mean([f.golden_ratio_score for f in feats]):.3f} ± {np.std([f.golden_ratio_score for f in feats]):.3f}")
            print(f"  harmonic_score:      {np.mean([f.harmonic_score for f in feats]):.3f} ± {np.std([f.harmonic_score for f in feats]):.3f}")
            print(f"  periodicity_score:   {np.mean([f.periodicity_score for f in feats]):.3f} ± {np.std([f.periodicity_score for f in feats]):.3f}")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    improvement = best_acc - current_acc
    print(f"\n1. Update thresholds in signal_classifier.py:")
    print(f"   - periodicity_threshold: 0.3 → {best_params['periodicity']:.3f}")
    print(f"   - sparsity_threshold: 0.3 → {best_params['sparsity']:.3f}")
    print(f"   - golden_threshold: 0.5 → {best_params['golden']:.3f}")
    print(f"\n2. Expected accuracy improvement: {100*current_acc:.1f}% → {100*best_acc:.1f}% (+{100*improvement:.1f}%)")
    
    if best_acc < 0.7:
        print("\n3. WARNING: Even calibrated accuracy is below 70%")
        print("   Consider using more sophisticated classifier (SVM, decision tree)")
        print("   Or: expand feature set (add envelope analysis, modulation detection)")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    run_experiment()
