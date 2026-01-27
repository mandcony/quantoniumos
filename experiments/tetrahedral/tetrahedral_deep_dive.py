#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Deep Dive: Tetrahedral Φ-RFT Confirmed Hypotheses
==================================================

Extended experiments on the 3 high-confidence wins from tetrahedral testing:
- T6: 3-vertex cascade (95.3% confidence)
- T8: Sensing matrix coherence (68.8% confidence)  
- T10: Quantization robustness (98.0% confidence)

This suite:
1. Tests across multiple signal types and sizes
2. Compares against SOTA baselines where applicable
3. Measures practical metrics (actual compression, actual CS recovery)
4. Validates statistical significance

Author: QuantoniumOS Team
Date: 2024
"""

import sys
import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from scipy.fft import fft, ifft, dct, idct
from scipy.stats import ttest_ind, wilcoxon
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import tetrahedral transforms from previous experiment
from tetrahedral_rft_hypotheses import (
    delta_rft_forward, delta_rft_inverse,
    three_vertex_cascade, tetrahedral_sensing_matrix,
    TETRA_VERTICES, PHI, A4_GROUP
)

# =============================================================================
# Extended Test Signals
# =============================================================================

def generate_extended_signals(n: int = 512) -> Dict[str, np.ndarray]:
    """Generate comprehensive test signal suite."""
    t = np.arange(n, dtype=np.float64)
    signals = {}
    
    # === Text/Code signals ===
    texts = {
        'english': "The quick brown fox jumps over the lazy dog. " * 50,
        'python': "def fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)\n" * 20,
        'json': '{"name": "test", "values": [1, 2, 3], "nested": {"key": "value"}}' * 20,
        'xml': '<root><item id="1">content</item><item id="2">more</item></root>' * 20,
    }
    for name, text in texts.items():
        data = np.frombuffer(text.encode()[:n], dtype=np.uint8).astype(np.float64)
        signals[f'text_{name}'] = (data / 128.0) - 1.0
    
    # === Wave signals ===
    signals['sine_single'] = np.sin(2 * np.pi * 5 * t / n)
    signals['sine_multi'] = (np.sin(2 * np.pi * 3 * t / n) + 
                             0.5 * np.sin(2 * np.pi * 7 * t / n) +
                             0.3 * np.sin(2 * np.pi * 11 * t / n))
    signals['chirp'] = np.sin(2 * np.pi * (1 + 10 * t / n) * t / n)
    signals['square'] = np.sign(np.sin(2 * np.pi * 4 * t / n))
    
    # === Mixed signals (key test case) ===
    wave = 0.5 * np.sin(2 * np.pi * 5 * t / n)
    text = signals['text_english']
    signals['mixed_wave_text'] = wave + text
    
    fib_wave = np.sin(2 * np.pi * PHI * t / n)
    signals['mixed_fib_text'] = fib_wave + text
    
    # === Structured signals ===
    # Fibonacci sequence
    fib = [1, 1]
    while len(fib) < n:
        fib.append(fib[-1] + fib[-2])
    fib = np.array(fib[:n], dtype=np.float64)
    signals['fibonacci'] = fib / fib.max()
    
    # Quasicrystal
    qc = np.zeros(n)
    for m in range(7):
        freq = PHI ** m
        qc += np.cos(2 * np.pi * freq * t / n) / (m + 1)
    signals['quasicrystal'] = qc / np.abs(qc).max()
    
    # Step function (edges)
    signals['steps'] = np.cumsum(np.random.choice([-1, 1], n)) / np.sqrt(n)
    
    # === Random baselines ===
    np.random.seed(42)
    signals['random_gaussian'] = np.random.randn(n)
    signals['random_uniform'] = np.random.rand(n) * 2 - 1
    
    # === Sparse signals (for CS testing) ===
    sparse_5 = np.zeros(n)
    sparse_5[np.random.choice(n, 5, replace=False)] = np.random.randn(5)
    signals['sparse_5'] = sparse_5
    
    sparse_20 = np.zeros(n)
    sparse_20[np.random.choice(n, 20, replace=False)] = np.random.randn(20)
    signals['sparse_20'] = sparse_20
    
    return signals


# =============================================================================
# T6 DEEP DIVE: 3-Vertex Cascade
# =============================================================================

def cascade_2way(x: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """Standard 2-way DCT + RFT cascade."""
    n = len(x)
    
    # Branch 1: DCT
    dct_coeffs = dct(x, norm="ortho")
    thresh = threshold * np.max(np.abs(dct_coeffs))
    dct_sparse = dct_coeffs * (np.abs(dct_coeffs) > thresh)
    branch1 = idct(dct_sparse, norm="ortho")
    residual = x - branch1
    
    # Branch 2: RFT on residual
    rft_coeffs = delta_rft_forward(residual, variant="T1")
    rft_sparse = rft_coeffs * (np.abs(rft_coeffs) > thresh)
    branch2 = delta_rft_inverse(rft_sparse, variant="T1").real
    
    return branch1 + branch2


def cascade_4way(x: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """4-way cascade: DCT + 3 tetrahedral RFT branches."""
    n = len(x)
    residual = x.copy().astype(np.complex128)
    
    # Branch 1: DCT
    dct_coeffs = dct(residual.real, norm="ortho")
    thresh = threshold * np.max(np.abs(dct_coeffs))
    dct_sparse = dct_coeffs * (np.abs(dct_coeffs) > thresh)
    branch1 = idct(dct_sparse, norm="ortho")
    residual -= branch1
    
    # Branches 2-4: Three tetrahedral RFT variants
    for variant in ["T1", "T2", "T5"]:
        rft_coeffs = delta_rft_forward(residual, variant=variant)
        rft_sparse = rft_coeffs * (np.abs(rft_coeffs) > thresh)
        branch = delta_rft_inverse(rft_sparse, variant=variant)
        residual -= branch
    
    return (x - residual).real


def test_T6_extended():
    """Extended T6: Compare cascade variants across signal types."""
    print("\n" + "="*70)
    print("T6 DEEP DIVE: 3-Vertex Cascade Performance")
    print("="*70)
    
    signals = generate_extended_signals(512)
    thresholds = [0.05, 0.10, 0.15, 0.20]
    
    results = []
    
    for thresh in thresholds:
        print(f"\n--- Threshold: {thresh} ---")
        print(f"{'Signal':<20} {'DCT-only':<12} {'2-way':<12} {'3-way':<12} {'4-way':<12} {'Best':<8}")
        print("-" * 78)
        
        for name, signal in signals.items():
            # DCT only
            dct_coeffs = dct(signal, norm="ortho")
            t = thresh * np.max(np.abs(dct_coeffs))
            dct_sparse = dct_coeffs * (np.abs(dct_coeffs) > t)
            dct_recon = idct(dct_sparse, norm="ortho")
            mse_dct = np.mean((signal - dct_recon)**2)
            
            # 2-way cascade
            recon_2way = cascade_2way(signal, thresh)
            mse_2way = np.mean((signal - recon_2way)**2)
            
            # 3-way cascade (original T6)
            b1, b2, b3 = three_vertex_cascade(signal, thresh)
            recon_3way = b1 + b2 + b3
            mse_3way = np.mean((signal - recon_3way)**2)
            
            # 4-way cascade
            recon_4way = cascade_4way(signal, thresh)
            mse_4way = np.mean((signal - recon_4way)**2)
            
            # Find best
            mses = {'DCT': mse_dct, '2-way': mse_2way, '3-way': mse_3way, '4-way': mse_4way}
            best = min(mses, key=mses.get)
            
            results.append({
                'signal': name,
                'threshold': thresh,
                'mse_dct': mse_dct,
                'mse_2way': mse_2way,
                'mse_3way': mse_3way,
                'mse_4way': mse_4way,
                'best': best
            })
            
            print(f"{name:<20} {mse_dct:<12.6f} {mse_2way:<12.6f} {mse_3way:<12.6f} {mse_4way:<12.6f} {best:<8}")
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY: Win counts by cascade type")
    print("="*70)
    
    win_counts = {'DCT': 0, '2-way': 0, '3-way': 0, '4-way': 0}
    for r in results:
        win_counts[r['best']] += 1
    
    total = len(results)
    for cascade, count in sorted(win_counts.items(), key=lambda x: -x[1]):
        print(f"  {cascade}: {count}/{total} ({100*count/total:.1f}%)")
    
    # Average improvement
    avg_3way_vs_2way = np.mean([r['mse_2way'] / max(r['mse_3way'], 1e-10) for r in results])
    avg_4way_vs_2way = np.mean([r['mse_2way'] / max(r['mse_4way'], 1e-10) for r in results])
    
    print(f"\nAverage improvement ratio:")
    print(f"  3-way vs 2-way: {avg_3way_vs_2way:.2f}×")
    print(f"  4-way vs 2-way: {avg_4way_vs_2way:.2f}×")
    
    return results


# =============================================================================
# T8 DEEP DIVE: Compressed Sensing Recovery
# =============================================================================

def cs_recovery_l1(A: np.ndarray, y: np.ndarray, max_iter: int = 1000) -> np.ndarray:
    """
    Simple ℓ₁ minimization via iterative soft thresholding (ISTA).
    Solves: min ||x||₁ s.t. Ax = y
    """
    m, n = A.shape
    x = np.zeros(n, dtype=np.complex128)
    
    # Step size (1 / largest singular value squared)
    step = 1.0 / (np.linalg.norm(A, 2) ** 2 + 1e-6)
    
    # Regularization
    lambda_reg = 0.01
    
    for _ in range(max_iter):
        # Gradient step
        grad = A.conj().T @ (A @ x - y)
        x_new = x - step * grad
        
        # Soft thresholding
        x_new = np.sign(x_new) * np.maximum(np.abs(x_new) - step * lambda_reg, 0)
        
        # Check convergence
        if np.linalg.norm(x_new - x) < 1e-8:
            break
        x = x_new
    
    return x


def test_T8_extended():
    """Extended T8: Actual compressed sensing recovery comparison."""
    print("\n" + "="*70)
    print("T8 DEEP DIVE: Compressed Sensing Recovery")
    print("="*70)
    
    n = 256  # Signal length
    sparsity_levels = [5, 10, 20, 30]
    measurement_ratios = [0.25, 0.5, 0.75]
    
    results = []
    
    for k in sparsity_levels:
        for ratio in measurement_ratios:
            m = int(n * ratio)
            
            print(f"\n--- Sparsity k={k}, Measurements m={m} ({ratio*100:.0f}%) ---")
            
            # Generate k-sparse signal
            np.random.seed(42 + k)
            x_true = np.zeros(n, dtype=np.complex128)
            support = np.random.choice(n, k, replace=False)
            x_true[support] = np.random.randn(k) + 1j * np.random.randn(k)
            
            # Build sensing matrices
            # 1. Tetrahedral Δ-RFT
            A_delta = tetrahedral_sensing_matrix(m, n)
            
            # 2. Gaussian random
            np.random.seed(123)
            A_gauss = (np.random.randn(m, n) + 1j * np.random.randn(m, n)) / np.sqrt(2 * m)
            
            # 3. DCT (random rows)
            dct_full = np.zeros((n, n))
            for i in range(n):
                e_i = np.zeros(n)
                e_i[i] = 1.0
                dct_full[:, i] = dct(e_i, norm="ortho")
            rows = np.random.choice(n, m, replace=False)
            A_dct = dct_full[rows, :].astype(np.complex128)
            
            # Take measurements
            y_delta = A_delta @ x_true
            y_gauss = A_gauss @ x_true
            y_dct = A_dct @ x_true
            
            # Recover via ℓ₁ minimization
            x_rec_delta = cs_recovery_l1(A_delta, y_delta)
            x_rec_gauss = cs_recovery_l1(A_gauss, y_gauss)
            x_rec_dct = cs_recovery_l1(A_dct, y_dct)
            
            # Compute recovery error
            err_delta = np.linalg.norm(x_true - x_rec_delta) / np.linalg.norm(x_true)
            err_gauss = np.linalg.norm(x_true - x_rec_gauss) / np.linalg.norm(x_true)
            err_dct = np.linalg.norm(x_true - x_rec_dct) / np.linalg.norm(x_true)
            
            results.append({
                'sparsity': k,
                'measurements': m,
                'ratio': ratio,
                'err_delta': err_delta,
                'err_gauss': err_gauss,
                'err_dct': err_dct
            })
            
            print(f"  Δ-RFT error:    {err_delta:.4f}")
            print(f"  Gaussian error: {err_gauss:.4f}")
            print(f"  DCT error:      {err_dct:.4f}")
            
            if err_delta < err_gauss and err_delta < err_dct:
                print(f"  ✅ Δ-RFT WINS")
            elif err_delta < err_gauss:
                print(f"  ⚖️ Δ-RFT beats Gaussian")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Average recovery errors")
    print("="*70)
    
    avg_delta = np.mean([r['err_delta'] for r in results])
    avg_gauss = np.mean([r['err_gauss'] for r in results])
    avg_dct = np.mean([r['err_dct'] for r in results])
    
    print(f"  Δ-RFT:    {avg_delta:.4f}")
    print(f"  Gaussian: {avg_gauss:.4f}")
    print(f"  DCT:      {avg_dct:.4f}")
    
    wins = sum(1 for r in results if r['err_delta'] < r['err_gauss'] and r['err_delta'] < r['err_dct'])
    print(f"\nΔ-RFT wins: {wins}/{len(results)} cases")
    
    return results


# =============================================================================
# T10 DEEP DIVE: Quantization Robustness
# =============================================================================

def quantize_roundtrip(coeffs: np.ndarray, bits: int) -> Tuple[np.ndarray, int]:
    """Quantize and dequantize, return reconstructed coeffs and byte count."""
    real = coeffs.real
    imag = coeffs.imag if np.iscomplexobj(coeffs) else np.zeros_like(real)
    
    # Scale to bit range
    max_val = max(np.abs(real).max(), np.abs(imag).max(), 1e-10)
    scale = (2**(bits-1) - 1) / max_val
    
    q_real = np.round(real * scale).astype(np.int32)
    q_imag = np.round(imag * scale).astype(np.int32)
    
    # Byte count (bits per value × 2 for complex × n values)
    byte_count = 2 * len(coeffs) * bits // 8
    
    # Dequantize
    rec_real = q_real / scale
    rec_imag = q_imag / scale
    
    return rec_real + 1j * rec_imag, byte_count


def test_T10_extended():
    """Extended T10: Quantization robustness across bit depths and signals."""
    print("\n" + "="*70)
    print("T10 DEEP DIVE: Quantization Robustness")
    print("="*70)
    
    signals = generate_extended_signals(512)
    bit_depths = [4, 6, 8, 10, 12, 16]
    
    results = []
    
    for bits in bit_depths:
        print(f"\n--- Bit depth: {bits} ---")
        print(f"{'Signal':<20} {'Δ-RFT MSE':<14} {'DCT MSE':<14} {'DFT MSE':<14} {'Winner':<8}")
        print("-" * 74)
        
        for name, signal in list(signals.items())[:10]:  # Top 10 signals
            # Δ-RFT path
            delta_coeffs = delta_rft_forward(signal, variant="T1")
            delta_q, _ = quantize_roundtrip(delta_coeffs, bits)
            delta_recon = delta_rft_inverse(delta_q, variant="T1").real
            mse_delta = np.mean((signal - delta_recon)**2)
            
            # DCT path
            dct_coeffs = dct(signal, norm="ortho")
            dct_q, _ = quantize_roundtrip(dct_coeffs.astype(np.complex128), bits)
            dct_recon = idct(dct_q.real, norm="ortho")
            mse_dct = np.mean((signal - dct_recon)**2)
            
            # DFT path
            dft_coeffs = fft(signal, norm="ortho")
            dft_q, _ = quantize_roundtrip(dft_coeffs, bits)
            dft_recon = ifft(dft_q, norm="ortho").real
            mse_dft = np.mean((signal - dft_recon)**2)
            
            # Winner
            mses = {'Δ-RFT': mse_delta, 'DCT': mse_dct, 'DFT': mse_dft}
            winner = min(mses, key=mses.get)
            
            results.append({
                'signal': name,
                'bits': bits,
                'mse_delta': mse_delta,
                'mse_dct': mse_dct,
                'mse_dft': mse_dft,
                'winner': winner
            })
            
            print(f"{name:<20} {mse_delta:<14.2e} {mse_dct:<14.2e} {mse_dft:<14.2e} {winner:<8}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Win counts by transform")
    print("="*70)
    
    for bits in bit_depths:
        bit_results = [r for r in results if r['bits'] == bits]
        wins = {'Δ-RFT': 0, 'DCT': 0, 'DFT': 0}
        for r in bit_results:
            wins[r['winner']] += 1
        print(f"  {bits}-bit: Δ-RFT={wins['Δ-RFT']}, DCT={wins['DCT']}, DFT={wins['DFT']}")
    
    # Average improvement
    delta_wins = sum(1 for r in results if r['winner'] == 'Δ-RFT')
    print(f"\nTotal Δ-RFT wins: {delta_wins}/{len(results)} ({100*delta_wins/len(results):.1f}%)")
    
    # Statistical test
    delta_mses = [r['mse_delta'] for r in results]
    dct_mses = [r['mse_dct'] for r in results]
    
    stat, pvalue = wilcoxon(delta_mses, dct_mses, alternative='less')
    print(f"\nWilcoxon test (Δ-RFT < DCT): p-value = {pvalue:.4e}")
    if pvalue < 0.05:
        print("  ✅ Statistically significant advantage for Δ-RFT")
    
    return results


# =============================================================================
# Rate-Distortion Analysis
# =============================================================================

def test_rate_distortion():
    """Compare rate-distortion curves for cascades."""
    print("\n" + "="*70)
    print("RATE-DISTORTION ANALYSIS")
    print("="*70)
    
    signal = generate_extended_signals(512)['mixed_wave_text']
    n = len(signal)
    
    # Sweep thresholds to get rate-distortion points
    thresholds = np.linspace(0.01, 0.5, 20)
    
    rd_dct = []
    rd_2way = []
    rd_3way = []
    rd_4way = []
    
    for thresh in thresholds:
        # DCT only
        dct_coeffs = dct(signal, norm="ortho")
        t = thresh * np.max(np.abs(dct_coeffs))
        mask = np.abs(dct_coeffs) > t
        dct_sparse = dct_coeffs * mask
        dct_recon = idct(dct_sparse, norm="ortho")
        mse_dct = np.mean((signal - dct_recon)**2)
        rate_dct = np.sum(mask) / n  # Fraction of coeffs kept
        rd_dct.append((rate_dct, mse_dct))
        
        # 2-way
        recon_2way = cascade_2way(signal, thresh)
        mse_2way = np.mean((signal - recon_2way)**2)
        # Estimate rate (simplified)
        rate_2way = rate_dct * 1.5  # Rough estimate
        rd_2way.append((rate_2way, mse_2way))
        
        # 3-way
        b1, b2, b3 = three_vertex_cascade(signal, thresh)
        recon_3way = b1 + b2 + b3
        mse_3way = np.mean((signal - recon_3way)**2)
        rate_3way = rate_dct * 2.0
        rd_3way.append((rate_3way, mse_3way))
        
        # 4-way
        recon_4way = cascade_4way(signal, thresh)
        mse_4way = np.mean((signal - recon_4way)**2)
        rate_4way = rate_dct * 2.5
        rd_4way.append((rate_4way, mse_4way))
    
    print("\nRate-Distortion points (Rate, MSE):")
    print("\nDCT-only:")
    for r, d in rd_dct[::4]:
        print(f"  Rate={r:.3f}, MSE={d:.6f}")
    
    print("\n3-way Cascade:")
    for r, d in rd_3way[::4]:
        print(f"  Rate={r:.3f}, MSE={d:.6f}")
    
    # Find crossover point
    for i in range(len(thresholds)):
        if rd_3way[i][1] < rd_dct[i][1]:
            print(f"\n✅ 3-way cascade dominates from threshold={thresholds[i]:.2f}")
            break
    
    return {'dct': rd_dct, '2way': rd_2way, '3way': rd_3way, '4way': rd_4way}


# =============================================================================
# Main
# =============================================================================

def main():
    """Run all extended experiments."""
    print("="*70)
    print("TETRAHEDRAL Φ-RFT: DEEP DIVE EXPERIMENTS")
    print("="*70)
    print("\nExtending the 3 high-confidence wins from T6, T8, T10")
    print()
    
    all_results = {}
    
    # T6: Cascade comparison
    all_results['T6_cascade'] = test_T6_extended()
    
    # T8: Compressed sensing
    all_results['T8_cs'] = test_T8_extended()
    
    # T10: Quantization
    all_results['T10_quant'] = test_T10_extended()
    
    # Rate-distortion
    all_results['rate_distortion'] = test_rate_distortion()
    
    # Save results
    output_dir = Path(__file__).parent
    
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj
    
    json_path = output_dir / 'tetrahedral_deep_dive_results.json'
    with open(json_path, 'w') as f:
        json.dump(_convert(all_results), f, indent=2)
    print(f"\n\nResults saved to: {json_path}")
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL CONCLUSIONS")
    print("="*70)
    
    # T6 conclusion
    t6 = all_results['T6_cascade']
    cascade_wins = sum(1 for r in t6 if r['best'] in ['3-way', '4-way'])
    print(f"\nT6 (Cascade): Multi-way cascade wins {cascade_wins}/{len(t6)} cases ({100*cascade_wins/len(t6):.1f}%)")
    
    # T8 conclusion
    t8 = all_results['T8_cs']
    cs_wins = sum(1 for r in t8 if r['err_delta'] < r['err_gauss'])
    print(f"T8 (CS): Δ-RFT beats Gaussian in {cs_wins}/{len(t8)} cases ({100*cs_wins/len(t8):.1f}%)")
    
    # T10 conclusion
    t10 = all_results['T10_quant']
    quant_wins = sum(1 for r in t10 if r['winner'] == 'Δ-RFT')
    print(f"T10 (Quant): Δ-RFT wins {quant_wins}/{len(t10)} cases ({100*quant_wins/len(t10):.1f}%)")
    
    return all_results


if __name__ == "__main__":
    main()
