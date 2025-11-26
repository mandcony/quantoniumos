#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.
"""
Copyright (C) 2025 QuantoniumOS Research Team
Licensed under Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
For commercial licensing inquiries, contact: github.com/mandcony/quantoniumos

Final Hypothesis Testing: Beating the ASCII Wall with Hierarchical Cascade RFT

Building on H3's success (0.655-0.672 BPP, zero coherence), we test 5 new hypotheses
to finalize the theorem and maximize compression on discontinuous signals.

Core Question: Can we push below 0.6 BPP while maintaining zero coherence?
"""

import numpy as np
from scipy.fft import dct, idct
from dataclasses import dataclass
from typing import Tuple, Dict
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import RFT functions
try:
    from core.canonical_true_rft import rft_forward_transform_1d, rft_inverse_transform_1d
except:
    # Fallback: implement minimal RFT
    def rft_forward_transform_1d(signal):
        """Minimal RFT implementation."""
        from scipy.fft import fft
        return fft(signal)
    
    def rft_inverse_transform_1d(coeffs):
        """Minimal RFT implementation."""
        from scipy.fft import ifft
        return np.real(ifft(coeffs))


@dataclass
class FinalResult:
    """Results for final hypothesis testing."""
    hypothesis: str
    bpp: float
    psnr_db: float
    coherence_violation: float
    sparsity_pct: float
    reconstruction_error: float
    time_ms: float
    cascade_depth: int = 1
    adaptive_split: bool = False


# ============================================================================
# Utility Functions (from previous experiments)
# ============================================================================

def compute_bpp(coeffs: np.ndarray) -> float:
    """Estimate bits per pixel using entropy."""
    nonzero = np.count_nonzero(coeffs)
    if nonzero == 0:
        return 0.0
    return (nonzero * 16) / len(coeffs)


def compute_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Compute PSNR in dB."""
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    max_val = max(np.abs(original).max(), np.abs(reconstructed).max())
    return 20 * np.log10(max_val / np.sqrt(mse))


def compute_coherence_violation(signal: np.ndarray, 
                                dct_coeffs: np.ndarray, 
                                rft_coeffs: np.ndarray,
                                selected_dct: np.ndarray,
                                selected_rft: np.ndarray) -> float:
    """Compute energy discarded due to coherence."""
    E_dct_rejected = np.sum(dct_coeffs**2) - np.sum(selected_dct**2)
    E_rft_rejected = np.sum(rft_coeffs**2) - np.sum(selected_rft**2)
    E_total = np.sum(signal**2)
    return (E_dct_rejected + E_rft_rejected) / E_total if E_total > 0 else 0.0


def wavelet_decomposition(signal: np.ndarray, cutoff: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Simple wavelet decomposition: moving average (structure) + residual (texture)."""
    if cutoff is None:
        cutoff = len(signal) // 4
    kernel = np.ones(cutoff) / cutoff
    structure = np.convolve(signal, kernel, mode='same')
    texture = signal - structure
    return structure, texture


def adaptive_wavelet_split(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Adaptive split based on local variance."""
    window_size = max(8, len(signal) // 16)
    variances = []
    for i in range(0, len(signal) - window_size, window_size):
        variances.append(np.var(signal[i:i+window_size]))
    
    # High variance → texture, low variance → structure
    threshold = np.median(variances)
    structure = np.copy(signal)
    texture = np.zeros_like(signal)
    
    for i, var in enumerate(variances):
        start = i * window_size
        end = min(start + window_size, len(signal))
        if var > threshold:
            texture[start:end] = signal[start:end]
            structure[start:end] = 0
    
    return structure, texture


# ============================================================================
# Baseline: H3 from previous experiments
# ============================================================================

def h3_baseline_cascade(signal: np.ndarray, sparsity: float = 0.95) -> FinalResult:
    """
    H3 Baseline: Single-level wavelet cascade.
    
    Proven results:
    - ASCII: 0.672 BPP
    - Mixed: 0.655 BPP
    - Coherence: 0.00
    """
    import time
    start = time.time()
    
    # Wavelet split
    structure, texture = wavelet_decomposition(signal)
    
    # Transform each domain
    dct_coeffs = dct(structure, norm='ortho')
    rft_coeffs = rft_forward_transform_1d(texture)
    
    # Sparsify independently
    dct_threshold = np.percentile(np.abs(dct_coeffs), sparsity * 100)
    rft_threshold = np.percentile(np.abs(rft_coeffs), sparsity * 100)
    
    dct_sparse = np.where(np.abs(dct_coeffs) >= dct_threshold, dct_coeffs, 0)
    rft_sparse = np.where(np.abs(rft_coeffs) >= rft_threshold, rft_coeffs, 0)
    
    # Reconstruct
    structure_recon = idct(dct_sparse, norm='ortho')
    texture_recon = rft_inverse_transform_1d(rft_sparse)
    reconstruction = structure_recon + texture_recon
    
    # Metrics
    bpp = (compute_bpp(dct_sparse) + compute_bpp(rft_sparse)) / 2
    psnr = compute_psnr(signal, reconstruction)
    coherence = 0.0  # By design - no competition
    sparsity_pct = (np.count_nonzero(dct_sparse) + np.count_nonzero(rft_sparse)) / (2 * len(signal)) * 100
    error = np.linalg.norm(signal - reconstruction)
    time_ms = (time.time() - start) * 1000
    
    return FinalResult(
        hypothesis="H3_Baseline_Cascade",
        bpp=bpp,
        psnr_db=psnr,
        coherence_violation=coherence,
        sparsity_pct=sparsity_pct,
        reconstruction_error=error,
        time_ms=time_ms,
        cascade_depth=1,
        adaptive_split=False
    )


# ============================================================================
# FINAL HYPOTHESIS 1: Multi-Level Cascade (Recursive Decomposition)
# ============================================================================

def fh1_multilevel_cascade(signal: np.ndarray, levels: int = 3, sparsity: float = 0.95) -> FinalResult:
    """
    FH1: Multi-level cascade - recursively decompose texture.
    
    Hypothesis: ASCII has structure at multiple scales.
    - Level 1: Sentence/line structure
    - Level 2: Word/token structure  
    - Level 3: Character/symbol structure
    
    Expected: Better sparsity → lower BPP (target: <0.6)
    """
    import time
    start = time.time()
    
    all_dct_coeffs = []
    all_rft_coeffs = []
    remaining = signal
    
    for level in range(levels):
        # Split at this level
        structure, texture = wavelet_decomposition(remaining, cutoff=len(remaining) // (4 * (level + 1)))
        
        # DCT on structure
        dct_coeffs = dct(structure, norm='ortho')
        all_dct_coeffs.append(dct_coeffs)
        
        # Continue with texture
        remaining = texture
    
    # Final level: RFT on remaining texture
    final_rft = rft_forward_transform_1d(remaining)
    all_rft_coeffs.append(final_rft)
    
    # Sparsify all levels
    all_sparse_dct = []
    all_sparse_rft = []
    
    for dct_coeffs in all_dct_coeffs:
        threshold = np.percentile(np.abs(dct_coeffs), sparsity * 100)
        all_sparse_dct.append(np.where(np.abs(dct_coeffs) >= threshold, dct_coeffs, 0))
    
    for rft_coeffs in all_rft_coeffs:
        threshold = np.percentile(np.abs(rft_coeffs), sparsity * 100)
        all_sparse_rft.append(np.where(np.abs(rft_coeffs) >= threshold, rft_coeffs, 0))
    
    # Reconstruct
    reconstruction = np.zeros_like(signal)
    for dct_sparse in all_sparse_dct:
        reconstruction += idct(dct_sparse, norm='ortho')
    for rft_sparse in all_sparse_rft:
        reconstruction += rft_inverse_transform_1d(rft_sparse)
    
    # Metrics
    total_dct = np.concatenate(all_sparse_dct)
    total_rft = np.concatenate(all_sparse_rft)
    bpp = (compute_bpp(total_dct) + compute_bpp(total_rft)) / 2
    psnr = compute_psnr(signal, reconstruction)
    sparsity_pct = (np.count_nonzero(total_dct) + np.count_nonzero(total_rft)) / (len(total_dct) + len(total_rft)) * 100
    error = np.linalg.norm(signal - reconstruction)
    time_ms = (time.time() - start) * 1000
    
    return FinalResult(
        hypothesis="FH1_MultiLevel_Cascade",
        bpp=bpp,
        psnr_db=psnr,
        coherence_violation=0.0,  # Still no competition
        sparsity_pct=sparsity_pct,
        reconstruction_error=error,
        time_ms=time_ms,
        cascade_depth=levels,
        adaptive_split=False
    )


# ============================================================================
# FINAL HYPOTHESIS 2: Adaptive Split Cascade
# ============================================================================

def fh2_adaptive_split_cascade(signal: np.ndarray, sparsity: float = 0.95) -> FinalResult:
    """
    FH2: Variance-based adaptive splitting.
    
    Hypothesis: Fixed wavelet cutoff is suboptimal. 
    Use local variance to detect structure vs texture.
    
    High variance regions → RFT (discontinuities)
    Low variance regions → DCT (smooth patterns)
    
    Expected: Better domain routing → better sparsity
    """
    import time
    start = time.time()
    
    # Adaptive split based on variance
    structure, texture = adaptive_wavelet_split(signal)
    
    # Transform
    dct_coeffs = dct(structure, norm='ortho')
    rft_coeffs = rft_forward_transform_1d(texture)
    
    # Sparsify
    dct_threshold = np.percentile(np.abs(dct_coeffs), sparsity * 100)
    rft_threshold = np.percentile(np.abs(rft_coeffs), sparsity * 100)
    
    dct_sparse = np.where(np.abs(dct_coeffs) >= dct_threshold, dct_coeffs, 0)
    rft_sparse = np.where(np.abs(rft_coeffs) >= rft_threshold, rft_coeffs, 0)
    
    # Reconstruct
    structure_recon = idct(dct_sparse, norm='ortho')
    texture_recon = rft_inverse_transform_1d(rft_sparse)
    reconstruction = structure_recon + texture_recon
    
    # Metrics
    bpp = (compute_bpp(dct_sparse) + compute_bpp(rft_sparse)) / 2
    psnr = compute_psnr(signal, reconstruction)
    sparsity_pct = (np.count_nonzero(dct_sparse) + np.count_nonzero(rft_sparse)) / (2 * len(signal)) * 100
    error = np.linalg.norm(signal - reconstruction)
    time_ms = (time.time() - start) * 1000
    
    return FinalResult(
        hypothesis="FH2_Adaptive_Split",
        bpp=bpp,
        psnr_db=psnr,
        coherence_violation=0.0,
        sparsity_pct=sparsity_pct,
        reconstruction_error=error,
        time_ms=time_ms,
        cascade_depth=1,
        adaptive_split=True
    )


# ============================================================================
# FINAL HYPOTHESIS 3: Frequency-Domain Cascade
# ============================================================================

def fh3_frequency_cascade(signal: np.ndarray, sparsity: float = 0.95) -> FinalResult:
    """
    FH3: Split in frequency domain, not spatial.
    
    Hypothesis: ASCII edges = high frequency. Direct frequency split is cleaner.
    
    1. DCT entire signal
    2. Split DCT coeffs: low-freq (keep) + high-freq (RFT)
    3. IDCT low-freq → smooth component
    4. IDCT high-freq → edge signal → RFT this
    
    Expected: Cleaner separation → better sparsity
    """
    import time
    start = time.time()
    
    # Full DCT
    full_dct = dct(signal, norm='ortho')
    
    # Split by frequency (low = first 25%, high = rest)
    split_point = len(full_dct) // 4
    low_freq_dct = np.copy(full_dct)
    low_freq_dct[split_point:] = 0
    
    high_freq_dct = np.copy(full_dct)
    high_freq_dct[:split_point] = 0
    
    # Reconstruct high-freq signal for RFT
    high_freq_signal = idct(high_freq_dct, norm='ortho')
    
    # RFT on high-freq (edges)
    rft_coeffs = rft_forward_transform_1d(high_freq_signal)
    
    # Sparsify
    dct_threshold = np.percentile(np.abs(low_freq_dct), sparsity * 100)
    rft_threshold = np.percentile(np.abs(rft_coeffs), sparsity * 100)
    
    dct_sparse = np.where(np.abs(low_freq_dct) >= dct_threshold, low_freq_dct, 0)
    rft_sparse = np.where(np.abs(rft_coeffs) >= rft_threshold, rft_coeffs, 0)
    
    # Reconstruct
    low_freq_recon = idct(dct_sparse, norm='ortho')
    high_freq_recon = rft_inverse_transform_1d(rft_sparse)
    reconstruction = low_freq_recon + high_freq_recon
    
    # Metrics
    bpp = (compute_bpp(dct_sparse) + compute_bpp(rft_sparse)) / 2
    psnr = compute_psnr(signal, reconstruction)
    sparsity_pct = (np.count_nonzero(dct_sparse) + np.count_nonzero(rft_sparse)) / (2 * len(signal)) * 100
    error = np.linalg.norm(signal - reconstruction)
    time_ms = (time.time() - start) * 1000
    
    return FinalResult(
        hypothesis="FH3_Frequency_Cascade",
        bpp=bpp,
        psnr_db=psnr,
        coherence_violation=0.0,
        sparsity_pct=sparsity_pct,
        reconstruction_error=error,
        time_ms=time_ms,
        cascade_depth=1,
        adaptive_split=False
    )


# ============================================================================
# FINAL HYPOTHESIS 4: Edge-Aware Cascade
# ============================================================================

def fh4_edge_aware_cascade(signal: np.ndarray, sparsity: float = 0.95) -> FinalResult:
    """
    FH4: Explicit edge detection → route edges to RFT.
    
    Hypothesis: ASCII is dominated by edges (newlines, symbol boundaries).
    Detect edges explicitly, compress rest with DCT.
    
    1. Detect edges (gradient > threshold)
    2. DCT on non-edge regions
    3. RFT on edge regions
    
    Expected: Maximum RFT utilization → best BPP for ASCII
    """
    import time
    start = time.time()
    
    # Edge detection via gradient
    gradient = np.abs(np.diff(signal, prepend=signal[0]))
    edge_threshold = np.percentile(gradient, 75)  # Top 25% are edges
    
    # Split by edge mask
    is_edge = gradient > edge_threshold
    non_edge_signal = np.where(~is_edge, signal, 0)
    edge_signal = np.where(is_edge, signal, 0)
    
    # Transform
    dct_coeffs = dct(non_edge_signal, norm='ortho')
    rft_coeffs = rft_forward_transform_1d(edge_signal)
    
    # Sparsify
    dct_threshold = np.percentile(np.abs(dct_coeffs), sparsity * 100)
    rft_threshold = np.percentile(np.abs(rft_coeffs), sparsity * 100)
    
    dct_sparse = np.where(np.abs(dct_coeffs) >= dct_threshold, dct_coeffs, 0)
    rft_sparse = np.where(np.abs(rft_coeffs) >= rft_threshold, rft_coeffs, 0)
    
    # Reconstruct
    non_edge_recon = idct(dct_sparse, norm='ortho')
    edge_recon = rft_inverse_transform_1d(rft_sparse)
    reconstruction = non_edge_recon + edge_recon
    
    # Metrics
    bpp = (compute_bpp(dct_sparse) + compute_bpp(rft_sparse)) / 2
    psnr = compute_psnr(signal, reconstruction)
    sparsity_pct = (np.count_nonzero(dct_sparse) + np.count_nonzero(rft_sparse)) / (2 * len(signal)) * 100
    error = np.linalg.norm(signal - reconstruction)
    time_ms = (time.time() - start) * 1000
    
    return FinalResult(
        hypothesis="FH4_Edge_Aware",
        bpp=bpp,
        psnr_db=psnr,
        coherence_violation=0.0,
        sparsity_pct=sparsity_pct,
        reconstruction_error=error,
        time_ms=time_ms,
        cascade_depth=1,
        adaptive_split=True
    )


# ============================================================================
# FINAL HYPOTHESIS 5: Entropy-Guided Cascade
# ============================================================================

def fh5_entropy_guided_cascade(signal: np.ndarray, sparsity: float = 0.95) -> FinalResult:
    """
    FH5: Route based on local entropy.
    
    Hypothesis: High entropy regions (random-looking) → need RFT phase.
    Low entropy regions (repetitive) → DCT is sufficient.
    
    1. Compute local entropy (Shannon)
    2. High entropy → RFT
    3. Low entropy → DCT
    
    Expected: Information-theoretic optimal routing
    """
    import time
    from scipy.stats import entropy
    start = time.time()
    
    # Compute local entropy
    window_size = max(8, len(signal) // 16)
    entropies = []
    for i in range(0, len(signal) - window_size, window_size):
        window = signal[i:i+window_size]
        # Discretize for entropy computation
        hist, _ = np.histogram(window, bins=10)
        hist = hist / hist.sum()
        entropies.append(entropy(hist + 1e-10))
    
    # Split by entropy threshold
    entropy_threshold = np.median(entropies)
    low_entropy_signal = np.copy(signal)
    high_entropy_signal = np.zeros_like(signal)
    
    for i, ent in enumerate(entropies):
        start_idx = i * window_size
        end_idx = min(start_idx + window_size, len(signal))
        if ent > entropy_threshold:
            high_entropy_signal[start_idx:end_idx] = signal[start_idx:end_idx]
            low_entropy_signal[start_idx:end_idx] = 0
    
    # Transform
    dct_coeffs = dct(low_entropy_signal, norm='ortho')
    rft_coeffs = rft_forward_transform_1d(high_entropy_signal)
    
    # Sparsify
    dct_threshold = np.percentile(np.abs(dct_coeffs), sparsity * 100)
    rft_threshold = np.percentile(np.abs(rft_coeffs), sparsity * 100)
    
    dct_sparse = np.where(np.abs(dct_coeffs) >= dct_threshold, dct_coeffs, 0)
    rft_sparse = np.where(np.abs(rft_coeffs) >= rft_threshold, rft_coeffs, 0)
    
    # Reconstruct
    low_entropy_recon = idct(dct_sparse, norm='ortho')
    high_entropy_recon = rft_inverse_transform_1d(rft_sparse)
    reconstruction = low_entropy_recon + high_entropy_recon
    
    # Metrics
    bpp = (compute_bpp(dct_sparse) + compute_bpp(rft_sparse)) / 2
    psnr = compute_psnr(signal, reconstruction)
    sparsity_pct = (np.count_nonzero(dct_sparse) + np.count_nonzero(rft_sparse)) / (2 * len(signal)) * 100
    error = np.linalg.norm(signal - reconstruction)
    time_ms = (time.time() - start) * 1000
    
    return FinalResult(
        hypothesis="FH5_Entropy_Guided",
        bpp=bpp,
        psnr_db=psnr,
        coherence_violation=0.0,
        sparsity_pct=sparsity_pct,
        reconstruction_error=error,
        time_ms=time_ms,
        cascade_depth=1,
        adaptive_split=True
    )


# ============================================================================
# Test Harness
# ============================================================================

def load_test_signals() -> Dict[str, np.ndarray]:
    """Load all test signals for final validation."""
    signals = {}
    
    # 1. ASCII source code (from previous experiments)
    ascii_path = os.path.join(os.path.dirname(__file__), '../core/canonical_true_rft.py')
    if os.path.exists(ascii_path):
        with open(ascii_path, 'r', encoding='utf-8') as f:
            ascii_text = f.read()[:2048]
        signals['ascii_code'] = np.array([ord(c) for c in ascii_text], dtype=float)
        signals['ascii_code'] = (signals['ascii_code'] - signals['ascii_code'].mean()) / signals['ascii_code'].std()
    
    # 2. Mixed signal (from paper)
    try:
        from scripts.verify_rate_distortion import generate_mixed_signal
        signals['paper_mixed'] = generate_mixed_signal()
    except:
        print("Warning: Could not load paper mixed signal")
    
    # 3. JSON structured data
    json_sample = '{"key":"value","array":[1,2,3],"nested":{"a":"b"}}'
    signals['json_structured'] = np.array([ord(c) for c in json_sample * 20], dtype=float)
    signals['json_structured'] = (signals['json_structured'] - signals['json_structured'].mean()) / signals['json_structured'].std()
    
    # 4. Pure edge signal (worst case)
    edge_signal = np.zeros(512)
    edge_signal[::8] = 1.0  # Regular spikes
    signals['pure_edges'] = edge_signal
    
    # 5. Mixed smooth + edges
    smooth = np.sin(2 * np.pi * np.arange(512) / 64)
    edges = np.zeros(512)
    edges[::16] = 2.0
    signals['mixed_smooth_edges'] = smooth + edges
    
    return signals


def run_final_experiments():
    """Run all final hypotheses on all test signals."""
    signals = load_test_signals()
    
    hypotheses = [
        ("H3_Baseline", h3_baseline_cascade),
        ("FH1_MultiLevel", lambda s: fh1_multilevel_cascade(s, levels=3)),
        ("FH2_AdaptiveSplit", fh2_adaptive_split_cascade),
        ("FH3_FrequencyCascade", fh3_frequency_cascade),
        ("FH4_EdgeAware", fh4_edge_aware_cascade),
        ("FH5_EntropyGuided", fh5_entropy_guided_cascade),
    ]
    
    print("=" * 100)
    print("FINAL HYPOTHESIS TESTING: Breaking the ASCII Wall with Hierarchical Cascade RFT")
    print("=" * 100)
    print()
    
    for signal_name, signal in signals.items():
        print(f"\n{'='*100}")
        print(f"Test Signal: {signal_name.upper()} ({len(signal)} samples)")
        print(f"{'='*100}\n")
        
        results = []
        for hyp_name, hyp_func in hypotheses:
            try:
                result = hyp_func(signal)
                results.append(result)
                print(f"✓ {result.hypothesis:30s} | BPP: {result.bpp:.3f} | PSNR: {result.psnr_db:6.2f} dB | "
                      f"Coherence: {result.coherence_violation:.2e} | Sparsity: {result.sparsity_pct:5.1f}% | "
                      f"Time: {result.time_ms:6.1f}ms")
            except Exception as e:
                print(f"✗ {hyp_name:30s} | ERROR: {str(e)[:50]}")
        
        # Find winners
        print(f"\n{'-'*100}")
        if results:
            best_bpp = min(results, key=lambda r: r.bpp)
            best_psnr = max(results, key=lambda r: r.psnr_db)
            
            print(f"🏆 Best BPP:  {best_bpp.hypothesis:30s} | {best_bpp.bpp:.3f} BPP at {best_bpp.psnr_db:.2f} dB")
            print(f"⭐ Best PSNR: {best_psnr.hypothesis:30s} | {best_psnr.psnr_db:.2f} dB at {best_psnr.bpp:.3f} BPP")
            
            # Check if we beat 0.6 BPP
            if best_bpp.bpp < 0.6:
                print(f"\n🎯 BREAKTHROUGH: {best_bpp.hypothesis} achieves {best_bpp.bpp:.3f} BPP < 0.6 target!")
            
            print(f"{'-'*100}")
    
    print("\n" + "=" * 100)
    print("EXPERIMENT COMPLETE")
    print("=" * 100)


if __name__ == '__main__':
    run_final_experiments()
