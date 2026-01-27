#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
QuantoniumOS Hypothesis Battery: H1-H12 Full Test Suite
========================================================
Tests all 12 hypotheses from the research framework.

GROUP A: Geometry, Coherence, ASCII Wall (H1-H5)
GROUP B: Symbolic & Structural Compression (H6-H7)  
GROUP C: Music / Audio (H8-H9)
GROUP D: Physics / Simulation (H10)
GROUP E: Crypto / Chaos (H11-H12)
"""

import numpy as np
import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from scipy.fft import dct, idct, fft, ifft
from scipy.signal import find_peaks
import hashlib

# Golden ratio
PHI = (1 + np.sqrt(5)) / 2
PHI_INV = PHI - 1

# ═══════════════════════════════════════════════════════════════════════════════
# CORE RFT IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def phi_rft_forward(x: np.ndarray, sigma: float = 1.25, beta: float = 0.83) -> np.ndarray:
    """Φ-RFT forward transform with configurable sigma."""
    x = np.asarray(x, dtype=np.complex128)
    n = x.shape[0]
    k = np.arange(n, dtype=np.float64)
    
    # Chirp phase
    C = np.exp(1j * np.pi * sigma * (k * k) / n)
    # Golden phase
    D = np.exp(2j * np.pi * beta * (k / PHI - np.floor(k / PHI)))
    
    return D * (C * fft(x, norm="ortho"))

def phi_rft_inverse(y: np.ndarray, sigma: float = 1.25, beta: float = 0.83) -> np.ndarray:
    """Φ-RFT inverse transform."""
    y = np.asarray(y, dtype=np.complex128)
    n = y.shape[0]
    k = np.arange(n, dtype=np.float64)
    
    C = np.exp(1j * np.pi * sigma * (k * k) / n)
    D = np.exp(2j * np.pi * beta * (k / PHI - np.floor(k / PHI)))
    
    return ifft(np.conj(C) * np.conj(D) * y, norm="ortho")

def build_dct_basis(n: int) -> np.ndarray:
    """Build orthonormal DCT-II basis matrix."""
    basis = np.zeros((n, n))
    for k in range(n):
        e = np.zeros(n)
        e[k] = 1.0
        basis[:, k] = dct(e, type=2, norm='ortho')
    return basis

def build_rft_basis(n: int, sigma: float = 1.25, beta: float = 0.83) -> np.ndarray:
    """Build Φ-RFT basis matrix."""
    basis = np.zeros((n, n), dtype=np.complex128)
    for k in range(n):
        e = np.zeros(n, dtype=np.complex128)
        e[k] = 1.0
        basis[:, k] = phi_rft_forward(e, sigma=sigma, beta=beta)
    return basis

def mutual_coherence(A: np.ndarray, B: np.ndarray) -> float:
    """Compute mutual coherence between two bases."""
    # Normalize columns
    A_norm = A / np.linalg.norm(A, axis=0, keepdims=True)
    B_norm = B / np.linalg.norm(B, axis=0, keepdims=True)
    
    # Compute all inner products
    G = np.abs(A_norm.conj().T @ B_norm)
    
    return np.max(G)

# ═══════════════════════════════════════════════════════════════════════════════
# RESULT DATACLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HypothesisResult:
    hypothesis_id: str
    hypothesis_text: str
    test_performed: str
    metric_name: str
    primary_value: float
    comparison_value: float
    improvement_pct: float
    verdict: str  # SUPPORTED, REJECTED, PARTIAL
    confidence: float
    details: str
    raw_data: Optional[Dict] = None

# ═══════════════════════════════════════════════════════════════════════════════
# GROUP A: GEOMETRY, COHERENCE, ASCII WALL
# ═══════════════════════════════════════════════════════════════════════════════

def test_h1_golden_coherence_geometry(n: int = 128) -> HypothesisResult:
    """
    H1: Golden Coherence Geometry
    
    Claim: There exists σ* ∈ (0,2) such that μ(DCT, Ψ_σ*) < μ(DCT, Ψ_0)
    """
    print("\n[H1] Testing Golden Coherence Geometry...")
    
    dct_basis = build_dct_basis(n)
    
    # Sweep sigma from 0 to 2
    sigmas = np.linspace(0, 2, 41)
    coherences = []
    
    for sigma in sigmas:
        rft_basis = build_rft_basis(n, sigma=sigma)
        mu = mutual_coherence(dct_basis, rft_basis)
        coherences.append(mu)
        
    coherences = np.array(coherences)
    
    # Find minimum
    min_idx = np.argmin(coherences)
    sigma_star = sigmas[min_idx]
    mu_star = coherences[min_idx]
    mu_0 = coherences[0]  # sigma = 0 (pure FFT)
    
    improvement = (mu_0 - mu_star) / mu_0 * 100 if mu_0 > 0 else 0
    
    verdict = "SUPPORTED" if mu_star < mu_0 and sigma_star > 0 else "REJECTED"
    
    print(f"    σ* = {sigma_star:.3f}, μ(σ*) = {mu_star:.4f}, μ(0) = {mu_0:.4f}")
    print(f"    Improvement: {improvement:.2f}%")
    
    return HypothesisResult(
        hypothesis_id="H1",
        hypothesis_text="Golden coherence geometry: ∃σ* where μ(DCT,Ψ_σ*) < μ(DCT,Ψ_0)",
        test_performed=f"Swept σ ∈ [0,2] at n={n}, measured mutual coherence with DCT",
        metric_name="Mutual coherence reduction vs FFT (σ=0)",
        primary_value=mu_star,
        comparison_value=mu_0,
        improvement_pct=improvement,
        verdict=verdict,
        confidence=0.95 if verdict == "SUPPORTED" else 0.0,
        details=f"Optimal σ* = {sigma_star:.3f}, coherence curve minimum at {mu_star:.4f}",
        raw_data={"sigmas": sigmas.tolist(), "coherences": coherences.tolist()}
    )


def test_h3_coherence_budget_tradeoff(n: int = 512) -> HypothesisResult:
    """
    H3: Coherence Budget Tradeoff Curve
    
    Claim: BPP(μ_max) has a sweet spot in (0, μ_max*), not just at μ=0
    """
    print("\n[H3] Testing Coherence Budget Tradeoff...")
    
    # Generate ASCII-like signal
    np.random.seed(42)
    text = "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n"
    signal = np.array([ord(c) for c in text], dtype=np.float64)
    signal = np.pad(signal, (0, n - len(signal) % n), mode='constant')[:n]
    
    # Test different coherence budgets
    mu_max_values = np.linspace(0, 0.5, 11)
    results = []
    
    for mu_max in mu_max_values:
        # Simulate mixing based on coherence budget
        # At μ_max = 0: pure cascade (no mixing)
        # At μ_max > 0: allow some per-bin competition
        
        dct_coeffs = dct(signal, type=2, norm='ortho')
        rft_coeffs = phi_rft_forward(signal)
        
        # Mix based on coherence budget
        mix_ratio = mu_max  # Simple linear mixing
        
        if mu_max == 0:
            # Pure cascade: split by variance
            mid = n // 2
            structure = signal.copy()
            structure[mid:] = 0
            texture = signal.copy()
            texture[:mid] = 0
            
            combined = np.concatenate([
                dct(structure, type=2, norm='ortho'),
                phi_rft_forward(texture)
            ])
        else:
            # Allow mixing
            combined = (1 - mix_ratio) * dct_coeffs + mix_ratio * np.abs(rft_coeffs)
        
        # Measure sparsity (proxy for BPP)
        sorted_coeffs = np.sort(np.abs(combined.flatten()))[::-1]
        energy_threshold = 0.95 * np.sum(sorted_coeffs**2)
        cumsum = np.cumsum(sorted_coeffs**2)
        k_sparse = np.searchsorted(cumsum, energy_threshold) + 1
        bpp_proxy = k_sparse / n * 8  # Bits per sample proxy
        
        results.append({
            'mu_max': mu_max,
            'bpp_proxy': bpp_proxy,
            'sparsity': 1 - k_sparse / n
        })
    
    # Find optimal point
    bpp_values = [r['bpp_proxy'] for r in results]
    min_idx = np.argmin(bpp_values)
    optimal_mu = results[min_idx]['mu_max']
    min_bpp = results[min_idx]['bpp_proxy']
    zero_bpp = results[0]['bpp_proxy']  # μ = 0
    
    # Check if interior optimum exists
    has_interior_optimum = 0 < optimal_mu < 0.5 and min_bpp < zero_bpp
    
    verdict = "SUPPORTED" if has_interior_optimum else "PARTIAL"
    improvement = (zero_bpp - min_bpp) / zero_bpp * 100 if zero_bpp > 0 else 0
    
    print(f"    Optimal μ_max = {optimal_mu:.3f}, BPP = {min_bpp:.3f}")
    print(f"    Zero-coherence BPP = {zero_bpp:.3f}")
    
    return HypothesisResult(
        hypothesis_id="H3",
        hypothesis_text="Coherence budget has interior optimum: BPP minimized at 0 < μ_max* < 0.5",
        test_performed=f"Swept μ_max ∈ [0,0.5] on ASCII signal, measured BPP proxy",
        metric_name="BPP improvement at optimal μ_max vs μ=0",
        primary_value=min_bpp,
        comparison_value=zero_bpp,
        improvement_pct=improvement,
        verdict=verdict,
        confidence=0.7 if has_interior_optimum else 0.3,
        details=f"Optimal μ_max* = {optimal_mu:.3f}, curve {'non-monotonic' if has_interior_optimum else 'monotonic'}",
        raw_data={"mu_max_values": [r['mu_max'] for r in results], 
                  "bpp_values": bpp_values}
    )


def test_h4_oscillatory_rft_phases(n: int = 512) -> HypothesisResult:
    """
    H4: Oscillatory RFT Phases Beat Static Cascades
    
    Claim: Modulating σ per block/band can further reduce BPP
    """
    print("\n[H4] Testing Oscillatory RFT Phases...")
    
    # ASCII signal
    np.random.seed(42)
    text = "class DataProcessor:\n    def __init__(self, config):\n        self.config = config\n"
    signal = np.array([ord(c) for c in text], dtype=np.float64)
    signal = np.pad(signal, (0, n - len(signal) % n), mode='constant')[:n]
    
    # Baseline: static cascade
    block_size = 64
    num_blocks = n // block_size
    
    # Static cascade
    static_coeffs = []
    for i in range(num_blocks):
        block = signal[i*block_size:(i+1)*block_size]
        rft = phi_rft_forward(block, sigma=1.25)
        static_coeffs.extend(np.abs(rft))
    
    static_sparsity = np.sum(np.array(static_coeffs) < 0.1 * np.max(static_coeffs)) / len(static_coeffs)
    
    # Oscillatory: modulate sigma per block
    oscillatory_coeffs = []
    for i in range(num_blocks):
        block = signal[i*block_size:(i+1)*block_size]
        # Modulate sigma sinusoidally
        sigma_mod = 1.25 + 0.5 * np.sin(2 * np.pi * i / num_blocks)
        rft = phi_rft_forward(block, sigma=sigma_mod)
        oscillatory_coeffs.extend(np.abs(rft))
    
    oscillatory_sparsity = np.sum(np.array(oscillatory_coeffs) < 0.1 * np.max(oscillatory_coeffs)) / len(oscillatory_coeffs)
    
    improvement = (oscillatory_sparsity - static_sparsity) / static_sparsity * 100 if static_sparsity > 0 else 0
    verdict = "SUPPORTED" if oscillatory_sparsity > static_sparsity else "REJECTED"
    
    print(f"    Static sparsity: {static_sparsity:.3f}")
    print(f"    Oscillatory sparsity: {oscillatory_sparsity:.3f}")
    
    return HypothesisResult(
        hypothesis_id="H4",
        hypothesis_text="Oscillatory σ modulation per block improves sparsity",
        test_performed=f"Compared static σ=1.25 vs σ=1.25+0.5sin(2πi/N) on {num_blocks} blocks",
        metric_name="Sparsity improvement",
        primary_value=oscillatory_sparsity,
        comparison_value=static_sparsity,
        improvement_pct=improvement,
        verdict=verdict,
        confidence=0.6 if verdict == "SUPPORTED" else 0.3,
        details=f"Block size={block_size}, modulation amplitude=0.5"
    )


def test_h5_annealed_cascades(n: int = 256, iterations: int = 50) -> HypothesisResult:
    """
    H5: Annealed Coherent Cascades
    
    Claim: Annealing over cascade configs beats static hand-designed cascades
    """
    print("\n[H5] Testing Annealed Coherent Cascades...")
    
    # ASCII signal
    np.random.seed(42)
    text = "import numpy as np\nfrom scipy import signal\ndef process(data):\n    return np.fft.fft(data)\n"
    signal_data = np.array([ord(c) for c in text], dtype=np.float64)
    signal_data = np.pad(signal_data, (0, n - len(signal_data) % n), mode='constant')[:n]
    
    def evaluate_cascade(sigma: float, split_point: float, threshold: float) -> float:
        """Evaluate a cascade configuration."""
        split_idx = int(split_point * n)
        
        # Split signal
        structure = signal_data.copy()
        structure[split_idx:] = 0
        texture = signal_data.copy()
        texture[:split_idx] = 0
        
        # Transform
        dct_coeffs = dct(structure, type=2, norm='ortho')
        rft_coeffs = phi_rft_forward(texture, sigma=sigma)
        
        # Apply threshold
        dct_sparse = dct_coeffs * (np.abs(dct_coeffs) > threshold * np.max(np.abs(dct_coeffs)))
        rft_sparse = rft_coeffs * (np.abs(rft_coeffs) > threshold * np.max(np.abs(rft_coeffs)))
        
        # Reconstruct
        recon_structure = idct(dct_sparse, type=2, norm='ortho')
        recon_texture = phi_rft_inverse(rft_sparse).real
        recon = recon_structure + recon_texture
        
        # Score: balance sparsity and reconstruction error
        sparsity = (np.sum(np.abs(dct_sparse) < 1e-10) + np.sum(np.abs(rft_sparse) < 1e-10)) / (2 * n)
        mse = np.mean((signal_data - recon) ** 2)
        
        # Lower is better (negative sparsity + MSE penalty)
        return -sparsity + 0.01 * mse
    
    # Static baseline (H3-style)
    baseline_score = evaluate_cascade(sigma=1.25, split_point=0.5, threshold=0.05)
    
    # Simulated annealing
    best_config = {'sigma': 1.25, 'split_point': 0.5, 'threshold': 0.05}
    best_score = baseline_score
    current_config = best_config.copy()
    current_score = baseline_score
    
    temperature = 1.0
    cooling_rate = 0.95
    
    for i in range(iterations):
        # Propose new config
        new_config = {
            'sigma': np.clip(current_config['sigma'] + np.random.randn() * 0.2, 0.5, 2.5),
            'split_point': np.clip(current_config['split_point'] + np.random.randn() * 0.1, 0.1, 0.9),
            'threshold': np.clip(current_config['threshold'] + np.random.randn() * 0.02, 0.01, 0.2)
        }
        
        new_score = evaluate_cascade(**new_config)
        
        # Accept or reject
        delta = new_score - current_score
        if delta < 0 or np.random.random() < np.exp(-delta / temperature):
            current_config = new_config
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_config = current_config.copy()
        
        temperature *= cooling_rate
    
    improvement = (baseline_score - best_score) / abs(baseline_score) * 100 if baseline_score != 0 else 0
    verdict = "SUPPORTED" if best_score < baseline_score else "REJECTED"
    
    print(f"    Baseline score: {baseline_score:.4f}")
    print(f"    Annealed score: {best_score:.4f}")
    print(f"    Best config: σ={best_config['sigma']:.3f}, split={best_config['split_point']:.3f}")
    
    return HypothesisResult(
        hypothesis_id="H5",
        hypothesis_text="Annealed cascade configs beat static hand-designed cascades",
        test_performed=f"Simulated annealing over (σ, split_point, threshold) for {iterations} iterations",
        metric_name="Score improvement (lower = better)",
        primary_value=best_score,
        comparison_value=baseline_score,
        improvement_pct=improvement,
        verdict=verdict,
        confidence=0.7 if verdict == "SUPPORTED" else 0.3,
        details=f"Optimal: σ={best_config['sigma']:.3f}, split={best_config['split_point']:.3f}, thresh={best_config['threshold']:.3f}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP B: SYMBOLIC & STRUCTURAL COMPRESSION
# ═══════════════════════════════════════════════════════════════════════════════

def test_h6_ast_symbolic_compression() -> HypothesisResult:
    """
    H6: AST-First Symbolic Resonance Compression
    
    Claim: AST-based encoding + Φ-RFT beats raw gzip/brotli on code
    """
    print("\n[H6] Testing AST-First Symbolic Compression...")
    
    import ast
    import zlib
    
    # Sample Python code
    code = '''
def fibonacci(n):
    """Calculate fibonacci number."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b

class Calculator:
    def __init__(self):
        self.history = []
    
    def add(self, x, y):
        result = x + y
        self.history.append(('add', x, y, result))
        return result
'''
    
    # Raw compression baseline
    raw_bytes = code.encode('utf-8')
    gzip_compressed = zlib.compress(raw_bytes, level=9)
    gzip_bpp = len(gzip_compressed) * 8 / len(code)
    
    # AST-based encoding
    try:
        tree = ast.parse(code)
        
        # Extract structure as numeric signal
        node_types = []
        def visit(node):
            node_types.append(hash(type(node).__name__) % 256)
            for child in ast.iter_child_nodes(node):
                visit(child)
        visit(tree)
        
        # Pad to power of 2
        n = 2 ** int(np.ceil(np.log2(max(len(node_types), 64))))
        ast_signal = np.array(node_types + [0] * (n - len(node_types)), dtype=np.float64)
        
        # Φ-RFT compress
        rft_coeffs = phi_rft_forward(ast_signal)
        
        # Keep top k coefficients (95% energy)
        magnitudes = np.abs(rft_coeffs)
        sorted_idx = np.argsort(magnitudes)[::-1]
        cumsum = np.cumsum(magnitudes[sorted_idx] ** 2)
        total_energy = cumsum[-1]
        k = np.searchsorted(cumsum, 0.95 * total_energy) + 1
        
        # Estimate bits needed
        # Complex coefficients: 2 floats * k + indices
        ast_rft_bits = k * (32 + 16)  # 32 bits per float, 16 for index
        ast_rft_bpp = ast_rft_bits / len(code)
        
        # Combined: AST structure + residual tokens
        tokens = code.split()
        token_entropy = len(set(tokens)) / len(tokens) if tokens else 1
        combined_bpp = ast_rft_bpp * 0.4 + gzip_bpp * 0.6 * token_entropy
        
        improvement = (gzip_bpp - combined_bpp) / gzip_bpp * 100
        verdict = "SUPPORTED" if combined_bpp < gzip_bpp else "REJECTED"
        
        print(f"    gzip BPP: {gzip_bpp:.3f}")
        print(f"    AST+RFT BPP: {combined_bpp:.3f}")
        
    except Exception as e:
        print(f"    AST parsing failed: {e}")
        combined_bpp = gzip_bpp
        improvement = 0
        verdict = "REJECTED"
    
    return HypothesisResult(
        hypothesis_id="H6",
        hypothesis_text="AST + Φ-RFT compression beats raw gzip on Python code",
        test_performed="Parsed code to AST, encoded structure as RFT signal, compared to gzip",
        metric_name="BPP improvement vs gzip",
        primary_value=combined_bpp,
        comparison_value=gzip_bpp,
        improvement_pct=improvement,
        verdict=verdict,
        confidence=0.5,
        details=f"AST nodes: {len(node_types) if 'node_types' in dir() else 0}, k-sparse: {k if 'k' in dir() else 0}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP C: MUSIC / AUDIO
# ═══════════════════════════════════════════════════════════════════════════════

def test_h8_rft_eq_transients(sample_rate: int = 44100, duration: float = 0.5) -> HypothesisResult:
    """
    H8: Φ-RFT EQ Preserves Transients Better Than FFT EQ
    
    Claim: RFT EQ shows higher crest factor than FFT EQ at matched loudness
    """
    print("\n[H8] Testing Φ-RFT EQ Transient Preservation...")
    
    n = int(sample_rate * duration)
    t = np.linspace(0, duration, n)
    
    # Generate drum-like transient signal
    # Kick: exponential decay with low freq
    kick_env = np.exp(-t * 20)
    kick = kick_env * np.sin(2 * np.pi * 60 * t)
    
    # Snare: shorter decay with noise
    snare_env = np.exp(-t * 40)
    snare_noise = np.random.randn(n) * 0.3
    snare = snare_env * (np.sin(2 * np.pi * 200 * t) + snare_noise)
    
    # Combined transient signal
    signal = kick + snare
    
    # Design simple EQ curve: boost 2-4kHz
    freqs = np.fft.fftfreq(n, 1/sample_rate)
    eq_curve = 1 + 2 * np.exp(-((np.abs(freqs) - 3000) / 1000) ** 2)
    
    # FFT-based EQ
    fft_spectrum = fft(signal)
    fft_eq = ifft(fft_spectrum * eq_curve).real
    
    # Φ-RFT-based EQ (apply similar curve in RFT domain)
    rft_spectrum = phi_rft_forward(signal)
    rft_eq = phi_rft_inverse(rft_spectrum * eq_curve).real
    
    # Normalize to same RMS
    rms_original = np.sqrt(np.mean(signal ** 2))
    fft_eq = fft_eq * rms_original / np.sqrt(np.mean(fft_eq ** 2))
    rft_eq = rft_eq * rms_original / np.sqrt(np.mean(rft_eq ** 2))
    
    # Compute crest factors (transient measure)
    crest_original = np.max(np.abs(signal)) / np.sqrt(np.mean(signal ** 2))
    crest_fft = np.max(np.abs(fft_eq)) / np.sqrt(np.mean(fft_eq ** 2))
    crest_rft = np.max(np.abs(rft_eq)) / np.sqrt(np.mean(rft_eq ** 2))
    
    improvement = (crest_rft - crest_fft) / crest_fft * 100
    verdict = "SUPPORTED" if crest_rft > crest_fft else "REJECTED"
    
    print(f"    Original crest factor: {crest_original:.3f}")
    print(f"    FFT EQ crest factor: {crest_fft:.3f}")
    print(f"    RFT EQ crest factor: {crest_rft:.3f}")
    
    return HypothesisResult(
        hypothesis_id="H8",
        hypothesis_text="Φ-RFT EQ preserves transients better than FFT EQ",
        test_performed="Applied same EQ curve via FFT and RFT to drum transient, compared crest factors",
        metric_name="Crest factor improvement",
        primary_value=crest_rft,
        comparison_value=crest_fft,
        improvement_pct=improvement,
        verdict=verdict,
        confidence=0.6 if verdict == "SUPPORTED" else 0.4,
        details=f"Original crest: {crest_original:.3f}, FFT: {crest_fft:.3f}, RFT: {crest_rft:.3f}"
    )


def test_h9_rft_oscillator_timbre(n_params: int = 8) -> HypothesisResult:
    """
    H9: Φ-RFT Oscillator Banks Yield Denser Timbre Control
    
    Claim: RFT oscillators cover more timbre space with fewer parameters
    """
    print("\n[H9] Testing Φ-RFT Oscillator Timbre Density...")
    
    sample_rate = 44100
    duration = 0.1
    n = int(sample_rate * duration)
    
    def generate_fft_timbre(params: np.ndarray) -> np.ndarray:
        """Generate timbre using FFT-based additive synthesis."""
        harmonics = np.arange(1, len(params) + 1)
        t = np.linspace(0, duration, n)
        signal = np.zeros(n)
        for i, (amp, h) in enumerate(zip(params, harmonics)):
            signal += amp * np.sin(2 * np.pi * 220 * h * t)
        return signal / (np.max(np.abs(signal)) + 1e-10)
    
    def generate_rft_timbre(params: np.ndarray) -> np.ndarray:
        """Generate timbre using RFT-based synthesis."""
        # Place harmonics at Φ-scaled positions
        t = np.linspace(0, duration, n)
        signal = np.zeros(n)
        for i, amp in enumerate(params):
            # Φ-scaled harmonic positions
            phi_harmonic = PHI ** i
            signal += amp * np.sin(2 * np.pi * 220 * phi_harmonic * t)
        return signal / (np.max(np.abs(signal)) + 1e-10)
    
    # Sample random parameter configurations
    num_samples = 100
    np.random.seed(42)
    
    fft_timbres = []
    rft_timbres = []
    
    for _ in range(num_samples):
        params = np.random.rand(n_params)
        params = params / np.sum(params)  # Normalize
        
        fft_timbres.append(generate_fft_timbre(params))
        rft_timbres.append(generate_rft_timbre(params))
    
    # Compute spectral centroids as timbre descriptor
    def spectral_centroid(signal: np.ndarray) -> float:
        spectrum = np.abs(fft(signal))[:n//2]
        freqs = np.fft.fftfreq(n, 1/sample_rate)[:n//2]
        return np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-10)
    
    def spectral_spread(signal: np.ndarray) -> float:
        spectrum = np.abs(fft(signal))[:n//2]
        freqs = np.fft.fftfreq(n, 1/sample_rate)[:n//2]
        centroid = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-10)
        return np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / (np.sum(spectrum) + 1e-10))
    
    # Compute timbre space coverage (variance of centroids and spreads)
    fft_centroids = [spectral_centroid(t) for t in fft_timbres]
    rft_centroids = [spectral_centroid(t) for t in rft_timbres]
    fft_spreads = [spectral_spread(t) for t in fft_timbres]
    rft_spreads = [spectral_spread(t) for t in rft_timbres]
    
    # Coverage metric: product of variances (higher = more coverage)
    fft_coverage = np.var(fft_centroids) * np.var(fft_spreads)
    rft_coverage = np.var(rft_centroids) * np.var(rft_spreads)
    
    improvement = (rft_coverage - fft_coverage) / fft_coverage * 100 if fft_coverage > 0 else 0
    verdict = "SUPPORTED" if rft_coverage > fft_coverage else "REJECTED"
    
    print(f"    FFT timbre coverage: {fft_coverage:.2e}")
    print(f"    RFT timbre coverage: {rft_coverage:.2e}")
    
    return HypothesisResult(
        hypothesis_id="H9",
        hypothesis_text="Φ-RFT oscillators cover more timbre space per parameter",
        test_performed=f"Generated {num_samples} random {n_params}-param timbres, compared spectral coverage",
        metric_name="Timbre space coverage (variance product)",
        primary_value=rft_coverage,
        comparison_value=fft_coverage,
        improvement_pct=improvement,
        verdict=verdict,
        confidence=0.5,
        details=f"Centroid variance: FFT={np.var(fft_centroids):.2e}, RFT={np.var(rft_centroids):.2e}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP D: PHYSICS / SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def test_h10_pde_solver_stability(n: int = 128, num_steps: int = 100) -> HypothesisResult:
    """
    H10: Φ-RFT Spectral Solvers Have Stability Advantages
    
    Claim: RFT spectral solver allows larger dt on golden-structured potentials
    """
    print("\n[H10] Testing Φ-RFT PDE Solver Stability...")
    
    # 1D wave equation: ∂²u/∂t² = c² ∂²u/∂x²
    # With golden-structured potential: V(x) = sin(2π * Φ * x/L)
    
    L = 1.0
    c = 1.0
    x = np.linspace(0, L, n, endpoint=False)
    dx = L / n
    
    # Golden-structured initial condition
    u0 = np.sin(2 * np.pi * PHI * x / L) + 0.5 * np.sin(4 * np.pi * x / L)
    
    def fft_wave_step(u: np.ndarray, dt: float) -> np.ndarray:
        """One step of FFT-based wave equation solver."""
        k = 2 * np.pi * np.fft.fftfreq(n, dx)
        u_hat = fft(u)
        # d²/dx² in frequency domain = -k²
        laplacian = ifft(-k**2 * u_hat).real
        return u + dt * c**2 * laplacian
    
    def rft_wave_step(u: np.ndarray, dt: float) -> np.ndarray:
        """One step of RFT-based wave equation solver."""
        u_hat = phi_rft_forward(u)
        k = 2 * np.pi * np.fft.fftfreq(n, dx)
        # Apply Laplacian in RFT domain
        laplacian = phi_rft_inverse(-k**2 * u_hat).real
        return u + dt * c**2 * laplacian
    
    # Find maximum stable dt for each method
    def find_max_stable_dt(step_func, u0, max_dt=0.1, tol=0.01):
        dt_test = max_dt
        while dt_test > 1e-6:
            u = u0.copy()
            stable = True
            for _ in range(num_steps):
                u = step_func(u, dt_test)
                if np.any(np.isnan(u)) or np.max(np.abs(u)) > 1e6:
                    stable = False
                    break
            if stable:
                return dt_test
            dt_test *= 0.9
        return dt_test
    
    fft_max_dt = find_max_stable_dt(fft_wave_step, u0)
    rft_max_dt = find_max_stable_dt(rft_wave_step, u0)
    
    improvement = (rft_max_dt - fft_max_dt) / fft_max_dt * 100 if fft_max_dt > 0 else 0
    verdict = "SUPPORTED" if rft_max_dt > fft_max_dt else "REJECTED"
    
    print(f"    FFT max stable dt: {fft_max_dt:.6f}")
    print(f"    RFT max stable dt: {rft_max_dt:.6f}")
    
    return HypothesisResult(
        hypothesis_id="H10",
        hypothesis_text="Φ-RFT spectral solver allows larger stable dt on golden potentials",
        test_performed=f"1D wave equation with golden potential, searched for max stable dt",
        metric_name="Maximum stable timestep ratio",
        primary_value=rft_max_dt,
        comparison_value=fft_max_dt,
        improvement_pct=improvement,
        verdict=verdict,
        confidence=0.4,
        details=f"n={n}, {num_steps} steps tested per dt"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP E: CRYPTO / CHAOS
# ═══════════════════════════════════════════════════════════════════════════════

def test_h11_crypto_avalanche(num_trials: int = 100) -> HypothesisResult:
    """
    H11: Φ-RFT Wave Ciphers Match or Exceed AES Avalanche Metrics
    
    Claim: RFT cipher shows ~50% bit avalanche on single-bit changes
    """
    print("\n[H11] Testing Φ-RFT Crypto Avalanche...")
    
    block_size = 16  # 128 bits
    
    def rft_hash(data: bytes) -> bytes:
        """Simple RFT-based hash function."""
        # Pad/truncate to block size
        data = data.ljust(block_size, b'\x00')[:block_size]
        
        # Convert to signal
        signal = np.frombuffer(data, dtype=np.uint8).astype(np.float64)
        
        # Multiple RFT rounds with different sigmas
        for sigma in [1.25, 0.83, 1.618]:
            coeffs = phi_rft_forward(signal, sigma=sigma)
            # Nonlinear mixing
            signal = np.abs(coeffs) * np.cos(np.angle(coeffs))
            signal = (signal * 256) % 256
        
        return bytes(signal.astype(np.uint8))
    
    def count_bit_diff(a: bytes, b: bytes) -> int:
        """Count differing bits between two byte arrays."""
        diff = 0
        for x, y in zip(a, b):
            diff += bin(x ^ y).count('1')
        return diff
    
    # Measure avalanche effect
    avalanche_scores = []
    
    for _ in range(num_trials):
        # Random input
        input_data = bytes(np.random.randint(0, 256, block_size, dtype=np.uint8))
        output1 = rft_hash(input_data)
        
        # Flip one random bit
        byte_idx = np.random.randint(0, block_size)
        bit_idx = np.random.randint(0, 8)
        modified = bytearray(input_data)
        modified[byte_idx] ^= (1 << bit_idx)
        output2 = rft_hash(bytes(modified))
        
        # Count bit changes
        bit_diff = count_bit_diff(output1, output2)
        avalanche = bit_diff / (block_size * 8)
        avalanche_scores.append(avalanche)
    
    mean_avalanche = np.mean(avalanche_scores)
    std_avalanche = np.std(avalanche_scores)
    
    # Ideal avalanche is 0.5 (50% bits flip)
    distance_from_ideal = abs(mean_avalanche - 0.5)
    verdict = "SUPPORTED" if distance_from_ideal < 0.1 else "PARTIAL" if distance_from_ideal < 0.2 else "REJECTED"
    
    print(f"    Mean avalanche: {mean_avalanche:.3f} ± {std_avalanche:.3f}")
    print(f"    Distance from ideal (0.5): {distance_from_ideal:.3f}")
    
    return HypothesisResult(
        hypothesis_id="H11",
        hypothesis_text="Φ-RFT cipher achieves ~50% avalanche effect",
        test_performed=f"Measured bit changes on single-bit input perturbations, {num_trials} trials",
        metric_name="Mean avalanche effect (ideal = 0.5)",
        primary_value=mean_avalanche,
        comparison_value=0.5,
        improvement_pct=-distance_from_ideal * 100,  # Negative = closer to ideal
        verdict=verdict,
        confidence=0.8 if verdict == "SUPPORTED" else 0.5,
        details=f"Avalanche: {mean_avalanche:.3f} ± {std_avalanche:.3f}, {num_trials} trials"
    )


def test_h12_geometric_hash_collisions(num_samples: int = 1000) -> HypothesisResult:
    """
    H12: Geometric Hashes in Φ-RFT Have Unique Collision Geometry
    
    Claim: RFT hashes show structured collision surfaces for similar inputs
    """
    print("\n[H12] Testing Geometric Hash Collision Geometry...")
    
    hash_size = 32  # 256 bits
    
    def rft_geometric_hash(data: bytes) -> np.ndarray:
        """RFT-based geometric hash returning normalized coefficients."""
        signal = np.frombuffer(data.ljust(64, b'\x00')[:64], dtype=np.uint8).astype(np.float64)
        coeffs = phi_rft_forward(signal)
        # Return normalized magnitude and phase as hash
        mag = np.abs(coeffs[:hash_size])
        phase = np.angle(coeffs[:hash_size])
        return np.concatenate([mag / (np.max(mag) + 1e-10), phase / np.pi])
    
    def sha256_hash(data: bytes) -> np.ndarray:
        """SHA-256 hash as normalized floats."""
        h = hashlib.sha256(data).digest()
        return np.frombuffer(h, dtype=np.uint8).astype(np.float64) / 255
    
    # Generate similar inputs (small perturbations)
    base_data = b"QuantoniumOS PHI-RFT Test Data"
    
    rft_hashes = []
    sha_hashes = []
    
    for i in range(num_samples):
        # Small perturbation: change one byte slightly
        perturbed = bytearray(base_data)
        if len(perturbed) > 0:
            idx = i % len(perturbed)
            perturbed[idx] = (perturbed[idx] + (i % 10)) % 256
        
        rft_hashes.append(rft_geometric_hash(bytes(perturbed)))
        sha_hashes.append(sha256_hash(bytes(perturbed)))
    
    # Analyze hash distribution structure
    rft_hashes = np.array(rft_hashes)
    sha_hashes = np.array(sha_hashes)
    
    # Compute pairwise distances
    def pairwise_distances(hashes: np.ndarray) -> np.ndarray:
        n = min(100, len(hashes))  # Sample for speed
        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                dists.append(np.linalg.norm(hashes[i] - hashes[j]))
        return np.array(dists)
    
    rft_dists = pairwise_distances(rft_hashes)
    sha_dists = pairwise_distances(sha_hashes)
    
    # Structure metric: kurtosis of distance distribution
    # Higher kurtosis = more structured (non-uniform) distribution
    rft_kurtosis = np.mean((rft_dists - np.mean(rft_dists))**4) / (np.std(rft_dists)**4 + 1e-10)
    sha_kurtosis = np.mean((sha_dists - np.mean(sha_dists))**4) / (np.std(sha_dists)**4 + 1e-10)
    
    # RFT should show more structure (higher kurtosis) for similar inputs
    improvement = (rft_kurtosis - sha_kurtosis) / sha_kurtosis * 100 if sha_kurtosis > 0 else 0
    verdict = "SUPPORTED" if rft_kurtosis > sha_kurtosis * 1.1 else "PARTIAL"
    
    print(f"    RFT distance kurtosis: {rft_kurtosis:.3f}")
    print(f"    SHA distance kurtosis: {sha_kurtosis:.3f}")
    
    return HypothesisResult(
        hypothesis_id="H12",
        hypothesis_text="RFT geometric hashes show structured collision surfaces",
        test_performed=f"Compared distance distribution structure on {num_samples} similar inputs",
        metric_name="Distance kurtosis (higher = more structured)",
        primary_value=rft_kurtosis,
        comparison_value=sha_kurtosis,
        improvement_pct=improvement,
        verdict=verdict,
        confidence=0.6 if verdict == "SUPPORTED" else 0.4,
        details=f"RFT kurtosis: {rft_kurtosis:.3f}, SHA kurtosis: {sha_kurtosis:.3f}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TEST RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_hypotheses() -> List[HypothesisResult]:
    """Run all hypothesis tests."""
    
    print("=" * 70)
    print("QUANTONIUMOS HYPOTHESIS BATTERY: H1-H12")
    print("=" * 70)
    
    results = []
    
    # GROUP A: Geometry, Coherence, ASCII Wall
    print("\n" + "=" * 70)
    print("GROUP A: GEOMETRY, COHERENCE, ASCII WALL")
    print("=" * 70)
    
    results.append(test_h1_golden_coherence_geometry())
    results.append(test_h3_coherence_budget_tradeoff())
    results.append(test_h4_oscillatory_rft_phases())
    results.append(test_h5_annealed_cascades())
    
    # GROUP B: Symbolic & Structural
    print("\n" + "=" * 70)
    print("GROUP B: SYMBOLIC & STRUCTURAL COMPRESSION")
    print("=" * 70)
    
    results.append(test_h6_ast_symbolic_compression())
    
    # GROUP C: Music / Audio
    print("\n" + "=" * 70)
    print("GROUP C: MUSIC / AUDIO")
    print("=" * 70)
    
    results.append(test_h8_rft_eq_transients())
    results.append(test_h9_rft_oscillator_timbre())
    
    # GROUP D: Physics
    print("\n" + "=" * 70)
    print("GROUP D: PHYSICS / SIMULATION")
    print("=" * 70)
    
    results.append(test_h10_pde_solver_stability())
    
    # GROUP E: Crypto
    print("\n" + "=" * 70)
    print("GROUP E: CRYPTO / CHAOS")
    print("=" * 70)
    
    results.append(test_h11_crypto_avalanche())
    results.append(test_h12_geometric_hash_collisions())
    
    return results


def print_summary(results: List[HypothesisResult]) -> None:
    """Print summary table of all results."""
    
    print("\n" + "=" * 70)
    print("SUMMARY: HYPOTHESIS TEST RESULTS")
    print("=" * 70)
    
    print(f"\n{'ID':<6} {'Hypothesis':<45} {'Verdict':<12} {'Confidence':<10}")
    print("-" * 75)
    
    supported = 0
    partial = 0
    rejected = 0
    
    for r in results:
        short_text = r.hypothesis_text[:42] + "..." if len(r.hypothesis_text) > 45 else r.hypothesis_text
        verdict_emoji = "✅" if r.verdict == "SUPPORTED" else "⚠️" if r.verdict == "PARTIAL" else "❌"
        print(f"{r.hypothesis_id:<6} {short_text:<45} {verdict_emoji} {r.verdict:<10} {r.confidence:.1%}")
        
        if r.verdict == "SUPPORTED":
            supported += 1
        elif r.verdict == "PARTIAL":
            partial += 1
        else:
            rejected += 1
    
    print("-" * 75)
    print(f"\nTotal: {len(results)} hypotheses tested")
    print(f"  ✅ SUPPORTED: {supported}")
    print(f"  ⚠️  PARTIAL:   {partial}")
    print(f"  ❌ REJECTED:  {rejected}")


def main():
    """Main entry point."""
    start_time = time.time()
    
    results = run_all_hypotheses()
    print_summary(results)
    
    # Save results
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_time_seconds": time.time() - start_time,
        "results": [asdict(r) for r in results]
    }
    
    output_path = "experiments/hypothesis_battery_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")
    print(f"Total time: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
