#!/usr/bin/env python3
"""
Rigorous Entropy and Rate Analysis for Φ-RFT Compression
=========================================================

This script provides proper information-theoretic evaluation:

1. STOCHASTIC SOURCE MODELS
   - Markov sources with known entropy H
   - Golden-structured sources (quasiperiodic)
   - Real-world ASCII/code sources

2. ENTROPY ESTIMATION
   - Empirical entropy H₀ (zero-order)
   - Conditional entropy H₁, H₂, ... (higher-order Markov)
   - Context-tree weighting (CTW) entropy bound

3. COMPRESSION RATE COMPARISON
   - Arithmetic coding baseline (near-optimal)
   - gzip/zlib (deflate)
   - brotli (modern)
   - zstd (fast modern)
   - Φ-RFT transform + quantize + entropy code

4. RATE vs ENTROPY ANALYSIS
   - Plot rate vs entropy for different sources
   - Show gap to theoretical limit
"""

import numpy as np
import zlib
import json
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import Counter
import struct

# Try to import optional compressors
try:
    import brotli
    BROTLI_AVAILABLE = True
except ImportError:
    BROTLI_AVAILABLE = False

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False

# Golden ratio
PHI = (1 + np.sqrt(5)) / 2

# ═══════════════════════════════════════════════════════════════════════════════
# ENTROPY ESTIMATION
# ═══════════════════════════════════════════════════════════════════════════════

def empirical_entropy_h0(data: bytes) -> float:
    """
    Compute zero-order empirical entropy H₀.
    
    H₀ = -Σ p(x) log₂ p(x)
    
    This is the entropy assuming i.i.d. symbols.
    """
    if len(data) == 0:
        return 0.0
    
    counts = Counter(data)
    n = len(data)
    
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / n
            entropy -= p * np.log2(p)
    
    return entropy


def empirical_entropy_h1(data: bytes) -> float:
    """
    Compute first-order conditional entropy H₁.
    
    H₁ = H(X|X_{-1}) = -Σ p(x_{-1}) Σ p(x|x_{-1}) log₂ p(x|x_{-1})
    
    This captures first-order Markov structure.
    """
    if len(data) < 2:
        return empirical_entropy_h0(data)
    
    # Count transitions
    transitions = Counter()
    contexts = Counter()
    
    for i in range(1, len(data)):
        prev = data[i-1]
        curr = data[i]
        transitions[(prev, curr)] += 1
        contexts[prev] += 1
    
    n = len(data) - 1
    entropy = 0.0
    
    for (prev, curr), count in transitions.items():
        p_joint = count / n
        p_cond = count / contexts[prev]
        if p_cond > 0:
            entropy -= p_joint * np.log2(p_cond)
    
    return entropy


def empirical_entropy_hk(data: bytes, k: int) -> float:
    """
    Compute k-th order conditional entropy H_k.
    
    H_k = H(X|X_{-1}, ..., X_{-k})
    """
    if len(data) <= k:
        return empirical_entropy_h0(data)
    
    # Count transitions from k-length contexts
    transitions = Counter()
    contexts = Counter()
    
    for i in range(k, len(data)):
        context = tuple(data[i-k:i])
        symbol = data[i]
        transitions[(context, symbol)] += 1
        contexts[context] += 1
    
    n = len(data) - k
    entropy = 0.0
    
    for (context, symbol), count in transitions.items():
        p_joint = count / n
        p_cond = count / contexts[context]
        if p_cond > 0:
            entropy -= p_joint * np.log2(p_cond)
    
    return entropy


def entropy_rate_estimate(data: bytes, max_order: int = 8) -> Tuple[float, List[float]]:
    """
    Estimate entropy rate using increasing context orders.
    
    Returns:
        (estimated_rate, [H0, H1, H2, ...])
    
    The entropy rate is approximated by H_k for large k.
    """
    entropies = []
    
    for k in range(max_order + 1):
        if k == 0:
            h = empirical_entropy_h0(data)
        else:
            h = empirical_entropy_hk(data, k)
        entropies.append(h)
        
        # Stop if we've converged
        if k > 0 and abs(entropies[-1] - entropies[-2]) < 0.001:
            break
    
    return entropies[-1], entropies


# ═══════════════════════════════════════════════════════════════════════════════
# STOCHASTIC SOURCE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SourceModel:
    name: str
    data: bytes
    true_entropy: Optional[float]  # Known if synthetic
    description: str


def generate_iid_source(n: int, alphabet_size: int = 256, seed: int = 42) -> SourceModel:
    """
    Generate i.i.d. uniform source.
    True entropy = log₂(alphabet_size)
    """
    np.random.seed(seed)
    data = bytes(np.random.randint(0, alphabet_size, n, dtype=np.uint8))
    true_entropy = np.log2(alphabet_size)
    
    return SourceModel(
        name=f"iid_uniform_{alphabet_size}",
        data=data,
        true_entropy=true_entropy,
        description=f"i.i.d. uniform over {alphabet_size} symbols, H = {true_entropy:.3f} bits/symbol"
    )


def generate_biased_source(n: int, p: float = 0.9, seed: int = 42) -> SourceModel:
    """
    Generate biased binary source.
    True entropy = -p log₂(p) - (1-p) log₂(1-p)
    """
    np.random.seed(seed)
    data = bytes(np.random.choice([0, 1], n, p=[1-p, p]).astype(np.uint8))
    true_entropy = -p * np.log2(p) - (1-p) * np.log2(1-p) if 0 < p < 1 else 0
    
    return SourceModel(
        name=f"biased_binary_p{p:.2f}",
        data=data,
        true_entropy=true_entropy,
        description=f"Biased binary, P(1)={p}, H = {true_entropy:.3f} bits/symbol"
    )


def generate_markov_source(n: int, transition_matrix: np.ndarray, seed: int = 42) -> SourceModel:
    """
    Generate first-order Markov source with given transition matrix.
    True entropy = Σ π_i Σ p_{ij} log₂ p_{ij}
    """
    np.random.seed(seed)
    k = transition_matrix.shape[0]
    
    # Compute stationary distribution
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    stationary_idx = np.argmin(np.abs(eigenvalues - 1))
    stationary = np.abs(eigenvectors[:, stationary_idx])
    stationary = stationary / np.sum(stationary)
    
    # Compute true entropy rate
    true_entropy = 0.0
    for i in range(k):
        for j in range(k):
            p = transition_matrix[i, j]
            if p > 0:
                true_entropy -= stationary[i] * p * np.log2(p)
    
    # Generate samples
    state = np.random.choice(k, p=stationary)
    samples = [state]
    for _ in range(n - 1):
        state = np.random.choice(k, p=transition_matrix[state])
        samples.append(state)
    
    data = bytes(np.array(samples, dtype=np.uint8))
    
    return SourceModel(
        name=f"markov_{k}state",
        data=data,
        true_entropy=true_entropy,
        description=f"{k}-state Markov, H = {true_entropy:.3f} bits/symbol"
    )


def generate_golden_source(n: int, seed: int = 42) -> SourceModel:
    """
    Generate golden-ratio structured source (Fibonacci word derivative).
    
    Based on Sturmian sequences with slope φ.
    """
    np.random.seed(seed)
    
    # Generate Fibonacci word: a→ab, b→a, starting from 'a'
    def fib_word(iterations: int) -> str:
        a, b = 'a', 'ab'
        for _ in range(iterations):
            a, b = b, b + a
        return b
    
    # Get enough characters
    word = fib_word(20)[:n]
    
    # Convert to bytes with some structure
    data = bytes([ord(c) for c in word])
    
    # Sturmian sequence entropy is 0 (deterministic), but with noise:
    # Add small perturbation
    noisy_data = bytes([(b + np.random.randint(-1, 2)) % 256 for b in data])
    
    return SourceModel(
        name="golden_sturmian",
        data=noisy_data,
        true_entropy=None,  # Not trivially computable
        description="Golden/Sturmian sequence with noise"
    )


def generate_ascii_source(n: int, source_type: str = "code") -> SourceModel:
    """
    Generate realistic ASCII source.
    """
    if source_type == "code":
        # Python-like code
        template = '''
def process_data(items):
    """Process a list of items."""
    results = []
    for i, item in enumerate(items):
        if item is not None:
            value = transform(item)
            results.append(value)
    return results

class DataHandler:
    def __init__(self, config):
        self.config = config
        self.cache = {}
    
    def handle(self, data):
        key = hash(str(data))
        if key in self.cache:
            return self.cache[key]
        result = self._process(data)
        self.cache[key] = result
        return result
'''
        # Repeat to get enough data
        text = (template * (n // len(template) + 1))[:n]
        
    elif source_type == "english":
        template = """The quick brown fox jumps over the lazy dog. This is a sample 
of English text that contains various common words and patterns. Natural language 
has significant redundancy that can be exploited for compression. The entropy of 
English is estimated to be around 1-1.5 bits per character when accounting for 
long-range dependencies and context. """
        text = (template * (n // len(template) + 1))[:n]
        
    else:
        text = ''.join(chr(np.random.randint(32, 127)) for _ in range(n))
    
    data = text.encode('utf-8')[:n]
    
    return SourceModel(
        name=f"ascii_{source_type}",
        data=data,
        true_entropy=None,
        description=f"ASCII {source_type} text"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# COMPRESSION METHODS
# ═══════════════════════════════════════════════════════════════════════════════

def compress_zlib(data: bytes, level: int = 9) -> bytes:
    """zlib/deflate compression."""
    return zlib.compress(data, level)


def compress_brotli(data: bytes, quality: int = 11) -> bytes:
    """Brotli compression (if available)."""
    if BROTLI_AVAILABLE:
        return brotli.compress(data, quality=quality)
    return data


def compress_zstd(data: bytes, level: int = 22) -> bytes:
    """Zstandard compression (if available)."""
    if ZSTD_AVAILABLE:
        cctx = zstd.ZstdCompressor(level=level)
        return cctx.compress(data)
    return data


def phi_rft_forward(x: np.ndarray, sigma: float = 1.25, beta: float = 0.83) -> np.ndarray:
    """Φ-RFT forward transform."""
    from scipy.fft import fft
    x = np.asarray(x, dtype=np.complex128)
    n = x.shape[0]
    k = np.arange(n, dtype=np.float64)
    C = np.exp(1j * np.pi * sigma * (k * k) / n)
    D = np.exp(2j * np.pi * beta * (k / PHI - np.floor(k / PHI)))
    return D * (C * fft(x, norm="ortho"))


def compress_rft_quantized(data: bytes, bits_per_coeff: int = 8) -> bytes:
    """
    Φ-RFT transform + uniform quantization + entropy coding.
    
    This is a LOSSY method unless bits_per_coeff is very high.
    For fair comparison, we track distortion.
    """
    # Convert to signal
    signal = np.frombuffer(data, dtype=np.uint8).astype(np.float64)
    n = len(signal)
    
    # Pad to power of 2
    n_padded = 2 ** int(np.ceil(np.log2(max(n, 64))))
    signal_padded = np.zeros(n_padded)
    signal_padded[:n] = signal
    
    # Transform
    coeffs = phi_rft_forward(signal_padded)
    
    # Quantize magnitude and phase separately
    mag = np.abs(coeffs)
    phase = np.angle(coeffs)
    
    # Normalize and quantize
    max_mag = np.max(mag) + 1e-10
    mag_quantized = np.round(mag / max_mag * (2**bits_per_coeff - 1)).astype(np.uint16)
    phase_quantized = np.round((phase + np.pi) / (2 * np.pi) * (2**bits_per_coeff - 1)).astype(np.uint16)
    
    # Pack and compress with zlib
    header = struct.pack('<IIf', n, n_padded, max_mag)
    coeff_data = mag_quantized.tobytes() + phase_quantized.tobytes()
    compressed = header + zlib.compress(coeff_data, 9)
    
    return compressed


def compress_rft_lossless(data: bytes) -> bytes:
    """
    Φ-RFT lossless compression attempt.
    
    Strategy: Transform, reorder by magnitude, delta-encode, entropy code.
    
    This is experimental and may not beat zlib.
    """
    signal = np.frombuffer(data, dtype=np.uint8).astype(np.float64)
    n = len(signal)
    
    # Pad to power of 2
    n_padded = 2 ** int(np.ceil(np.log2(max(n, 64))))
    signal_padded = np.zeros(n_padded)
    signal_padded[:n] = signal
    
    # Transform
    coeffs = phi_rft_forward(signal_padded)
    
    # For lossless: store full precision
    # This will likely be LARGER than original!
    real_part = coeffs.real
    imag_part = coeffs.imag
    
    # Convert to bytes and compress
    header = struct.pack('<II', n, n_padded)
    coeff_bytes = real_part.tobytes() + imag_part.tobytes()
    compressed = header + zlib.compress(coeff_bytes, 9)
    
    return compressed


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CompressionResult:
    method: str
    compressed_size: int
    original_size: int
    rate_bps: float  # bits per symbol
    is_lossless: bool
    distortion: Optional[float] = None


def analyze_source(source: SourceModel, verbose: bool = True) -> Dict:
    """
    Comprehensive entropy and compression analysis of a source.
    """
    data = source.data
    n = len(data)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"SOURCE: {source.name}")
        print(f"{'='*70}")
        print(f"Description: {source.description}")
        print(f"Size: {n} bytes")
    
    # Entropy estimation
    h0 = empirical_entropy_h0(data)
    h1 = empirical_entropy_h1(data)
    h_rate, h_sequence = entropy_rate_estimate(data, max_order=6)
    
    if verbose:
        print(f"\nENTROPY ESTIMATES:")
        print(f"  H₀ (zero-order):    {h0:.4f} bits/symbol")
        print(f"  H₁ (first-order):   {h1:.4f} bits/symbol")
        print(f"  H_rate (converged): {h_rate:.4f} bits/symbol")
        if source.true_entropy is not None:
            print(f"  H_true (known):     {source.true_entropy:.4f} bits/symbol")
    
    # Theoretical minimum
    h_best = source.true_entropy if source.true_entropy is not None else h_rate
    min_bytes = h_best * n / 8
    
    if verbose:
        print(f"\nTHEORETICAL LIMITS:")
        print(f"  Min achievable size: {min_bytes:.1f} bytes ({h_best:.4f} bps)")
    
    # Compression tests
    results = []
    
    # zlib (baseline - close to optimal for many sources)
    start = time.time()
    zlib_compressed = compress_zlib(data)
    zlib_time = time.time() - start
    zlib_rate = len(zlib_compressed) * 8 / n
    results.append(CompressionResult(
        method="zlib-9",
        compressed_size=len(zlib_compressed),
        original_size=n,
        rate_bps=zlib_rate,
        is_lossless=True
    ))
    
    # brotli
    if BROTLI_AVAILABLE:
        start = time.time()
        brotli_compressed = compress_brotli(data)
        brotli_time = time.time() - start
        brotli_rate = len(brotli_compressed) * 8 / n
        results.append(CompressionResult(
            method="brotli-11",
            compressed_size=len(brotli_compressed),
            original_size=n,
            rate_bps=brotli_rate,
            is_lossless=True
        ))
    
    # zstd
    if ZSTD_AVAILABLE:
        start = time.time()
        zstd_compressed = compress_zstd(data)
        zstd_time = time.time() - start
        zstd_rate = len(zstd_compressed) * 8 / n
        results.append(CompressionResult(
            method="zstd-22",
            compressed_size=len(zstd_compressed),
            original_size=n,
            rate_bps=zstd_rate,
            is_lossless=True
        ))
    
    # RFT lossless attempt
    start = time.time()
    rft_lossless = compress_rft_lossless(data)
    rft_lossless_time = time.time() - start
    rft_lossless_rate = len(rft_lossless) * 8 / n
    results.append(CompressionResult(
        method="rft-lossless",
        compressed_size=len(rft_lossless),
        original_size=n,
        rate_bps=rft_lossless_rate,
        is_lossless=True
    ))
    
    # RFT quantized (lossy)
    for bits in [8, 12, 16]:
        start = time.time()
        rft_quant = compress_rft_quantized(data, bits_per_coeff=bits)
        rft_quant_time = time.time() - start
        rft_quant_rate = len(rft_quant) * 8 / n
        results.append(CompressionResult(
            method=f"rft-quant-{bits}bit",
            compressed_size=len(rft_quant),
            original_size=n,
            rate_bps=rft_quant_rate,
            is_lossless=False
        ))
    
    if verbose:
        print(f"\nCOMPRESSION RESULTS:")
        print(f"  {'Method':<20} {'Size':>10} {'Rate (bps)':>12} {'Gap to H':>12} {'Lossless':>10}")
        print(f"  {'-'*66}")
        
        for r in results:
            gap = r.rate_bps - h_best
            gap_str = f"+{gap:.4f}" if gap > 0 else f"{gap:.4f}"
            lossless_str = "✓" if r.is_lossless else "✗"
            print(f"  {r.method:<20} {r.compressed_size:>10} {r.rate_bps:>12.4f} {gap_str:>12} {lossless_str:>10}")
        
        # Best lossless
        lossless_results = [r for r in results if r.is_lossless]
        if lossless_results:
            best_lossless = min(lossless_results, key=lambda r: r.rate_bps)
            print(f"\n  BEST LOSSLESS: {best_lossless.method} at {best_lossless.rate_bps:.4f} bps")
            print(f"  Gap to entropy: {best_lossless.rate_bps - h_best:+.4f} bps")
            
            if best_lossless.rate_bps < h_best:
                print(f"  ⚠️ WARNING: Rate < estimated entropy! Check entropy estimation.")
    
    return {
        "source": source.name,
        "description": source.description,
        "size": n,
        "entropy": {
            "H0": h0,
            "H1": h1,
            "H_rate": h_rate,
            "H_sequence": h_sequence,
            "H_true": source.true_entropy
        },
        "compression": [asdict(r) for r in results],
        "best_lossless_rate": min(r.rate_bps for r in results if r.is_lossless) if any(r.is_lossless for r in results) else None
    }


def main():
    """Run comprehensive entropy and rate analysis."""
    
    print("=" * 70)
    print("RIGOROUS ENTROPY AND RATE ANALYSIS FOR Φ-RFT")
    print("=" * 70)
    
    n = 10000  # Sample size
    
    # Define source models
    sources = [
        # Synthetic sources with known entropy
        generate_iid_source(n, alphabet_size=256),
        generate_iid_source(n, alphabet_size=16),
        generate_biased_source(n, p=0.9),
        generate_biased_source(n, p=0.99),
        
        # Markov sources
        generate_markov_source(n, np.array([
            [0.9, 0.1],
            [0.3, 0.7]
        ])),
        generate_markov_source(n, np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.1, 0.7]
        ])),
        
        # Structured sources
        generate_golden_source(n),
        
        # Real-world sources
        generate_ascii_source(n, "code"),
        generate_ascii_source(n, "english"),
    ]
    
    all_results = []
    
    for source in sources:
        result = analyze_source(source)
        all_results.append(result)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: COMPRESSION GAP TO ENTROPY")
    print("=" * 70)
    
    print(f"\n{'Source':<25} {'H_est':>8} {'Best Rate':>10} {'Gap':>8} {'Method':<15}")
    print("-" * 70)
    
    for result in all_results:
        h_est = result['entropy']['H_rate']
        best_rate = result['best_lossless_rate']
        if best_rate:
            gap = best_rate - h_est
            best_method = min(
                [r for r in result['compression'] if r['is_lossless']],
                key=lambda r: r['rate_bps']
            )['method']
            print(f"{result['source']:<25} {h_est:>8.4f} {best_rate:>10.4f} {gap:>+8.4f} {best_method:<15}")
    
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    
    # Check if RFT ever wins
    rft_wins = []
    for result in all_results:
        lossless = [r for r in result['compression'] if r['is_lossless']]
        if lossless:
            best = min(lossless, key=lambda r: r['rate_bps'])
            if 'rft' in best['method']:
                rft_wins.append(result['source'])
    
    if rft_wins:
        print(f"\n✅ Φ-RFT wins on: {', '.join(rft_wins)}")
    else:
        print(f"\n❌ Φ-RFT does NOT beat standard compressors on any tested source.")
        print("   This is expected: RFT is a TRANSFORM, not a compression algorithm.")
        print("   To beat entropy coders, we need:")
        print("   1. Sources where RFT provides better energy compaction than DCT")
        print("   2. Followed by optimal quantization + entropy coding")
        print("   3. Or: RFT as preprocessing for learned compression")
    
    # Save results
    output_path = "experiments/entropy_rate_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sample_size": n,
            "results": all_results
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
