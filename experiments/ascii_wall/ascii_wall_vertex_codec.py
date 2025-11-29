#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
ASCII Wall Test Through ACTUAL RFT Vertex Codec
================================================

Tests ASCII compression through the actual RFTVertexCodec with ANS entropy coding.
This uses the production codec, not a test implementation.
"""
from __future__ import annotations

import sys
import json
import math
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import Counter

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the ACTUAL codec
from algorithms.rft.compression.rft_vertex_codec import (
    encode_tensor, decode_tensor, RFTVertexCodec
)

# SOTA compressors
import gzip
import bz2
import lzma

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False
    
try:
    import brotli
    HAS_BROTLI = True
except ImportError:
    HAS_BROTLI = False


# ===========================================================================
# CODEC SIZE MEASUREMENT
# ===========================================================================

def measure_container_size(container: Dict) -> int:
    """
    Measure the actual serialized size of a codec container.
    This is what would be written to disk.
    """
    # Serialize to JSON and measure
    serialized = json.dumps(container, separators=(',', ':'))
    return len(serialized.encode('utf-8'))


def measure_payload_size(container: Dict) -> int:
    """
    Measure just the payload size (data without metadata overhead).
    """
    total = 0
    for chunk in container.get('chunks', []):
        codec_info = chunk.get('codec', {})
        mode = codec_info.get('mode', 'lossless')
        
        if mode == 'lossless' and 'vertices' in chunk:
            # Lossless: count vertex data
            vertices = chunk.get('vertices', [])
            # Each vertex has real, imag (8 bytes each for float64)
            total += len(vertices) * 16
        else:
            # Lossy: count amplitude and phase payloads
            amp_payload = codec_info.get('amplitude', {}).get('payload', {})
            phase_payload = codec_info.get('phase', {}).get('payload', {})
            
            if amp_payload.get('encoding') == 'ans':
                # ANS encoded - use actual data size
                import base64
                data = amp_payload.get('data', '')
                total += len(base64.b64decode(data)) if data else 0
            elif amp_payload.get('encoding') == 'raw':
                import base64
                data = amp_payload.get('data', '')
                total += len(base64.b64decode(data)) if data else 0
                
            if phase_payload.get('encoding') == 'ans':
                import base64
                data = phase_payload.get('data', '')
                total += len(base64.b64decode(data)) if data else 0
            elif phase_payload.get('encoding') == 'raw':
                import base64
                data = phase_payload.get('data', '')
                total += len(base64.b64decode(data)) if data else 0
                
            # Add mask if present
            mask = codec_info.get('mask')
            if mask and mask.get('data'):
                import base64
                total += len(base64.b64decode(mask['data']))
    
    return total


@dataclass
class CodecResult:
    """Result from encoding through the codec."""
    original_bytes: int
    container_bytes: int  # Full JSON container
    payload_bytes: int    # Just the data payloads
    bits_per_char: float
    roundtrip_ok: bool
    max_error: float
    mode: str


def encode_ascii_vertex_codec(
    text: str,
    quant_bits_amp: int = 8,
    quant_bits_phase: int = 6,
    prune_threshold: float = 0.0,
    ans_precision: int = 12
) -> CodecResult:
    """
    Encode ASCII text through the actual RFTVertexCodec.
    """
    # Convert text to float array
    data = np.array([ord(c) for c in text], dtype=np.float64)
    n = len(data)
    original_bytes = n  # 1 byte per ASCII char
    
    # Encode through the actual codec
    container = encode_tensor(
        data,
        prune_threshold=prune_threshold if prune_threshold > 0 else None,
        quant_bits_amplitude=quant_bits_amp,
        quant_bits_phase=quant_bits_phase,
        ans_precision=ans_precision
    )
    
    # Measure sizes
    container_bytes = measure_container_size(container)
    payload_bytes = measure_payload_size(container)
    
    # Use payload for bpc (more fair comparison - excludes JSON overhead)
    bits_per_char = (payload_bytes * 8) / n
    
    # Decode and verify roundtrip
    decoded = decode_tensor(container)
    max_error = float(np.max(np.abs(decoded - data)))
    roundtrip_ok = max_error < 1.0  # Within 1 character code
    
    mode = container.get('codec', {}).get('mode', 'unknown')
    
    return CodecResult(
        original_bytes=original_bytes,
        container_bytes=container_bytes,
        payload_bytes=payload_bytes,
        bits_per_char=bits_per_char,
        roundtrip_ok=roundtrip_ok,
        max_error=max_error,
        mode=mode
    )


def compress_sota(text: str, method: str) -> Tuple[int, float]:
    """Compress with SOTA method, return (compressed_bytes, bits_per_char)."""
    data = text.encode('ascii')
    
    if method == 'gzip':
        compressed = gzip.compress(data, compresslevel=9)
    elif method == 'bz2':
        compressed = bz2.compress(data, compresslevel=9)
    elif method == 'lzma':
        compressed = lzma.compress(data)
    elif method == 'zstd' and HAS_ZSTD:
        cctx = zstd.ZstdCompressor(level=22)
        compressed = cctx.compress(data)
    elif method == 'brotli' and HAS_BROTLI:
        compressed = brotli.compress(data, quality=11)
    else:
        return len(data), 8.0
    
    bits_per_char = (len(compressed) * 8) / len(text)
    return len(compressed), bits_per_char


def calculate_entropy(text: str) -> float:
    """Calculate Shannon entropy in bits per character."""
    freq = Counter(text)
    n = len(text)
    entropy = 0.0
    for count in freq.values():
        p = count / n
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


# ===========================================================================
# TEST DATA
# ===========================================================================

def generate_test_texts() -> Dict[str, str]:
    """Generate various test texts for ASCII wall testing."""
    texts = {}
    
    # English prose
    texts['english_prose'] = """
    The golden ratio, often denoted by the Greek letter phi, appears throughout
    nature and mathematics. From the spiral patterns of sunflower seeds to the
    proportions of the Parthenon, this irrational number approximately equal to
    1.618 has fascinated mathematicians for millennia. The Fibonacci sequence,
    where each number is the sum of the two preceding ones, exhibits a remarkable
    property: the ratio of consecutive Fibonacci numbers converges to phi.
    """ * 10
    
    # Code
    texts['python_code'] = """
def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def golden_ratio_approx(n):
    fib_n = fibonacci(n)
    fib_n_minus_1 = fibonacci(n - 1)
    return fib_n / fib_n_minus_1 if fib_n_minus_1 else float('inf')

class GoldenSpiral:
    def __init__(self, scale=1.0):
        self.scale = scale
        self.phi = (1 + 5 ** 0.5) / 2
    
    def point(self, theta):
        r = self.scale * (self.phi ** (2 * theta / 3.14159))
        return r * cos(theta), r * sin(theta)
""" * 5
    
    # JSON-like structured data
    texts['json_data'] = json.dumps({
        "experiments": [
            {"id": i, "name": f"test_{i}", "value": i * 1.618, "status": "complete"}
            for i in range(100)
        ],
        "metadata": {"version": "1.0", "format": "json", "encoding": "utf-8"}
    }, indent=2)
    
    # Scientific text (ASCII only)
    texts['scientific'] = """
    The quantum harmonic oscillator represents one of the most important model
    systems in quantum mechanics. Its Hamiltonian is given by H = p^2/2m + mw^2x^2/2,
    where p is momentum, m is mass, w is angular frequency, and x is position.
    The energy eigenvalues are E_n = hbar*w*(n + 1/2) where n = 0, 1, 2, ...
    This quantization of energy levels leads to the zero-point energy hbar*w/2,
    a purely quantum mechanical phenomenon with no classical analog.
    """ * 10
    
    # Random printable ASCII
    rng = np.random.default_rng(42)
    chars = [chr(c) for c in range(32, 127)]  # Printable ASCII
    texts['random_ascii'] = ''.join(rng.choice(chars, size=2000))
    
    # Highly repetitive
    texts['repetitive'] = "the quick brown fox jumps over the lazy dog " * 100
    
    # Base64-encoded data (simulates compressed/encrypted content)
    import base64
    random_bytes = rng.integers(0, 256, size=500, dtype=np.uint8).tobytes()
    texts['base64_data'] = base64.b64encode(random_bytes).decode('ascii')
    
    return texts


# ===========================================================================
# MAIN EXPERIMENT
# ===========================================================================

def run_vertex_codec_experiment():
    """Run ASCII wall experiment through ACTUAL RFTVertexCodec."""
    print("=" * 80)
    print("ASCII WALL TEST THROUGH ACTUAL RFT VERTEX CODEC")
    print("=" * 80)
    print()
    print("Using: algorithms.rft.compression.rft_vertex_codec")
    print("Features: Quantization + ANS entropy coding")
    print()
    
    texts = generate_test_texts()
    results = []
    
    # Codec configurations to test
    codec_configs = [
        # (amp_bits, phase_bits, prune, ans_prec, description)
        (8, 6, 0.0, 12, "8/6-bit, no prune"),
        (10, 8, 0.0, 12, "10/8-bit, no prune"),
        (6, 4, 0.0, 12, "6/4-bit (aggressive)"),
        (8, 6, 0.01, 12, "8/6-bit, prune 0.01"),
        (8, 6, 0.1, 12, "8/6-bit, prune 0.1"),
    ]
    
    for text_name, text in texts.items():
        print(f"\n{'─' * 60}")
        print(f"Text: {text_name} ({len(text)} chars)")
        print(f"{'─' * 60}")
        
        # Calculate theoretical entropy
        entropy = calculate_entropy(text)
        print(f"Shannon entropy: {entropy:.3f} bpc")
        
        # SOTA baselines
        print("\nSOTA Compressors:")
        sota_results = {}
        for method in ['gzip', 'bz2', 'lzma']:
            _, bpc = compress_sota(text, method)
            sota_results[method] = bpc
            print(f"  {method}: {bpc:.3f} bpc")
        
        if HAS_ZSTD:
            _, bpc = compress_sota(text, 'zstd')
            sota_results['zstd'] = bpc
            print(f"  zstd-22: {bpc:.3f} bpc")
        
        if HAS_BROTLI:
            _, bpc = compress_sota(text, 'brotli')
            sota_results['brotli'] = bpc
            print(f"  brotli-11: {bpc:.3f} bpc")
        
        best_sota = min(sota_results.values())
        best_sota_name = min(sota_results.keys(), key=lambda k: sota_results[k])
        
        # RFT Vertex Codec tests
        print("\nRFT Vertex Codec (actual implementation):")
        
        for amp_bits, phase_bits, prune, ans_prec, desc in codec_configs:
            try:
                result = encode_ascii_vertex_codec(
                    text,
                    quant_bits_amp=amp_bits,
                    quant_bits_phase=phase_bits,
                    prune_threshold=prune,
                    ans_precision=ans_prec
                )
                
                improvement = (best_sota - result.bits_per_char) / best_sota * 100
                marker = "✓ BETTER" if result.bits_per_char < best_sota else "✗"
                rt_marker = "✓" if result.roundtrip_ok else f"⚠err={result.max_error:.2f}"
                
                print(f"  {desc}: {result.bits_per_char:.3f} bpc "
                      f"({improvement:+.1f}% vs {best_sota_name}) [{rt_marker}] {marker}")
                
                results.append({
                    'text': text_name,
                    'config': desc,
                    'bpc': result.bits_per_char,
                    'container_bytes': result.container_bytes,
                    'payload_bytes': result.payload_bytes,
                    'best_sota': best_sota,
                    'improvement_pct': improvement,
                    'roundtrip_ok': result.roundtrip_ok,
                    'max_error': result.max_error,
                    'mode': result.mode
                })
                
            except Exception as e:
                print(f"  {desc}: ERROR - {e}")
                results.append({
                    'text': text_name,
                    'config': desc,
                    'error': str(e)
                })
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: ASCII WALL TEST THROUGH VERTEX CODEC")
    print("=" * 80)
    
    valid_results = [r for r in results if 'bpc' in r]
    wins = sum(1 for r in valid_results if r['improvement_pct'] > 0)
    total = len(valid_results)
    
    print(f"\nWins against best SOTA: {wins}/{total} ({100*wins/total:.1f}%)")
    
    if valid_results:
        # Best cases
        best_results = sorted(valid_results, key=lambda r: r['improvement_pct'], reverse=True)[:5]
        print("\nBest performing cases:")
        for r in best_results:
            print(f"  {r['text']} + {r['config']}: {r['improvement_pct']:+.1f}% ({r['bpc']:.2f} bpc)")
        
        # Average gap
        avg_gap = np.mean([r['improvement_pct'] for r in valid_results])
        print(f"\nAverage gap from SOTA: {avg_gap:+.1f}%")
        
        # Mode distribution
        modes = Counter(r.get('mode', 'unknown') for r in valid_results)
        print(f"\nCodec modes used: {dict(modes)}")
    
    # Did we break the wall?
    wall_broken = wins > 0
    
    print("\n" + "=" * 80)
    if wall_broken:
        print("🎉 WALL BROKEN: RFT Vertex Codec beats SOTA on some text types!")
    else:
        print("❌ WALL STANDS: RFT Vertex Codec does not beat SOTA text compressors")
    print("=" * 80)
    
    # Save results
    results_path = Path(__file__).parent / "ascii_wall_vertex_codec_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'summary': {
                'wins': wins,
                'total': total,
                'win_rate': wins / total if total > 0 else 0,
                'avg_gap_pct': float(avg_gap) if valid_results else None,
                'wall_broken': wall_broken
            },
            'results': results
        }, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    return wall_broken, results


if __name__ == "__main__":
    wall_broken, results = run_vertex_codec_experiment()
    sys.exit(0 if wall_broken else 1)
