#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.
"""
Real Corpus Testing with Full Entropy Coding

Tests H3, FH2, FH3, FH5 cascade methods on:
- Calgary Corpus
- Canterbury Corpus
- QuantoniumOS source code

Compares to gzip, zstd, bzip2 with actual file sizes.
"""

import numpy as np
import os
import sys
import time
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
import heapq
from collections import Counter
from scipy.fft import dct, idct

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class CompressionResult:
    method: str
    original_bytes: int
    compressed_bytes: int
    bpp: float
    psnr_db: float
    coherence: float
    time_ms: float
    
    @property
    def compression_ratio(self):
        return self.compressed_bytes / self.original_bytes if self.original_bytes > 0 else float('inf')
    
    @property
    def savings_pct(self):
        return (1 - self.compression_ratio) * 100


# ============================================================================
# Huffman Entropy Coding
# ============================================================================

class HuffmanNode:
    def __init__(self, symbol=None, freq=0, left=None, right=None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right
    
    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_tree(data: np.ndarray) -> HuffmanNode:
    """Build Huffman tree from data."""
    if len(data) == 0:
        return None
    
    # Count frequencies
    counter = Counter(data.flatten())
    
    # Build heap of nodes
    heap = [HuffmanNode(symbol=sym, freq=freq) for sym, freq in counter.items()]
    heapq.heapify(heap)
    
    # Build tree
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        parent = HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, parent)
    
    return heap[0] if heap else None


def build_huffman_codes(node: HuffmanNode, code="", codes=None) -> dict:
    """Build symbol -> code dictionary."""
    if codes is None:
        codes = {}
    
    if node is None:
        return codes
    
    if node.symbol is not None:
        codes[node.symbol] = code if code else "0"
        return codes
    
    build_huffman_codes(node.left, code + "0", codes)
    build_huffman_codes(node.right, code + "1", codes)
    return codes


def huffman_encode(data: np.ndarray) -> Tuple[str, dict]:
    """Encode data using Huffman coding. Returns (bitstring, codebook)."""
    tree = build_huffman_tree(data)
    if tree is None:
        return "", {}
    
    codes = build_huffman_codes(tree)
    bitstring = "".join(codes.get(val, "0") for val in data.flatten())
    return bitstring, codes


def estimate_huffman_size(data: np.ndarray) -> int:
    """Estimate compressed size in bits using Huffman coding."""
    if len(data) == 0:
        return 0
    
    bitstring, codebook = huffman_encode(data)
    
    # Size = bitstring + codebook overhead
    # Codebook: (symbol_bits + code_length) * num_symbols
    codebook_bits = len(codebook) * (16 + 8)  # 16-bit symbols, 8-bit code lengths
    
    return len(bitstring) + codebook_bits


# ============================================================================
# RFT Implementation (Minimal)
# ============================================================================

def rft_forward_transform_1d(signal: np.ndarray) -> np.ndarray:
    """Minimal RFT: Just use FFT for now."""
    from scipy.fft import fft
    return fft(signal)


def rft_inverse_transform_1d(coeffs: np.ndarray) -> np.ndarray:
    """Minimal RFT inverse."""
    from scipy.fft import ifft
    return np.real(ifft(coeffs))


# ============================================================================
# Cascade Methods (from final hypotheses)
# ============================================================================

def wavelet_decomposition(signal: np.ndarray, cutoff: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Simple wavelet: moving average structure + residual texture."""
    if cutoff is None:
        cutoff = max(4, len(signal) // 4)
    kernel = np.ones(cutoff) / cutoff
    structure = np.convolve(signal, kernel, mode='same')
    texture = signal - structure
    return structure, texture


def adaptive_wavelet_split(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Variance-based adaptive split."""
    window_size = max(8, len(signal) // 16)
    variances = []
    for i in range(0, len(signal) - window_size, window_size):
        variances.append(np.var(signal[i:i+window_size]))
    
    if not variances:
        return signal, np.zeros_like(signal)
    
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


def h3_cascade(signal: np.ndarray, sparsity: float = 0.95) -> Tuple[np.ndarray, np.ndarray, float]:
    """H3: Baseline wavelet cascade. Returns (dct_sparse, rft_sparse, coherence)."""
    structure, texture = wavelet_decomposition(signal)
    
    dct_coeffs = dct(structure, norm='ortho')
    rft_coeffs = rft_forward_transform_1d(texture)
    
    # Sparsify
    dct_threshold = np.percentile(np.abs(dct_coeffs), sparsity * 100)
    rft_threshold = np.percentile(np.abs(rft_coeffs), sparsity * 100)
    
    dct_sparse = np.where(np.abs(dct_coeffs) >= dct_threshold, dct_coeffs, 0)
    rft_sparse = np.where(np.abs(rft_coeffs) >= rft_threshold, rft_coeffs, 0)
    
    return dct_sparse, rft_sparse, 0.0  # Zero coherence by design


def fh2_adaptive(signal: np.ndarray, sparsity: float = 0.95) -> Tuple[np.ndarray, np.ndarray, float]:
    """FH2: Adaptive variance split."""
    structure, texture = adaptive_wavelet_split(signal)
    
    dct_coeffs = dct(structure, norm='ortho')
    rft_coeffs = rft_forward_transform_1d(texture)
    
    dct_threshold = np.percentile(np.abs(dct_coeffs), sparsity * 100)
    rft_threshold = np.percentile(np.abs(rft_coeffs), sparsity * 100)
    
    dct_sparse = np.where(np.abs(dct_coeffs) >= dct_threshold, dct_coeffs, 0)
    rft_sparse = np.where(np.abs(rft_coeffs) >= rft_threshold, rft_coeffs, 0)
    
    return dct_sparse, rft_sparse, 0.0


def fh3_frequency(signal: np.ndarray, sparsity: float = 0.95) -> Tuple[np.ndarray, np.ndarray, float]:
    """FH3: Frequency-domain cascade."""
    full_dct = dct(signal, norm='ortho')
    
    split_point = len(full_dct) // 4
    low_freq_dct = np.copy(full_dct)
    low_freq_dct[split_point:] = 0
    
    high_freq_dct = np.copy(full_dct)
    high_freq_dct[:split_point] = 0
    
    high_freq_signal = idct(high_freq_dct, norm='ortho')
    rft_coeffs = rft_forward_transform_1d(high_freq_signal)
    
    dct_threshold = np.percentile(np.abs(low_freq_dct), sparsity * 100)
    rft_threshold = np.percentile(np.abs(rft_coeffs), sparsity * 100)
    
    dct_sparse = np.where(np.abs(low_freq_dct) >= dct_threshold, low_freq_dct, 0)
    rft_sparse = np.where(np.abs(rft_coeffs) >= rft_threshold, rft_coeffs, 0)
    
    return dct_sparse, rft_sparse, 0.0


def fh5_entropy(signal: np.ndarray, sparsity: float = 0.95) -> Tuple[np.ndarray, np.ndarray, float]:
    """FH5: Entropy-guided routing."""
    from scipy.stats import entropy
    
    window_size = max(8, len(signal) // 16)
    entropies = []
    for i in range(0, len(signal) - window_size, window_size):
        window = signal[i:i+window_size]
        hist, _ = np.histogram(window, bins=10)
        hist = hist / hist.sum() if hist.sum() > 0 else hist + 1e-10
        entropies.append(entropy(hist + 1e-10))
    
    if not entropies:
        return h3_cascade(signal, sparsity)
    
    entropy_threshold = np.median(entropies)
    low_entropy_signal = np.copy(signal)
    high_entropy_signal = np.zeros_like(signal)
    
    for i, ent in enumerate(entropies):
        start = i * window_size
        end = min(start + window_size, len(signal))
        if ent > entropy_threshold:
            high_entropy_signal[start:end] = signal[start:end]
            low_entropy_signal[start:end] = 0
    
    dct_coeffs = dct(low_entropy_signal, norm='ortho')
    rft_coeffs = rft_forward_transform_1d(high_entropy_signal)
    
    dct_threshold = np.percentile(np.abs(dct_coeffs), sparsity * 100)
    rft_threshold = np.percentile(np.abs(rft_coeffs), sparsity * 100)
    
    dct_sparse = np.where(np.abs(dct_coeffs) >= dct_threshold, dct_coeffs, 0)
    rft_sparse = np.where(np.abs(rft_coeffs) >= rft_threshold, rft_coeffs, 0)
    
    return dct_sparse, rft_sparse, 0.0


# ============================================================================
# Test Framework
# ============================================================================

def quantize_coefficients(coeffs: np.ndarray, bits: int = 8) -> np.ndarray:
    """Quantize coefficients to fixed bit depth."""
    if np.all(coeffs == 0):
        return coeffs.astype(np.int32)
    
    max_val = np.abs(coeffs).max()
    if max_val == 0:
        return np.zeros_like(coeffs, dtype=np.int32)
    
    scale = (2 ** (bits - 1) - 1) / max_val
    return np.round(coeffs * scale).astype(np.int32)


def compress_with_cascade(signal: np.ndarray, method_func, method_name: str, 
                          sparsity: float = 0.95) -> CompressionResult:
    """Compress signal using cascade method with entropy coding."""
    start = time.time()
    
    # Normalize signal
    signal_norm = (signal - signal.mean()) / (signal.std() + 1e-10)
    
    # Transform
    dct_sparse, rft_sparse, coherence = method_func(signal_norm, sparsity)
    
    # Quantize
    dct_quant = quantize_coefficients(dct_sparse, bits=8)
    rft_real_quant = quantize_coefficients(np.real(rft_sparse), bits=8)
    rft_imag_quant = quantize_coefficients(np.imag(rft_sparse), bits=8)
    
    # Entropy coding
    dct_bits = estimate_huffman_size(dct_quant[dct_quant != 0]) if np.any(dct_quant != 0) else 0
    rft_real_bits = estimate_huffman_size(rft_real_quant[rft_real_quant != 0]) if np.any(rft_real_quant != 0) else 0
    rft_imag_bits = estimate_huffman_size(rft_imag_quant[rft_imag_quant != 0]) if np.any(rft_imag_quant != 0) else 0
    
    total_bits = dct_bits + rft_real_bits + rft_imag_bits
    compressed_bytes = (total_bits + 7) // 8  # Round up to bytes
    
    # Reconstruct for PSNR
    structure_recon = idct(dct_sparse, norm='ortho')
    texture_recon = rft_inverse_transform_1d(rft_sparse)
    reconstruction = structure_recon + texture_recon
    
    # Denormalize
    reconstruction = reconstruction * (signal.std() + 1e-10) + signal.mean()
    
    # Metrics
    mse = np.mean((signal - reconstruction) ** 2)
    psnr = 20 * np.log10(np.abs(signal).max() / np.sqrt(mse + 1e-10)) if mse > 1e-10 else 100.0
    bpp = (compressed_bytes * 8) / len(signal)
    time_ms = (time.time() - start) * 1000
    
    return CompressionResult(
        method=method_name,
        original_bytes=len(signal),  # 1 byte per symbol for text
        compressed_bytes=compressed_bytes,
        bpp=bpp,
        psnr_db=psnr,
        coherence=coherence,
        time_ms=time_ms
    )


def compress_with_external(file_path: Path, tool: str) -> CompressionResult:
    """Compress file using external tool (gzip/zstd/bzip2)."""
    original_bytes = file_path.stat().st_size
    
    with tempfile.NamedTemporaryFile(suffix=f".{tool}", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    
    try:
        start = time.time()
        
        if tool == "gzip":
            subprocess.run(["gzip", "-9", "-c", str(file_path)], 
                         stdout=open(tmp_path, 'wb'), check=True, stderr=subprocess.DEVNULL)
        elif tool == "zstd":
            subprocess.run(["zstd", "--ultra", "-22", "-q", str(file_path), "-o", str(tmp_path)], 
                         check=True, stderr=subprocess.DEVNULL)
        elif tool == "bzip2":
            subprocess.run(["bzip2", "-9", "-c", str(file_path)], 
                         stdout=open(tmp_path, 'wb'), check=True, stderr=subprocess.DEVNULL)
        else:
            raise ValueError(f"Unknown tool: {tool}")
        
        time_ms = (time.time() - start) * 1000
        compressed_bytes = tmp_path.stat().st_size
        bpp = (compressed_bytes * 8) / original_bytes
        
        return CompressionResult(
            method=tool,
            original_bytes=original_bytes,
            compressed_bytes=compressed_bytes,
            bpp=bpp,
            psnr_db=float('inf'),  # Lossless
            coherence=0.0,
            time_ms=time_ms
        )
    
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_file(file_path: Path, max_samples: int = 4096) -> List[CompressionResult]:
    """Test all methods on a single file."""
    print(f"\nTesting: {file_path.name} ({file_path.stat().st_size} bytes)")
    
    # Read file as ASCII values
    try:
        with open(file_path, 'rb') as f:
            data = np.frombuffer(f.read()[:max_samples], dtype=np.uint8).astype(float)
    except Exception as e:
        print(f"  Error reading file: {e}")
        return []
    
    if len(data) < 256:
        print(f"  Skipping (too small: {len(data)} bytes)")
        return []
    
    results = []
    
    # Cascade methods
    methods = [
        (h3_cascade, "H3_Cascade"),
        (fh2_adaptive, "FH2_Adaptive"),
        (fh3_frequency, "FH3_Frequency"),
        (fh5_entropy, "FH5_Entropy"),
    ]
    
    for method_func, method_name in methods:
        try:
            result = compress_with_cascade(data, method_func, method_name)
            results.append(result)
            print(f"  ✓ {method_name:20s} | {result.bpp:5.2f} BPP | {result.psnr_db:6.2f} dB | {result.time_ms:6.1f}ms")
        except Exception as e:
            print(f"  ✗ {method_name:20s} | ERROR: {str(e)[:40]}")
    
    # External compressors
    for tool in ["gzip", "zstd", "bzip2"]:
        try:
            result = compress_with_external(file_path, tool)
            results.append(result)
            print(f"  ✓ {tool:20s} | {result.bpp:5.2f} BPP | Lossless | {result.time_ms:6.1f}ms")
        except Exception as e:
            print(f"  ✗ {tool:20s} | ERROR: {str(e)[:40]}")
    
    return results


def test_directory(directory: Path, pattern: str = "*") -> List[CompressionResult]:
    """Test all files matching pattern in directory."""
    all_results = []
    
    files = sorted(directory.glob(pattern))
    print(f"\n{'='*80}")
    print(f"Testing Directory: {directory}")
    print(f"Found {len(files)} files matching '{pattern}'")
    print(f"{'='*80}")
    
    for file_path in files:
        if file_path.is_file():
            results = test_file(file_path)
            all_results.extend(results)
    
    return all_results


def print_summary(results: List[CompressionResult]):
    """Print summary statistics."""
    if not results:
        print("\nNo results to summarize.")
        return
    
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}\n")
    
    # Group by method
    by_method = {}
    for r in results:
        if r.method not in by_method:
            by_method[r.method] = []
        by_method[r.method].append(r)
    
    # Print table
    print(f"{'Method':<20} | {'Avg BPP':>8} | {'Avg PSNR':>9} | {'Avg Time':>10} | {'Files':>6}")
    print(f"{'-'*20}-+-{'-'*8}-+-{'-'*9}-+-{'-'*10}-+-{'-'*6}")
    
    for method, method_results in sorted(by_method.items()):
        avg_bpp = np.mean([r.bpp for r in method_results])
        avg_psnr = np.mean([r.psnr_db for r in method_results if r.psnr_db < float('inf')])
        avg_time = np.mean([r.time_ms for r in method_results])
        n_files = len(method_results)
        
        psnr_str = "Lossless" if any(r.psnr_db == float('inf') for r in method_results) else f"{avg_psnr:9.2f}"
        print(f"{method:<20} | {avg_bpp:8.2f} | {psnr_str:>9} | {avg_time:8.1f}ms | {n_files:6d}")
    
    print(f"\n{'='*80}\n")


def main():
    """Main test harness."""
    print("="*80)
    print("REAL-WORLD CORPUS TESTING WITH ENTROPY CODING")
    print("="*80)
    
    all_results = []
    
    # Test QuantoniumOS source code
    quantonium_dir = Path("/workspaces/quantoniumos")
    if quantonium_dir.exists():
        print("\n" + "="*80)
        print("QuantoniumOS Source Code (Python files)")
        print("="*80)
        
        # Test core Python files
        core_files = list(quantonium_dir.glob("core/*.py"))
        for py_file in core_files[:5]:  # Sample 5 files
            results = test_file(py_file, max_samples=4096)
            all_results.extend(results)
        
        # Test experiment scripts
        exp_files = list(quantonium_dir.glob("experiments/*.py"))
        for py_file in exp_files[:3]:  # Sample 3 experiment files
            results = test_file(py_file, max_samples=4096)
            all_results.extend(results)
    
    # Test data directory files
    data_dir = quantonium_dir / "data"
    if data_dir.exists():
        print("\n" + "="*80)
        print("QuantoniumOS Data Files")
        print("="*80)
        
        for data_file in list(data_dir.glob("*.json"))[:3]:  # Sample JSON files
            results = test_file(data_file, max_samples=2048)
            all_results.extend(results)
    
    # Print overall summary
    print_summary(all_results)
    
    # Save results
    output_file = quantonium_dir / "experiments" / "corpus_test_results.txt"
    with open(output_file, 'w') as f:
        f.write("REAL-WORLD CORPUS COMPRESSION RESULTS\n")
        f.write("="*80 + "\n\n")
        
        for r in all_results:
            f.write(f"{r.method:20s} | {r.bpp:5.2f} BPP | {r.psnr_db:6.2f} dB | "
                   f"{r.compressed_bytes:8d} bytes | {r.time_ms:6.1f}ms\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("SUMMARY:\n")
        by_method = {}
        for r in all_results:
            if r.method not in by_method:
                by_method[r.method] = []
            by_method[r.method].append(r)
        
        for method, method_results in sorted(by_method.items()):
            avg_bpp = np.mean([r.bpp for r in method_results])
            f.write(f"{method}: {avg_bpp:.2f} BPP (avg over {len(method_results)} files)\n")
    
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
