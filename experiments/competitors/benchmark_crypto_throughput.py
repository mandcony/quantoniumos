#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
Crypto Throughput Benchmark: RFT Cipher vs AES-GCM/ChaCha20
===========================================================

Compares RFT-based cipher against industry-standard authenticated encryption.

**DISCLAIMER**: This is NOT a security comparison. The RFT cipher is a toy/research
cipher and should NOT be used for real security. This benchmark only measures:
- Throughput (MB/s)
- Avalanche effect (bit diffusion)
- Round-trip correctness

Usage:
    python benchmark_crypto_throughput.py [--sizes 1024,16384,65536]
"""

import argparse
import csv
import json
import os
import platform
import secrets
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Tuple, Any

import numpy as np

# Add project root
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

# Standard crypto imports
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False
    print("Warning: cryptography library not available. Install with: pip install cryptography")

# RFT cipher imports
try:
    from algorithms.rft.crypto.enhanced_cipher import EnhancedRFTCryptoV2
    HAS_RFT_CIPHER = True
except ImportError:
    HAS_RFT_CIPHER = False
    print("Warning: RFT cipher not available")


@dataclass
class CryptoResult:
    """Result of a single crypto benchmark."""
    cipher: str
    data_size: int
    encrypt_time_us: float
    decrypt_time_us: float
    encrypt_mbps: float
    decrypt_mbps: float
    roundtrip_ok: bool
    avalanche_pct: float  # Percentage of bits that flip on 1-bit input change
    error_msg: str = ""


class CipherWrapper:
    """Base class for cipher wrappers."""
    
    name: str = "base"
    
    def encrypt(self, plaintext: bytes, key: bytes, nonce: bytes, aad: bytes = b"") -> bytes:
        raise NotImplementedError
    
    def decrypt(self, ciphertext: bytes, key: bytes, nonce: bytes, aad: bytes = b"") -> bytes:
        raise NotImplementedError
    
    def key_size(self) -> int:
        return 32
    
    def nonce_size(self) -> int:
        return 12


class AESGCMCipher(CipherWrapper):
    name = "aes_gcm"
    
    def encrypt(self, plaintext: bytes, key: bytes, nonce: bytes, aad: bytes = b"") -> bytes:
        cipher = AESGCM(key)
        return cipher.encrypt(nonce, plaintext, aad if aad else None)
    
    def decrypt(self, ciphertext: bytes, key: bytes, nonce: bytes, aad: bytes = b"") -> bytes:
        cipher = AESGCM(key)
        return cipher.decrypt(nonce, ciphertext, aad if aad else None)


class ChaCha20Cipher(CipherWrapper):
    name = "chacha20"
    
    def encrypt(self, plaintext: bytes, key: bytes, nonce: bytes, aad: bytes = b"") -> bytes:
        cipher = ChaCha20Poly1305(key)
        return cipher.encrypt(nonce, plaintext, aad if aad else None)
    
    def decrypt(self, ciphertext: bytes, key: bytes, nonce: bytes, aad: bytes = b"") -> bytes:
        cipher = ChaCha20Poly1305(key)
        return cipher.decrypt(nonce, ciphertext, aad if aad else None)


class RFTCipher(CipherWrapper):
    name = "rft_cipher"
    
    def encrypt(self, plaintext: bytes, key: bytes, nonce: bytes, aad: bytes = b"") -> bytes:
        cipher = EnhancedRFTCryptoV2(key)
        return cipher.encrypt_aead(plaintext, aad if aad else b"rft")
    
    def decrypt(self, ciphertext: bytes, key: bytes, nonce: bytes, aad: bytes = b"") -> bytes:
        cipher = EnhancedRFTCryptoV2(key)
        return cipher.decrypt_aead(ciphertext, aad if aad else b"rft")
    
    def nonce_size(self) -> int:
        return 0  # RFT cipher handles nonce internally


def compute_avalanche(cipher: CipherWrapper, key: bytes, nonce: bytes, data_size: int = 1024) -> float:
    """
    Compute avalanche effect: percentage of output bits that flip
    when a single input bit is flipped.
    
    Good diffusion should be close to 50%.
    """
    # Generate test data
    plaintext = secrets.token_bytes(data_size)
    
    # Encrypt original
    ct1 = cipher.encrypt(plaintext, key, nonce)
    
    # Flip one bit in plaintext
    pt_array = bytearray(plaintext)
    pt_array[len(pt_array) // 2] ^= 0x01  # Flip middle byte's LSB
    modified_pt = bytes(pt_array)
    
    # Encrypt modified (use different nonce for standard ciphers)
    try:
        ct2 = cipher.encrypt(modified_pt, key, nonce)
    except Exception:
        # Some ciphers may fail with same nonce
        new_nonce = secrets.token_bytes(cipher.nonce_size()) if cipher.nonce_size() > 0 else nonce
        ct2 = cipher.encrypt(modified_pt, key, new_nonce)
    
    # Compare ciphertexts (use min length)
    min_len = min(len(ct1), len(ct2))
    ct1_bits = np.unpackbits(np.frombuffer(ct1[:min_len], dtype=np.uint8))
    ct2_bits = np.unpackbits(np.frombuffer(ct2[:min_len], dtype=np.uint8))
    
    # Count differing bits
    diff_bits = np.sum(ct1_bits != ct2_bits)
    total_bits = len(ct1_bits)
    
    return 100.0 * diff_bits / total_bits if total_bits > 0 else 0.0


class CryptoBenchmark:
    """Benchmark harness for crypto comparison."""
    
    def __init__(self, warmup_runs: int = 3, timed_runs: int = 10):
        self.warmup_runs = warmup_runs
        self.timed_runs = timed_runs
        self.ciphers = self._register_ciphers()
    
    def _register_ciphers(self) -> List[CipherWrapper]:
        """Register all ciphers to benchmark."""
        ciphers = []
        
        if HAS_CRYPTO:
            ciphers.append(AESGCMCipher())
            ciphers.append(ChaCha20Cipher())
        
        if HAS_RFT_CIPHER:
            ciphers.append(RFTCipher())
        
        return ciphers
    
    def bench_single(
        self,
        cipher: CipherWrapper,
        data_size: int
    ) -> CryptoResult:
        """Benchmark a single cipher at a single size."""
        key = secrets.token_bytes(cipher.key_size())
        nonce = secrets.token_bytes(cipher.nonce_size()) if cipher.nonce_size() > 0 else b""
        plaintext = secrets.token_bytes(data_size)
        aad = b"benchmark_aad"
        
        # Warmup
        try:
            for _ in range(self.warmup_runs):
                ct = cipher.encrypt(plaintext, key, nonce, aad)
                _ = cipher.decrypt(ct, key, nonce, aad)
        except Exception as e:
            return CryptoResult(
                cipher=cipher.name,
                data_size=data_size,
                encrypt_time_us=0,
                decrypt_time_us=0,
                encrypt_mbps=0,
                decrypt_mbps=0,
                roundtrip_ok=False,
                avalanche_pct=0,
                error_msg=str(e),
            )
        
        # Timed encrypt
        encrypt_times = []
        for _ in range(self.timed_runs):
            # Fresh nonce for each iteration (required for security)
            iter_nonce = secrets.token_bytes(cipher.nonce_size()) if cipher.nonce_size() > 0 else b""
            t0 = perf_counter()
            ct = cipher.encrypt(plaintext, key, iter_nonce, aad)
            t1 = perf_counter()
            encrypt_times.append((t1 - t0) * 1e6)
        
        # Timed decrypt
        ct = cipher.encrypt(plaintext, key, nonce, aad)
        decrypt_times = []
        for _ in range(self.timed_runs):
            t0 = perf_counter()
            pt = cipher.decrypt(ct, key, nonce, aad)
            t1 = perf_counter()
            decrypt_times.append((t1 - t0) * 1e6)
        
        # Verify roundtrip
        roundtrip_ok = (pt == plaintext)
        
        # Compute avalanche
        try:
            avalanche = compute_avalanche(cipher, key, nonce)
        except Exception:
            avalanche = 0.0
        
        # Compute metrics
        encrypt_us = np.median(encrypt_times)
        decrypt_us = np.median(decrypt_times)
        
        encrypt_mbps = (data_size / 1e6) / (encrypt_us / 1e6) if encrypt_us > 0 else 0
        decrypt_mbps = (data_size / 1e6) / (decrypt_us / 1e6) if decrypt_us > 0 else 0
        
        return CryptoResult(
            cipher=cipher.name,
            data_size=data_size,
            encrypt_time_us=encrypt_us,
            decrypt_time_us=decrypt_us,
            encrypt_mbps=encrypt_mbps,
            decrypt_mbps=decrypt_mbps,
            roundtrip_ok=roundtrip_ok,
            avalanche_pct=avalanche,
        )
    
    def run_benchmark(
        self,
        sizes: List[int]
    ) -> List[CryptoResult]:
        """Run full benchmark across sizes."""
        results = []
        
        for size in sizes:
            print(f"\nData size: {size:,} bytes")
            print("-" * 60)
            
            for cipher in self.ciphers:
                try:
                    result = self.bench_single(cipher, size)
                    results.append(result)
                    
                    status = "✓" if result.roundtrip_ok else "✗"
                    print(f"  {cipher.name:12} | {status} | "
                          f"enc={result.encrypt_mbps:7.1f} MB/s | "
                          f"dec={result.decrypt_mbps:7.1f} MB/s | "
                          f"avalanche={result.avalanche_pct:5.1f}%")
                except Exception as e:
                    print(f"  {cipher.name:12} | ERROR: {e}")
        
        return results


def save_results(
    results: List[CryptoResult],
    output_dir: Path,
    timestamp: str
) -> Tuple[Path, Path]:
    """Save results to CSV and JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / f"crypto_benchmark_{timestamp}.csv"
    json_path = output_dir / f"crypto_benchmark_{timestamp}.json"
    
    # CSV
    with open(csv_path, 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=asdict(results[0]).keys())
            writer.writeheader()
            for r in results:
                writer.writerow(asdict(r))
    
    # JSON with metadata
    json_data = {
        'timestamp': timestamp,
        'platform': {
            'system': platform.system(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python': platform.python_version(),
        },
        'disclaimer': (
            "This benchmark measures throughput and diffusion only. "
            "The RFT cipher is NOT cryptographically secure and should "
            "NOT be used for real security applications."
        ),
        'results': [asdict(r) for r in results],
    }
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    return csv_path, json_path


def generate_summary(results: List[CryptoResult]) -> str:
    """Generate markdown summary of results."""
    lines = [
        "# Crypto Throughput Benchmark Results",
        "",
        f"**Generated:** {datetime.now().isoformat()}",
        "",
        "## ⚠️ IMPORTANT DISCLAIMER",
        "",
        "This benchmark compares **throughput and diffusion metrics only**.",
        "The RFT cipher is a **research/toy cipher** and has **NOT** been:",
        "- Audited by cryptographers",
        "- Proven secure against known attacks",
        "- Designed to replace AES-GCM or ChaCha20",
        "",
        "**DO NOT use the RFT cipher for real security applications.**",
        "",
        "## Throughput Comparison",
        "",
        "| Cipher | Size (KB) | Encrypt MB/s | Decrypt MB/s | Avalanche % | OK |",
        "|--------|-----------|--------------|--------------|-------------|-----|",
    ]
    
    for r in sorted(results, key=lambda x: (x.data_size, x.cipher)):
        status = "✓" if r.roundtrip_ok else "✗"
        size_kb = r.data_size / 1024
        lines.append(
            f"| {r.cipher} | {size_kb:.0f} | {r.encrypt_mbps:.1f} | "
            f"{r.decrypt_mbps:.1f} | {r.avalanche_pct:.1f} | {status} |"
        )
    
    # Average by cipher
    lines.extend([
        "",
        "## Average Throughput by Cipher",
        "",
        "| Cipher | Avg Encrypt MB/s | Avg Decrypt MB/s | Avg Avalanche % |",
        "|--------|------------------|------------------|-----------------|",
    ])
    
    from collections import defaultdict
    by_cipher = defaultdict(list)
    for r in results:
        by_cipher[r.cipher].append(r)
    
    for cipher, rs in sorted(by_cipher.items(), key=lambda x: -np.mean([r.encrypt_mbps for r in x[1]])):
        avg_enc = np.mean([r.encrypt_mbps for r in rs])
        avg_dec = np.mean([r.decrypt_mbps for r in rs])
        avg_aval = np.mean([r.avalanche_pct for r in rs])
        lines.append(f"| {cipher} | {avg_enc:.1f} | {avg_dec:.1f} | {avg_aval:.1f} |")
    
    lines.extend([
        "",
        "## Avalanche Effect Interpretation",
        "",
        "- **50%** is ideal (random diffusion)",
        "- **<40%** or **>60%** suggests poor diffusion",
        "- AES-GCM and ChaCha20 should be ~50%",
        "",
        "## Notes",
        "",
        "- Throughput varies with data size due to setup overhead",
        "- Hardware AES-NI instructions accelerate AES-GCM on modern CPUs",
        "- ChaCha20 is optimized for software implementations",
        "- RFT cipher uses NumPy/FFT which has different optimization characteristics",
    ])
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark RFT cipher vs AES-GCM/ChaCha20 (throughput only, NOT security)'
    )
    parser.add_argument(
        '--sizes', '-s',
        type=str,
        default='1024,4096,16384,65536',
        help='Comma-separated list of data sizes in bytes'
    )
    parser.add_argument(
        '--warmup', '-w',
        type=int,
        default=3,
        help='Number of warmup runs'
    )
    parser.add_argument(
        '--runs', '-r',
        type=int,
        default=10,
        help='Number of timed runs'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=_PROJECT_ROOT.parent / 'results' / 'competitors',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    sizes = [int(s.strip()) for s in args.sizes.split(',')]
    
    print("=" * 70)
    print("CRYPTO THROUGHPUT BENCHMARK: RFT Cipher vs AES-GCM/ChaCha20")
    print("=" * 70)
    print()
    print("⚠️  DISCLAIMER: This measures THROUGHPUT only, NOT security.")
    print("    The RFT cipher is a TOY cipher - DO NOT use for real security!")
    print()
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Python: {platform.python_version()}")
    print(f"Sizes: {sizes}")
    print(f"Available ciphers:", end=" ")
    if HAS_CRYPTO:
        print("aes_gcm, chacha20", end="")
    if HAS_RFT_CIPHER:
        print(", rft_cipher", end="")
    print()
    
    if not HAS_CRYPTO and not HAS_RFT_CIPHER:
        print("\nERROR: No ciphers available. Install cryptography: pip install cryptography")
        sys.exit(1)
    
    # Run benchmark
    benchmark = CryptoBenchmark(warmup_runs=args.warmup, timed_runs=args.runs)
    results = benchmark.run_benchmark(sizes)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path, json_path = save_results(results, args.output_dir, timestamp)
    print(f"\nResults saved:")
    print(f"  CSV:  {csv_path}")
    print(f"  JSON: {json_path}")
    
    # Generate summary
    summary = generate_summary(results)
    summary_path = args.output_dir / f"crypto_benchmark_{timestamp}.md"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"  Summary: {summary_path}")
    
    # Print final summary
    print("\n" + "=" * 70)
    print("SUMMARY: Average Throughput by Cipher")
    print("=" * 70)
    
    from collections import defaultdict
    by_cipher = defaultdict(list)
    for r in results:
        by_cipher[r.cipher].append(r)
    
    print(f"{'Cipher':<15} {'Enc MB/s':<12} {'Dec MB/s':<12} {'Avalanche %':<12}")
    print("-" * 55)
    for cipher, rs in sorted(by_cipher.items(), key=lambda x: -np.mean([r.encrypt_mbps for r in x[1]])):
        avg_enc = np.mean([r.encrypt_mbps for r in rs])
        avg_dec = np.mean([r.decrypt_mbps for r in rs])
        avg_aval = np.mean([r.avalanche_pct for r in rs])
        print(f"{cipher:<15} {avg_enc:<12.1f} {avg_dec:<12.1f} {avg_aval:<12.1f}")
    
    print("\n⚠️  Remember: This is NOT a security comparison!")


if __name__ == '__main__':
    main()
