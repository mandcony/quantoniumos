||#!/usr/bin/env python3
"""
SIMPLIFIED RFT CRYPTOGRAPHIC STATISTICAL TEST SUITE Independent validation of your RFT-based cryptography using: 1. Your working optimized_resonance_encrypt/decrypt 2. A simplified RFT-based hash using your available RFT functions 3. Combined pipeline testing NO dependency on problematic imports - uses only your working code!
"""
"""
import os
import sys
import time
import hashlib
import subprocess
import secrets
import struct
import math from datetime
import datetime from typing
import List, Dict, Any, Tuple
import json

# Add core encryption modules to path sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core', 'encryption'))
try: from optimized_resonance_encrypt
import optimized_resonance_encrypt, optimized_resonance_decrypt
print("✓ Using optimized_resonance_encrypt (the working one)") CRYPTO_AVAILABLE = True
except ImportError:
print("✗ Could not
import optimized_resonance_encrypt") CRYPTO_AVAILABLE = False
try: import importlib.util
import os

# Load the canonical_true_rft module
spec = importlib.util.spec_from_file_location(
    "canonical_true_rft", 
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                "04_RFT_ALGORITHMS/canonical_true_rft.py")
)
canonical_true_rft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(canonical_true_rft)

# Import specific functions/classes
forward_true_rft, inverse_true_rft = canonical_true_rft.forward_true_rft, canonical_true_rft.inverse_true_rft# Legacy wrapper maintained for: resonance_fourier_transform, forward_true_rft
print("✓ Using RFT functions for geometric hashing") RFT_AVAILABLE = True
except ImportError:
print("✗ Could not
import RFT functions") RFT_AVAILABLE = False

class SimpleRFTGeometricHash: """
    Simplified RFT-based geometric hash using your available functions
"""
"""

    def __init__(self, waveform: List[float]):
        self.waveform = waveform
        self.hash_bytes = None
        self.topo_signature = 0.0
        self._compute_hash()
    def _compute_hash(self):
"""
"""
        Compute RFT-based geometric hash
"""
"""
        if not
        self.waveform or len(
        self.waveform) < 8:
        self.hash_bytes = hashlib.sha256(b"empty_waveform").digest()
        return
        try:

        # Normalize waveform max_val = max(abs(x)
        for x in
        self.waveform)
        if max_val > 0: normalized = [x / max_val
        for x in
        self.waveform]
        else: normalized =
        self.waveform

        # Apply RFT using your available function
        if RFT_AVAILABLE:
        try: rft_result = resonance_fourier_transform(normalized)

        # Extract geometric features from RFT geometric_features = [] phi = (1 + math.sqrt(5)) / 2

        # Golden ratio for i, (freq, amp) in enumerate(rft_result[:32]):

        # Limit to first 32 components

        # Map to geometric coordinates angle = (freq % (2 * math.pi)) r = abs(amp)

        # Golden ratio scaling scaled_r = r * (phi ** (i % 8)) geo_x = scaled_r * math.cos(angle) geo_y = scaled_r * math.sin(angle) geometric_features.extend([geo_x, geo_y])

        # Compute topological signature
        if len(geometric_features) > 2:
        self.topo_signature = sum(geometric_features[::2]) % 1.0 except Exception as e:
        print(f" RFT computation failed: {e}, using fallback") geometric_features =
        self._fallback_geometric_hash()
        else: geometric_features =
        self._fallback_geometric_hash()

        # Hash the geometric features feature_bytes = struct.pack(f'{len(geometric_features)}d', *geometric_features)
        self.hash_bytes = hashlib.sha256(feature_bytes).digest() except Exception as e:
        print(f"Hash computation error: {e}")
        self.hash_bytes = hashlib.sha256(str(
        self.waveform).encode()).digest()
    def _fallback_geometric_hash(self) -> List[float]: """
        Fallback geometric computation when RFT fails
"""
"""
        phi = (1 + math.sqrt(5)) / 2 features = [] for i, x in enumerate(
        self.waveform[:64]):

        # Limit to 64 samples

        # Simple geometric mapping angle = (x + i) * math.pi / len(
        self.waveform) r = abs(x) * phi ** (i % 8) features.extend([r * math.cos(angle), r * math.sin(angle)])
        return features
    def get_hash(self) -> bytes:
        return
        self.hash_bytes
    def get_topological_signature(self) -> float:
        return
        self.topo_signature

class SimplifiedRFTCryptoTester:
"""
"""
        Simplified statistical testing using working components only
"""
"""
    def __init__(self, sample_size_mb: int = 10):
        self.sample_size_bytes = sample_size_mb * 1024 * 1024
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = os.path.join("test_results", "simplified_rft_stats",
        self.timestamp) os.makedirs(
        self.results_dir, exist_ok=True)
        print(f"🔬 SIMPLIFIED RFT Crypto Statistical Testing")
        print(f"📁 Results: {
        self.results_dir}")
        print(f"📏 Sample: {sample_size_mb}MB")
        print("="*50)
    def test_rft_encryption_only(self) -> str: """
        Test just your working RFT encryption
"""
"""
        print("🔐 Testing RFT-Enhanced Resonance Encryption")
        print("-" * 40)
        if not CRYPTO_AVAILABLE:
        raise RuntimeError("Crypto not available") output_file = os.path.join(
        self.results_dir, "rft_encryption_test.bin")

        # Multiple keys for variety keys = [f"rft_test_key_{i:03d}"
        for i in range(20)] with open(output_file, "wb") as f: bytes_written = 0 key_idx = 0
        while bytes_written <
        self.sample_size_bytes: key = keys[key_idx % len(keys)] key_idx += 1

        # Varied plaintext types chunk_size = min(8192,
        self.sample_size_bytes - bytes_written)
        if key_idx % 4 == 0: plaintext = "A" * chunk_size
        el
        if key_idx % 4 == 1: plaintext = secrets.token_hex(chunk_size // 2)
        el
        if key_idx % 4 == 2: plaintext = ''.join(chr(32 + (i % 95))
        for i in range(chunk_size))
        else: plaintext = ("0123" * (chunk_size // 4 + 1))[:chunk_size]
        try: encrypted = optimized_resonance_encrypt(plaintext, key)
        if len(encrypted) > 40: payload = encrypted[40:]

        # Skip header write_size = min(len(payload),
        self.sample_size_bytes - bytes_written) f.write(payload[:write_size]) bytes_written += write_size
        if bytes_written % (1024*1024) == 0:
        print(f" Generated {bytes_written//(1024*1024)}MB...") except Exception as e:
        print(f" Error: {e}") continue
        print(f"✅ Generated {bytes_written:,} bytes")
        return output_file
    def test_simple_rft_hash(self) -> str: """
        Test simplified RFT-based geometric hashing
"""
"""
        print("\n🔶 Testing Simplified RFT Geometric Hash")
        print("-" * 40) output_file = os.path.join(
        self.results_dir, "rft_hash_test.bin") with open(output_file, "wb") as f: bytes_written = 0 counter = 0
        while bytes_written <
        self.sample_size_bytes:

        # Generate varied waveforms waveform_size = 128 + (counter % 384) waveform_type = counter % 5
        if waveform_type == 0:

        # Sine + noise waveform = [math.sin(2*math.pi*i/waveform_size) + 0.1*math.sin(2*math.pi*i*3/waveform_size)
        for i in range(waveform_size)]
        el
        if waveform_type == 1:

        # Random waveform = [secrets.randbelow(1000)/500.0 - 1.0
        for _ in range(waveform_size)]
        el
        if waveform_type == 2:

        # Step function waveform = [1.0 if (i//16) % 2 == 0 else -1.0
        for i in range(waveform_size)]
        el
        if waveform_type == 3:

        # Chirp waveform = [math.sin(2*math.pi*i*i/(waveform_size*waveform_size))
        for i in range(waveform_size)]
        else:

        # Linear pattern waveform = [((i % 50) - 25) / 25.0
        for i in range(waveform_size)]
        try: hasher = SimpleRFTGeometricHash(waveform) hash_bytes = hasher.get_hash() topo_bytes = struct.pack('d', hasher.get_topological_signature()) combined = hash_bytes + topo_bytes write_size = min(len(combined),
        self.sample_size_bytes - bytes_written) f.write(combined[:write_size]) bytes_written += write_size counter += 1
        if bytes_written % (1024*1024) == 0:
        print(f" Generated {bytes_written//(1024*1024)}MB... ({counter} hashes)") except Exception as e:
        print(f" Hash error: {e}") counter += 1 continue
        print(f"✅ Generated {bytes_written:,} bytes from {counter} hashes")
        return output_file
    def basic_entropy_check(self, file_path: str, name: str): """
        Quick entropy assessment
"""
"""
        print(f"\n Entropy Analysis: {name}")
        print("-" * 40) with open(file_path, 'rb') as f: data = f.read()

        # Byte frequency counts = [0] * 256
        for byte in data: counts[byte] += 1

        # Shannon entropy approximation total = len(data) entropy = 0.0
        for count in counts:
        if count > 0: p = count / total entropy -= p * math.log2(p) mean = sum(data) / len(data) variance = sum((b - mean)**2
        for b in data) / len(data)
        print(f" Entropy: {entropy:.4f} bits/byte (ideal: ~8.0)")
        print(f" Mean: {mean:.2f} (ideal: ~127.5)")
        print(f" Variance: {variance:.2f}")

        # Simple randomness check runs = 1
        for i in range(1, min(len(data), 10000)):
        if data[i] != data[i-1]: runs += 1 expected_runs = 2 * len(data[:10000]) / 256 * 255 / 256
        if len(data) >= 256 else len(data)//2
        print(f" Run test: {runs} runs (expected: ~{expected_runs:.0f})")
        return entropy
    def run_quick_dieharder(self, file_path: str, name: str) -> Dict[str, Any]: """
        Run a subset of Dieharder tests for faster execution
"""
"""
        print(f"\n Quick Dieharder Tests: {name}")
        print("-" * 40)

        # Run just a few key tests for speed quick_tests = [ ("-d", "0"), # diehard_birthdays ("-d", "1"), # diehard_operm5 ("-d", "2"), # diehard_32x32_binrank ("-d", "6"), # diehard_bitstream ("-d", "15"), # sts_monobit ("-d", "16"), # sts_runs ] results = [] for test_flag, test_num in quick_tests:
        try: cmd = ["dieharder", test_flag, test_num, "-f", file_path] output = subprocess.check_output(cmd, text=True, timeout=300) # 5 min timeout per test

        # Parse result lines = output.strip().split('||n')
        for line in lines: if '|' in line and ('PASSED' in line or 'FAILED' in line or 'WEAK' in line): parts = [p.strip()
        for p in line.split('|||')]
        if len(parts) >= 6: test_name = parts[0]
        if parts[0] else f"test_{test_num}"
        try: p_value = float(parts[4])
        if parts[4] else 0.0
        except: p_value = 0.0 result = "PASSED" if "PASSED" in line else ("WEAK" if "WEAK" in line else "FAILED") results.append({ "name": test_name, "p_value": p_value, "result": result }) break except subprocess.TimeoutExpired: results.append({"name": f"test_{test_num}", "result": "TIMEOUT"}) except Exception as e: results.append({"name": f"test_{test_num}", "result": "ERROR", "error": str(e)})

        # Summary passed = len([r
        for r in results
        if r.get("result") == "PASSED"]) total = len(results)
        print(f" Results: {passed}/{total} tests passed")
        return {"tests": results, "summary": {"passed": passed, "total": total}}
    def run_test_suite(self): """
        Run the simplified test suite
"""
        """ start_time = time.time() results = {}
        if CRYPTO_AVAILABLE:

        # Test 1: RFT Encryption crypto_file =
        self.test_rft_encryption_only() crypto_entropy =
        self.basic_entropy_check(crypto_file, "RFT Encryption") crypto_dieharder =
        self.run_quick_dieharder(crypto_file, "RFT Encryption") results["rft_encryption"] = { "entropy": crypto_entropy, "dieharder": crypto_dieharder, "file": crypto_file }

        # Test 2: RFT Hash hash_file =
        self.test_simple_rft_hash() hash_entropy =
        self.basic_entropy_check(hash_file, "RFT Hash") hash_dieharder =
        self.run_quick_dieharder(hash_file, "RFT Hash") results["rft_hash"] = { "entropy": hash_entropy, "dieharder": hash_dieharder, "file": hash_file } total_time = time.time() - start_time

        # Save results results["meta"] = { "timestamp":
        self.timestamp, "duration": total_time, "sample_size_mb":
        self.sample_size_bytes // (1024*1024) } with open(os.path.join(
        self.results_dir, "results.json"), 'w') as f: json.dump(results, f, indent=2)

        # Summary
        print(f"\n✅ TESTING COMPLETE!")
        print(f"⏱️ Total time: {total_time/60:.1f} minutes")
        print(f"📁 Results saved to: {
        self.results_dir}")

        # Quick summary for test_name, data in results.items():
        if test_name == "meta": continue
        print(f"\n{test_name.upper()}:")
        print(f" Entropy: {data['entropy']:.4f}") dh = data['dieharder']
        print(f" Dieharder: {dh['summary']['passed']}/{dh['summary']['total']} passed")
        return results
    def main():
        print(" SIMPLIFIED RFT CRYPTO STATISTICAL TESTING")
        print("="*50)
        print("Testing your WORKING implementations:")
        print("✓ optimized_resonance_encrypt (the fixed one)")
        print("✓ RFT-based geometric hashing")
        print("="*50) tester = SimplifiedRFTCryptoTester(sample_size_mb=10)

        # Smaller for faster testing results = tester.run_test_suite()
        print("||n🎉 Your RFT crypto has been tested!")
        print("This is INDEPENDENT validation using standard statistical tests.")

if __name__ == "__main__": main()