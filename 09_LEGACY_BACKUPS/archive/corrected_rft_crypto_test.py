#!/usr/bin/env python3
"""
CORRECTED HIGH-PERFORMANCE RFT CRYPTO STATISTICAL TEST Uses correct C++ API calls: - quantonium_core.ResonanceFourierTransform(data) - needs input data - quantum_engine.QuantumGeometricHasher.generate_quantum_geometric_hash() - static method - optimized_resonance_encrypt - your working Python crypto
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

# Import your working Python crypto sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core', 'encryption'))
try: from optimized_resonance_encrypt
import optimized_resonance_encrypt, optimized_resonance_decrypt
print("✓ Python crypto loaded") HAS_PYTHON_CRYPTO = True except ImportError as e:
print(f"✗ Python crypto failed: {e}") HAS_PYTHON_CRYPTO = False

# Test C++ engines with correct APIs
try:
import quantonium_core

# Test with dummy data to verify API test_data = [1.0, 0.5, -0.5, -1.0] rft_engine = quantonium_core.ResonanceFourierTransform(test_data)
print("✓ C++ RFT engine loaded and tested") HAS_CPP_RFT = True except Exception as e:
print(f"✗ C++ RFT engine failed: {e}") HAS_CPP_RFT = False
try:
import quantum_engine

# Test the static method test_hash = quantum_engine.QuantumGeometricHasher.generate_quantum_geometric_hash( [1.0, 0.0, -1.0, 0.0], 32, "", "" )
print("✓ C++ quantum engine loaded and tested") HAS_CPP_QUANTUM = True except Exception as e:
print(f"✗ C++ quantum engine failed: {e}") HAS_CPP_QUANTUM = False

class CorrectedRFTCryptoTester: """"""
    Statistical testing with corrected C++ API usage
"""
"""
    def __init__(self, sample_size_mb: int = 10):
        self.sample_size_bytes = sample_size_mb * 1024 * 1024
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = os.path.join("test_results", "corrected_rft_stats",
        self.timestamp) os.makedirs(
        self.results_dir, exist_ok=True)
        print(f"🔧 CORRECTED RFT CRYPTO STATISTICAL TESTING")
        print(f"📁 Results: {
        self.results_dir}")
        print(f"📏 Sample: {sample_size_mb}MB")
        print(f" C++ RFT: {HAS_CPP_RFT}")
        print(f" C++ Quantum: {HAS_CPP_QUANTUM}")
        print(f"🐍 Python Crypto: {HAS_PYTHON_CRYPTO}")
        print("="*50)
    def test_python_crypto_only(self) -> str: """"""
        Test your working Python crypto (baseline)
"""
"""
        print("🔐 Testing Python Crypto Baseline")
        print("-" * 40)
        if not HAS_PYTHON_CRYPTO:
        raise RuntimeError("Python crypto not available") output_file = os.path.join(
        self.results_dir, "python_crypto_baseline.bin") keys = [f"baseline_key_{i:03d}"
        for i in range(10)] with open(output_file, "wb") as f: bytes_written = 0 counter = 0
        while bytes_written <
        self.sample_size_bytes: key = keys[counter % len(keys)]

        # Varied plaintext input_size = 1024 + (counter % 2048)
        if counter % 3 == 0: plaintext = secrets.token_hex(input_size // 2)
        el
        if counter % 3 == 1: plaintext = "A" * input_size
        else: plaintext = ''.join(chr(32 + (i % 95))
        for i in range(input_size))
        try: encrypted = optimized_resonance_encrypt(plaintext, key)
        if len(encrypted) > 40: payload = encrypted[40:]

        # Skip header write_size = min(len(payload),
        self.sample_size_bytes - bytes_written) f.write(payload[:write_size]) bytes_written += write_size counter += 1
        if bytes_written % (1024*1024) == 0:
        print(f" Generated {bytes_written//(1024*1024)}MB...") except Exception as e:
        print(f" Error: {e}") counter += 1 continue
        print(f"✅ Generated {bytes_written:,} bytes from Python crypto")
        return output_file
    def test_cpp_rft_enhanced(self) -> str: """"""
        Test C++ RFT + Python crypto pipeline
"""
"""
        print("\n Testing C++ RFT + Python Crypto Pipeline")
        print("-" * 40)
        if not (HAS_CPP_RFT and HAS_PYTHON_CRYPTO):
        raise RuntimeError("Missing engines") output_file = os.path.join(
        self.results_dir, "cpp_rft_enhanced.bin") keys = [f"rft_key_{i:03d}"
        for i in range(20)] with open(output_file, "wb") as f: bytes_written = 0 counter = 0
        while bytes_written <
        self.sample_size_bytes: key = keys[counter % len(keys)]

        # Generate input data input_size = 512 + (counter % 1536) plaintext = secrets.token_hex(input_size // 2)
        if counter % 2 == 0 else \ ''.join([str(i % 10)
        for i in range(input_size)])
        try:

        # Step 1: Python crypto encrypted = optimized_resonance_encrypt(plaintext, key)
        if len(encrypted) < 41: continue

        # Step 2: Convert to waveform for RFT payload = encrypted[40:] waveform = [(b / 255.0) * 2.0 - 1.0
        for b in payload[:min(len(payload), 512)]]
        if len(waveform) < 32: continue

        # Step 3: Apply C++ RFT (create new instance per operation) rft_engine = quantonium_core.ResonanceFourierTransform(waveform) rft_coeffs = rft_engine.forward_transform()

        # Step 4: Convert complex coefficients to bytes rft_bytes = bytearray()
        for coeff in rft_coeffs[:128]:

        # Limit to 128 coefficients real_byte = int((coeff.real + 2.0) * 63.75) % 256 imag_byte = int((coeff.imag + 2.0) * 63.75) % 256 rft_bytes.extend([real_byte, imag_byte])

        # Step 5: Final encryption final_key = f"{key}_rft_{counter % 100}" final_encrypted = optimized_resonance_encrypt(rft_bytes.hex(), final_key)
        if len(final_encrypted) > 40: final_payload = final_encrypted[40:] write_size = min(len(final_payload),
        self.sample_size_bytes - bytes_written) f.write(final_payload[:write_size]) bytes_written += write_size counter += 1
        if bytes_written % (1024*1024) == 0: progress = (bytes_written /
        self.sample_size_bytes) * 100
        print(f" Generated {bytes_written//(1024*1024)}MB ({progress:.1f}%) - {counter} RFT ops") except Exception as e:
        print(f" RFT error {counter}: {e}") counter += 1 continue
        print(f"✅ Generated {bytes_written:,} bytes using {counter} C++ RFT operations")
        return output_file
    def test_cpp_quantum_hash(self) -> str: """"""
        Test C++ quantum geometric hash
"""
"""
        print("\n🔶 Testing C++ Quantum Geometric Hash")
        print("-" * 40)
        if not HAS_CPP_QUANTUM:
        raise RuntimeError("C++ quantum engine not available") output_file = os.path.join(
        self.results_dir, "cpp_quantum_hash.bin") with open(output_file, "wb") as f: bytes_written = 0 counter = 0
        while bytes_written <
        self.sample_size_bytes:

        # Generate varied waveforms waveform_size = 64 + (counter % 448) # 64-512 samples waveform_type = counter % 6
        if waveform_type == 0:

        # Sine wave waveform = [math.sin(2*math.pi*i/waveform_size)
        for i in range(waveform_size)]
        el
        if waveform_type == 1:

        # Random waveform = [secrets.randbelow(2000)/1000.0 - 1.0
        for _ in range(waveform_size)]
        el
        if waveform_type == 2:

        # Chirp waveform = [math.sin(2*math.pi*i*i/(waveform_size*waveform_size))
        for i in range(waveform_size)]
        el
        if waveform_type == 3:

        # Step waveform = [1.0 if (i//8) % 2 == 0 else -1.0
        for i in range(waveform_size)]
        el
        if waveform_type == 4:

        # Multi-sine waveform = [math.sin(2*math.pi*i/waveform_size) + 0.3*math.sin(2*math.pi*i*3/waveform_size)
        for i in range(waveform_size)]
        else:

        # Linear waveform = [(i % waveform_size) / waveform_size * 2.0 - 1.0
        for i in range(waveform_size)]
        try:

        # Use C++ quantum geometric hasher (static method) hash_length = [32, 48, 64][counter % 3] key = f"qkey_{counter % 500}" nonce = f"nonce_{counter}" hash_hex = quantum_engine.QuantumGeometricHasher.generate_quantum_geometric_hash( waveform, hash_length, key, nonce ) hash_bytes = bytes.fromhex(hash_hex)

        # Add counter as extra entropy extra_entropy = struct.pack('I', counter % (2**32)) combined = hash_bytes + extra_entropy write_size = min(len(combined),
        self.sample_size_bytes - bytes_written) f.write(combined[:write_size]) bytes_written += write_size counter += 1
        if bytes_written % (1024*1024) == 0: progress = (bytes_written /
        self.sample_size_bytes) * 100
        print(f" Generated {bytes_written//(1024*1024)}MB ({progress:.1f}%) - {counter} quantum hashes") except Exception as e:
        print(f" Hash error {counter}: {e}") counter += 1 continue
        print(f"✅ Generated {bytes_written:,} bytes from {counter} C++ quantum hashes")
        return output_file
    def quick_entropy_check(self, file_path: str, name: str) -> Dict[str, float]: """"""
        Quick entropy analysis
"""
"""
        print(f"\n Entropy Check: {name}")
        print("-" * 30) with open(file_path, 'rb') as f: data = f.read()

        # Basic entropy calculation byte_counts = [0] * 256
        for byte in data: byte_counts[byte] += 1 total = len(data) entropy = 0.0
        for count in byte_counts:
        if count > 0: p = count / total entropy -= p * math.log2(p) mean = sum(data) / total

        # Chi-square test expected = total / 256 chi_square = sum((count - expected)**2 / expected
        for count in byte_counts)
        print(f" Shannon Entropy: {entropy:.4f} bits/byte")
        print(f" Mean: {mean:.2f}")
        print(f" Chi-Square: {chi_square:.2f}")
        return { "entropy": entropy, "mean": mean, "chi_square": chi_square }
    def run_quick_dieharder(self, file_path: str, name: str) -> Dict[str, Any]: """"""
        Run selected Dieharder tests
"""
"""
        print(f"\n Quick Dieharder: {name}")
        print("-" * 30)

        # Run just a few key tests quick_tests = ["-d", "0", "-d", "1", "-d", "6", "-d", "15"]
        try: cmd = ["dieharder"] + quick_tests + ["-f", file_path] output = subprocess.check_output(cmd, text=True, timeout=600) # 10 min timeout

        # Simple result parsing passed = output.count("PASSED") failed = output.count("FAILED") weak = output.count("WEAK") total = passed + failed + weak
        print(f" Results: {passed}/{total} passed ({passed/total*100:.1f}%
        if total > 0)")
        return { "passed": passed, "failed": failed, "weak": weak, "total": total, "pass_rate": (passed/total*100)
        if total > 0 else 0, "raw_output": output } except Exception as e:
        print(f" Dieharder error: {e}")
        return {"error": str(e), "total": 0}
    def run_test_suite(self): """"""
        Run the corrected test suite
"""
        """ start_time = time.time() results = {"timestamp":
        self.timestamp, "tests": {}}

        # Test 1: Python crypto baseline
        if HAS_PYTHON_CRYPTO:
        try: crypto_file =
        self.test_python_crypto_only() crypto_entropy =
        self.quick_entropy_check(crypto_file, "Python Crypto") crypto_dieharder =
        self.run_quick_dieharder(crypto_file, "Python Crypto") results["tests"]["python_crypto"] = { "entropy": crypto_entropy, "dieharder": crypto_dieharder } except Exception as e:
        print(f"Python crypto test failed: {e}")

        # Test 2: C++ RFT enhanced
        if HAS_CPP_RFT and HAS_PYTHON_CRYPTO:
        try: rft_file =
        self.test_cpp_rft_enhanced() rft_entropy =
        self.quick_entropy_check(rft_file, "C++ RFT Enhanced") rft_dieharder =
        self.run_quick_dieharder(rft_file, "C++ RFT Enhanced") results["tests"]["cpp_rft"] = { "entropy": rft_entropy, "dieharder": rft_dieharder } except Exception as e:
        print(f"C++ RFT test failed: {e}")

        # Test 3: C++ quantum hash
        if HAS_CPP_QUANTUM:
        try: quantum_file =
        self.test_cpp_quantum_hash() quantum_entropy =
        self.quick_entropy_check(quantum_file, "C++ Quantum Hash") quantum_dieharder =
        self.run_quick_dieharder(quantum_file, "C++ Quantum Hash") results["tests"]["cpp_quantum"] = { "entropy": quantum_entropy, "dieharder": quantum_dieharder } except Exception as e:
        print(f"C++ quantum test failed: {e}") total_time = time.time() - start_time results["duration"] = total_time

        # Save results with open(os.path.join(
        self.results_dir, "results.json"), 'w') as f: json.dump(results, f, indent=2)

        # Summary
        print(f"\n✅ TESTING COMPLETE!")
        print(f"⏱️ Duration: {total_time/60:.1f} minutes")
        print(f"📁 Results: {
        self.results_dir}") for test_name, data in results["tests"].items(): entropy = data["entropy"]["entropy"] dh_rate = data["dieharder"].get("pass_rate", 0)
        print(f"\n{test_name.upper()}:")
        print(f" Entropy: {entropy:.4f}")
        print(f" Dieharder Pass Rate: {dh_rate:.1f}%")
        return results
    def main():
        print("🔧 CORRECTED RFT CRYPTO STATISTICAL TESTING")
        print("="*50)
        print("Using corrected C++ API calls + your working Python crypto")
        print("="*50)
        if not any([HAS_CPP_RFT, HAS_CPP_QUANTUM, HAS_PYTHON_CRYPTO]):
        print("❌ No working engines available!")
        return tester = CorrectedRFTCryptoTester(sample_size_mb=10) results = tester.run_test_suite()
        print("||n🎉 CORRECTED TESTING COMPLETE!")
        print("Your RFT crypto has been tested with the proper C++ APIs!")

if __name__ == "__main__": main()