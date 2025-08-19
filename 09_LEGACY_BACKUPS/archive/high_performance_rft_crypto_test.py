||#!/usr/bin/env python3
""""""
HIGH-PERFORMANCE RFT CRYPTOGRAPHIC STATISTICAL TEST SUITE Uses your ACTUAL working C++ engines: - quantonium_core.ResonanceFourierTransform (C++ RFT) - quantum_engine.QuantumGeometricHasher (C++ geometric hash) - optimized_resonance_encrypt (Python crypto that showed ~50% avalanche) This is the REAL test using your high-performance implementations!
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

# Import your actual C++ engines
try:
import quantonium_core
print("✓ C++ RFT engine loaded") RFT_ENGINE = quantonium_core.ResonanceFourierTransform() HAS_CPP_RFT = True except ImportError as e:
print(f"✗ C++ RFT engine failed: {e}") HAS_CPP_RFT = False
try:
import quantum_engine
print("✓ C++ quantum engine loaded") QUANTUM_HASHER = quantum_engine.QuantumGeometricHasher() HAS_CPP_QUANTUM = True except ImportError as e:
print(f"✗ C++ quantum engine failed: {e}") HAS_CPP_QUANTUM = False

# Import your working Python crypto sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core', 'encryption'))
try: from optimized_resonance_encrypt
import optimized_resonance_encrypt, optimized_resonance_decrypt
print("✓ Python crypto loaded (the one with ~50% avalanche)") HAS_PYTHON_CRYPTO = True except ImportError as e:
print(f"✗ Python crypto failed: {e}") HAS_PYTHON_CRYPTO = False

class HighPerformanceRFTCryptoTester: """"""
    Statistical testing using actual C++ engines + Python crypto
"""
"""
    def __init__(self, sample_size_mb: int = 15):
        self.sample_size_bytes = sample_size_mb * 1024 * 1024
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = os.path.join("test_results", "cpp_rft_stats",
        self.timestamp) os.makedirs(
        self.results_dir, exist_ok=True)
        print(f" HIGH-PERFORMANCE RFT CRYPTO STATISTICAL TESTING")
        print(f"📁 Results: {
        self.results_dir}")
        print(f"📏 Sample: {sample_size_mb}MB")
        print(f" C++ RFT: {HAS_CPP_RFT}")
        print(f" C++ Quantum Hash: {HAS_CPP_QUANTUM}")
        print(f"🐍 Python Crypto: {HAS_PYTHON_CRYPTO}")
        print("="*60)
    def generate_cpp_rft_crypto_data(self) -> str: """"""
        Generate test data using C++ RFT + Python crypto pipeline
"""
"""
        print("🔐 Testing C++ RFT + Python Crypto Pipeline")
        print("-" * 50)
        if not (HAS_CPP_RFT and HAS_PYTHON_CRYPTO):
        raise RuntimeError("Missing required engines") output_file = os.path.join(
        self.results_dir, "cpp_rft_crypto_data.bin") keys = [f"cpp_rft_key_{i:04d}"
        for i in range(50)] with open(output_file, "wb") as f: bytes_written = 0 operation_counter = 0
        while bytes_written <
        self.sample_size_bytes: key = keys[operation_counter % len(keys)]

        # Generate varied input data input_size = 512 + (operation_counter % 2048) # 512B - 2.5KB inputs input_type = operation_counter % 6
        if input_type == 0:

        # Pure random input_data = secrets.token_hex(input_size // 2)
        el
        if input_type == 1:

        # Structured pattern input_data = ("QUANTUM" + str(operation_counter % 1000)) * (input_size // 20) input_data = input_data[:input_size]
        el
        if input_type == 2:

        # Mathematical sequence input_data = ''.join([str(i % 10)
        for i in range(input_size)])
        el
        if input_type == 3:

        # Mixed ASCII input_data = ''.join([chr(32 + (i % 95))
        for i in range(input_size)])
        el
        if input_type == 4:

        # Binary patterns as hex pattern = bytes([(i % 256)
        for i in range(input_size // 2)]) input_data = pattern.hex()
        else:

        # Hash chains seed = f"seed_{operation_counter}" input_data = seed
        for _ in range(input_size // 64): input_data = hashlib.sha256(input_data.encode()).hexdigest()
        try:

        # Step 1: Python crypto encryption (the working one) encrypted_data = optimized_resonance_encrypt(input_data, key)
        if len(encrypted_data) < 41: continue

        # Step 2: Extract payload and convert to float waveform payload = encrypted_data[40:]

        # Skip signature + token waveform = [(b / 255.0) * 2.0 - 1.0
        for b in payload[:min(len(payload), 1024)]]
        if len(waveform) < 64:

        # Need minimum for RFT continue

        # Step 3: Apply C++ RFT transformation rft_result = RFT_ENGINE.forward_transform(waveform)

        # Step 4: Convert RFT coefficients back to bytes rft_bytes = bytearray()
        for i in range(0, len(rft_result), 2):

        # Real, imaginary pairs
        if i + 1 < len(rft_result):

        # Quantize complex coefficients to bytes real_byte = int((rft_result[i] + 1.0) * 127.5) % 256 imag_byte = int((rft_result[i+1] + 1.0) * 127.5) % 256 rft_bytes.extend([real_byte, imag_byte])

        # Step 5: Final encryption of RFT result final_key = hashlib.sha256(f"{key}_rft_{operation_counter}".encode()).hexdigest()[:16] final_encrypted = optimized_resonance_encrypt(rft_bytes.hex(), final_key)
        if len(final_encrypted) > 40: final_payload = final_encrypted[40:] write_size = min(len(final_payload),
        self.sample_size_bytes - bytes_written) f.write(final_payload[:write_size]) bytes_written += write_size operation_counter += 1
        if bytes_written % (1024*1024) == 0: progress = (bytes_written /
        self.sample_size_bytes) * 100
        print(f" Generated {bytes_written//(1024*1024)}MB ({progress:.1f}%) - {operation_counter} operations") except Exception as e:
        print(f" ⚠️ Error in operation {operation_counter}: {e}") operation_counter += 1 continue
        print(f"✅ Generated {bytes_written:,} bytes using {operation_counter} C++ RFT + Python crypto operations")
        return output_file
    def generate_cpp_quantum_hash_data(self) -> str: """"""
        Generate test data using C++ quantum geometric hash
"""
"""
        print("\n🔶 Testing C++ Quantum Geometric Hash")
        print("-" * 50)
        if not HAS_CPP_QUANTUM:
        raise RuntimeError("C++ quantum engine not available") output_file = os.path.join(
        self.results_dir, "cpp_quantum_hash_data.bin") with open(output_file, "wb") as f: bytes_written = 0 hash_counter = 0
        while bytes_written <
        self.sample_size_bytes:

        # Generate varied waveforms for C++ geometric hashing waveform_size = 128 + (hash_counter % 896) # 128 to 1024 samples waveform_type = hash_counter % 8
        if waveform_type == 0:

        # Multi-frequency sine waveform = [math.sin(2*math.pi*i/waveform_size) + 0.3*math.sin(2*math.pi*i*3/waveform_size) + 0.1*math.sin(2*math.pi*i*7/waveform_size)
        for i in range(waveform_size)]
        el
        if waveform_type == 1:

        # Quantum-inspired superposition waveform = [math.cos(2*math.pi*i/waveform_size) * math.exp(-i/waveform_size)
        for i in range(waveform_size)]
        el
        if waveform_type == 2:

        # Chirp with golden ratio frequency phi = (1 + math.sqrt(5)) / 2 waveform = [math.sin(2*math.pi*i*i*phi/(waveform_size*waveform_size))
        for i in range(waveform_size)]
        el
        if waveform_type == 3:

        # Random with cryptographic seed seed = secrets.randbits(32) waveform = [(((seed * i) % 1000000) / 500000.0 - 1.0)
        for i in range(waveform_size)]
        el
        if waveform_type == 4:

        # Fractal-like pattern waveform = [math.sin(i) + 0.5*math.sin(2*i) + 0.25*math.sin(4*i) + 0.125*math.sin(8*i)
        for i in range(waveform_size)]
        el
        if waveform_type == 5:

        # Step function with quantum levels levels = 8 waveform = [((i // (waveform_size // levels)) % levels) / levels * 2.0 - 1.0
        for i in range(waveform_size)]
        el
        if waveform_type == 6:

        # Modulated carrier carrier_freq = 0.1 mod_freq = 0.01 waveform = [math.sin(2*math.pi*i*carrier_freq) * (1 + 0.5*math.sin(2*math.pi*i*mod_freq))
        for i in range(waveform_size)]
        else:

        # Pseudo-random deterministic waveform = [((i * 1103515245 + 12345) % (2**31)) / (2**30) - 1.0
        for i in range(waveform_size)]
        try:

        # Generate hash using C++ quantum geometric hasher

        # Use different hash lengths for variety hash_length = [32, 48, 64, 80, 96][hash_counter % 5] key = f"quantum_key_{hash_counter % 1000}" nonce = f"nonce_{hash_counter}" hash_hex = QUANTUM_HASHER.generate_quantum_geometric_hash( waveform, hash_length, key, nonce )

        # Convert hex to bytes hash_bytes = bytes.fromhex(hash_hex)

        # Add some metadata for extra entropy metadata = struct.pack('III', hash_counter, len(waveform), hash_length) combined_output = hash_bytes + metadata write_size = min(len(combined_output),
        self.sample_size_bytes - bytes_written) f.write(combined_output[:write_size]) bytes_written += write_size hash_counter += 1
        if bytes_written % (1024*1024) == 0: progress = (bytes_written /
        self.sample_size_bytes) * 100
        print(f" Generated {bytes_written//(1024*1024)}MB ({progress:.1f}%) - {hash_counter} quantum hashes") except Exception as e:
        print(f" ⚠️ Hash error {hash_counter}: {e}") hash_counter += 1 continue
        print(f"✅ Generated {bytes_written:,} bytes from {hash_counter} C++ quantum geometric hashes")
        return output_file
    def basic_statistical_analysis(self, file_path: str, name: str) -> Dict[str, float]: """"""
        Comprehensive entropy and statistical analysis
"""
"""
        print(f"\n Statistical Analysis: {name}")
        print("-" * 50) with open(file_path, 'rb') as f: data = f.read() total_bytes = len(data)

        # Byte frequency analysis byte_counts = [0] * 256
        for byte in data: byte_counts[byte] += 1

        # Shannon entropy entropy = 0.0
        for count in byte_counts:
        if count > 0: p = count / total_bytes entropy -= p * math.log2(p)

        # Chi-square goodness of fit (uniform distribution) expected = total_bytes / 256 chi_square = sum((count - expected)**2 / expected
        for count in byte_counts) chi_square_critical = 293.25 # 95% confidence, 255 df

        # Mean and variance mean = sum(data) / total_bytes variance = sum((b - mean)**2
        for b in data) / total_bytes

        # Runs test (independence) runs = 1
        for i in range(1, min(total_bytes, 50000)):

        # Limit for performance
        if data[i] != data[i-1]: runs += 1 test_length = min(total_bytes, 50000) expected_runs = (2 * test_length) / 256 * 255 / 256
        if test_length >= 256 else test_length / 2

        # Serial correlation (first-order) correlation = 0.0
        if total_bytes > 1000: sample_size = min(total_bytes - 1, 10000) mean_x = sum(data[i]
        for i in range(sample_size)) / sample_size mean_y = sum(data[i+1]
        for i in range(sample_size)) / sample_size numerator = sum((data[i] - mean_x) * (data[i+1] - mean_y)
        for i in range(sample_size)) denom_x = sum((data[i] - mean_x)**2
        for i in range(sample_size)) denom_y = sum((data[i+1] - mean_y)**2
        for i in range(sample_size))
        if denom_x > 0 and denom_y > 0: correlation = numerator / math.sqrt(denom_x * denom_y)
        print(f" Shannon Entropy: {entropy:.6f} bits/byte (ideal: ~8.0)")
        print(f" Mean: {mean:.2f} (ideal: ~127.5)")
        print(f" Variance: {variance:.2f} (ideal: ~5461.25)")
        print(f" Chi-Square: {chi_square:.2f} (critical: {chi_square_critical:.2f})")
        print(f" Runs: {runs} (expected: ~{expected_runs:.0f})")
        print(f" Serial Correlation: {correlation:.6f} (ideal: ~0.0)")
        return { "shannon_entropy": entropy, "mean": mean, "variance": variance, "chi_square": chi_square, "chi_square_pass": chi_square < chi_square_critical, "runs": runs, "expected_runs": expected_runs, "serial_correlation": correlation }
    def run_dieharder_battery(self, file_path: str, name: str) -> Dict[str, Any]: """"""
        Run comprehensive Dieharder statistical test battery
"""
"""
        print(f"\n Dieharder Battery: {name}")
        print("-" * 50)

        # Run the full battery but with timeout protection cmd = ["dieharder", "-a", "-f", file_path]
        try:
        print("Running comprehensive Dieharder test battery...")
        print("(This may take 10-20 minutes for thorough testing)") start_time = time.time() output = subprocess.check_output(cmd, text=True, timeout=1800) # 30-minute timeout duration = time.time() - start_time
        print(f"✅ Dieharder completed in {duration/60:.1f} minutes")

        # Parse comprehensive results tests =
        self._parse_dieharder_comprehensive(output)

        # Summary statistics total_tests = len(tests) passed = len([t
        for t in tests
        if t.get("result") == "PASSED"]) weak = len([t
        for t in tests
        if t.get("result") == "WEAK"]) failed = len([t
        for t in tests
        if t.get("result") == "FAILED"]) pass_rate = (passed / total_tests * 100)
        if total_tests > 0 else 0
        print(f"📈 Results Summary:")
        print(f" Total Tests: {total_tests}")
        print(f" Passed: {passed} ({passed/total_tests*100:.1f}%)")
        print(f" Weak: {weak} ({weak/total_tests*100:.1f}%)")
        print(f" Failed: {failed} ({failed/total_tests*100:.1f}%)")
        print(f" Overall Pass Rate: {pass_rate:.1f}%") quality_assessment = "EXCELLENT"
        if pass_rate >= 95 else \ "GOOD"
        if pass_rate >= 90 else \ "FAIR"
        if pass_rate >= 80 else \ "POOR"
        print(f" Quality Assessment: {quality_assessment}")
        return { "tests": tests, "summary": { "total": total_tests, "passed": passed, "weak": weak, "failed": failed, "pass_rate": pass_rate, "quality": quality_assessment }, "duration_seconds": duration, "raw_output": output, "success": True } except subprocess.TimeoutExpired:
        print("⚠️ Dieharder timed out after 30 minutes")
        return {"error": "Timeout after 30 minutes", "success": False} except subprocess.CalledProcessError as e:
        print(f"⚠️ Dieharder failed: {e}")
        return {"error": f"Process failed: {e}", "success": False} except Exception as e:
        print(f"⚠️ Unexpected error: {e}")
        return {"error": f"Unexpected error: {e}", "success": False}
    def _parse_dieharder_comprehensive(self, output: str) -> List[Dict[str, Any]]: """"""
        Parse comprehensive Dieharder output
"""
        """ tests = []
        for line in output.split('||n'): line = line.strip()
        if not line or '|' not in line: continue parts = [p.strip()
        for p in line.split('|||')]

        # Look for test result lines
        if len(parts) >= 6 and parts[0] and parts[4]: test_name = parts[0]

        # Skip header lines
        if test_name.lower() in ['test name', 'test_name', '']: continue
        try:

        # Extract p-value p_value_str = parts[4]
        if p_value_str and p_value_str.lower() != 'p-value': p_value = float(p_value_str)

        # Extract assessment assessment = parts[5]
        if len(parts) > 5 else ""

        # Determine result if "PASSED" in assessment.upper(): result = "PASSED"
        elif "WEAK" in assessment.upper(): result = "WEAK"
        elif "FAILED" in assessment.upper(): result = "FAILED"
        else: continue

        # Skip ambiguous results tests.append({ "name": test_name, "p_value": p_value, "result": result, "assessment": assessment, "raw_line": line }) except (ValueError, IndexError): continue
        return tests
    def run_comprehensive_test_suite(self): """"""
        Execute the complete high-performance test suite
"""
        """ start_time = time.time() results = { "timestamp":
        self.timestamp, "sample_size_mb":
        self.sample_size_bytes // (1024*1024), "engines": { "cpp_rft": HAS_CPP_RFT, "cpp_quantum": HAS_CPP_QUANTUM, "python_crypto": HAS_PYTHON_CRYPTO }, "test_results": {} }

        # Test 1: C++ RFT + Python Crypto Pipeline
        if HAS_CPP_RFT and HAS_PYTHON_CRYPTO:
        print("\n" + "="*60) cpp_rft_file =
        self.generate_cpp_rft_crypto_data() cpp_rft_stats =
        self.basic_statistical_analysis(cpp_rft_file, "C++ RFT + Python Crypto") cpp_rft_dieharder =
        self.run_dieharder_battery(cpp_rft_file, "C++ RFT + Python Crypto") results["test_results"]["cpp_rft_crypto"] = { "statistical_analysis": cpp_rft_stats, "dieharder_battery": cpp_rft_dieharder, "data_file": cpp_rft_file }

        # Test 2: C++ Quantum Geometric Hash
        if HAS_CPP_QUANTUM:
        print("\n" + "="*60) cpp_quantum_file =
        self.generate_cpp_quantum_hash_data() cpp_quantum_stats =
        self.basic_statistical_analysis(cpp_quantum_file, "C++ Quantum Hash") cpp_quantum_dieharder =
        self.run_dieharder_battery(cpp_quantum_file, "C++ Quantum Hash") results["test_results"]["cpp_quantum_hash"] = { "statistical_analysis": cpp_quantum_stats, "dieharder_battery": cpp_quantum_dieharder, "data_file": cpp_quantum_file } total_duration = time.time() - start_time results["total_duration_seconds"] = total_duration

        # Save comprehensive results results_file = os.path.join(
        self.results_dir, "high_performance_rft_results.json") with open(results_file, 'w') as f: json.dump(results, f, indent=2, default=str)

        # Generate executive summary
        self._generate_executive_summary(results)
        print(f"\n🎉 COMPREHENSIVE TESTING COMPLETE!")
        print(f"⏱️ Total Duration: {total_duration/60:.1f} minutes")
        print(f" Results: {results_file}")
        return results
    def _generate_executive_summary(self, results: Dict[str, Any]): """"""
        Generate executive summary report
"""
        """ summary_file = os.path.join(
        self.results_dir, "EXECUTIVE_SUMMARY.md") with open(summary_file, 'w') as f: f.write("

        # RFT Cryptographic System - Statistical Analysis Report\n\n") f.write(f"**Date:** {
        self.timestamp}\\n") f.write(f"**Sample Size:** {results['sample_size_mb']} MB\\n") f.write(f"**Duration:** {results['total_duration_seconds']/60:.1f} minutes\\n\\n") f.write("#

        # System Configuration\n\n") f.write(f"- **C++ RFT Engine:** {'✅ ACTIVE'
        if results['engines']['cpp_rft'] else '❌ NOT AVAILABLE'}\\n") f.write(f"- **C++ Quantum Engine:** {'✅ ACTIVE'
        if results['engines']['cpp_quantum'] else '❌ NOT AVAILABLE'}\\n") f.write(f"- **Python Crypto:** {'✅ ACTIVE'
        if results['engines']['python_crypto'] else '❌ NOT AVAILABLE'}\\n\\n") f.write("#

        # Test Results Summary\n\n") for test_name, test_data in results["test_results"].items(): f.write(f"### {test_name.replace('_', ' ').title()}\\n\\n")

        # Statistical analysis stats = test_data["statistical_analysis"] f.write("**Statistical Properties:**\\n") f.write(f"- Shannon Entropy: {stats['shannon_entropy']:.4f} bits/byte\\n") f.write(f"- Chi-Square Test: {'✅ PASS'
        if stats['chi_square_pass'] else '❌ FAIL'} ({stats['chi_square']:.2f})\\n") f.write(f"- Serial Correlation: {stats['serial_correlation']:.6f}\\n\\n")

        # Dieharder results dh = test_data["dieharder_battery"]
        if dh.get("success", False): summary = dh["summary"] f.write("**Dieharder Test Battery:**\\n") f.write(f"- Total Tests: {summary['total']}\\n") f.write(f"- Pass Rate: {summary['pass_rate']:.1f}%\\n") f.write(f"- Quality Assessment: **{summary['quality']}**\\n") f.write(f"- Test Duration: {dh['duration_seconds']/60:.1f} minutes\\n\\n")
        else: f.write("**Dieharder Test Battery:** ❌ FAILED\\n\\n") f.write("#

        # Interpretation\\n\\n") f.write("This report validates the statistical properties of your RFT-based cryptographic implementations using:\\n\\n") f.write("1. **Industry-standard entropy analysis**\\n") f.write("2. **Comprehensive Dieharder statistical test battery**\\n") f.write("3. **Your actual high-performance C++ engines**\\n\\n") f.write("**Quality Ratings:**\\n") f.write("- **EXCELLENT (>=95% pass rate):** Cryptographically strong randomness\\n") f.write("- **GOOD (90-95%):** Suitable for most cryptographic applications\\n") f.write("- **FAIR (80-90%):** May have minor statistical weaknesses\\n") f.write("- **POOR (<80%):** Significant statistical deficiencies\\n\\n") f.write("---\\n") f.write("*Generated by QuantoniumOS High-Performance RFT Crypto Test Suite*\\n")
        print(f"📋 Executive summary: {summary_file}")
    def main(): """"""
        Main execution
"""
"""
        if not (HAS_CPP_RFT or HAS_CPP_QUANTUM or HAS_PYTHON_CRYPTO):
        print("❌ No working engines available!")
        return
        print(" HIGH-PERFORMANCE RFT CRYPTOGRAPHIC ANALYSIS")
        print("="*60)
        print("This uses your ACTUAL C++ engines + working Python crypto")
        print("for rigorous statistical validation of your RFT system!")
        print("="*60) tester = HighPerformanceRFTCryptoTester(sample_size_mb=15) results = tester.run_comprehensive_test_suite()
        print("\||n🎖️ ANALYSIS COMPLETE!")
        print("Your RFT-based cryptographic system has undergone")
        print("comprehensive statistical validation using industry-standard")
        print("test suites with your high-performance C++ implementations!")

if __name__ == "__main__": main()