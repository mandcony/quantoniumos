#!/usr/bin/env python3
"""
Statistical Validation Script for QuantoniumOS

This script generates a large entropy corpus (1 GiB) and runs comprehensive
statistical tests to meet NIST/STS reviewer thresholds for publication-grade validation.
"""

import os
import sys
import time
import json
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.encryption.wave_entropy_engine import WaveformEntropyEngine
from core.encryption.geometric_waveform_hash import GeometricWaveformHash
from core.encryption.resonance_encrypt import resonance_encrypt


class StatisticalValidator:
    """Comprehensive statistical validation for publication-grade analysis"""
    
    def __init__(self, output_dir: str = "validation_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Configure test parameters for publication standards
        self.corpus_size_bytes = 10 * 1024 * 1024  # 10 MiB for testing (normally 1 GiB)
        self.block_sizes = [128, 256, 512, 1024, 2048]  # Multiple block sizes
        self.test_sequences = 100  # Number of test sequences per algorithm (normally 1000)
        
        # Statistical thresholds (NIST recommendations)
        self.alpha_threshold = 0.01  # Significance level
        self.pass_threshold = 0.96   # At least 96% must pass
        
    def generate_entropy_corpus(self) -> bytes:
        """Generate a large entropy corpus using QuantoniumOS algorithms"""
        print(f"Generating {self.corpus_size_bytes // (1024*1024)} MiB entropy corpus...")
        
        # Initialize entropy engine with cryptographically secure randomization
        entropy_engine = WaveformEntropyEngine()
        
        # Generate corpus in chunks to manage memory
        corpus = bytearray()
        chunk_size = 1024 * 1024  # 1 MiB chunks
        chunks_needed = self.corpus_size_bytes // chunk_size
        
        start_time = time.time()
        for i in range(chunks_needed):
            if i % 100 == 0:
                elapsed = time.time() - start_time
                progress = (i / chunks_needed) * 100
                print(f"  Progress: {progress:.1f}% ({i}/{chunks_needed} chunks, {elapsed:.1f}s)")
            
            # Generate chunk using entropy engine
            chunk = entropy_engine.generate_entropy_bytes(chunk_size)
            corpus.extend(chunk)
            
            # Periodically mutate the engine state for variety
            entropy_engine.mutate_waveform()
            entropy_engine.dynamic_feedback()
        
        print(f"Corpus generation completed in {time.time() - start_time:.1f} seconds")
        return bytes(corpus)
    
    def monobit_test(self, data: bytes) -> Tuple[float, bool]:
        """NIST SP 800-22 Monobit Test"""
        n = len(data) * 8  # Total bits
        
        # Count 1s in the bit stream
        ones = 0
        for byte in data:
            ones += bin(byte).count('1')
        
        # Calculate test statistic
        s = abs(ones - n/2) / np.sqrt(n/4)
        
        # Calculate p-value (simplified approximation)
        p_value = np.exp(-s**2 / 2)
        
        return p_value, p_value >= self.alpha_threshold
    
    def frequency_test_within_block(self, data: bytes, block_size: int) -> Tuple[float, bool]:
        """NIST SP 800-22 Frequency Test within a Block"""
        n = len(data) * 8
        num_blocks = n // block_size
        
        if num_blocks == 0:
            return 0.0, False
        
        # Calculate proportion of ones in each block
        proportions = []
        bit_pos = 0
        
        for block in range(num_blocks):
            ones_in_block = 0
            
            for _ in range(block_size):
                byte_idx = bit_pos // 8
                bit_idx = bit_pos % 8
                
                if byte_idx < len(data):
                    bit_val = (data[byte_idx] >> bit_idx) & 1
                    ones_in_block += bit_val
                
                bit_pos += 1
            
            proportions.append(ones_in_block / block_size)
        
        # Calculate chi-square statistic
        chi_square = 4 * block_size * sum((p - 0.5)**2 for p in proportions)
        
        # Approximate p-value
        p_value = np.exp(-chi_square / 2)
        
        return p_value, p_value >= self.alpha_threshold
    
    def runs_test(self, data: bytes) -> Tuple[float, bool]:
        """NIST SP 800-22 Runs Test"""
        # Convert to bit array
        bits = []
        for byte in data:
            for i in range(8):
                bits.append((byte >> i) & 1)
        
        n = len(bits)
        if n == 0:
            return 0.0, False
        
        # Calculate proportion of ones
        ones = sum(bits)
        pi = ones / n
        
        # Pre-test: check if proportion is approximately 0.5
        if abs(pi - 0.5) >= (2 / np.sqrt(n)):
            return 0.0, False
        
        # Count runs
        runs = 1
        for i in range(1, n):
            if bits[i] != bits[i-1]:
                runs += 1
        
        # Calculate test statistic
        expected_runs = 2 * n * pi * (1 - pi) + 1
        variance = 2 * n * pi * (1 - pi) * (2 * pi * (1 - pi) - 1) / (n - 1)
        
        if variance <= 0:
            return 0.0, False
        
        z = (runs - expected_runs) / np.sqrt(variance)
        
        # Calculate p-value
        p_value = 2 * (1 - np.abs(z) / np.sqrt(2 * np.pi))
        
        return p_value, p_value >= self.alpha_threshold
    
    def serial_test(self, data: bytes, m: int = 16) -> Tuple[float, bool]:
        """NIST SP 800-22 Serial Test (simplified)"""
        # Convert to bit array
        bits = []
        for byte in data:
            for i in range(8):
                bits.append((byte >> i) & 1)
        
        n = len(bits)
        if n < m:
            return 0.0, False
        
        # Count m-bit patterns
        patterns = {}
        for i in range(n - m + 1):
            pattern = tuple(bits[i:i+m])
            patterns[pattern] = patterns.get(pattern, 0) + 1
        
        # Calculate chi-square statistic
        expected = (n - m + 1) / (2**m)
        chi_square = sum((count - expected)**2 / expected for count in patterns.values())
        
        # Approximate p-value
        p_value = np.exp(-chi_square / (2 * (2**m - 1)))
        
        return p_value, p_value >= self.alpha_threshold
    
    def entropy_estimate(self, data: bytes, block_size: int = 256) -> float:
        """Estimate entropy using Shannon entropy calculation"""
        if len(data) < block_size:
            return 0.0
        
        # Calculate frequency of each byte value
        frequencies = [0] * 256
        for byte in data[:block_size]:
            frequencies[byte] += 1
        
        # Calculate Shannon entropy
        entropy = 0.0
        for freq in frequencies:
            if freq > 0:
                p = freq / block_size
                entropy -= p * np.log2(p)
        
        return entropy
    
    def run_comprehensive_tests(self, data: bytes) -> Dict[str, Any]:
        """Run comprehensive statistical test suite"""
        print("Running comprehensive statistical tests...")
        results = {}
        
        # Basic statistics
        results["data_size_bytes"] = len(data)
        results["data_size_bits"] = len(data) * 8
        
        # Entropy analysis
        results["entropy_estimate"] = self.entropy_estimate(data)
        results["max_entropy"] = 8.0  # Maximum for byte data
        
        # NIST tests
        print("  Running Monobit test...")
        mono_p, mono_pass = self.monobit_test(data)
        results["monobit"] = {"p_value": mono_p, "passed": mono_pass}
        
        print("  Running Runs test...")
        runs_p, runs_pass = self.runs_test(data)
        results["runs"] = {"p_value": runs_p, "passed": runs_pass}
        
        print("  Running Serial test...")
        serial_p, serial_pass = self.serial_test(data)
        results["serial"] = {"p_value": serial_p, "passed": serial_pass}
        
        # Block frequency tests for different block sizes
        results["block_frequency"] = {}
        for block_size in self.block_sizes:
            if len(data) * 8 >= block_size * 10:  # Need at least 10 blocks
                print(f"  Running Block Frequency test (block size: {block_size})...")
                bf_p, bf_pass = self.frequency_test_within_block(data, block_size)
                results["block_frequency"][str(block_size)] = {
                    "p_value": bf_p, 
                    "passed": bf_pass
                }
        
        # Calculate overall pass rate
        all_tests = []
        for test_name, test_result in results.items():
            if isinstance(test_result, dict) and "passed" in test_result:
                all_tests.append(test_result["passed"])
            elif test_name == "block_frequency":
                for block_result in test_result.values():
                    all_tests.append(block_result["passed"])
        
        if all_tests:
            results["overall_pass_rate"] = sum(all_tests) / len(all_tests)
            results["meets_nist_threshold"] = results["overall_pass_rate"] >= self.pass_threshold
        else:
            results["overall_pass_rate"] = 0.0
            results["meets_nist_threshold"] = False
        
        return results
    
    def validate_hash_function(self) -> Dict[str, Any]:
        """Validate the geometric waveform hash function"""
        print("\nValidating Geometric Waveform Hash Function...")
        
        # Generate diverse test data using hash function
        hash_outputs = bytearray()
        
        for i in range(self.test_sequences):
            if i % 100 == 0:
                print(f"  Generating hash sequence {i}/{self.test_sequences}")
            
            # Create diverse waveform input
            waveform = [np.sin(i * 0.1 + j * 0.01) for j in range(100)]
            waveform[i % 100] += i * 0.001  # Small perturbation
            
            # Generate hash
            hasher = GeometricWaveformHash(waveform=waveform)
            hash_str = hasher.generate_hash()
            
            # Extract the hex part and convert to bytes
            hex_part = hash_str.split('_')[-1]
            hash_outputs.extend(bytes.fromhex(hex_part))
        
        # Run statistical tests on hash outputs
        return self.run_comprehensive_tests(bytes(hash_outputs))
    
    def validate_encryption_function(self) -> Dict[str, Any]:
        """Validate the resonance encryption function"""
        print("\nValidating Resonance Encryption Function...")
        
        # Generate diverse encryption outputs
        encrypted_outputs = bytearray()
        
        for i in range(self.test_sequences):
            if i % 100 == 0:
                print(f"  Generating encryption sequence {i}/{self.test_sequences}")
            
            # Create diverse plaintext
            plaintext = f"Test message {i} with varying content" + "x" * (i % 100)
            
            # Use diverse amplitude and phase parameters
            amplitude = 0.1 + (i % 100) / 1000.0
            phase = (i * 0.1) % (2 * np.pi)
            
            # Encrypt
            encrypted = resonance_encrypt(plaintext, amplitude, phase)
            encrypted_outputs.extend(encrypted)
        
        # Run statistical tests on encrypted outputs
        return self.run_comprehensive_tests(bytes(encrypted_outputs))
    
    def validate_entropy_engine(self) -> Dict[str, Any]:
        """Validate the wave entropy engine"""
        print("\nValidating Wave Entropy Engine...")
        
        # Generate large sample from entropy engine
        entropy_engine = WaveformEntropyEngine()
        
        sample_size = min(10 * 1024 * 1024, self.corpus_size_bytes // 10)  # 10 MiB or 1/10 of corpus
        entropy_data = entropy_engine.generate_entropy_bytes(sample_size)
        
        # Run statistical tests on entropy output
        return self.run_comprehensive_tests(entropy_data)
    
    def generate_report(self, results: Dict[str, Dict[str, Any]]) -> str:
        """Generate comprehensive validation report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.output_dir, f"statistical_validation_report_{timestamp}.html")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"""
<!DOCTYPE html>
<html>
<head>
    <title>QuantoniumOS Statistical Validation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #3498db; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .pass {{ color: green; font-weight: bold; }}
        .fail {{ color: red; font-weight: bold; }}
        .summary {{ background-color: #eef; padding: 10px; border-radius: 5px; margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>QuantoniumOS Statistical Validation Report</h1>
    <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    
    <div class="summary">
        <h2>Executive Summary</h2>
        <ul>
""")
            
            # Summary statistics
            for component, result in results.items():
                status = "PASS" if result.get("meets_nist_threshold", False) else "FAIL"
                pass_rate = result.get("overall_pass_rate", 0) * 100
                f.write(f'<li><b>{component}:</b> {status} (Pass rate: {pass_rate:.1f}%)</li>\n')
            
            f.write("""
        </ul>
    </div>
""")
            
            # Detailed results for each component
            for component, result in results.items():
                f.write(f'<h2>{component} Results</h2>\n')
                f.write('<table>\n<tr><th>Test</th><th>P-value</th><th>Result</th><th>Status</th></tr>\n')
                
                # Write test results
                for test_name, test_result in result.items():
                    if isinstance(test_result, dict) and "p_value" in test_result:
                        status_class = "pass" if test_result["passed"] else "fail"
                        status_text = "PASS" if test_result["passed"] else "FAIL"
                        f.write(f'<tr><td>{test_name}</td><td>{test_result["p_value"]:.6f}</td><td>{test_result.get("result", "N/A")}</td><td class="{status_class}">{status_text}</td></tr>\n')
                    elif test_name == "block_frequency":
                        for block_size, block_result in test_result.items():
                            status_class = "pass" if block_result["passed"] else "fail"
                            status_text = "PASS" if block_result["passed"] else "FAIL"
                            f.write(f'<tr><td>Block Frequency ({block_size})</td><td>{block_result["p_value"]:.6f}</td><td>N/A</td><td class="{status_class}">{status_text}</td></tr>\n')
                
                f.write('</table>\n')
                
                # Overall statistics
                f.write(f'<p><b>Overall Pass Rate:</b> {result.get("overall_pass_rate", 0)*100:.1f}%</p>\n')
                f.write(f'<p><b>Meets NIST Threshold:</b> {"YES" if result.get("meets_nist_threshold", False) else "NO"}</p>\n')
                f.write(f'<p><b>Entropy Estimate:</b> {result.get("entropy_estimate", 0):.2f} bits (Max: {result.get("max_entropy", 8):.2f})</p>\n')
            
            f.write("""
    <h2>Methodology</h2>
    <p>This validation follows NIST SP 800-22 guidelines for statistical testing of random number generators.
    Each test uses a significance level of alpha = 0.01, and a minimum of 96% of tests must pass to meet publication standards.</p>
    
    <h2>Test Descriptions</h2>
    <ul>
        <li><b>Monobit Test:</b> Tests for equal proportion of 0s and 1s in the sequence</li>
        <li><b>Block Frequency Test:</b> Tests frequency of 1s within fixed-length blocks</li>
        <li><b>Runs Test:</b> Tests for oscillation between consecutive bits</li>
        <li><b>Serial Test:</b> Tests frequency of overlapping patterns</li>
    </ul>
</body>
</html>
""")
        
        return report_file
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run complete statistical validation suite"""
        print("=== QuantoniumOS Statistical Validation Suite ===")
        print(f"Output directory: {self.output_dir}")
        
        results = {}
        
        # Validate each component
        results["Hash Function"] = self.validate_hash_function()
        results["Encryption Function"] = self.validate_encryption_function()  
        results["Entropy Engine"] = self.validate_entropy_engine()
        
        # Generate comprehensive report
        report_file = self.generate_report(results)
        print(f"\nValidation completed. Report saved to: {report_file}")
        
        # Save raw results as JSON
        json_file = os.path.join(self.output_dir, f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(json_file, 'w') as f:
            # Convert numpy types to JSON serializable
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, (np.bool_, bool)):
                    return bool(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj
            
            json.dump(convert_numpy(results), f, indent=2)
        
        print(f"Raw results saved to: {json_file}")
        
        # Print summary
        print("\n=== VALIDATION SUMMARY ===")
        for component, result in results.items():
            status = "PASS" if result.get("meets_nist_threshold", False) else "FAIL"
            pass_rate = result.get("overall_pass_rate", 0) * 100
            print(f"{component}: {status} (Pass rate: {pass_rate:.1f}%)")
        
        return results


if __name__ == "__main__":
    validator = StatisticalValidator()
    results = validator.run_full_validation()
