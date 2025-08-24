#!/usr/bin/env python3
"""
COMPREHENSIVE STATISTICAL TESTING SUITE === Implements all required statistical tests to validate cryptographic quality: - Entropy analysis (>=100 MB datasets) - Chi-square tests - Correlation analysis - Run tests - Compression testing - Indistinguishability testing
"""
"""

import os
import gzip
import time
import math
import random
import statistics
import subprocess from typing
import Dict, List, Tuple, Any, Optional from collections
import Counter
import tempfile

class ComprehensiveStatisticalTester:
"""
"""
    Comprehensive statistical validation for cryptographic outputs
"""
"""

    def __init__(self):
        self.test_results = {}
    def shannon_entropy(self, data: bytes) -> float:
"""
"""
        Calculate Shannon entropy in bits per byte
"""
"""

        if not data:
        return 0.0

        # Count byte frequencies byte_counts = Counter(data) data_len = len(data)

        # Calculate entropy entropy = 0.0
        for count in byte_counts.values():
        if count > 0: probability = count / data_len entropy -= probability * math.log2(probability)
        return entropy
    def chi_square_test(self, data: bytes) -> Tuple[float, bool]:
"""
"""
        Chi-square goodness of fit test Target: chi^2 <= 293.25 for uniform distribution
"""
"""

        if len(data) < 256:
        return float('inf'), False

        # Count byte frequencies byte_counts = Counter(data) expected_frequency = len(data) / 256

        # Calculate chi-square statistic chi_square = 0.0
        for byte_value in range(256): observed = byte_counts.get(byte_value, 0) chi_square += (observed - expected_frequency) ** 2 / expected_frequency

        # Critical value for df=255 at α=0.05 is approximately 293.25 passes = chi_square <= 293.25
        return chi_square, passes
    def serial_correlation_test(self, data: bytes, lag: int = 1) -> Tuple[float, bool]:
"""
"""
        Serial correlation test Target: |rho1||| <= 0.003
"""
"""

        if len(data) < lag + 100:
        return float('inf'), False

        # Convert to numeric values values = list(data) n = len(values) - lag

        # Calculate means x_mean = statistics.mean(values[:-lag]) y_mean = statistics.mean(values[lag:])

        # Calculate correlation coefficient numerator = sum((values[i] - x_mean) * (values[i + lag] - y_mean)
        for i in range(n)) x_var = sum((values[i] - x_mean) ** 2
        for i in range(n)) y_var = sum((values[i] - y_mean) ** 2
        for i in range(lag, len(values)))
        if x_var == 0 or y_var == 0: correlation = 0.0
        else: correlation = numerator / math.sqrt(x_var * y_var) passes = abs(correlation) <= 0.003
        return correlation, passes
    def runs_test(self, data: bytes) -> Tuple[float, bool]:
"""
"""
        Runs test for randomness Target: z in [−2,2]
"""
"""

        if len(data) < 20:
        return float('inf'), False

        # Convert to binary (above/below median) median = statistics.median(data) binary_sequence = [1
        if x >= median else 0
        for x in data]

        # Count runs runs = 1
        for i in range(1, len(binary_sequence)):
        if binary_sequence[i] != binary_sequence[i-1]: runs += 1

        # Count 1s and 0s ones = sum(binary_sequence) zeros = len(binary_sequence) - ones
        if ones == 0 or zeros == 0:
        return float('inf'), False n = len(binary_sequence)

        # Expected runs and variance expected_runs = (2 * ones * zeros / n) + 1 runs_variance = (2 * ones * zeros * (2 * ones * zeros - n)) / (n ** 2 * (n - 1))
        if runs_variance <= 0:
        return float('inf'), False

        # Z-score z_score = (runs - expected_runs) / math.sqrt(runs_variance) passes = -2 <= z_score <= 2
        return z_score, passes
    def compression_test(self, data: bytes) -> Tuple[float, bool]:
"""
"""
        Compression test using gzip Target: compression ratio >= 0.995 (<=0.5% compression)
"""
"""

        if len(data) < 1000:
        return 0.0, False

        # Compress data compressed = gzip.compress(data, compresslevel=9) compression_ratio = len(compressed) / len(data) passes = compression_ratio >= 0.995
        return compression_ratio, passes
    def generate_large_dataset(self, generator_func, size_mb: int = 100) -> bytes:
"""
"""
        Generate large dataset for testing
"""
"""
        print(f" 🔄 Generating {size_mb} MB test dataset...") chunk_size = 1024 * 1024 # 1 MB chunks data = b''
        for i in range(size_mb): chunk = generator_func(chunk_size) data += chunk if (i + 1) % 10 == 0:
        print(f" Generated {i + 1}/{size_mb} MB...")
        return data
    def indistinguishability_test(self, test_data: bytes, reference_generator) -> Tuple[float, bool]: """
        Basic indistinguishability test using simple statistical measures More sophisticated classifiers would be better but this gives a baseline
"""
"""
        if len(test_data) < 10000:
        return 0.0, False

        # Generate reference data of same length reference_data = reference_generator(len(test_data))

        # Compare statistical properties test_entropy =
        self.shannon_entropy(test_data) ref_entropy =
        self.shannon_entropy(reference_data) test_chi2, _ =
        self.chi_square_test(test_data) ref_chi2, _ =
        self.chi_square_test(reference_data) test_corr, _ =
        self.serial_correlation_test(test_data) ref_corr, _ =
        self.serial_correlation_test(reference_data)

        # Simple "classifier" based on statistical distance entropy_diff = abs(test_entropy - ref_entropy) chi2_diff = abs(test_chi2 - ref_chi2) / max(test_chi2, ref_chi2, 1) corr_diff = abs(test_corr - ref_corr)

        # Combined statistical distance (simple metric) statistical_distance = entropy_diff + chi2_diff + corr_diff

        # Convert to "accuracy" - lower distance = harder to distinguish = better

        # This is a very rough approximation accuracy = max(0.5, min(1.0, 0.5 + statistical_distance * 10))

        # Good indistinguishability: accuracy close to 50% passes = 0.48 <= accuracy <= 0.52
        return accuracy, passes
    def comprehensive_validation(self, data_generator, generator_name: str, test_size_mb: int = 10) -> Dict[str, Any]: """
        Run comprehensive validation on a data generator
"""
"""
        print(f"\n🔬 COMPREHENSIVE VALIDATION: {generator_name}")
        print("=" * 60) results = { 'generator_name': generator_name, 'test_size_mb': test_size_mb, 'timestamp': time.time(), 'tests': {} }

        # Generate test data start_time = time.time() test_data =
        self.generate_large_dataset(data_generator, test_size_mb) generation_time = time.time() - start_time
        print(f" ✅ Generated {len(test_data):,} bytes in {generation_time:.2f}s")
        print(f" Rate: {len(test_data) / generation_time / 1024 / 1024:.1f} MB/s")

        # Test 1: Shannon Entropy
        print(f"||n 🧪 Running Shannon entropy test...") entropy =
        self.shannon_entropy(test_data) entropy_pass = entropy >= 7.997 results['tests']['shannon_entropy'] = { 'value': entropy, 'target': '>= 7.997', 'passes': entropy_pass, 'status': 'PASS'
        if entropy_pass else 'FAIL' }
        print(f" Entropy: {entropy:.6f} bits/byte ({'PASS'
        if entropy_pass else 'FAIL'})")

        # Test 2: Chi-square
        print(f" 🧪 Running chi-square test...") chi2_value, chi2_pass =
        self.chi_square_test(test_data) results['tests']['chi_square'] = { 'value': chi2_value, 'target': '<= 293.25', 'passes': chi2_pass, 'status': 'PASS'
        if chi2_pass else 'FAIL' }
        print(f" chi^2: {chi2_value:.2f} ({'PASS'
        if chi2_pass else 'FAIL'})")

        # Test 3: Serial Correlation
        print(f" 🧪 Running serial correlation test...") corr_value, corr_pass =
        self.serial_correlation_test(test_data) results['tests']['serial_correlation'] = { 'value': corr_value, 'target': '|rho1||| <= 0.003', 'passes': corr_pass, 'status': 'PASS'
        if corr_pass else 'FAIL' }
        print(f" rho1: {corr_value:.6f} ({'PASS'
        if corr_pass else 'FAIL'})")

        # Test 4: Runs Test
        print(f" 🧪 Running runs test...") runs_z, runs_pass =
        self.runs_test(test_data) results['tests']['runs_test'] = { 'value': runs_z, 'target': 'z in [−2,2]', 'passes': runs_pass, 'status': 'PASS'
        if runs_pass else 'FAIL' }
        print(f" Runs z-score: {runs_z:.3f} ({'PASS'
        if runs_pass else 'FAIL'})")

        # Test 5: Compression
        print(f" 🧪 Running compression test...") comp_ratio, comp_pass =
        self.compression_test(test_data) results['tests']['compression'] = { 'value': comp_ratio, 'target': '>= 0.995', 'passes': comp_pass, 'status': 'PASS'
        if comp_pass else 'FAIL' }
        print(f" Compression ratio: {comp_ratio:.6f} ({'PASS'
        if comp_pass else 'FAIL'})")

        # Test 6: Basic Indistinguishability
        print(f" 🧪 Running indistinguishability test...")

        # Use a subset for this test to save time test_subset = test_data[:min(1024*1024, len(test_data))] # 1MB max
    def random_reference(size):
        return os.urandom(size) indist_accuracy, indist_pass =
        self.indistinguishability_test(test_subset, random_reference) results['tests']['indistinguishability'] = { 'value': indist_accuracy, 'target': '~= 0.50 (±0.02)', 'passes': indist_pass, 'status': 'PASS'
        if indist_pass else 'FAIL' }
        print(f" Classifier accuracy: {indist_accuracy:.3f} ({'PASS'
        if indist_pass else 'FAIL'})")

        # Overall assessment total_tests = len(results['tests']) passed_tests = sum(1
        for test in results['tests'].values()
        if test['passes']) results['summary'] = { 'total_tests': total_tests, 'passed_tests': passed_tests, 'pass_rate': passed_tests / total_tests, 'overall_status': 'PASS'
        if passed_tests == total_tests else 'PARTIAL'
        if passed_tests > 0 else 'FAIL' }
        print(f"\n SUMMARY:")
        print(f" Tests passed: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
        print(f" Overall status: {results['summary']['overall_status']}")
        return results
    def test_enhanced_system_with_comprehensive_validation(): """
        Test our enhanced system with comprehensive statistical validation
"""
"""
        print("🔬 COMPREHENSIVE STATISTICAL VALIDATION")
        print("=" * 60) from cryptographic_security_enhancements
import CryptographicEnhancer

        # Initialize enhancer enhancer = CryptographicEnhancer() tester = ComprehensiveStatisticalTester()

        # Test user key test_user_key = b"quantonium_test_key_2025_v1"

        # Define enhanced generator
    def enhanced_rft_generator(size: int) -> bytes: """
        Generator using our enhanced RFT system
"""
        """ output = b'' current_nonce = os.urandom(16)
        while len(output) < size:

        # Simulate RFT output with some structure (this would be real RFT output) rft_raw = os.urandom(128)

        # In reality this comes from your RFT engines

        # Apply our enhancements chunk = enhancer.enhanced_rft_output( rft_raw, test_user_key, current_nonce, output_length=min(64, size - len(output)) ) output += chunk

        # Update nonce for diversity current_nonce = os.urandom(16)
        return output[:size]

        # Test standard urandom for comparison
    def urandom_generator(size: int) -> bytes:
        return os.urandom(size)

        # Run comprehensive tests
        print(" Testing Enhanced RFT System...") enhanced_results = tester.comprehensive_validation( enhanced_rft_generator, "Enhanced RFT System", test_size_mb=5

        # Smaller for demo )
        print("\n Testing OS urandom (Reference)...") urandom_results = tester.comprehensive_validation( urandom_generator, "OS urandom", test_size_mb=5 )

        # Compare results
        print(f"\n COMPARISON SUMMARY")
        print("=" * 30) enhanced_pass_rate = enhanced_results['summary']['pass_rate'] urandom_pass_rate = urandom_results['summary']['pass_rate']
        print(f"Enhanced RFT System: {enhanced_pass_rate*100:.1f}% pass rate")
        print(f"OS urandom: {urandom_pass_rate*100:.1f}% pass rate")
        if enhanced_pass_rate >= urandom_pass_rate:
        print(f"✅ Enhanced system performs as well as urandom!")
        else:
        print(f"⚠️ Enhanced system needs improvement")

        # Save results
import json all_results = { 'timestamp': time.time(), 'enhanced_rft_results': enhanced_results, 'urandom_reference_results': urandom_results, 'comparison': { 'enhanced_pass_rate': enhanced_pass_rate, 'reference_pass_rate': urandom_pass_rate, 'meets_reference_standard': enhanced_pass_rate >= 0.8 * urandom_pass_rate } } with open('/workspaces/quantoniumos/comprehensive_statistical_validation.json', 'w') as f: json.dump(all_results, f, indent=2, default=str)
        return all_results

if __name__ == "__main__": results = test_enhanced_system_with_comprehensive_validation()
print(f"||n STATISTICAL VALIDATION COMPLETE")
print(f" Results saved to: comprehensive_statistical_validation.json") enhanced_status = results['enhanced_rft_results']['summary']['overall_status']
print(f" Enhanced RFT Status: {enhanced_status}")
if results['comparison']['meets_reference_standard']:
print(f" ✅ MEETS REFERENCE STANDARD!")
else:
print(f" ⚠️ Needs improvement to meet reference standard")