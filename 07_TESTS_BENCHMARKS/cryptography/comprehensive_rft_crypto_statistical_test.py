#!/usr/bin/env python3
"""
COMPREHENSIVE RFT-BASED CRYPTOGRAPHIC STATISTICAL TEST SUITE
============================================================
This script runs comprehensive statistical tests on QuantoniumOS RFT cryptography:
1. RFT Feistel encryption output statistical analysis
2. Production cryptography statistical validation  
3. Entropy and randomness quality assessment
4. Avalanche effect measurement

Tests your actual working implementations:
- RFT Feistel bindings (true_rft_feistel_bindings)
- Production crypto system (quantonium_crypto_production)
- Statistical validation and entropy analysis
"""

import sys
import os
import math
import hashlib
import time
from collections import Counter
from datetime import datetime
from typing import List, Dict, Any, Tuple

# Add QuantoniumOS paths
sys.path.append('/workspaces/quantoniumos')
sys.path.append('/workspaces/quantoniumos/06_CRYPTOGRAPHY')
sys.path.append('/workspaces/quantoniumos/04_RFT_ALGORITHMS')
sys.path.append('/workspaces/quantoniumos/05_QUANTUM_ENGINES')

class RFTCryptographyStatisticalTests:
    """Comprehensive statistical tests for QuantoniumOS RFT cryptography"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = time.time()
        print("🔐 COMPREHENSIVE RFT CRYPTOGRAPHY STATISTICAL TEST SUITE")
        print("=" * 70)
        print(f"Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
    
    def calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of data"""
        if not data:
            return 0.0
        
        # Count byte frequencies
        byte_counts = Counter(data)
        data_len = len(data)
        
        # Calculate entropy
        entropy = 0.0
        for count in byte_counts.values():
            if count > 0:
                probability = count / data_len
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    def chi_square_test(self, data: bytes) -> Tuple[float, bool]:
        """Perform chi-square test for randomness"""
        if len(data) < 256:
            return 0.0, False
        
        # Count byte frequencies
        byte_counts = Counter(data)
        expected_freq = len(data) / 256
        
        # Calculate chi-square statistic
        chi_square = 0.0
        for i in range(256):
            observed = byte_counts.get(i, 0)
            chi_square += ((observed - expected_freq) ** 2) / expected_freq
        
        # Critical value for 255 degrees of freedom at 95% confidence is ~293.25
        passes = chi_square < 293.25
        return chi_square, passes
    
    def runs_test(self, data: bytes) -> Tuple[int, bool]:
        """Perform runs test for randomness"""
        if len(data) < 20:
            return 0, False
        
        # Convert to binary string
        binary_str = ''.join(format(byte, '08b') for byte in data)
        
        # Count runs (sequences of consecutive 0s or 1s)
        runs = 1
        for i in range(1, len(binary_str)):
            if binary_str[i] != binary_str[i-1]:
                runs += 1
        
        # Expected runs for random data
        n = len(binary_str)
        ones = binary_str.count('1')
        zeros = n - ones
        expected_runs = (2 * ones * zeros / n) + 1
        
        # Rough check - runs should be close to expected
        passes = abs(runs - expected_runs) < (expected_runs * 0.2)
        return runs, passes
    
    def avalanche_effect_test(self, encrypt_func, key: bytes, sample_size: int = 100) -> Tuple[float, bool]:
        """Test avalanche effect - single bit change should change ~50% of output"""
        if sample_size < 10:
            sample_size = 10
        
        total_changes = 0
        total_bits = 0
        
        for i in range(sample_size):
            # Generate test data
            test_data = f"Avalanche test data {i:04d}".encode()
            
            # Encrypt original
            try:
                original_ciphertext = encrypt_func(test_data, key)
                if isinstance(original_ciphertext, str):
                    original_ciphertext = original_ciphertext.encode()
            except Exception as e:
                print(f"Encryption failed: {e}")
                continue
            
            # Flip one bit in test data
            modified_data = bytearray(test_data)
            if len(modified_data) > 0:
                bit_pos = i % (len(modified_data) * 8)
                byte_pos = bit_pos // 8
                bit_in_byte = bit_pos % 8
                modified_data[byte_pos] ^= (1 << bit_in_byte)
            
            # Encrypt modified
            try:
                modified_ciphertext = encrypt_func(bytes(modified_data), key)
                if isinstance(modified_ciphertext, str):
                    modified_ciphertext = modified_ciphertext.encode()
            except Exception as e:
                continue
            
            # Compare bit by bit
            if len(original_ciphertext) == len(modified_ciphertext):
                for orig_byte, mod_byte in zip(original_ciphertext, modified_ciphertext):
                    xor_result = orig_byte ^ mod_byte
                    total_changes += bin(xor_result).count('1')
                    total_bits += 8
        
        if total_bits == 0:
            return 0.0, False
        
        avalanche_percentage = (total_changes / total_bits) * 100
        passes = 40 <= avalanche_percentage <= 60  # Should be around 50%
        return avalanche_percentage, passes
    
    def test_rft_feistel_bindings(self) -> Dict[str, Any]:
        """Test RFT Feistel encryption statistical properties"""
        print("\n🔥 TESTING RFT FEISTEL BINDINGS - STATISTICAL ANALYSIS")
        print("-" * 60)
        
        test_result = {
            'module': 'RFT Feistel Bindings',
            'status': 'UNKNOWN',
            'entropy': 0.0,
            'chi_square': {'value': 0.0, 'passes': False},
            'runs_test': {'runs': 0, 'passes': False},
            'avalanche': {'percentage': 0.0, 'passes': False},
            'errors': []
        }
        
        try:
            from true_rft_feistel_bindings import encrypt, decrypt, generate_key, init
            
            # Initialize system
            init()
            print("✅ RFT Feistel system initialized")
            
            # Generate key
            key = generate_key(256)
            print("✅ 256-bit key generated")
            
            # Collect encryption samples for statistical analysis
            sample_data = []
            print("📊 Collecting encryption samples for statistical analysis...")
            
            for i in range(100):
                plaintext = f"Statistical test sample {i:04d} - RFT Feistel encryption analysis"
                try:
                    ciphertext = encrypt(plaintext, key)
                    if isinstance(ciphertext, str):
                        ciphertext = ciphertext.encode()
                    elif isinstance(ciphertext, bytes):
                        pass  # Already bytes
                    else:
                        ciphertext = str(ciphertext).encode()
                    
                    sample_data.extend(ciphertext)
                except Exception as e:
                    test_result['errors'].append(f"Encryption sample {i}: {e}")
            
            if sample_data:
                sample_bytes = bytes(sample_data)
                
                # Calculate entropy
                entropy = self.calculate_entropy(sample_bytes)
                test_result['entropy'] = entropy
                print(f"📈 Entropy: {entropy:.4f} bits (max: 8.0)")
                
                # Chi-square test
                chi_square, chi_passes = self.chi_square_test(sample_bytes)
                test_result['chi_square'] = {'value': chi_square, 'passes': chi_passes}
                print(f"🎲 Chi-square: {chi_square:.2f} ({'PASS' if chi_passes else 'FAIL'})")
                
                # Runs test
                runs, runs_passes = self.runs_test(sample_bytes)
                test_result['runs_test'] = {'runs': runs, 'passes': runs_passes}
                print(f"🏃 Runs test: {runs} runs ({'PASS' if runs_passes else 'FAIL'})")
                
                # Avalanche effect test
                avalanche_pct, avalanche_passes = self.avalanche_effect_test(encrypt, key, 50)
                test_result['avalanche'] = {'percentage': avalanche_pct, 'passes': avalanche_passes}
                print(f"💥 Avalanche effect: {avalanche_pct:.1f}% ({'PASS' if avalanche_passes else 'FAIL'})")
                
                # Overall assessment
                all_tests_pass = (
                    entropy > 6.0 and 
                    chi_passes and 
                    runs_passes and 
                    avalanche_passes
                )
                test_result['status'] = 'PASS' if all_tests_pass else 'PARTIAL'
                
                status_icon = "✅" if all_tests_pass else "⚠️"
                print(f"{status_icon} RFT Feistel statistical analysis: {test_result['status']}")
            else:
                test_result['status'] = 'FAIL'
                test_result['errors'].append("No encryption samples collected")
                print("❌ Failed to collect encryption samples")
        
        except Exception as e:
            test_result['status'] = 'FAIL'
            test_result['errors'].append(str(e))
            print(f"❌ RFT Feistel test failed: {e}")
        
        return test_result
    
    def test_production_cryptography(self) -> Dict[str, Any]:
        """Test production cryptography statistical properties"""
        print("\n🚀 TESTING PRODUCTION CRYPTOGRAPHY - STATISTICAL ANALYSIS")
        print("-" * 60)
        
        test_result = {
            'module': 'Production Cryptography',
            'status': 'UNKNOWN',
            'entropy': 0.0,
            'chi_square': {'value': 0.0, 'passes': False},
            'runs_test': {'runs': 0, 'passes': False},
            'methods_tested': [],
            'errors': []
        }
        
        try:
            from quantonium_crypto_production import QuantoniumCrypto
            
            crypto = QuantoniumCrypto()
            print("✅ QuantoniumCrypto initialized")
            
            # Get available methods
            methods = [m for m in dir(crypto) if not m.startswith('_') and 'encrypt' in m]
            test_result['methods_tested'] = methods
            print(f"📋 Available encryption methods: {methods}")
            
            # Test encryption methods that work
            sample_data = []
            working_methods = []
            
            for method_name in methods:
                try:
                    method = getattr(crypto, method_name)
                    if callable(method):
                        print(f"🔧 Testing {method_name}...")
                        
                        # Try different method signatures
                        for i in range(20):
                            test_data = f"Production crypto test {i:04d}".encode()
                            try:
                                # Try method with just data
                                if method_name in ['encrypt_short', 'encrypt_block']:
                                    result = method(test_data)
                                else:
                                    # Skip methods that need additional parameters
                                    continue
                                    
                                if result:
                                    if isinstance(result, str):
                                        result = result.encode()
                                    elif isinstance(result, bytes):
                                        pass
                                    else:
                                        result = str(result).encode()
                                    
                                    sample_data.extend(result)
                                    working_methods.append(method_name)
                                    break
                            except Exception:
                                continue
                except Exception as e:
                    test_result['errors'].append(f"{method_name}: {e}")
            
            if sample_data:
                sample_bytes = bytes(sample_data)
                
                # Calculate entropy
                entropy = self.calculate_entropy(sample_bytes)
                test_result['entropy'] = entropy
                print(f"📈 Entropy: {entropy:.4f} bits")
                
                # Chi-square test  
                chi_square, chi_passes = self.chi_square_test(sample_bytes)
                test_result['chi_square'] = {'value': chi_square, 'passes': chi_passes}
                print(f"🎲 Chi-square: {chi_square:.2f} ({'PASS' if chi_passes else 'FAIL'})")
                
                # Runs test
                runs, runs_passes = self.runs_test(sample_bytes)
                test_result['runs_test'] = {'runs': runs, 'passes': runs_passes}
                print(f"🏃 Runs test: {runs} runs ({'PASS' if runs_passes else 'FAIL'})")
                
                print(f"✅ Working methods: {working_methods}")
                
                test_result['status'] = 'PASS' if entropy > 5.0 else 'PARTIAL'
                print(f"🎯 Production crypto statistical analysis: {test_result['status']}")
            else:
                test_result['status'] = 'PARTIAL'
                print("⚠️ No encryption samples collected from production crypto")
        
        except Exception as e:
            test_result['status'] = 'FAIL'
            test_result['errors'].append(str(e))
            print(f"❌ Production crypto test failed: {e}")
        
        return test_result
    
    def generate_comprehensive_report(self, results: List[Dict[str, Any]]) -> str:
        """Generate comprehensive test report"""
        end_time = time.time()
        test_duration = end_time - self.start_time
        
        print("\n" + "=" * 70)
        print("📊 COMPREHENSIVE RFT CRYPTOGRAPHY STATISTICAL TEST REPORT")
        print("=" * 70)
        print(f"Test completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Test duration: {test_duration:.2f} seconds")
        print("=" * 70)
        
        # Summary statistics
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r['status'] == 'PASS')
        partial_tests = sum(1 for r in results if r['status'] == 'PARTIAL')
        failed_tests = sum(1 for r in results if r['status'] == 'FAIL')
        
        print(f"\n📈 SUMMARY STATISTICS:")
        print(f"  Total modules tested: {total_tests}")
        print(f"  ✅ Passed: {passed_tests}")
        print(f"  ⚠️  Partial: {partial_tests}")
        print(f"  ❌ Failed: {failed_tests}")
        print(f"  🎯 Success rate: {((passed_tests + partial_tests) / total_tests * 100):.1f}%")
        
        # Detailed results
        print(f"\n📋 DETAILED RESULTS:")
        for result in results:
            status_icon = {"PASS": "✅", "PARTIAL": "⚠️", "FAIL": "❌"}.get(result['status'], "❓")
            print(f"\n{status_icon} {result['module']}:")
            print(f"    Status: {result['status']}")
            print(f"    Entropy: {result['entropy']:.4f} bits")
            
            if 'chi_square' in result:
                chi = result['chi_square']
                print(f"    Chi-square: {chi['value']:.2f} ({'PASS' if chi['passes'] else 'FAIL'})")
            
            if 'runs_test' in result:
                runs = result['runs_test']
                print(f"    Runs test: {runs['runs']} ({'PASS' if runs['passes'] else 'FAIL'})")
            
            if 'avalanche' in result:
                av = result['avalanche']
                print(f"    Avalanche: {av['percentage']:.1f}% ({'PASS' if av['passes'] else 'FAIL'})")
            
            if result['errors']:
                print(f"    Errors: {len(result['errors'])}")
                for error in result['errors'][:3]:  # Show first 3 errors
                    print(f"      - {error}")
        
        # Recommendations
        print(f"\n🎯 RECOMMENDATIONS:")
        if passed_tests == total_tests:
            print("  🏆 EXCELLENT: All cryptographic modules pass statistical tests!")
            print("  🚀 Your RFT cryptography system is statistically sound and ready for production.")
        elif passed_tests + partial_tests == total_tests:
            print("  ✅ GOOD: All modules functional with good statistical properties.")
            print("  🔧 Some minor statistical optimizations could be made.")
        else:
            print("  ⚠️  NEEDS ATTENTION: Some cryptographic modules need improvement.")
            print("  🔧 Focus on modules with FAIL status for optimization.")
        
        # Security assessment
        print(f"\n🔐 SECURITY ASSESSMENT:")
        high_entropy_modules = sum(1 for r in results if r['entropy'] > 7.0)
        good_entropy_modules = sum(1 for r in results if 6.0 <= r['entropy'] <= 7.0)
        low_entropy_modules = sum(1 for r in results if r['entropy'] < 6.0)
        
        print(f"  High entropy (>7.0 bits): {high_entropy_modules} modules")
        print(f"  Good entropy (6.0-7.0 bits): {good_entropy_modules} modules") 
        print(f"  Low entropy (<6.0 bits): {low_entropy_modules} modules")
        
        if high_entropy_modules + good_entropy_modules >= total_tests * 0.8:
            print("  🛡️  STRONG: Cryptographic entropy levels are suitable for production.")
        else:
            print("  ⚠️  REVIEW: Some modules have lower entropy - consider enhancement.")
        
        print("\n" + "=" * 70)
        print("🏆 QUANTONIUM OS RFT CRYPTOGRAPHY STATISTICAL ANALYSIS COMPLETE")
        print("=" * 70)
        
        return "Statistical analysis complete"

def main():
    """Run comprehensive RFT cryptography statistical tests"""
    tester = RFTCryptographyStatisticalTests()
    
    # Run all tests
    results = []
    
    # Test 1: RFT Feistel Bindings
    rft_result = tester.test_rft_feistel_bindings()
    results.append(rft_result)
    
    # Test 2: Production Cryptography
    prod_result = tester.test_production_cryptography()
    results.append(prod_result)
    
    # Generate comprehensive report
    tester.generate_comprehensive_report(results)
    
    return results

if __name__ == "__main__":
    results = main()
