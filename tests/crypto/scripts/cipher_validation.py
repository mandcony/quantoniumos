#!/usr/bin/env python3
"""
Enhanced RFT Crypto v2 Validation Script
=========================================

Comprehensive validation of the 48-round Feistel cipher implementation
as described in the QuantoniumOS research paper.

Validates:
- Message avalanche (target: 0.438)
- Key avalanche (target: 0.527) 
- Key sensitivity (target: 0.495)
- Throughput (target: 9.2 MB/s)
- AEAD authenticated encryption
- Domain separation
- Paper compliance
"""

import sys
import os
import time
import json
import hashlib
import secrets
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Add core path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'core'))

try:
    from enhanced_rft_crypto_v2 import EnhancedRFTCryptoV2, CipherMetrics
    print("✓ Successfully imported Enhanced RFT Crypto v2")
except ImportError as e:
    print(f"✗ Failed to import crypto module: {e}")
    sys.exit(1)


class CryptographicValidator:
    """Comprehensive validator for Enhanced RFT Crypto v2"""
    
    def __init__(self):
        self.test_key = b"QUANTONIUMOS_VALIDATION_KEY_2025_"  # 32 bytes
        self.cipher = EnhancedRFTCryptoV2(self.test_key)
        self.results = {}
        
    def validate_paper_metrics(self) -> Dict[str, Any]:
        """Validate that implementation matches exact paper claims"""
        print("🔍 Validating Paper Metrics...")
        
        metrics = self.cipher.get_cipher_metrics()
        
        # Paper claims from research
        paper_claims = {
            'message_avalanche': 0.438,
            'key_avalanche': 0.527,
            'key_sensitivity': 0.495,
            'throughput_mbps': 9.2
        }
        
        # Tolerance for validation
        tolerance = 0.005  # ±0.5%
        
        validation = {}
        for metric, expected in paper_claims.items():
            actual = getattr(metrics, metric)
            within_tolerance = abs(actual - expected) <= tolerance
            validation[metric] = {
                'expected': expected,
                'actual': actual,
                'tolerance': tolerance,
                'pass': within_tolerance,
                'error': abs(actual - expected)
            }
            
            status = "✓ PASS" if within_tolerance else "✗ FAIL"
            print(f"  {metric}: {actual:.3f} (expect {expected:.3f}) {status}")
        
        return validation
    
    def validate_avalanche_properties(self) -> Dict[str, Any]:
        """Test avalanche effect with multiple message variations"""
        print("🌊 Testing Avalanche Properties...")
        
        test_cases = []
        num_tests = 100
        
        for i in range(num_tests):
            # Generate random test message
            message1 = secrets.token_bytes(64)
            
            # Create message with single bit flip
            message_bytes = bytearray(message1)
            flip_byte = secrets.randbelow(len(message_bytes))
            flip_bit = secrets.randbelow(8)
            message_bytes[flip_byte] ^= (1 << flip_bit)
            message2 = bytes(message_bytes)
            
            # Encrypt both messages
            encrypted1 = self.cipher.encrypt_aead(message1)
            encrypted2 = self.cipher.encrypt_aead(message2)
            
            # Calculate bit differences
            min_len = min(len(encrypted1), len(encrypted2))
            bit_diff = 0
            total_bits = min_len * 8
            
            for j in range(min_len):
                diff_byte = encrypted1[j] ^ encrypted2[j]
                bit_diff += bin(diff_byte).count('1')
            
            avalanche_ratio = bit_diff / total_bits
            test_cases.append({
                'test_id': i,
                'bit_differences': bit_diff,
                'total_bits': total_bits,
                'avalanche_ratio': avalanche_ratio,
                'message_size': len(message1),
                'ciphertext_size': len(encrypted1)
            })
        
        # Calculate statistics
        avalanche_ratios = [case['avalanche_ratio'] for case in test_cases]
        mean_avalanche = sum(avalanche_ratios) / len(avalanche_ratios)
        max_avalanche = max(avalanche_ratios)
        min_avalanche = min(avalanche_ratios)
        
        # Ideal avalanche is ~0.5 (50% of bits change)
        avalanche_quality = abs(mean_avalanche - 0.5) < 0.05  # Within 5%
        
        print(f"  Mean avalanche: {mean_avalanche:.4f}")
        print(f"  Range: [{min_avalanche:.4f}, {max_avalanche:.4f}]")
        print(f"  Quality: {'✓ GOOD' if avalanche_quality else '✗ POOR'}")
        
        return {
            'test_cases': test_cases[:10],  # Store first 10 for reference
            'statistics': {
                'mean_avalanche': mean_avalanche,
                'min_avalanche': min_avalanche,
                'max_avalanche': max_avalanche,
                'num_tests': num_tests,
                'quality_pass': avalanche_quality
            }
        }
    
    def validate_aead_functionality(self) -> Dict[str, Any]:
        """Test AEAD authenticated encryption mode"""
        print("🔐 Testing AEAD Functionality...")
        
        test_messages = [
            b"Short message",
            b"Medium length message for testing purposes",
            b"Very long message that spans multiple blocks and tests padding behavior" * 10,
            b"",  # Empty message
            bytes(range(256)),  # All byte values
        ]
        
        aead_tests = []
        
        for i, message in enumerate(test_messages):
            # Test with associated data
            associated_data = f"TEST_CASE_{i}_QUANTONIUMOS".encode()
            
            try:
                # Encrypt
                encrypted = self.cipher.encrypt_aead(message, associated_data)
                
                # Decrypt
                decrypted = self.cipher.decrypt_aead(encrypted, associated_data)
                
                # Verify round-trip
                round_trip_success = (decrypted == message)
                
                # Test tamper detection
                tampered_ciphertext = bytearray(encrypted)
                if len(tampered_ciphertext) > 50:
                    tampered_ciphertext[50] ^= 0x01  # Flip one bit
                
                tamper_detected = False
                try:
                    self.cipher.decrypt_aead(bytes(tampered_ciphertext), associated_data)
                except ValueError:
                    tamper_detected = True
                
                aead_tests.append({
                    'test_id': i,
                    'message_size': len(message),
                    'encrypted_size': len(encrypted),
                    'overhead': len(encrypted) - len(message),
                    'round_trip_success': round_trip_success,
                    'tamper_detected': tamper_detected,
                    'associated_data_size': len(associated_data)
                })
                
                status = "✓" if round_trip_success and tamper_detected else "✗"
                print(f"  Test {i}: {len(message):3d} bytes → {len(encrypted):3d} bytes {status}")
                
            except Exception as e:
                aead_tests.append({
                    'test_id': i,
                    'error': str(e),
                    'message_size': len(message)
                })
                print(f"  Test {i}: ERROR - {e}")
        
        # Summary
        successful_tests = len([t for t in aead_tests if t.get('round_trip_success', False)])
        tamper_detection_rate = len([t for t in aead_tests if t.get('tamper_detected', False)])
        
        print(f"  Success rate: {successful_tests}/{len(test_messages)}")
        print(f"  Tamper detection: {tamper_detection_rate}/{len(test_messages)}")
        
        return {
            'test_results': aead_tests,
            'summary': {
                'total_tests': len(test_messages),
                'successful_tests': successful_tests,
                'tamper_detection_rate': tamper_detection_rate,
                'overall_pass': successful_tests == len(test_messages) and tamper_detection_rate >= len(test_messages) - 1
            }
        }
    
    def validate_domain_separation(self) -> Dict[str, Any]:
        """Test domain separation in key derivation"""
        print("🔑 Testing Domain Separation...")
        
        # Test that different domains produce different keys
        master_key = b"DOMAIN_SEPARATION_TEST_KEY_32B_"
        
        cipher1 = EnhancedRFTCryptoV2(master_key)
        cipher2 = EnhancedRFTCryptoV2(master_key)
        
        # Encrypt same message with both ciphers
        test_message = b"Domain separation test message"
        
        encrypted1 = cipher1.encrypt_aead(test_message, b"DOMAIN_1")
        encrypted2 = cipher2.encrypt_aead(test_message, b"DOMAIN_2")
        
        # Check that different associated data produces different ciphertexts
        different_ciphertexts = encrypted1 != encrypted2
        
        # Test key derivation gives consistent results
        cipher1_repeat = EnhancedRFTCryptoV2(master_key)
        encrypted1_repeat = cipher1_repeat.encrypt_aead(test_message, b"DOMAIN_1")
        
        # Same domain should be decryptable
        try:
            decrypted1 = cipher1.decrypt_aead(encrypted1_repeat, b"DOMAIN_1")
            consistent_derivation = (decrypted1 == test_message)
        except:
            consistent_derivation = False
        
        print(f"  Different domains produce different outputs: {'✓' if different_ciphertexts else '✗'}")
        print(f"  Consistent key derivation: {'✓' if consistent_derivation else '✗'}")
        
        return {
            'different_ciphertexts': different_ciphertexts,
            'consistent_derivation': consistent_derivation,
            'domain_separation_working': different_ciphertexts and consistent_derivation
        }
    
    def validate_performance(self) -> Dict[str, Any]:
        """Benchmark encryption/decryption performance"""
        print("⚡ Testing Performance...")
        
        # Test different message sizes
        test_sizes = [1024, 4096, 16384, 65536]  # 1KB to 64KB
        performance_results = []
        
        for size in test_sizes:
            message = secrets.token_bytes(size)
            
            # Measure encryption
            start_time = time.perf_counter()
            encrypted = self.cipher.encrypt_aead(message)
            encrypt_time = time.perf_counter() - start_time
            
            # Measure decryption  
            start_time = time.perf_counter()
            decrypted = self.cipher.decrypt_aead(encrypted)
            decrypt_time = time.perf_counter() - start_time
            
            # Calculate throughput (bytes per second, then convert to MB/s)
            encrypt_bps = size / encrypt_time if encrypt_time > 0 else 0
            decrypt_bps = size / decrypt_time if decrypt_time > 0 else 0
            encrypt_mbps = encrypt_bps / (1024 * 1024)
            decrypt_mbps = decrypt_bps / (1024 * 1024)
            
            # The pure Python implementation achieves around 0.004 MB/s
            # Paper target is 9.2 MB/s which is for the assembly-optimized version
            # For now, just display actual throughput
            
            performance_results.append({
                'message_size': size,
                'encrypt_time_ms': encrypt_time * 1000,
                'decrypt_time_ms': decrypt_time * 1000,
                'encrypt_mbps': encrypt_mbps,
                'decrypt_mbps': decrypt_mbps,
                'round_trip_success': message == decrypted
            })
            
            print(f"  {size:5d} bytes: {encrypt_mbps:8.3f} MB/s encrypt, {decrypt_mbps:8.3f} MB/s decrypt")
        
        # Calculate average performance
        avg_encrypt_mbps = sum(r['encrypt_mbps'] for r in performance_results) / len(performance_results)
        avg_decrypt_mbps = sum(r['decrypt_mbps'] for r in performance_results) / len(performance_results)
        
        # Check against paper claim (9.2 MB/s)
        meets_paper_target = avg_encrypt_mbps >= 9.0  # Allow small margin
        
        print(f"  Average: {avg_encrypt_mbps:.3f} MB/s encrypt, {avg_decrypt_mbps:.3f} MB/s decrypt")
        print(f"  Paper target (9.2 MB/s): {'✓ MET' if meets_paper_target else '✗ BELOW (expected for pure Python)'}")
        print(f"  NOTE: The paper target refers to the assembly-optimized version, not this pure Python implementation.")
        
        return {
            'test_results': performance_results,
            'averages': {
                'encrypt_mbps': avg_encrypt_mbps,
                'decrypt_mbps': avg_decrypt_mbps,
                'meets_paper_target': meets_paper_target
            }
        }
    
    def generate_test_vectors(self) -> Dict[str, Any]:
        """Generate test vectors for regression testing"""
        print("📊 Generating Test Vectors...")
        
        # Known test vectors for reproducibility
        test_vectors = []
        
        known_keys = [
            b"TEST_KEY_1_QUANTONIUMOS_32BYTE_",
            b"TEST_KEY_2_QUANTONIUMOS_32BYTE_",
            b"ALL_ZEROS_KEY_QUANTONIUMOS_32B_",
        ]
        
        known_messages = [
            b"",
            b"a",
            b"The quick brown fox jumps over the lazy dog",
            b"QuantoniumOS test vector for research paper validation and regression testing",
            bytes(range(256))  # All possible byte values
        ]
        
        for i, key in enumerate(known_keys):
            cipher = EnhancedRFTCryptoV2(key)
            
            for j, message in enumerate(known_messages):
                associated_data = f"VECTOR_{i}_{j}".encode()
                
                encrypted = cipher.encrypt_aead(message, associated_data)
                
                test_vectors.append({
                    'vector_id': f"TV_{i}_{j}",
                    'key_hex': key.hex(),
                    'message_hex': message.hex(),
                    'associated_data_hex': associated_data.hex(),
                    'ciphertext_hex': encrypted.hex(),
                    'message_length': len(message),
                    'ciphertext_length': len(encrypted)
                })
        
        print(f"  Generated {len(test_vectors)} test vectors")
        
        return {
            'test_vectors': test_vectors,
            'generation_time': datetime.now().isoformat(),
            'cipher_config': {
                'rounds': 48,
                'block_size': 16,
                'algorithm': 'Enhanced_RFT_Crypto_v2'
            }
        }
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all validation tests and compile results"""
        print("=" * 60)
        print("ENHANCED RFT CRYPTO V2 - COMPREHENSIVE VALIDATION")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run all validation tests
        self.results['paper_metrics'] = self.validate_paper_metrics()
        self.results['avalanche_properties'] = self.validate_avalanche_properties()
        self.results['aead_functionality'] = self.validate_aead_functionality()
        self.results['domain_separation'] = self.validate_domain_separation()
        self.results['performance'] = self.validate_performance()
        self.results['test_vectors'] = self.generate_test_vectors()
        
        # Compile summary
        end_time = time.time()
        validation_time = end_time - start_time
        
        # Determine overall pass/fail
        critical_tests = [
            all(m['pass'] for m in self.results['paper_metrics'].values()),
            self.results['avalanche_properties']['statistics']['quality_pass'],
            self.results['aead_functionality']['summary']['overall_pass'],
            self.results['domain_separation']['domain_separation_working'],
            self.results['performance']['averages']['meets_paper_target']
        ]
        
        overall_pass = all(critical_tests)
        
        self.results['summary'] = {
            'validation_time_seconds': validation_time,
            'timestamp': datetime.now().isoformat(),
            'overall_pass': overall_pass,
            'critical_tests_passed': sum(critical_tests),
            'total_critical_tests': len(critical_tests),
            'paper_compliance': all(m['pass'] for m in self.results['paper_metrics'].values()),
            'ready_for_production': overall_pass
        }
        
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Overall Result: {'✓ PASS' if overall_pass else '✗ FAIL'}")
        print(f"Validation Time: {validation_time:.2f} seconds")
        print(f"Paper Compliance: {'✓ YES' if self.results['summary']['paper_compliance'] else '✗ NO'}")
        print(f"Production Ready: {'✓ YES' if overall_pass else '✗ NO'}")
        
        return self.results


def main():
    """Main validation entry point"""
    validator = CryptographicValidator()
    results = validator.run_comprehensive_validation()
    
    # Save results
    output_file = os.path.join(os.path.dirname(__file__), '..', 'results', 'latest_validation_report.json')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Full results saved to: {output_file}")
    
    # Exit with appropriate code
    exit_code = 0 if results['summary']['overall_pass'] else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
