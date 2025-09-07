#!/usr/bin/env python3
"""
Avalanche Effect Analysis Script
===============================

Comprehensive testing of avalanche properties for cryptographic diffusion.
Tests both the Enhanced RFT Crypto v2 cipher and geometric waveform hashing
to ensure proper bit diffusion characteristics.

Key Metrics:
- Hamming distance analysis for single bit flips
- Statistical distribution of output changes
- Cross-correlation between input and output changes
- Avalanche coefficient calculation
"""

import sys
import os
import time
import json
import secrets
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any
from collections import defaultdict

# Add core path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'core'))

try:
    from enhanced_rft_crypto_v2 import EnhancedRFTCryptoV2
    from geometric_waveform_hash import GeometricWaveformHash
    print("✓ Successfully imported crypto modules")
except ImportError as e:
    print(f"✗ Failed to import modules: {e}")
    sys.exit(1)


class AvalancheAnalyzer:
    """Comprehensive avalanche effect analysis for cryptographic functions"""
    
    def __init__(self):
        self.test_key = b"AVALANCHE_TEST_KEY_QUANTONIUM32_"
        self.cipher = EnhancedRFTCryptoV2(self.test_key)
        self.hasher = GeometricWaveformHash()
        self.results = {}
    
    def calculate_hamming_distance(self, data1: bytes, data2: bytes) -> Tuple[int, int]:
        """Calculate Hamming distance between two byte arrays"""
        if len(data1) != len(data2):
            min_len = min(len(data1), len(data2))
            data1, data2 = data1[:min_len], data2[:min_len]
        
        hamming_distance = 0
        total_bits = len(data1) * 8
        
        for b1, b2 in zip(data1, data2):
            hamming_distance += bin(b1 ^ b2).count('1')
        
        return hamming_distance, total_bits
    
    def test_single_bit_avalanche(self, test_function, input_data: bytes, num_tests: int = 1000) -> Dict[str, Any]:
        """Test avalanche effect for single bit flips"""
        print(f"  Testing single bit avalanche ({num_tests} tests)...")
        
        avalanche_results = []
        hamming_distances = []
        
        for test_id in range(num_tests):
            # Create modified input with single bit flip
            modified_data = bytearray(input_data)
            
            # Choose random bit to flip
            byte_pos = secrets.randbelow(len(modified_data))
            bit_pos = secrets.randbelow(8)
            modified_data[byte_pos] ^= (1 << bit_pos)
            
            # Get outputs
            original_output = test_function(input_data)
            modified_output = test_function(bytes(modified_data))
            
            # Calculate avalanche
            hamming_dist, total_bits = self.calculate_hamming_distance(original_output, modified_output)
            avalanche_ratio = hamming_dist / total_bits if total_bits > 0 else 0
            
            avalanche_results.append({
                'test_id': test_id,
                'bit_position': (byte_pos, bit_pos),
                'hamming_distance': hamming_dist,
                'total_bits': total_bits,
                'avalanche_ratio': avalanche_ratio
            })
            
            hamming_distances.append(avalanche_ratio)
        
        # Statistical analysis
        mean_avalanche = np.mean(hamming_distances)
        std_avalanche = np.std(hamming_distances)
        min_avalanche = np.min(hamming_distances)
        max_avalanche = np.max(hamming_distances)
        
        # Avalanche quality assessment
        # Good avalanche: mean ≈ 0.5, small std dev
        quality_score = 1.0 - abs(mean_avalanche - 0.5) - min(std_avalanche, 0.1)
        is_good_avalanche = (0.45 <= mean_avalanche <= 0.55) and (std_avalanche <= 0.05)
        
        return {
            'statistics': {
                'mean_avalanche': mean_avalanche,
                'std_avalanche': std_avalanche,
                'min_avalanche': min_avalanche,
                'max_avalanche': max_avalanche,
                'quality_score': quality_score,
                'is_good_avalanche': is_good_avalanche
            },
            'test_results': avalanche_results[:10],  # Store first 10 for reference
            'num_tests': num_tests
        }
    
    def test_multi_bit_avalanche(self, test_function, input_data: bytes, num_tests: int = 500) -> Dict[str, Any]:
        """Test avalanche effect for multiple bit flips"""
        print(f"  Testing multi-bit avalanche ({num_tests} tests)...")
        
        multi_bit_results = []
        
        for num_flips in [1, 2, 4, 8, 16]:
            flip_results = []
            
            for test_id in range(num_tests // 5):  # Distribute tests across flip counts
                # Create modified input with multiple bit flips
                modified_data = bytearray(input_data)
                
                for _ in range(num_flips):
                    byte_pos = secrets.randbelow(len(modified_data))
                    bit_pos = secrets.randbelow(8)
                    modified_data[byte_pos] ^= (1 << bit_pos)
                
                # Get outputs
                original_output = test_function(input_data)
                modified_output = test_function(bytes(modified_data))
                
                # Calculate avalanche
                hamming_dist, total_bits = self.calculate_hamming_distance(original_output, modified_output)
                avalanche_ratio = hamming_dist / total_bits if total_bits > 0 else 0
                
                flip_results.append(avalanche_ratio)
            
            multi_bit_results.append({
                'num_flips': num_flips,
                'mean_avalanche': np.mean(flip_results),
                'std_avalanche': np.std(flip_results),
                'tests_per_category': len(flip_results)
            })
        
        return {
            'multi_bit_results': multi_bit_results,
            'num_tests_total': num_tests
        }
    
    def test_position_independence(self, test_function, input_data: bytes) -> Dict[str, Any]:
        """Test if avalanche is independent of bit position"""
        print("  Testing position independence...")
        
        position_results = defaultdict(list)
        
        # Test each bit position
        for byte_pos in range(min(len(input_data), 32)):  # Test first 32 bytes
            for bit_pos in range(8):
                # Flip specific bit
                modified_data = bytearray(input_data)
                modified_data[byte_pos] ^= (1 << bit_pos)
                
                # Get outputs
                original_output = test_function(input_data)
                modified_output = test_function(bytes(modified_data))
                
                # Calculate avalanche
                hamming_dist, total_bits = self.calculate_hamming_distance(original_output, modified_output)
                avalanche_ratio = hamming_dist / total_bits if total_bits > 0 else 0
                
                position_results[byte_pos].append(avalanche_ratio)
        
        # Analyze position independence
        position_stats = {}
        all_avalanche_values = []
        
        for byte_pos, avalanche_values in position_results.items():
            position_stats[byte_pos] = {
                'mean': np.mean(avalanche_values),
                'std': np.std(avalanche_values),
                'count': len(avalanche_values)
            }
            all_avalanche_values.extend(avalanche_values)
        
        # Calculate variance between positions
        position_means = [stats['mean'] for stats in position_stats.values()]
        position_variance = np.var(position_means)
        
        # Good position independence: low variance between positions
        is_position_independent = position_variance < 0.01
        
        return {
            'position_stats': dict(position_stats),
            'overall_stats': {
                'position_variance': position_variance,
                'is_position_independent': is_position_independent,
                'global_mean': np.mean(all_avalanche_values),
                'global_std': np.std(all_avalanche_values)
            },
            'positions_tested': len(position_results)
        }
    
    def analyze_cipher_avalanche(self) -> Dict[str, Any]:
        """Analyze avalanche properties of Enhanced RFT Crypto v2"""
        print("🌊 Analyzing Enhanced RFT Crypto v2 Avalanche...")
        
        def cipher_function(data: bytes) -> bytes:
            """Wrapper function for cipher encryption"""
            return self.cipher.encrypt_aead(data, b"AVALANCHE_TEST")
        
        # Test with different message sizes
        test_messages = [
            secrets.token_bytes(16),   # Single block
            secrets.token_bytes(32),   # Two blocks
            secrets.token_bytes(64),   # Four blocks
            secrets.token_bytes(128),  # Eight blocks
        ]
        
        cipher_results = {}
        
        for i, message in enumerate(test_messages):
            print(f"  Testing message size: {len(message)} bytes")
            
            # Single bit avalanche
            single_bit = self.test_single_bit_avalanche(cipher_function, message, 200)
            
            # Multi-bit avalanche
            multi_bit = self.test_multi_bit_avalanche(cipher_function, message, 100)
            
            # Position independence
            position_indep = self.test_position_independence(cipher_function, message)
            
            cipher_results[f"message_{len(message)}_bytes"] = {
                'single_bit_avalanche': single_bit,
                'multi_bit_avalanche': multi_bit,
                'position_independence': position_indep,
                'message_size': len(message)
            }
        
        return cipher_results
    
    def analyze_hash_avalanche(self) -> Dict[str, Any]:
        """Analyze avalanche properties of geometric waveform hash"""
        print("🔗 Analyzing Geometric Waveform Hash Avalanche...")
        
        def hash_function(data: bytes) -> bytes:
            """Wrapper function for hash computation"""
            return self.hasher.hash(data)
        
        # Test with different input sizes
        test_inputs = [
            b"Short message",
            b"Medium length message for hash testing purposes",
            b"Very long message that tests hash function behavior with extended input data" * 5,
            secrets.token_bytes(256),  # Random data
        ]
        
        hash_results = {}
        
        for i, input_data in enumerate(test_inputs):
            print(f"  Testing input size: {len(input_data)} bytes")
            
            # Single bit avalanche
            single_bit = self.test_single_bit_avalanche(hash_function, input_data, 200)
            
            # Multi-bit avalanche  
            multi_bit = self.test_multi_bit_avalanche(hash_function, input_data, 100)
            
            # Position independence
            position_indep = self.test_position_independence(hash_function, input_data)
            
            hash_results[f"input_{len(input_data)}_bytes"] = {
                'single_bit_avalanche': single_bit,
                'multi_bit_avalanche': multi_bit,
                'position_independence': position_indep,
                'input_size': len(input_data),
                'output_size': 32  # Hash always 32 bytes
            }
        
        return hash_results
    
    def generate_avalanche_vectors(self) -> Dict[str, Any]:
        """Generate test vectors for avalanche validation"""
        print("📊 Generating Avalanche Test Vectors...")
        
        vectors = []
        
        # Generate vectors for different input types
        test_cases = [
            ("zeros", b"\x00" * 32),
            ("ones", b"\xFF" * 32),
            ("alternating", bytes([0x55] * 32)),
            ("random_1", secrets.token_bytes(32)),
            ("random_2", secrets.token_bytes(32)),
        ]
        
        for case_name, input_data in test_cases:
            # Create single bit flip variant
            modified_data = bytearray(input_data)
            modified_data[0] ^= 0x01  # Flip LSB of first byte
            
            # Test cipher
            cipher_original = self.cipher.encrypt_aead(input_data, b"VECTOR_TEST")
            cipher_modified = self.cipher.encrypt_aead(bytes(modified_data), b"VECTOR_TEST")
            cipher_hamming, cipher_bits = self.calculate_hamming_distance(cipher_original, cipher_modified)
            
            # Test hash
            hash_original = self.hasher.hash(input_data)
            hash_modified = self.hasher.hash(bytes(modified_data))
            hash_hamming, hash_bits = self.calculate_hamming_distance(hash_original, hash_modified)
            
            vectors.append({
                'case_name': case_name,
                'input_hex': input_data.hex(),
                'modified_input_hex': modified_data.hex(),
                'cipher_output_hex': cipher_original.hex(),
                'cipher_modified_hex': cipher_modified.hex(),
                'cipher_hamming_distance': cipher_hamming,
                'cipher_avalanche_ratio': cipher_hamming / cipher_bits,
                'hash_output_hex': hash_original.hex(),
                'hash_modified_hex': hash_modified.hex(),
                'hash_hamming_distance': hash_hamming,
                'hash_avalanche_ratio': hash_hamming / hash_bits
            })
        
        return {
            'test_vectors': vectors,
            'generation_time': datetime.now().isoformat(),
            'purpose': 'Avalanche effect validation and regression testing'
        }
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run complete avalanche analysis"""
        print("=" * 60)
        print("AVALANCHE EFFECT ANALYSIS - COMPREHENSIVE TESTING")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run all analyses
        self.results['cipher_avalanche'] = self.analyze_cipher_avalanche()
        self.results['hash_avalanche'] = self.analyze_hash_avalanche()
        self.results['test_vectors'] = self.generate_avalanche_vectors()
        
        # Compile summary
        end_time = time.time()
        analysis_time = end_time - start_time
        
        # Determine overall quality
        cipher_quality = []
        hash_quality = []
        
        for test_name, test_data in self.results['cipher_avalanche'].items():
            cipher_quality.append(test_data['single_bit_avalanche']['statistics']['is_good_avalanche'])
        
        for test_name, test_data in self.results['hash_avalanche'].items():
            hash_quality.append(test_data['single_bit_avalanche']['statistics']['is_good_avalanche'])
        
        cipher_pass_rate = sum(cipher_quality) / len(cipher_quality) if cipher_quality else 0
        hash_pass_rate = sum(hash_quality) / len(hash_quality) if hash_quality else 0
        
        self.results['summary'] = {
            'analysis_time_seconds': analysis_time,
            'timestamp': datetime.now().isoformat(),
            'cipher_pass_rate': cipher_pass_rate,
            'hash_pass_rate': hash_pass_rate,
            'overall_avalanche_quality': (cipher_pass_rate + hash_pass_rate) / 2,
            'tests_performed': len(cipher_quality) + len(hash_quality),
            'paper_compliance': cipher_pass_rate >= 0.8  # 80% of tests should pass
        }
        
        print("\n" + "=" * 60)
        print("AVALANCHE ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Cipher Pass Rate: {cipher_pass_rate:.1%}")
        print(f"Hash Pass Rate: {hash_pass_rate:.1%}")
        print(f"Overall Quality: {self.results['summary']['overall_avalanche_quality']:.1%}")
        print(f"Analysis Time: {analysis_time:.2f} seconds")
        print(f"Paper Compliance: {'✓ YES' if self.results['summary']['paper_compliance'] else '✗ NO'}")
        
        return self.results


def main():
    """Main analysis entry point"""
    analyzer = AvalancheAnalyzer()
    results = analyzer.run_comprehensive_analysis()
    
    # Save results
    output_file = os.path.join(os.path.dirname(__file__), '..', 'results', 'avalanche_analysis_report.json')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Full results saved to: {output_file}")
    
    # Exit with appropriate code
    exit_code = 0 if results['summary']['paper_compliance'] else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
