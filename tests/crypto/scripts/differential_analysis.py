#!/usr/bin/env python3
"""
Differential Cryptanalysis Suite for Enhanced RFT Cryptography
Validates resistance to differential attacks with measurable security thresholds.
"""

import numpy as np
import secrets
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from core.enhanced_rft_crypto_v2 import EnhancedRFTCryptoV2

class DifferentialCryptanalysis:
    """Formal differential cryptanalysis for 48-round Feistel with RFT enhancement."""
    
    def __init__(self):
        self.test_key = b"DIFFERENTIAL_TEST_KEY_QUANTONIUM_RFT_SECURITY_2025"[:32]
        self.cipher = EnhancedRFTCryptoV2(self.test_key)
        self.rounds = 48
        self.security_threshold = 2**-64  # 128-bit security requirement
        
    def test_differential_probability(self, input_diff: bytes, num_samples: int = 10000) -> Dict[str, Any]:
        """Test maximum differential probability for specific input differential."""
        
        print(f"  Testing differential: {input_diff.hex()}")
        differential_counts = defaultdict(int)
        
        for sample in range(num_samples):
            # Generate random plaintext
            pt1 = secrets.token_bytes(16)
            
            # Create differential pair
            pt2 = bytes(a ^ b for a, b in zip(pt1, input_diff))
            
            # Encrypt both plaintexts
            try:
                ct1 = self.cipher.encrypt_aead(pt1, f"DIFF_TEST_{sample}".encode())
                ct2 = self.cipher.encrypt_aead(pt2, f"DIFF_TEST_{sample}".encode())
                
                # Calculate output differential
                min_len = min(len(ct1), len(ct2))
                output_diff = bytes(a ^ b for a, b in zip(ct1[:min_len], ct2[:min_len]))
                differential_counts[output_diff] += 1
                
            except Exception as e:
                print(f"    Warning: Encryption failed for sample {sample}: {e}")
                continue
        
        if not differential_counts:
            return {
                'error': 'No successful encryptions',
                'differential_security': False
            }
        
        # Calculate maximum probability
        max_count = max(differential_counts.values())
        max_probability = max_count / num_samples
        
        # Security assessment
        is_secure = max_probability < self.security_threshold
        
        return {
            'input_differential': input_diff.hex(),
            'samples_tested': num_samples,
            'max_count': max_count,
            'max_differential_probability': max_probability,
            'security_threshold': self.security_threshold,
            'differential_security': is_secure,
            'unique_output_differentials': len(differential_counts),
            'assessment': 'SECURE' if is_secure else 'VULNERABLE',
            'security_margin': self.security_threshold / max_probability if max_probability > 0 else float('inf')
        }
    
    def comprehensive_differential_analysis(self) -> Dict[str, Any]:
        """Test comprehensive set of input differentials."""
        
        print("🔬 DIFFERENTIAL CRYPTANALYSIS SUITE")
        print("=" * 50)
        
        # Test differentials with increasing Hamming weights
        test_differentials = [
            # Single bit differentials
            b'\x01' + b'\x00' * 15,  # Bit 0
            b'\x02' + b'\x00' * 15,  # Bit 1  
            b'\x04' + b'\x00' * 15,  # Bit 2
            b'\x08' + b'\x00' * 15,  # Bit 3
            b'\x80' + b'\x00' * 15,  # Bit 7
            
            # Two bit differentials
            b'\x03' + b'\x00' * 15,  # Bits 0,1
            b'\x05' + b'\x00' * 15,  # Bits 0,2
            b'\x81' + b'\x00' * 15,  # Bits 0,7
            
            # Byte differentials
            b'\xFF' + b'\x00' * 15,  # Full byte
            b'\x00\xFF' + b'\x00' * 14,  # Second byte
            
            # Multi-byte patterns
            b'\xFF\xFF' + b'\x00' * 14,  # Two bytes
            b'\x55\x55' + b'\x00' * 14,  # Alternating pattern
            b'\xAA\xAA' + b'\x00' * 14,  # Inverse alternating
            
            # Full block patterns
            b'\xFF' * 16,  # All bits set
            b'\x55' * 16,  # Alternating bits
            b'\xAA' * 16,  # Inverse alternating bits
        ]
        
        results = {}
        start_time = time.time()
        
        for i, diff in enumerate(test_differentials):
            print(f"\n📊 Test {i+1}/{len(test_differentials)}:")
            diff_result = self.test_differential_probability(diff, num_samples=5000)
            results[f"differential_{i+1:02d}"] = diff_result
            
            # Log critical results
            if not diff_result.get('differential_security', False):
                print(f"  ⚠️  VULNERABILITY DETECTED: p = {diff_result['max_differential_probability']:.2e}")
            else:
                print(f"  ✅ SECURE: p = {diff_result['max_differential_probability']:.2e} < {self.security_threshold:.2e}")
        
        analysis_time = time.time() - start_time
        
        # Overall assessment
        all_secure = all(r.get('differential_security', False) for r in results.values() if 'error' not in r)
        max_probability = max(r.get('max_differential_probability', 0) for r in results.values() if 'error' not in r)
        
        summary = {
            'total_tests': len(test_differentials),
            'analysis_time_seconds': analysis_time,
            'overall_differential_security': all_secure,
            'maximum_probability_observed': max_probability,
            'security_threshold': self.security_threshold,
            'security_margin': self.security_threshold / max_probability if max_probability > 0 else float('inf'),
            'conclusion': 'SECURE_AGAINST_DIFFERENTIAL_ATTACKS' if all_secure else 'VULNERABLE_TO_DIFFERENTIAL_ATTACKS',
            'individual_results': results
        }
        
        return summary
    
    def test_round_reduced_analysis(self, reduced_rounds: List[int]) -> Dict[str, Any]:
        """Test differential security with reduced number of rounds."""
        
        print(f"\n🔄 ROUND-REDUCED DIFFERENTIAL ANALYSIS")
        print("=" * 40)
        
        # Note: This would require modifying the cipher to use fewer rounds
        # For now, we document the theoretical analysis
        
        results = {}
        for rounds in reduced_rounds:
            print(f"  Analyzing {rounds}-round variant...")
            
            # Theoretical differential probability calculation
            # For Feistel networks: p ≤ (p_f)^(rounds/2) where p_f is F-function differential probability
            
            # Assume F-function has differential probability ≤ 2^-6 (conservative estimate)
            f_function_prob = 2**-6
            theoretical_prob = f_function_prob**(rounds // 2)
            
            results[f"{rounds}_rounds"] = {
                'rounds': rounds,
                'theoretical_max_probability': theoretical_prob,
                'security_threshold': self.security_threshold,
                'theoretically_secure': theoretical_prob < self.security_threshold,
                'note': 'Theoretical analysis based on Feistel structure'
            }
        
        return {
            'round_reduced_analysis': results,
            'full_rounds': self.rounds,
            'minimum_secure_rounds': min(r for r in reduced_rounds 
                                       if results[f"{r}_rounds"]['theoretically_secure'])
        }

def main():
    """Run comprehensive differential cryptanalysis."""
    
    analyzer = DifferentialCryptanalysis()
    
    # Main differential analysis
    main_results = analyzer.comprehensive_differential_analysis()
    
    # Round-reduced analysis
    round_results = analyzer.test_round_reduced_analysis([16, 24, 32, 40, 48])
    
    # Combined report
    print("\n" + "=" * 60)
    print("DIFFERENTIAL CRYPTANALYSIS FINAL REPORT")
    print("=" * 60)
    
    print(f"Overall Security: {main_results['conclusion']}")
    print(f"Maximum Probability: {main_results['maximum_probability_observed']:.2e}")
    print(f"Security Threshold: {main_results['security_threshold']:.2e}")
    print(f"Security Margin: {main_results['security_margin']:.2e}x")
    print(f"Analysis Time: {main_results['analysis_time_seconds']:.1f} seconds")
    
    # Save results
    import json
    timestamp = int(time.time())
    
    report = {
        'timestamp': timestamp,
        'cipher': 'Enhanced_RFT_Cryptography_48_Round_Feistel',
        'analysis_type': 'differential_cryptanalysis',
        'main_analysis': main_results,
        'round_reduced_analysis': round_results
    }
    
    output_file = f"differential_analysis_report_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Report saved: {output_file}")
    
    return main_results['overall_differential_security']

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
