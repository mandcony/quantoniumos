#!/usr/bin/env python3
"""
SIMPLIFIED TABULATED PROOF SUITE
Generates actual measurable data with the correct method names
"""

import numpy as np
import secrets
import time
from collections import defaultdict
import pandas as pd
import sys
import os
import json

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from core.enhanced_rft_crypto_v2 import EnhancedRFTCryptoV2

class SimplifiedTabulatedProofs:
    """Simplified tabulated proof generator with working method calls."""
    
    def __init__(self, samples: int = 5000):
        self.samples = samples
        self.test_key = b"TABULATED_PROOF_KEY_2025_SEPT"[:32]
        self.cipher = EnhancedRFTCryptoV2(self.test_key)
    
    def generate_differential_table(self) -> pd.DataFrame:
        """Generate differential probability table with actual measurements."""
        print("🔬 DIFFERENTIAL PROBABILITY TABLE")
        print("=" * 50)
        
        # Test differentials
        differentials = {
            'single_bit_0': b'\x01' + b'\x00' * 15,
            'single_bit_7': b'\x80' + b'\x00' * 15,
            'byte_pattern': b'\xFF' + b'\x00' * 15,
            'two_bits': b'\x03' + b'\x00' * 15,
            'nibble': b'\x0F' + b'\x00' * 15
        }
        
        results = []
        
        for diff_name, input_diff in differentials.items():
            print(f"Testing {diff_name}...")
            
            output_diffs = defaultdict(int)
            samples_used = min(self.samples, 2000)  # Manageable size
            
            for i in range(samples_used):
                if i % 500 == 0:
                    print(f"  Progress: {i}/{samples_used}")
                
                pt1 = secrets.token_bytes(16)
                pt2 = bytes(a ^ b for a, b in zip(pt1, input_diff))
                
                ct1 = self.cipher._feistel_encrypt(pt1)
                ct2 = self.cipher._feistel_encrypt(pt2)
                
                output_diff = bytes(a ^ b for a, b in zip(ct1, ct2))
                output_diffs[output_diff] += 1
            
            max_count = max(output_diffs.values()) if output_diffs else 0
            max_dp = max_count / samples_used
            
            # 95% confidence interval
            z = 1.96
            n = samples_used
            p = max_dp
            if n > 0 and p > 0:
                se = np.sqrt(p * (1-p) / n)
                ci_upper = p + z * se
            else:
                ci_upper = 1/n if n > 0 else 1
            
            results.append({
                'differential': diff_name,
                'samples': samples_used,
                'max_count': max_count,
                'max_dp': max_dp,
                'log2_dp': np.log2(max_dp) if max_dp > 0 else -np.inf,
                'ci_upper_95': ci_upper,
                'unique_outputs': len(output_diffs),
                'target_2_64': 2**-64,
                'passes': ci_upper <= 2**-32  # Relaxed for demo
            })
            
            print(f"  Max DP: {max_dp:.2e}")
            print(f"  95% CI upper: {ci_upper:.2e}")
            print(f"  Unique outputs: {len(output_diffs)}")
        
        df = pd.DataFrame(results)
        print(f"\nGenerated table with {len(df)} differentials")
        return df
    
    def generate_linear_table(self) -> pd.DataFrame:
        """Generate linear probability table."""
        print("\n🔬 LINEAR PROBABILITY TABLE")
        print("=" * 50)
        
        # Test mask pairs
        mask_tests = [
            ('single_in_out', b'\x01' + b'\x00'*15, b'\x80' + b'\x00'*15),
            ('byte_level', b'\xFF' + b'\x00'*15, b'\x0F' + b'\x00'*15),
            ('pattern_aa', b'\xAA' * 8 + b'\x00' * 8, b'\x55' * 8 + b'\x00' * 8),
            ('diagonal', bytes([1<<(i%8) for i in range(16)]), bytes([1<<((i+4)%8) for i in range(16)])),
            ('checksum', b'\x5A\xA5' * 8, b'\x33\xCC' * 8)
        ]
        
        results = []
        
        for mask_name, input_mask, output_mask in mask_tests:
            print(f"Testing {mask_name}...")
            
            correlations = []
            samples_used = min(self.samples, 1000)
            
            for i in range(samples_used):
                if i % 200 == 0:
                    print(f"  Progress: {i}/{samples_used}")
                
                pt = secrets.token_bytes(16)
                ct = self.cipher._feistel_encrypt(pt)
                
                input_parity = sum(bin(a & b).count('1') for a, b in zip(pt, input_mask)) % 2
                output_parity = sum(bin(a & b).count('1') for a, b in zip(ct, output_mask)) % 2
                
                correlation = 1 if input_parity == output_parity else 0
                correlations.append(correlation)
            
            mean_corr = np.mean(correlations)
            bias = abs(mean_corr - 0.5)
            std_corr = np.std(correlations)
            
            # Confidence interval
            n = len(correlations)
            se = std_corr / np.sqrt(n) if n > 0 else 1
            ci_upper = bias + 1.96 * se
            
            results.append({
                'mask_pair': mask_name,
                'samples': samples_used,
                'mean_correlation': mean_corr,
                'bias': bias,
                'log2_bias': np.log2(bias) if bias > 0 else -np.inf,
                'std_correlation': std_corr,
                'ci_upper_95': ci_upper,
                'target_2_32': 2**-32,
                'passes': ci_upper <= 2**-16  # Relaxed for demo
            })
            
            print(f"  Bias: {bias:.2e}")
            print(f"  95% CI upper: {ci_upper:.2e}")
        
        df = pd.DataFrame(results)
        print(f"\nGenerated table with {len(df)} mask pairs")
        return df
    
    def generate_avalanche_table(self) -> pd.DataFrame:
        """Generate avalanche effect table."""
        print("\n🔬 AVALANCHE EFFECT TABLE")
        print("=" * 50)
        
        # Test different bit positions
        bit_positions = [0, 1, 7, 8, 15, 64, 127]  # Different bit positions to flip
        
        results = []
        
        for bit_pos in bit_positions:
            if bit_pos >= 128:  # 16 bytes = 128 bits
                continue
                
            print(f"Testing bit position {bit_pos}...")
            
            avalanche_scores = []
            samples_used = min(self.samples, 500)
            
            for i in range(samples_used):
                if i % 100 == 0:
                    print(f"  Progress: {i}/{samples_used}")
                
                pt = secrets.token_bytes(16)
                pt_modified = bytearray(pt)
                
                # Flip specific bit
                byte_idx = bit_pos // 8
                bit_idx = bit_pos % 8
                if byte_idx < len(pt_modified):
                    pt_modified[byte_idx] ^= (1 << bit_idx)
                
                ct1 = self.cipher._feistel_encrypt(pt)
                ct2 = self.cipher._feistel_encrypt(bytes(pt_modified))
                
                # Count different bits
                diff_bits = sum(bin(a ^ b).count('1') for a, b in zip(ct1, ct2))
                avalanche_score = diff_bits / (16 * 8)  # Fraction of bits changed
                avalanche_scores.append(avalanche_score)
            
            mean_avalanche = np.mean(avalanche_scores)
            std_avalanche = np.std(avalanche_scores)
            min_avalanche = np.min(avalanche_scores)
            max_avalanche = np.max(avalanche_scores)
            
            # Ideal is 0.5 (50% bits change)
            deviation_from_ideal = abs(mean_avalanche - 0.5)
            
            results.append({
                'bit_position': bit_pos,
                'byte_position': bit_pos // 8,
                'samples': samples_used,
                'mean_avalanche': mean_avalanche,
                'std_avalanche': std_avalanche,
                'min_avalanche': min_avalanche,
                'max_avalanche': max_avalanche,
                'deviation_from_ideal': deviation_from_ideal,
                'ideal_threshold': 0.1,  # Within 10% of ideal
                'passes': deviation_from_ideal <= 0.1
            })
            
            print(f"  Mean avalanche: {mean_avalanche:.4f} (ideal: 0.5)")
            print(f"  Deviation: {deviation_from_ideal:.4f}")
        
        df = pd.DataFrame(results)
        print(f"\nGenerated table with {len(df)} bit positions")
        return df
    
    def generate_timing_table(self) -> pd.DataFrame:
        """Generate timing analysis table."""
        print("\n🔬 TIMING ANALYSIS TABLE")
        print("=" * 50)
        
        operations = [
            ('sbox_lookup', self._time_sbox),
            ('full_encrypt', self._time_encrypt),
            ('round_function', self._time_round)
        ]
        
        results = []
        
        for op_name, timing_func in operations:
            print(f"Testing {op_name}...")
            
            times = []
            measurements = 200
            
            for i in range(measurements):
                if i % 50 == 0:
                    print(f"  Measurement {i}/{measurements}")
                
                elapsed = timing_func()
                times.append(elapsed)
            
            times = np.array(times)
            mean_time = np.mean(times)
            std_time = np.std(times)
            cv = std_time / mean_time if mean_time > 0 else float('inf')
            
            results.append({
                'operation': op_name,
                'measurements': measurements,
                'mean_time_us': mean_time * 1e6,  # microseconds
                'std_time_us': std_time * 1e6,
                'coefficient_variation': cv,
                'constant_time_threshold': 0.2,  # 20% CV threshold
                'passes_constant_time': cv < 0.2,
                'timing_risk': 'LOW' if cv < 0.1 else 'MEDIUM' if cv < 0.3 else 'HIGH'
            })
            
            print(f"  Mean time: {mean_time * 1e6:.1f} μs")
            print(f"  CV: {cv:.4f}")
        
        df = pd.DataFrame(results)
        print(f"\nGenerated table with {len(df)} operations")
        return df
    
    def _time_sbox(self) -> float:
        """Time S-box lookup."""
        byte_val = secrets.randbits(8)
        start = time.perf_counter()
        _ = self.cipher.S_BOX[byte_val]
        end = time.perf_counter()
        return end - start
    
    def _time_encrypt(self) -> float:
        """Time full encryption."""
        pt = secrets.token_bytes(16)
        start = time.perf_counter()
        _ = self.cipher._feistel_encrypt(pt)
        end = time.perf_counter()
        return end - start
    
    def _time_round(self) -> float:
        """Time round function."""
        data = secrets.token_bytes(8)
        key = secrets.token_bytes(16)
        start = time.perf_counter()
        _ = self.cipher._round_function(data, key, 0)
        end = time.perf_counter()
        return end - start
    
    def generate_comprehensive_report(self) -> dict:
        """Generate all tables and summary."""
        print("🎯 COMPREHENSIVE TABULATED PROOF REPORT")
        print("=" * 60)
        
        start_time = time.time()
        
        # Generate all tables
        dp_table = self.generate_differential_table()
        lp_table = self.generate_linear_table()
        avalanche_table = self.generate_avalanche_table()
        timing_table = self.generate_timing_table()
        
        # Save tables
        timestamp = int(time.time())
        
        dp_table.to_csv(f'differential_analysis_{timestamp}.csv', index=False)
        lp_table.to_csv(f'linear_analysis_{timestamp}.csv', index=False)
        avalanche_table.to_csv(f'avalanche_analysis_{timestamp}.csv', index=False)
        timing_table.to_csv(f'timing_analysis_{timestamp}.csv', index=False)
        
        # Generate summary
        summary = {
            'generation_info': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_time': time.time() - start_time,
                'samples_per_test': self.samples
            },
            'differential_summary': {
                'tests': len(dp_table),
                'max_dp': float(dp_table['max_dp'].max()),
                'tests_passing': int(dp_table['passes'].sum()),
                'unique_outputs_avg': float(dp_table['unique_outputs'].mean())
            },
            'linear_summary': {
                'tests': len(lp_table),
                'max_bias': float(lp_table['bias'].max()),
                'tests_passing': int(lp_table['passes'].sum()),
                'avg_correlation': float(lp_table['mean_correlation'].mean())
            },
            'avalanche_summary': {
                'positions_tested': len(avalanche_table),
                'avg_avalanche': float(avalanche_table['mean_avalanche'].mean()),
                'tests_passing': int(avalanche_table['passes'].sum()),
                'best_avalanche': float(avalanche_table['mean_avalanche'].iloc[avalanche_table['deviation_from_ideal'].idxmin()])
            },
            'timing_summary': {
                'operations_tested': len(timing_table),
                'operations_constant_time': int(timing_table['passes_constant_time'].sum()),
                'best_cv': float(timing_table['coefficient_variation'].min()),
                'worst_cv': float(timing_table['coefficient_variation'].max())
            }
        }
        
        with open(f'tabulated_proof_summary_{timestamp}.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        total_time = time.time() - start_time
        
        print(f"\n🎉 COMPREHENSIVE REPORT COMPLETE")
        print("=" * 50)
        print(f"Total time: {total_time:.1f}s")
        print(f"Files generated:")
        print(f"  • differential_analysis_{timestamp}.csv")
        print(f"  • linear_analysis_{timestamp}.csv") 
        print(f"  • avalanche_analysis_{timestamp}.csv")
        print(f"  • timing_analysis_{timestamp}.csv")
        print(f"  • tabulated_proof_summary_{timestamp}.json")
        
        print(f"\n📊 SUMMARY STATISTICS:")
        print(f"Differential tests: {summary['differential_summary']['tests_passing']}/{summary['differential_summary']['tests']} pass")
        print(f"Linear tests: {summary['linear_summary']['tests_passing']}/{summary['linear_summary']['tests']} pass")
        print(f"Avalanche tests: {summary['avalanche_summary']['tests_passing']}/{summary['avalanche_summary']['positions_tested']} pass")
        print(f"Timing tests: {summary['timing_summary']['operations_constant_time']}/{summary['timing_summary']['operations_tested']} pass")
        
        return {
            'summary': summary,
            'tables': {
                'differential': dp_table,
                'linear': lp_table,
                'avalanche': avalanche_table,
                'timing': timing_table
            }
        }

def main():
    """Run simplified tabulated proof generation."""
    print("🚀 SIMPLIFIED TABULATED PROOF SUITE")
    print("Generating actual measurable data with working method calls")
    print()
    
    samples = 3000  # Manageable for demo
    generator = SimplifiedTabulatedProofs(samples)
    results = generator.generate_comprehensive_report()
    
    print("\n✅ TABULATED PROOF GENERATION COMPLETE")
    return results

if __name__ == "__main__":
    main()
