import pytest
#!/usr/bin/env python3
"""
RFT Round-trip Test Suite
Tests the mathematical correctness of forward and inverse RFT operations
"""

import sys
import os
import math
import json
from typing import List, Tuple, Dict, Any

# Add core modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

try:
    from encryption.resonance_fourier import (
        resonance_fourier_transform, 
        inverse_resonance_fourier_transform,
        perform_rft_list,
        perform_irft_list
    )
    RFT_AVAILABLE = True
except ImportError:
    RFT_AVAILABLE = False

class RFTRoundTripTest:
    """Test suite for RFT mathematical correctness"""
    
    def __init__(self):
        self.test_results = []
        self.tolerance = 1e-10  # High precision for mathematical correctness
        
    def generate_test_signals(self) -> List[Tuple[str, List[float]]]:
        """Generate various test signals for comprehensive validation"""
        test_signals = []
        
        # Test 1: Simple sine wave
        sine_wave = [math.sin(2 * math.pi * i / 16) for i in range(16)]
        test_signals.append(("sine_wave", sine_wave))
        
        # Test 2: Cosine wave
        cosine_wave = [math.cos(2 * math.pi * i / 16) for i in range(16)]
        test_signals.append(("cosine_wave", cosine_wave))
        
        # Test 3: Delta function (impulse)
        delta = [1.0 if i == 0 else 0.0 for i in range(16)]
        test_signals.append(("delta_function", delta))
        
        # Test 4: Step function
        step = [1.0 if i >= 8 else 0.0 for i in range(16)]
        test_signals.append(("step_function", step))
        
        # Test 5: Random-like signal
        random_signal = [0.5, -0.3, 0.8, -0.1, 0.4, -0.7, 0.2, -0.9, 
                        0.6, -0.4, 0.1, -0.8, 0.3, -0.5, 0.7, -0.2]
        test_signals.append(("random_signal", random_signal))
        
        # Test 6: Constant signal
        constant = [0.5] * 16
        test_signals.append(("constant_signal", constant))
        
        # Test 7: Linear ramp
        linear_ramp = [i / 15.0 for i in range(16)]
        test_signals.append(("linear_ramp", linear_ramp))
        
        # Test 8: Complex waveform
        complex_wave = [math.sin(2 * math.pi * i / 8) + 0.5 * math.cos(2 * math.pi * i / 4) 
                       for i in range(16)]
        test_signals.append(("complex_waveform", complex_wave))
        
        return test_signals
    
    def calculate_mse(self, original: List[float], reconstructed: List[float]) -> float:
        """Calculate Mean Squared Error between original and reconstructed signals"""
        if len(original) != len(reconstructed):
            return float('inf')
        
        mse = sum((a - b) ** 2 for a, b in zip(original, reconstructed)) / len(original)
        return mse
    
    def test_basic_rft_roundtrip(self):
        """Test basic RFT round-trip with low-level functions"""
        if not RFT_AVAILABLE:
            self.test_results.append({
                'test': 'basic_rft_roundtrip',
                'status': 'SKIPPED',
                'reason': 'RFT modules not available'
            })
            return
        
        test_signals = self.generate_test_signals()
        
        for signal_name, signal in test_signals:
            try:
                # Forward transform
                frequency_components = resonance_fourier_transform(signal)
                
                # Inverse transform
                reconstructed = inverse_resonance_fourier_transform(frequency_components)
                
                # Calculate error
                mse = self.calculate_mse(signal, reconstructed)
                
                # Test passes if MSE is below tolerance
                passed = mse < self.tolerance
                
                self.test_results.append({
                    'test': f'basic_rft_roundtrip_{signal_name}',
                    'status': 'PASS' if passed else 'FAIL',
                    'mse': mse,
                    'tolerance': self.tolerance,
                    'original_length': len(signal),
                    'reconstructed_length': len(reconstructed),
                    'frequency_components': len(frequency_components)
                })
                
            except Exception as e:
                self.test_results.append({
                    'test': f'basic_rft_roundtrip_{signal_name}',
                    'status': 'ERROR',
                    'error': str(e)
                })
    
    def test_high_level_rft_roundtrip(self):
        """Test high-level RFT round-trip with perform_rft/perform_irft"""
        if not RFT_AVAILABLE:
            self.test_results.append({
                'test': 'high_level_rft_roundtrip',
                'status': 'SKIPPED',
                'reason': 'RFT modules not available'
            })
            return
        
        test_signals = self.generate_test_signals()
        
        for signal_name, signal in test_signals:
            try:
                # Forward transform using high-level API
                rft_result = perform_rft_list(signal)
                
                # Inverse transform using high-level API
                reconstructed = perform_irft_list(rft_result)
                
                # Calculate error
                mse = self.calculate_mse(signal, reconstructed)
                
                # Test passes if MSE is below tolerance
                passed = mse < self.tolerance
                
                self.test_results.append({
                    'test': f'high_level_rft_roundtrip_{signal_name}',
                    'status': 'PASS' if passed else 'FAIL',
                    'mse': mse,
                    'tolerance': self.tolerance,
                    'original_length': len(signal),
                    'reconstructed_length': len(reconstructed),
                    'rft_components': len(rft_result)
                })
                
            except Exception as e:
                self.test_results.append({
                    'test': f'high_level_rft_roundtrip_{signal_name}',
                    'status': 'ERROR',
                    'error': str(e)
                })
    
    def test_parseval_theorem(self):
        """Test Parseval's theorem: energy conservation in frequency domain"""
        if not RFT_AVAILABLE:
            self.test_results.append({
                'test': 'parseval_theorem',
                'status': 'SKIPPED',
                'reason': 'RFT modules not available'
            })
            return
        
        test_signals = self.generate_test_signals()
        
        for signal_name, signal in test_signals:
            try:
                # Calculate time-domain energy
                time_energy = sum(x ** 2 for x in signal)
                
                # Forward transform
                frequency_components = resonance_fourier_transform(signal)
                
                # Calculate frequency-domain energy
                freq_energy = sum(abs(complex_val) ** 2 for _, complex_val in frequency_components)
                
                # Energy difference (should be close to 0)
                energy_diff = abs(time_energy - freq_energy)
                
                # Test passes if energy is conserved within tolerance
                passed = energy_diff < self.tolerance
                
                self.test_results.append({
                    'test': f'parseval_theorem_{signal_name}',
                    'status': 'PASS' if passed else 'FAIL',
                    'time_energy': time_energy,
                    'freq_energy': freq_energy,
                    'energy_difference': energy_diff,
                    'tolerance': self.tolerance
                })
                
            except Exception as e:
                self.test_results.append({
                    'test': f'parseval_theorem_{signal_name}',
                    'status': 'ERROR',
                    'error': str(e)
                })
    
    def test_linearity_property(self):
        """Test linearity: RFT(a*x + b*y) = a*RFT(x) + b*RFT(y)"""
        if not RFT_AVAILABLE:
            self.test_results.append({
                'test': 'linearity_property',
                'status': 'SKIPPED',
                'reason': 'RFT modules not available'
            })
            return
        
        try:
            # Create two test signals
            signal1 = [math.sin(2 * math.pi * i / 8) for i in range(16)]
            signal2 = [math.cos(2 * math.pi * i / 8) for i in range(16)]
            
            # Scaling factors
            a, b = 2.0, 3.0
            
            # Combined signal
            combined = [a * x + b * y for x, y in zip(signal1, signal2)]
            
            # Transform combined signal
            rft_combined = resonance_fourier_transform(combined)
            
            # Transform individual signals
            rft_signal1 = resonance_fourier_transform(signal1)
            rft_signal2 = resonance_fourier_transform(signal2)
            
            # Calculate linear combination of transforms
            rft_linear = [(f1, a * c1 + b * c2) for (f1, c1), (f2, c2) in zip(rft_signal1, rft_signal2)]
            
            # Compare results
            max_diff = max(abs(c1 - c2) for (_, c1), (_, c2) in zip(rft_combined, rft_linear))
            
            passed = max_diff < self.tolerance
            
            self.test_results.append({
                'test': 'linearity_property',
                'status': 'PASS' if passed else 'FAIL',
                'max_difference': max_diff,
                'tolerance': self.tolerance,
                'scaling_factors': [a, b]
            })
            
        except Exception as e:
            self.test_results.append({
                'test': 'linearity_property',
                'status': 'ERROR',
                'error': str(e)
            })
    
    def run_all_tests(self):
        """Run all RFT round-trip tests"""
        print("Running RFT Round-trip Test Suite...")
        print("=" * 50)
        
        # Run all test methods
        self.test_basic_rft_roundtrip()
        self.test_high_level_rft_roundtrip()
        self.test_parseval_theorem()
        self.test_linearity_property()
        
        # Calculate summary statistics
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['status'] == 'PASS')
        failed_tests = sum(1 for result in self.test_results if result['status'] == 'FAIL')
        error_tests = sum(1 for result in self.test_results if result['status'] == 'ERROR')
        skipped_tests = sum(1 for result in self.test_results if result['status'] == 'SKIPPED')
        
        print(f"Test Results Summary:")
        print(f"  Total: {total_tests}")
        print(f"  Passed: {passed_tests}")
        print(f"  Failed: {failed_tests}")
        print(f"  Errors: {error_tests}")
        print(f"  Skipped: {skipped_tests}")
        
        # Print individual test results
        print("\nDetailed Results:")
        for result in self.test_results:
            status = result['status']
            test_name = result['test']
            print(f"  {test_name}: {status}")
            
            if status == 'PASS' and 'mse' in result:
                print(f"    MSE: {result['mse']:.2e}")
            elif status == 'FAIL' and 'mse' in result:
                print(f"    MSE: {result['mse']:.2e} (tolerance: {result['tolerance']:.2e})")
            elif status == 'ERROR':
                print(f"    Error: {result['error']}")
            elif status == 'SKIPPED':
                print(f"    Reason: {result['reason']}")
        
        # Save results to JSON file
        with open('rft_roundtrip_test_results.json', 'w') as f:
            json.dump({
                'summary': {
                    'total': total_tests,
                    'passed': passed_tests,
                    'failed': failed_tests,
                    'errors': error_tests,
                    'skipped': skipped_tests,
                    'success_rate': passed_tests / total_tests if total_tests > 0 else 0
                },
                'results': self.test_results
            }, f, indent=2)
        
        print(f"\nResults saved to: rft_roundtrip_test_results.json")
        
        # Return success if all non-skipped tests passed
        return failed_tests == 0 and error_tests == 0

def main():
    """Run RFT round-trip tests"""
    tester = RFTRoundTripTest()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ All RFT round-trip tests passed!")
        return 0
    else:
        print("\n❌ Some RFT round-trip tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())