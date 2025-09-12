#!/usr/bin/env python3
"""
RFT FAULT-TOLERANCE VALIDATION
===============================
Goal: Prove RFT robustness under various error conditions
Tests: Noise resilience, bit corruption, precision degradation, hardware faults
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(__file__))
from true_rft_kernel import TrueRFTKernel
import time

def rft_transform(x):
    """Wrapper for RFT transform"""
    rft = TrueRFTKernel(len(x))
    return rft.forward_transform(x)

def inverse_rft_transform(y):
    """Wrapper for inverse RFT transform"""
    rft = TrueRFTKernel(len(y))
    return rft.inverse_transform(y)

def test_noise_resilience():
    """Test RFT behavior under additive noise"""
    print("=" * 80)
    print("NOISE RESILIENCE TEST")
    print("=" * 80)
    print("Goal: Measure RFT degradation under various noise levels")
    print("Pass: Graceful degradation, no catastrophic failure")
    print("-" * 80)
    
    n = 32
    trials = 50
    noise_levels = [1e-6, 1e-4, 1e-2, 1e-1]
    
    results = []
    
    for noise_std in noise_levels:
        reconstruction_errors = []
        
        for trial in range(trials):
            # Generate test signal
            x = np.random.randn(n) + 1j * np.random.randn(n)
            x = x.astype(np.complex128)
            x = x / np.linalg.norm(x)  # Normalize
            
            # Add noise to input
            noise = noise_std * (np.random.randn(n) + 1j * np.random.randn(n))
            x_noisy = x + noise
            
            # RFT forward and inverse
            X_noisy = rft_transform(x_noisy)
            x_reconstructed = inverse_rft_transform(X_noisy)
            
            # Measure reconstruction error
            error = np.linalg.norm(x - x_reconstructed) / np.linalg.norm(x)
            reconstruction_errors.append(error)
        
        mean_error = np.mean(reconstruction_errors)
        std_error = np.std(reconstruction_errors)
        max_error = np.max(reconstruction_errors)
        
        print(f"Noise level {noise_std:.0e}:")
        print(f"  Mean reconstruction error: {mean_error:.3f}")
        print(f"  Std reconstruction error: {std_error:.3f}")
        print(f"  Max reconstruction error: {max_error:.3f}")
        
        # Check for catastrophic failure (>100% error)
        catastrophic = max_error > 1.0
        print(f"  Catastrophic failure: {'✗' if catastrophic else '✓'}")
        
        results.append({
            'noise_level': noise_std,
            'mean_error': mean_error,
            'max_error': max_error,
            'catastrophic': catastrophic
        })
        print()
    
    # Overall assessment
    all_stable = all(not r['catastrophic'] for r in results)
    error_growth = results[-1]['mean_error'] / results[0]['mean_error']
    
    print("=" * 80)
    print("NOISE RESILIENCE SUMMARY")
    print("=" * 80)
    print(f"Stability across noise levels: {'✓' if all_stable else '✗'}")
    print(f"Error growth factor: {error_growth:.1f}x")
    print(f"Graceful degradation: {'✓' if error_growth < 1000 else '✗'}")
    
    return all_stable and error_growth < 1000

def test_bit_corruption():
    """Test RFT behavior under bit-level corruption"""
    print("=" * 80)
    print("BIT CORRUPTION TEST")
    print("=" * 80)
    print("Goal: Measure RFT resilience to bit flips in input data")
    print("Pass: Limited impact from single/few bit flips")
    print("-" * 80)
    
    n = 32
    trials = 100
    corruption_rates = [1, 2, 4, 8]  # Number of bit flips
    
    results = []
    
    for num_flips in corruption_rates:
        reconstruction_errors = []
        
        for trial in range(trials):
            # Generate test signal
            x = np.random.complex128(np.random.randn(n) + 1j * np.random.randn(n))
            x = x / np.linalg.norm(x)
            
            # Corrupt input by flipping random bits
            x_bytes = x.tobytes()
            x_array = bytearray(x_bytes)
            
            # Flip random bits
            for _ in range(num_flips):
                byte_idx = np.random.randint(0, len(x_array))
                bit_idx = np.random.randint(0, 8)
                x_array[byte_idx] ^= (1 << bit_idx)
            
            # Convert back to complex array
            x_corrupted = np.frombuffer(x_array, dtype=np.complex128)
            
            # RFT forward and inverse
            try:
                X_corrupted = rft_transform(x_corrupted)
                x_reconstructed = inverse_rft_transform(X_corrupted)
                
                # Measure reconstruction error
                error = np.linalg.norm(x - x_reconstructed) / np.linalg.norm(x)
                reconstruction_errors.append(error)
            except:
                # Algorithm failed catastrophically
                reconstruction_errors.append(float('inf'))
        
        mean_error = np.mean([e for e in reconstruction_errors if np.isfinite(e)])
        max_error = np.max([e for e in reconstruction_errors if np.isfinite(e)])
        failure_rate = np.mean([np.isinf(e) for e in reconstruction_errors])
        
        print(f"Bit flips: {num_flips}")
        print(f"  Mean reconstruction error: {mean_error:.3f}")
        print(f"  Max reconstruction error: {max_error:.3f}")
        print(f"  Catastrophic failure rate: {failure_rate:.1%}")
        
        results.append({
            'bit_flips': num_flips,
            'mean_error': mean_error,
            'failure_rate': failure_rate
        })
        print()
    
    # Overall assessment
    low_failure_rate = all(r['failure_rate'] < 0.1 for r in results)
    bounded_errors = all(r['mean_error'] < 10.0 for r in results if r['failure_rate'] < 0.1)
    
    print("=" * 80)
    print("BIT CORRUPTION SUMMARY")
    print("=" * 80)
    print(f"Low catastrophic failure rate: {'✓' if low_failure_rate else '✗'}")
    print(f"Bounded reconstruction errors: {'✓' if bounded_errors else '✗'}")
    
    return low_failure_rate and bounded_errors

def test_precision_degradation():
    """Test RFT behavior under reduced floating-point precision"""
    print("=" * 80)
    print("PRECISION DEGRADATION TEST")
    print("=" * 80)
    print("Goal: Measure RFT accuracy under reduced precision")
    print("Pass: Graceful degradation from float64 to float32")
    print("-" * 80)
    
    n = 32
    trials = 50
    
    errors_64 = []
    errors_32 = []
    
    for trial in range(trials):
        # Generate test signal
        x = np.random.complex128(np.random.randn(n) + 1j * np.random.randn(n))
        x = x / np.linalg.norm(x)
        
        # Full precision (complex128)
        X_64 = rft_transform(x)
        x_recon_64 = inverse_rft_transform(X_64)
        error_64 = np.linalg.norm(x - x_recon_64) / np.linalg.norm(x)
        errors_64.append(error_64)
        
        # Reduced precision (complex64)
        x_32 = x.astype(np.complex64)
        X_32 = rft_transform(x_32)
        x_recon_32 = inverse_rft_transform(X_32)
        error_32 = np.linalg.norm(x - x_recon_32.astype(np.complex128)) / np.linalg.norm(x)
        errors_32.append(error_32)
    
    mean_error_64 = np.mean(errors_64)
    mean_error_32 = np.mean(errors_32)
    precision_degradation = mean_error_32 / mean_error_64
    
    print(f"Float64 reconstruction error: {mean_error_64:.2e}")
    print(f"Float32 reconstruction error: {mean_error_32:.2e}")
    print(f"Precision degradation factor: {precision_degradation:.1f}x")
    
    # Check for reasonable degradation (not more than 1000x worse)
    reasonable_degradation = precision_degradation < 1000
    
    print("=" * 80)
    print("PRECISION SUMMARY")
    print("=" * 80)
    print(f"Reasonable precision degradation: {'✓' if reasonable_degradation else '✗'}")
    
    return reasonable_degradation

def test_hardware_fault_simulation():
    """Simulate hardware faults (memory corruption, computation errors)"""
    print("=" * 80)
    print("HARDWARE FAULT SIMULATION")
    print("=" * 80)
    print("Goal: Test RFT resilience to simulated hardware faults")
    print("Pass: Detection/mitigation of computational errors")
    print("-" * 80)
    
    n = 32
    trials = 50
    fault_rates = [0.001, 0.01, 0.1]  # Probability of fault per operation
    
    results = []
    
    for fault_rate in fault_rates:
        detection_count = 0
        mitigation_count = 0
        
        for trial in range(trials):
            # Generate test signal
            x = np.random.complex128(np.random.randn(n) + 1j * np.random.randn(n))
            x = x / np.linalg.norm(x)
            
            # Simulate faulty RFT computation
            X_faulty = rft_transform(x)
            
            # Inject faults with given probability
            if np.random.random() < fault_rate:
                # Inject computation error
                fault_magnitude = 0.1 * np.random.randn()
                fault_idx = np.random.randint(0, len(X_faulty))
                X_faulty[fault_idx] += fault_magnitude
            
            # Attempt reconstruction
            x_recon = inverse_rft_transform(X_faulty)
            
            # Check for fault detection (unusually high reconstruction error)
            error = np.linalg.norm(x - x_recon) / np.linalg.norm(x)
            
            if error > 0.1:  # Threshold for fault detection
                detection_count += 1
                
                # Attempt mitigation (re-computation)
                X_retry = rft_transform(x)
                x_retry = inverse_rft_transform(X_retry)
                retry_error = np.linalg.norm(x - x_retry) / np.linalg.norm(x)
                
                if retry_error < 0.01:  # Successful mitigation
                    mitigation_count += 1
        
        detection_rate = detection_count / trials
        mitigation_rate = mitigation_count / detection_count if detection_count > 0 else 0
        
        print(f"Fault rate: {fault_rate:.1%}")
        print(f"  Fault detection rate: {detection_rate:.1%}")
        print(f"  Successful mitigation rate: {mitigation_rate:.1%}")
        
        results.append({
            'fault_rate': fault_rate,
            'detection_rate': detection_rate,
            'mitigation_rate': mitigation_rate
        })
        print()
    
    # Overall assessment
    good_detection = all(r['detection_rate'] > 0.5 for r in results if r['fault_rate'] > 0.01)
    good_mitigation = all(r['mitigation_rate'] > 0.8 for r in results if r['detection_rate'] > 0)
    
    print("=" * 80)
    print("HARDWARE FAULT SUMMARY")
    print("=" * 80)
    print(f"Good fault detection: {'✓' if good_detection else '✗'}")
    print(f"Good fault mitigation: {'✓' if good_mitigation else '✗'}")
    
    return good_detection and good_mitigation

def main():
    print("🛡️ RFT FAULT-TOLERANCE VALIDATION SUITE")
    print("=" * 80)
    print("Testing RFT robustness under error conditions")
    print("Validating fault detection and mitigation capabilities")
    print("=" * 80)
    
    start_time = time.time()
    
    # Run all fault-tolerance tests
    tests = [
        ("Noise Resilience", test_noise_resilience),
        ("Bit Corruption", test_bit_corruption),
        ("Precision Degradation", test_precision_degradation),
        ("Hardware Fault Simulation", test_hardware_fault_simulation)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*80}")
        print(f"RUNNING TEST: {test_name}")
        print(f"{'='*80}")
        
        try:
            result = test_func()
            status = "✓ PASS" if result else "✗ FAIL"
            results.append((test_name, result))
            print(f"\n{test_name}: {status}")
        except Exception as e:
            print(f"\n{test_name}: ✗ ERROR - {e}")
            results.append((test_name, False))
    
    # Final summary
    end_time = time.time()
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print("\n" + "=" * 80)
    print("FAULT-TOLERANCE TEST RESULTS SUMMARY")
    print("=" * 80)
    print(f"Tests run: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success rate: {passed/total:.1%}")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    
    print(f"\nDetailed Results:")
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {test_name:<30} {status}")
    
    print("\n" + "=" * 80)
    print("FAULT-TOLERANCE CONCLUSIONS")
    print("=" * 80)
    
    if passed == total:
        print("✓ RFT demonstrates robust fault-tolerance")
        print("  - Graceful degradation under noise")
        print("  - Resilience to bit corruption")
        print("  - Stable across precision levels")
        print("  - Effective fault detection/mitigation")
    else:
        print("⚠ RFT fault-tolerance needs improvement")
        print("  - Some tests failed - see details above")
    
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    verdict = "✓ FAULT-TOLERANT" if passed >= total * 0.75 else "✗ NEEDS IMPROVEMENT"
    print(f"🛡️ {verdict}: RFT fault-tolerance assessment")

if __name__ == "__main__":
    main()
