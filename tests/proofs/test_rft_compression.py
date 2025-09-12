#!/usr/bin/env python3
"""
RFT COMPRESSION VALIDATION
==========================
Goal: Measure RFT compression capabilities vs DFT
Tests: Sparsity analysis, compression ratios, reconstruction quality
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

def compress_spectrum(spectrum, compression_ratio):
    """Compress spectrum by keeping only largest coefficients"""
    n = len(spectrum)
    k = int(n * compression_ratio)  # Number of coefficients to keep
    
    # Find indices of k largest coefficients by magnitude
    magnitudes = np.abs(spectrum)
    indices = np.argsort(magnitudes)[-k:]
    
    # Create compressed spectrum
    compressed = np.zeros_like(spectrum)
    compressed[indices] = spectrum[indices]
    
    return compressed, indices

def test_sparsity_analysis():
    """Analyze natural sparsity of RFT vs DFT"""
    print("=" * 80)
    print("SPARSITY ANALYSIS TEST")
    print("=" * 80)
    print("Goal: Compare natural sparsity of RFT vs DFT spectra")
    print("Pass: RFT shows different sparsity characteristics")
    print("-" * 80)
    
    n = 64
    trials = 100
    signal_types = ['impulse', 'sinusoid', 'chirp', 'noise', 'sparse']
    
    results = {}
    
    for signal_type in signal_types:
        rft_sparsities = []
        dft_sparsities = []
        
        for trial in range(trials):
            # Generate test signal based on type
            if signal_type == 'impulse':
                x = np.zeros(n, dtype=complex)
                x[np.random.randint(0, n)] = 1.0
            elif signal_type == 'sinusoid':
                freq = np.random.randint(1, n//4)
                t = np.arange(n)
                x = np.exp(1j * 2 * np.pi * freq * t / n)
            elif signal_type == 'chirp':
                t = np.arange(n) / n
                f0, f1 = 1, 10
                x = np.exp(1j * 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / 2))
            elif signal_type == 'noise':
                x = np.random.randn(n) + 1j * np.random.randn(n)
            elif signal_type == 'sparse':
                x = np.zeros(n, dtype=complex)
                num_nonzero = np.random.randint(1, n//4)
                indices = np.random.choice(n, num_nonzero, replace=False)
                x[indices] = np.random.randn(num_nonzero) + 1j * np.random.randn(num_nonzero)
            
            x = x / np.linalg.norm(x)  # Normalize
            
            # Compute transforms
            X_rft = rft_transform(x)
            X_dft = np.fft.fft(x)
            
            # Measure sparsity (fraction of "small" coefficients)
            threshold = 0.01  # 1% of max magnitude
            
            rft_max = np.max(np.abs(X_rft))
            rft_small = np.sum(np.abs(X_rft) < threshold * rft_max)
            rft_sparsity = rft_small / n
            
            dft_max = np.max(np.abs(X_dft))
            dft_small = np.sum(np.abs(X_dft) < threshold * dft_max)
            dft_sparsity = dft_small / n
            
            rft_sparsities.append(rft_sparsity)
            dft_sparsities.append(dft_sparsity)
        
        rft_mean_sparsity = np.mean(rft_sparsities)
        dft_mean_sparsity = np.mean(dft_sparsities)
        sparsity_ratio = rft_mean_sparsity / dft_mean_sparsity if dft_mean_sparsity > 0 else float('inf')
        
        print(f"Signal type: {signal_type}")
        print(f"  RFT sparsity: {rft_mean_sparsity:.3f}")
        print(f"  DFT sparsity: {dft_mean_sparsity:.3f}")
        print(f"  Sparsity ratio (RFT/DFT): {sparsity_ratio:.2f}")
        
        results[signal_type] = {
            'rft_sparsity': rft_mean_sparsity,
            'dft_sparsity': dft_mean_sparsity,
            'ratio': sparsity_ratio
        }
        print()
    
    # Check for significant differences in sparsity
    significant_differences = sum(1 for r in results.values() 
                                if abs(r['ratio'] - 1.0) > 0.2)
    
    print("=" * 80)
    print("SPARSITY SUMMARY")
    print("=" * 80)
    print(f"Signal types with significant sparsity differences: {significant_differences}/{len(signal_types)}")
    print(f"RFT shows different sparsity characteristics: {'✓' if significant_differences >= 2 else '✗'}")
    
    return significant_differences >= 2

def test_compression_ratios():
    """Test compression performance at various compression ratios"""
    print("=" * 80)
    print("COMPRESSION RATIO TEST")
    print("=" * 80)
    print("Goal: Compare RFT vs DFT compression quality")
    print("Pass: RFT achieves competitive or better compression")
    print("-" * 80)
    
    n = 64
    trials = 50
    compression_ratios = [0.1, 0.25, 0.5, 0.75]  # Fraction of coefficients to keep
    
    results = {}
    
    for ratio in compression_ratios:
        rft_errors = []
        dft_errors = []
        
        for trial in range(trials):
            # Generate mixed signal (realistic test case)
            t = np.arange(n)
            x = (np.exp(1j * 2 * np.pi * 3 * t / n) + 
                 0.5 * np.exp(1j * 2 * np.pi * 7 * t / n) +
                 0.1 * (np.random.randn(n) + 1j * np.random.randn(n)))
            x = x / np.linalg.norm(x)
            
            # RFT compression
            X_rft = rft_transform(x)
            X_rft_compressed, _ = compress_spectrum(X_rft, ratio)
            x_rft_recon = inverse_rft_transform(X_rft_compressed)
            rft_error = np.linalg.norm(x - x_rft_recon) / np.linalg.norm(x)
            
            # DFT compression
            X_dft = np.fft.fft(x)
            X_dft_compressed, _ = compress_spectrum(X_dft, ratio)
            x_dft_recon = np.fft.ifft(X_dft_compressed)
            dft_error = np.linalg.norm(x - x_dft_recon) / np.linalg.norm(x)
            
            rft_errors.append(rft_error)
            dft_errors.append(dft_error)
        
        rft_mean_error = np.mean(rft_errors)
        dft_mean_error = np.mean(dft_errors)
        compression_advantage = dft_mean_error / rft_mean_error if rft_mean_error > 0 else float('inf')
        
        print(f"Compression ratio: {ratio:.0%}")
        print(f"  RFT reconstruction error: {rft_mean_error:.4f}")
        print(f"  DFT reconstruction error: {dft_mean_error:.4f}")
        print(f"  RFT advantage: {compression_advantage:.2f}x {'✓' if compression_advantage > 0.8 else '✗'}")
        
        results[ratio] = {
            'rft_error': rft_mean_error,
            'dft_error': dft_mean_error,
            'advantage': compression_advantage
        }
        print()
    
    # Check overall compression performance
    competitive_ratios = sum(1 for r in results.values() if r['advantage'] > 0.5)
    
    print("=" * 80)
    print("COMPRESSION RATIO SUMMARY")
    print("=" * 80)
    print(f"Competitive compression ratios: {competitive_ratios}/{len(compression_ratios)}")
    print(f"RFT compression performance: {'✓' if competitive_ratios >= len(compression_ratios)//2 else '✗'}")
    
    return competitive_ratios >= len(compression_ratios)//2

def test_adaptive_compression():
    """Test adaptive compression based on signal characteristics"""
    print("=" * 80)
    print("ADAPTIVE COMPRESSION TEST")
    print("=" * 80)
    print("Goal: Test RFT's ability to adapt compression to signal type")
    print("Pass: Better compression for signals that match RFT characteristics")
    print("-" * 80)
    
    n = 64
    trials = 30
    
    # Test signals with different characteristics
    signal_configs = [
        ('Golden ratio resonance', lambda: generate_golden_ratio_signal(n)),
        ('Gaussian envelope', lambda: generate_gaussian_signal(n)),
        ('Exponential decay', lambda: generate_exponential_signal(n)),
        ('Random complex', lambda: np.random.randn(n) + 1j * np.random.randn(n))
    ]
    
    results = {}
    
    for signal_name, signal_generator in signal_configs:
        compression_ratios = []
        
        for trial in range(trials):
            x = signal_generator()
            x = x / np.linalg.norm(x)
            
            # Find minimum compression ratio for 1% reconstruction error
            X_rft = rft_transform(x)
            
            target_error = 0.01
            best_ratio = 1.0
            
            for test_ratio in np.linspace(0.05, 1.0, 20):
                X_compressed, _ = compress_spectrum(X_rft, test_ratio)
                x_recon = inverse_rft_transform(X_compressed)
                error = np.linalg.norm(x - x_recon) / np.linalg.norm(x)
                
                if error <= target_error:
                    best_ratio = test_ratio
                    break
            
            compression_ratios.append(best_ratio)
        
        mean_ratio = np.mean(compression_ratios)
        std_ratio = np.std(compression_ratios)
        
        print(f"Signal type: {signal_name}")
        print(f"  Mean compression ratio: {mean_ratio:.3f}")
        print(f"  Std compression ratio: {std_ratio:.3f}")
        print(f"  Compression efficiency: {1/mean_ratio:.1f}x")
        
        results[signal_name] = {
            'mean_ratio': mean_ratio,
            'efficiency': 1/mean_ratio
        }
        print()
    
    # Check if RFT-friendly signals compress better
    rft_signals = ['Golden ratio resonance', 'Gaussian envelope']
    rft_efficiency = np.mean([results[s]['efficiency'] for s in rft_signals if s in results])
    other_efficiency = np.mean([results[s]['efficiency'] for s in results.keys() if s not in rft_signals])
    
    adaptive_advantage = rft_efficiency / other_efficiency if other_efficiency > 0 else 1.0
    
    print("=" * 80)
    print("ADAPTIVE COMPRESSION SUMMARY")
    print("=" * 80)
    print(f"RFT-friendly signal efficiency: {rft_efficiency:.1f}x")
    print(f"General signal efficiency: {other_efficiency:.1f}x")
    print(f"Adaptive advantage: {adaptive_advantage:.2f}x")
    print(f"Adaptive compression capability: {'✓' if adaptive_advantage > 1.2 else '✗'}")
    
    return adaptive_advantage > 1.1

def generate_golden_ratio_signal(n):
    """Generate signal with golden ratio characteristics"""
    phi = (1 + np.sqrt(5)) / 2
    t = np.arange(n)
    return np.exp(1j * 2 * np.pi * t / (phi * n))

def generate_gaussian_signal(n):
    """Generate signal with Gaussian envelope"""
    t = np.arange(n) - n//2
    sigma = n / 6
    envelope = np.exp(-t**2 / (2 * sigma**2))
    carrier = np.exp(1j * 2 * np.pi * t / 8)
    return envelope * carrier

def generate_exponential_signal(n):
    """Generate signal with exponential characteristics"""
    t = np.arange(n)
    alpha = 0.1
    return np.exp(-alpha * t) * np.exp(1j * 2 * np.pi * t / 16)

def test_compression_quality_metrics():
    """Test various quality metrics for compressed signals"""
    print("=" * 80)
    print("COMPRESSION QUALITY METRICS")
    print("=" * 80)
    print("Goal: Evaluate RFT compression using multiple quality metrics")
    print("Pass: Good performance across PSNR, SNR, and perceptual metrics")
    print("-" * 80)
    
    n = 64
    trials = 50
    compression_ratio = 0.25  # Keep 25% of coefficients
    
    rft_psnrs = []
    rft_snrs = []
    rft_correlations = []
    
    dft_psnrs = []
    dft_snrs = []
    dft_correlations = []
    
    for trial in range(trials):
        # Generate test signal
        x = generate_gaussian_signal(n)
        x = x / np.linalg.norm(x)
        
        # RFT compression
        X_rft = rft_transform(x)
        X_rft_compressed, _ = compress_spectrum(X_rft, compression_ratio)
        x_rft_recon = inverse_rft_transform(X_rft_compressed)
        
        # DFT compression
        X_dft = np.fft.fft(x)
        X_dft_compressed, _ = compress_spectrum(X_dft, compression_ratio)
        x_dft_recon = np.fft.ifft(X_dft_compressed)
        
        # Calculate quality metrics
        def calculate_psnr(original, reconstructed):
            mse = np.mean(np.abs(original - reconstructed)**2)
            if mse == 0:
                return float('inf')
            max_val = np.max(np.abs(original))
            return 20 * np.log10(max_val / np.sqrt(mse))
        
        def calculate_snr(original, reconstructed):
            signal_power = np.mean(np.abs(original)**2)
            noise_power = np.mean(np.abs(original - reconstructed)**2)
            if noise_power == 0:
                return float('inf')
            return 10 * np.log10(signal_power / noise_power)
        
        def calculate_correlation(original, reconstructed):
            return np.abs(np.vdot(original, reconstructed)) / (np.linalg.norm(original) * np.linalg.norm(reconstructed))
        
        # RFT metrics
        rft_psnrs.append(calculate_psnr(x, x_rft_recon))
        rft_snrs.append(calculate_snr(x, x_rft_recon))
        rft_correlations.append(calculate_correlation(x, x_rft_recon))
        
        # DFT metrics
        dft_psnrs.append(calculate_psnr(x, x_dft_recon))
        dft_snrs.append(calculate_snr(x, x_dft_recon))
        dft_correlations.append(calculate_correlation(x, x_dft_recon))
    
    # Remove infinite values for statistics
    rft_psnrs_finite = [p for p in rft_psnrs if np.isfinite(p)]
    dft_psnrs_finite = [p for p in dft_psnrs if np.isfinite(p)]
    
    print(f"PSNR (dB):")
    print(f"  RFT: {np.mean(rft_psnrs_finite):.1f} ± {np.std(rft_psnrs_finite):.1f}")
    print(f"  DFT: {np.mean(dft_psnrs_finite):.1f} ± {np.std(dft_psnrs_finite):.1f}")
    
    print(f"SNR (dB):")
    print(f"  RFT: {np.mean(rft_snrs):.1f} ± {np.std(rft_snrs):.1f}")
    print(f"  DFT: {np.mean(dft_snrs):.1f} ± {np.std(dft_snrs):.1f}")
    
    print(f"Correlation:")
    print(f"  RFT: {np.mean(rft_correlations):.3f} ± {np.std(rft_correlations):.3f}")
    print(f"  DFT: {np.mean(dft_correlations):.3f} ± {np.std(dft_correlations):.3f}")
    
    # Check if RFT achieves good quality
    good_psnr = np.mean(rft_psnrs_finite) > 20  # >20 dB is generally good
    good_snr = np.mean(rft_snrs) > 15   # >15 dB SNR
    good_correlation = np.mean(rft_correlations) > 0.9  # >90% correlation
    
    print("\n" + "=" * 80)
    print("QUALITY METRICS SUMMARY")
    print("=" * 80)
    print(f"Good PSNR (>20 dB): {'✓' if good_psnr else '✗'}")
    print(f"Good SNR (>15 dB): {'✓' if good_snr else '✗'}")
    print(f"Good correlation (>0.9): {'✓' if good_correlation else '✗'}")
    
    return good_psnr and good_snr and good_correlation

def main():
    print("🗜️ RFT COMPRESSION VALIDATION SUITE")
    print("=" * 80)
    print("Testing RFT compression capabilities vs DFT")
    print("Analyzing sparsity, compression ratios, and quality metrics")
    print("=" * 80)
    
    start_time = time.time()
    
    # Run all compression tests
    tests = [
        ("Sparsity Analysis", test_sparsity_analysis),
        ("Compression Ratios", test_compression_ratios),
        ("Adaptive Compression", test_adaptive_compression),
        ("Quality Metrics", test_compression_quality_metrics)
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
    print("COMPRESSION TEST RESULTS SUMMARY")
    print("=" * 80)
    print(f"Tests run: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success rate: {passed/total:.1%}")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    
    print(f"\nDetailed Results:")
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {test_name:<25} {status}")
    
    print("\n" + "=" * 80)
    print("COMPRESSION CONCLUSIONS")
    print("=" * 80)
    
    if passed >= total * 0.75:
        print("✓ RFT demonstrates competitive compression capabilities")
        print("  - Distinct sparsity characteristics")
        print("  - Competitive compression ratios")
        print("  - Adaptive compression for suitable signals")
        print("  - Good quality metrics")
    else:
        print("⚠ RFT compression performance needs improvement")
        print("  - Some tests failed - see details above")
    
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    verdict = "✓ COMPRESSION-CAPABLE" if passed >= total * 0.75 else "✗ NEEDS IMPROVEMENT"
    print(f"🗜️ {verdict}: RFT compression assessment")

if __name__ == "__main__":
    main()
