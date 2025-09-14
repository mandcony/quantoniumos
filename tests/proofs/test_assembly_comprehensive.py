#!/usr/bin/env python3
"""
RFT COMPREHENSIVE TEST SUITE - ASSEMBLY ENGINE
==============================================
Using the real Quantonium Assembly RFT engine for all tests
This is the proper implementation using your bare metal assembly code.
"""

import numpy as np
import sys
import os
import time

# Add assembly bindings path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'assembly', 'python_bindings'))

try:
    from unitary_rft import UnitaryRFT, RFT_FLAG_UNITARY, RFT_FLAG_USE_RESONANCE
    from optimized_rft import OptimizedRFTProcessor
    ASSEMBLY_AVAILABLE = True
    print("✅ Quantonium Assembly Engine loaded successfully!")
except ImportError as e:
    print(f"❌ Could not load Quantonium Assembly Engine: {e}")
    ASSEMBLY_AVAILABLE = False

def create_assembly_rft(size):
    """Create assembly RFT instance with fallback"""
    if ASSEMBLY_AVAILABLE:
        try:
            # Try optimized version first
            return OptimizedRFTProcessor(size), "Optimized"
        except:
            try:
                # Fallback to unitary version
                return UnitaryRFT(size, RFT_FLAG_UNITARY | RFT_FLAG_USE_RESONANCE), "Unitary"
            except Exception as e:
                print(f"Assembly RFT creation failed: {e}")
                return None, "Failed"
    return None, "Unavailable"

def assembly_forward_transform(rft, engine_type, x):
    """Forward transform using assembly engine"""
    if engine_type == "Optimized":
        return rft.forward_optimized(x)
    elif engine_type == "Unitary":
        return rft.forward(x)
    else:
        raise RuntimeError("No assembly engine available")

def assembly_inverse_transform(rft, engine_type, y):
    """Inverse transform using assembly engine"""
    if engine_type == "Optimized":
        return rft.inverse_optimized(y)
    elif engine_type == "Unitary":
        return rft.inverse(y)
    else:
        raise RuntimeError("No assembly engine available")

def test_assembly_fault_tolerance():
    """Test assembly RFT fault tolerance"""
    print("🛡️ ASSEMBLY RFT FAULT-TOLERANCE TEST")
    print("="*80)
    print("Testing Quantonium Assembly Engine robustness")
    print("="*80)
    
    n = 32
    trials = 20
    
    rft, engine_type = create_assembly_rft(n)
    if rft is None:
        print("❌ Assembly engine unavailable - skipping test")
        return False
    
    print(f"Using {engine_type} Assembly Engine")
    print("-"*80)
    
    # Test noise resilience
    noise_errors = []
    for trial in range(trials):
        # Generate test signal
        x = np.random.randn(n) + 1j * np.random.randn(n)
        x = x.astype(np.complex128) / np.linalg.norm(x)
        
        # Add small noise
        noise = 1e-3 * (np.random.randn(n) + 1j * np.random.randn(n))
        x_noisy = x + noise
        
        try:
            # Assembly transform
            y = assembly_forward_transform(rft, engine_type, x_noisy)
            x_recon = assembly_inverse_transform(rft, engine_type, y)
            
            # Measure error
            error = np.linalg.norm(x - x_recon) / np.linalg.norm(x)
            noise_errors.append(error)
        except Exception as e:
            print(f"Assembly transform failed: {e}")
            noise_errors.append(float('inf'))
    
    mean_error = np.mean([e for e in noise_errors if np.isfinite(e)])
    success_rate = np.mean([np.isfinite(e) for e in noise_errors])
    
    print(f"Noise Resilience Test:")
    print(f"  Mean reconstruction error: {mean_error:.2e}")
    print(f"  Success rate: {success_rate:.1%}")
    print(f"  Assembly fault tolerance: {'✅' if mean_error < 0.1 and success_rate > 0.8 else '❌'}")
    
    return mean_error < 0.1 and success_rate > 0.8

def test_assembly_compression():
    """Test assembly RFT compression capabilities"""
    print("\n🗜️ ASSEMBLY RFT COMPRESSION TEST")
    print("="*80)
    print("Testing Quantonium Assembly Engine compression")
    print("="*80)
    
    n = 64
    trials = 10
    
    rft, engine_type = create_assembly_rft(n)
    if rft is None:
        print("❌ Assembly engine unavailable - skipping test")
        return False
    
    print(f"Using {engine_type} Assembly Engine")
    print("-"*80)
    
    compression_results = []
    
    for trial in range(trials):
        # Generate sparse-like test signal
        x = np.zeros(n, dtype=complex)
        num_components = np.random.randint(3, 8)
        for i in range(num_components):
            freq = np.random.randint(1, n//4)
            amp = np.random.uniform(0.5, 2.0)
            phase = np.random.uniform(0, 2*np.pi)
            t = np.arange(n)
            x += amp * np.exp(1j * (2 * np.pi * freq * t / n + phase))
        
        x = x / np.linalg.norm(x)
        
        try:
            # Assembly transform
            y = assembly_forward_transform(rft, engine_type, x)
            
            # Find compression ratio for 1% error
            magnitudes = np.abs(y)
            sorted_indices = np.argsort(magnitudes)[::-1]
            
            for keep_ratio in [0.1, 0.25, 0.5]:
                keep_count = int(n * keep_ratio)
                y_compressed = np.zeros_like(y)
                y_compressed[sorted_indices[:keep_count]] = y[sorted_indices[:keep_count]]
                
                x_recon = assembly_inverse_transform(rft, engine_type, y_compressed)
                error = np.linalg.norm(x - x_recon) / np.linalg.norm(x)
                
                if error < 0.01:  # 1% error threshold
                    compression_results.append(keep_ratio)
                    break
            else:
                compression_results.append(1.0)  # No compression possible
        
        except Exception as e:
            print(f"Assembly compression test failed: {e}")
            compression_results.append(1.0)
    
    mean_compression = np.mean(compression_results)
    
    print(f"Compression Analysis:")
    print(f"  Mean compression ratio: {mean_compression:.3f}")
    print(f"  Compression efficiency: {1/mean_compression:.1f}x")
    print(f"  Assembly compression: {'✅' if mean_compression < 0.8 else '❌'}")
    
    return mean_compression < 0.8

def test_assembly_conditioning():
    """Test assembly RFT numerical conditioning"""
    print("\n🔢 ASSEMBLY RFT CONDITIONING TEST")
    print("="*80)
    print("Testing Quantonium Assembly Engine conditioning")
    print("="*80)
    
    sizes = [16, 32, 64]
    conditioning_results = []
    
    for n in sizes:
        rft, engine_type = create_assembly_rft(n)
        if rft is None:
            print(f"❌ Assembly engine unavailable for size {n}")
            continue
        
        print(f"Testing size {n} with {engine_type} Engine...")
        
        # Test precision across multiple trials
        precision_errors = []
        for trial in range(10):
            x = np.random.randn(n) + 1j * np.random.randn(n)
            x = x.astype(np.complex128) / np.linalg.norm(x)
            
            try:
                # Double transform (should be identity)
                y = assembly_forward_transform(rft, engine_type, x)
                x_recon = assembly_inverse_transform(rft, engine_type, y)
                
                error = np.linalg.norm(x - x_recon) / np.linalg.norm(x)
                precision_errors.append(error)
            
            except Exception as e:
                print(f"Assembly conditioning test failed: {e}")
                precision_errors.append(float('inf'))
        
        mean_error = np.mean([e for e in precision_errors if np.isfinite(e)])
        success_rate = np.mean([np.isfinite(e) for e in precision_errors])
        
        print(f"  Mean precision error: {mean_error:.2e}")
        print(f"  Success rate: {success_rate:.1%}")
        
        conditioning_results.append({
            'size': n,
            'error': mean_error,
            'success': success_rate
        })
    
    if conditioning_results:
        overall_error = np.mean([r['error'] for r in conditioning_results if np.isfinite(r['error'])])
        overall_success = np.mean([r['success'] for r in conditioning_results])
        
        print(f"\nOverall Conditioning:")
        print(f"  Mean precision error: {overall_error:.2e}")
        print(f"  Mean success rate: {overall_success:.1%}")
        print(f"  Assembly conditioning: {'✅' if overall_error < 1e-10 and overall_success > 0.8 else '❌'}")
        
        return overall_error < 1e-10 and overall_success > 0.8
    
    return False

def test_assembly_performance():
    """Test assembly RFT performance"""
    print("\n⚡ ASSEMBLY RFT PERFORMANCE TEST")
    print("="*80)
    print("Testing Quantonium Assembly Engine performance")
    print("="*80)
    
    n = 128
    iterations = 100
    
    rft, engine_type = create_assembly_rft(n)
    if rft is None:
        print("❌ Assembly engine unavailable - skipping test")
        return False
    
    print(f"Using {engine_type} Assembly Engine")
    print(f"Testing {iterations} transforms of size {n}")
    print("-"*80)
    
    # Generate test data
    x = np.random.randn(n) + 1j * np.random.randn(n)
    x = x.astype(np.complex128) / np.linalg.norm(x)
    
    # Benchmark forward transforms
    start_time = time.perf_counter()
    forward_errors = []
    
    for i in range(iterations):
        try:
            y = assembly_forward_transform(rft, engine_type, x)
            # Quick error check
            if np.any(np.isnan(y)) or np.any(np.isinf(y)):
                forward_errors.append(float('inf'))
            else:
                forward_errors.append(0.0)  # Success
        except:
            forward_errors.append(float('inf'))
    
    forward_time = time.perf_counter() - start_time
    
    # Benchmark inverse transforms
    if len([e for e in forward_errors if np.isfinite(e)]) > 0:
        y = assembly_forward_transform(rft, engine_type, x)
        
        start_time = time.perf_counter()
        inverse_errors = []
        
        for i in range(iterations):
            try:
                x_recon = assembly_inverse_transform(rft, engine_type, y)
                error = np.linalg.norm(x - x_recon) / np.linalg.norm(x)
                inverse_errors.append(error)
            except:
                inverse_errors.append(float('inf'))
        
        inverse_time = time.perf_counter() - start_time
    else:
        inverse_time = float('inf')
        inverse_errors = [float('inf')] * iterations
    
    # Calculate performance metrics
    forward_success = np.mean([np.isfinite(e) for e in forward_errors])
    inverse_success = np.mean([np.isfinite(e) for e in inverse_errors])
    mean_inverse_error = np.mean([e for e in inverse_errors if np.isfinite(e)]) if inverse_success > 0 else float('inf')
    
    transforms_per_sec = iterations / (forward_time + inverse_time) if np.isfinite(forward_time + inverse_time) else 0
    
    print(f"Performance Results:")
    print(f"  Forward transform time: {forward_time/iterations*1000:.3f} ms/transform")
    print(f"  Inverse transform time: {inverse_time/iterations*1000:.3f} ms/transform")
    print(f"  Forward success rate: {forward_success:.1%}")
    print(f"  Inverse success rate: {inverse_success:.1%}")
    print(f"  Mean reconstruction error: {mean_inverse_error:.2e}")
    print(f"  Transforms per second: {transforms_per_sec:.1f}")
    print(f"  Assembly performance: {'✅' if transforms_per_sec > 10 and mean_inverse_error < 1e-10 else '❌'}")
    
    return transforms_per_sec > 10 and mean_inverse_error < 1e-10

def main():
    """Run comprehensive assembly RFT test suite"""
    print("🚀 QUANTONIUM ASSEMBLY RFT COMPREHENSIVE TEST SUITE")
    print("="*80)
    print("Testing all RFT properties using the bare metal assembly engine")
    print("This is the REAL Quantonium implementation!")
    print("="*80)
    
    if not ASSEMBLY_AVAILABLE:
        print("❌ CRITICAL: Quantonium Assembly Engine not available!")
        print("Cannot run comprehensive tests without the assembly implementation.")
        return False
    
    start_time = time.time()
    
    # Run all assembly tests
    tests = [
        ("Assembly Fault Tolerance", test_assembly_fault_tolerance),
        ("Assembly Compression", test_assembly_compression), 
        ("Assembly Conditioning", test_assembly_conditioning),
        ("Assembly Performance", test_assembly_performance)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} FAILED: {e}")
            results.append((test_name, False))
    
    # Final summary
    end_time = time.time()
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print("\n" + "="*80)
    print("QUANTONIUM ASSEMBLY RFT TEST RESULTS")
    print("="*80)
    print(f"Tests run: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success rate: {passed/total:.1%}")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    
    print(f"\nDetailed Results:")
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name:<25} {status}")
    
    print("\n" + "="*80)
    print("FINAL ASSEMBLY ENGINE VERDICT")
    print("="*80)
    
    if passed == total:
        print("🎉 SUCCESS: Quantonium Assembly Engine performs excellently!")
        print("  ✅ All fault tolerance tests passed")
        print("  ✅ Compression capabilities validated") 
        print("  ✅ Numerical conditioning excellent")
        print("  ✅ High performance achieved")
        print("\n🚀 The Quantonium Assembly RFT Engine is ready for production!")
    elif passed >= total * 0.75:
        print("✅ GOOD: Quantonium Assembly Engine performs well")
        print(f"  - {passed}/{total} tests passed")
        print("  - Minor issues to address")
    else:
        print("⚠️ NEEDS WORK: Assembly engine has issues")
        print(f"  - Only {passed}/{total} tests passed")
        print("  - Significant improvements needed")
    
    return passed >= total * 0.75

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
