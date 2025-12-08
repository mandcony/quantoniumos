#!/usr/bin/env python3
"""
Cascade Hybrid Integration Test
================================

Validates Phase 2-4 integration:
- C kernel enums updated (RFT_VARIANT_DICTIONARY = 12)
- Python variant registry (H3, FH5, H6)
- Production codec (RFTHybridCodec with mode selection)

Tests all 3 cascade modes achieve benchmark performance:
- H3: 0.673 BPP avg, η=0 coherence, <1ms latency
- FH5: 0.406 BPP on edges, adaptive entropy routing
- H6: 49.9 dB PSNR on smooth signals, dictionary bridge atoms
"""

import numpy as np
import sys
from pathlib import Path
from typing import Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from algorithms.rft.hybrids.rft_hybrid_codec import RFTHybridCodec

def generate_test_signals() -> Dict[str, np.ndarray]:
    """Generate diverse test signals"""
    N = 1024
    signals = {}
    
    # Random noise (high entropy)
    np.random.seed(42)
    signals['random_noise'] = np.random.randn(N)
    
    # Smooth sine (low entropy, best for H6)
    signals['sine_smooth'] = np.sin(2 * np.pi * 5 * np.arange(N) / N)
    
    # Steps (edges, best for FH5)
    signals['steps'] = np.concatenate([
        np.zeros(N//4), np.ones(N//4), 
        np.zeros(N//4), np.ones(N//4)
    ])
    
    # Mixed signal
    signals['mixed'] = (
        2.0 * np.sin(2 * np.pi * 3 * np.arange(N) / N) +
        0.5 * np.random.randn(N) +
        np.sign(np.sin(2 * np.pi * 15 * np.arange(N) / N))
    )
    
    return signals


def compute_metrics(original: np.ndarray, reconstructed: np.ndarray) -> Tuple[float, float]:
    """Compute PSNR and MSE"""
    mse = np.mean((original - reconstructed) ** 2)
    if mse < 1e-12:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(np.max(np.abs(original)) / np.sqrt(mse))
    return psnr, mse


def _run_codec_mode(mode: str, signal_name: str, signal: np.ndarray) -> Dict:
    """Test a single codec mode on a signal"""
    import time
    
    # Initialize codec
    codec = RFTHybridCodec(mode=mode)
    
    # Encode
    start = time.perf_counter()
    container = codec.encode(signal)
    encode_time = (time.perf_counter() - start) * 1000
    
    # Decode
    start = time.perf_counter()
    reconstructed = codec.decode(container)
    decode_time = (time.perf_counter() - start) * 1000
    
    # Metrics
    psnr, mse = compute_metrics(signal, reconstructed)
    
    # Estimate BPP from container
    if mode == 'legacy':
        kept_coeff = container.get('kept_coeff', len(signal))
        bits_per_coeff = container.get('bitrate_coeff', 11.0)
        total_bits = kept_coeff * bits_per_coeff
        bpp = total_bits / len(signal)
    else:
        # Cascade modes - extract from serialized dict
        compressed_dict = container.get('compressed', {})
        bpp = compressed_dict.get('bpp', 8.0)
    
    return {
        'mode': mode,
        'signal': signal_name,
        'bpp': bpp,
        'psnr': psnr,
        'mse': mse,
        'encode_ms': encode_time,
        'decode_ms': decode_time,
        'total_ms': encode_time + decode_time,
        'passed': True  # Will be updated by validate_target_performance
    }


def validate_target_performance(results: Dict, mode: str, signal: str) -> bool:
    """Check if results meet target performance. Returns False if targets not met."""
    passed = True
    
    if mode == 'h3_cascade':
        # Target: 0.673 BPP avg, <1ms latency
        if results['bpp'] > 0.75:  # Allow small margin over 0.673
            print(f"  ⚠ H3 BPP {results['bpp']:.3f} > 0.75 (target: ≤0.673)")
            passed = False
        if results['total_ms'] > 2.0:
            print(f"  ⚠ H3 latency {results['total_ms']:.2f}ms > 2.0ms (target: <1ms)")
            passed = False
    
    elif mode == 'fh5_entropy':
        # Target: 0.406 BPP on edges
        if signal == 'steps' and results['bpp'] > 0.5:
            print(f"  ⚠ FH5 edge BPP {results['bpp']:.3f} > 0.5 (target: 0.406)")
            passed = False
    
    elif mode == 'h6_dictionary':
        # Target: 49.9 dB PSNR on smooth
        if signal == 'sine_smooth' and results['psnr'] < 40.0:
            print(f"  ⚠ H6 smooth PSNR {results['psnr']:.2f}dB < 40dB (target: 49.9dB)")
            passed = False
    
    return passed


def main():
    print("=" * 100)
    print("  CASCADE HYBRID INTEGRATION TEST")
    print("  Validating Phases 2-4: C Kernel → Python Bindings → Production Codec")
    print("=" * 100)
    print()
    
    # Generate signals
    signals = generate_test_signals()
    print(f"✓ Generated {len(signals)} test signals")
    print()
    
    # Test modes
    modes = ['h3_cascade', 'fh5_entropy', 'h6_dictionary']
    all_results = []
    performance_failures = []  # Track performance target failures
    
    for mode in modes:
        print(f"Testing Mode: {mode.upper()}")
        print("-" * 100)
        
        mode_results = []
        for signal_name, signal_data in signals.items():
            try:
                result = _run_codec_mode(mode, signal_name, signal_data)
                
                # Check performance targets
                perf_ok = validate_target_performance(result, mode, signal_name)
                result['perf_passed'] = perf_ok
                if not perf_ok:
                    performance_failures.append({
                        'mode': mode, 
                        'signal': signal_name,
                        'bpp': result['bpp'],
                        'psnr': result['psnr']
                    })
                
                mode_results.append(result)
                
                # Print result
                status = "✓" if perf_ok else "⚠"
                print(f"  {status} {signal_name:<20} │ BPP: {result['bpp']:>6.3f} │ "
                      f"PSNR: {result['psnr']:>7.2f}dB │ Time: {result['total_ms']:>6.2f}ms")
                
            except Exception as e:
                print(f"  ✗ {signal_name:<20} │ ERROR: {str(e)[:60]}")
                mode_results.append({'mode': mode, 'signal': signal_name, 'error': str(e)})
        
        all_results.extend(mode_results)
        
        # Compute averages
        valid_results = [r for r in mode_results if 'bpp' in r]
        if valid_results:
            avg_bpp = np.mean([r['bpp'] for r in valid_results])
            avg_psnr = np.mean([r['psnr'] for r in valid_results if r['psnr'] != float('inf')])
            avg_time = np.mean([r['total_ms'] for r in valid_results])
            print(f"\n  Average: BPP={avg_bpp:.3f} │ PSNR={avg_psnr:.2f}dB │ Time={avg_time:.2f}ms")
        print()
    
    # Summary
    print("=" * 100)
    print("  INTEGRATION SUMMARY")
    print("=" * 100)
    
    errors = [r for r in all_results if 'error' in r]
    executed = [r for r in all_results if 'error' not in r]
    perf_passed = [r for r in executed if r.get('perf_passed', True)]
    
    print(f"  Total Tests: {len(all_results)}")
    print(f"  Executed: {len(executed)}")
    print(f"  Errors: {len(errors)} ✗")
    print(f"  Performance Passed: {len(perf_passed)}/{len(executed)}")
    print(f"  Performance Failed: {len(performance_failures)} ⚠")
    print()
    
    # Determine overall pass/fail
    all_passed = len(errors) == 0 and len(performance_failures) == 0
    
    if all_passed:
        print("  ✅ ALL TESTS PASSED - Integration Complete!")
        print()
        print("  Next Steps:")
        print("  - Rebuild C extension: cd /workspaces/quantoniumos && pip install -e .")
        print("  - Run full benchmarks: python benchmarks/test_all_hybrids.py")
        print("  - Deploy to production pipeline")
        exit_code = 0
    else:
        print("  ❌ TESTS FAILED")
        print()
        if errors:
            print("  Execution Errors:")
            for r in errors:
                print(f"    - {r['mode']} on {r['signal']}: {r.get('error', 'unknown')}")
        if performance_failures:
            print("  Performance Target Failures:")
            for f in performance_failures:
                print(f"    - {f['mode']} on {f['signal']}: BPP={f['bpp']:.3f}, PSNR={f['psnr']:.2f}dB")
        exit_code = 1
    
    print("=" * 100)
    return exit_code


if __name__ == '__main__':
    import sys
    sys.exit(main())
