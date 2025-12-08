"""
Benchmark Script: H3 vs H3-ARFT
Compares the standard H3 cascade (DCT+RFT) against the new H3-ARFT (DCT+OperatorARFT).
"""

import numpy as np
import sys
import os

# Add workspace root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from algorithms.rft.hybrids.cascade_hybrids import H3HierarchicalCascade
from algorithms.rft.hybrids.h3_arft_cascade import H3ARFTCascade

def benchmark_h3_vs_arft():
    print("="*60)
    print("H3 vs H3-ARFT BENCHMARK")
    print("="*60)
    
    N = 256
    t = np.linspace(0, 1, N)
    phi = (1 + np.sqrt(5)) / 2
    
    # Test Signal: Golden Quasi-Periodic (The "Home Turf" for RFT)
    signal = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 10 * phi * t)
    
    # 1. Run Standard H3
    print("\n[1] Running Standard H3 (DCT + RFT)...")
    h3 = H3HierarchicalCascade()
    # Note: H3 implementation in repo might need adaptation to match the simplified logic
    # We'll use the encode method if available, or simulate it
    try:
        res_h3 = h3.encode(signal)
        print(f"    BPP:  {res_h3.bpp:.4f}")
        print(f"    PSNR: {res_h3.psnr:.2f} dB")
    except Exception as e:
        print(f"    Standard H3 failed: {e}")
        res_h3 = None
        
    # 2. Run H3-ARFT
    print("\n[2] Running H3-ARFT (DCT + OperatorARFT)...")
    h3_arft = H3ARFTCascade()
    res_arft = h3_arft.encode(signal)
    
    print(f"    BPP:  {res_arft.bpp:.4f}")
    print(f"    PSNR: {res_arft.psnr:.2f} dB")
    
    # 3. Comparison
    if res_h3 and res_arft:
        print("\n" + "="*60)
        print("COMPARISON RESULTS")
        print("="*60)
        
        # Compare PSNR at similar BPP (or efficiency)
        # Efficiency = PSNR / BPP
        eff_h3 = res_h3.psnr / res_h3.bpp
        eff_arft = res_arft.psnr / res_arft.bpp
        
        print(f"H3 Efficiency:      {eff_h3:.2f} dB/BPP")
        print(f"H3-ARFT Efficiency: {eff_arft:.2f} dB/BPP")
        
        gain = (eff_arft / eff_h3 - 1) * 100
        print(f"Improvement:        {gain:.2f}%")
        
        if gain > 0:
            print("✅ H3-ARFT outperforms Standard H3!")
        else:
            print("❌ H3-ARFT did not outperform Standard H3.")

if __name__ == "__main__":
    benchmark_h3_vs_arft()
