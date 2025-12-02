#!/usr/bin/env python3
"""
Test ALL RFT Hybrid Variants
=============================

Systematically tests every hybrid RFT implementation found in the repository:
- H1-H10: hypothesis_testing/hybrid_mca_fixes.py
- FH1-FH5: ascii_wall/ascii_wall_final_hypotheses.py
- Legacy: hybrids/rft_hybrid_codec.py
- Variants: algorithms/rft/variants/

Finds the best performing hybrid across different signal types.
"""

import numpy as np
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Callable
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

@dataclass
class HybridResult:
    """Result from testing a hybrid variant"""
    name: str
    bpp: float
    psnr: float
    time_ms: float
    coherence: float
    sparsity: float
    error: float
    status: str = "OK"
    
    def __str__(self):
        return (f"{self.name:<35} │ {self.bpp:>6.3f} │ {self.psnr:>7.2f}dB │ "
                f"{self.time_ms:>8.2f}ms │ {self.coherence:>8.2e} │ {self.status}")


# ============================================================================
# Import all hybrid variants
# ============================================================================

HYBRIDS = {}

# Try to import H1-H10 from hypothesis testing
try:
    sys.path.insert(0, 'experiments/hypothesis_testing')
    from hybrid_mca_fixes import (
        baseline_greedy_hybrid,
        hypothesis1_coherence_aware,
        hypothesis2_phase_adaptive,
        hypothesis3_hierarchical_cascade,
        hypothesis4_quantum_superposition,
        hypothesis5_attention_gating,
        hypothesis6_dictionary_learning,
        hypothesis7_cascade_attention,
        hypothesis8_aggressive_cascade,
        hypothesis9_iterative_refinement,
        hypothesis10_quality_cascade
    )
    
    HYBRIDS.update({
        'H0_Baseline_Greedy': baseline_greedy_hybrid,
        'H1_Coherence_Aware': hypothesis1_coherence_aware,
        'H2_Phase_Adaptive': hypothesis2_phase_adaptive,
        'H3_Hierarchical_Cascade': hypothesis3_hierarchical_cascade,
        'H4_Quantum_Superposition': hypothesis4_quantum_superposition,
        'H5_Attention_Gating': hypothesis5_attention_gating,
        'H6_Dictionary_Learning': hypothesis6_dictionary_learning,
        'H7_Cascade_Attention': hypothesis7_cascade_attention,
        'H8_Aggressive_Cascade': hypothesis8_aggressive_cascade,
        'H9_Iterative_Refinement': hypothesis9_iterative_refinement,
        'H10_Quality_Cascade': hypothesis10_quality_cascade,
    })
    print(f"✓ Loaded H1-H10 (11 variants)")
except Exception as e:
    print(f"✗ Could not load H1-H10: {e}")

# Try to import FH1-FH5 from ascii wall final hypotheses
try:
    sys.path.insert(0, 'experiments/ascii_wall')
    from ascii_wall_final_hypotheses import (
        fh1_multilevel_cascade,
        fh2_adaptive_split_cascade,
        fh3_frequency_cascade,
        fh4_edge_aware_cascade,
        fh5_entropy_guided_cascade
    )
    
    # Wrap FH variants to match ExperimentResult interface
    def wrap_fh(fh_func):
        def wrapper(signal, target_sparsity=0.95):
            result = fh_func(signal, sparsity=target_sparsity)
            # Convert FinalResult to ExperimentResult-like object
            class ExperimentResult:
                def __init__(self, fr):
                    self.name = fr.hypothesis
                    self.bpp = fr.bpp
                    self.psnr = fr.psnr_db
                    self.time_ms = fr.time_ms
                    self.coherence_violation = fr.coherence_violation
                    self.sparsity_pct = fr.sparsity_pct
                    self.reconstruction_error = fr.reconstruction_error
            return ExperimentResult(result)
        return wrapper
    
    HYBRIDS.update({
        'FH1_MultiLevel_Cascade': wrap_fh(fh1_multilevel_cascade),
        'FH2_Adaptive_Split': wrap_fh(fh2_adaptive_split_cascade),
        'FH3_Frequency_Cascade': wrap_fh(fh3_frequency_cascade),
        'FH4_Edge_Aware': wrap_fh(fh4_edge_aware_cascade),
        'FH5_Entropy_Guided': wrap_fh(fh5_entropy_guided_cascade),
    })
    print(f"✓ Loaded FH1-FH5 (5 variants)")
except Exception as e:
    print(f"✗ Could not load FH1-FH5: {e}")

# Try to import legacy hybrid codec
try:
    from algorithms.rft.hybrids.rft_hybrid_codec import RFTHybridCodec
    
    def legacy_hybrid_wrapper(signal, target_sparsity=0.95):
        """Wrap legacy codec"""
        start = time.perf_counter()
        codec = RFTHybridCodec()
        
        # Encode/decode
        try:
            compressed = codec.encode(signal, quality=0.95)
            reconstructed = codec.decode(compressed)
            elapsed = (time.perf_counter() - start) * 1000
            
            # Compute metrics
            mse = np.mean((signal - reconstructed)**2)
            psnr = 20 * np.log10(np.max(np.abs(signal)) / np.sqrt(mse)) if mse > 0 else float('inf')
            
            class ExperimentResult:
                def __init__(self):
                    self.name = "Legacy_Hybrid_Codec"
                    self.bpp = len(compressed) * 8 / len(signal)
                    self.psnr = psnr
                    self.time_ms = elapsed
                    self.coherence_violation = 0.0  # Unknown
                    self.sparsity_pct = 95.0
                    self.reconstruction_error = np.linalg.norm(signal - reconstructed)
            return ExperimentResult()
        except Exception as e:
            raise Exception(f"Legacy codec failed: {e}")
    
    HYBRIDS['Legacy_Hybrid_Codec'] = legacy_hybrid_wrapper
    print(f"✓ Loaded Legacy Hybrid Codec (1 variant)")
except Exception as e:
    print(f"✗ Could not load legacy codec: {e}")

print(f"\nTotal hybrids loaded: {len(HYBRIDS)}")
print()


# ============================================================================
# Test Signals
# ============================================================================

def generate_test_signals(n=1024) -> Dict[str, np.ndarray]:
    """Generate diverse test signals"""
    signals = {}
    
    # Random noise
    signals['random_noise'] = np.random.randn(n)
    
    # Sine wave (smooth)
    t = np.linspace(0, 10*np.pi, n)
    signals['sine_smooth'] = np.sin(t) + 0.3*np.sin(3*t)
    
    # ASCII-like (discrete jumps)
    signals['ascii_code'] = np.array([ord(c) for c in ('x=42;' * 200)[:n]], dtype=float)
    
    # Step function (discontinuous)
    signals['steps'] = np.zeros(n)
    for i in range(0, n, n//8):
        signals['steps'][i:i+n//16] = np.random.randn() * 10
    
    # Chirp (frequency sweep)
    signals['chirp'] = np.sin(t * t / 10)
    
    # Sparse impulses
    signals['sparse_impulse'] = np.zeros(n)
    impulse_indices = np.arange(0, n, n//20)
    signals['sparse_impulse'][impulse_indices] = np.random.randn(len(impulse_indices)) * 5
    
    # Mixed smooth+edges
    signals['mixed'] = signals['sine_smooth'].copy()
    signals['mixed'][n//4:n//4+10] += 20  # Add sharp edge
    signals['mixed'][3*n//4:3*n//4+10] -= 15
    
    # Fibonacci-like (structured)
    fib = [1, 1]
    for i in range(n-2):
        fib.append((fib[-1] + fib[-2]) % 256)
    signals['fibonacci'] = np.array(fib[:n], dtype=float)
    
    return signals


# ============================================================================
# Benchmark Runner
# ============================================================================

def test_hybrid(hybrid_func: Callable, signal: np.ndarray, name: str) -> HybridResult:
    """Test a single hybrid variant"""
    try:
        start = time.perf_counter()
        result = hybrid_func(signal, target_sparsity=0.95)
        elapsed = (time.perf_counter() - start) * 1000
        
        return HybridResult(
            name=name,
            bpp=result.bpp,
            psnr=result.psnr,
            time_ms=result.time_ms if hasattr(result, 'time_ms') else elapsed,
            coherence=result.coherence_violation,
            sparsity=result.sparsity_pct,
            error=result.reconstruction_error,
            status="✓"
        )
    except Exception as e:
        return HybridResult(
            name=name,
            bpp=999.0,
            psnr=-999.0,
            time_ms=0.0,
            coherence=999.0,
            sparsity=0.0,
            error=999.0,
            status=f"✗ {str(e)[:30]}"
        )


def run_comprehensive_benchmark():
    """Run all hybrids on all signals"""
    
    print("=" * 100)
    print("  COMPREHENSIVE HYBRID RFT BENCHMARK")
    print("  Testing ALL hybrid variants across diverse signals")
    print("=" * 100)
    print()
    
    if len(HYBRIDS) == 0:
        print("ERROR: No hybrids loaded! Check imports.")
        return
    
    signals = generate_test_signals(1024)
    
    print(f"Testing {len(HYBRIDS)} hybrids on {len(signals)} signal types\n")
    
    # Results storage
    all_results = {}
    
    for signal_name, signal in signals.items():
        print(f"{'━' * 100}")
        print(f"  SIGNAL: {signal_name.upper()} (N={len(signal)})")
        print(f"{'━' * 100}")
        print()
        print(f"  {'Hybrid Variant':<35} │ {'BPP':<6} │ {'PSNR':<8} │ {'Time':<9} │ Coherence │ Status")
        print(f"  {'-' * 98}")
        
        signal_results = []
        
        for hybrid_name, hybrid_func in HYBRIDS.items():
            result = test_hybrid(hybrid_func, signal, hybrid_name)
            signal_results.append(result)
            print(f"  {result}")
        
        all_results[signal_name] = signal_results
        print()
    
    # ========================================================================
    # SUMMARY ANALYSIS
    # ========================================================================
    
    print("=" * 100)
    print("  SUMMARY: BEST PERFORMING HYBRIDS")
    print("=" * 100)
    print()
    
    # Find best by BPP for each signal type
    print("  BEST COMPRESSION (Lowest BPP):")
    print(f"  {'-' * 98}")
    for signal_name, results in all_results.items():
        valid_results = [r for r in results if r.status == "✓"]
        if valid_results:
            best = min(valid_results, key=lambda r: r.bpp)
            print(f"  {signal_name:<20} → {best.name:<30} ({best.bpp:.3f} BPP, {best.coherence:.2e} coherence)")
    
    print()
    print("  BEST QUALITY (Highest PSNR):")
    print(f"  {'-' * 98}")
    for signal_name, results in all_results.items():
        valid_results = [r for r in results if r.status == "✓"]
        if valid_results:
            best = max(valid_results, key=lambda r: r.psnr)
            print(f"  {signal_name:<20} → {best.name:<30} ({best.psnr:.2f} dB)")
    
    print()
    print("  FASTEST (Lowest Latency):")
    print(f"  {'-' * 98}")
    for signal_name, results in all_results.items():
        valid_results = [r for r in results if r.status == "✓"]
        if valid_results:
            best = min(valid_results, key=lambda r: r.time_ms)
            print(f"  {signal_name:<20} → {best.name:<30} ({best.time_ms:.2f} ms)")
    
    print()
    print("  ZERO COHERENCE (η = 0):")
    print(f"  {'-' * 98}")
    for signal_name, results in all_results.items():
        zero_coherence = [r for r in results if r.status == "✓" and r.coherence < 1e-10]
        if zero_coherence:
            print(f"  {signal_name:<20} → {len(zero_coherence)} variants: {', '.join([r.name for r in zero_coherence[:5]])}")
    
    # ========================================================================
    # OVERALL WINNER
    # ========================================================================
    
    print()
    print("=" * 100)
    print("  OVERALL WINNER")
    print("=" * 100)
    print()
    
    # Average BPP across all signals
    hybrid_avg_bpp = {}
    hybrid_count = {}
    
    for signal_name, results in all_results.items():
        for result in results:
            if result.status == "✓":
                if result.name not in hybrid_avg_bpp:
                    hybrid_avg_bpp[result.name] = 0
                    hybrid_count[result.name] = 0
                hybrid_avg_bpp[result.name] += result.bpp
                hybrid_count[result.name] += 1
    
    # Compute averages
    for name in hybrid_avg_bpp:
        hybrid_avg_bpp[name] /= hybrid_count[name]
    
    # Sort by average BPP
    sorted_hybrids = sorted(hybrid_avg_bpp.items(), key=lambda x: x[1])
    
    print(f"  {'Rank':<6} │ {'Hybrid Variant':<35} │ {'Avg BPP':<10} │ Signals Tested")
    print(f"  {'-' * 98}")
    
    for rank, (name, avg_bpp) in enumerate(sorted_hybrids[:10], 1):
        marker = "🏆" if rank == 1 else "  "
        print(f"  {marker} {rank:<3} │ {name:<35} │ {avg_bpp:>10.3f} │ {hybrid_count[name]}")
    
    print()
    
    if sorted_hybrids:
        winner_name, winner_bpp = sorted_hybrids[0]
        print(f"  🏆 CHAMPION: {winner_name}")
        print(f"     Average BPP: {winner_bpp:.3f}")
        print(f"     Tested on: {hybrid_count[winner_name]} signals")
        
        # Get detailed stats for winner
        winner_results = []
        for results in all_results.values():
            winner_result = next((r for r in results if r.name == winner_name and r.status == "✓"), None)
            if winner_result:
                winner_results.append(winner_result)
        
        if winner_results:
            avg_psnr = np.mean([r.psnr for r in winner_results])
            avg_time = np.mean([r.time_ms for r in winner_results])
            avg_coherence = np.mean([r.coherence for r in winner_results])
            
            print(f"     Average PSNR: {avg_psnr:.2f} dB")
            print(f"     Average Time: {avg_time:.2f} ms")
            print(f"     Average Coherence: {avg_coherence:.2e}")
    
    print()
    print("=" * 100)
    print()


if __name__ == "__main__":
    run_comprehensive_benchmark()
