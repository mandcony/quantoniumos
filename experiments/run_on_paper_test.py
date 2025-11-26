#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.
"""
Copyright (C) 2025 QuantoniumOS Research Team
Licensed under Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
For commercial licensing inquiries, contact: github.com/mandcony/quantoniumos

Run hybrid MCA experiments on paper's rate-distortion test signal.

This tests all 10 hypotheses on the actual signal used in Section VI-C
(mixed ASCII steps + Fibonacci waves) to compare against paper's 4.96 BPP claim.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import paper's test signal generator
from scripts.verify_rate_distortion import generate_mixed_signal

# Import our experiment framework
sys.path.insert(0, str(Path(__file__).parent))
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
)


def run_paper_experiments(target_sparsity=0.95):
    """Run all hypotheses on paper's test signal."""
    
    # Generate paper's test signal (ASCII steps + Fibonacci waves)
    print("Generating paper's test signal (512 samples)...")
    signal = generate_mixed_signal(n=512)
    
    print(f"Signal stats:")
    print(f"  Length: {len(signal)}")
    print(f"  Mean: {np.mean(signal):.3f}")
    print(f"  Std: {np.std(signal):.3f}")
    print(f"  Min/Max: {np.min(signal):.3f} / {np.max(signal):.3f}")
    print()
    
    print("="*100)
    print("PAPER TEST SIGNAL EXPERIMENTS")
    print(f"Signal: Mixed ASCII steps + Fibonacci waves (as used in Section VI-C)")
    print(f"Target sparsity: {target_sparsity*100:.1f}%")
    print("="*100)
    print()
    
    experiments = [
        ("Baseline (Greedy)", baseline_greedy_hybrid),
        ("H1: Coherence-Aware", hypothesis1_coherence_aware),
        ("H2: Phase-Adaptive", hypothesis2_phase_adaptive),
        ("H3: Hierarchical Cascade", hypothesis3_hierarchical_cascade),
        ("H4: Quantum Superposition", hypothesis4_quantum_superposition),
        ("H5: Attention Gating", hypothesis5_attention_gating),
        ("H6: Dictionary Learning", hypothesis6_dictionary_learning),
        ("H7: Cascade + Attention", hypothesis7_cascade_attention),
        ("H8: Aggressive Cascade", hypothesis8_aggressive_cascade),
        ("H9: Iterative Refinement", hypothesis9_iterative_refinement),
    ]
    
    results = []
    
    for name, func in experiments:
        print(f"Running {name}...", end=" ", flush=True)
        try:
            result = func(signal, target_sparsity)
            results.append(result)
            print(f"✓ {result.bpp:.3f} BPP, {result.psnr:.2f} dB ({result.time_ms:.1f}ms)")
        except Exception as e:
            print(f"✗ FAILED: {e}")
    
    print()
    print("="*120)
    print("RESULTS TABLE")
    print("="*120)
    print(f"{'Method':<35} {'BPP':>8} {'PSNR':>8} {'Time':>10} {'Sparsity':>10} {'Recon Err':>12} {'Coherence':>12}")
    print("-"*120)
    
    # Sort by BPP
    results_sorted = sorted(results, key=lambda r: r.bpp)
    
    # Find baseline
    baseline = next((r for r in results if "Baseline" in r.name), None)
    
    # Mark winners
    best_bpp = min(r.bpp for r in results)
    best_psnr = max(r.psnr for r in results)
    
    for result in results_sorted:
        marker = ""
        if result.bpp == best_bpp:
            marker = "🏆"
        elif result.psnr == best_psnr:
            marker = "⭐"
        else:
            marker = "  "
        
        print(f"{marker} {result.name:<33} "
              f"{result.bpp:>8.3f} "
              f"{result.psnr:>8.2f} "
              f"{result.time_ms:>9.1f}ms "
              f"{result.sparsity_pct:>9.1f}% "
              f"{result.reconstruction_error:>12.4e} "
              f"{result.coherence_violation:>12.4e}")
    
    print("="*120)
    print()
    
    # Comparison to paper's claim
    print("COMPARISON TO PAPER (Section VI-C):")
    print("-" * 80)
    print(f"Paper's claim (Table VI):")
    print(f"  Pure DCT:     4.83 BPP at 39.21 dB PSNR")
    print(f"  Pure RFT:     7.72 BPP at 31.04 dB PSNR") 
    print(f"  Hybrid:       4.96 BPP at 38.52 dB PSNR  ← Current paper claim")
    print()
    
    if baseline:
        print(f"Our baseline (greedy hybrid):  {baseline.bpp:.3f} BPP at {baseline.psnr:.2f} dB")
    
    best_result = results_sorted[0]
    print(f"🏆 Best method: {best_result.name}")
    print(f"   BPP:  {best_result.bpp:.3f} ({(4.96 - best_result.bpp) / 4.96 * 100:.1f}% improvement over paper)")
    print(f"   PSNR: {best_result.psnr:.2f} dB")
    print(f"   Coherence: {best_result.coherence_violation:.2e}")
    print()
    
    # Analysis
    print("KEY FINDINGS:")
    print("-" * 80)
    
    cascade_methods = [r for r in results if "Cascade" in r.name or "H3" in r.name or "H7" in r.name]
    if cascade_methods:
        avg_coherence = np.mean([r.coherence_violation for r in cascade_methods])
        print(f"✓ Cascade methods achieve {avg_coherence:.2e} avg coherence violation (vs baseline {baseline.coherence_violation:.2e})")
    
    improvements = [(r.name, (4.96 - r.bpp) / 4.96 * 100) for r in results if r.bpp < 4.96]
    if improvements:
        print(f"✓ {len(improvements)}/{len(results)} methods beat paper's 4.96 BPP claim:")
        for name, pct in sorted(improvements, key=lambda x: -x[1])[:5]:
            print(f"    {name}: {pct:+.1f}%")
    
    print()
    
    # Save results
    output_file = Path(__file__).parent / "paper_test_results.txt"
    with open(output_file, 'w') as f:
        f.write("PAPER TEST SIGNAL RESULTS (Mixed ASCII + Fibonacci)\n")
        f.write("="*80 + "\n\n")
        
        f.write("Paper's baseline (Table VI):\n")
        f.write("  Pure DCT:  4.83 BPP at 39.21 dB\n")
        f.write("  Pure RFT:  7.72 BPP at 31.04 dB\n")
        f.write("  Hybrid:    4.96 BPP at 38.52 dB\n\n")
        
        for result in results_sorted:
            f.write(f"{result.name}:\n")
            f.write(f"  BPP: {result.bpp:.3f}\n")
            f.write(f"  PSNR: {result.psnr:.2f} dB\n")
            f.write(f"  Time: {result.time_ms:.1f} ms\n")
            f.write(f"  Sparsity: {result.sparsity_pct:.1f}%\n")
            f.write(f"  Reconstruction error: {result.reconstruction_error:.4e}\n")
            f.write(f"  Coherence violation: {result.coherence_violation:.4e}\n")
            f.write(f"  Improvement over paper: {(4.96 - result.bpp) / 4.96 * 100:+.1f}%\n")
            f.write("\n")
    
    print(f"Results saved to: {output_file}")
    

if __name__ == "__main__":
    run_paper_experiments(target_sparsity=0.95)
