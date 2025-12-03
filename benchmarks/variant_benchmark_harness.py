#!/usr/bin/env python3
"""
Variant Benchmark Harness
=========================

Provides unified testing infrastructure for all 14 Φ-RFT variants and 17 hybrids
across all benchmark classes (A-E).

This module is imported by each class_*.py benchmark to wire in full variant coverage.
"""

import sys
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'experiments' / 'hypothesis_testing'))

# ============================================================================
# Variant Registry - All 14 Φ-RFT Variants
# ============================================================================

VARIANT_CODES = [
    "STANDARD", "HARMONIC", "FIBONACCI", "CHAOTIC", "GEOMETRIC",
    "PHI_CHAOTIC", "HYPERBOLIC", "LOG_PERIODIC", "CONVEX_MIX", "GOLDEN_EXACT",
    "CASCADE", "ADAPTIVE_SPLIT", "ENTROPY_GUIDED", "DICTIONARY"
]

# Skip these slow variants in benchmark runs (O(N³) complexity)
SLOW_VARIANTS = {"GOLDEN_EXACT"}  # Takes 21+ seconds even for small sizes

# Map variant codes to codec modes
VARIANT_TO_MODE = {
    "STANDARD": "legacy",
    "HARMONIC": "legacy",
    "FIBONACCI": "legacy",
    "CHAOTIC": "legacy",
    "GEOMETRIC": "legacy",
    "PHI_CHAOTIC": "legacy",
    "HYPERBOLIC": "legacy",
    "LOG_PERIODIC": "legacy",
    "CONVEX_MIX": "legacy",
    "GOLDEN_EXACT": "legacy",
    "CASCADE": "h3_cascade",
    "ADAPTIVE_SPLIT": "fh5_entropy",  # Uses adaptive split internally
    "ENTROPY_GUIDED": "fh5_entropy",
    "DICTIONARY": "h6_dictionary",
}

# ============================================================================
# Hybrid Registry - All 17 Hybrids
# ============================================================================

HYBRID_NAMES = [
    "H0_Baseline_Greedy",
    "H1_Coherence_Aware",
    "H2_Phase_Adaptive",
    "H3_Hierarchical_Cascade",
    "H4_Quantum_Superposition",
    "H5_Attention_Gating",
    "H6_Dictionary_Learning",
    "H7_Cascade_Attention",
    "H8_Aggressive_Cascade",
    "H9_Iterative_Refinement",
    "H10_Quality_Cascade",
    "FH1_MultiLevel_Cascade",
    "FH2_Adaptive_Split",
    "FH3_Frequency_Cascade",
    "FH4_Edge_Aware",
    "FH5_Entropy_Guided",
    "Legacy_Hybrid_Codec",
]

# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class VariantResult:
    """Result from a single variant benchmark run."""
    variant: str
    signal_type: str
    bpp: float
    psnr: float
    time_ms: float
    coherence: float
    success: bool
    error: Optional[str] = None


@dataclass
class HybridResult:
    """Result from a single hybrid benchmark run."""
    hybrid: str
    signal_type: str
    bpp: float
    psnr: float
    time_ms: float
    coherence: float
    success: bool
    error: Optional[str] = None


# ============================================================================
# Signal Generators
# ============================================================================

def generate_benchmark_signals(size: int = 1024) -> Dict[str, np.ndarray]:
    """Generate standard benchmark signal corpus."""
    np.random.seed(42)
    
    signals = {}
    
    # Basic signals
    signals['random_noise'] = np.random.randn(size)
    signals['sine_smooth'] = np.sin(2 * np.pi * 5 * np.arange(size) / size)
    signals['steps'] = np.concatenate([np.zeros(size//4), np.ones(size//4)] * 2)
    signals['chirp'] = np.sin(np.linspace(0, 50, size) ** 2 / 100)
    signals['sparse_impulse'] = np.zeros(size); signals['sparse_impulse'][::size//10] = 1.0
    
    # Domain-specific
    signals['ascii_code'] = np.array([ord(c) for c in ('x=42;' * (size // 5 + 1))[:size]], dtype=float)
    signals['fibonacci'] = np.array([_fib(i % 20) for i in range(size)], dtype=float)
    signals['mixed'] = (
        2.0 * np.sin(2 * np.pi * 3 * np.arange(size) / size) +
        0.5 * np.random.randn(size) +
        np.sign(np.sin(2 * np.pi * 15 * np.arange(size) / size))
    )
    
    return signals


def _fib(n: int) -> int:
    """Compute Fibonacci number."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b


def generate_quantum_state(n_qubits: int) -> np.ndarray:
    """Generate synthetic quantum state vector."""
    dim = 2 ** n_qubits
    # Random complex amplitudes, normalized
    state = np.random.randn(dim) + 1j * np.random.randn(dim)
    state /= np.linalg.norm(state)
    return state


def generate_audio_signals(sample_rate: int = 44100, duration: float = 1.0) -> Dict[str, np.ndarray]:
    """Generate audio test signals."""
    samples = int(sample_rate * duration)
    t = np.linspace(0, duration, samples, dtype=np.float32)
    
    signals = {}
    signals['sine_440hz'] = 0.8 * np.sin(2 * np.pi * 440 * t)
    signals['harmonic'] = sum(np.sin(2 * np.pi * 440 * (i+1) * t) / (i+1) for i in range(8)).astype(np.float32) * 0.5
    signals['noise'] = np.random.randn(samples).astype(np.float32) * 0.3
    signals['speech_like'] = (0.5 * np.sin(2 * np.pi * 500 * t) + 0.3 * np.sin(2 * np.pi * 1500 * t)).astype(np.float32)
    signals['chirp'] = np.sin(2 * np.pi * (100 + 1900 * t / duration) * t).astype(np.float32)
    
    return signals


# ============================================================================
# Variant Loading
# ============================================================================

def load_variant_generators() -> Dict[str, Callable]:
    """Load all variant basis generators from registry."""
    generators = {}
    
    try:
        from algorithms.rft.variants.manifest import iter_variants
        for entry in iter_variants(include_experimental=True):
            generators[entry.code] = entry.info.generator
    except ImportError:
        # Fallback to direct registry import
        try:
            from algorithms.rft.variants.registry import VARIANTS
            for key, info in VARIANTS.items():
                generators[info.name.upper().replace(' ', '_').replace('-', '_')] = info.generator
        except ImportError:
            pass
    
    return generators


def load_hybrid_functions() -> Dict[str, Callable]:
    """Load all hybrid encode functions."""
    hybrids = {}
    
    import sys
    import os
    import time
    
    # Get absolute path to experiments directories
    this_file = os.path.abspath(__file__)
    benchmarks_dir = os.path.dirname(this_file)
    repo_root = os.path.dirname(benchmarks_dir)
    
    # Add experiments/hypothesis_testing to path
    exp_hyp_path = os.path.join(repo_root, 'experiments', 'hypothesis_testing')
    if exp_hyp_path not in sys.path:
        sys.path.insert(0, exp_hyp_path)
    
    # Add experiments/ascii_wall to path
    exp_ascii_path = os.path.join(repo_root, 'experiments', 'ascii_wall')
    if exp_ascii_path not in sys.path:
        sys.path.insert(0, exp_ascii_path)
    
    # Load H0-H10 from hybrid_mca_fixes
    try:
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
            hypothesis10_quality_cascade,
        )
        
        hybrids.update({
            "H0_Baseline_Greedy": baseline_greedy_hybrid,
            "H1_Coherence_Aware": hypothesis1_coherence_aware,
            "H2_Phase_Adaptive": hypothesis2_phase_adaptive,
            "H3_Hierarchical_Cascade": hypothesis3_hierarchical_cascade,
            "H4_Quantum_Superposition": hypothesis4_quantum_superposition,
            "H5_Attention_Gating": hypothesis5_attention_gating,
            "H6_Dictionary_Learning": hypothesis6_dictionary_learning,
            "H7_Cascade_Attention": hypothesis7_cascade_attention,
            "H8_Aggressive_Cascade": hypothesis8_aggressive_cascade,
            "H9_Iterative_Refinement": hypothesis9_iterative_refinement,
            "H10_Quality_Cascade": hypothesis10_quality_cascade,
        })
    except Exception as e:
        print(f"WARNING: Failed to load H0-H10: {e}")
    
    # Load FH1-FH5 from ascii_wall_final_hypotheses
    try:
        from ascii_wall_final_hypotheses import (
            fh1_multilevel_cascade,
            fh2_adaptive_split_cascade,
            fh3_frequency_cascade,
            fh4_edge_aware_cascade,
            fh5_entropy_guided_cascade,
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
        
        hybrids.update({
            "FH1_MultiLevel_Cascade": wrap_fh(fh1_multilevel_cascade),
            "FH2_Adaptive_Split": wrap_fh(fh2_adaptive_split_cascade),
            "FH3_Frequency_Cascade": wrap_fh(fh3_frequency_cascade),
            "FH4_Edge_Aware": wrap_fh(fh4_edge_aware_cascade),
            "FH5_Entropy_Guided": wrap_fh(fh5_entropy_guided_cascade),
        })
    except Exception as e:
        print(f"WARNING: Failed to load FH1-FH5: {e}")
    
    return hybrids


# ============================================================================
# Benchmark Runners
# ============================================================================

def benchmark_variant_on_signal(
    variant_code: str,
    signal: np.ndarray,
    signal_name: str,
    generators: Dict[str, Callable]
) -> VariantResult:
    """Run a single variant on a single signal."""
    
    if variant_code not in generators:
        return VariantResult(
            variant=variant_code,
            signal_type=signal_name,
            bpp=999.0, psnr=-999.0, time_ms=0.0, coherence=999.0,
            success=False, error=f"Generator not found for {variant_code}"
        )
    
    try:
        generator = generators[variant_code]
        n = len(signal)
        
        start = time.perf_counter()
        
        # Generate basis and transform
        basis = generator(n)
        
        # Forward transform
        coeffs = basis @ signal
        
        # Compute sparsity (BPP proxy)
        threshold = 0.01 * np.max(np.abs(coeffs))
        kept = np.sum(np.abs(coeffs) > threshold)
        bpp = (kept * 11.0) / n  # 11 bits per coefficient estimate
        
        # Inverse transform
        reconstructed = np.real(basis.conj().T @ coeffs)
        
        elapsed = (time.perf_counter() - start) * 1000
        
        # PSNR
        mse = np.mean((signal - reconstructed) ** 2)
        if mse < 1e-12:
            psnr = float('inf')
        else:
            psnr = 20 * np.log10(np.max(np.abs(signal)) / np.sqrt(mse))
        
        # Coherence (off-diagonal norm)
        identity = basis.conj().T @ basis
        coherence = np.linalg.norm(identity - np.eye(n), 'fro') / n
        
        return VariantResult(
            variant=variant_code,
            signal_type=signal_name,
            bpp=bpp,
            psnr=psnr,
            time_ms=elapsed,
            coherence=coherence,
            success=True
        )
        
    except Exception as e:
        return VariantResult(
            variant=variant_code,
            signal_type=signal_name,
            bpp=999.0, psnr=-999.0, time_ms=0.0, coherence=999.0,
            success=False, error=str(e)[:60]
        )


def benchmark_hybrid_on_signal(
    hybrid_name: str,
    signal: np.ndarray,
    signal_name: str,
    hybrids: Dict[str, Callable]
) -> HybridResult:
    """Run a single hybrid on a single signal."""
    
    if hybrid_name not in hybrids:
        return HybridResult(
            hybrid=hybrid_name,
            signal_type=signal_name,
            bpp=999.0, psnr=-999.0, time_ms=0.0, coherence=999.0,
            success=False, error=f"Hybrid not found: {hybrid_name}"
        )
    
    try:
        func = hybrids[hybrid_name]
        
        start = time.perf_counter()
        result = func(signal, target_sparsity=0.95)
        elapsed = (time.perf_counter() - start) * 1000
        
        return HybridResult(
            hybrid=hybrid_name,
            signal_type=signal_name,
            bpp=result.bpp,
            psnr=result.psnr,
            time_ms=elapsed,
            coherence=result.coherence_violation,
            success=True
        )
        
    except Exception as e:
        return HybridResult(
            hybrid=hybrid_name,
            signal_type=signal_name,
            bpp=999.0, psnr=-999.0, time_ms=0.0, coherence=999.0,
            success=False, error=str(e)[:60]
        )


def run_all_variants_benchmark(signals: Dict[str, np.ndarray], skip_slow: bool = True) -> List[VariantResult]:
    """Run all 14 variants on all signals.
    
    Args:
        signals: Dictionary of signal name -> signal array
        skip_slow: If True, skip GOLDEN_EXACT and other O(N³) variants
    """
    generators = load_variant_generators()
    results = []
    
    for variant_code in VARIANT_CODES:
        # Skip slow variants unless explicitly requested
        if skip_slow and variant_code in SLOW_VARIANTS:
            # Add placeholder result
            for signal_name in signals.keys():
                results.append(VariantResult(
                    variant=variant_code,
                    signal_type=signal_name,
                    bpp=0.0, psnr=0.0, time_ms=0.0, coherence=0.0,
                    success=False, error="Skipped (O(N³) complexity)"
                ))
            continue
            
        for signal_name, signal in signals.items():
            result = benchmark_variant_on_signal(variant_code, signal, signal_name, generators)
            results.append(result)
    
    return results


def run_all_hybrids_benchmark(signals: Dict[str, np.ndarray]) -> List[HybridResult]:
    """Run all 17 hybrids on all signals."""
    hybrids = load_hybrid_functions()
    results = []
    
    for hybrid_name in HYBRID_NAMES[:-1]:  # Exclude Legacy which needs special handling
        for signal_name, signal in signals.items():
            result = benchmark_hybrid_on_signal(hybrid_name, signal, signal_name, hybrids)
            results.append(result)
    
    return results


# ============================================================================
# Reporting
# ============================================================================

def print_variant_results(results: List[VariantResult], title: str = "VARIANT BENCHMARK"):
    """Print formatted variant results."""
    print()
    print("=" * 100)
    print(f"  {title}")
    print("=" * 100)
    print()
    
    # Group by variant
    by_variant = {}
    for r in results:
        if r.variant not in by_variant:
            by_variant[r.variant] = []
        by_variant[r.variant].append(r)
    
    print(f"  {'Variant':<20} │ {'Avg BPP':>8} │ {'Avg PSNR':>10} │ {'Avg Time':>10} │ {'Coherence':>10} │ Status")
    print("  " + "─" * 90)
    
    for variant in VARIANT_CODES:
        if variant not in by_variant:
            print(f"  {variant:<20} │ {'N/A':>8} │ {'N/A':>10} │ {'N/A':>10} │ {'N/A':>10} │ ✗ Not loaded")
            continue
        
        vr = by_variant[variant]
        successful = [r for r in vr if r.success]
        
        if not successful:
            print(f"  {variant:<20} │ {'N/A':>8} │ {'N/A':>10} │ {'N/A':>10} │ {'N/A':>10} │ ✗ All failed")
            continue
        
        avg_bpp = np.mean([r.bpp for r in successful])
        avg_psnr = np.mean([r.psnr for r in successful if r.psnr != float('inf')])
        avg_time = np.mean([r.time_ms for r in successful])
        avg_coh = np.mean([r.coherence for r in successful])
        
        status = "✓" if len(successful) == len(vr) else f"⚠ {len(successful)}/{len(vr)}"
        
        print(f"  {variant:<20} │ {avg_bpp:>8.3f} │ {avg_psnr:>9.2f}dB │ {avg_time:>9.2f}ms │ {avg_coh:>10.2e} │ {status}")
    
    print()


def print_hybrid_results(results: List[HybridResult], title: str = "HYBRID BENCHMARK"):
    """Print formatted hybrid results."""
    print()
    print("=" * 100)
    print(f"  {title}")
    print("=" * 100)
    print()
    
    # Group by hybrid
    by_hybrid = {}
    for r in results:
        if r.hybrid not in by_hybrid:
            by_hybrid[r.hybrid] = []
        by_hybrid[r.hybrid].append(r)
    
    print(f"  {'Hybrid':<30} │ {'Avg BPP':>8} │ {'Avg PSNR':>10} │ {'Avg Time':>10} │ {'Coherence':>10} │ Status")
    print("  " + "─" * 95)
    
    for hybrid in HYBRID_NAMES[:-1]:
        if hybrid not in by_hybrid:
            print(f"  {hybrid:<30} │ {'N/A':>8} │ {'N/A':>10} │ {'N/A':>10} │ {'N/A':>10} │ ✗ Not loaded")
            continue
        
        hr = by_hybrid[hybrid]
        successful = [r for r in hr if r.success]
        
        if not successful:
            err = hr[0].error if hr else "Unknown"
            print(f"  {hybrid:<30} │ {'N/A':>8} │ {'N/A':>10} │ {'N/A':>10} │ {'N/A':>10} │ ✗ {err[:20]}")
            continue
        
        avg_bpp = np.mean([r.bpp for r in successful])
        avg_psnr = np.mean([r.psnr for r in successful if r.psnr != float('inf')])
        avg_time = np.mean([r.time_ms for r in successful])
        avg_coh = np.mean([r.coherence for r in successful])
        
        status = "✓" if len(successful) == len(hr) else f"⚠ {len(successful)}/{len(hr)}"
        
        coh_str = "η=0" if avg_coh < 1e-6 else f"{avg_coh:.2e}"
        
        print(f"  {hybrid:<30} │ {avg_bpp:>8.3f} │ {avg_psnr:>9.2f}dB │ {avg_time:>9.2f}ms │ {coh_str:>10} │ {status}")
    
    print()


# ============================================================================
# Self-test
# ============================================================================

if __name__ == "__main__":
    print("=" * 100)
    print("  VARIANT BENCHMARK HARNESS - Self Test")
    print("=" * 100)
    
    # Load generators
    generators = load_variant_generators()
    print(f"\n  Loaded {len(generators)} variant generators:")
    for code in generators:
        print(f"    ✓ {code}")
    
    # Load hybrids
    hybrids = load_hybrid_functions()
    print(f"\n  Loaded {len(hybrids)} hybrid functions:")
    for name in hybrids:
        print(f"    ✓ {name}")
    
    # Quick benchmark
    signals = generate_benchmark_signals(256)  # Small for quick test
    print(f"\n  Generated {len(signals)} test signals")
    
    # Run variants
    variant_results = run_all_variants_benchmark(signals)
    print_variant_results(variant_results, "VARIANT QUICK TEST (N=256)")
    
    # Run hybrids
    hybrid_results = run_all_hybrids_benchmark(signals)
    print_hybrid_results(hybrid_results, "HYBRID QUICK TEST (N=256)")
    
    # Summary
    v_success = sum(1 for r in variant_results if r.success)
    h_success = sum(1 for r in hybrid_results if r.success)
    
    print("=" * 100)
    print(f"  SUMMARY: {v_success}/{len(variant_results)} variant tests, {h_success}/{len(hybrid_results)} hybrid tests")
    print("=" * 100)
