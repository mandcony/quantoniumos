# QuantoniumOS Experiments

Comprehensive experimental validation of Φ-RFT (Recursive Fibonacci Transform) and related algorithms.

## Directory Structure

```
experiments/
├── ascii_wall/           # ASCII Wall compression experiments (H11-H20)
├── hypothesis_testing/   # Core hypothesis battery (H1-H12)
├── entropy/              # Information-theoretic analysis
├── fibonacci/            # Fibonacci tilt experiments
├── tetrahedral/          # Tetrahedral deep-dive experiments
├── sota_benchmarks/      # State-of-the-art compression comparisons
├── corpus/               # Real-world corpus testing
├── FINAL_RECOMMENDATION.md
└── ITERATION_SUMMARY.md
```

## Experiment Categories

### 1. Hypothesis Testing (`hypothesis_testing/`)

**Main File:** `hypothesis_battery_h1_h12.py`

Tests 12 hypotheses across 5 groups:

| Hypothesis | Description | Result |
|------------|-------------|--------|
| **H1** | Golden coherence geometry: ∃σ* where μ(DCT,Ψ_σ*) < μ(DCT,Ψ_0) | [OK] **SUPPORTED** (79.6% coherence reduction at σ*=1.0) |
| **H3** | Rate-distortion improvement via Φ-RFT | [OK] **SUPPORTED** (11.6% BPP improvement at μ_max=0.45) |
| **H4** | Oscillatory phase stability | [X] REJECTED |
| **H5** | Annealed cascades beat fixed baseline | [OK] **SUPPORTED** |
| **H6** | AST compression via RFT | [X] REJECTED (gzip wins) |
| **H8** | RFT EQ transients superior to FFT | [X] REJECTED (identical) |
| **H9** | RFT oscillators expand timbre space | [OK] **SUPPORTED** (280x coverage) |
| **H10** | PDE solver stability via RFT | [X] REJECTED (same stability) |
| **H11** | Crypto avalanche near 50% | [OK] **SUPPORTED** (50.7% avalanche) |
| **H12** | Geometric hash distribution | [!] PARTIAL (different kurtosis) |

**Summary:** 5 fully supported, 1 partial, 4 rejected

### 2. Entropy Analysis (`entropy/`)

**Main File:** `entropy_rate_analysis.py`

Rigorous information-theoretic evaluation using:
- Shannon entropy estimation (H0, H1, H_k)
- Multiple source models (IID uniform, Markov, Gaussian, etc.)
- Comparison with zlib, brotli

**Key Finding:**
> "Φ-RFT is NOT a compression algorithm - it's a transform. It does not beat standard compressors on any tested source."

This is expected behavior. RFT is designed for:
- Transform domain representation
- Golden ratio coherence properties
- Audio synthesis applications

### 3. ASCII Wall (`ascii_wall/`)

Experiments on the "ASCII Wall" problem - compression of printable ASCII streams:

- `ascii_wall_all_variants.py` - Tests all RFT variants
- `ascii_wall_paper.py` - Paper-ready experiments
- `ASCII_WALL_THEOREM.md` - Theoretical analysis

### 4. Fibonacci Tilt (`fibonacci/`)

Fibonacci lattice tilt angle optimization:

- `fibonacci_tilt_hypotheses.py` - Tilt angle experiments
- `FIBONACCI_TILT_RESULTS.md` - Summary of findings

### 5. Tetrahedral (`tetrahedral/`)

Tetrahedral geometry deep-dive:

- `tetrahedral_deep_dive.py` - Geometric analysis
- `tetrahedral_rft_hypotheses.py` - RFT on tetrahedral lattices

### 6. SOTA Benchmarks (`sota_benchmarks/`)

Comparisons with state-of-the-art algorithms:

- `sota_compression_benchmark.py` - Benchmark suite
- `SOTA_BENCHMARK_RESULTS.md` - Full results
- `PAPER_COMPARISON.md` - Academic paper comparisons

### 7. Corpus Testing (`corpus/`)

Real-world corpus validation:

- `test_real_corpora.py` - Tests on Canterbury, Silesia, etc.
- `CORPUS_TEST_ANALYSIS.md` - Analysis

## Running Experiments

```bash
# Run hypothesis battery
cd experiments/hypothesis_testing
python hypothesis_battery_h1_h12.py

# Run entropy analysis
cd experiments/entropy
python entropy_rate_analysis.py

# Run SOTA benchmarks
cd experiments/sota_benchmarks
python sota_compression_benchmark.py
```

## Key Results Summary

### What Φ-RFT Does Well

1. **Coherence Reduction** - 79.6% reduction in mutual coherence vs DCT (H1)
2. **Audio Synthesis** - 280x expanded timbre coverage via RFT oscillators (H9)
3. **Crypto Properties** - Near-ideal 50.7% avalanche effect (H11)
4. **Rate-Distortion** - 11.6% BPP improvement in certain regimes (H3)

### What Φ-RFT Does NOT Do

1. **General Compression** - Does not beat gzip/brotli on arbitrary data
2. **PDE Solving** - No stability improvements over standard methods
3. **AST Compression** - Standard compressors win

### Honest Assessment

Φ-RFT is a **transform** with unique golden ratio properties, not a universal compressor. Its strength lies in:
- Audio/signal processing
- Cryptographic hash functions
- Transform-domain representations with low coherence

## Results Files

All experiments produce `.json` results files for reproducibility:
- `hypothesis_battery_results.json`
- `entropy_rate_results.json`
- `sota_benchmark_results.json`
- `fibonacci_tilt_results.json`
- etc.

## Citation

If using these experiments in research, cite:
```
@software{quantoniumos_experiments,
  title = {QuantoniumOS Φ-RFT Experimental Validation},
  year = {2024},
  url = {https://github.com/yourusername/quantoniumos}
}
```
