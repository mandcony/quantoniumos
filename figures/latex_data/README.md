# QuantoniumOS Figure Generation Workflow

This directory contains the complete pipeline for generating publication-ready figures from validation data.

## Quick Start

### Step 1: Generate Validation Data (Python)
```bash
# From repository root
python scripts/verify_rate_distortion.py --export figures/latex_data/rate_distortion.csv
python tests/benchmarks/rft_wave_computer_demo.py --export figures/latex_data/wave_computer.csv
```

### Step 2: Render Figures (MATLAB)
```matlab
% Open MATLAB and navigate to figures/latex_data/
cd figures/latex_data/

% Generate rate-distortion plot
run plot_rate_distortion.m

% Generate wave computer benchmark plot
run plot_wave_computer.m
```

### Step 3: Compile Paper (LaTeX)
```bash
cd docs/research
pdflatex THE_PHI_RFT_FRAMEWORK_PAPER.tex
pdflatex THE_PHI_RFT_FRAMEWORK_PAPER.tex  # Run twice for references
```

## Automated Pipeline

Run the complete validation suite:
```bash
bash scripts/generate_paper_figures.sh
```

## Output Files

### CSV Data (Input to MATLAB)
- `figures/latex_data/rate_distortion.csv` - Rate vs Distortion for DCT/RFT/Hybrid
- `figures/latex_data/wave_computer.csv` - Reconstruction error vs number of modes

### MATLAB Figures (Embedded in LaTeX)
- `figures/rft_rate_distortion_matlab.pdf` - Publication-quality R-D curves
- `figures/rft_wave_computer_matlab.pdf` - Wave computer efficiency plot

### Python Figures (Optional)
- `rft_wave_computer_benchmark.png` - Direct Python/Matplotlib output
- `figures/transform_fingerprints.png` - Variant visualization

## What Each Test Validates

### 1. `irrevocable_truths.py`
**Claims tested:**
- Unitarity of all 7 transforms (Frobenius error < 1e-14)
- Exact diagonalization of Golden Ratio Hamiltonians
- Sparsity on quasi-periodic signals (>89% for N=64)

**Honest assessment:** The transforms are unitary by QR construction. The diagonalization test is circular (builds H from U, then checks U diagonalizes H). Sparsity is proven for signals built from the basis itself.

### 2. `verify_variant_claims.py`
**Claims tested:**
- Entropy/whitening (all variants achieve ~6 bits for impulse)
- Nonlinear response (Harmonic-Phase vs Original for cubic signals)
- Fibonacci lattice resonance (Fibonacci Tilt isolates integer modes)
- Adaptive selection (meta-layer chooses correct basis per signal type)
- Quantum chaos statistics (level spacing)
- Cryptographic avalanche (Fibonacci Tilt ~57%)

**Honest assessment:** Tests confirm the variants are mathematically distinct and occupy different representational niches. The adaptive selection works because we're testing on synthetic signals designed for each basis.

### 3. `verify_rate_distortion.py`
**Claims tested:**
- The "ASCII Bottleneck" (RFT fails on text, DCT fails on waves)
- Theorem 10: Hybrid achieves R_hybrid ≤ min(R_DCT, R_RFT)

**Honest assessment:** This is the most practical result. It proves that using the right transform for each signal component (DCT for steps, RFT for waves) beats using either alone. This is solid engineering, not circular reasoning.

### 4. `rft_wave_computer_demo.py`
**Claims tested:**
- Graph RFT exactly solves Fibonacci graph dynamics (MSE < 1e-10 with 5 modes)
- FFT fails (MSE > 1e-5 with 5 modes) because it's mismatched to the topology

**Honest assessment:** The system is defined on a graph whose Laplacian eigenvectors ARE the RFT basis. So yes, RFT wins by construction. But this is a valid demonstration of "quantum advantage on classical hardware" IF your physics matches this graph topology.

## Provenance Chain

```
Python Scripts → CSV Export → MATLAB Rendering → LaTeX Embedding → PDF Paper
```

This workflow ensures:
1. All figures are reproducible from source code
2. Reviewers can regenerate figures using either Python or MATLAB
3. The mathematical claims are traceable to specific test outputs
