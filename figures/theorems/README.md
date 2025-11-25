# Theorem Visualization Figures

Publication-quality figures for all RFT theorems and experimental results.

## Theorem 10: Hybrid Φ-RFT / DCT Decomposition

### Figure 1: Rate-Distortion Analysis (ASCII Bottleneck)
**File:** `theorem10_rate_distortion.[png/pdf]`

**Shows:** The catastrophic failure of pure RFT on text data and how Hybrid solves it.

**Key Results:**
- **DCT Only:** 4.83 BPP (baseline)
- **RFT Only:** 7.72 BPP (**+60% catastrophic failure** ❌)
- **Hybrid:** 4.96 BPP (≈ DCT, problem solved ✅)
- **Overhead:** ε ≈ 0.13 BPP (< 0.2 BPP threshold ✓)

**Reference:** Section 4 of THEOREM_10_HYBRID.md

---

### Figure 2: Greedy vs Braided Comprehensive Comparison
**File:** `theorem10_greedy_vs_braided.[png/pdf]`

**Shows:** Why parallel competition (braided) fails catastrophically on all metrics.

**Contains 4 panels:**

1. **Test 1: Compression Efficiency** (% coefficients used)
   - Braided requires **2× MORE coefficients** than Greedy
   - Fails on all 4 datasets (Natural Text, Python Code, Random ASCII, Mixed Signal)

2. **Test 2: Reconstruction Quality** (error)
   - Braided has **100× HIGHER error** than Greedy
   - Logarithmic scale shows magnitude of failure

3. **Test 3: Source Separation (MCA)**
   - Braided error: ~0.91 vs Greedy: ~0.03 (**30× worse**)
   - Consistent across all sparsity levels (Ks=4/8, Kt=4/8)

4. **Test 4: Rate-Distortion Pareto Frontier**
   - Braided is **Pareto-dominated** (worse on BOTH rate AND distortion)
   - Arrows show domination relationship

**Verdict:** Braided is catastrophically worse on ALL metrics.

**Root Cause:** Winner-takes-all hard thresholding in frequency domain creates destructive interference for non-orthogonal bases.

**Reference:** Section 5.3 of THEOREM_10_HYBRID.md

---

### Figure 3: Soft vs Hard Braided Thresholding
**File:** `theorem10_soft_braided.[png/pdf]`

**Shows:** Soft thresholding is theoretically interesting but still not practical.

**Key Results:**
- **Greedy:** 0.0402 reconstruction error (best)
- **Hard Braid:** 0.8518 error (catastrophic)
- **Soft Braid:** 0.7255 error (**1.17× better than hard**)

**Insights:**
- ✅ **Scientific contribution:** Proves parallel competition CAN work with proper smoothing
- ⚠️ **Engineering reality:** Greedy still **18× better** than Soft Braid
- 📌 **Publication strategy:** Present Greedy as main result, document Soft as theoretical exploration

**Reference:** Section 5.3.2 of THEOREM_10_HYBRID.md

---

### Figure 4: RFT Phase Kernel Variants
**File:** `theorem10_phase_variants.[png/pdf]`

**Shows:** Three phase modulation variants maintain unitarity and achieve performance parity.

**Contains 3 rows:**

1. **Phase Distributions θ(k):**
   - Standard: 2πβ·frac(k/φ)
   - LogPhi: 2πβ·log(1+k)/log(1+N)
   - Mixed: (1-α)·standard + α·logphi

2. **Unit Circle Representation:**
   - All variants populate unit circle uniformly
   - Color-coded by index k showing phase progression

3. **Performance Comparison:**
   - All variants achieve **same sparsity** on text datasets
   - Confirms adaptive routing (not phase kernel) dominates for text
   - Hypothesis: Divergence expected on mixed signals (wave + text)

**Properties Validated:**
- ✅ Unit modulus: |exp(iθ)| = 1
- ✅ Unitary transforms: Ψ†Ψ = I (error < 10⁻¹⁴)
- ✅ Numerically stable
- ✅ Extensible framework

**Reference:** Section 6 of THEOREM_10_HYBRID.md

---

### Figure 5: MCA Separation Failure Analysis
**File:** `theorem10_mca_failure.[png/pdf]`

**Shows:** Root cause analysis of why greedy sequential fails at source separation.

**Contains 4 rows:**

1. **Ground Truth Components:**
   - Structure (DCT-sparse): Step functions
   - Texture (RFT-sparse): Golden ratio wave
   - Mixture: x = x_s + x_t

2. **Spectral Analysis:**
   - DCT domain: Structure dominates
   - RFT domain: Texture dominates
   - Mutual Coherence: Overlap in coefficient energies

3. **Performance Metrics:**
   - ✅ Total Error: 0.05 (reconstruction WORKS)
   - ❌ Struct Error: 1.5 (separation FAILS)
   - ❌ Texture Error: 1.0 (separation FAILS)
   - ⚠️ DCT F1: 0.65 (partial capture)
   - ❌ RFT F1: 0.05 (complete failure)

4. **Root Cause Explanation:**
   ```
   1. DCT BIAS: DCT captures BOTH structure AND texture
   2. GREEDY SUBTRACTION: DCT wins → subtracts ALL energy
   3. RFT STARVATION: RFT never claims its atoms
   4. RESULT: "DCT-First Codec" not true MCA separator
   ```

**Solution Required:** Replace greedy/parallel with L1-minimization (BPDN):
```
min_{s,t} ||s||_1 + ||t||_1   s.t.   ||x - (Ψ_S·s + Ψ_T·t)||_2 < ε
```

**Reference:** Section 5.2 and 7.8 of THEOREM_10_HYBRID.md

---

## Technical Specifications

### Figure Quality
- **Resolution:** 300 DPI (publication-ready)
- **Formats:** PNG (web), PDF (vector, LaTeX)
- **Color schemes:**
  - Blue: DCT/Structure/Baseline
  - Red: RFT/Texture/Failure
  - Green: Hybrid/Success/Solution
  - Gray: Reference/Comparison

### Data Sources
All figures generated from empirical results in:
- `scripts/verify_ascii_bottleneck.py`
- `scripts/verify_hybrid_mca_recovery.py`
- `tests/rft/test_hybrid_basis.py`

### Reproducibility
Regenerate all figures:
```bash
python3 generate_all_theorem_figures.py
```

All tests run at N=256 with default parameters (β=0.83, σ=1.25).

---

## Usage in Publications

### LaTeX Integration
```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.8\textwidth]{figures/theorems/theorem10_rate_distortion.pdf}
  \caption{Rate-Distortion analysis showing RFT catastrophic failure on text (7.72 BPP) 
           and Hybrid solution (4.96 BPP ≈ DCT baseline).}
  \label{fig:theorem10_rd}
\end{figure}
```

### Markdown/GitHub
```markdown
![Rate-Distortion](figures/theorems/theorem10_rate_distortion.png)
*Figure 1: Hybrid solves the ASCII bottleneck with minimal overhead (ε ≈ 0.13 BPP)*
```

---

## Summary of Key Results

| Figure | Main Finding | Status |
|:-------|:-------------|:------:|
| **Rate-Distortion** | Hybrid achieves ≈DCT rate, avoiding RFT's +60% failure | ✅ SOLVED |
| **Greedy vs Braided** | Parallel competition fails catastrophically (30× worse) | ❌ FAILED |
| **Soft Braided** | Soft thresholding 1.17× better than hard, but 18× worse than greedy | ⚠️ PARTIAL |
| **Phase Variants** | All variants achieve parity on text (adaptive routing dominates) | ✅ VALIDATED |
| **MCA Failure** | Greedy works for compression but fails at separation (RFT F1 = 0.05) | ❌ NEEDS BPDN |

---

## Future Work

Additional figures needed for complete theorem coverage:

1. **Theorem 1-9:** Core RFT properties (unitarity already in GIFs)
2. **Scaling experiments:** N ∈ {256, 512, 1024, 2048, 4096}
3. **Real-world datasets:** Natural images, audio, video
4. **Parameter sensitivity:** β, σ grid search heatmaps
5. **BPDN comparison:** When L1-minimization is implemented

---

Generated by `generate_all_theorem_figures.py`  
Date: November 25, 2025
