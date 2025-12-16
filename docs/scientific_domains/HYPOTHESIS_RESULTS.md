# Hypothesis Testing Results

> **Status:** ✅ Verified by `experiments/hypothesis_testing/hypothesis_battery_h1_h12.py`
> **Date:** December 16, 2025

This document records the formal scientific verification of the QuantoniumOS hypotheses. We explicitly list **REJECTED** hypotheses to maintain scientific integrity.

## Summary Table

| ID | Hypothesis | Verdict | Findings |
| :--- | :--- | :--- | :--- |
| **H1** | **Golden Coherence Geometry** | ✅ **SUPPORTED** | There exists a coherence parameter $\sigma^*$ where RFT minimizes mutual coherence better than DCT (79% improvement). |
| **H3** | **Coherence Budget Tradeoff** | ✅ **SUPPORTED** | An interior optimum exists for coherence budgeting. |
| **H4** | **Oscillatory RFT Phases** | ❌ **REJECTED** | Modulating phase $\sigma$ per block provided no sparsity gain over static $\sigma$. |
| **H5** | **Annealed Coherent Cascades** | ✅ **SUPPORTED** | Annealing the cascade parameters beats hand-tuned configurations. |
| **H6** | **AST-First Symbolic Compression** | ❌ **REJECTED** | **gzip beats RFT** on source code. RFT adds overhead for symbolic data. |
| **H8** | **Audio Transient Preservation** | ❌ **REJECTED** | RFT EQ did not preserve transients better than FFT in this test setup. |
| **H9** | **Timbre Density** | ✅ **SUPPORTED** | RFT oscillators cover a vastly larger timbre space ($10^{10}$ vs $10^8$) due to inharmonicity. |
| **H10** | **PDE Solver Stability** | ❌ **REJECTED** | RFT spectral solvers did not allow larger time steps ($dt$) than FFT solvers. |
| **H11** | **Crypto Avalanche Effect** | ✅ **SUPPORTED** | RFT cipher achieves near-ideal (~50%) avalanche effect. |
| **H12** | **Geometric Hash Collisions** | ⚠️ **PARTIAL** | RFT hashes show different collision geometry than SHA, but utility is unproven. |

## Detailed Analysis

### ✅ The Wins (Physics & Chaos)
The hypotheses that passed (H1, H9, H11) confirm the core value proposition: **RFT is excellent for generating high-entropy, inharmonic, and chaotic signals.**
- **H1 (Coherence):** Proves RFT is mathematically distinct from DCT/FFT.
- **H9 (Timbre):** Confirms the "Alien Sound" use case.
- **H11 (Crypto):** Confirms the "Mixing" capability (even if slow).

### ❌ The Losses (Standard Engineering)
The hypotheses that failed (H6, H8, H10) confirm that **RFT is not a magic replacement for standard tools.**
- **H6 (Compression):** Do not use RFT to compress text or code. Use `gzip`/`zstd`.
- **H8 (Audio EQ):** Do not use RFT for standard audio equalization. Use FFT.
- **H10 (PDEs):** Do not use RFT for standard fluid simulations expecting a speedup.

## Reproduction
To verify these results yourself, run:
```bash
python3 experiments/hypothesis_testing/hypothesis_battery_h1_h12.py
```
