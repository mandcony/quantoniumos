# RFT Validation Snapshot (Dec 2025)

## Transform Coverage
- Patent variants: 20/20 unitary (err < 1e-12)
- New experimental variants: Noise-Shrink Manifold (1D), Robust Manifold 2D (non-separable blend)

## Test Suite
- Pytest: 89 passed, 1 skipped, 0 failed
- Warnings addressed:
  - Deprecated `closed_form_rft` import → switched to `phi_phase_fft`
  - Vertex codec checksum now clips to int64 range
  - Avalanche tests now byte-view complex to avoid cast warnings

## Cross-Class & Confusion
- Manifold wins 8/12 signals; non-construction wins 5/9
- Confusion best basis counts: M-Torus 7, FFT 2, M-Lissa 3, others <=2
- Cross-validation wins: 5/9 (manifold operators generalize)

## Stability (Test 4)
- Noise σ vs SNDR (torus): 0.00→57.6 dB; 0.05→22.7 dB; 0.10→25.3 dB
- Degradation: ~39–66% across σ∈[0.01,0.50]; noise robustness remains a target

## 2D Extension
- Kronecker 2D: DCT-2D still wins on torus_2d/spiral_2d/wave_2d/radial_2d/checker_2d
- Added Robust Manifold 2D (radial blend) for follow-up benchmarking

## Fast Projection (Toeplitz–FFT Lanczos)
- k=16 eigenvectors, rel err ~3e-16 vs dense
- Matvec speedup: ~7–19× over dense for N=512–2048; eigsolve: up to 8.5× (N=2048)

## Open Follow-Ups
- Tune Noise-Shrink Manifold (σ, λ) to hit <25% degradation at σ≈0.05
- Benchmark Robust Manifold 2D vs DCT-2D on structured images
- Consider 2D operator learned directly from manifold surfaces (non-Kronecker)
