# Benchmark Methodology (Experimental)

This document defines a reproducible plan for comparing experimental Φ-RFT-based codecs against established baselines (JPEG XL, AVIF) and, in future, model quantization approaches (e.g., Llama.cpp).

## Scope
- Current focus: small image set compression.
- Future: AI weight tensors, text embeddings, audio segments.

## Baselines
- **JPEG XL** (`cjxl` / `djxl`) – reference high-performance perceptual/efficient image codec.
- **AVIF** (`avifenc` / `avifdec`) – AV1-based still image coding.
- **Φ-RFT Vertex / Hybrid Codecs** – experimental, not production-ready.

## Metrics
| Category | Metric | Description |
|----------|--------|-------------|
| Size | Bytes | Compressed artifact size |
| Ratio | CR | Original bytes / compressed bytes |
| Quality | PSNR | Peak Signal-to-Noise Ratio (RGB) |
| Quality | SSIM | Structural Similarity Index |
| Time | Encode / Decode (s) | Wall-clock times per image |
| Future | LPIPS / VMAF | Optional perceptual quality (not yet integrated) |

## Data Sets
Initial evaluation will use a handful (5–20) of small, public-domain images placed under `tests/data/images/`. Larger sets (Kodak, Tecnick) may be added later with clear licensing notes.

## Procedure
1. Ensure system packages installed: `libjxl-tools`, `libavif-tools`.
2. Populate `tests/data/images/` with PNG/JPG files.
3. Run harness:
   ```bash
   python tests/benchmarks/rft_sota_comparison.py --images tests/data/images --quality 75 --rft-mode vertex
   ```
4. Inspect outputs: `benchmark_results_rft_vs_sota.json`, `benchmark_results_rft_vs_sota.md`.
5. Iterate with different `--quality` values (e.g. 50, 90) and `--rft-mode hybrid`.

## Verification Levels
- Baseline tool metrics (JPEG XL, AVIF) are deterministic given identical binaries and inputs.
- Φ-RFT metrics use the **verified closed-form implementation** (`rft_forward`/`rft_inverse` from `closed_form_rft.py`).
- Current benchmarks use separable 2D RFT (row-wise then column-wise) with top-K coefficient retention.
- No performance advantage claims are to be labeled **[VERIFIED]** until:
  - A CI job runs the harness on a fixed dataset.
  - Statistical variance across ≥3 runs is reported.

## Planned Extensions
| Area | Description | Status |
|------|-------------|--------|
| Tensor Compression | Evaluate model weight blocks (e.g., 4K parameter slices) | Planned |
| Quantization Comparison | Compare Φ-RFT transform preconditioning vs Llama.cpp quant outputs | Planned |
| Perceptual Metrics | Integrate LPIPS for visual similarity | Planned |
| Parallel Scaling | Benchmark multi-core throughput (batch encode) | Planned |

## Limitations
- RFT codec uses simple top-K coefficient retention (no entropy coding yet).
- External tool versions (libjxl, libavif) affect results; pin versions in future lockfile.
- Image set choice can bias outcomes; must disclose dataset list with each published table.

## Next Steps
1. ~~Replace stub with actual vertex codec forward + inverse path.~~ **DONE** - Now uses `closed_form_rft.py`.
2. Add integrity check (RMS / PSNR vs original) for claimed "near-lossless" cases.
3. Automate environment setup script to ensure reproducibility.
4. Add CLI option to emit raw metric CSV for external statistical analysis.
5. Integrate ANS entropy coding from `algorithms/rft/compression/ans.py`.

---
*This methodology file is experimental and will evolve as codecs mature.*