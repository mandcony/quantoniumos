# Hybrid Quant + RFT Codec

This document describes the implemented Hybrid (RFT + Learned Residual) compression path.

## Overview
Hybrid compression stores a **coarsely quantized spectral representation** (after RFT transform) plus a **tiny residual predictor** that reconstructs fine detail lost through pruning/quantization. This yields higher compression ratios than pure scalar quantization while preserving controllable reconstruction fidelity.

Pipeline:
1. Flatten tensor and perform deterministic RFT forward transform.
2. Partition coefficient indices into geometric bands.
3. Prune coefficients below amplitude threshold (optional).
4. Convert amplitudes to log domain; quantize log amplitude and phase separately.
5. Collect residual targets: (true_logA - coarse_logA, true_phase - coarse_phase).
6. Train a small MLP (shared across tensors) on aggregated samples.
7. Store: tensor hybrid containers + predictor JSON.
8. Decode: coarse dequantization + residual prediction + inverse RFT.

## File Artifacts
Directory produced by `rft_hybrid_compress.py`:
```
<out>/
  manifest_hybrid.json
  tensors/<tensor>.json            # per-tensor hybrid containers
  predictors/global_residual_predictor.json
```

Per tensor container (`tensors/<name>.json`):
```
{
  "type": "rft_hybrid_tensor",
  "version": 1,
  "dtype": "float32",
  "original_shape": [...],
  "bands": [ {"id":0, "start":0, "end":64}, ... ],
  "codec": {
      "mode": "hybrid",
      "quant_amp_bits": 6,
      "quant_phase_bits": 5,
      "prune_threshold": 1e-4,
      "residual_predictor_ref": "predictors/global_residual_predictor.json"
  },
  "sparsity": 0.82,
  "kept_coeff": 12345,
  "total_coeff": 67890,
  "bitrate_coeff": 11.0,
  "payload": {
      "indices": "...",          # base64 uint32 kept indices
      "amp_codes": "...",         # base64 uint{bits} amplitude codes
      "phase_codes": "...",       # base64 uint{bits} phase codes
      "amp_scale": [min_logA, max_logA],
      "phase_scale": [-3.14159, 3.14159]
  }
}
```

Predictor JSON (`predictors/global_residual_predictor.json`) includes architecture, weights (base64), training stats, and checksum.

## CLI Usage
### Compression
```
python tools/rft_hybrid_compress.py \
  --model-id <model or path> \
  --out encoded_models/my_model_hybrid \
  --prune-threshold 1e-4 \
  --quant-amp-bits 6 --quant-phase-bits 5 \
  --train-residual --residual-hidden 32 --residual-epochs 5
```

### Decompression
```
python tools/rft_hybrid_decode.py \
  --input-dir encoded_models/my_model_hybrid \
  --output-file restored_model.bin
```
Disable residual correction:
```
python tools/rft_hybrid_decode.py --input-dir ... --no-predictor
```

## Residual Predictor
* Architecture: tiny MLP (NumPy implementation) with configurable hidden dim and layers.
* Inputs: [idx_norm, logA_coarse, phase_coarse, band_one_hot...]
* Outputs: [delta_logA, delta_phase]
* Loss: MSE

## Rate-Distortion Control
Current implementation provides static bit allocation (uniform bits across bands). Future work: iterative re-allocation based on residual energy per band.

## Integrity
Each predictor JSON includes a SHA256 over architectural + weight payload. Tensor containers can be externally validated by recomputing transforms.

## Future Extensions
* Adaptive per-band bitwidth search.
* Entropy coding for index and code streams (rANS / Huffman).
* Streaming inverse with layer-by-layer decode.
* Direct spectral inference (operate in RFT domain without full inverse).

## License
MIT – Inherits project license.
