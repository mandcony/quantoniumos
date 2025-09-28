# QuantoniumOS – Model & Codec Inventory (September 2025)

Earlier versions of this document referenced dozens of quantum-compressed models and multi-gigabyte HuggingFace checkpoints that are **not** present in the current repository snapshot. This page lists the assets that actually exist, along with the evidence collected during the latest audit.

## Summary

| Category | Assets on disk | Verified details |
| --- | --- | --- |
| Quantum state bundles | `ai/models/quantum/quantonium_120b_quantum_states.json` | 2.3 MB JSON; 4 096 symbolic states; metadata claims 120 B original params and 351.9 M effective. Loaded via `json.load` to confirm structure. |
| RFT-encoded tensors | `encoded_models/tiny_gpt2_lossless/` | 33 chunk files + manifest for `sshleifer/tiny-gpt2`; manifest reports ≈2.9 MB original → ≈29.0 MB encoded, lossless (no lossy chunks). |
| Decoded checkpoints | `decoded_models/tiny_gpt2_lossless/state_dict.pt` | PyTorch state dict with 2 300 382 parameters (validated with `torch.load`). |
| Tokenizer resources | `ai/models/huggingface/tokenizer_fine_tuned.json`, `.../vocab_fine_tuned.json` | Custom JSON token data. |
| HuggingFace cache skeleton | `hf_models/models--runwayml--stable-diffusion-v1-5/` | Config directories present; large weight files omitted (≈68 KB on disk). Use `tools/real_model_downloader.py` to fetch full assets if required. |

All other directories mentioned in older reports (`ai/models/compressed/`, additional quantum JSON files, multi-GB checkpoints) are absent.

## Detailed notes

### Quantum bundle

```bash
python - <<'PY'
import json
path = 'ai/models/quantum/quantonium_120b_quantum_states.json'
with open(path) as fh:
    data = json.load(fh)
print('states:', len(data['quantum_states']))
print('metadata:', data['metadata'])
PY
```

Output confirms 4 096 states and the metadata values listed above. This file is a symbolic representation only; reconstructing the full 120 B-parameter model would require additional tooling.

### Tiny GPT‑2 RFT archive

```bash
python - <<'PY'
import json
with open('encoded_models/tiny_gpt2_lossless/manifest.json') as fh:
    bundle = json.load(fh)
manifest = bundle['manifests'][0]
print('original_bytes', manifest['metrics']['original_size_bytes'])
print('encoded_bytes', manifest['metrics']['encoded_size_bytes'])
print('tensor_count', len(manifest['tensors']))
PY
```

This verifies that the repository contains all 33 lossless tensors for `sshleifer/tiny-gpt2`. `decoded_models/tiny_gpt2_lossless/state_dict.pt` reconstructs to a 2.3 M-parameter PyTorch checkpoint:

```bash
python - <<'PY'
import torch
state = torch.load('decoded_models/tiny_gpt2_lossless/state_dict.pt', map_location='cpu')
print(sum(t.numel() for t in state.values()))
PY
```

### Removed DistilGPT‑2 fragment

An orphaned 224 MB chunk (`transformer_h_0_attn_c_attn_weight.json`) formerly stored in `encoded_models/distilgpt2_lossless/` has been deleted (28 September 2025). No manifest or companion tensors were available, so the asset could not be decoded or validated. Regenerate a fresh DistilGPT-2 bundle with the compression scripts if you require that model.

### Tooling expectations

`tools/real_hf_model_compressor.py` expects a downloaded model at `hf_models/downloaded/DialoGPT-small`. Running the script without that directory prints:

```
✅ Using QuantoniumOS RFT engine
❌ Model not found at .../hf_models/downloaded/DialoGPT-small
```

Download the checkpoint before invoking the compressor.

## Action items

1. Regenerate any missing quantum or PKL archives if you rely on the legacy documentation.
2. Update this file whenever new assets or validations are added. Include exact command outputs, file sizes, and parameter counts so future audits remain transparent.

_Last updated: 28 September 2025_