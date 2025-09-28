# QuantoniumOS AI Models Directory
# QuantoniumOS AI Models (September 2025)

This directory now contains a minimal, reality-checked set of model assets. Earlier documentation referenced many additional quantum-compressed files, but those artifacts are **not** present in the repository snapshot you are reading. Use this page to understand exactly what exists on disk today and what needs to be recreated if you require richer inventories.

## Current contents

```
ai/models/
├── quantum/
│   └── quantonium_120b_quantum_states.json   # 2.3 MB JSON, 4 096 symbolic states
├── huggingface/
│   ├── tokenizer_fine_tuned.json            # Custom vocabulary (JSON)
│   └── vocab_fine_tuned.json                # Companion vocab file
└── scripts/
    ├── gpt_oss_120b_memory_optimized.py
    ├── gpt_oss_120b_quantum_integrator.py
    └── gpt_oss_120b_streamlined_integrator.py
```

That’s it. There is currently **one** quantum-state bundle, a pair of tokenizer assets, and three Python utilities.

### `quantum/quantonium_120b_quantum_states.json`

* Size: `2.3 MB`
* Metadata (parsed at runtime):
  * `original_parameters`: 120 000 000 000 (reported by the file, not reconstructed here)
  * `effective_parameters`: 351 906 158
  * `quantum_states_count`: 4 096
  * `compression_method`: `rft_golden_ratio_streaming`
* Contents: per-layer resonance/amplitude/phase records produced by `tools/generate_gpt_oss_quantum_sample.py`.
* Verified via:

  ```bash
  python - <<'PY'
  import json
  with open('ai/models/quantum/quantonium_120b_quantum_states.json') as fh:
      data = json.load(fh)
  print(len(data['quantum_states']))
  PY
  ```

### Tokenizer assets

`ai/models/huggingface/tokenizer_fine_tuned.json` and `.../vocab_fine_tuned.json` hold small, project-specific tokenizer adjustments. No full HuggingFace checkpoints live inside `ai/models/` right now.

## Related assets outside `ai/models/`

| Location | Description | Verification |
| --- | --- | --- |
| `encoded_models/tiny_gpt2_lossless/` | 33 JSON chunk files + manifest for `sshleifer/tiny-gpt2`. Lossless RFT encoding. | `python - <<'PY' ...` shows 33 tensors, original ≈2.9 MB → encoded ≈29 MB. |
| `decoded_models/tiny_gpt2_lossless/state_dict.pt` | PyTorch state dict reconstructed from encoded bundle. | `torch.load(...); sum(numel)` reports 2 300 382 parameters. |
| `hf_models/` | Partial HuggingFace cache for Stable Diffusion v1-5 (config + directories, heavy weights absent). | `du -sh hf_models/...` reports ≈68 KB; expect to download weights separately. |

If you need other quantum-compressed JSON files or PKL.GZ archives mentioned in historical docs, you must regenerate them with the scripts in `tools/`—they are not tracked in this commit. The orphaned DistilGPT‑2 chunk that previously lived under `encoded_models/distilgpt2_lossless/` was removed on 28 September 2025 to keep this tree limited to reproducible assets.

## Regenerating additional assets

1. Download the desired base model into `hf_models/downloaded/<ModelName>` using `tools/real_model_downloader.py` or the HuggingFace CLI.
2. Run the relevant compressor script (for example, `python tools/real_hf_model_compressor.py`) to create new JSON/PKL outputs under `data/` or `ai/models/`.
3. Move or rename the generated files into the structure above and update the documentation with sizes, dates, and verification steps.

Keep this README in sync with the actual tree so other contributors can understand which artifacts are available without rerunning large pipelines.