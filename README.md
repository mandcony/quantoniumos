# QuantoniumOS: Symbolic Quantum-Inspired Research Platform

QuantoniumOS explores quantum-inspired data compression, simulation, and tooling on standard hardware. The platform focuses on symbolic techniques built around a Resonance Fourier Transform (RFT) rather than physical quantum processors. The repository contains executable Python and C components, PyQt5 utilities, and lossless/lossy tensor codecs used for experiments with compressed neural network models.

> **Scope statement:** Everything in this repository runs on classical CPUs. Any references to “qubits,” “quantum compression,” or extremely large parameter counts refer to symbolic encodings or compressed artifacts, not deployed quantum hardware.

## Verified functionality (September 2025)

The following capabilities were verified in this workspace by executing automated tests:

| Component | Evidence | Notes |
| --- | --- | --- |
| RFT vertex codec (`src/core/rft_vertex_codec.py`) | `pytest tests/tests/test_rft_vertex_codec.py` | Confirms round-trip accuracy for float/int/bool tensors, checksum fallbacks, and lossy mode error bounds. |
| Compressed model router (`src/apps/compressed_model_router.py`) | `pytest tests/apps/test_compressed_model_router.py` | Exercises manifest discovery, HuggingFace stub loading, and hybrid tensor reconstruction using `encode_tensor_hybrid`. |
| Console launcher (`quantonium_boot.py`) | `QT_QPA_PLATFORM=offscreen python quantonium_boot.py` | Runs in console mode when PyQt5 is missing; dependency check and validation suite succeed, GUI disabled with clear warning. |
| Model/codec inventory audit | `python - <<'PY' ... json.load(...)` / `torch.load('decoded_models/tiny_gpt2_lossless/state_dict.pt')` | Confirms 4 096 symbolic states in `quantonium_120b_quantum_states.json` and 2 300 382 parameters in the decoded tiny GPT‑2 state dict. See “Current model & codec assets”. |
| Entangled assembly harness (`tests/proofs/test_entangled_assembly.py`) | `make -C src/assembly all` then `pytest tests/proofs/test_entangled_assembly.py` | Native kernels load without crashes; entangled vertex pairs maintain fidelity error < 1e-9 via the ctypes bridge. |
| Bell CHSH check (`direct_bell_test.py`) | `make -C src/assembly all` then `python direct_bell_test.py` | Reports CHSH ≈ 2.828 with compiled kernels; pure-Python fallback score matches within 1e-7. |

Warnings observed during the suite (ANS fallback and checksum messages) are expected in lossy/quantized paths and do not impact the pass/fail status.

## Experimental or unverified areas

Large portions of the prior documentation described results that have not been reproduced in this session:

- Extended entanglement protocols (multi-party or QuTiP-driven flows under `src/engine/` and `tests/proofs/`) remain experimental beyond the single-pair CHSH harness validated above.
- Claims of tens of billions of effective parameters or million-vertex simulations originate from compressed metadata and should be treated as theoretical targets. Only the smaller HuggingFace checkpoints included in `ai/models/` and `hf_models/` are confirmed assets.
- Performance numbers in `results/` and benchmark scripts were not rerun here; treat them as historical data points until regenerated.
- Assembly bindings under `src/assembly/` provide optional optimisations. The default build (`make -C src/assembly all`) and AddressSanitizer variant (`make -C src/assembly asan`) were exercised; other optimisation flags remain untested.
- `tools/real_hf_model_compressor.py` exits early unless `hf_models/downloaded/DialoGPT-small/` exists. Running it today prints a missing-model warning instead of argument help.

If you depend on any of the above features, re-run the corresponding scripts/tests and update this README with fresh evidence.

## Current model & codec assets

| Asset | Location | Verification | Notes |
| --- | --- | --- | --- |
| GPT‑OSS quantum sample | `ai/models/quantum/quantonium_120b_quantum_states.json` | `python - <<'PY' ... json.load(...)` → 4 096 symbolic states, metadata claims 120 B original parameters and 351.9 M effective. | 2.3 MB JSON produced by `tools/generate_gpt_oss_quantum_sample.py`. |
| Tiny GPT‑2 lossless archive | `encoded_models/tiny_gpt2_lossless/` | Manifest reports 33 tensors, ≈2.9 MB original → ≈29 MB encoded; verified with `json.load`. | Complete lossless RFT bundle for `sshleifer/tiny-gpt2` plus manifest. |
| Tiny GPT‑2 decoded weights | `decoded_models/tiny_gpt2_lossless/state_dict.pt` | `torch.load(...); sum(t.numel() ...)` → 2 300 382 parameters. | Reconstructed PyTorch checkpoint derived from the encoded bundle. |
| Tokenizer tweaks | `ai/models/huggingface/tokenizer_fine_tuned.json`, `.../vocab_fine_tuned.json` | File presence only. | No full HuggingFace checkpoints reside in `ai/models/` right now; see `hf_models/` for partial caches. |

The incomplete DistilGPT‑2 chunk previously stored in `encoded_models/distilgpt2_lossless/` was deleted during this audit to avoid shipping orphaned tensors. See `FINAL_AI_MODEL_INVENTORY.md` for the full audit log and reproduction commands.

## Repository layout (abridged)

```
quantoniumos/
├── quantonium_boot.py            # Desktop launcher entry point
├── src/
│   ├── apps/                     # PyQt5 apps and tooling
│   ├── core/                     # RFT codecs, math utilities, crypto
│   ├── assembly/                 # Optional C/AVX kernels + bindings
│   ├── engine/                   # Experimental vertex + entanglement engines
│   └── frontend/                 # Desktop shell
├── tests/                        # Pytest suites and analysis scripts
├── tools/                        # Model compression/management scripts
├── ai/, encoded_models/, decoded_models/   # Sample model assets
└── docs/                         # Technical notes and reports
```

## Getting started

```bash
git clone https://github.com/mandcony/quantoniumos.git
cd quantoniumos
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

From there you can:

- Launch the desktop shell (experimental):

	```bash
	python quantonium_boot.py
	```

- Run the validated unit tests:

	```bash
	pytest tests/tests/test_rft_vertex_codec.py
	pytest tests/apps/test_compressed_model_router.py
	pytest tests/proofs/test_entangled_assembly.py
	python direct_bell_test.py
	```

	- Build the native kernels first if you want the accelerated paths:

	```bash
	make -C src/assembly all
	```

	For debugging native issues, rebuild with AddressSanitizer enabled and re-run the tests:

	```bash
	make -C src/assembly asan
	```

- Prepare compression experiments:

	1. Download `microsoft/DialoGPT-small` into `hf_models/downloaded/DialoGPT-small/` (see `tools/real_model_downloader.py`).
	2. Run `python tools/real_hf_model_compressor.py` to generate a compressed artifact and update `results/`.

## How the verified pieces fit together

- **Tensor codecs** encode NumPy arrays or PyTorch state dictionaries into JSON-friendly containers with optional pruning/quantization. They provide both lossless and lossy modes and are exercised in the unit tests listed above.
- **Model router** discovers decoded/encoded artifacts on disk, reads manifests, and instantiates lightweight HuggingFace stubs for testing. Real deployments should replace the dummy tokenizer/model with actual checkpoints.
- **Hybrid codec** (see `src/core/rft_hybrid_codec.py`) mixes RFT coefficient sparsification with amplitude/phase quantization; the router test confirms the JSON manifest format and reconstruction path.

## Recommended validation workflow

1. Run the two fast pytest modules showcased above.
2. If you need desktop features, smoke-test `quantonium_boot.py` inside a Python virtual environment with PyQt5 available.
3. To benchmark compression performance, execute the scripts under `tests/tests/` or `tests/analysis/` that correspond to your focus area, then update `docs/` with fresh measurements.

## Known limitations

- Entanglement, open quantum systems, and million-vertex benchmarks are not covered by the automated tests that currently pass. Treat them as prototypes.
- Assembly acceleration requires compiling the optional kernels; fallback Python paths are slower but functional.
- The repository ships large model artifacts that may not be fully licensed for redistribution; review `FINAL_AI_MODEL_INVENTORY.md` before packaging releases.

## Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/my-improvement`).
3. Add tests for any behavior you modify.
4. Run the validation steps above.
5. Open a pull request describing which claims you re-validated.

## License

QuantoniumOS ships under a custom license — see `LICENSE.md` for details.

---

If you extend the system or reproduce additional benchmarks, please document the commands, hardware, and outcomes so this README stays tied to verifiable results.
