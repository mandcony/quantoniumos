# QuantoniumOS Benchmark & Proof Report
_Last updated: 28 September 2025_

This report documents the reproducible benchmarks and proofs executed during the September 2025 validation effort. It supersedes earlier drafts that referenced unverified assets or results. For a living view of individual test runs, see the “Benchmark & proof run log” in `docs/TECHNICAL_SUMMARY.md`.

## 1. Validated runs

| Command | Category | Outcome | Key metrics / notes |
| --- | --- | --- | --- |
| `pytest tests/tests/test_rft_vertex_codec.py` | Tensor codec regression | ✅ 8 passed, 11 warnings (expected ANS fallback behaviour) | Runtime ≈ 1.9 s. Confirms lossless round-trips, lossy error bounds, and checksum fallbacks. |
| `pytest tests/apps/test_compressed_model_router.py` | Hybrid manifest routing | ✅ 3 passed | Runtime ≈ 4.4 s. Exercises manifest discovery, HuggingFace stubs, and tensor reconstruction via `encode_tensor_hybrid`. |
| `pytest tests/proofs/test_entangled_assembly.py` | Entangled vertex proofs | ✅ 20 passed, 1 warning | Runtime ≈ 1.4 s. Warning reports QuTiP comparison fidelity ≈ 0.468 (expected for the current harness). |
| `python direct_bell_test.py` | Bell CHSH demonstration | ✅ CHSH = 2.828427, fidelity = 1.000000 | Uses compiled kernels; pure-Python fallback matches within 1e‑7. |
| `QT_QPA_PLATFORM=offscreen python quantonium_boot.py` | Desktop boot smoke test | ⚠️ Console fallback (PyQt5 missing), validation checklist completed | Displays dependency diagnostics and confirms quick assembly validation pass. |

All commands were executed on Ubuntu 24.04 inside the dev container using Python 3.12.1 and GCC 13.2. Native kernels were built beforehand with `make -C src/assembly all`; the AddressSanitizer variant (`make -C src/assembly asan`) is also available and was exercised without error.

## 2. Reproduction guide

1. Create and activate a virtual environment; install dependencies:

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

2. (Optional) Build the native kernels:

    ```bash
    make -C src/assembly all
    ```

   Use `make -C src/assembly asan` for AddressSanitizer instrumentation when debugging native changes.

3. Execute the validation commands in the table above and compare outputs with the “Validated runs” section. Capture warning text verbatim when documenting new sessions.

## 3. Asset references

- `FINAL_AI_MODEL_INVENTORY.md` — authoritative list of model/codec artifacts in the repository.
- `ai/models/README.md` — directory-level inventory for quantum bundles and tokenizer tweaks.
- `docs/TECHNICAL_SUMMARY.md` — consolidated status report, including the benchmark log maintained alongside this file.

## 4. Historical content

Earlier benchmark narratives that mentioned unverified parameter counts or unavailable artifacts have been removed. If you need legacy material for research context, consult the “Historical appendix” within `docs/DEVELOPMENT_MANUAL.md` or the archived notes under `docs/reports/`.