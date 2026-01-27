# Component Inventory

Concise map of key paths → purpose → primary routes/links. Scope: top-level files and core directories (code, docs, tests, data, tooling, hardware).

## Top-Level Files
| Path | Purpose | Routes/Notes |
| --- | --- | --- |
| `README.md` | Project overview, canonical RFT definition update, quick start | Links to `GETTING_STARTED.md`, `SETUP_GUIDE.md`, `docs/ARCHITECTURE*.md`, `algorithms/rft/README_RFT.md`, build/bench/test commands |
| `GETTING_STARTED.md` | First-use guide; quick verify and examples | Points to `SETUP_GUIDE.md`, demos, benchmarks, experiments |
| `SETUP_GUIDE.md` | Install + architecture (ASM→C→C++→Python), native builds, troubleshooting | References `algorithms/rft/kernels/*`, `src/rftmw_native` |
| `QUICK_REFERENCE.md` | Daily commands for install/test/bench/hardware; API snippets | Routes to `benchmarks/run_all_benchmarks.py`, `validate_system.py`, `algorithms/rft/core/*` |
| `DOCUMENTATION_INDEX.md` | Master doc index and task navigation | Links to system map, cleanup plans, validation docs |
| `SYSTEM_STATUS_SUMMARY.md` | Repo metrics, working/broken items, cleanup opportunities | Routes to bootstrap/verify scripts, benchmarks, system map |
| `RFT_VALIDATION_SNAPSHOT.md` | Latest validation snapshot (unitarity, robustness) | Notes deprecated import fix; follow-ups |
| `CLAIMS_AUDIT_REPORT.md` | Claim assessment (proven/overstated/false/experimental) | Highlights no sparsity advantage for φ-phase FFT |
| `PATENT_COMPLIANCE_REPORT.md` | Maps patent claims to implementations/tests | Cites `algorithms/rft/*`, kernels, crypto, hardware |
| `PATENT_NOTICE.md` | Patent notice and licensing pointer | References `CLAIMS_PRACTICING_FILES.txt`, `LICENSE-CLAIMS-NC.md` |
| `CLAIMS_PRACTICING_FILES.txt` | Files under non-commercial patent license | Applies `LICENSE-CLAIMS-NC.md` to listed paths |
| `HYBRID_INTEGRATION_PLAN.md` | Hybrid variants plan (H3/FH5/H6) | Targets `algorithms/rft/variants/registry.py`, hybrids |
| `NEW_TRANSFORM_DISCOVERY.md` | Operator-ARFT derivation/results | Adaptive eigenbasis transform notes |
| `docs/FINAL_VALIDATION_REPORT.md` | Phase 4 validation summary | Tests/benchmarks status, native regression note |
| `algorithms/rft/RFT_RESEARCH_SUMMARY.md` | Canonical RFT status (theory/fast/apps) | Lists created analysis/fast/application files |
| `LICENSE.md` | AGPL-3.0-or-later for non-claim files | Mentions split with claims list |
| `LICENSE-CLAIMS-NC.md` | Research-only license for claim files | Contact for commercial license |
| `pyproject.toml` | Project metadata, deps, package include/exclude | Runtime deps pinned; dev/ai/image extras |
| `requirements*.txt|in|lock` | Dependency pins/ranges | `requirements.in` roots; `lock` minimal pins (regenerate with pip-compile for hashes) |
| `pytest.ini` | Pytest config/markers | `integration`, `slow` markers |
| `Dockerfile` | Quick validation container | Installs dev extras; builds kernels; runs `pytest -m "not slow"` |
| `Dockerfile.papers` | Paper validation container | Minimal deps; curated validation script |
| `quantoniumos-bootstrap.sh` | Full setup (deps, venv, native builds, HW tools) | Modes: full/minimal/dev/hardware |
| `verify_setup.sh` | Quick setup check | RFT core sanity; native engines optional |
| `verify_test_wiring.py` | Ensures test discovery/import validity | Pytest collect dry-run |
| `validate_system.py` | Full system validation | Native UnitaryRFT, QuantSoundDesign engines, variants |
| `run_demo.sh` | Runs power demo | Executes `demo_rft_power.py` |
| `verify_medical_tests.py` | Imports medical test modules | Verifies key functions |
| `test_imports.py` | Sanity imports (codec, PyQt editor, Bell tests) | Hypothesis presence check |
| `codec_pipeline_results.json` | Codec benchmark outputs (rates/MSE) | Data only |
| `CITATION.cff` | Citation metadata (Zenodo DOI) | |
| `SECURITY.md` | Security policy; crypto disclaimer | |
| `DOCKER_PAPERS.md` | Paper validation via Docker | |
| `REPRODUCING_RESULTS.md` | Steps to reproduce benchmarks/tests | |

## Core Directories
| Path | Purpose | Routes/Notes |
| --- | --- | --- |
| `algorithms/` | RFT core/variants/fast/theory/hybrids/crypto/compression | See `algorithms/rft/README_RFT.md`; variants manifest; patent list applies |
| `quantoniumos/` | Package root (minimal) | Contains `quantum_search.py`; init |
| `quantonium_os_src/` | Organized apps/engine/frontend | `apps/`, `engine/`, `frontend/`; used by QuantSoundDesign and engines |
| `src/` | Native bindings + apps | `rftmw_native/` C++/ASM; `apps/` additional apps |
| `benchmarks/` | Benchmark harnesses (classes A–E, hybrids, crypto/audio/compression) | `run_all_benchmarks.py`, `test_all_hybrids.py` |
| `experiments/` | Research experiments (entropy, fibonacci, runtime, SOTA, etc.) | Generates results under `results/` |
| `examples/` | Usage demos | e.g., `routing_integration_demo.py` |
| `tests/` | Pytest suites for algorithms/benchmarks/validation | Markers `slow`, `integration`; wiring verified by `verify_test_wiring.py` |
| `docs/` | Documentation tree (architecture, proofs, validation, manuals, reports) | Indexed by `DOCUMENTATION_INDEX.md` |
| `hardware/` | RTL, testbenches, simulation outputs, viz scripts, synthesis reports | `fpga_top.sv`, `tb_*`, `verify_fixes.sh`, viz scripts, WebFPGA reports |
| `papers/` | Paper assets (LaTeX, figures) | Used by validation container |
| `figures/` | Images, GIFs, LaTeX data, analysis plots | |
| `data/` | Result JSONs and configs | `quantum_compression_results.json`, scaling configs |
| `tools/` | Utilities: benchmarking, compression, crypto, model mgmt, dev tools | Mixed scripts/modules |
| `scripts/` | Entry-point scripts: validation, benchmarks, figures, paper helpers | `run_full_suite.sh`, `verify_scaling_laws.py`, figure generators |
| `ui/` | UI assets (styles, icons) | |
| `quantonium-mobile/` | React Native mobile app | `package.json`, TS/JS src |
| `release/` | Release artifacts/notes | Benchmark zip, README |

## Licensing Split (reference)
- AGPL-3.0-or-later: default for all files not listed in `CLAIMS_PRACTICING_FILES.txt`.
- Research-only (non-commercial) patent-practicing files: listed in `CLAIMS_PRACTICING_FILES.txt`, governed by `LICENSE-CLAIMS-NC.md`.

## How to Use This Inventory
- For navigation: jump to path → see purpose → follow routes/notes for deeper docs/code.
- For compliance: check `CLAIMS_PRACTICING_FILES.txt` when touching RFT core, kernels, crypto, hybrids, hardware, DAW files.
- For onboarding: start at `README.md` → `GETTING_STARTED.md` → `SETUP_GUIDE.md` → `DOCUMENTATION_INDEX.md`.
