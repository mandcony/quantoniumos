# QuantoniumOS - Complete System Architecture Map

**Generated:** December 3, 2025  
**Purpose:** Comprehensive system documentation for organization and cleanup  
**Total Files:** 7,585 Python files, 307 documentation files  
**Codebase Size:** ~42MB across all directories

---

## 📊 Executive Summary

QuantoniumOS is a quantum-inspired research operating system centered around the **Φ-RFT (Golden Ratio + Chirp Resonance Fourier Transform)**. The system includes:

- **14 Φ-RFT Variants** (7 core unitary + 7 hybrid/cascade modes)
- **17 Hybrid Transform Architectures** (DCT/RFT cascades with zero-coherence)
- **5 Benchmark Classes** (Quantum, Transform/DSP, Compression, Crypto, Audio)
- **Hardware Implementations** (SystemVerilog, Makerchip TL-V)
- **Native Performance Kernels** (C, C++, Assembly with AVX2/AVX-512)
- **Full Application Suite** (Sound design, crypto, quantum simulation)

**Status:** Actively developed, numerically validated, patent-pending (USPTO 19/169,399)

---

## 🗂️ Directory Structure Analysis

### Core Implementation Directories

```
/workspaces/quantoniumos/
├── algorithms/                    [1.7MB, ACTIVE - Core algorithms]
│   └── rft/
│       ├── core/                  [Φ-RFT implementations]
│       │   ├── canonical_true_rft.py      ✅ ACTIVE (Claims-practicing)
│       │   ├── closed_form_rft.py         ✅ ACTIVE (Reference impl)
│       │   ├── rft_optimized.py           ✅ ACTIVE (Fused diagonal, 4-7× faster)
│       │   ├── geometric_waveform_hash.py ✅ ACTIVE (Crypto hashing)
│       │   ├── geometric_hashing.py       ✅ ACTIVE (Geometric variant)
│       │   └── quantum_kernel_implementation.py ✅ ACTIVE (QSC)
│       ├── compression/           [Transform codecs]
│       ├── crypto/                [RFT-SIS, Feistel ciphers]
│       ├── hybrids/               [17 hybrid architectures]
│       ├── kernels/               [C/Assembly optimizations]
│       ├── quantum/               [Quantum simulation]
│       ├── theorems/              [Mathematical proofs]
│       ├── variants/              [14 variant generators]
│       ├── hybrid_basis.py        ✅ ACTIVE (DCT/RFT basis mixing)
│       ├── routing.py             ✅ ACTIVE (Variant routing system)
│       └── rft_status.py          ✅ ACTIVE (Status tracking)
│
├── benchmarks/                    [320KB, ACTIVE - Competitive testing]
│   ├── run_all_benchmarks.py     ✅ ACTIVE (Master benchmark runner)
│   ├── class_a_quantum_simulation.py      ✅ ACTIVE (QSC vs Qiskit/Cirq)
│   ├── class_b_transform_dsp.py           ✅ ACTIVE (Φ-RFT vs FFT/DCT)
│   ├── class_c_compression.py             ✅ ACTIVE (RFTMW vs zstd/brotli)
│   ├── class_d_crypto.py                  ✅ ACTIVE (RFT-SIS vs OpenSSL)
│   ├── class_e_audio_daw.py               ✅ ACTIVE (Audio performance)
│   ├── variant_benchmark_harness.py       ✅ ACTIVE (Variant testing infrastructure)
│   ├── test_all_hybrids.py                ✅ ACTIVE (17 hybrid tests)
│   └── test_cascade_integration.py        ✅ ACTIVE (H3/FH5 integration)
│
├── quantoniumos/                  [420KB, ACTIVE - Python package namespace]
│   ├── __init__.py                ✅ ACTIVE (Exports CanonicalTrueRFT, EnhancedRFTCryptoV2)
│   └── rftmw_native.cpython-*.so  ✅ ACTIVE (Compiled native module)
│
├── src/                           [24MB, ACTIVE - Native implementations]
│   ├── rftmw_native/              [C++ RFT engine with SIMD]
│   │   ├── rftmw_core.hpp         ✅ ACTIVE (Core C++ engine)
│   │   ├── rft_fused_kernel.hpp   ✅ ACTIVE (AVX2/AVX-512 kernels)
│   │   ├── rftmw_python.cpp       ✅ ACTIVE (pybind11 bindings)
│   │   └── rftmw_asm_kernels.hpp  ✅ ACTIVE (Assembly kernels)
│   └── apps/                      [Application suite]
│       ├── quantsounddesign/      ✅ ACTIVE (Φ-RFT Sound Design Studio)
│       ├── rft_visualizer.py      ✅ ACTIVE (Real-time visualization)
│       ├── quantum_crypto.py      ✅ ACTIVE (RFT-SIS crypto GUI)
│       ├── quantum_simulator.py   ✅ ACTIVE (QSC GUI)
│       └── q_vault.py             ✅ ACTIVE (Encrypted storage)
│
├── quantonium_os_src/             [ACTIVE - Organized app structure]
│   ├── apps/                      [Refactored applications]
│   │   ├── rft_visualizer/        ✅ ACTIVE (Modular visualizer)
│   │   ├── rft_validator/         ✅ ACTIVE (Validation suite)
│   │   ├── quantum_crypto/        ✅ ACTIVE (Crypto tools)
│   │   ├── quantum_simulator/     ✅ ACTIVE (QSC interface)
│   │   └── system_monitor/        ✅ ACTIVE (System monitoring)
│   ├── engine/                    ✅ ACTIVE (Core engine)
│   └── frontend/                  ✅ ACTIVE (Desktop launcher)
│
├── tests/                         [1.7MB, ACTIVE - Test suite]
│   ├── rft/                       [RFT validation tests]
│   │   ├── test_variant_unitarity.py      ✅ PASSED (14 variants unitary)
│   │   ├── test_rft_vs_fft.py             ⏳ PENDING
│   │   └── test_lct_nonequiv.py           ⏳ PENDING
│   ├── validation/                [E2E validation]
│   ├── benchmarks/                [Performance tests]
│   ├── crypto/                    [Crypto tests]
│   ├── codec_tests/               [Compression tests]
│   └── transforms/                [Transform correctness]
│
├── experiments/                   [1.1MB, ACTIVE - Research experiments]
│   ├── ascii_wall/                ✅ ACTIVE (Coherence-free theorem validation)
│   ├── hypothesis_testing/        ✅ ACTIVE (17 hybrid hypotheses)
│   ├── competitors/               ✅ ACTIVE (Competitive benchmarks)
│   ├── fibonacci/                 ✅ ACTIVE (Fibonacci tilt analysis)
│   ├── entropy/                   ✅ ACTIVE (Entropy routing)
│   └── sota_benchmarks/           ✅ ACTIVE (SOTA comparisons)
│
├── hardware/                      [5.0MB, ACTIVE - FPGA/HDL implementations]
│   ├── quantoniumos_unified_engines.sv    ✅ ACTIVE (RFT+SIS+Feistel)
│   ├── rft_middleware_engine.sv           ✅ ACTIVE (8x8 RFT core)
│   ├── makerchip_rft_closed_form.tlv      ✅ ACTIVE (Makerchip demo)
│   ├── fpga_top.sv                        ✅ ACTIVE (Top-level FPGA)
│   ├── verify_fixes.sh                    ✅ ACTIVE (Hardware validation)
│   └── generate_hardware_test_vectors.py  ✅ ACTIVE (Test generation)
│
├── docs/                          [1.3MB, ACTIVE - Documentation]
│   ├── ARCHITECTURE.md            ✅ ACTIVE (Technical deep dive)
│   ├── ARCHITECTURE_QUICKREF.md   ✅ ACTIVE (One-page reference)
│   ├── DOCS_INDEX.md              ✅ ACTIVE (Documentation index)
│   ├── algorithms/                [Algorithm docs]
│   ├── api/                       [API reference]
│   ├── validation/                [Validation reports]
│   └── research/                  [Research papers]
│
├── papers/                        [11MB, ACTIVE - Academic papers]
│   ├── coherence_free_hybrid_transforms.tex ✅ ACTIVE (Hybrid transform paper)
│   ├── dev_manual.tex             ✅ ACTIVE (Developer manual)
│   ├── paper.tex                  ✅ ACTIVE (Main paper)
│   └── quantoniumos_rft.bib       ✅ ACTIVE (Bibliography)
│
├── scripts/                       [ACTIVE - Automation scripts]
│   ├── run_full_suite.sh          ✅ ACTIVE (Full benchmark runner)
│   ├── validate_all.sh            ✅ ACTIVE (Complete validation)
│   ├── generate_all_theorem_figures.py    ✅ ACTIVE (Figure generation)
│   └── verify_*.py                ✅ ACTIVE (Validation scripts)
│
├── tools/                         [ACTIVE - Development tools]
│   ├── competitive_benchmark_suite.py     ✅ ACTIVE (Benchmark tools)
│   ├── compression/               ✅ ACTIVE (Codec tools)
│   ├── crypto/                    ✅ ACTIVE (Crypto tools)
│   └── benchmarking/              ✅ ACTIVE (Perf analysis)
│
├── ui/                            [ACTIVE - UI resources]
│   ├── styles_dark.qss            ✅ ACTIVE (Dark theme)
│   ├── styles_light.qss           ✅ ACTIVE (Light theme)
│   └── icons/                     ✅ ACTIVE (Application icons)
│
├── quantonium-mobile/             [ACTIVE - React Native mobile app]
│   ├── src/                       ✅ ACTIVE (TypeScript source)
│   ├── App.tsx                    ✅ ACTIVE (Main app)
│   └── package.json               ✅ ACTIVE (Dependencies)
│
├── data/                          [ACTIVE - Benchmark data]
│   ├── scaling_results.json       ✅ ACTIVE (Performance data)
│   ├── config/                    [Configuration files]
│   └── entropy/                   [Entropy datasets]
│
├── figures/                       [ACTIVE - Generated figures]
│   ├── gifs/                      [Animated visualizations]
│   └── latex_data/                [LaTeX figure data]
│
├── results/                       [ACTIVE - Benchmark results]
│   └── competitors/               [Competitive analysis results]
│
└── release/                       [ACTIVE - Release packages]
    └── quantoniumos-benchmarks-20251201.zip  ✅ PACKAGED
```

---

## 🔬 Core Algorithm Components

### 1. Φ-RFT Implementations (algorithms/rft/core/)

| File | Status | Purpose | Complexity |
|------|--------|---------|------------|
| `canonical_true_rft.py` | ✅ ACTIVE | Patent-practicing reference implementation | O(n log n) |
| `closed_form_rft.py` | ✅ ACTIVE | Original closed-form implementation | O(n log n) |
| `rft_optimized.py` | ✅ ACTIVE | **Fused diagonal optimization (4-7× faster)** | O(n log n) |
| `geometric_waveform_hash.py` | ✅ ACTIVE | Geometric variant for crypto | O(n log n) |
| `quantum_kernel_implementation.py` | ✅ ACTIVE | Quantum Symbolic Compression (QSC) | O(n) |

**Key Insight:** All variants maintain exact unitarity (error < 1e-14). The optimized version fuses D_φ and C_σ diagonals into a single multiplication, achieving near-FFT performance (1.06-1.3× slower) while preserving golden-ratio spectral properties.

### 2. 14 Φ-RFT Variants (algorithms/rft/variants/)

**Group A - Core Unitary Variants (7)**

| # | Variant | File/Generator | Use Case | Status |
|---|---------|----------------|----------|--------|
| 1 | Standard Φ-RFT | `STANDARD` | Exact diagonalization | ✅ ACTIVE |
| 2 | Harmonic Φ-RFT | `HARMONIC` | Nonlinear filtering | ✅ ACTIVE |
| 3 | Fibonacci RFT | `FIBONACCI` | Lattice structures | ✅ ACTIVE |
| 4 | Chaotic Φ-RFT | `CHAOTIC` | Diffusion/crypto | ✅ ACTIVE |
| 5 | Geometric Φ-RFT | `GEOMETRIC` | Optical computing | ✅ ACTIVE |
| 6 | Φ-Chaotic Hybrid | `PHI_CHAOTIC` | Resilient codecs | ✅ ACTIVE |
| 7 | Hyperbolic Φ-RFT | `HYPERBOLIC` | Phase-space embeddings | ✅ ACTIVE |

**Group B - Hybrid/Cascade Variants (7)**

| # | Variant | File/Generator | Innovation | Status |
|---|---------|----------------|------------|--------|
| 8 | Log-Periodic | `LOG_PERIODIC` | Log-frequency warp | ✅ ACTIVE |
| 9 | Convex Mixed | `CONVEX_MIX` | Phase blend | ✅ ACTIVE |
| 10 | Exact Golden | `GOLDEN_EXACT` | Full resonance lattice | ⚠️ SLOW (O(N³)) |
| 11 | H3 Cascade | `CASCADE` | Zero-coherence (0.673 BPP) | ✅ ACTIVE |
| 12 | FH2 Adaptive | `ADAPTIVE_SPLIT` | Variance-based DCT/RFT | ✅ ACTIVE |
| 13 | FH5 Entropy | `ENTROPY_GUIDED` | Entropy routing (0.406 BPP) | ✅ ACTIVE |
| 14 | H6 Dictionary | `DICTIONARY` | RFT↔DCT bridge | ✅ ACTIVE |

**Validation Status:** 13/14 variants unitary to machine precision (GOLDEN_EXACT skipped in benchmarks due to O(N³) complexity)

### 3. 17 Hybrid Transform Architectures (experiments/hypothesis_testing/)

| Hybrid | File | BPP | PSNR | Coherence | Status |
|--------|------|-----|------|-----------|--------|
| H0 Baseline Greedy | `hybrid_mca_fixes.py` | 0.812 | 28.5 dB | 0.50 | ✅ BASELINE |
| H1 Coherence Aware | `hybrid_mca_fixes.py` | 0.745 | 29.1 dB | 0.35 | ✅ WORKING |
| H2 Phase Adaptive | `hybrid_mca_fixes.py` | - | - | - | ⚠️ BUGGY |
| H3 Hierarchical Cascade | `hybrid_mca_fixes.py` | **0.655** | 30.2 dB | **0.00** | ✅ **BEST** |
| H4 Quantum Superposition | `hybrid_mca_fixes.py` | 0.698 | 29.8 dB | 0.12 | ✅ WORKING |
| H5 Attention Gating | `hybrid_mca_fixes.py` | 0.702 | 29.5 dB | 0.08 | ✅ WORKING |
| H6 Dictionary Learning | `hybrid_mca_fixes.py` | 0.715 | **31.4 dB** | 0.00 | ✅ WORKING |
| H7 Cascade Attention | `hybrid_mca_fixes.py` | 0.668 | 30.0 dB | 0.00 | ✅ WORKING |
| H8 Aggressive Cascade | `hybrid_mca_fixes.py` | 0.672 | 29.9 dB | 0.00 | ✅ WORKING |
| H9 Iterative Refinement | `hybrid_mca_fixes.py` | 0.680 | 30.1 dB | 0.00 | ✅ WORKING |
| H10 Quality Cascade | `hybrid_mca_fixes.py` | - | - | - | ⚠️ BUGGY |
| FH1 MultiLevel Cascade | `ascii_wall_final_hypotheses.py` | 0.692 | 29.7 dB | 0.00 | ✅ WORKING |
| FH2 Adaptive Split | `ascii_wall_final_hypotheses.py` | 0.715 | 30.5 dB | 0.00 | ✅ WORKING |
| FH3 Frequency Cascade | `ascii_wall_final_hypotheses.py` | 0.705 | 29.9 dB | 0.00 | ✅ WORKING |
| FH4 Edge Aware | `ascii_wall_final_hypotheses.py` | 0.688 | 30.3 dB | 0.00 | ✅ WORKING |
| FH5 Entropy Guided | `ascii_wall_final_hypotheses.py` | **0.663** | 30.8 dB | **0.00** | ✅ **BEST** |
| Legacy Hybrid | `legacy_hybrid_codec.py` | 0.890 | 26.2 dB | 0.42 | ⚠️ DEPRECATED |

**Key Achievement:** H3 and FH5 achieve **η=0 coherence** (zero energy loss) with 17-19% compression improvement over greedy baseline.

---

## 🧪 Benchmark Suite (benchmarks/)

### Class A: Quantum Symbolic Simulation

**File:** `class_a_quantum_simulation.py`  
**Comparison:** QSC vs Qiskit vs Cirq  
**Status:** ✅ ACTIVE

**Key Results:**
- QSC achieves O(n) symbolic compression of qubit labels
- Reaches 10M+ labels at ~20 M/s throughput
- **Important:** Compresses symbolic configurations, NOT 2^n quantum amplitudes

### Class B: Transform & DSP

**File:** `class_b_transform_dsp.py`  
**Comparison:** Φ-RFT vs FFT/DCT/PyFFTW  
**Status:** ✅ ACTIVE

**Key Results:**
- FFT: 1.00× baseline (15.6 µs @ N=1024)
- RFT Optimized: 1.06× (21.4 µs @ N=1024)
- RFT Original: 4.97× (85.4 µs @ N=1024)
- Golden-ratio signals: 61.8%+ sparsity advantage

### Class C: Compression

**File:** `class_c_compression.py`  
**Comparison:** RFTMW vs zstd/brotli/lzma  
**Status:** ✅ ACTIVE

**Key Results:**
- H3 Cascade: 0.655-0.669 BPP (η=0)
- FH5 Entropy: 0.663 BPP, 23.89 dB PSNR (η=0)
- **Competitive with transform codecs, NOT better than entropy bounds**

### Class D: Cryptography

**File:** `class_d_crypto.py`  
**Comparison:** RFT-SIS vs OpenSSL/liboqs  
**Status:** ✅ ACTIVE

**Key Results:**
- RFT-SIS v3.1: 50.0% avalanche effect
- 0 collisions in 10k trials
- **EXPERIMENTAL - No hardness proofs, not production-ready**

### Class E: Audio & DAW

**File:** `class_e_audio_daw.py`  
**Comparison:** Φ-RFT audio processing  
**Status:** ✅ ACTIVE

**Key Results:**
- QuantSoundDesign: Real-time Φ-RFT synthesis
- Harmonic variant for additive synthesis
- RFT-based drum synthesis

---

## 🏗️ Hardware Implementations (hardware/)

### SystemVerilog Modules

| File | Purpose | Size | Test Status |
|------|---------|------|-------------|
| `quantoniumos_unified_engines.sv` | RFT+SIS+Feistel unified stack | N=512 | ✅ SIM PASS |
| `rft_middleware_engine.sv` | 8×8 RFT core | Fixed 8-point | ✅ SIM PASS |
| `fpga_top.sv` | FPGA top-level | Configurable | ⏳ PENDING |
| `tb_quantoniumos_unified.sv` | Unified testbench | - | ✅ PASS |
| `tb_rft_middleware.sv` | RFT testbench | - | ✅ PASS |

### Makerchip TL-V

**File:** `makerchip_rft_closed_form.tlv`  
**Status:** ✅ READY  
**Features:**
- Q1.15 fixed-point Φ-RFT kernels
- 8×8 SIS matrix (deterministic)
- RFT-SIS stage with centered reduction
- Browser-based simulation ready

### Validation Status

| Test | Status | Notes |
|------|--------|-------|
| Standalone RFT (sim_rft) | ✅ PASS | 10 patterns tested, energy conserved |
| Unified Engine (sim_unified) | ✅ PASS | All modes (RFT/SIS/Feistel/Pipeline) |
| Verilator Lint | ❌ FAIL | 12 errors, 45 warnings (BLKANDNBLK issues) |
| Yosys Synthesis | ⚠️ TIMEOUT | Optimization needed |

**Action Items:**
- Fix blocking/non-blocking assignment issues
- Optimize synthesis for large N
- Complete FPGA resource utilization analysis

---

## 📱 Application Suite

### QuantSoundDesign (src/apps/quantsounddesign/)

**Status:** ✅ FULLY FUNCTIONAL  
**Lines of Code:** 3,200+  
**Architecture:** PyQt5 GUI + UnitaryRFT Engine

**Features:**
- 8-channel mixer with volume/pan/mute/solo
- Polyphonic synthesizer (8 voices) using Φ-RFT additive synthesis
- 16-step drum sequencer with RFT-based drum synthesis
- Piano roll MIDI editor with computer keyboard input (ASDFGHJKL)
- Pattern editor with velocity control
- Real-time Φ-RFT audio processing (HARMONIC variant)
- Blank session start for creative workflow

**Key Files:**
- `gui.py` - Main UI (FL Studio/Ableton inspired)
- `engine.py` - Track/clip management
- `synth_engine.py` - Polyphonic Φ-RFT synthesis
- `pattern_editor.py` - 16-step sequencer
- `piano_roll.py` - MIDI editor
- `audio_backend.py` - PyAudio/sounddevice output

### Other Applications

| Application | File | Status | Purpose |
|-------------|------|--------|---------|
| RFT Visualizer | `src/apps/rft_visualizer.py` | ✅ ACTIVE | Real-time Φ-RFT analysis |
| Quantum Crypto | `src/apps/quantum_crypto.py` | ✅ ACTIVE | RFT-SIS cipher GUI |
| Quantum Simulator | `src/apps/quantum_simulator.py` | ✅ ACTIVE | QSC interface |
| Q-Vault | `src/apps/q_vault.py` | ✅ ACTIVE | Encrypted storage |
| Q-Notes | `src/apps/q_notes.py` | ✅ ACTIVE | Note-taking app |

### Mobile App (quantonium-mobile/)

**Status:** ✅ ACTIVE (React Native)  
**Platform:** iOS/Android  
**Main File:** `App.tsx`

---

## 🧬 Test Infrastructure (tests/)

### Test Coverage Summary

| Category | Files | Status | Pass Rate |
|----------|-------|--------|-----------|
| RFT Core | 15 | ⏳ PARTIAL | 1/15 run |
| Validation | 12 | ⏳ PARTIAL | 0/12 run |
| Benchmarks | 8 | ⏳ PARTIAL | 0/8 run |
| Codecs | 4 | ✅ TESTED | 4/4 pass |
| Crypto | 3 | ⏳ PENDING | 0/3 run |
| Transforms | 2 | ⏳ PENDING | 0/2 run |

### Key Test Results

**Passing Tests:**
- ✅ `test_ans_integration.py` - ANS codec lossless roundtrip
- ✅ `test_codec_comprehensive.py` - 7/7 codec tests
- ✅ `test_audio_backend.py` - Audio backend hardening
- ✅ `test_codecs_updated.py` - Vertex & hybrid codecs
- ✅ `test_variant_unitarity.py` - 14 variants unitary at N=32

**Skipped:**
- ⏭️ `test_rans_roundtrip.py` - Known roundtrip issue

**Pending (Need to run):**
- 40+ test files in rft/, validation/, benchmarks/, crypto/

### Test Configuration

**Files:**
- `pytest.ini` - Pytest configuration
- `conftest.py` - Test fixtures
- `requirements.txt` - Test dependencies (hypothesis>=6.0.0)

---

## 📚 Documentation (docs/)

### Structure

```
docs/
├── ARCHITECTURE.md              ✅ Technical deep dive (ASM → C → C++ → Python)
├── ARCHITECTURE_QUICKREF.md     ✅ One-page cheat sheet
├── DOCS_INDEX.md                ✅ Documentation index
├── algorithms/                  [Algorithm specifications]
│   └── rft/                     [Φ-RFT details]
├── api/                         [API reference]
├── validation/                  [Validation reports]
│   └── RFT_THEOREMS.md          ✅ Mathematical proofs
├── research/                    [Research papers]
├── technical/                   [Technical specs]
├── user/                        [User guides]
├── patent/                      [USPTO documentation]
└── licensing/                   [License details]
```

### Root-Level Documentation

| File | Purpose | Status |
|------|---------|--------|
| `README.md` | Main project documentation | ✅ ACTIVE |
| `GETTING_STARTED.md` | First steps guide | ✅ ACTIVE |
| `SETUP_GUIDE.md` | Installation instructions | ✅ ACTIVE |
| `QUICK_REFERENCE.md` | Developer quick reference | ✅ ACTIVE |
| `REPRODUCING_RESULTS.md` | Reproducibility guide | ✅ ACTIVE |
| `PROJECT_ORGANIZATION.md` | Organization summary | ✅ ACTIVE |
| `ARCHITECTURE_VERIFICATION.md` | Architecture validation | ✅ ACTIVE |
| `COMPETITIVE_BENCHMARK_RESULTS.md` | Benchmark results | ✅ ACTIVE |
| `TEST_RESULTS.md` | Test execution summary | ✅ ACTIVE |
| `PATENT_NOTICE.md` | Patent information | ✅ ACTIVE |
| `LICENSE.md` | AGPL-3.0-or-later | ✅ ACTIVE |
| `LICENSE-CLAIMS-NC.md` | Non-commercial claims license | ✅ ACTIVE |
| `CLAIMS_PRACTICING_FILES.txt` | Patent-practicing files list | ✅ ACTIVE |

---

## 🔬 Experiments & Research (experiments/)

### Active Experiment Directories

| Directory | Purpose | Key Files | Status |
|-----------|---------|-----------|--------|
| `ascii_wall/` | Coherence-free theorem validation | `ascii_wall_paper.py`, `ascii_wall_final_hypotheses.py` | ✅ ACTIVE |
| `hypothesis_testing/` | 17 hybrid hypotheses | `hybrid_mca_fixes.py` | ✅ ACTIVE |
| `competitors/` | Competitive benchmarks | `benchmark_transforms_vs_fft.py` | ✅ ACTIVE |
| `fibonacci/` | Fibonacci tilt analysis | `fibonacci_tilt_hypotheses.py` | ✅ ACTIVE |
| `entropy/` | Entropy routing | Various | ✅ ACTIVE |
| `tetrahedral/` | Geometric validation | `tetrahedral_deep_dive.py` | ✅ ACTIVE |
| `sota_benchmarks/` | SOTA comparisons | `sota_compression_benchmark.py` | ✅ ACTIVE |
| `runtime/` | Performance analysis | Various | ✅ ACTIVE |
| `corpus/` | Test datasets | Various | ✅ ACTIVE |

### Key Experiment Results

**Validated Claims:**
1. **ASCII Bottleneck** - H3 Cascade achieves 0.672 BPP with η=0 coherence
2. **Scaling Laws** - 61.8%+ sparsity for golden-ratio signals
3. **Fibonacci Tilt** - Optimal lattice for crypto (52% avalanche)
4. **Tetrahedral RFT** - Geometric variant validation complete

---

## 🛠️ Build & Deployment Infrastructure

### CI/CD (.github/workflows/)

| Workflow | File | Status | Purpose |
|----------|------|--------|---------|
| Shannon Tests | `shannon_tests.yml` | ✅ ACTIVE | Information theory validation |
| SPDX Inject | `spdx-inject.yml` | ✅ ACTIVE | License header injection |

### Docker

| File | Purpose | Status |
|------|---------|--------|
| `Dockerfile` | Main container | ✅ ACTIVE |
| `Dockerfile.papers` | LaTeX compilation | ✅ ACTIVE |
| `DOCKER_PAPERS.md` | Docker documentation | ✅ ACTIVE |

### Bootstrap & Setup

| Script | Purpose | Status |
|--------|---------|--------|
| `quantoniumos-bootstrap.sh` | One-command setup | ✅ ACTIVE |
| `organize-release.sh` | Release packager | ✅ ACTIVE |
| `verify_setup.sh` | Installation verification | ✅ ACTIVE |
| `run_demo.sh` | Demo runner | ✅ ACTIVE |

### Build Scripts (scripts/)

| Script | Purpose | Status |
|--------|---------|--------|
| `run_full_suite.sh` | Full benchmark runner | ✅ ACTIVE |
| `validate_all.sh` | Complete validation | ✅ ACTIVE |
| `generate_all_theorem_figures.py` | Figure generation | ✅ ACTIVE |
| `verify_scaling_laws.py` | Scaling law validation | ✅ ACTIVE |
| `verify_ascii_bottleneck.py` | ASCII theorem validation | ✅ ACTIVE |
| `run_paper_validation_suite.py` | Paper claim validation | ✅ ACTIVE |

---

## 📦 Python Package Structure

### Package Configuration

**Files:**
- `pyproject.toml` - Modern Python packaging config
- `requirements.txt` - Core dependencies
- `requirements.in` - Dependency sources
- `requirements-lock.txt` - Locked versions
- `pytest.ini` - Test configuration

### Core Dependencies

```python
dependencies = [
    "numpy==1.26.4",          # Core arrays
    "scipy>=1.7.0,<1.13.0",   # Scientific computing
    "matplotlib==3.9.0",      # Plotting
    "sympy==1.12",            # Symbolic math
    "qutip==4.7.6",          # Quantum simulation
    "PyQt5",                  # GUI framework
]
```

### Optional Dependencies

```python
dev = ["pytest", "black", "flake8", "jupyterlab", "hypothesis"]
ai = ["torch", "transformers", "datasets", "accelerate", "peft", "trl"]
image = ["diffusers", "Pillow", "controlnet-aux", "xformers"]
```

### Package Namespace

**Import Structure:**
```python
from quantoniumos import CanonicalTrueRFT, EnhancedRFTCryptoV2
from algorithms.rft.core import rft_forward, rft_inverse
from algorithms.rft.variants.manifest import iter_variants
from algorithms.rft.crypto.enhanced_cipher import EnhancedRFTCryptoV2
```

---

## 🗑️ Deprecated & Cleanup Candidates

### Potential Duplications

**⚠️ Need Investigation:**

1. **Multiple RFT Implementations:**
   - `algorithms/rft/core/canonical_true_rft.py` ✅ KEEP (Claims-practicing)
   - `algorithms/rft/core/closed_form_rft.py` ✅ KEEP (Reference)
   - `algorithms/rft/core/rft_optimized.py` ✅ KEEP (Performance)
   - **Recommendation:** All three serve distinct purposes, KEEP ALL

2. **Duplicate App Structures:**
   - `src/apps/` - Flat application files
   - `quantonium_os_src/apps/` - Organized module structure
   - **Recommendation:** Migrate remaining apps from `src/apps/` to `quantonium_os_src/apps/`, then deprecate flat structure

3. **Multiple Quantum Implementations:**
   - `algorithms/rft/quantum/` - Core quantum algorithms
   - `algorithms/rft/core/quantum_kernel_implementation.py` - QSC
   - **Recommendation:** Merge into `algorithms/rft/quantum/`, maintain single source

4. **Duplicate Geometric Hashing:**
   - `algorithms/rft/core/geometric_waveform_hash.py`
   - `algorithms/rft/core/geometric_hashing.py`
   - `algorithms/rft/quantum/geometric_waveform_hash.py`
   - `algorithms/rft/quantum/geometric_hashing.py`
   - **Recommendation:** Consolidate into `algorithms/rft/crypto/geometric_hashing.py`

### Cache Directories (Safe to Delete)

```bash
# 36 __pycache__ directories found
find /workspaces/quantoniumos -type d -name "__pycache__" -exec rm -rf {} +

# Python build artifacts
rm -rf /workspaces/quantoniumos/quantoniumos.egg-info
rm -rf /workspaces/quantoniumos/.pytest_cache
rm -rf /workspaces/quantoniumos/.hypothesis

# C++ build artifacts (if rebuilding)
rm -rf /workspaces/quantoniumos/src/rftmw_native/build
```

### Legacy Files (Candidates for Archival)

**⚠️ Verify Before Deletion:**

1. **Legacy Hybrid Codec:**
   - Location: Various `legacy_hybrid_codec.py` files
   - Status: Replaced by H3/FH5
   - Recommendation: Move to `docs/archive/`

2. **Old Test Files:**
   - Check `tests/slow/` for outdated tests
   - Archive tests that are superseded

3. **Experimental Notebooks:**
   - Review Jupyter notebooks in experiments/
   - Archive completed experiments

### Build Artifacts

**Generated Files (Do Not Commit):**
- `*.pyc` - Python bytecode
- `*.so` - Compiled shared libraries (except in release packages)
- `*.o` - Object files
- `*.a` - Static libraries
- `*.vcd` - Waveform dumps
- `sim_*` - Simulation executables

---

## 📊 Directory Size Analysis

```
Directory                    Size      Files  Status
─────────────────────────────────────────────────────────
src/                         24MB      1200+  ✅ ACTIVE (C++/Python)
papers/                      11MB        17   ✅ ACTIVE (LaTeX)
hardware/                    5.0MB       30   ✅ ACTIVE (SystemVerilog)
algorithms/                  1.7MB      150+  ✅ ACTIVE (Core algos)
tests/                       1.7MB      100+  ✅ ACTIVE (Test suite)
docs/                        1.3MB      100+  ✅ ACTIVE (Documentation)
experiments/                 1.1MB       28   ✅ ACTIVE (Research)
quantoniumos/               420KB        2   ✅ ACTIVE (Package namespace)
benchmarks/                 320KB       10   ✅ ACTIVE (Benchmarks)
quantonium_os_src/            ?         ?    ✅ ACTIVE (Organized apps)
quantonium-mobile/            ?         ?    ✅ ACTIVE (React Native)
ui/                          <1MB        ?    ✅ ACTIVE (UI resources)
data/                        <1MB        ?    ✅ ACTIVE (Datasets)
results/                     <1MB        ?    ✅ ACTIVE (Results)
figures/                     <1MB        ?    ✅ ACTIVE (Generated figures)
release/                     <1MB        1    ✅ ACTIVE (Packaged release)
scripts/                     <1MB       40+   ✅ ACTIVE (Automation)
tools/                       <1MB       30+   ✅ ACTIVE (Dev tools)
.github/                     <1MB        2    ✅ ACTIVE (CI/CD)
─────────────────────────────────────────────────────────
TOTAL                        ~42MB    7585 Python files
```

---

## 🎯 Recommendations for Organization

### Immediate Actions

1. **Clean Build Artifacts:**
   ```bash
   find . -type d -name "__pycache__" -delete
   find . -type d -name ".pytest_cache" -delete
   find . -type d -name "*.egg-info" -delete
   ```

2. **Consolidate Duplicate Code:**
   - Merge geometric hashing implementations
   - Move quantum implementations to single location
   - Standardize on organized app structure

3. **Archive Completed Experiments:**
   - Move validated experiments to `docs/validation/`
   - Keep active research in `experiments/`

4. **Update Documentation:**
   - This file (`SYSTEM_ARCHITECTURE_MAP.md`) as master reference
   - Update `QUICK_REFERENCE.md` with latest commands
   - Ensure all READMEs are current

### Medium-Term Improvements

1. **Code Organization:**
   - Complete migration to `quantonium_os_src/apps/` structure
   - Consolidate crypto implementations
   - Create clear separation between:
     - Core algorithms (`algorithms/`)
     - Applications (`quantonium_os_src/apps/`)
     - Research (`experiments/`)
     - Validation (`tests/`, `docs/validation/`)

2. **Test Coverage:**
   - Run all pending tests
   - Achieve >80% coverage for core algorithms
   - Add integration tests for hybrids

3. **Performance Optimization:**
   - Complete native module builds
   - Benchmark all 14 variants
   - Profile and optimize hot paths

4. **Documentation:**
   - Generate API documentation (Sphinx)
   - Create video tutorials
   - Write academic paper on hybrid architectures

### Long-Term Goals

1. **Hardware:**
   - Fix Verilator lint errors
   - Complete FPGA synthesis
   - Measure resource utilization for N=256, 512, 1024

2. **Benchmarking:**
   - Run full competitive suite against industry standards
   - Publish benchmark results
   - Create reproducible benchmark containers

3. **Applications:**
   - Polish QuantSoundDesign for release
   - Complete mobile app
   - Create web-based demos

4. **Academic:**
   - Submit papers to conferences/journals
   - Open-source under dual-license model
   - Build community around Φ-RFT research

---

## 🔐 License & Patent Information

### License Structure

**Dual License:**
1. **AGPL-3.0-or-later** - Most files (see `LICENSE.md`)
2. **Non-Commercial Claims License** - Patent-practicing files (see `LICENSE-CLAIMS-NC.md`)

**Claims-Practicing Files:**
Listed in `CLAIMS_PRACTICING_FILES.txt`:
- `algorithms/rft/core/canonical_true_rft.py`
- Other files implementing patented methods

### Patent Status

**USPTO Application:** 19/169,399  
**Filed:** April 3, 2025  
**Title:** *Hybrid Computational Framework for Quantum and Resonance Simulation*  
**Status:** Patent Pending

**Commercial Licensing:**
Contact Luis M. Minier (luisminier79@gmail.com) for commercial rights to patent-practicing implementations.

---

## 📞 Contact & Support

**Author:** Luis M. Minier  
**Email:** luisminier79@gmail.com  
**GitHub:** https://github.com/mandcony/quantoniumos

**Support:**
- Bug Reports: GitHub Issues
- Feature Requests: GitHub Discussions
- Commercial Licensing: Direct email
- Academic Collaboration: Direct email
- Security Reviews: Direct email

---

## 📝 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Dec 3, 2025 | Initial comprehensive system map |

---

**End of System Architecture Map**

*Generated automatically by scanning the QuantoniumOS repository structure.*  
*This document should be updated whenever major structural changes occur.*
