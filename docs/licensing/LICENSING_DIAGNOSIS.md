# QuantoniumOS Licensing Diagnosis
**Analysis Date:** December 3, 2025  
**Patent Application:** U.S. 19/169,399 (Filed April 3, 2025)  
**Title:** "Hybrid Computational Framework for Quantum and Resonance Simulation"

---

## Executive Summary

Based on analysis of your patent claims, research documents, and codebase, this document provides a comprehensive diagnosis of which files should be:

1. **AGPL-3.0-or-later** (Open source, free for all uses including commercial)
2. **LicenseRef-QuantoniumOS-Claims-NC** (Research-only, commercial requires patent license)

### Key Principle
**Claims-NC applies to files that PRACTICE the patent claims**, not files that merely:
- Benchmark/test claimed algorithms
- Use standard algorithms (FFT, rANS, SHA-256)
- Provide infrastructure/utilities

---

## Patent Claims Summary (U.S. 19/169,399)

### Claim 1: Symbolic Resonance Fourier Transform Engine
**What's protected:**
- φ^(kn) golden-ratio phase modulation in unitary transform
- Symbolic amplitude decomposition (not amplitude-level simulation)
- Topological embedding layer with manifold mapping
- Symbolic gate propagation without collapse

### Claim 2: Resonance-Based Cryptographic Subsystem
**What's protected:**
- Amplitude-phase modulated waveform generation using φ
- Geometric coordinate transformations (polar-to-Cartesian with φ scaling)
- Topological hashing via manifold mapping
- Dynamic entropy mapping with recursive modulation

### Claim 3: Geometric Structures for Waveform Hashing
**What's protected:**
- Topological winding number computation (Wilson loops)
- Manifold-based hash generation preserving topological structure
- Complex geometric coordinate generation with φ

### Claim 4: Hybrid Mode Integration
**What's protected:**
- Unified framework combining Claims 1-3
- Coherent propagation across encryption layers
- Dynamic resource allocation between symbolic/crypto modes

---

## File Classification Matrix

### ✅ SHOULD BE Claims-NC (Practicing Patent Claims)

| Category | Files | Reason |
|----------|-------|--------|
| **Core Φ-RFT Engine** | `algorithms/rft/core/canonical_true_rft.py` | Claim 1: φ-modulated unitary transform |
| | `algorithms/rft/core/closed_form_rft.py` | Claim 1: Core φ^(kn) implementation |
| | `algorithms/rft/core/rft_optimized.py` | Claim 1: Optimized RFT kernel |
| **Quantum Symbolic** | `algorithms/rft/core/quantum_kernel_implementation.py` | Claim 1: Symbolic amplitude decomposition |
| | `algorithms/rft/core/topological_quantum_kernel.py` | Claim 3: Topological embedding |
| | `algorithms/rft/core/enhanced_topological_qubit.py` | Claim 3: Topological structure |
| | `algorithms/rft/core/quantum_gates.py` | Claim 1: Symbolic gate propagation |
| **Geometric Crypto** | `algorithms/rft/core/geometric_waveform_hash.py` | Claims 2+3: Waveform hashing |
| | `algorithms/rft/core/geometric_hashing.py` | Claim 3: Geometric coordinate transforms |
| **Unitary Variants** | `algorithms/rft/variants/golden_ratio_unitary.py` | Claim 1: φ-parameterized unitary |
| | `algorithms/rft/variants/symbolic_unitary.py` | Claim 1: Symbolic decomposition |
| | `algorithms/rft/variants/entropic_unitary.py` | Claim 2: Entropy-based modulation |
| **Compression (Hybrid)** | `algorithms/rft/compression/rft_quantum_sim.py` | Claim 4: QSC integration |
| | `algorithms/rft/compression/rft_vertex_codec.py` | Claims 1+3: Vertex-based RFT codec |
| **Hybrid Codecs** | `algorithms/rft/hybrids/rft_hybrid_codec.py` | Claim 4: Unified framework |
| | `algorithms/rft/hybrids/cascade_hybrids.py` | Claim 4: Cascade integration |
| | `algorithms/rft/hybrids/hybrid_residual_predictor.py` | Claim 4: Hybrid mode |
| **Crypto Primitives** | `algorithms/rft/crypto/enhanced_cipher.py` | Claim 2: RFT-based cipher |
| | `algorithms/rft/crypto/primitives/quantum_prng.py` | Claims 1+2: φ-based PRNG |
| | `algorithms/rft/crypto/benchmarks/rft_sis/rft_sis_hash_v31.py` | Claims 2+3: RFT-SIS |
| **Hardware RTL** | `hardware/fpga_top.sv` | All Claims: Hardware implementation |
| | `hardware/quantoniumos_unified_engines.sv` | Claim 4: Unified engine |
| | `hardware/rft_middleware_engine.sv` | Claims 1+4: RFT middleware |
| | `hardware/makerchip_rft_closed_form.tlv` | Claim 1: TL-Verilog RFT |
| **Applications** | `src/apps/quantsounddesign/*.py` | Claims 1+4: DAW using Φ-RFT |
| **Research Experiments** | `experiments/hypothesis_testing/hybrid_mca_fixes.py` | Claim 4: Hybrid implementations |
| | `experiments/ascii_wall/*.py` | All Claims: Practicing hybrids |

### ✅ SHOULD BE AGPL (Standard Algorithms / Infrastructure)

| Category | Files | Reason |
|----------|-------|--------|
| **Standard Compression** | `algorithms/rft/compression/ans.py` | rANS is public domain algorithm |
| | `algorithms/rft/compression/lossless/rans_stream.py` | rANS streaming is standard |
| **Benchmark Runners** | `benchmarks/*.py` | Test harnesses, don't practice claims |
| | `benchmarks/run_all_benchmarks.py` | Benchmark orchestration only |
| | `benchmarks/variant_benchmark_harness.py` | Testing infrastructure |
| **Competitor Benchmarks** | `experiments/competitors/*.py` | Comparing to standard algorithms |
| **Entropy Measurement** | `experiments/entropy/*.py` | Information theory (standard math) |
| **Runtime Benchmarks** | `experiments/runtime/*.py` | Performance measurement tools |
| **Documentation Demos** | `docs/archive/*.py` | Demo/proof scripts |
| **Test Infrastructure** | `tests/conftest.py` | pytest configuration |
| | `tests/transforms/*.py` | Testing utilities |
| **Build/Config** | `Dockerfile`, `Dockerfile.papers` | Container infrastructure |
| | `pyproject.toml`, `requirements*.txt` | Python packaging |
| **Visualization** | `hardware/visualize_*.py` | Plotting/visualization tools |
| | `hardware/generate_hardware_test_vectors.py` | Test vector generation |

---

## Current Issues Identified

### Issue 1: Native C++ Files Have Wrong License
**Files:** `src/rftmw_native/*.hpp`, `src/rftmw_native/*.cpp`

**Current:** AGPL-3.0-or-later (per SPDX headers)  
**Should be:** Claims-NC (listed in CLAIMS_PRACTICING_FILES.txt)

**Contradiction:** These files ARE in CLAIMS_PRACTICING_FILES.txt but have AGPL headers.

**Recommendation:** 
- If they implement Φ-RFT core → Change headers to Claims-NC
- If they're just FFT wrappers → Remove from CLAIMS_PRACTICING_FILES.txt

### Issue 2: rANS Files Have Conflicting Licenses
**Files:** 
- `algorithms/rft/compression/lossless/rans_stream.py`
- `algorithms/rft/compression/ans.py`

**Current:** 
- SPDX header says Claims-NC
- Docstring says "License: MIT"
- Listed in CLAIMS_PRACTICING_FILES.txt  

**Should be:** AGPL (rANS is Jarek Duda's public domain algorithm)

**The patent covers using RFT coefficients as input to entropy coding, not rANS itself.**

**Fix:** Remove from CLAIMS list, change header to AGPL, remove "License: MIT" from docstring.

### Issue 3: legacy_mca.py IS Claims-Practicing (Confirmed)
**File:** `algorithms/rft/hybrids/legacy_mca.py`

**Status:** Correctly labeled Claims-NC ✅

**Reason:** Uses canonical Φ-RFT with all 7 variants, implements hybrid DCT + Φ-RFT decomposition per Claims 1 and 4.

### Issue 4: Missing From CLAIMS List
**Files that should be added:**
```
experiments/hypothesis_testing/hypothesis_battery_h1_h12.py
experiments/sota_benchmarks/*.py (if practicing claims)
algorithms/rft/theorems/*.py (if any exist)
```

### Issue 5: Files With Claims-NC That Shouldn't Be
**Files that are pure infrastructure:**
```
# These should be AGPL:
tests/validation/test_rft_vertex_codec_roundtrip.py  # Test harness
tests/test_audio_backend.py                          # Test harness
```

---

## Recommended Actions

### Phase 1: Fix Contradictions (High Priority)

1. **Fix src/rftmw_native/ headers:**
   ```cpp
   // OLD:
   // SPDX-License-Identifier: AGPL-3.0-or-later
   
   // NEW (if practicing claims):
   // SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
   ```

2. **Remove rANS from CLAIMS list:**
   ```
   # Remove from CLAIMS_PRACTICING_FILES.txt:
   algorithms/rft/compression/lossless/rans_stream.py
   algorithms/rft/compression/ans.py
   ```

### Phase 2: Add Missing Files (Medium Priority)

3. **Add to CLAIMS_PRACTICING_FILES.txt:**
   ```
   # Hybrid experiments that practice claims
   experiments/hypothesis_testing/hypothesis_battery_h1_h12.py
   experiments/hypothesis_testing/verify_h11_claims.py
   
   # ASCII wall experiments using full hybrid stack
   experiments/ascii_wall/ascii_wall_through_codec.py
   experiments/ascii_wall/ascii_wall_final_hypotheses.py
   experiments/ascii_wall/ascii_wall_vertex_codec.py
   experiments/ascii_wall/ascii_wall_h11_h20.py
   experiments/ascii_wall/ascii_wall_paper.py
   experiments/ascii_wall/ascii_wall_all_variants.py
   
   # SOTA benchmarks if they run claims-practicing code
   experiments/sota_benchmarks/run_on_paper_test.py
   ```

### Phase 3: Update Test Files (Low Priority)

4. **Change test harnesses to AGPL:**
   Test files that only import and test (don't implement) claims-practicing code should be AGPL so users can freely write and share tests.

---

## Decision Tree for New Files

```
Does the file IMPLEMENT (not just call) one of these?
│
├─ φ^(kn) phase modulation in transforms? → Claims-NC (Claim 1)
├─ Symbolic amplitude decomposition? → Claims-NC (Claim 1)
├─ Geometric waveform hashing with φ? → Claims-NC (Claims 2+3)
├─ Topological manifold mapping? → Claims-NC (Claim 3)
├─ Unified RFT+Crypto hybrid orchestration? → Claims-NC (Claim 4)
│
└─ No to all above:
   │
   ├─ Uses standard algorithms (FFT, rANS, SHA-256)? → AGPL
   ├─ Is test/benchmark infrastructure? → AGPL
   ├─ Is documentation/demo? → AGPL
   └─ Is build/config tooling? → AGPL
```

---

## Wave Computer Theorem Status

The **RFT Wave Computer Theorem** (docs/research/RFT_WAVE_COMPUTER_THEOREM.md) describes:

- Diagonalization of golden-ratio dynamics systems
- O(N) complexity for systems with φ-eigenspectrum

**This theorem is NOT currently in the patent claims** but could be a continuation/CIP.

**Recommendation:** 
- Keep research documents under AGPL (public disclosure)
- If you file a continuation, add implementing files to Claims-NC

---

## Summary

| License | File Count | Purpose |
|---------|------------|---------|
| **Claims-NC** | ~95 files | Core Φ-RFT, QSC, crypto, hardware, apps |
| **AGPL** | ~200+ files | Infrastructure, tests, standard algorithms |

**Key Insight:** The patent protects the φ-modulated unitary transform and its integration with geometric hashing - NOT standard algorithms like rANS, FFT, or SHA-256 that you use as building blocks.
