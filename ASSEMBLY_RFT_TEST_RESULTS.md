# Assembly/C RFT vs Classical Transforms: Test Results

**Date:** 2025-11-24  
**Test Suite:** `test_assembly_rft_vs_classical_transforms.py`  
**Assembly Library:** `libquantum_symbolic.so` (compiled with NASM + GCC)

## Executive Summary

✅ **Correctness:** Assembly RFT passes all mathematical correctness tests  
⚠️  **Performance:** Assembly RFT is **300-700x slower** than FFT (needs optimization)  
❌ **Sparsity:** Assembly RFT shows **0% sparsity** for quasi-periodic signals (unexpected)

---

## Test Results

### TEST 1: Unitarity / Round-Trip Reconstruction

All transforms successfully reconstruct input signals:

| Size | RFT (C/ASM) Error | FFT Error | DCT Error | Status |
|:-----|:------------------|:----------|:----------|:-------|
| 64   | 9.49e-15          | 2.15e-16  | 3.51e-16  | ✓ PASS |
| 128  | 1.92e-14          | 3.42e-16  | 2.91e-16  | ✓ PASS |
| 256  | 3.58e-14          | 2.60e-16  | 3.33e-16  | ✓ PASS |
| 512  | 7.58e-14          | 3.47e-16  | 3.34e-16  | ✓ PASS |

**Conclusion:** RFT maintains unitarity within tolerance (< 1e-10), though 2-3 orders of magnitude less precise than FFT.

---

### TEST 2: Sparsity Comparison

#### Quasi-Periodic Signals (φ-based harmonics)

**N=128:**
| Transform | Sparsity % | Gini Coef | 90% Energy (# coeffs) |
|:----------|:-----------|:----------|:----------------------|
| RFT (C/ASM) | 0.0%  | 0.231     | 83                    |
| FFT (FFTW)  | 37.5% | 0.661     | 3                     |
| DCT (FFTPACK) | 96.9% | 0.981   | 2                     |

**N=256:**
| Transform | Sparsity % | Gini Coef | 90% Energy (# coeffs) |
|:----------|:-----------|:----------|:----------------------|
| RFT (C/ASM) | 0.0%  | 0.236     | 162                   |
| FFT (FFTW)  | 72.7% | 0.692     | 3                     |
| DCT (FFTPACK) | 98.4% | 0.990   | 2                     |

**⚠️  CRITICAL ISSUE:** RFT shows 0% sparsity for quasi-periodic signals where it should excel!

#### Periodic Signals (integer harmonics)

**N=128:**
| Transform | Sparsity % | Gini Coef | 90% Energy (# coeffs) |
|:----------|:-----------|:----------|:----------------------|
| RFT (C/ASM) | 0.0%  | 0.382     | 58                    |
| FFT (FFTW)  | 97.7% | 0.977     | 3                     |
| DCT (FFTPACK) | 93.8% | 0.971   | 3                     |

**✓ Expected:** FFT dominates for periodic signals.

#### Random Signals

**N=128:**
| Transform | Sparsity % | Gini Coef | 90% Energy (# coeffs) |
|:----------|:-----------|:----------|:----------------------|
| RFT (C/ASM) | 0.0%  | 0.298     | 76                    |
| FFT (FFTW)  | 0.0%  | 0.279     | 78                    |
| DCT (FFTPACK) | 1.6% | 0.399    | 60                    |

**✓ Expected:** No transform is sparse for random signals.

---

### TEST 3: Performance Comparison

Execution time (milliseconds, averaged over 100 iterations):

| Size | Transform | Forward (ms) | Inverse (ms) | Total (ms) | Speedup vs RFT |
|:-----|:----------|:-------------|:-------------|:-----------|:---------------|
| 64   | RFT (C/ASM) | 0.177     | 0.177        | 0.353      | 1.0x           |
|      | FFT (FFTW)  | 0.006     | 0.006        | 0.012      | **29.4x**      |
|      | DCT (FFTPACK) | 0.008   | 0.008        | 0.016      | **22.1x**      |
| 128  | RFT (C/ASM) | 0.566     | 0.565        | 1.131      | 1.0x           |
|      | FFT (FFTW)  | 0.007     | 0.007        | 0.013      | **87.0x**      |
|      | DCT (FFTPACK) | 0.009   | 0.008        | 0.017      | **66.5x**      |
| 256  | RFT (C/ASM) | 1.980     | 2.003        | 3.983      | 1.0x           |
|      | FFT (FFTW)  | 0.007     | 0.007        | 0.014      | **284.5x**     |
|      | DCT (FFTPACK) | 0.012   | 0.014        | 0.027      | **147.5x**     |
| 512  | RFT (C/ASM) | 7.381     | 7.353        | 14.735     | 1.0x           |
|      | FFT (FFTW)  | 0.009     | 0.009        | 0.018      | **818.6x**     |
|      | DCT (FFTPACK) | 0.011   | 0.010        | 0.020      | **736.8x**     |

**⚠️  CRITICAL ISSUE:** Assembly RFT is 30-800x slower than FFT, despite claimed O(N log N) complexity!

**Complexity Analysis:**
- Expected: O(N log N) for all transforms
- Observed: RFT appears to scale worse than O(N log N)
- N=64→128: 3.2x slower (expected 2.2x)
- N=128→256: 3.5x slower (expected 2.0x)
- N=256→512: 3.7x slower (expected 2.0x)

---

### TEST 4: Energy Preservation (Parseval's Theorem)

| Size | Transform | Energy Error | Status |
|:-----|:----------|:-------------|:-------|
| 64   | RFT (C/ASM) | 3.80e-16   | ✓ PASS |
|      | FFT (FFTW)  | 0.00e+00   | ✓ PASS |
|      | DCT (FFTPACK) | 5.18e-01 | ✗ FAIL |
| 128  | RFT (C/ASM) | 2.15e-16   | ✓ PASS |
|      | FFT (FFTW)  | 2.15e-16   | ✓ PASS |
|      | DCT (FFTPACK) | 5.18e-01 | ✗ FAIL |
| 256  | RFT (C/ASM) | 6.37e-16   | ✓ PASS |
|      | FFT (FFTW)  | 0.00e+00   | ✓ PASS |
|      | DCT (FFTPACK) | 5.29e-01 | ✗ FAIL |
| 512  | RFT (C/ASM) | 4.09e-15   | ✓ PASS |
|      | FFT (FFTW)  | 4.30e-16   | ✓ PASS |
|      | DCT (FFTPACK) | 4.73e-01 | ✗ FAIL |

**Note:** DCT fails because our test uses complex signals but DCT only processes real part.

---

## Critical Issues Identified

### 1. **Zero Sparsity for Quasi-Periodic Signals**

**Problem:** The assembly RFT implementation shows 0% sparsity for quasi-periodic (φ-based) signals, where it should theoretically excel (Theorem 3 claims >61.8% sparsity).

**Possible Causes:**
- Assembly implementation may not be using the correct phase progression (φ^(-k))
- Basis vectors may be incorrectly computed
- Transform may be implementing standard FFT instead of φ-RFT

**Evidence:** Python RFT implementation shows 89-98% sparsity for quasi-periodic signals (as validated in `scripts/irrevocable_truths.py`), but assembly shows 0%.

### 2. **Severe Performance Degradation**

**Problem:** Assembly RFT is 30-800x slower than optimized FFT, contradicting the O(N log N) complexity claim.

**Possible Causes:**
- Missing SIMD vectorization (AVX2/AVX512)
- No FFT factorization (may be using O(N²) matrix multiply)
- Memory access patterns not cache-optimized
- Excessive function call overhead
- No pre-computation of phase tables

**Evidence:** Scaling behavior suggests O(N²) or worse, not O(N log N).

### 3. **Precision Issues**

**Problem:** Assembly RFT has 2-3 orders of magnitude worse numerical precision than FFT.

**Impact:** Acceptable for most applications (< 1e-10) but indicates potential accumulated rounding errors.

---

## Recommendations

### Immediate Actions

1. **Verify Assembly Implementation:**
   ```bash
   # Check if assembly is computing correct basis
   python -c "from algorithms.rft.kernels.python_bindings.unitary_rft import UnitaryRFT; \
              import numpy as np; \
              rft = UnitaryRFT(16, 0); \
              x = np.ones(16, dtype=np.complex128); \
              print(rft.forward(x))"
   ```

2. **Compare Against Python Reference:**
   - Test same input on both implementations
   - Verify spectral outputs match
   - Check if assembly is actually using φ-based phases

3. **Profile Assembly Code:**
   ```bash
   cd algorithms/rft/kernels
   gcc -pg -O3 kernel/rft_kernel_fixed.c -o profile_test
   gprof profile_test
   ```

### Performance Optimization Path

1. **Implement FFT Factorization:**
   - Current: Direct matrix-vector multiply O(N²)
   - Target: Ψ = D_φ C_σ F factorization O(N log N)

2. **Add SIMD Instructions:**
   - Use AVX2 for 4x complex128 parallelism
   - Use AVX512 for 8x complex128 parallelism

3. **Pre-compute Phase Tables:**
   - Store D_φ and C_σ diagonals
   - Align to cache line boundaries (64 bytes)

4. **Use FFTW for FFT Component:**
   - Link against FFTW3 library
   - Apply phase corrections as post-processing

### Correctness Validation

1. **Add Sparsity Tests to CI:**
   ```python
   def test_assembly_rft_sparsity():
       from algorithms.rft.kernels.python_bindings.unitary_rft import UnitaryRFT
       x = generate_quasi_periodic(128)
       rft = UnitaryRFT(128, 0)
       X = rft.forward(x)
       sparsity = compute_sparsity(X)
       assert sparsity > 60.0, f"Sparsity {sparsity}% below theoretical minimum 61.8%"
   ```

2. **Cross-Validate with Python:**
   - Run same test vectors through both implementations
   - Assert spectral similarity (not just round-trip error)

---

## Conclusion

The assembly RFT implementation **passes unitarity and energy preservation tests** but has **critical performance and sparsity issues** that must be addressed before it can be considered production-ready.

**Status Summary:**
- ✅ Mathematical correctness (unitarity, energy)
- ❌ Performance (800x slower than FFT)
- ❌ Sparsity (0% instead of >60%)
- ⚠️  Numerical precision (acceptable but suboptimal)

**Next Steps:**
1. Diagnose why assembly shows 0% sparsity
2. Implement FFT factorization for O(N log N) performance
3. Add comprehensive regression tests
4. Profile and optimize hot paths

## Variant Implementation Status (New)

**Date:** 2025-11-24
**Backend:** C/Assembly (`libquantum_symbolic.so`)

All 7 RFT variants have been translated to the C backend.

| Variant | ID | Status | Unitarity Error | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Original** | 0 | ⚠️ Partial | `1.82e-06` | Working, but sparsity is lower than Python due to rank deficiency handling. |
| **Harmonic** | 1 | ⚠️ Partial | `1.07e-05` | Working, acceptable unitarity. |
| **Fibonacci** | 2 | ❌ Fail | `1.42e+00` | Formula is highly rank-deficient; requires robust QR (SVD) to fix. |
| **Chaotic** | 3 | ✅ **PASS** | `1.41e-08` | Excellent unitarity. |
| **Geometric** | 4 | ✅ **PASS** | `1.00e-14` | **Perfect unitarity.** |
| **Hybrid** | 5 | ✅ **PASS** | `2.69e-10` | Excellent unitarity. |
| **Adaptive** | 6 | ✅ **PASS** | `2.69e-10` | Excellent unitarity. |

**Key Findings:**
- **Geometric, Chaotic, Hybrid, and Adaptive** variants are fully functional and mathematically correct in the Assembly backend.
- **Original and Fibonacci** variants suffer from numerical rank deficiency in the C-based Gram-Schmidt process. The Python implementation uses LAPACK's robust SVD/QR which handles these cases better.
- **Recommendation:** Use **Hybrid** or **Geometric** variants for high-performance testing in the Assembly backend.

---

**Test Files:**
- Main test: `tests/validation/test_assembly_rft_vs_classical_transforms.py`
- Results log: `assembly_vs_classical_results.txt`
- Comparison test: `tests/validation/test_assembly_vs_python_comprehensive.py`

**Related Documentation:**
- Developer Manual: `docs/algorithms/rft/RFT_DEVELOPER_MANUAL.md` Section 5 (Performance)
- Theorem Proofs: `docs/validation/RFT_THEOREMS.md`
- Assembly Code: `algorithms/rft/kernels/kernel/rft_kernel_fixed.c`
