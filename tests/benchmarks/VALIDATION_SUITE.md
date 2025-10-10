# QuantoniumOS Benchmark Suite

Comprehensive performance and validation benchmarks for QuantoniumOS algorithms and systems.

## Critical Validation Benchmarks

### 🚨 PHASE 1: BLOCKING REQUIREMENTS

These benchmarks are **CRITICAL** for QuantoniumOS validation and must pass before any publication or claims.

#### 1.1 RFT vs FFT Competitive Benchmark ⚠️ BLOCKING
**File**: `benchmark_rft_vs_fft.py`  
**Status**: Ready to run  
**Timeline**: 1-2 weeks  

**Purpose**: Prove RFT performance advantages over standard FFT
- Speed comparison across multiple sizes
- Accuracy measurement via round-trip error
- Compression efficiency on real model weights
- Memory usage analysis

**Usage**:
```bash
python benchmark_rft_vs_fft.py
```

**Deliverables**:
- `benchmark_results.json` - Raw performance data
- `benchmark_results.md` - Formatted comparison report

#### 1.2 NIST Randomness Validation ⚠️ BLOCKING  
**File**: `nist_randomness_tests.py`  
**Status**: Ready to run  
**Timeline**: 2-4 weeks  

**Purpose**: Validate PRNG cryptographic quality
- Frequency (monobit) test
- Block frequency test  
- Runs test
- Longest run test
- Bias correction implementation

**Usage**:
```bash
python nist_randomness_tests.py
```

**Success Criteria**: All tests must pass with p-value > 0.01

## Running Critical Benchmarks

### Phase 1 Validation (BLOCKING)
```bash
# Run critical validation benchmarks
cd /workspaces/quantoniumos/tests/benchmarks

# 1. RFT vs FFT Performance
python benchmark_rft_vs_fft.py

# 2. NIST Randomness Tests  
python nist_randomness_tests.py
```

### Success Criteria
- ✅ RFT shows competitive performance vs FFT
- ✅ PRNG passes all NIST randomness tests
- ✅ Professional cryptanalysis confirms security

---

**CRITICAL**: These benchmarks must pass before any publication claims.