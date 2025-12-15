# Non-Claims and Limitations

> **Purpose:** Explicitly state what Φ-RFT does NOT do to prevent overclaiming by omission.

---

## What Φ-RFT Does NOT Claim

### 1. Not a Replacement for FFT

Φ-RFT does not replace the FFT for general-purpose spectral analysis. The FFT remains:
- Faster (O(N log N) vs O(N²) for naive RFT)
- Universally applicable
- The correct choice for most signal processing tasks

**Use Φ-RFT only when:** Your signal class matches the golden-ratio autocorrelation model.

### 2. Not Faster Asymptotically

| Transform | Complexity | Notes |
|-----------|------------|-------|
| FFT | O(N log N) | Optimal for uniformly-sampled sinusoids |
| Φ-RFT (naive) | O(N²) | Eigendecomposition cost |
| Φ-RFT (fast) | O(N log N) | Exploits Toeplitz structure (experimental) |

Even with fast algorithms, Φ-RFT carries higher constant factors.

### 3. Not Quantum Computing

Despite the "Quantum" in project name, this work is:
- **Classical computation only**
- No qubits, no quantum gates, no quantum speedup
- "Quantum-inspired" refers to mathematical structure, not physics

The project name is historical; the mathematics is purely classical.

### 4. Not Universally Optimal

Φ-RFT achieves good results **only on specific signal classes:**

| Signal Type | Φ-RFT Performance | Better Alternative |
|-------------|-------------------|-------------------|
| Golden-ratio quasi-periodic | ✅ See [VERIFIED_BENCHMARKS](research/benchmarks/VERIFIED_BENCHMARKS.md) | — |
| Smooth piecewise signals | ⚠️ Comparable | DCT |
| White noise | ❌ No advantage | Any transform |
| High-entropy random | ❌ No advantage | Entropy coding |
| Natural images | ⚠️ Domain-dependent | DCT/Wavelet |
| Audio/speech | ⚠️ Domain-dependent | MDCT |

### 5. Not a Cryptographic Primitive

The RFT-SIS hashing experiments are:
- Research explorations only
- Not proven secure
- Not recommended for production cryptography
- Not a replacement for SHA-256, BLAKE3, etc.

### 6. Not Production Software

This codebase is:
- A research framework for experiments
- Not hardened for production deployment
- Not audited for security vulnerabilities
- Not optimized for all edge cases

### 7. Not Novel in Every Aspect

Parts of this work build on well-established foundations:
- Eigendecomposition (linear algebra)
- Toeplitz matrix structure (1900s)
- Golden ratio in signal processing (existing literature)
- Transform coding (JPEG, etc.)

**The claimed novelty is narrow:** A specific combination producing a data-independent transform with KLT-like properties on certain signal classes.

---

## Known Limitations

### Computational

1. **O(N²) complexity** for naive implementation
2. **Memory overhead** for storing eigenbasis
3. **Precomputation required** for eigenbasis (one-time cost)
4. **Numerical precision** degrades for very large N

### Signal Classes

1. **No advantage on white noise** — expected, not a bug
2. **No advantage on fully random signals** — information-theoretic limit
3. **Reduced advantage on non-golden-ratio structures**
4. **Domain mismatch** causes performance regression

### Empirical

1. **Benchmarks are domain-specific** — not universal claims
2. **Hardware results are simulated** — no physical ASIC validation
3. **Medical data results** require clinical validation
4. **Compression ratios** depend heavily on signal statistics

---

## Honest Failure Cases

We explicitly document where Φ-RFT loses:

| Benchmark | Φ-RFT Result | Winner | Reason |
|-----------|--------------|--------|--------|
| White noise compression | 0% improvement | Tie | No structure to exploit |
| Random permutation | No sparsity gain | FFT | Basis mismatch |
| High-entropy text | 0.99 BPP | gzip | Entropy-limited |
| Out-of-family signals | Typically loses | FFT/DCT | Domain mismatch |

> **Note**: For current reproducible metrics, see [VERIFIED_BENCHMARKS](research/benchmarks/VERIFIED_BENCHMARKS.md).

**This is expected behavior, not a failure of the method.**

---

## Reviewer Concerns Pre-Answered

### "Isn't this just a windowed FFT?"

No. The phase structure is non-quadratic and derived from golden-ratio autocorrelation eigendecomposition, not windowing. See [GLOSSARY.md](docs/GLOSSARY.md) for precise definition.

### "Why is it slower?"

Because eigendecomposition is O(N²) naive. Fast algorithms exist but carry higher constant factors than FFT. We trade speed for sparsity on specific signal classes.

### "Isn't this cherry-picked?"

We explicitly show failure cases (see above). The claims are narrow: specific signal classes, specific metrics, specific conditions.

### "What's the point if FFT is faster?"

Sparsity matters for:
- Compression (fewer coefficients = smaller files)
- Denoising (thresholding in sparse domain)
- Feature extraction (compact representation)

Speed matters less than representation quality for these applications.

### "How do I know this isn't vaporware?"

- All code is open source
- Benchmarks are reproducible (see BENCHMARK_PROTOCOL.md)
- Results include failure cases
- No claims without validation data

---

## Summary

| Claim | Status |
|-------|--------|
| Replaces FFT | ❌ FALSE |
| Faster than FFT | ❌ FALSE |
| Quantum computing | ❌ FALSE |
| Universally optimal | ❌ FALSE |
| Cryptographically secure | ❌ FALSE |
| Production-ready | ❌ FALSE |
| Novel operator-eigenbasis transform | ✅ TRUE |
| KLT-like compaction on specific signals | ✅ TRUE |
| Data-independent basis | ✅ TRUE |
| Outside LCT/FrFT family | ✅ TRUE |

---

*Last updated: December 2025*
