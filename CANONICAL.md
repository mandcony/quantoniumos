# Canonical Implementation Reference

> **Purpose:** Define which code is claim-bearing and which is exploratory.

---

## Claim Status

**Only code marked CANONICAL is claim-bearing.**

Everything else is exploratory, demonstrative, or historical.

---

## CANONICAL (Claim-Bearing)

These files contain the authoritative implementation of Φ-RFT.

### Core Transform

| File | Status | Description |
|------|--------|-------------|
| `algorithms/rft/core/canonical_true_rft.py` | **CANONICAL** | Reference RFT implementation |
| `algorithms/rft/kernels/resonant_fourier_transform.py` | **CANONICAL** | Production kernel |
| `algorithms/rft/theory/phi_rft_formal_spec.py` | **CANONICAL** | Mathematical specification |

### Validation

| File | Status | Description |
|------|--------|-------------|
| `tests/rft/test_canonical_rft.py` | **CANONICAL** | Core correctness tests |
| `tests/rft/test_variant_unitarity.py` | **CANONICAL** | Unitarity verification |
| `tests/validation/test_rft_validation.py` | **CANONICAL** | Cross-validation suite |

### Benchmarks

| File | Status | Description |
|------|--------|-------------|
| `benchmarks/rft_realworld_benchmark.py` | **CANONICAL** | Primary benchmark suite |
| `benchmarks/class_a_quantum_simulation.py` | **CANONICAL** | Class A validation |
| `benchmarks/class_b_transform_dsp.py` | **CANONICAL** | Class B validation |

### Documentation

| File | Status | Description |
|------|--------|-------------|
| `algorithms/rft/README_RFT.md` | **CANONICAL** | Authoritative RFT definition |
| `docs/GLOSSARY.md` | **CANONICAL** | Term definitions |
| `docs/NON_CLAIMS.md` | **CANONICAL** | Explicit non-claims |
| `BENCHMARK_PROTOCOL.md` | **CANONICAL** | Benchmark methodology |

---

## EXPLORATORY (Non-Claim-Bearing)

These files are research explorations. Results are not peer-reviewed claims.

### Experimental Variants

| Path | Status | Description |
|------|--------|-------------|
| `algorithms/rft/experiments/` | EXPLORATORY | Variant research |
| `algorithms/rft/fast/` | EXPLORATORY | Fast algorithm R&D |
| `experiments/` | EXPLORATORY | General experiments |
| `rft-experiments/` | EXPLORATORY | Variant index |

### Applications

| Path | Status | Description |
|------|--------|-------------|
| `algorithms/rft/applications/` | EXPLORATORY | Domain-specific research |
| `examples/` | DEMONSTRATION | Usage examples only |

### Crypto

| Path | Status | Description |
|------|--------|-------------|
| `algorithms/rft/crypto/` | EXPLORATORY | **NOT cryptographically validated** |

---

## DEMONSTRATION (Not Scientific Claims)

These exist for visualization and UI purposes only.

| Path | Status | Description |
|------|--------|-------------|
| `quantonium_os_src/` | DEMONSTRATION | Desktop UI demo |
| `quantonium-mobile/` | DEMONSTRATION | Mobile prototype |
| `ui/` | DEMONSTRATION | UI components |
| `demos/` | DEMONSTRATION | Demo scripts |

---

## HARDWARE (Feasibility Study)

| Path | Status | Description |
|------|--------|-------------|
| `hardware/` | FEASIBILITY STUDY | Not fabricated, not validated |
| `hardware/rftpu-3d-viewer/` | DEMONSTRATION | Visualization only |

---

## DEPRECATED (Historical Only)

| File | Status | Reason |
|------|--------|--------|
| `algorithms/rft/kernels/phi_phase_fft.py` | DEPRECATED | No sparsity advantage |
| `algorithms/rft/kernels/closed_form_rft.py` | DEPRECATED | Renamed to phi_phase_fft |

---

## Reference Implementation Definition

The **canonical RFT** is defined by:

```python
# Authoritative definition in canonical_true_rft.py

def build_resonance_operator(N, f0, decay_rate, phi=PHI):
    """Build the Toeplitz autocorrelation matrix K."""
    r = np.zeros(N)
    for k in range(N):
        t_k = k / N
        r[k] = (np.cos(2*np.pi*f0*t_k) + np.cos(2*np.pi*f0*phi*t_k)) * np.exp(-decay_rate*k)
    r[0] = 1.0
    return toeplitz(r)

def build_rft_basis(N, f0=10.0, decay_rate=0.05):
    """Compute eigenbasis of resonance operator."""
    K = build_resonance_operator(N, f0, decay_rate)
    eigenvalues, eigenvectors = np.linalg.eigh(K)
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvectors[:, idx]

def rft_transform(x, basis):
    """Apply RFT: project onto eigenbasis."""
    return basis.T @ x

def rft_inverse(coeffs, basis):
    """Inverse RFT: reconstruct from coefficients."""
    return basis @ coeffs
```

**Parameters:**
- `N`: Signal length (power of 2 recommended)
- `f0`: Base frequency (default: 10.0)
- `decay_rate`: Autocorrelation decay (default: 0.05)
- `phi`: Golden ratio (fixed: 1.618...)

**This is the only authoritative definition.**

---

## Version

| Component | Version | Date |
|-----------|---------|------|
| Canonical RFT | 2.0 | December 2025 |
| Benchmark Protocol | 1.0 | December 2025 |
| API | Stable | December 2025 |

---

## Modification Policy

Changes to CANONICAL files require:
1. Pull request with justification
2. Passing all validation tests
3. Updated documentation
4. Version bump

Exploratory code may be modified freely.

---

*Last updated: December 2025*
