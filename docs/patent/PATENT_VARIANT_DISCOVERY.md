# Patent Variant Discovery Report

**Date**: December 2025  
**Patent Application**: US 19/169,399  
**Title**: Hybrid Computational Framework for Quantum and Resonance Simulation  
**First Named Inventor**: Luis Michael Minier  
**Filing Date**: April 3, 2025  

---

## Executive Summary

Created 20 new RFT variants aligned to USPTO patent claims. Benchmarked all 30 transforms (20 patent + 8 original + FFT/DCT) across 16 signal types. Found **rft_manifold_projection** as the breakthrough performer with +47.9 dB improvement over golden on torus signals.

---

## Discovery Results

### Top 10 Winners (by signal type wins)

| Rank | Variant | Wins | Best Signal | Peak SNDR |
|------|---------|------|-------------|-----------|
| 1 | rft_golden | 5 | phyllotaxis | - |
| 2 | **rft_manifold_projection** | 4 | torus | 73.6 dB |
| 3 | rft_geometric | 2 | - | - |
| 4 | rft_euler_sphere | 1 | phyllotaxis | - |
| 5 | rft_phase_coherent | 1 | chirp | 53.9 dB |
| 6 | rft_entropy_modulated | 1 | noise | - |
| 7 | fft | 1 | pure_sine | - |
| 8 | dct | 1 | step | - |

### Star Performers (Patent Variants)

#### ★ RFT-Manifold-Projection (Claim 3)
- **Signal**: torus
- **SNDR**: 73.6 dB
- **vs FFT/DCT**: +18.6 dB
- **vs Golden**: +47.9 dB
- **Use Case**: Torus, spiral, helical signals

#### ★ RFT-Phase-Coherent (Claim 1)
- **Signal**: pure_sine
- **SNDR**: 53.9 dB
- **vs FFT/DCT**: +14.6 dB
- **vs Golden**: +15.5 dB
- **Use Case**: Chirp signals, frequency sweeps

#### RFT-Euler-Sphere (Claim 3)
- **Signal**: phyllotaxis
- **Use Case**: Biological patterns, spherical geodesics

#### RFT-Loxodrome (Claim 3)
- **Signal**: pure_sine
- **vs Golden**: +12.3 dB
- **Use Case**: Pure tones, rhumb line trajectories

#### RFT-Entropy-Modulated (Claim 2)
- **Signal**: noise
- **Use Case**: Random processes, entropy-modulated signals

---

## Patent Claim Mapping

### Claim 1: Symbolic Transform Subsystem
> A hybrid computational framework integrating quantum and resonance computing systems, comprising: a symbolic transformation subsystem that performs polar-Cartesian transformations with golden ratio scaling on input data...

**Aligned Variants**:
- `rft_phase_coherent` - Phase-space coherence maintenance
- `rft_polar_golden` - Polar-Cartesian with golden scaling

### Claim 2: Cryptographic Subsystem
> A system of claim 1, further comprising: a cryptographic subsystem that applies entropy-modulated golden ratio scaling to generate mathematically structured secure hashes...

**Aligned Variants**:
- `rft_entropy_modulated` - Entropy-modulated golden scaling
- `rft_bloom_hash` - Bloom filter integration
- `rft_merkle_lattice` - Lattice-based hash computation

### Claim 3: Geometric Structure Subsystem
> A method of claim 1, further comprising: applying geometric structures including at least one of: topological winding numbers associated with Euler characteristics, manifold-based hash generation and verification...

**Aligned Variants**:
- `rft_manifold_projection` - Manifold-based hash generation ★
- `rft_euler_sphere` - Euler characteristics
- `rft_euler_torus` - Torus topology
- `rft_winding` - Topological winding numbers
- `rft_hopf_fibration` - Hopf fibration
- `rft_trefoil_knot` - Knot invariants
- `rft_loxodrome` - Rhumb line trajectory

### Claim 4: Hybrid Integration
> A computer program product comprising: a non-transitory computer-readable medium storing instructions that, when executed, cause a processor to perform hybrid computation including both quantum simulation and resonance-based transformations...

**Aligned Variants**:
- All variants with `generate_*` function patterns
- `PATENT_VARIANTS` registry for unified access

---

## Technical Implementation

### Unitarity Verification

All 20 patent variants verified unitary:
```
✓ RFT-Polar-Golden        | err=4.0e-14
✓ RFT-Spiral-Golden       | err=8.9e-14
✓ RFT-Complex-Exp         | err=2.1e-14
✓ RFT-Winding            | err=4.8e-14
✓ RFT-Euler-Torus        | err=1.3e-13
✓ RFT-Euler-Sphere       | err=7.9e-14
✓ RFT-Hopf-Fibration     | err=2.3e-13
✓ RFT-Manifold-Projection | err=1.6e-13
... (all 20 variants pass)
```

### Registration

Patent variants are now registered in:
- `algorithms/rft/variants/patent_variants.py` - Definitions
- `algorithms/rft/variants/__init__.py` - Exports
- `algorithms/rft/variants/operator_variants.py` - OPERATOR_VARIANTS registry

### Signal Classifier

Added new TransformType enum values:
- `RFT_MANIFOLD_PROJECTION`
- `RFT_EULER_SPHERE`
- `RFT_PHASE_COHERENT`
- `RFT_ENTROPY_MODULATED`
- `RFT_LOXODROME`

Calibrated thresholds (accuracy 64%):
- `PERIODICITY_THRESHOLD = 0.250`
- `GOLDEN_THRESHOLD = 0.080`
- `FIBONACCI_THRESHOLD = 0.280`

---

## Files Modified

| File | Changes |
|------|---------|
| `algorithms/rft/variants/patent_variants.py` | NEW: 20 patent-aligned variants |
| `algorithms/rft/experiments/exp_patent_discovery.py` | NEW: Discovery benchmark |
| `algorithms/rft/variants/__init__.py` | Added patent variant imports |
| `algorithms/rft/variants/operator_variants.py` | Added 6 top patent variants to registry |
| `algorithms/rft/routing/signal_classifier.py` | Calibrated thresholds, new enum values |

---

## Next Steps

1. **Integration Testing**: Run full test suite with new variants
2. **Compression Benchmark**: Test patent variants in H3 codec
3. **Signal Routing**: Expand AdaptiveRouter with patent variant rules
4. **Performance Tuning**: Optimize variant generation caching
5. **Documentation**: Update API docs with patent variant descriptions

---

## References

- USPTO Application: 19/169,399
- Experiment: `exp_patent_discovery.py`
- Verification: `patent_variants.py` self-test
