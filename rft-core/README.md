# /rft-core/

## CANONICAL SCIENTIFIC REFERENCE IMPLEMENTATION

This directory is the **canonical scientific reference** for the Resonant Fourier Transform (RFT).

### Scope (MUST)
This section contains **only**:
- Formal RFT definitions
- Operator-family specification
- Proofs / propositions / assumptions
- Canonical reference implementation
- Unit tests proving invariants (unitarity, energy, stability)
- Reproducible benchmarks **with fairness notes**

### Language policy
Allowed language in this section:
- “Defined”
- “Proven”
- “Demonstrated under X conditions”
- “Fails under Y conditions”

Not allowed in this section:
- “Breakthrough”
- “Better than FFT” (without a regime qualifier)
- “OS”, “platform”, “ecosystem”

---

## Canonical Locations (authoritative code lives here)

This repo historically stores the canonical implementation under `algorithms/` and the tests under `tests/`.
This directory is an **explicit scientific index** that points to those canonical sources (no duplication).

### Formal definitions / operator family
- `algorithms/rft/kernels/resonant_fourier_transform.py`
- `algorithms/rft/core/canonical_true_rft.py` (API-stable canonical wrapper)
- `algorithms/rft/theory/formal_framework.py`
- `algorithms/rft/theory/theoretical_analysis.py`

### Proofs / propositions / assumptions
- `algorithms/rft/theorems/`
- `docs/proofs/`

### Unit tests (invariants)
Focus: tests that assert **invariants** (unitarity/energy/stability) rather than applications.
- `tests/rft/test_variant_unitarity.py`
- `tests/validation/`

### Reproducible benchmarks (fair comparisons)
Benchmarks belong under `benchmarks/` and should include fairness notes.
- `benchmarks/`
- `algorithms/rft/benchmarks/`

---

## Contribution rules
- If you add an application demo/UI/OS feature, it **does not** belong here (see `/demos/`).
- If you add exploratory or speculative variants, they **do not** belong here (see `/rft-experiments/`).
- If you add empirical evaluations, datasets, or sweeps, they **do not** belong here (see `/rft-validation/`).
