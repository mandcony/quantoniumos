# /rft-validation/

## REPRODUCIBLE EMPIRICAL EVALUATION

This directory is the empirical evaluation layer for RFT.

### Scope
Contains:
- Medical, signal, edge benchmarks
- Parameter sweeps
- Negative results
- Runtime/energy comparisons
- Dataset citations

### Mandatory comparison rules
Every comparison MUST document:
- Compute budget (hardware, threads, time budget)
- Parameter matching (what was held constant, what was tuned)
- When RFT loses (regimes / datasets / parameter regions)

---

## Canonical locations (evaluation code lives here)

- Benchmarks:
  - `benchmarks/` (Class A–E suites)
  - `algorithms/rft/benchmarks/`
- Validation suites:
  - `tests/validation/`
  - `tests/medical/`
- Datasets and citations:
  - `data/README.md`
  - `docs/medical/`

---

## Output / artifacts
Store outputs under:
- `data/` (JSON, configs, result dumps)
- `figures/` (plots)

Do not commit large generated artifacts here unless they are explicitly part of a reproducibility package.
