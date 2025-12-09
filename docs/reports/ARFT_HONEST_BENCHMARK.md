# ARFT vs DCT: Honest Benchmark Summary

## Executive Summary

> **"On the tested ECG and EEG segments, a per-signal ARFT preserves basic diagnostic proxies (R-peaks, RR-intervals, and band powers) at least as well as DCT at normal compression rates, and degrades them less than DCT at moderate compression (10% coefficients), but both transforms fail at extreme compression (5%) and none of this constitutes clinical validation."**

---

## Scope & Limitations

- **ECG**: ONE 30s segment (10,800 samples, 37 beats) from MIT-BIH record 100
- **EEG**: ONE 39.1s segment (10,000 samples), synthetic with known band powers
- **No multi-patient cohort, no real clinical endpoints**
- **This is transform behavior evidence, not clinical validation**

---

## ECG Results (Detailed)

### R-Peak Detection Sensitivity

| Keep | DCT | ARFT | Verdict |
|------|-----|------|---------|
| 5%   | 1.000 | 1.000 | **Tie** - Both perfect |
| 10%  | 1.000 | 1.000 | **Tie** - Both perfect |
| 20%  | 1.000 | 1.000 | **Tie** - Both perfect |
| 30%  | 1.000 | 1.000 | **Tie** - Both perfect |
| 50%  | 1.000 | 1.000 | **Tie** - Both perfect |

Neither method misses beats at any compression level in this segment.

### RR-Interval Correlation

| Keep | DCT RR-corr | ARFT RR-corr | Honest Verdict |
|------|-------------|--------------|----------------|
| 5%   | 0.244       | -0.148       | **Both BAD** - Not diagnostically safe |
| 10%  | 0.048       | 0.281        | **ARFT better** - But still weak |
| 20%  | 0.9993      | 0.9991       | **Tie** - Both perfect |
| 30%  | 0.9986      | 0.9987       | **Tie** - Both perfect |
| 50%  | 0.9986      | 0.9987       | **Tie** - Both perfect |

### PRD (Signal Distortion)

| Keep | DCT PRD | ARFT PRD | Winner |
|------|---------|----------|--------|
| 5%   | 20.63%  | 16.01%   | ARFT (-22% distortion) |
| 10%  | 11.53%  | 8.75%    | ARFT (-24% distortion) |
| 20%  | 3.75%   | 2.96%    | ARFT (-21% distortion) |
| 30%  | 1.56%   | 1.33%    | ARFT (-15% distortion) |
| 50%  | 0.68%   | 0.51%    | ARFT (-25% distortion) |

### ECG Interpretation

- **5% keep**: RR timing is garbage for BOTH. Neither is diagnostically safe.
- **10% keep**: ARFT is clearly better (lower PRD, higher RR-corr), but still not great.
- **20-50% keep**: Both are essentially perfect. Differences are noise in 4th decimal place.

**The only real ARFT advantage is at 10% keep. At 20-50% it's a tie. At 5% both fail.**

---

## EEG Results (Band Power Preservation)

| Keep | DCT Avg | ARFT Avg | Gap |
|------|---------|----------|-----|
| 5%   | 0.978   | 0.987    | +0.009 |
| 10%  | 0.984   | 0.991    | +0.007 |
| 20%  | 0.990   | 0.995    | +0.005 |
| 30%  | 0.993   | 0.998    | +0.005 |
| 50%  | 0.998   | 0.999    | +0.001 |

### EEG Interpretation

- DCT is already very good (>97.8% at all levels)
- ARFT is consistently ~0.5-1% better
- Single segment, no real sleep-stage classifier tested

---

## What We Can Actually Claim

### ✅ Legit, Defensible Claims:

1. **ARFT is not breaking clinical proxies in the normal range**
   - At 20-50% keep: R-peak sensitivity 100%, RR-corr ≈ 1, ARFT has lower PRD
   - ARFT behaves at least as well as DCT for basic ECG diagnostics

2. **ARFT improves when compression is pushed moderately hard**
   - At 10% keep: ARFT keeps all beats, better RR-corr, lower PRD
   - If someone needs aggressive compression, ARFT hurts less

3. **Extreme compression is unsafe for BOTH transforms**
   - At 5%: RR-corr is bad for both (ARFT worse)
   - Neither should be advertised as preserving diagnostic fidelity here

### ❌ Claims We Cannot Make:

- "ARFT is clinically validated" - **No, this is ONE segment**
- "ARFT wins 8/10 tests" - **Misleading** - 3 ECG wins are not equally meaningful
- "ARFT beats DCT on medical data" - **Overstated** - 20-50% are ties

---

## Compression Benchmark (Separate from Diagnostics)

On 20 unseen ECG segments (512 samples each), ARFT beats DCT on pure PRD:

| Keep | DCT PRD | ARFT PRD | Improvement |
|------|---------|----------|-------------|
| 5%   | 52.1%   | 40.5%    | +22% |
| 10%  | 28.4%   | 21.5%    | +24% |
| 20%  | 9.2%    | 7.2%     | +21% |
| 30%  | 3.6%    | 3.1%     | +16% |
| 50%  | 1.4%    | 1.2%     | +16% |

**This is verified with proper normalization and unitarity checks.**

---

## Technical Verification

All transforms passed bug checks:
- Roundtrip error: DCT=5e-15, FFT=3e-15, ARFT=9e-13 (all unitary)
- Parseval identity: All preserve energy to machine precision
- PRD computed identically for all transforms

---

## Conclusion

ARFT provides **measurable compression improvement** (~20% lower PRD) over DCT on real ECG data, which translates to **slightly better diagnostic proxy preservation at moderate compression** (10-20%), but at normal compression rates (30-50%) the clinical difference is negligible, and at extreme compression (5%) both fail.

**This is transform behavior evidence, not clinical validation.**

---

FOR RESEARCH USE ONLY - NOT FOR CLINICAL DIAGNOSIS

Generated: December 2025
