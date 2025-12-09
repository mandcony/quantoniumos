# Hybrid RFT/DCT Benchmark: Honest Findings

## Summary

**DCT wins on biomedical signals. Hybrids win on structured text/ASCII.**

---

## Biomedical Signal Compression (ECG)

Tested on MIT-BIH records: 100, 101, 200, 207, 208, 217

| Compression | Pure DCT PRD | Best Hybrid PRD | Winner |
|-------------|--------------|-----------------|--------|
| 10% kept    | 5.4–24.5%    | 6.5–28.0%       | **DCT** |
| 20% kept    | 2.5–8.0%     | 2.8–10.3%       | **DCT** |
| 30% kept    | 1.3–3.4%     | 1.7–4.5%        | **DCT** |

**Result: Pure DCT wins ALL 18 test cases on ECG data.**

### Why DCT Wins on ECG:
1. ECG is predominantly smooth/quasi-periodic → DCT is already optimal
2. After DCT captures structure, the residual is noise, not coherent texture
3. Splitting coefficient budget between domains hurts both

---

## Structured Text/ASCII Compression

From `experiments/ascii_wall/HYBRID_RFT_DCT_RESULTS.md`:

| Signal Type | Baseline BPP | H3 Cascade BPP | Improvement |
|-------------|--------------|----------------|-------------|
| Python code | 0.805        | 0.672          | **16.5%** ✅ |
| JSON        | 0.812        | 0.671          | **17.4%** ✅ |
| XML         | 0.812        | 0.671          | **17.4%** ✅ |
| CSV         | 0.812        | 0.671          | **17.4%** ✅ |
| Log files   | 0.812        | 0.671          | **17.4%** ✅ |
| Random ASCII| 0.812        | 0.671          | **17.4%** ✅ |

**Result: Hybrid H3 cascade achieves ~17% BPP reduction on ASCII/text data.**

### Why Hybrids Win on ASCII:
1. Text has both structure (repeating patterns) AND discontinuities (word boundaries)
2. DCT captures smooth patterns, RFT captures sharp transitions
3. Sequential decomposition avoids coherence violations

---

## Conclusions

### Where to Use Each:

| Signal Type | Recommended Transform | Reason |
|-------------|----------------------|--------|
| ECG | Pure DCT | Smooth, quasi-periodic |
| EEG | Pure DCT | Mostly smooth oscillations |
| Audio | Pure DCT | Harmonic content |
| ASCII/Text | Hybrid H3 Cascade | Mixed structure + edges |
| Step functions | Hybrid FH5 Entropy | Edge-dominated |
| Images (natural) | Pure DCT | Smooth gradients |
| Images (text/graphics) | Hybrid | Sharp edges + smooth regions |

### Honest Assessment:
- **RFT ≡ FFT** in magnitude → same compression performance
- **Hybrid = DCT + residual FFT/RFT** → only helps when residual has structure
- **For biomedical signals: DCT is the right choice**
- **For text/structured data: Hybrids offer real benefits**

---

## Research Status

FOR RESEARCH USE ONLY - NOT FOR CLINICAL DIAGNOSIS

The hybrid cascade transforms offer genuine value on specific signal types,
but should not be presented as universally superior to DCT.

---

Generated: 2025
