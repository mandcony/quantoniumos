# Breaking the ASCII Wall: Experimental Results

## Summary by Signal Category


### ASCII Signals


#### Python Source Code

| Method | BPP | PSNR (dB) | Coherence | Sparsity | Time (ms) |
|--------|-----|-----------|-----------|----------|-----------|
| Baseline_Greedy | 0.805 | 3.07 | 5.00e-01 | 95.0% | 2.02 |
| H3_Cascade | 0.672 | 3.81 | 0.00e+00 | 96.7% | 0.68 |
| H5_Attention | 0.805 | 5.05 | 5.00e-01 | 95.0% | 0.79 |
| H6_Dictionary | 0.806 | 5.05 | 5.00e-01 | 95.0% | 1.66 |
| H7_Cascade_Attention | 0.805 | 5.04 | 0.00e+00 | 95.0% | 1.30 |

**Winner:** H3_Cascade (0.672 BPP)
**Best PSNR:** H6_Dictionary (5.05 dB)


#### JSON Data

| Method | BPP | PSNR (dB) | Coherence | Sparsity | Time (ms) |
|--------|-----|-----------|-----------|----------|-----------|
| Baseline_Greedy | 0.812 | 0.25 | 5.00e-01 | 94.9% | 0.31 |
| H3_Cascade | 0.671 | 2.69 | 0.00e+00 | 96.7% | 0.42 |
| H5_Attention | 0.812 | 5.37 | 5.00e-01 | 94.9% | 0.57 |
| H6_Dictionary | 0.815 | 10.34 | 5.00e-01 | 94.9% | 1.06 |
| H7_Cascade_Attention | 0.812 | 5.39 | 0.00e+00 | 94.9% | 1.00 |

**Winner:** H3_Cascade (0.671 BPP)
**Best PSNR:** H6_Dictionary (10.34 dB)


#### XML Data

| Method | BPP | PSNR (dB) | Coherence | Sparsity | Time (ms) |
|--------|-----|-----------|-----------|----------|-----------|
| Baseline_Greedy | 0.812 | 1.36 | 5.00e-01 | 94.9% | 0.33 |
| H3_Cascade | 0.671 | 4.67 | 0.00e+00 | 96.7% | 0.45 |
| H5_Attention | 0.812 | 6.38 | 5.00e-01 | 94.9% | 0.58 |
| H6_Dictionary | 0.815 | 12.89 | 5.00e-01 | 94.9% | 0.98 |
| H7_Cascade_Attention | 0.812 | 6.37 | 0.00e+00 | 94.9% | 0.98 |

**Winner:** H3_Cascade (0.671 BPP)
**Best PSNR:** H6_Dictionary (12.89 dB)


#### CSV Data

| Method | BPP | PSNR (dB) | Coherence | Sparsity | Time (ms) |
|--------|-----|-----------|-----------|----------|-----------|
| Baseline_Greedy | 0.812 | 15.44 | 5.00e-01 | 94.9% | 0.29 |
| H3_Cascade | 0.671 | 17.47 | 0.00e+00 | 96.7% | 0.45 |
| H5_Attention | 0.812 | 16.93 | 5.00e-01 | 94.9% | 0.57 |
| H6_Dictionary | 0.815 | 19.34 | 5.00e-01 | 94.9% | 1.00 |
| H7_Cascade_Attention | 0.812 | 16.96 | 0.00e+00 | 94.9% | 0.96 |

**Winner:** H3_Cascade (0.671 BPP)
**Best PSNR:** H6_Dictionary (19.34 dB)


#### Log File

| Method | BPP | PSNR (dB) | Coherence | Sparsity | Time (ms) |
|--------|-----|-----------|-----------|----------|-----------|
| Baseline_Greedy | 0.812 | 6.99 | 5.00e-01 | 94.9% | 0.30 |
| H3_Cascade | 0.671 | 9.87 | 0.00e+00 | 96.7% | 0.41 |
| H5_Attention | 0.812 | 9.60 | 5.00e-01 | 94.9% | 0.59 |
| H6_Dictionary | 0.815 | 11.88 | 5.00e-01 | 94.9% | 0.97 |
| H7_Cascade_Attention | 0.812 | 9.50 | 0.00e+00 | 94.9% | 1.05 |

**Winner:** H3_Cascade (0.671 BPP)
**Best PSNR:** H6_Dictionary (11.88 dB)


#### Random ASCII

| Method | BPP | PSNR (dB) | Coherence | Sparsity | Time (ms) |
|--------|-----|-----------|-----------|----------|-----------|
| Baseline_Greedy | 0.812 | 3.41 | 5.00e-01 | 94.9% | 0.33 |
| H3_Cascade | 0.671 | 4.81 | 0.00e+00 | 96.7% | 0.44 |
| H5_Attention | 0.812 | 5.65 | 5.00e-01 | 94.9% | 0.60 |
| H6_Dictionary | 0.830 | 5.67 | 5.00e-01 | 94.8% | 0.97 |
| H7_Cascade_Attention | 0.812 | 5.63 | 0.00e+00 | 94.9% | 0.97 |

**Winner:** H3_Cascade (0.671 BPP)
**Best PSNR:** H6_Dictionary (5.67 dB)


### CONTINUOUS Signals


#### Fibonacci Wave

| Method | BPP | PSNR (dB) | Coherence | Sparsity | Time (ms) |
|--------|-----|-----------|-----------|----------|-----------|
| Baseline_Greedy | 0.812 | 2.16 | 5.00e-01 | 94.9% | 0.29 |
| H3_Cascade | 0.655 | 23.19 | 0.00e+00 | 96.7% | 0.35 |
| H5_Attention | 0.812 | 8.04 | 5.00e-01 | 94.9% | 0.55 |
| H6_Dictionary | 0.817 | 23.52 | 5.00e-01 | 94.9% | 0.86 |
| H7_Cascade_Attention | 0.812 | 8.03 | 0.00e+00 | 94.9% | 0.89 |

**Winner:** H3_Cascade (0.655 BPP)
**Best PSNR:** H6_Dictionary (23.52 dB)


#### Chirp Signal

| Method | BPP | PSNR (dB) | Coherence | Sparsity | Time (ms) |
|--------|-----|-----------|-----------|----------|-----------|
| Baseline_Greedy | 0.812 | 1.66 | 5.00e-01 | 94.9% | 0.24 |
| H3_Cascade | 0.655 | 4.54 | 0.00e+00 | 96.7% | 0.44 |
| H5_Attention | 0.812 | 5.43 | 5.00e-01 | 94.9% | 0.63 |
| H6_Dictionary | 0.817 | 4.94 | 5.00e-01 | 94.9% | 1.07 |
| H7_Cascade_Attention | 0.812 | 5.15 | 0.00e+00 | 94.9% | 1.02 |

**Winner:** H3_Cascade (0.655 BPP)
**Best PSNR:** H5_Attention (5.43 dB)


#### Sine Wave

| Method | BPP | PSNR (dB) | Coherence | Sparsity | Time (ms) |
|--------|-----|-----------|-----------|----------|-----------|
| Baseline_Greedy | 0.812 | -2.99 | 5.00e-01 | 94.9% | 0.28 |
| H3_Cascade | 0.655 | 32.04 | 0.00e+00 | 96.7% | 0.38 |
| H5_Attention | 0.812 | 3.03 | 5.00e-01 | 94.9% | 0.53 |
| H6_Dictionary | 0.817 | 47.55 | 5.00e-01 | 94.9% | 0.93 |
| H7_Cascade_Attention | 0.812 | 3.03 | 0.00e+00 | 94.9% | 0.81 |

**Winner:** H3_Cascade (0.655 BPP)
**Best PSNR:** H6_Dictionary (47.55 dB)


### MIXED Signals


#### ASCII + Fibonacci

| Method | BPP | PSNR (dB) | Coherence | Sparsity | Time (ms) |
|--------|-----|-----------|-----------|----------|-----------|
| Baseline_Greedy | 0.812 | 10.30 | 5.00e-01 | 94.9% | 0.29 |
| H3_Cascade | 0.655 | 22.63 | 0.00e+00 | 96.7% | 0.35 |
| H5_Attention | 0.812 | 15.82 | 5.00e-01 | 94.9% | 0.55 |
| H6_Dictionary | 0.817 | 22.53 | 5.00e-01 | 94.9% | 0.84 |
| H7_Cascade_Attention | 0.812 | 15.85 | 0.00e+00 | 94.9% | 0.85 |

**Winner:** H3_Cascade (0.655 BPP)
**Best PSNR:** H3_Cascade (22.63 dB)


#### JSON + Sine

| Method | BPP | PSNR (dB) | Coherence | Sparsity | Time (ms) |
|--------|-----|-----------|-----------|----------|-----------|
| Baseline_Greedy | 0.812 | 2.86 | 5.00e-01 | 94.9% | 0.25 |
| H3_Cascade | 0.655 | 5.60 | 0.00e+00 | 96.7% | 0.37 |
| H5_Attention | 0.812 | 6.77 | 5.00e-01 | 94.9% | 0.51 |
| H6_Dictionary | 0.817 | 9.48 | 5.00e-01 | 94.9% | 0.85 |
| H7_Cascade_Attention | 0.812 | 6.76 | 0.00e+00 | 94.9% | 0.88 |

**Winner:** H3_Cascade (0.655 BPP)
**Best PSNR:** H6_Dictionary (9.48 dB)


## Overall Statistics

| Method | Avg BPP | Avg PSNR (dB) | Avg Coherence | Win Rate |
|--------|---------|---------------|---------------|----------|
| Baseline_Greedy | 0.812 | 4.05 | 5.00e-01 | 0.0% |
| H3_Cascade | 0.664 | 11.94 | 0.00e+00 | 100.0% |
| H5_Attention | 0.812 | 8.01 | 5.00e-01 | 0.0% |
| H6_Dictionary | 0.816 | 15.74 | 5.00e-01 | 0.0% |
| H7_Cascade_Attention | 0.812 | 7.97 | 0.00e+00 | 0.0% |