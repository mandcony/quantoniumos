# RFT vs FFT Benchmark Results

## Executive Summary

This report presents competitive benchmarks between QuantoniumOS RFT and standard FFT implementations.

## Speed Performance

| Size | FFT Time (s) | RFT Time (s) | Ratio |
|------|-------------|-------------|-------|
| 64 | 0.0044 | 0.0124 | 2.79x |
| 128 | 0.0075 | 0.0353 | 4.74x |
| 256 | 0.0086 | 0.2669 | 31.07x |
| 512 | 0.0108 | 0.3483 | 32.11x |
| 1024 | 0.0154 | 2.3098 | 149.79x |
| 2048 | 0.0280 | 22.9849 | 821.33x |

## Accuracy Results

| Size | FFT Error | RFT Error |
|------|-----------|----------|
| 256 | 1.19e-15 | 4.71e-15 |
| 512 | 1.24e-15 | 4.98e-15 |
| 1024 | 1.40e-15 | 6.72e-15 |

## Compression Efficiency

- Original size: 147.2 MB
- FFT compressed: 73.6 MB (MSE: 0.625021)
- RFT compressed: 147.2 MB (MSE: 0.625021)
