# SOTA Compression Benchmark Results

**Date:** 2025-11-27 21:07:42

## Summary

**Conclusion:** INCOMPLETE: Missing RFT or SOTA results for comparison.

## Results by Corpus

### english_prose

- **Size:** 89,000 bytes
- **H₀:** 4.3834 bpc
- **H₃:** 0.4479 bpc

| Compressor | BPC | Ratio | Verified |
|------------|-----|-------|----------|
| brotli | 0.0335 | 0.0042 | ✓ |
| brotli | 0.0390 | 0.0049 | ✓ |
| zstd | 0.0454 | 0.0057 | ✓ |
| zstd | 0.0455 | 0.0057 | ✓ |
| lzma | 0.0608 | 0.0076 | ✓ |
| gzip | 0.0908 | 0.0113 | ✓ |
| bz2 | 0.1121 | 0.0140 | ✓ |
| rft_sparse | 0.5019 | 0.0627 | ✗ |

### source_code

- **Size:** 48,500 bytes
- **H₀:** 4.5745 bpc
- **H₃:** 0.4255 bpc

| Compressor | BPC | Ratio | Verified |
|------------|-----|-------|----------|
| brotli | 0.0632 | 0.0079 | ✓ |
| brotli | 0.0699 | 0.0087 | ✓ |
| zstd | 0.0749 | 0.0094 | ✓ |
| zstd | 0.0774 | 0.0097 | ✓ |
| lzma | 0.0963 | 0.0120 | ✓ |
| gzip | 0.1209 | 0.0151 | ✓ |
| bz2 | 0.1623 | 0.0203 | ✓ |
| rft_sparse | 0.9066 | 0.1133 | ✗ |

### xml_markup

- **Size:** 90,600 bytes
- **H₀:** 4.3208 bpc
- **H₃:** 0.3697 bpc

| Compressor | BPC | Ratio | Verified |
|------------|-----|-------|----------|
| brotli | 0.0259 | 0.0032 | ✓ |
| brotli | 0.0291 | 0.0036 | ✓ |
| zstd | 0.0340 | 0.0042 | ✓ |
| zstd | 0.0350 | 0.0044 | ✓ |
| lzma | 0.0459 | 0.0057 | ✓ |
| bz2 | 0.0742 | 0.0093 | ✓ |
| gzip | 0.0765 | 0.0096 | ✓ |
| rft_sparse | 0.4203 | 0.0525 | ✗ |

### json_data

- **Size:** 100,000 bytes
- **H₀:** 4.2153 bpc
- **H₃:** 0.6963 bpc

| Compressor | BPC | Ratio | Verified |
|------------|-----|-------|----------|
| brotli | 0.3853 | 0.0482 | ✓ |
| lzma | 0.4083 | 0.0510 | ✓ |
| zstd | 0.4097 | 0.0512 | ✓ |
| bz2 | 0.4126 | 0.0516 | ✓ |
| brotli | 0.4546 | 0.0568 | ✓ |
| zstd | 0.5466 | 0.0683 | ✓ |
| gzip | 0.6208 | 0.0776 | ✓ |
| rft_sparse | 26.6502 | 3.3313 | ✗ |

### random_ascii

- **Size:** 100,000 bytes
- **H₀:** 6.5691 bpc
- **H₃:** 0.1106 bpc

| Compressor | BPC | Ratio | Verified |
|------------|-----|-------|----------|
| bz2 | 6.6406 | 0.8301 | ✓ |
| brotli | 6.6439 | 0.8305 | ✓ |
| brotli | 6.6442 | 0.8305 | ✓ |
| zstd | 6.6458 | 0.8307 | ✓ |
| zstd | 6.6522 | 0.8315 | ✓ |
| gzip | 6.6574 | 0.8322 | ✓ |
| lzma | 6.7066 | 0.8383 | ✓ |
| rft_sparse | 46.4026 | 5.8003 | ✗ |

### repetitive

- **Size:** 99,099 bytes
- **H₀:** 3.3300 bpc
- **H₃:** 0.0081 bpc

| Compressor | BPC | Ratio | Verified |
|------------|-----|-------|----------|
| rft_sparse | 0.0013 | 0.0002 | ✗ |
| brotli | 0.0022 | 0.0003 | ✓ |
| brotli | 0.0025 | 0.0003 | ✓ |
| zstd | 0.0028 | 0.0004 | ✓ |
| zstd | 0.0033 | 0.0004 | ✓ |
| bz2 | 0.0062 | 0.0008 | ✓ |
| lzma | 0.0139 | 0.0017 | ✓ |
| gzip | 0.0269 | 0.0034 | ✓ |

### base64

- **Size:** 100,000 bytes
- **H₀:** 6.0000 bpc
- **H₃:** 0.0000 bpc

| Compressor | BPC | Ratio | Verified |
|------------|-----|-------|----------|
| brotli | 0.0590 | 0.0074 | ✓ |
| brotli | 0.0642 | 0.0080 | ✓ |
| zstd | 0.0658 | 0.0082 | ✓ |
| zstd | 0.0658 | 0.0082 | ✓ |
| lzma | 0.0755 | 0.0094 | ✓ |
| gzip | 0.1133 | 0.0142 | ✓ |
| bz2 | 0.1403 | 0.0175 | ✓ |
| rft_sparse | 0.3949 | 0.0494 | ✗ |

### mixed

- **Size:** 30,000 bytes
- **H₀:** 4.6448 bpc
- **H₃:** 0.3102 bpc

| Compressor | BPC | Ratio | Verified |
|------------|-----|-------|----------|
| brotli | 0.0803 | 0.0100 | ✓ |
| brotli | 0.0920 | 0.0115 | ✓ |
| zstd | 0.1072 | 0.0134 | ✓ |
| zstd | 0.1083 | 0.0135 | ✓ |
| lzma | 0.1419 | 0.0177 | ✓ |
| gzip | 0.1557 | 0.0195 | ✓ |
| bz2 | 0.2352 | 0.0294 | ✓ |
| rft_sparse | 0.9387 | 0.1173 | ✗ |

## Theoretical Gaps

| Corpus | RFT BPC | SOTA BPC | Gap | H₀ | H₃ |
|--------|---------|----------|-----|----|----|n
## Evidence

