# Audio Signal Processing with RFT

## Executive Summary
**Status:** 🟡 Niche / Creative
**Verdict:** RFT is **inferior** to FFT/DCT for standard audio compression (MP3/AAC) and real-time EQ because audio is fundamentally harmonic (strings, pipes, voice). However, RFT offers unique **inharmonic spectral analysis** useful for sound design, metallic timbre synthesis, and transient detection.

## The Physics of Sound
- **Harmonic Sound:** Musical instruments (guitar, piano, voice) produce overtones at integer multiples ($f, 2f, 3f...$).
  - **Best Tool:** FFT (matches the physics).
- **Inharmonic Sound:** Bells, gongs, cymbals, and percussive transients have non-integer overtones.
  - **Best Tool:** RFT (matches the inharmonicity).

## Benchmark Results

| Task | FFT/DCT Performance | RFT Performance | Winner |
| :--- | :--- | :--- | :--- |
| **Compression (MP3)** | High Ratio, Low Artifacts | Low Ratio, Ringing Artifacts | 🏆 **FFT/DCT** |
| **Real-time EQ** | <1ms Latency | ~10ms Latency | 🏆 **FFT** |
| **Transient Analysis** | Smearing (Gibbs Phenomenon) | Sharp Localization | 🏆 **RFT** |
| **Timbre Synthesis** | Standard Sounds | Metallic/Alien Textures | 🤝 **Tie (Creative)** |

## Use Cases
1. **Sound Design:** Creating "alien" or "metallic" soundscapes by filtering in the $\phi$-domain.
2. **Transient Detection:** Identifying the exact onset of a drum hit where FFT windows might smear the energy.
3. **Audio Watermarking:** Embedding data in the "irrational" parts of the spectrum that MP3 encoders might ignore or preserve differently.

## Recommendation
Do not use RFT for building a standard music player or DAW. Use it as a **VST plugin** or offline processor for creative sound design and specific transient analysis tasks.
