# QuantoniumOS Unified Engines - Hardware Implementation

## 📊 Visualization & Results

**Complete hardware test visualizations are available in [`figures/README.md`](figures/README.md)**

Key figures include:
- 🎯 **Frequency Domain Analysis** - 10 test patterns with full spectral analysis
- ⚡ **Energy Comparison** - Performance across diverse input patterns
- 🔄 **Phase Analysis** - Complex frequency domain representation
- 📈 **Test Suite Overview** - Comprehensive dashboard (100% pass rate)
- 🏗️ **Architecture Diagram** - Complete hardware block diagram
- 🔬 **Synthesis Metrics** - FPGA resource utilization and timing

See [`HW_TEST_RESULTS.md`](HW_TEST_RESULTS.md) and [`HW_VISUALIZATION_REPORT.md`](HW_VISUALIZATION_REPORT.md) for detailed analysis.

---

## Overview

This Verilog design integrates the complete QuantoniumOS cryptographic stack into a unified hardware architecture:

1. **Canonical RFT Core** - Unitary Resonance Fourier Transform with golden-ratio parameterization
2. **RFT-SIS Hash v3.1** - Cryptographic hash with lattice-based security (SIS hardness)
3. **Feistel-48 Cipher** - 48-round Feistel network with AES S-box and ARX operations
4. **Compression Engine** - Quantum-symbolic compression interface
5. **Unified Controller** - Pipeline orchestration and mode selection

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                QuantoniumOS Unified Core                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌────────────┐   ┌──────────────┐   ┌──────────────┐    │
│  │  RFT Core  │──▶│  SIS Hash    │──▶│  Feistel-48  │    │
│  │  (N=64)    │   │  (SIS-512)   │   │  (48 rounds) │    │
│  └────────────┘   └──────────────┘   └──────────────┘    │
│        │                  │                   │            │
│        └──────────────────┴───────────────────┘            │
│                           ▼                                │
│                  ┌─────────────────┐                       │
│                  │  Unified Output │                       │
│                  │  (256-bit)      │                       │
│                  └─────────────────┘                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

### 1. Canonical RFT Core
- **Algorithm**: Unitary RFT with Ψ = Σ w_i D_φi C_σi D†_φi
- **Golden Ratio**: φ = (1 + √5) / 2 ≈ 1.618034
- **Unitarity**: Error < 10^-12 (verified in Python)
- **Transform Size**: Configurable (N=64 default)
- **Precision**: Q16.16 fixed-point arithmetic

**Implementation Details**:
- Precomputed golden-ratio phase sequence
- Gaussian kernel weights for locality
- QR-based orthonormalization
- Pipelined matrix-vector multiplication

### 2. RFT-SIS Hash v3.1
- **Security**: Based on Short Integer Solution (SIS) lattice problem
- **Parameters**: n=512, m=1024, q=3329, β=100
- **Avalanche**: 40-60% bit flip for 10^-15 coordinate changes
- **Collision Resistance**: Cryptographic expansion before normalization
- **Output**: 256-bit SHA3 digest

**Novel Features**:
- Cryptographic coordinate expansion (prevents normalization loss)
- RFT transformation for non-linear mixing
- Lattice-point computation: As mod q
- Domain separation with golden-ratio phases

### 3. Feistel-48 Cipher
- **Rounds**: 48 (exceeds AES-256 security margin)
- **Block Size**: 128 bits (64-bit L/R halves)
- **Key Size**: 256 bits master key
- **F-function**: AES S-box + MixColumns + ARX
- **Key Schedule**: HKDF with domain separation

**Cryptographic Layers**:
1. Pre/post whitening with derived keys
2. Per-round key injection (48 unique round keys)
3. Byte substitution (AES S-box)
4. MixColumns-style diffusion
5. ARX operations (Add-Rotate-XOR with φ)
6. Golden-ratio mixing for non-periodicity

### 4. Unified Pipeline
- **Mode 0**: RFT transform only
- **Mode 1**: RFT-SIS hash only
- **Mode 2**: Feistel-48 cipher only
- **Mode 3**: Full pipeline (RFT → SIS → Feistel)
- **Mode 4**: Compression (future expansion)

## Build Instructions

### Prerequisites
- **Icarus Verilog**: Open-source Verilog simulator
- **GTKWave**: Waveform viewer
- **Yosys** (optional): Open-source synthesis
- **Verilator** (optional): Linting and verification

Install on Ubuntu/Debian:
```bash
sudo apt-get install iverilog gtkwave yosys verilator
```

### Quick Start

```bash
# Clone or navigate to the QuantoniumOS directory
cd /workspaces/quantoniumos

# Run simulation
make -f quantoniumos_engines_makefile sim

# View waveforms
make -f quantoniumos_engines_makefile view

# Synthesize (resource estimation)
make -f quantoniumos_engines_makefile synth

# Lint with Verilator
make -f quantoniumos_engines_makefile verilate

# Clean build artifacts
make -f quantoniumos_engines_makefile clean
```

## Usage Examples

### Example 1: RFT Transform
```verilog
// Mode 0: RFT only
mode = 3'd0;
data_in = 128'hDEADBEEFCAFEBABEDEADBEEFCAFEBABE;
start = 1;
wait(done);
// data_out contains RFT coefficients
```

### Example 2: Cryptographic Hash
```verilog
// Mode 1: RFT-SIS Hash
mode = 3'd1;
data_in = {coordinate_x[63:0], coordinate_y[63:0]};
start = 1;
wait(done);
// data_out[255:0] contains SHA3-256 hash
```

### Example 3: Encryption
```verilog
// Mode 2: Feistel-48 cipher
mode = 3'd2;
master_key = 256'h0123456789ABCDEF...;
data_in = 128'hPLAINTEXT_BLOCK;
start = 1;
wait(done);
// data_out[127:0] contains ciphertext
```

### Example 4: Full Pipeline
```verilog
// Mode 3: RFT → SIS Hash → Feistel
mode = 3'd3;
master_key = 256'h...;
data_in = 128'h...;
start = 1;
wait(done);
// data_out[255:0] contains final authenticated output
```

## Performance Metrics

### Estimated Resource Usage (Artix-7 100T)
- **LUTs**: ~45,000 (44% utilization)
- **FFs**: ~28,000 (27% utilization)
- **DSPs**: 48 (for fixed-point multiply-accumulate)
- **BRAM**: 96 blocks (for S-boxes, RFT matrix cache)

### Throughput Estimates
- **RFT Transform**: ~10 MB/s @ 100 MHz
- **SIS Hash**: ~5 MB/s @ 100 MHz
- **Feistel-48**: ~9.2 MB/s @ 100 MHz (matches C implementation)
- **Full Pipeline**: ~3 MB/s @ 100 MHz (bottlenecked by SIS)

### Latency
- **RFT**: ~6,400 cycles (64×64 matrix multiply + orthonormalization)
- **SIS Hash**: ~15,000 cycles (expansion + RFT + lattice)
- **Feistel-48**: ~240 cycles (48 rounds × 5 pipeline stages)
- **Full Pipeline**: ~22,000 cycles (sequential execution)

## Optimization Opportunities

### 1. RFT Acceleration
- **FFT-style butterfly network**: Reduce O(N²) to O(N log N)
- **Precomputed basis cache**: Store RFT matrix in BRAM
- **Parallel column processing**: 4-8 columns simultaneously

### 2. SIS Hash Speedup
- **SHA3 Keccak core**: Use optimized IP core
- **Sparse matrix A**: Reduce lattice multiplication cost
- **Early expansion pruning**: Skip low-entropy dimensions

### 3. Feistel Pipelining
- **4-stage pipeline**: Process 4 rounds simultaneously
- **AES-NI instructions**: Use hardware AES S-box on Zynq
- **Key pre-generation**: Derive all 48 keys at startup

### 4. Full Pipeline Parallelization
- **Streaming architecture**: Overlap RFT/SIS/Feistel stages
- **Multi-block buffering**: Process 4 blocks in parallel
- **DMA integration**: Direct memory access for high throughput

## Testing

### Included Tests
1. **RFT Unitarity**: Verify Ψ^H Ψ = I (error < 10^-12)
2. **SIS Avalanche**: Check 40-60% bit flip for small input changes
3. **Feistel Correctness**: Encrypt/decrypt round-trip
4. **Pipeline Integration**: Full end-to-end data flow

### Running Tests
```bash
make -f quantoniumos_engines_makefile test
```

Expected output:
```
=== Test 1: RFT Transform ===
RFT Output: 0x...
RFT Energy: 12345

=== Test 2: RFT-SIS Hash ===
SIS Hash: 0xabcd...

=== Test 3: Feistel-48 Cipher ===
Ciphertext: 0x1234...

=== Test 4: Full Pipeline ===
Pipeline Output: 0x...
Throughput: 3000000 bits/sec
```

## Integration with Python Implementation

This Verilog design matches the Python reference implementations:
- **canonical_true_rft.py**: Golden-ratio RFT with QR orthonormalization
- **rft_sis_hash_v31.py**: Cryptographic expansion + SIS lattice
- **enhanced_rft_crypto_v2.py**: 48-round Feistel with HKDF

### Verification Strategy
1. Run Python tests to generate ground truth
2. Export test vectors to CSV
3. Import vectors into Verilog testbench
4. Compare hardware outputs bit-exact

## FPGA Deployment

### Xilinx Vivado Workflow
```bash
vivado -mode batch -source quantoniumos_unified_engines_synthesis.tcl
```

### Intel Quartus Workflow
```bash
quartus_sh --flow compile quantoniumos_unified_core
```

### Lattice Diamond Workflow
```bash
diamondc quantoniumos_unified_core.tcl
```

## 📊 Visualization & Analysis

### Generate Hardware Figures

To create comprehensive visualizations of test results:

```bash
cd hardware
python visualize_hardware_results.py
```

**Generated Outputs:**
- `figures/hw_rft_frequency_spectra.png/pdf` - Frequency domain analysis for all tests
- `figures/hw_rft_energy_comparison.png/pdf` - Energy distribution across patterns
- `figures/hw_rft_phase_analysis.png/pdf` - Complex phase representation
- `figures/hw_rft_test_overview.png/pdf` - Comprehensive test dashboard
- `figures/hw_architecture_diagram.png/pdf` - Hardware block diagram
- `figures/hw_synthesis_metrics.png/pdf` - FPGA metrics and timing
- `HW_VISUALIZATION_REPORT.md` - Detailed analysis report

**Features:**
- ✅ Parses simulation logs automatically
- ✅ Generates publication-quality figures (PNG + PDF)
- ✅ Comprehensive statistical analysis
- ✅ Hardware architecture diagrams
- ✅ Synthesis and timing metrics
- ✅ Test coverage visualization

See [`figures/README.md`](figures/README.md) for detailed documentation of each figure.

---

## Security Considerations

### Side-Channel Resistance
- **Constant-time operations**: All crypto paths take fixed cycles
- **No data-dependent branches**: Avoid timing leaks
- **Masked S-box**: Consider threshold implementation for AES S-box
- **TRNG integration**: Add true random number generator for nonces

### Formal Verification
- **Bounded model checking**: Verify state machine correctness
- **Equivalence checking**: Match against Python reference
- **Coverage analysis**: Ensure all branches tested

## Future Enhancements

1. **Quantum Entanglement Module**: Add Bell state preparation circuit
2. **ML Inference Engine**: Integrate TinyML for adaptive cryptography
3. **AEAD Mode**: Implement authenticated encryption with associated data
4. **Post-Quantum Extensions**: Add Kyber/Dilithium integration
5. **Hardware Security Module**: Secure key storage and anti-tampering

## License

```
SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
Copyright (C) 2025 Luis M. Minier

This file is licensed under LICENSE-CLAIMS-NC.md (research/education only).
Commercial rights require a separate patent license from the author.
```

## References

- **QuantoniumOS Paper**: See `docs/TECHNICAL_SUMMARY.md`
- **RFT Theory**: See `docs/RFT_THEOREMS.md`
- **Cryptographic Analysis**: See `docs/PATENT_CLAIMS_IMPLEMENTATION_ANALYSIS.md`
- **Benchmarks**: See `COMPETITIVE_BENCHMARKS_ADDED.md`

## Contact

For hardware-specific questions or FPGA optimization consulting:
- Email: [Author contact from docs/Author]
- GitHub: https://github.com/mandcony/quantoniumos
- Issues: Submit hardware bugs to GitHub issues

---

**Note**: This is a research implementation. For production deployment, conduct thorough security audits and side-channel analysis.
