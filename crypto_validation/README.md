# QuantoniumOS Cryptographic Validation Suite

## 📁 Directory Structure

This folder contains all cryptographic validation artifacts for easy discovery and reproducible research:

```
crypto_validation/
├── README.md                  # This overview (you are here)
├── scripts/                   # Validation and test scripts
│   ├── avalanche_analysis.py     # Avalanche effect testing
│   ├── cipher_validation.py      # Enhanced RFT Crypto v2 validation
│   ├── performance_benchmarks.py # Throughput and timing analysis
│   ├── geometric_hash_tests.py   # Geometric waveform hashing tests
│   └── comprehensive_crypto_suite.py # Master validation runner
├── test_vectors/              # Reference test vectors and data
│   ├── paper_validation_vectors.json # Exact vectors from research paper
│   ├── known_answer_tests.json      # KAT for regression testing
│   └── edge_case_vectors.json       # Boundary condition tests
├── results/                   # Output artifacts and reports
│   ├── latest_validation_report.json # Most recent results
│   ├── historical_results/          # Timestamped validation runs
│   └── paper_metrics_verification.json # Paper claim verification
└── benchmarks/               # Performance analysis artifacts
    ├── throughput_analysis.json     # MB/s measurements
    ├── memory_usage_profiles.json   # Memory efficiency data
    └── comparative_benchmarks.json  # vs other crypto systems
```

## 🎯 Purpose

This validation suite serves three primary purposes:

1. **Research Reproducibility**: Exact reproduction of paper metrics (avalanche 0.438/0.527, throughput 9.2 MB/s)
2. **Artifact Discovery**: Centralized location for all crypto validation evidence
3. **Regression Testing**: Continuous validation that implementation remains consistent

## 📊 Key Validation Metrics

Based on the QuantoniumOS research paper, this suite validates:

### Enhanced RFT Crypto v2 (48-Round Feistel)
- ✅ **Message Avalanche**: Target 0.438, Range [0.43-0.45]
- ✅ **Key Avalanche**: Target 0.527, Range [0.52-0.53] 
- ✅ **Key Sensitivity**: Target 0.495, Range [0.49-0.50]
- ✅ **Throughput**: Target 9.2 MB/s, Min 8.0 MB/s
- ✅ **48 Rounds**: Exact implementation compliance
- ✅ **AEAD Mode**: Authenticated encryption validation
- ✅ **Domain Separation**: HKDF key derivation verification

### Geometric Waveform Hashing
- ✅ **RFT Pipeline**: x → Ψ(x) → Manifold → Topological → Digest
- ✅ **Deterministic**: Same input produces same output
- ✅ **Diffusion**: Small input changes cause large output changes
- ✅ **Avalanche**: Bit-flip propagation analysis

### Unitary RFT Integration
- ✅ **Unitarity**: ∥Ψ†Ψ − I∥₂ < 10⁻¹² for crypto transforms
- ✅ **Golden Ratio**: φ = (1 + √5)/2 parameterization
- ✅ **Reconstruction**: Perfect round-trip accuracy

## 🚀 Quick Start

### Run Complete Validation Suite
```bash
cd /workspaces/quantoniumos/crypto_validation/scripts
python comprehensive_crypto_suite.py
```

### Check Paper Metrics Compliance
```bash
python cipher_validation.py --verify-paper-claims
```

### Generate Test Vectors
```bash
python avalanche_analysis.py --generate-vectors --output ../test_vectors/
```

### Performance Benchmarking
```bash
python performance_benchmarks.py --full-suite --output ../benchmarks/
```

## 📈 Expected Results

### Passing Validation Criteria
- Message avalanche: 0.438 ± 0.005
- Key avalanche: 0.527 ± 0.005  
- Throughput: ≥ 9.0 MB/s
- Unitarity error: < 1e-12
- Round-trip accuracy: < 1e-15

### Paper Compliance Verification
All scripts verify implementation matches exact specifications from:
> "QuantoniumOS: A Hybrid Resonance–Symbolic Framework with a Unitary Resonance Fourier Transform and an Enhanced Feistel-Based Cryptosystem"

## 🔬 Research Integration

### For Academic Publication
- Results stored in reproducible JSON format
- Timestamped validation runs for verification
- Statistical analysis with confidence intervals
- Comparative benchmarks vs classical systems

### For Patent Documentation
- Exact implementation verification
- Performance characterization data
- Security property validation
- Test vector generation for claims

## 📋 Validation Checklist

Before considering crypto implementation complete:

- [ ] **Avalanche Analysis**: All tests pass within tolerance
- [ ] **Performance**: Meets or exceeds paper benchmarks
- [ ] **Unitarity**: Mathematical properties preserved
- [ ] **AEAD**: Authenticated encryption working correctly
- [ ] **Test Vectors**: Known answer tests pass
- [ ] **Edge Cases**: Boundary conditions handled
- [ ] **Regression**: Historical consistency maintained

## 🔗 Related Components

This validation suite integrates with:
- `/core/enhanced_rft_crypto_v2.py` - Implementation under test
- `/core/canonical_true_rft.py` - Unitary transform foundation
- `/core/geometric_waveform_hash.py` - Hashing pipeline
- `/tests/rft_scientific_validation.py` - Mathematical validation
- `/ASSEMBLY/` - High-performance implementation

---

**Status**: Ready for comprehensive cryptographic validation and research artifact generation.
