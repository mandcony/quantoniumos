# Contributing to QuantoniumOS

Thank you for your interest in contributing to QuantoniumOS symbolic resonance computing research!

## Quick Start

1. Fork the repository
2. Clone: `git clone https://github.com/[your-username]/quantoniumos.git`
3. Build: See [README.md](README.md) for build instructions
4. Test: `python3 test_v2_comprehensive.py`

## Development Setup

### Dependencies

```bash
# Essential tools
sudo apt install build-essential python3 python3-pip
pip install pybind11 numpy scipy

# Build C++ engine
c++ -O3 -march=native -flto -DNDEBUG -Wall -shared -std=c++17 -fPIC \
  $(python3 -m pybind11 --includes) enhanced_rft_crypto_bindings_v2.cpp \
  -o enhanced_rft_crypto$(python3-config --extension-suffix)
```

## Code Standards

### C++ (Cryptographic Engine)

- **C++17** standard
- **Optimization**: Use `-O3 -march=native -flto` for performance
- **Testing**: Validate cryptographic properties (avalanche, etc.)
- **Documentation**: Comment mathematical operations

### Python (Wrappers & Research)

- **PEP 8** style
- **Type hints** for function signatures
- **Docstrings** for mathematical operations
- **Unit tests** for new functionality

## Testing Requirements

Before submitting changes:

```bash
# Comprehensive validation
python3 test_v2_comprehensive.py

# RFT mathematical validation
python3 publication_ready_validation.py

# Performance benchmarks
python3 test_final_v2.py
```

## Patent Considerations

⚠️ **Important**: This repository implements patented technology (Application 19/169,399).

- **Research contributions**: Welcome under fair use
- **Commercial modifications**: May require licensing
- **Core algorithms**: Ensure patent claims remain validated

## Pull Request Process

1. **Create branch**: `git checkout -b feature/your-feature`
2. **Make changes**: Implement your improvements
3. **Test thoroughly**: All validation tests must pass
4. **Document**: Update relevant documentation
5. **Submit PR**: Include clear description and test results

## Areas for Contribution

### Research Components

- RFT mathematical analysis and optimization
- Golden ratio manifold mapping improvements
- Validation framework enhancements

### Performance Optimization

- C++ engine optimizations
- Parallel processing implementations
- Memory usage improvements

### Security Analysis

- Cryptographic property analysis
- Side-channel resistance testing
- Formal verification methods

## Questions?

Open an issue for:

- Technical questions about the implementation
- Research collaboration inquiries
- Bug reports or performance issues

---

For detailed development setup, see [QUANTONIUM_DEVELOPER_GUIDE.md](QUANTONIUM_DEVELOPER_GUIDE.md). 