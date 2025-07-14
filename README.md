# QuantoniumOS
[![License: Patent Pending](https://img.shields.io/badge/License-Patent%20Pending-orange.svg)](LICENSE) [![Patent: USPTO #19/169,399](https://img.shields.io/badge/Patent-USPTO%20%2319%2F169%2C399-red.svg)](https://patents.uspto.gov/) [![Version](https://img.shields.io/badge/Version-1.0.0-blue)](https://github.com/mandcony/quantoniumos) [![API Status](https://img.shields.io/badge/API-Live-brightgreen)](http://localhost:5000/health)

**Patent-protected quantum cryptographic platform seeking academic peer review and validation.**

## 🚀 Quick Start

```bash
# Clone and run
git clone https://github.com/mandcony/quantoniumos.git
cd quantoniumos
python setup_local_env.py
python main.py

# Test the API
curl http://localhost:5000/api/health
curl http://localhost:5000/docs  # Interactive API documentation
```

## ⚡ What This Is

**USPTO Patent Application #19/169,399** - Working implementation of novel cryptographic algorithms:

✅ **Resonance-based encryption** using geometric waveforms  
✅ **Quantum-inspired entropy generation** for secure randomness  
✅ **High-performance C++ engine** with Python API  
✅ **Production-ready infrastructure** with Docker deployment  

## 🎯 Academic Research Platform

**Seeking validation from:**
- **Cryptographers** - Security analysis of patented algorithms
- **Mathematicians** - Formal verification of foundations
- **Performance researchers** - Benchmarking vs. established methods
- **Quantum scientists** - Validation of quantum-inspired approaches

## 🏗️ Architecture

```
Web API (Python Flask) → Core Engine (C++) → Cryptographic Algorithms
     ↓                       ↓                      ↓
  REST Interface        Eigen Math Library    Patent-Protected Methods
```

**Key Components:**
- `/api/` - REST API endpoints
- `/core/` - C++ cryptographic engine  
- `/secure_core/` - Patent-protected implementations
- `/auth/` - Authentication and security
- `/tests/` - Comprehensive validation suite

## 📖 Documentation

- **[API Docs](http://localhost:5000/docs)** - Interactive Swagger interface
- **[Developer Guide](QUANTONIUM_DEVELOPER_GUIDE.md)** - Setup and contribution
- **[License Details](LICENSE)** - Patent and usage terms

## 🔒 Intellectual Property

- **Patent Pending:** USPTO Application #19/169,399
- **Non-Commercial Use:** Free for research, education, and validation
- **Commercial License:** Contact for business applications
- **Academic Collaboration:** Joint research opportunities available

## 🎯 Project Status

🟢 **Production Ready** - Full API with Docker deployment  
🟢 **Patent Protected** - USPTO application filed  
🟢 **Seeking Review** - Academic validation in progress  
🟢 **Open Research** - Code available for peer review

## License

- **Non-Commercial:** [LICENSE](LICENSE) - Free for education and research
- **Commercial:** [LICENSE_COMMERCIAL.md](LICENSE_COMMERCIAL.md) - Contact for business use

To verify the C++ implementation:

```powershell
.\run_simple_test.bat
```

For more comprehensive testing:

```powershell
.\run_robust_tests.bat
```

## API Endpoints

The simplified app exposes these key endpoints:

- `/api/status` - Check system status
- `/api/wave/compute` - Compute wave properties
- `/api/resonance/check` - Verify resonance calculations
- `/api/symbolic/eigenvector` - Calculate symbolic eigenvectors

## Implementation Notes

This repository contains:

- Complete wave-based mathematics
- Quantum-inspired geometric transformations
- Resonance-based process management
- High-performance C++ linear algebra operations using Eigen

All core scientific components have been verified through automated tests.

## Documentation

For more detailed information, see:
- [Local Setup Guide](LOCAL_SETUP.md)
- [Simplified Solution](SIMPLIFIED_SOLUTION.md)
- [Developer Guide](QUANTONIUM_DEVELOPER_GUIDE.md)

## License

See [LICENSE](LICENSE) for terms of use.
