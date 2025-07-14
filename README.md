# QuantoniumOS

**A quantum computing development platform that makes quantum concepts accessible through simple APIs.**

## What This Does

QuantoniumOS is an educational platform that simulates quantum computing operations and shows how to integrate them with classical systems. Perfect for developers learning quantum programming.

**Quick Demo:**
```bash
# Get quantum randomness
curl http://localhost:5000/api/quantum/entropy

# Simulate quantum gates  
curl -X POST http://localhost:5000/api/quantum/simulate \
  -d '{"qubits": 2, "gates": [{"type": "hadamard", "target": 0}]}'
```

## Key Features

✅ **Quantum Simulation** - Hadamard, CNOT, Pauli gates and more  
✅ **Random Number Generation** - Quantum-inspired entropy  
✅ **REST API** - Easy integration with any programming language  
✅ **Cross-Platform** - Windows, Linux, macOS  
✅ **Fast Setup** - Running in 2 minutes  

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/mandcony/quantoniumos.git
cd quantoniumos
python setup_local_env.py

# 2. Start the server
python start_dev.py

# 3. Test it works
curl http://localhost:5000/api/health
```

**→ Open http://localhost:5000/docs for interactive API documentation**

## What's Quantum vs Classical

**Quantum Parts (Simulated):**
- Random number generation using quantum algorithms
- Quantum gate operations (Hadamard, CNOT, etc.)
- Quantum state vector calculations

**Classical Parts:**
- REST API server (Python Flask)
- Database storage (SQLite/PostgreSQL) 
- Web interface and documentation
- C++ performance engine (Eigen library)

**How They Work Together:**
```python
# Generate quantum randomness
quantum_key = get_quantum_entropy(256)

# Use in classical encryption (demo only - not secure)
encrypted = xor_encrypt(data, quantum_key)
```

## Use Cases

- **Learning quantum programming** without complex setup
- **Prototyping quantum algorithms** before hardware access
- **Understanding quantum-classical integration**
- **Teaching quantum computing concepts**

## Technology Stack

- **Backend:** C++ (Eigen) + Python (Flask)
- **Frontend:** HTML5/React with quantum visualizations  
- **Database:** SQLite (dev) / PostgreSQL (prod)
- **Deployment:** Docker, GitHub Actions CI/CD
- **Testing:** 90%+ coverage with pytest

## Important Notes

⚠️ **Educational Purpose:** This is for learning, not production cryptography  
⚠️ **Simulation Only:** Not running on real quantum hardware  
⚠️ **Not Cryptographically Secure:** Use established crypto libraries for real security  

## Documentation

- **[Developer Guide](QUANTONIUM_DEVELOPER_GUIDE.md)** - Complete setup and contribution guide
- **[API Docs](http://localhost:5000/docs)** - Interactive API documentation (when running)
- **[Security Notes](SECURITY_DISCLAIMER.md)** - Important security considerations

## Project Status

🟢 **Active Development** - Ready for contributors and learners  
🟢 **CI/CD Pipeline** - Automated testing and deployment  
🟢 **Cross-Platform** - Tested on Windows, Linux, macOS  

## Quick Architecture

```
Web Browser → REST API → Quantum Engine (C++) → Results
     ↓              ↓            ↓
  HTML/React   Python Flask   Eigen Math
```

## Contributing

We welcome contributions! See [QUANTONIUM_DEVELOPER_GUIDE.md](QUANTONIUM_DEVELOPER_GUIDE.md) for:
- Development environment setup
- Code style guidelines  
- Testing requirements
- Contribution workflow

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
