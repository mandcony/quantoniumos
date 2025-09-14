
# QuantoniumOS Quick Start

## Launch the System

```bash
cd quantoniumos
python quantonium_boot.py
```

This starts the full desktop environment with:
- System initialization and dependency checks
- Assembly engine compilation
- Desktop interface with app launcher

Click the Q logo in the center to access applications.

## Individual Applications

```bash
# Quantum simulator (1000+ qubit vertex encoding)
python src/apps/quantum_simulator.py

# Secure note-taking
python src/apps/q_notes.py

# Encrypted storage  
python src/apps/q_vault.py

# AI chat interface
python src/apps/qshll_chatbox.py

# System monitoring
python src/apps/qshll_system_monitor.py

# Cryptography tools
python src/apps/quantum_crypto.py
```

## Run Validation Tests

```bash
# Mathematical validation of RFT
python tests/tests/comprehensive_validation_suite.py

# Cryptographic testing
python tests/crypto/crypto_performance_test.py

# Performance benchmarks  
python tests/benchmarks/QUANTONIUM_BENCHMARK_SUITE.py
```

## Key Features to Try

### Quantum Simulator
- Set qubit count up to 1000
- Run Grover's search on vertex encoding
- Test quantum algorithms with RFT acceleration
- Compare classical vs RFT scaling

### Q-Notes
- Markdown editing with live preview
- Automatic saving and search
- Research note organization

### Q-Vault  
- AES-256 encrypted storage
- Master password protection
- RFT-enhanced key derivation

## System Requirements

- Python 3.8+ with PyQt5, NumPy, SciPy, matplotlib
- C compiler for assembly components (optional but recommended)
- 4GB+ RAM for large quantum simulations
- Windows or Linux

## Troubleshooting

**Missing dependencies**: Install with `pip install PyQt5 numpy scipy matplotlib`

**Assembly compilation fails**: System falls back to Python implementation

**Apps don't launch**: Run `python quantonium_boot.py` for integrated environment

**Performance issues**: Check `tests/benchmarks/` for optimization settings
