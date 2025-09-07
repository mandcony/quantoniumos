
# QUANTONIUMOS QUICK START COMMANDS

## 🚀 Unified Boot Script (RECOMMENDED)
```bash
# Complete system boot with all engines and validation
python quantonium_boot.py

# Desktop mode (default)
python quantonium_boot.py --mode desktop

# Console mode  
python quantonium_boot.py --mode console

# Assembly engines only
python quantonium_boot.py --assembly-only

# System status check
python quantonium_boot.py --status
```

## 🎯 Individual Component Launchers
```bash
# Main frontend launcher
python frontend/launch_quantonium_os.py

# 3-Engine system launcher (OS + Crypto + Quantum)
python ASSEMBLY/quantonium_os.py

# Console interface
python frontend/quantonium_os_main.py --console
```

## 🧪 Run System Benchmarks  
```bash
# Full system validation
python validation/benchmarks/QUANTONIUM_BENCHMARK_SUITE.py

# ASSEMBLY 3-engine benchmarks
python ASSEMBLY/quantonium_os.py --benchmark

# View final validation results
python validation/analysis/QUANTONIUM_FINAL_VALIDATION.py
```

## 💡 Launch Individual Apps
```bash
# Productivity
python apps/q_notes.py           # Q-Notes text editor
python apps/q_vault.py           # Q-Vault secure storage

# Visualization & Monitoring  
python apps/qshll_system_monitor.py  # System monitor + 3D RFT visualizer
python apps/quantum_simulator.py     # Quantum circuit simulator

# Cryptography & Security
python apps/enhanced_rft_crypto.py   # RFT crypto engine
python apps/quantum_crypto.py        # Quantum crypto tools
python apps/rft_validation_suite.py  # RFT validation
```

## 📋 Check Project Status
```bash
type PROJECT_STATUS.json
cat PROJECT_SUMMARY.json
```

## ✨ Current Features
- **Unified Frontend**: Single centered quantum logo with expandable app dock ✅
- **Golden Ratio Design**: Mathematically precise proportions ✅  
- **3-Engine Architecture**: OS + Crypto + Quantum in streamlined ASSEMBLY ✅
- **Assembly Optimization**: C/Assembly quantum compression (1M+ qubits) ✅
- **Complete Validation**: Performance benchmarks and mathematical proofs ✅
- **Production Ready**: Streamlined architecture with single frontend ✅
