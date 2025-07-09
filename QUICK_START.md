# Quick Start Guide for Quantonium OS

## 🚀 Get Started in 5 Minutes

### Prerequisites
- Python 3.11 or 3.12
- 8GB+ RAM recommended
- Git

### 1. Clone and Setup
```bash
git clone https://github.com/mandcony/quantoniumos.git
cd quantoniumos
```

### 2. Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run Tests (Optional but Recommended)
```bash
# Test core functionality
python tests/test_rft_roundtrip.py
python tests/test_geowave_kat.py
python test_quantonium_analysis.py

# Run performance benchmark
python benchmark_throughput.py
```

### 4. Start the Application
```bash
# Simple start
python main.py

# Production mode with Gunicorn
gunicorn --bind 0.0.0.0:5000 main:app

# With database (optional)
export DATABASE_URL="sqlite:///quantonium.db"
python main.py
```

### 5. Access the Interface
- **Web Interface**: http://localhost:5000
- **API Documentation**: http://localhost:5000/docs
- **Health Check**: http://localhost:5000/health

## 🧪 Try the Quantum Features

### Generate Community Summary
```bash
python community_summary_generator.py
```

### Run Quantum Desktop Demo
```bash
python live_quantum_demo.py
```

### Launch Qt Application
```bash
python qt_app_example.py
```

## 📊 Validate Performance Claims
```bash
# Full validation suite
python test_quantonium_analysis.py

# Benchmark throughput
python benchmark_throughput.py

# Expected results:
# - SHA-256: >1GB/s throughput
# - RFT: Energy preservation <0.001%
# - Quantum entropy: Chi-square <4.0
```

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure you're in the right directory
   cd quantoniumos
   # Reinstall dependencies
   pip install -r requirements.txt
   ```

2. **Permission Errors**
   ```bash
   # Make scripts executable
   chmod +x *.sh
   ```

3. **Performance Issues**
   ```bash
   # Check available modules
   python -c "from core.encryption import *; print('Core modules loaded')"
   ```

### Need Help?
- Check the [full documentation](QUANTONIUM_OS_DOCUMENTATION.md)
- Review [troubleshooting guide](troubleshoot_all_modules.py)
- Submit issues on GitHub

## ⚡ What You Get

- **Quantum Algorithms**: RFT, geometric waveform encryption
- **High Performance**: Multi-GB/s cryptographic throughput  
- **Security**: Enterprise-grade container security
- **Patent Protection**: USPTO #19/169,399
- **Full Source**: MIT license for academic use

Ready to explore quantum computing? Start with `python main.py`! 🚀
