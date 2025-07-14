# QuantoniumOS Developer Guide

## 🚀 Quick Start (Just Want to Run It?)

**Get QuantoniumOS running in 5 minutes:**

```bash
# 1. Clone and setup
git clone https://github.com/mandcony/quantoniumos.git
cd quantoniumos
python -m venv venv && source venv/bin/activate  # or venv\Scripts\activate on Windows

# 2. One-command setup
python setup_local_env.py

# 3. Start development server
python start_dev.py
```

🌐 **Access:** http://localhost:5000  
📚 **API Docs:** http://localhost:5000/docs  
🔐 **Admin:** http://localhost:5000/admin  

---

## 🏗️ Architecture Overview

```
+-------------------+    +--------------------+    +-------------------+
|   Web Browser     |    |   Desktop App      |    |   Mobile App      |
|  (React/HTML)     |    |    (PyQt5)         |    |   (PWA)           |
+---------+---------+    +---------+----------+    +---------+---------+
          |                        |                         |
          +------------------------+-------------------------+
                                   |
          +---------------------+--+--+---------------------+
          |                     |  |  |                     |
    +-----v-----+         +-----v--v-----+           +-----v-----+
    |   NGINX   |         |   Flask App  |           |   Auth    |
    | (Static)  |         |  (Python)    |           |  Service  |
    +-----------+         +-----+--------+           +-----------+
                                |
                    +-----------+-------+
                    |           |       |
            +-------v---+  +----v----+  +--v------+
            | Quantum   |  |  C++    |  | Redis   |
            | Engine    |  | Core    |  | Cache   |
            | (Python)  |  | (Eigen) |  |         |
            +-----------+  +---------+  +---------+
                                |
                        +-------v--------+
                        |  PostgreSQL    |
                        |   Database     |
                        +----------------+
```

**Data Flow:**
1. **Frontend** → Flask routes (`/api/*`)
2. **Flask** → C++ quantum core (via Python bindings)
3. **C++** → Mathematical computations (Eigen, OpenMP)
4. **Results** → PostgreSQL (encrypted) + Redis (cached)
5. **Response** → JSON API → Frontend

---

## 📋 Table of Contents

- [🚀 Quick Start](#-quick-start-just-want-to-run-it)
- [🛠️ Full Installation](#️-full-installation)
- [🔧 Development Setup](#-development-setup-want-to-contribute)
- [🏗️ Architecture](#️-architecture-overview)
- [📡 API Reference](#-api-reference)
- [🧪 Testing](#-testing)
- [🚀 Deployment](#-deployment)

---

## 🛠️ Full Installation

### System Requirements
- **OS:** Windows 10+, Ubuntu 20.04+, macOS 11+
- **Python:** 3.9+ (3.11 recommended)
- **Memory:** 4GB+ RAM (8GB+ for quantum simulations)
- **Storage:** 2GB free space

### Core Dependencies

**Database & Cache:**
```bash
# PostgreSQL (optional - SQLite used for development)
# Ubuntu/Debian:
sudo apt update && sudo apt install postgresql postgresql-contrib

# Windows (via Chocolatey):
choco install postgresql

# macOS (via Homebrew):
brew install postgresql

# Redis (optional - memory fallback available)
# Ubuntu/Debian:
sudo apt install redis-server

# Windows:
# Download from: https://github.com/tporadowski/redis/releases

# macOS:
brew install redis
```

**Python Environment:**
```bash
# Create isolated environment
python -m venv quantonium_env
source quantonium_env/bin/activate  # Linux/Mac
# quantonium_env\Scripts\activate    # Windows

# Install all dependencies (command-verified from requirements.txt)
pip install -r requirements.txt
```

**C++ Build Tools (for quantum core):**
```bash
# Ubuntu/Debian:
sudo apt install build-essential cmake libeigen3-dev

# Windows (Visual Studio 2019+):
# Install via Visual Studio Installer or Build Tools

# macOS:
xcode-select --install
brew install cmake eigen
```

### Quick Setup
```bash
# Automated development environment setup
python setup_local_env.py

# Verify everything is working
python scripts/verify_setup.py
```

---

## 🔧 Development Setup (Want to Contribute?)

### Advanced Environment Setup

**1. Fork & Clone:**
```bash
# Fork on GitHub first, then:
git clone https://github.com/YOUR_USERNAME/quantoniumos.git
cd quantoniumos

# Add upstream remote
git remote add upstream https://github.com/mandcony/quantoniumos.git
```

**2. Development Dependencies:**
```bash
# Install dev tools
pip install black isort flake8 mypy pytest-cov pre-commit mkdocs

# Setup pre-commit hooks
pre-commit install

# Install package in editable mode
pip install -e .
```

**3. Build C++ Quantum Core:**
```bash
# Build quantum computational engine
cd quantoniumos
.\build_cpp.ps1 -Debug  # Windows
./build_cpp.sh --debug  # Linux/Mac

# Verify C++ integration
python -c "from quantonium_engine import test_quantum_operations; test_quantum_operations()"
```

**4. Development Workflow:**
```bash
# Run tests before coding
pytest tests/ -v --cov=quantoniumos

# Start development server with auto-reload
python start_dev.py  # Includes hot reload, debug mode, test data
```

### Project Structure (for Contributors)
```
quantoniumos/
├── quantoniumos/              # Main application package
│   ├── __init__.py
│   ├── app.py                 # Flask application factory
│   ├── main.py               # Entry point
│   ├── models.py             # SQLAlchemy models
│   ├── security.py           # WAF, encryption, auth
│   ├── routes/               # API route blueprints
│   │   ├── api.py           # Core API endpoints
│   │   ├── auth.py          # Authentication routes
│   │   ├── quantum.py       # Quantum algorithm APIs
│   │   └── encrypt.py       # Encryption services
│   ├── quantum/              # Quantum computing core
│   │   ├── engine_adapter.py # Python-C++ bridge
│   │   ├── algorithms.py    # Quantum algorithms
│   │   └── resonance.py     # Proprietary resonance theory
│   ├── static/              # Frontend assets
│   │   ├── js/quantum-*.js  # Quantum UI components
│   │   ├── css/theme.css    # QuantoniumOS styling
│   │   └── html/            # Application pages
│   ├── templates/           # Jinja2 templates
│   └── utils/               # Utility modules
├── cpp_core/                 # C++ quantum engine
│   ├── CMakeLists.txt       # Build configuration
│   ├── src/                 # C++ source files
│   ├── include/             # Header files
│   └── bindings/            # Python bindings (pybind11)
├── tests/                   # Test suite
├── docs/                   # Documentation
├── scripts/               # Development scripts
├── .github/              # CI/CD workflows
├── requirements.txt     # Python dependencies
├── CMakeLists.txt      # Root build file
└── README.md
```

---

## 📡 API Reference

### Interactive Documentation
- **OpenAPI Spec:** `http://localhost:5000/docs` (Swagger UI)
- **ReDoc:** `http://localhost:5000/redoc` (Alternative UI)
- **JSON Schema:** `http://localhost:5000/openapi.json`

### Core Endpoints

**Authentication:**
```bash
# Get API key
curl -X POST http://localhost:5000/auth/api-key \
  -H "Content-Type: application/json" \
  -d '{"name": "my-app"}'

# Use API key
curl -H "Authorization: Bearer YOUR_API_KEY" \
  http://localhost:5000/api/quantum/entropy
```

**Quantum Operations:**
```bash
# Quantum encryption
curl -X POST http://localhost:5000/api/quantum/encrypt \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{"data": "secret message", "algorithm": "resonance"}'

# System status
curl http://localhost:5000/api/health
```

### Rate Limits
- **Unauthenticated:** 100 requests/hour
- **Authenticated:** 1000 requests/hour  
- **Enterprise:** 10,000 requests/hour

---

## 🧪 Testing

### Test Suites

**Quick Test:**
```bash
# Run core tests (< 30 seconds)
pytest tests/unit/ -x --tb=short

# Integration tests (< 2 minutes)  
pytest tests/integration/ --maxfail=3
```

**Full Test Suite:**
```bash
# All tests with coverage
pytest tests/ --cov=quantoniumos --cov-report=html --cov-report=term

# Performance benchmarks
pytest tests/performance/ --benchmark-only
```

---

## 🚀 Deployment

### Production Deployment

**Docker (Recommended):**
```bash
# Build production image
docker build -t quantoniumos:latest .

# Run with docker-compose
docker-compose up -d
```

**Manual Deployment:**
```bash
# Install production dependencies
pip install -r requirements.txt

# Run with Gunicorn
gunicorn --workers 4 --bind 0.0.0.0:8000 quantoniumos.app:create_app()
```

---

## 🔧 Troubleshooting

### Common Issues

**Database Connection Failed:**
```bash
# Check if setup was run
python setup_local_env.py

# Verify setup
python scripts/verify_setup.py
```

**Import Errors:**
```bash
# Verify Python path
python -c "import sys; print('\n'.join(sys.path))"

# Reinstall in development mode
pip install -e .
```

### Performance Optimization

**Database:**
```sql
-- Add indexes for common queries
CREATE INDEX idx_api_key_hash ON api_key(key_hash);
```

---

## 🤝 Contributing

### Development Workflow

1. **Fork & Branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Code & Test:**
   ```bash
   # Make changes
   vim quantoniumos/your_module.py
   
   # Test changes
   pytest tests/test_your_module.py -v
   
   # Format code
   black . && isort .
   ```

3. **Commit & Push:**
   ```bash
   git add .
   git commit -m "feat: add quantum entanglement API"
   git push origin feature/your-feature-name
   ```

---

## 📚 Additional Resources

### Documentation
- **Architecture Deep Dive:** `/docs/architecture.md`
- **API Specification:** `/docs/api-spec.yaml`
- **Security Guide:** `/docs/security.md`

### Community
- **GitHub Issues:** [Report bugs & feature requests](https://github.com/mandcony/quantoniumos/issues)
- **Discussions:** [Ask questions & share ideas](https://github.com/mandcony/quantoniumos/discussions)

---

**🎯 Goal: Make QuantoniumOS the most accessible quantum-classical computing platform for developers.**

*This guide is automatically updated with each release. Last updated: July 14, 2025*