# QuantoniumOS Development Setup Guide

## Quick Start

### Prerequisites
- Python 3.9+ (3.11 recommended)
- Git
- C++ compiler (GCC/Clang on Linux, MSVC on Windows)
- CMake
- PostgreSQL (for full functionality)

### Local Development Setup

1. **Clone and Install**
   ```bash
   git clone https://github.com/quantonium/quantonium-os.git
   cd quantonium-os
   pip install -e .
   ```

2. **Verify Installation**
   ```bash
   python scripts/verify_cli.py --verbose
   ```

3. **Run Tests**
   ```bash
   # Unit tests only (no server required)
   python -m pytest tests/ -v
   
   # Full integration tests (requires PostgreSQL)
   export DATABASE_URL="postgresql://user:pass@localhost:5432/quantonium_dev"
   python scripts/smoke_test.py --skip-auth
   ```

### CI/CD Development

#### Understanding the Pipeline

1. **Validation Phase**: Code quality, security scanning, CLI verification
2. **Testing Phase**: Multi-version testing (Python 3.9-3.11)
3. **Integration Phase**: Server startup and smoke tests
4. **Build Phase**: Package and Docker image creation

#### Running CI Tests Locally

```bash
# Validate (like CI Phase 1)
python scripts/verify_cli.py --verbose
bandit -r core/ || echo "Security issues found"

# Test (like CI Phase 2)
python -m pytest tests/ -v --cov=core

# Integration (like CI Phase 3) - requires PostgreSQL
export DATABASE_URL="postgresql://user:pass@localhost:5432/test_db"
gunicorn --bind 0.0.0.0:5000 main:app --daemon
python scripts/smoke_test.py --url http://localhost:5000 --skip-auth
```

### Troubleshooting Common Issues

#### C++ Extension Build Failures
```bash
# Install build dependencies
sudo apt-get install build-essential cmake libpython3-dev  # Ubuntu
brew install cmake                                          # macOS

# Manual build
python setup.py build_ext --inplace
```

#### Server Won't Start in CI
- Check the health endpoint: `curl http://localhost:5000/health`
- Verify database connection
- Check logs: `cat error.log access.log`

#### Test Connection Failures
- Ensure PostgreSQL is running
- Verify DATABASE_URL format
- Check firewall/port settings

### GitHub Security Scanning Setup

1. **Enable GitHub Advanced Security** (if available)
2. **Repository Settings** → Security & Analysis → Enable CodeQL
3. **Permissions**: Ensure workflows have `security-events: write`

### Development Workflow

1. **Feature Development**
   ```bash
   git checkout -b feature/your-feature
   # Make changes
   python scripts/verify_cli.py  # Quick validation
   python -m pytest tests/      # Run tests
   ```

2. **Pre-commit Checks**
   ```bash
   black core/ tests/           # Format code
   isort core/ tests/           # Sort imports
   bandit -r core/              # Security scan
   ```

3. **CI Pipeline Debugging**
   - Check workflow runs in GitHub Actions
   - Download artifacts for investigation
   - Compare local vs CI environment differences

### Architecture Notes

- **Core**: Patent-protected quantum algorithms (C++ with Python bindings)
- **API**: Flask-based REST API with authentication
- **Tests**: Isolated unit tests + integration tests with server
- **Security**: Multi-layer scanning (Bandit, Safety, Trivy)

### Performance Considerations

- C++ extensions provide ~100x speedup for core algorithms
- PostgreSQL required for production performance
- Redis caching for API rate limiting
- Docker deployment for consistent environments
