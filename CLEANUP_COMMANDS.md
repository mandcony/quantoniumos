# QuantoniumOS - Quick Cleanup Commands

**Generated:** December 3, 2025  
**Purpose:** Copy-paste commands for repository cleanup

---

## 🧹 Phase 1: Safe Cleanup (Run Immediately)

### Remove Python Cache
```bash
cd /workspaces/quantoniumos

# Remove all __pycache__ directories
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# Remove pytest cache
rm -rf .pytest_cache

# Remove hypothesis cache
rm -rf .hypothesis

# Remove egg-info
rm -rf quantoniumos.egg-info

# Remove Python compiled files
find . -name "*.pyc" -delete
find . -name "*.pyo" -delete

echo "✅ Python cache cleaned"
```

### Remove Temporary Files
```bash
# Remove editor temporary files
find . -name "*~" -delete
find . -name "*.swp" -delete
find . -name ".DS_Store" -delete

# Remove LaTeX temporary files
cd papers
rm -f *.aux *.log *.out *.toc *.bbl *.blg *.synctex.gz
cd ..

echo "✅ Temporary files cleaned"
```

### Clean Hardware Artifacts
```bash
cd hardware

# Remove simulation executables
rm -f sim_rft sim_unified rft_tb

# Remove waveform dumps
rm -f *.vcd

# Keep directory structure but clean logs
find test_logs -name "*.log" -delete 2>/dev/null

cd ..

echo "✅ Hardware artifacts cleaned"
```

### Disk Space Report
```bash
# Show space saved
du -sh . 2>/dev/null
echo ""
echo "Cache directories removed: $(find . -type d -name "__pycache__" 2>/dev/null | wc -l)"
```

---

## 🔍 Phase 2: Analysis Commands

### Find Duplicates
```bash
# Find duplicate geometric hashing files
find . -name "*geometric*hash*.py" -type f | grep -E "(core|quantum)"

# Find duplicate quantum files
find . -name "*quantum*" -type f | head -20

# Find legacy files
find . -name "*legacy*.py" -type f
```

### Count Files by Type
```bash
echo "Python files: $(find . -name "*.py" -type f | wc -l)"
echo "Markdown files: $(find . -name "*.md" -type f | wc -l)"
echo "C/C++ files: $(find . \( -name "*.c" -o -name "*.cpp" -o -name "*.h" -o -name "*.hpp" \) -type f | wc -l)"
echo "Test files: $(find tests -name "*.py" -type f | wc -l)"
echo "Benchmark files: $(find benchmarks -name "*.py" -type f | wc -l)"
```

### Directory Sizes
```bash
du -sh algorithms benchmarks quantoniumos src tests experiments hardware docs papers 2>/dev/null | sort -hr
```

---

## 🧪 Phase 3: Testing Commands

### Run All Tests
```bash
# Activate environment
source .venv/bin/activate

# Run all tests with verbose output
pytest tests/ -v --tb=short 2>&1 | tee test_results_full.log

# Run specific test categories
pytest tests/rft/ -v --tb=short 2>&1 | tee test_results_rft.log
pytest tests/validation/ -v --tb=short 2>&1 | tee test_results_validation.log
pytest tests/benchmarks/ -v --tb=short 2>&1 | tee test_results_benchmarks.log
pytest tests/crypto/ -v --tb=short 2>&1 | tee test_results_crypto.log

echo "✅ Test results saved to test_results_*.log"
```

### Run Benchmarks
```bash
# Run all benchmark classes
python benchmarks/run_all_benchmarks.py

# Run with variants (comprehensive)
python benchmarks/run_all_benchmarks.py --variants

# Run specific classes
python benchmarks/run_all_benchmarks.py A B C

# Save results to JSON
python benchmarks/run_all_benchmarks.py --variants --json results/benchmark_$(date +%Y%m%d).json
```

### Validate System
```bash
# Quick validation
python validate_system.py

# Full validation
bash scripts/validate_all.sh

# Hardware validation
cd hardware && bash verify_fixes.sh && cd ..

# Verify specific claims
python scripts/verify_scaling_laws.py
python scripts/verify_ascii_bottleneck.py
python scripts/verify_variant_claims.py
```

---

## 📊 Phase 4: Analysis & Reporting

### Generate Coverage Report
```bash
# Install coverage tools
pip install pytest-cov

# Generate HTML coverage report
pytest tests/ --cov=algorithms --cov=quantoniumos --cov-report=html --cov-report=term

# Open report
# Linux: xdg-open htmlcov/index.html
# macOS: open htmlcov/index.html
# Windows: start htmlcov/index.html

echo "✅ Coverage report generated in htmlcov/"
```

### Count Test Status
```bash
cd tests

echo "=== Test File Status ==="
echo "Total test files: $(find . -name "test_*.py" | wc -l)"
echo ""

# Find tests with PASS/FAIL markers
grep -r "✅ PASSED" ../TEST_RESULTS.md | wc -l | xargs echo "Passing tests:"
grep -r "⏳ PENDING" ../TEST_RESULTS.md | wc -l | xargs echo "Pending tests:"
grep -r "❌ FAILED" ../TEST_RESULTS.md | wc -l | xargs echo "Failed tests:"

cd ..
```

### Benchmark Analysis
```bash
# Find all benchmark result files
find results -name "*.json" -type f

# Latest benchmark
ls -lt results/*.json | head -1

# Compare benchmarks
python tools/benchmarking/compare_results.py \
    results/benchmark_old.json \
    results/benchmark_new.json
```

---

## 🔄 Phase 5: Code Consolidation

### Consolidate Geometric Hashing
```bash
# Step 1: Compare implementations
diff algorithms/rft/core/geometric_hashing.py \
     algorithms/rft/quantum/geometric_hashing.py

# Step 2: Create consolidated version (manual merge required)
mkdir -p algorithms/rft/crypto
# Then manually merge files

# Step 3: Find all imports
echo "=== Finding import statements ==="
grep -r "from algorithms.rft.core.geometric" . --include="*.py"
grep -r "from algorithms.rft.quantum.geometric" . --include="*.py"

# Step 4: Update imports (after manual review)
# find . -name "*.py" -exec sed -i 's/OLD_IMPORT/NEW_IMPORT/g' {} +
```

### Consolidate Quantum Code
```bash
# Move quantum kernel to quantum directory
git mv algorithms/rft/core/quantum_kernel_implementation.py \
       algorithms/rft/quantum/symbolic_compression.py

# Update imports (requires manual review of output first)
echo "=== Files importing quantum_kernel_implementation ==="
grep -r "quantum_kernel_implementation" . --include="*.py" | cut -d: -f1 | sort -u
```

---

## 🏗️ Phase 6: Build Commands

### Rebuild Native Modules
```bash
cd src/rftmw_native

# Clean previous build
rm -rf build
mkdir build
cd build

# Configure with optimizations
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-O3 -march=native" \
    -DRFTMW_ENABLE_ASM=ON \
    -DRFTMW_ENABLE_AVX512=ON

# Build
make -j$(nproc)

# Install to venv
cp rftmw_native.cpython-*-linux-gnu.so \
   ../../../.venv/lib/python*/site-packages/

cd ../../..

# Test
python -c "import rftmw_native; print('✅ Native module loaded')"
```

### Rebuild Hardware
```bash
cd hardware

# Simulate RFT core
iverilog -o sim_rft tb_rft_middleware.sv rft_middleware_engine.sv
./sim_rft

# Simulate unified engine
iverilog -o sim_unified tb_quantoniumos_unified.sv quantoniumos_unified_engines.sv
./sim_unified

# Lint with Verilator
verilator --lint-only quantoniumos_unified_engines.sv 2>&1 | tee verilator_lint.log

cd ..
```

---

## 📝 Phase 7: Documentation Commands

### Generate API Documentation
```bash
# Install Sphinx
pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints

# Initialize Sphinx (if not done)
cd docs
# sphinx-quickstart api_reference  # Skip if already exists

# Generate API docs
sphinx-apidoc -f -o api_reference/source ../algorithms ../quantoniumos

# Build HTML docs
cd api_reference
make clean
make html

# Open docs
# xdg-open build/html/index.html  # Linux
# open build/html/index.html      # macOS

cd ../..
```

### Update Documentation Index
```bash
# List all markdown files
find . -name "*.md" -type f | sort > docs/markdown_inventory.txt

# Find broken links
grep -r "\[.*\](.*\.md)" docs --include="*.md" | \
    grep -v "^docs/.*:.*docs/" | \
    head -20

# Check for TODOs
find docs -name "*.md" -exec grep -l "TODO\|FIXME\|XXX" {} \;
```

---

## 🎨 Phase 8: Application Testing

### Test QuantSoundDesign
```bash
# Launch application
python src/apps/quantsounddesign/gui.py &

# Or via desktop launcher
python quantonium_os_src/frontend/quantonium_desktop.py
# Click "QuantSoundDesign"
```

### Test Other Applications
```bash
# RFT Visualizer
python src/apps/rft_visualizer.py &

# Quantum Crypto
python src/apps/quantum_crypto.py &

# Quantum Simulator
python src/apps/quantum_simulator.py &

# Q-Vault
python src/apps/q_vault.py &
```

### Test Mobile App
```bash
cd quantonium-mobile

# Install dependencies
npm install

# Run on iOS simulator
npm run ios

# Run on Android emulator
npm run android

cd ..
```

---

## 🔐 Phase 9: Security & Compliance

### Run Security Audit
```bash
# Install tools
pip install bandit safety pip-audit

# Run Bandit (Python security linter)
bandit -r algorithms/ src/ quantoniumos/ -f json -o security_report.json

# Check dependencies for vulnerabilities
pip-audit

# Run safety check
safety check --json > safety_report.json

echo "✅ Security reports generated"
```

### Verify License Headers
```bash
# Check SPDX headers
echo "=== Checking license headers ==="
python tools/spdx_inject.py --check

# Count files with/without headers
total=$(find . -name "*.py" -type f | wc -l)
with_header=$(find . -name "*.py" -type f -exec head -5 {} \; | grep -c "SPDX-License-Identifier")
echo "Files with headers: $with_header / $total"

# Verify claims files
echo ""
echo "=== Verifying claims-practicing files ==="
while read file; do
    if [ -f "$file" ]; then
        if head -10 "$file" | grep -q "LicenseRef-QuantoniumOS-Claims-NC"; then
            echo "✅ $file"
        else
            echo "❌ $file (missing claims license)"
        fi
    else
        echo "⚠️ $file (file not found)"
    fi
done < CLAIMS_PRACTICING_FILES.txt
```

---

## 📦 Phase 10: Release Preparation

### Create Release Package
```bash
# Run organize script
bash organize-release.sh

# Check output
ls -lh release/quantoniumos-complete-*.{tar.gz,zip}

# Verify checksums
cd release
sha256sum quantoniumos-complete-*.tar.gz
sha256sum quantoniumos-complete-*.zip
cd ..
```

### Tag Release
```bash
# Create version tag
VERSION="v0.1.0"
git tag -a $VERSION -m "QuantoniumOS $VERSION - Initial release"

# Push tag
git push origin $VERSION

# Create GitHub release (requires gh CLI)
gh release create $VERSION \
    release/quantoniumos-complete-*.tar.gz \
    release/quantoniumos-complete-*.zip \
    --title "QuantoniumOS $VERSION" \
    --notes "See CHANGELOG.md for details"
```

---

## 🎯 Complete Workflow Example

### Full Cleanup + Test + Benchmark
```bash
#!/bin/bash
set -e  # Exit on error

echo "=== QuantoniumOS Complete Workflow ==="
echo ""

# 1. Cleanup
echo "[1/5] Cleaning cache and temporary files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
rm -rf .pytest_cache .hypothesis
echo "✅ Cleanup complete"
echo ""

# 2. Activate environment
echo "[2/5] Activating environment..."
source .venv/bin/activate
echo "✅ Environment activated"
echo ""

# 3. Run tests
echo "[3/5] Running test suite..."
pytest tests/ -v --tb=short 2>&1 | tee test_results.log
echo "✅ Tests complete"
echo ""

# 4. Run benchmarks
echo "[4/5] Running benchmarks..."
python benchmarks/run_all_benchmarks.py --json results/benchmark_$(date +%Y%m%d).json
echo "✅ Benchmarks complete"
echo ""

# 5. Generate report
echo "[5/5] Generating summary..."
echo "==================="
echo "Test Results:"
grep -E "(PASSED|FAILED)" test_results.log | tail -5
echo ""
echo "Benchmark Results:"
ls -lh results/benchmark_*.json | tail -1
echo "==================="
echo ""
echo "✅ Complete workflow finished!"
```

Save as `scripts/complete_workflow.sh` and run:
```bash
chmod +x scripts/complete_workflow.sh
./scripts/complete_workflow.sh
```

---

## 🚨 Emergency Commands

### Rollback Changes
```bash
# View recent commits
git log --oneline -10

# Rollback to specific commit
git reset --hard <commit-hash>

# Or rollback last commit
git reset --hard HEAD~1

# Force push (USE WITH CAUTION)
# git push -f origin main
```

### Restore Deleted Files
```bash
# Find deleted file
git log --all --full-history -- path/to/file

# Restore file
git checkout <commit-hash> -- path/to/file
```

### Check Disk Space
```bash
# Current directory size
du -sh .

# Largest directories
du -h . | sort -hr | head -20

# Largest files
find . -type f -exec du -h {} + | sort -hr | head -20
```

---

## 📞 Quick Reference

### Common Paths
```bash
ALGORITHMS="algorithms/rft"
BENCHMARKS="benchmarks"
TESTS="tests"
APPS="src/apps"
DOCS="docs"
HARDWARE="hardware"
```

### Quick Tests
```bash
# Test RFT core
python -c "from algorithms.rft.core.canonical_true_rft import CanonicalTrueRFT; print('✅ RFT works')"

# Test native module
python -c "import rftmw_native; print('✅ Native module works')"

# Test variant manifest
python -c "from algorithms.rft.variants.manifest import iter_variants; print(f'✅ {len(list(iter_variants()))} variants loaded')"
```

### Quick Info
```bash
# System info
python --version
pytest --version
git branch --show-current
git log --oneline -1

# Package info
pip show quantoniumos
pip list | grep -E "(numpy|scipy|torch|pytest)"
```

---

**End of Quick Cleanup Commands**

*Copy and paste these commands as needed for repository maintenance.*  
*Always review output and test on a branch before applying to main.*
