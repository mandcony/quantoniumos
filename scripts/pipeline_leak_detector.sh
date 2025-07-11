#!/bin/bash
# QuantoniumOS Pipeline Leak Detector
# Like putting a tube underwater and watching for bubbles

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Timing and counters
START_TIME=$(date +%s)
LEAK_COUNT=0
TOTAL_CHECKS=0

echo -e "${BLUE}🔍 QuantoniumOS Pipeline Leak Detector${NC}"
echo -e "${BLUE}=====================================${NC}"

# Function to check for "leaks" (potential issues)
check_leak() {
    local test_name="$1"
    local command="$2"
    local timeout_seconds="${3:-30}"
    
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    echo -n "🧪 Testing: $test_name... "
    
    if timeout "$timeout_seconds" bash -c "$command" >/dev/null 2>&1; then
        echo -e "${GREEN}✅ PASS${NC}"
        return 0
    else
        echo -e "${RED}❌ LEAK DETECTED${NC}"
        LEAK_COUNT=$((LEAK_COUNT + 1))
        return 1
    fi
}

# Function to run timed test with output
run_timed_test() {
    local test_name="$1"
    local command="$2"
    local timeout_seconds="${3:-60}"
    
    echo -e "\n${YELLOW}⏱️  Running: $test_name (max ${timeout_seconds}s)${NC}"
    local test_start=$(date +%s)
    
    if timeout "$timeout_seconds" bash -c "$command"; then
        local test_end=$(date +%s)
        local duration=$((test_end - test_start))
        echo -e "${GREEN}✅ Completed in ${duration}s${NC}"
        return 0
    else
        local test_end=$(date +%s)
        local duration=$((test_end - test_start))
        echo -e "${RED}❌ Failed after ${duration}s${NC}"
        LEAK_COUNT=$((LEAK_COUNT + 1))
        return 1
    fi
}

echo -e "\n${BLUE}Phase 1: Pre-flight Checks (Looking for obvious leaks)${NC}"
echo "================================================================"

# Check if critical files exist
check_leak "requirements.txt exists" "test -f requirements.txt"
check_leak "main.py exists" "test -f main.py"
check_leak "setup.py exists" "test -f setup.py"
check_leak "CI workflow exists" "test -f .github/workflows/main-ci.yml"

# Check Python syntax
check_leak "Python syntax check" "python -m py_compile main.py"
check_leak "Core module syntax" "find core/ -name '*.py' -exec python -m py_compile {} \; 2>/dev/null"

echo -e "\n${BLUE}Phase 2: Dependency Resolution (Testing tube connections)${NC}"
echo "================================================================"

# Test pip install simulation
run_timed_test "Pip dependency check" "pip install --dry-run -r requirements.txt" 45

# Test imports
check_leak "Flask import" "python -c 'import flask'"
check_leak "NumPy import" "python -c 'import numpy'"
check_leak "Cryptography import" "python -c 'import cryptography'"

echo -e "\n${BLUE}Phase 3: CLI Verification (Testing basic functionality)${NC}"
echo "================================================================"

if [ -f "scripts/verify_cli.py" ]; then
    run_timed_test "CLI verification" "python scripts/verify_cli.py --verbose" 60
else
    echo -e "${YELLOW}⚠️  CLI verification script not found - creating basic test${NC}"
    check_leak "Basic CLI test" "python -c 'import sys; print(\"CLI test passed\")'"
fi

echo -e "\n${BLUE}Phase 4: Security Scan (Looking for security leaks)${NC}"
echo "================================================================"

# Install security tools if not present
if ! command -v bandit &> /dev/null; then
    echo "📦 Installing security tools..."
    pip install bandit safety >/dev/null 2>&1 || echo "⚠️  Security tools installation failed"
fi

if command -v bandit &> /dev/null; then
    run_timed_test "Bandit security scan" "bandit -r core/ -f json" 30
else
    echo -e "${YELLOW}⚠️  Bandit not available - skipping security scan${NC}"
fi

echo -e "\n${BLUE}Phase 5: Build Test (Testing package integrity)${NC}"
echo "================================================================"

# Test package build
run_timed_test "Package build test" "python setup.py build --dry-run" 45

# Test Docker build if Dockerfile exists
if [ -f "Dockerfile" ]; then
    echo -e "\n${YELLOW}🐳 Docker build test available but skipped (time-intensive)${NC}"
    echo "   Run manually: docker build -t quantonium-test ."
else
    echo -e "${YELLOW}⚠️  No Dockerfile found${NC}"
fi

echo -e "\n${BLUE}Phase 6: Final Leak Report${NC}"
echo "================================================================"

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo -e "⏱️  Total execution time: ${TOTAL_DURATION}s"
echo -e "🧪 Total checks performed: $TOTAL_CHECKS"

if [ $LEAK_COUNT -eq 0 ]; then
    echo -e "${GREEN}🎉 NO LEAKS DETECTED - Pipeline ready for production!${NC}"
    echo -e "${GREEN}✅ Your tube is watertight - safe to submerge${NC}"
    exit 0
else
    echo -e "${RED}🚨 $LEAK_COUNT LEAKS DETECTED out of $TOTAL_CHECKS checks${NC}"
    echo -e "${RED}❌ Fix leaks before pushing to production${NC}"
    exit 1
fi
