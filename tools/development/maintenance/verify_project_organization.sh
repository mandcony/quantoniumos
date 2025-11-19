#!/bin/bash
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier
# Project Organization Verification Script

set -e

# Normalize to repo root so the script works from any location
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
cd "$REPO_ROOT"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  QuantoniumOS Project Organization Verification            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
PASS=0
FAIL=0
WARN=0

# Test function
test_check() {
    local test_name="$1"
    local test_cmd="$2"
    
    if eval "$test_cmd" &>/dev/null; then
        echo -e "${GREEN}✅ PASS${NC}: $test_name"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}❌ FAIL${NC}: $test_name"
        FAIL=$((FAIL + 1))
    fi
}

test_warn() {
    local test_name="$1"
    local test_cmd="$2"
    
    if eval "$test_cmd" &>/dev/null; then
        echo -e "${GREEN}✅ PASS${NC}: $test_name"
        PASS=$((PASS + 1))
    else
        echo -e "${YELLOW}⚠️  WARN${NC}: $test_name"
        WARN=$((WARN + 1))
    fi
}

echo "📁 1. DIRECTORY STRUCTURE CHECKS"
echo "─────────────────────────────────────────────────────────────"
test_check "algorithms/ directory exists" "[ -d algorithms ]"
test_check "core/ directory exists" "[ -d core ]"
test_check "tests/ directory exists" "[ -d tests ]"
test_check "tools/ directory exists" "[ -d tools ]"
test_check "docs/ directory exists" "[ -d docs ]"
test_check "data/ directory exists" "[ -d data ]"
echo ""

echo "📄 2. CORE FILE EXISTENCE CHECKS"
echo "─────────────────────────────────────────────────────────────"
test_check "README.md exists" "[ -f README.md ]"
test_check "LICENSE.md exists" "[ -f LICENSE.md ]"
test_check "LICENSE-CLAIMS-NC.md exists" "[ -f LICENSE-CLAIMS-NC.md ]"
test_check "PATENT_NOTICE.md exists" "[ -f PATENT_NOTICE.md ]"
test_check "requirements.txt exists" "[ -f requirements.txt ]"
test_check "validate_all.sh exists" "[ -f validate_all.sh ]"
test_check "validate_all.sh is executable" "[ -x validate_all.sh ]"
echo ""

echo "🔬 3. ALGORITHM COMPONENT CHECKS"
echo "─────────────────────────────────────────────────────────────"
test_check "algorithms/rft/core/canonical_true_rft.py exists" "[ -f algorithms/rft/core/canonical_true_rft.py ]"
test_check "algorithms/rft/core/enhanced_rft_crypto_v2.py exists" "[ -f algorithms/rft/core/enhanced_rft_crypto_v2.py ]"
test_check "algorithms/compression/hybrid/ exists" "[ -d algorithms/compression/hybrid ]"
test_check "algorithms/compression/vertex/ exists" "[ -d algorithms/compression/vertex ]"
test_check "algorithms/crypto/crypto_benchmarks/rft_sis/ exists" "[ -d algorithms/crypto/crypto_benchmarks/rft_sis ]"
echo ""

echo "🧪 4. TEST INFRASTRUCTURE CHECKS"
echo "─────────────────────────────────────────────────────────────"
test_check "tests/validation/ exists" "[ -d tests/validation ]"
test_check "tests/benchmarks/ exists" "[ -d tests/benchmarks ]"
test_check "tests/proofs/ exists" "[ -d tests/proofs ]"
test_check "tests/rft/ exists" "[ -d tests/rft ]"
test_check "tests/crypto/ exists" "[ -d tests/crypto ]"
test_check "pytest.ini exists" "[ -f pytest.ini ]"
echo ""

echo "🔧 5. HARDWARE FILES CHECKS"
echo "─────────────────────────────────────────────────────────────"
test_check "hardware/quantoniumos_unified_engines.sv exists" "[ -f hardware/quantoniumos_unified_engines.sv ]"
test_check "hardware/quantoniumos_engines_README.md exists" "[ -f hardware/quantoniumos_engines_README.md ]"
test_check "hardware/CRITICAL_FIXES_REPORT.md exists" "[ -f hardware/CRITICAL_FIXES_REPORT.md ]"
echo ""

echo "📚 6. DOCUMENTATION CHECKS"
echo "─────────────────────────────────────────────────────────────"
test_check "docs/COMPLETE_DEVELOPER_MANUAL.md exists" "[ -f docs/COMPLETE_DEVELOPER_MANUAL.md ]"
test_check "docs/QUICK_START.md exists" "[ -f docs/QUICK_START.md ]"
test_check "docs/TECHNICAL_SUMMARY.md exists" "[ -f docs/TECHNICAL_SUMMARY.md ]"
test_check "docs/RFT_THEOREMS.md exists" "[ -f docs/RFT_THEOREMS.md ]"
test_check "DOCUMENTATION_AUDIT.md exists" "[ -f DOCUMENTATION_AUDIT.md ]"
echo ""

echo "🛠️  7. UTILITY SCRIPT CHECKS"
echo "─────────────────────────────────────────────────────────────"
test_check "cleanup_docs.sh exists" "[ -f cleanup_docs.sh ]"
test_check "cleanup_docs.sh is executable" "[ -x cleanup_docs.sh ]"
test_check "hardware/verify_fixes.sh exists" "[ -f hardware/verify_fixes.sh ]"
test_check "hardware/verify_fixes.sh is executable" "[ -x hardware/verify_fixes.sh ]"
test_check "tools/development/maintenance/generate_project_map.py exists" "[ -f tools/development/maintenance/generate_project_map.py ]"
test_check "PROJECT_STRUCTURE.json exists" "[ -f PROJECT_STRUCTURE.json ]"
echo ""

echo "📦 8. PYTHON IMPORT CHECKS"
echo "─────────────────────────────────────────────────────────────"
test_warn "algorithms module imports" "python3 -c 'import algorithms' 2>&1 | grep -v UserWarning | grep -v DeprecationWarning"
test_warn "core module imports" "python3 -c 'import core' 2>&1 | grep -v UserWarning | grep -v DeprecationWarning"
test_warn "tests module imports" "python3 -c 'import tests' 2>&1 | grep -v UserWarning | grep -v DeprecationWarning"
echo ""

echo "⚖️  9. LICENSE HEADER SAMPLING"
echo "─────────────────────────────────────────────────────────────"
if grep -q "SPDX-License-Identifier" algorithms/rft/core/canonical_true_rft.py 2>/dev/null; then
    echo -e "${GREEN}✅ PASS${NC}: algorithms/rft/core/canonical_true_rft.py has license header"
    PASS=$((PASS + 1))
else
    echo -e "${YELLOW}⚠️  WARN${NC}: algorithms/rft/core/canonical_true_rft.py missing license header"
    WARN=$((WARN + 1))
fi

if grep -q "SPDX-License-Identifier" hardware/quantoniumos_unified_engines.sv 2>/dev/null; then
    echo -e "${GREEN}✅ PASS${NC}: hardware/quantoniumos_unified_engines.sv has license header"
    PASS=$((PASS + 1))
else
    echo -e "${YELLOW}⚠️  WARN${NC}: quantoniumos_unified_engines.sv missing license header"
    WARN=$((WARN + 1))
fi
echo ""

echo "🔍 10. PROJECT STRUCTURE JSON VALIDATION"
echo "─────────────────────────────────────────────────────────────"
if [ -f PROJECT_STRUCTURE.json ]; then
    if python3 -m json.tool PROJECT_STRUCTURE.json > /dev/null 2>&1; then
        echo -e "${GREEN}✅ PASS${NC}: PROJECT_STRUCTURE.json is valid JSON"
        PASS=$((PASS + 1))
        
        # Extract stats from JSON
        TOTAL_FILES=$(python3 -c "import json; data=json.load(open('PROJECT_STRUCTURE.json')); print(data['summary']['total_files'])" 2>/dev/null || echo "N/A")
        PYTHON_MODULES=$(python3 -c "import json; data=json.load(open('PROJECT_STRUCTURE.json')); print(data['summary']['python_modules'])" 2>/dev/null || echo "N/A")
        WITH_LICENSE=$(python3 -c "import json; data=json.load(open('PROJECT_STRUCTURE.json')); print(data['summary']['with_license_headers'])" 2>/dev/null || echo "N/A")
        
        echo -e "   Total files: ${YELLOW}${TOTAL_FILES}${NC}"
        echo -e "   Python modules: ${YELLOW}${PYTHON_MODULES}${NC}"
        echo -e "   With license headers: ${YELLOW}${WITH_LICENSE}${NC}"
    else
        echo -e "${RED}❌ FAIL${NC}: PROJECT_STRUCTURE.json is invalid JSON"
        FAIL=$((FAIL + 1))
    fi
else
    echo -e "${RED}❌ FAIL${NC}: PROJECT_STRUCTURE.json not found"
    FAIL=$((FAIL + 1))
fi
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "                     VERIFICATION SUMMARY                       "
echo "═══════════════════════════════════════════════════════════════"
echo -e "✅ PASS: ${GREEN}${PASS}${NC}"
echo -e "⚠️  WARN: ${YELLOW}${WARN}${NC}"
echo -e "❌ FAIL: ${RED}${FAIL}${NC}"
echo ""

if [ "$FAIL" -eq 0 ]; then
    if [ "$WARN" -eq 0 ]; then
        echo -e "${GREEN}🎉 ALL CHECKS PASSED!${NC} Project organization is complete."
    else
        echo -e "${YELLOW}✅ CHECKS PASSED WITH WARNINGS${NC}"
        echo "   Warnings are non-critical issues (imports, license headers)."
        echo "   Review warnings above for details."
    fi
    exit 0
else
    echo -e "${RED}❌ VERIFICATION FAILED${NC}"
    echo "   Please fix the failed checks above before proceeding."
    exit 1
fi
