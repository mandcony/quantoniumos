#!/bin/bash

# QuantoniumOS RFT Cleanup Plan
# Remove duplicate RFT implementations and consolidate to canonical source

echo "=== RFT CONSOLIDATION CLEANUP ==="
echo "Problem: 30+ duplicate RFT implementations confusing reviewers"
echo "Solution: Replace all with imports from canonical_true_rft.py"
echo ""

# Phase 1: Critical test files (keep these, fix imports)
echo "PHASE 1: Critical test files - fix imports to use canonical"
echo "- tests/test_rft_non_equivalence.py"
echo "- tests/test_rft_roundtrip.py"
echo "- publication_ready_validation.py"
echo "- comprehensive_non_equivalence_validation.py"

# Phase 2: Legacy test files (delete or merge)
echo ""
echo "PHASE 2: Legacy test files - DELETE duplicates"
echo "- rft_enhanced_crypto_suite.py"
echo "- rft_ablation_study.py"
echo "- simplified_rft_crypto_test.py"
echo "- enhanced_hash_test.py"
echo "- debug_sigma_stages.py"
echo "- comprehensive_rft_crypto_statistical_test.py"
echo "- high_performance_rft_crypto_test.py"
echo "- test_direct_hash.py"
echo "- corrected_rft_crypto_test.py"
echo "- mathematically_rigorous_rft.py"

# Phase 3: Hash/encryption modules (fix imports)
echo ""
echo "PHASE 3: Encryption modules - fix to use canonical"
echo "- encryption/improved_geometric_hash.py"

# Phase 4: Core modules (audit and fix)
echo ""
echo "PHASE 4: Core modules - consolidate with canonical"
echo "- core/true_rft.py"
echo "- core/python_bindings/engine_core.py"
echo "- core/high_performance_engine.py"

echo ""
echo "TARGET: Single RFT source truth prevents reviewer confusion"
echo "RESULT: canonical_true_rft.py + minimal_true_rft.py ONLY"
