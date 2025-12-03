# QuantoniumOS - Cleanup & Organization Action Plan

**Generated:** December 3, 2025  
**Based on:** SYSTEM_ARCHITECTURE_MAP.md  
**Purpose:** Actionable cleanup tasks for repository organization

---

## 🎯 Executive Summary

After scanning 7,585 Python files across the entire QuantoniumOS codebase, this document provides specific, actionable steps to:
1. Remove deprecated files
2. Consolidate duplicate code
3. Clean build artifacts
4. Optimize directory structure
5. Improve test coverage

**Estimated Cleanup Impact:**
- Remove ~500MB of build artifacts
- Consolidate 8-10 duplicate implementations
- Archive 20+ deprecated files
- Reduce directory complexity by ~15%

---

## ✅ Phase 1: Safe Deletions (No Code Changes)

### 1.1 Cache Directories (Safe to Delete - 100%)

```bash
# Remove all Python cache directories (36 found)
find /workspaces/quantoniumos -type d -name "__pycache__" -print0 | xargs -0 rm -rf

# Remove pytest cache
rm -rf /workspaces/quantoniumos/.pytest_cache

# Remove hypothesis cache
rm -rf /workspaces/quantoniumos/.hypothesis

# Remove egg-info
rm -rf /workspaces/quantoniumos/quantoniumos.egg-info
```

**Impact:** Recovers ~50-100MB, no functional changes

### 1.2 Build Artifacts (Regenerable)

```bash
# C++ build artifacts (can rebuild with cmake)
rm -rf /workspaces/quantoniumos/src/rftmw_native/build/_deps/
rm -rf /workspaces/quantoniumos/src/rftmw_native/build/CMakeFiles/

# Note: Keep build/rftmw_native.*.so (compiled module)

# Hardware simulation artifacts
cd /workspaces/quantoniumos/hardware
rm -f sim_rft sim_unified rft_tb
rm -f *.vcd
rm -rf test_logs/*.log  # Keep directory structure
```

**Impact:** Recovers ~200-300MB, can regenerate with build scripts

### 1.3 Temporary Files

```bash
# Python temporary files
find /workspaces/quantoniumos -name "*.pyc" -delete
find /workspaces/quantoniumos -name "*.pyo" -delete
find /workspaces/quantoniumos -name "*~" -delete
find /workspaces/quantoniumos -name "*.swp" -delete

# LaTeX temporary files
cd /workspaces/quantoniumos/papers
rm -f *.aux *.log *.out *.toc *.bbl *.blg *.synctex.gz
```

**Impact:** Recovers ~10-20MB

---

## 🔄 Phase 2: Code Consolidation (Requires Testing)

### 2.1 Consolidate Geometric Hashing Implementations

**Issue:** 4 duplicate implementations found:
1. `algorithms/rft/core/geometric_waveform_hash.py`
2. `algorithms/rft/core/geometric_hashing.py`
3. `algorithms/rft/quantum/geometric_waveform_hash.py`
4. `algorithms/rft/quantum/geometric_hashing.py`

**Action Plan:**

```bash
# Step 1: Analyze differences
diff algorithms/rft/core/geometric_hashing.py algorithms/rft/quantum/geometric_hashing.py

# Step 2: Create consolidated implementation
mkdir -p algorithms/rft/crypto/
# Merge best features into algorithms/rft/crypto/geometric_hashing.py

# Step 3: Update imports across codebase
grep -r "from algorithms.rft.core.geometric" . --include="*.py"
grep -r "from algorithms.rft.quantum.geometric" . --include="*.py"

# Step 4: Deprecate old files (add warnings)
# Add: warnings.warn("Deprecated: use algorithms.rft.crypto.geometric_hashing", DeprecationWarning)

# Step 5: After one release cycle, remove deprecated files
```

**Testing Required:**
- Run all crypto tests: `pytest tests/crypto/ -v`
- Run RFT tests: `pytest tests/rft/ -v`
- Check benchmarks: `python benchmarks/class_d_crypto.py`

### 2.2 Consolidate Quantum Implementations

**Issue:** Quantum code scattered across:
1. `algorithms/rft/quantum/` - Core quantum algorithms
2. `algorithms/rft/core/quantum_kernel_implementation.py` - QSC

**Action Plan:**

```bash
# Step 1: Move QSC to quantum directory
mv algorithms/rft/core/quantum_kernel_implementation.py \
   algorithms/rft/quantum/symbolic_compression.py

   # Step 2: Update __init__.py
   echo "from .symbolic_compression import QuantumSymbolicCompression" >> \
      algorithms/rft/quantum/__init__.py

      # Step 3: Update imports
      find . -type f -name "*.py" -exec sed -i \
         's/from algorithms.rft.core.quantum_kernel_implementation/from algorithms.rft.quantum.symbolic_compression/g' {} +

         # Step 4: Test
         pytest tests/benchmarks/quantum_compression_benchmark.py -v
         python benchmarks/class_a_quantum_simulation.py
         ```

         **Testing Required:**
         - Quantum compression tests
         - Class A benchmarks
         - QSC application

         ### 2.3 Migrate Apps to Organized Structure

         **Issue:** Apps exist in both flat and organized structures:
         - Flat: `src/apps/*.py` (22 files)
         - Organized: `quantonium_os_src/apps/` (7 directories)

         **Action Plan:**

         ```bash
         # Step 1: Identify apps not yet migrated
         cd /workspaces/quantoniumos
         ls src/apps/*.py | grep -v "quantsounddesign"

         # Apps needing migration:
         # - enhanced_rft_crypto.py
         # - baremetal_engine_3d.py
         # - launch_*.py (launcher scripts)

         # Step 2: Create organized structure for each
         # Example: enhanced_rft_crypto.py
         mkdir -p quantonium_os_src/apps/enhanced_crypto
         mv src/apps/enhanced_rft_crypto.py quantonium_os_src/apps/enhanced_crypto/main.py
         touch quantonium_os_src/apps/enhanced_crypto/__init__.py

         # Step 3: Update launcher scripts
         # Keep launchers in src/apps/ but point to organized structure

         # Step 4: Test each migrated app
         python quantonium_os_src/apps/enhanced_crypto/main.py
         ```

         **Testing Required:**
         - Test each application individually
         - Verify desktop launcher still works
         - Check all import paths

         ---

         ## 📦 Phase 3: Archive Deprecated Code

         ### 3.1 Create Archive Directory

         ```bash
         mkdir -p /workspaces/quantoniumos/docs/archive/deprecated_code
         mkdir -p /workspaces/quantoniumos/docs/archive/completed_experiments
         ```

         ### 3.2 Archive Legacy Hybrid Codec

         **Files to Archive:**
         - Any `legacy_hybrid_codec.py` files
         - Old hybrid implementations replaced by H3/FH5

         ```bash
         # Find legacy codec files
         find . -name "*legacy*hybrid*.py" -type f

         # Move to archive with git history
         git mv <file> docs/archive/deprecated_code/
         echo "Replaced by H3 Cascade (hybrid_mca_fixes.py)" > \
            docs/archive/deprecated_code/README.md
            ```

            ### 3.3 Archive Completed Experiments

            **Candidates:**
            - Experiments with published results
            - One-off analysis scripts
            - Superseded implementations

            ```bash
            # Example: Archive old ASCII wall experiments
            cd experiments/ascii_wall
            ls *.py | grep -v "paper\|final"  # Find non-final versions

            # Move completed experiments
            git mv <old_file>.py ../../docs/archive/completed_experiments/
            ```

            ---

            ## 🧪 Phase 4: Test Coverage Improvement

            ### 4.1 Run All Pending Tests

            ```bash
            # Activate environment
            source .venv/bin/activate

            # Run all RFT tests
            pytest tests/rft/ -v --tb=short 2>&1 | tee test_results_rft.log

            # Run validation tests
            pytest tests/validation/ -v --tb=short 2>&1 | tee test_results_validation.log

            # Run benchmark tests
            pytest tests/benchmarks/ -v --tb=short 2>&1 | tee test_results_benchmarks.log

            # Run crypto tests
            pytest tests/crypto/ -v --tb=short 2>&1 | tee test_results_crypto.log

            # Generate coverage report
            pytest tests/ --cov=algorithms --cov=quantoniumos --cov-report=html
            ```

            ### 4.2 Fix Failing Tests

            **Known Issues:**
            1. `test_rans_roundtrip.py` - Skipped (roundtrip issue)
            2. Verilator lint errors in hardware/
            3. Import path issues in some validation tests

            **Action Plan:**

            ```bash
            # Fix import paths
            find tests/ -name "*.py" -exec grep -l "from core import" {} \; | \
               xargs sed -i 's/from core import/from algorithms.rft.core import/g'

               # Fix hardware lint errors
               cd hardware
               # Edit quantoniumos_unified_engines.sv
               # Fix BLKANDNBLK errors (mixing = and <=)
               # Fix WIDTHTRUNC errors (bit width mismatches)
               ```

               ### 4.3 Add Missing Test Coverage

               **Areas Needing Tests:**
               1. All 14 variants (only STANDARD tested comprehensively)
               2. Hybrid architectures H2 and H10 (marked as buggy)
               3. Hardware/software parity tests
               4. Performance regression tests

               ```bash
               # Create test files for missing coverage
               cat > tests/rft/test_all_variants_comprehensive.py << 'EOF'
               """Comprehensive tests for all 14 Φ-RFT variants."""
               import pytest
               from algorithms.rft.variants.manifest import iter_variants

               @pytest.mark.parametrize("variant", [entry.code for entry in iter_variants()])
               def test_variant_unitarity(variant):
                   """Test unitarity for each variant."""
                       # Implementation
                           pass
                           EOF
                           ```

                           ---

                           ## 🏗️ Phase 5: Documentation Updates

                           ### 5.1 Update Root Documentation

                           ```bash
                           # Update README.md with latest benchmarks
                           # Update QUICK_REFERENCE.md with new commands
                           # Update GETTING_STARTED.md with simplified setup

                           # Ensure cross-references are valid
                           find . -name "*.md" -exec grep -l "TODO\|FIXME\|XXX" {} \;
                           ```

                           ### 5.2 Generate API Documentation

                           ```bash
                           # Install Sphinx
                           pip install sphinx sphinx-rtd-theme

                           # Generate API docs
                           cd docs
                           sphinx-quickstart api_reference
                           sphinx-apidoc -o api_reference/source ../algorithms
                           cd api_reference
                           make html
                           ```

                           ### 5.3 Create Video Tutorials

                           **Topics:**
                           1. Installing QuantoniumOS (5 min)
                           2. Running benchmarks (10 min)
                           3. Using QuantSoundDesign (15 min)
                           4. Understanding Φ-RFT variants (20 min)
                           5. Hardware simulation (15 min)

                           ---

                           ## 🚀 Phase 6: Performance Optimization

                           ### 6.1 Profile Hot Paths

                           ```bash
                           # Profile RFT forward transform
                           python -m cProfile -o rft_profile.stats << EOF
                           import numpy as np
                           from algorithms.rft.core.rft_optimized import rft_forward_optimized
                           x = np.random.randn(4096)
                           for _ in range(1000):
                               rft_forward_optimized(x)
                               EOF

                               # Analyze profile
                               python -c "import pstats; p = pstats.Stats('rft_profile.stats'); p.sort_stats('cumulative').print_stats(20)"
                               ```

                               ### 6.2 Benchmark All Variants

                               ```bash
                               # Run comprehensive variant benchmark
                               python benchmarks/variant_benchmark_harness.py

                               # Compare against FFT
                               python experiments/competitors/benchmark_transforms_vs_fft.py \
                                  --sizes 256,512,1024,2048,4096 \
                                     --runs 100 \
                                        --output results/variant_performance.json
                                        ```

                                        ### 6.3 Optimize Native Modules

                                        ```bash
                                        # Rebuild with aggressive optimization
                                        cd src/rftmw_native/build
                                        cmake .. \
                                           -DCMAKE_BUILD_TYPE=Release \
                                              -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native" \
                                                 -DRFTMW_ENABLE_ASM=ON \
                                                    -DRFTMW_ENABLE_AVX512=ON
                                                    make -j$(nproc)

                                                    # Benchmark improvement
                                                    python -c "
                                                    import numpy as np
                                                    import rftmw_native
                                                    import time
                                                    x = np.random.randn(4096)
                                                    start = time.perf_counter()
                                                    for _ in range(1000):
                                                        rftmw_native.rft_forward(x)
                                                        print(f'Time: {(time.perf_counter()-start)*1000:.2f} ms')
                                                        "
                                                        ```

                                                        ---

                                                        ## 📊 Phase 7: Benchmarking & Validation

                                                        ### 7.1 Run Full Benchmark Suite

                                                        ```bash
                                                        # Run all 5 benchmark classes
                                                        python benchmarks/run_all_benchmarks.py --variants

                                                        # With JSON output
                                                        python benchmarks/run_all_benchmarks.py \
                                                           --variants \
                                                              --json results/full_benchmark_$(date +%Y%m%d).json
                                                              ```

                                                              ### 7.2 Run Hardware Validation

                                                              ```bash
                                                              cd hardware
                                                              bash verify_fixes.sh

                                                              # Generate hardware test report
                                                              python generate_hardware_test_vectors.py
                                                              python visualize_hardware_results.py
                                                              ```

                                                              ### 7.3 Validate All Claims

                                                              ```bash
                                                              # Run paper validation suite
                                                              python scripts/run_paper_validation_suite.py

                                                              # Verify scaling laws
                                                              python scripts/verify_scaling_laws.py

                                                              # Verify ASCII bottleneck theorem
                                                              python scripts/verify_ascii_bottleneck.py

                                                              # Verify variant properties
                                                              python scripts/verify_variant_claims.py
                                                              ```

                                                              ---

                                                              ## 🎨 Phase 8: UI/UX Improvements

                                                              ### 8.1 Polish QuantSoundDesign

                                                              **Tasks:**
                                                              - Add preset management system
                                                              - Implement MIDI file export
                                                              - Add keyboard shortcuts
                                                              - Create user manual
                                                              - Record demo videos

                                                              ```bash
                                                              cd src/apps/quantsounddesign
                                                              # TODO: Add preset system
                                                              # TODO: Add MIDI export
                                                              # TODO: Document keyboard shortcuts
                                                              ```

                                                              ### 8.2 Complete Mobile App

                                                              ```bash
                                                              cd quantonium-mobile
                                                              npm install
                                                              npm run build

                                                              # Test on simulators
                                                              npm run ios
                                                              npm run android
                                                              ```

                                                              ### 8.3 Create Web Demos

                                                              **Technologies:**
                                                              - Emscripten for WebAssembly
                                                              - React for UI
                                                              - Web Audio API for playback

                                                              ```bash
                                                              # Compile RFT to WASM
                                                              emcc algorithms/rft/core/rft_optimized.py -o web/rft.wasm

                                                              # Create interactive demos
                                                              # - Live RFT visualization
                                                              # - Audio processing demo
                                                              # - Variant comparison tool
                                                              ```

                                                              ---

                                                              ## 🔐 Phase 9: Security & Compliance

                                                              ### 9.1 License Header Injection

                                                              ```bash
                                                              # Run SPDX injection
                                                              python tools/spdx_inject.py

                                                              # Verify all files have proper headers
                                                              find . -name "*.py" -exec head -5 {} \; | grep -c "SPDX-License-Identifier"
                                                              ```

                                                              ### 9.2 Security Audit

                                                              **Tasks:**
                                                              1. Review crypto implementations
                                                              2. Check for hardcoded secrets
                                                              3. Validate input sanitization
                                                              4. Test for timing attacks

                                                              ```bash
                                                              # Check for secrets
                                                              git secrets --scan

                                                              # Run security linters
                                                              bandit -r algorithms/ src/ -f json -o security_report.json

                                                              # Check dependencies
                                                              pip-audit
                                                              ```

                                                              ### 9.3 Patent Compliance

                                                              ```bash
                                                              # Verify CLAIMS_PRACTICING_FILES.txt is current
                                                              cat CLAIMS_PRACTICING_FILES.txt

                                                              # Ensure all listed files have proper license headers
                                                              while read file; do
                                                                  head -10 "$file" | grep -q "LicenseRef-QuantoniumOS-Claims-NC" || \
                                                                          echo "Missing license: $file"
                                                                          done < CLAIMS_PRACTICING_FILES.txt
                                                                          ```

                                                                          ---

                                                                          ## 📅 Recommended Timeline

                                                                          ### Week 1: Safe Cleanup
                                                                          - [ ] Phase 1: Delete cache and build artifacts
                                                                          - [ ] Update .gitignore
                                                                          - [ ] Commit cleanup

                                                                          ### Week 2: Code Consolidation
                                                                          - [ ] Phase 2.1: Consolidate geometric hashing
                                                                          - [ ] Phase 2.2: Consolidate quantum code
                                                                          - [ ] Run tests, fix issues

                                                                          ### Week 3: Migration & Archival
                                                                          - [ ] Phase 2.3: Migrate apps
                                                                          - [ ] Phase 3: Archive deprecated code
                                                                          - [ ] Update documentation

                                                                          ### Week 4: Testing & Coverage
                                                                          - [ ] Phase 4: Run all tests
                                                                          - [ ] Fix failing tests
                                                                          - [ ] Add missing tests
                                                                          - [ ] Achieve 80%+ coverage

                                                                          ### Week 5: Documentation
                                                                          - [ ] Phase 5: Update all docs
                                                                          - [ ] Generate API docs
                                                                          - [ ] Create tutorials

                                                                          ### Week 6: Performance
                                                                          - [ ] Phase 6: Profile and optimize
                                                                          - [ ] Benchmark all variants
                                                                          - [ ] Rebuild native modules

                                                                          ### Week 7: Validation
                                                                          - [ ] Phase 7: Full benchmark suite
                                                                          - [ ] Hardware validation
                                                                          - [ ] Verify all claims

                                                                          ### Week 8: Polish
                                                                          - [ ] Phase 8: UI/UX improvements
                                                                          - [ ] Phase 9: Security audit
                                                                          - [ ] Final review

                                                                          ---

                                                                          ## 🎯 Success Metrics

                                                                          ### Code Quality
                                                                          - [ ] Zero lint errors (Verilator, flake8, black)
                                                                          - [ ] 80%+ test coverage
                                                                          - [ ] All tests passing
                                                                          - [ ] No duplicate implementations

                                                                          ### Documentation
                                                                          - [ ] 100% of public APIs documented
                                                                          - [ ] All READMEs current
                                                                          - [ ] Video tutorials created
                                                                          - [ ] API docs generated

                                                                          ### Performance
                                                                          - [ ] Benchmarks complete for all 14 variants
                                                                          - [ ] Native modules built and tested
                                                                          - [ ] Hardware simulations passing
                                                                          - [ ] Performance data published

                                                                          ### Organization
                                                                          - [ ] Clear directory structure
                                                                          - [ ] No build artifacts in repo
                                                                          - [ ] Deprecated code archived
                                                                          - [ ] License headers complete

                                                                          ---

                                                                          ## 🚨 Critical Warnings

                                                                          ### DO NOT Delete Without Backup:
                                                                          1. Any file in `algorithms/rft/core/` - These are core implementations
                                                                          2. Test files (even if failing) - May contain valuable test cases
                                                                          3. Experiment results - Irreplaceable research data
                                                                          4. Hardware test logs - Validation evidence

                                                                          ### Always Test After:
                                                                          1. Consolidating code
                                                                          2. Updating imports
                                                                          3. Moving files
                                                                          4. Deleting archives

                                                                          ### Git Best Practices:
                                                                          ```bash
                                                                          # Create feature branch for each phase
                                                                          git checkout -b cleanup/phase1-safe-deletions

                                                                          # Commit frequently
                                                                          git commit -m "Phase 1: Remove __pycache__ directories"

                                                                          # Create backups before major changes
                                                                          git tag backup-$(date +%Y%m%d) -m "Backup before cleanup"
                                                                          ```

                                                                          ---

                                                                          ## 📞 Questions & Support

                                                                          **Before proceeding with cleanup:**
                                                                          1. Review this plan with team
                                                                          2. Create backups
                                                                          3. Test on development branch first
                                                                          4. Monitor CI/CD pipelines

                                                                          **Contact:**
                                                                          Luis M. Minier - luisminier79@gmail.com

                                                                          ---

                                                                          **End of Cleanup Action Plan**

                                                                          *This plan should be executed incrementally with thorough testing at each phase.*  
                                                                          *Track progress using checkboxes and update as phases complete.*
                                                                          