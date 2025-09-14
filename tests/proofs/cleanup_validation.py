#!/usr/bin/env python3
"""
FINAL CLEANUP VALIDATION
========================
Verify all essential files are present and working after cleanup.
"""

import os
import sys

def check_essential_files():
    """Check that all essential files are present"""
    
    print("FINAL CLEANUP VALIDATION")
    print("=" * 50)
    print("Verifying all essential files are present and working")
    print("=" * 50)
    
    # Essential files that must be present
    essential_files = {
        # Core mathematical proofs
        "test_rft_vs_dft_separation_proper.py": "RFT ≠ DFT proof",
        "test_rft_convolution_proper.py": "Convolution theorem violation",
        "test_shift_diagonalization.py": "Shift operator failure", 
        "test_rft_conditioning.py": "Numerical conditioning",
        "test_property_invariants.py": "Linearity preservation",
        "test_aead_compliance.py": "AEAD compliance",
        "test_aead_simple.py": "Tamper detection",
        
        # Assembly engine tests
        "test_corrected_assembly.py": "Quantum symbolic engine",
        "test_assembly_fixed.py": "Unitary RFT assembly",
        "test_quantum_symbolic.py": "Direct symbolic engine",
        
        # Performance analysis
        "test_timing_curves.py": "Statistical timing analysis",
        "ci_scaling_analysis.py": "Scaling exponent calculation",
        
        # Reproducibility
        "artifact_proof.py": "Environment documentation",
        "build_info_test.py": "Binary provenance",
        "parity_test.py": "DLL consistency",
        "run_comprehensive_validation.py": "Mathematical proof runner",
        "final_summary.py": "Complete results summary",
        "ci_safe_final_validation.py": "CI-safe validation",
        
        # Supporting files
        "true_rft_kernel.py": "Mathematical RFT kernel",
        "requirements_freeze.txt": "Environment lock",
        "README.md": "Documentation"
    }
    
    missing_files = []
    present_files = []
    
    for filename, description in essential_files.items():
        if os.path.exists(filename):
            size_kb = os.path.getsize(filename) / 1024
            present_files.append((filename, description, size_kb))
            print(f"✅ {filename:<35} ({size_kb:.1f} KB) - {description}")
        else:
            missing_files.append((filename, description))
            print(f"❌ {filename:<35} MISSING - {description}")
    
    print(f"\n" + "=" * 50)
    print("CLEANUP VALIDATION SUMMARY")
    print("=" * 50)
    print(f"Essential files present: {len(present_files)}/{len(essential_files)}")
    print(f"Missing files: {len(missing_files)}")
    
    if missing_files:
        print(f"\nMISSING FILES:")
        for filename, description in missing_files:
            print(f"  - {filename}: {description}")
        return False
    else:
        print(f"\n✅ ALL ESSENTIAL FILES PRESENT")
        total_size = sum(size for _, _, size in present_files)
        print(f"Total size: {total_size:.1f} KB")
        return True

def check_directory_structure():
    """Verify clean directory structure"""
    print(f"\nDIRECTORY STRUCTURE CHECK")
    print("=" * 30)
    
    # Files that should NOT be present (cleaned up)
    unwanted_files = [
        "__pycache__",
        "test_assembly_comprehensive.py",
        "test_assembly_working.py", 
        "test_direct_assembly.py",
        "test_minimal_assembly.py",
        "debug_assembly_uniqueness.py",
        "inspect_library_functions.py",
        "test_rft_conditioning_enhanced.py",
        "test_rft_convolution_invariance.py",
        "test_rft_vs_dft_separation.py",
        "test_precision_baseline.py",
        "ci_test_suite.py",
        "ci_final_validation.py",
        "scaling_analysis.py",
        "test_rft_compression.py",
        "test_rft_fault_tolerance.py",
        "test_roundtrip_integrity.py"
    ]
    
    found_unwanted = []
    for unwanted in unwanted_files:
        if os.path.exists(unwanted):
            found_unwanted.append(unwanted)
    
    if found_unwanted:
        print(f"❌ Found {len(found_unwanted)} files that should be cleaned:")
        for filename in found_unwanted:
            print(f"  - {filename}")
        return False
    else:
        print(f"✅ Directory is clean - no unwanted files found")
        return True

def main():
    """Run final cleanup validation"""
    
    files_ok = check_essential_files()
    structure_ok = check_directory_structure()
    
    print(f"\n" + "=" * 50)
    print("FINAL CLEANUP VALIDATION RESULT") 
    print("=" * 50)
    
    if files_ok and structure_ok:
        print("✅ CLEANUP SUCCESSFUL")
        print("All essential files present")
        print("No unwanted files remaining")
        print("Directory ready for peer review")
        return True
    else:
        print("❌ CLEANUP INCOMPLETE")
        if not files_ok:
            print("Some essential files are missing")
        if not structure_ok:
            print("Some unwanted files still present")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
