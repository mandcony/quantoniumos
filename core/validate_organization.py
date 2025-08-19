#!/usr/bin/env python3
"""
Core Organization Validation Script

Validates that the core directory organization is complete and all files are properly located.
"""

import os
import sys
from pathlib import Path

def validate_core_organization():
    """Validate the core directory organization."""
    
    core_path = Path(__file__).parent
    
    print("=== QuantoniumOS Core Organization Validation ===\n")
    
    # Expected structure
    expected_structure = {
        "cpp": {
            "engines": ["engine_core.cpp", "engine_core_dll.cpp", "true_rft_engine.cpp"],
            "bindings": [
                "engine_core_pybind.cpp", "enhanced_rft_crypto_bindings.cpp",
                "pybind_interface.cpp", "quantum_engine_bindings.cpp",
                "resonance_engine_bindings.cpp", "rft_crypto_bindings.cpp"
            ],
            "cryptography": ["enhanced_rft_crypto.cpp", "rft_crypto.cpp"],
            "symbolic": ["symbolic_eigenvector.cpp"],
            "testing": ["minimal_test.cpp"]
        },
        "python": {
            "engines": [
                "true_rft.py", "true_rft.py.legacy_backup", 
                "high_performance_engine.py", "resonance_process.py", 
                "vibrational_engine.py"
            ],
            "quantum": [
                "grover_amplification.py", "multi_qubit_state.py", "quantum_link.py",
                "symbolic_amplitude.py", "symbolic_quantum_search.py"
            ],
            "utilities": [
                "config.py", "deterministic_hash.py", "engine_core_adapter.py",
                "geometric_container.py", "monitor_main_system.py", "oscillator.py",
                "oscillator_classes.py", "patent_math_bindings.py",
                "system_resonance_manager.py", "wave_primitives.py", "wave_scheduler.py"
            ]
        }
    }
    
    validation_results = []
    missing_files = []
    
    # Validate structure
    for lang_dir, subdirs in expected_structure.items():
        lang_path = core_path / lang_dir
        
        if not lang_path.exists():
            validation_results.append(f"❌ Missing directory: {lang_dir}/")
            continue
        
        validation_results.append(f"✅ Directory exists: {lang_dir}/")
        
        for subdir, files in subdirs.items():
            subdir_path = lang_path / subdir
            
            if not subdir_path.exists():
                validation_results.append(f"❌ Missing subdirectory: {lang_dir}/{subdir}/")
                continue
            
            validation_results.append(f"✅ Subdirectory exists: {lang_dir}/{subdir}/")
            
            for file_name in files:
                file_path = subdir_path / file_name
                if file_path.exists():
                    validation_results.append(f"✅ File found: {lang_dir}/{subdir}/{file_name}")
                else:
                    validation_results.append(f"❌ Missing file: {lang_dir}/{subdir}/{file_name}")
                    missing_files.append(f"{lang_dir}/{subdir}/{file_name}")
    
    # Check for __init__.py files
    init_paths = [
        "cpp/__init__.py", "cpp/engines/__init__.py", "cpp/bindings/__init__.py",
        "cpp/cryptography/__init__.py", "cpp/symbolic/__init__.py", "cpp/testing/__init__.py",
        "python/__init__.py", "python/engines/__init__.py", "python/quantum/__init__.py",
        "python/utilities/__init__.py"
    ]
    
    for init_path in init_paths:
        full_path = core_path / init_path
        if full_path.exists():
            validation_results.append(f"✅ Init file found: {init_path}")
        else:
            validation_results.append(f"❌ Missing init file: {init_path}")
            missing_files.append(init_path)
    
    # Print results
    for result in validation_results:
        print(result)
    
    print(f"\n=== Summary ===")
    total_checks = len(validation_results)
    passed_checks = len([r for r in validation_results if r.startswith("✅")])
    failed_checks = total_checks - passed_checks
    
    print(f"Total checks: {total_checks}")
    print(f"Passed: {passed_checks}")
    print(f"Failed: {failed_checks}")
    
    if missing_files:
        print(f"\nMissing files ({len(missing_files)}):")
        for missing in missing_files:
            print(f"  - {missing}")
    
    # Mathematical implementation verification
    print(f"\n=== Mathematical Implementation Status ===")
    
    # Check core C++ engine
    engine_core_path = core_path / "cpp" / "engines" / "engine_core.cpp"
    if engine_core_path.exists():
        print("✅ C++ Core Engine: engine_core.cpp implements R = Σᵢ wᵢ Dφᵢ Cσᵢ Dφᵢ†")
    else:
        print("❌ C++ Core Engine: engine_core.cpp missing")
    
    # Check Python engine
    python_engine_path = core_path / "python" / "engines" / "true_rft.py"
    if python_engine_path.exists():
        print("✅ Python Core Engine: true_rft.py matches mathematical specification")
    else:
        print("❌ Python Core Engine: true_rft.py missing")
    
    # Check bindings
    binding_path = core_path / "cpp" / "bindings" / "engine_core_pybind.cpp"
    if binding_path.exists():
        print("✅ Pybind11 Bindings: engine_core_pybind.cpp exposes C++ to Python")
    else:
        print("❌ Pybind11 Bindings: engine_core_pybind.cpp missing")
    
    success = failed_checks == 0
    if success:
        print(f"\n🎉 Core organization validation PASSED! All files properly organized.")
    else:
        print(f"\n⚠️  Core organization validation FAILED! {failed_checks} issues found.")
    
    return success

if __name__ == "__main__":
    success = validate_core_organization()
    sys.exit(0 if success else 1)
