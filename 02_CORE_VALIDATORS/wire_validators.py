#!/usr/bin/env python3
"""
QuantoniumOS Validation Wiring Module
=====================================
This module ensures all engines and Python scripts are properly wired to run the validation tests.
"""

import importlib
import os
import subprocess
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Define the list of all validators
VALIDATORS = [
    "basic_scientific_validator",
    "definitive_quantum_validation",
    "final_paper_compliance_test",
    "patent_validation_summary",
    "phd_level_scientific_validator",
    "publication_ready_validation",
    "true_rft_patent_validator",
    "validate_complete_hpc_pipeline",
    "validate_energy_conservation",
    "validate_system",
    "verify_breakthrough",
    "verify_system",
]

# Define key algorithms and engines to be validated
KEY_ALGORITHMS = [
    "04_RFT_ALGORITHMS/true_rft_exact.py",
    "04_RFT_ALGORITHMS/canonical_resonance_fourier_transform.py",
    "04_RFT_ALGORITHMS/canonical_true_rft.py",
    "04_RFT_ALGORITHMS/energy_conserving_rft_adapter.py",
    "04_RFT_ALGORITHMS/mathematically_rigorous_rft.py",
    "04_RFT_ALGORITHMS/production_canonical_rft.py",
    "04_RFT_ALGORITHMS/resonance_fourier_transform.py",
    "04_RFT_ALGORITHMS/rft_final_validation.py",
    "05_QUANTUM_ENGINES/quantum_circuit_engine.py",
    "05_QUANTUM_ENGINES/quantum_simulator.py",
    "05_QUANTUM_ENGINES/quantum_vertex_engine.py",
]


def run_validator(validator_name):
    """Run a specific validator and return the result"""
    try:
        validator_path = f"02_CORE_VALIDATORS.{validator_name}"
        module = importlib.import_module(validator_path)

        if hasattr(module, "run_validation"):
            print(f"Running validator: {validator_name}")
            return module.run_validation()
        elif hasattr(module, "main"):
            print(f"Running validator: {validator_name}")
            return module.main()
        else:
            print(
                f"Validator {validator_name} has no run_validation() or main() function"
            )
            return {"status": "ERROR", "message": "No entry point found"}
    except Exception as e:
        print(f"Error running validator {validator_name}: {e}")
        return {"status": "ERROR", "message": str(e)}


def run_algorithm_with_validation(algorithm_path):
    """Run an algorithm with validation"""
    try:
        full_path = project_root / algorithm_path
        if not full_path.exists():
            print(f"Algorithm not found: {algorithm_path}")
            return False

        print(f"Running algorithm with validation: {algorithm_path}")
        result = subprocess.run(
            [sys.executable, str(full_path), "--validate"],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print(f"✅ {algorithm_path} passed")
            return True
        else:
            print(f"❌ {algorithm_path} failed:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"Error running algorithm {algorithm_path}: {e}")
        return False


def ensure_imports_available():
    """Ensure all required imports are available"""
    try:
        # Try to import core modules
        import_modules = [
            "bulletproof_quantum_kernel",
            "topological_quantum_kernel",
            "paper_compliant_rft_fixed",
            "quantonium_os_unified",
            "quantonium_design_system",
            "quantonium_hpc_pipeline",
        ]

        for module in import_modules:
            try:
                importlib.import_module(module)
                print(f"✅ Successfully imported {module}")
            except ImportError:
                print(f"❌ Failed to import {module}")

        return True
    except Exception as e:
        print(f"Error checking imports: {e}")
        return False


def run_all():
    """Run all validators on all key algorithms"""
    print("\n" + "=" * 80)
    print(" QuantoniumOS Validation Wiring - Full Test Suite ".center(80, "="))
    print("=" * 80 + "\n")

    # First ensure all imports are available
    print("Checking required imports...")
    ensure_imports_available()

    # Then run the direct validators
    print("Running all validators directly...")
    validator_results = {}
    for validator in VALIDATORS:
        validator_results[validator] = run_validator(validator)
    for validator in VALIDATORS:
        validator_results[validator] = run_validator(validator)

    # Then run all algorithms with validation
    print("\nRunning all algorithms with validation...")
    algorithm_results = {}
    for algorithm in KEY_ALGORITHMS:
        algorithm_results[algorithm] = run_algorithm_with_validation(algorithm)

    # Print summary
    print("\n" + "=" * 80)
    print(" Validation Summary ".center(80, "="))
    print("-" * 80)

    validator_passed = sum(
        1
        for r in validator_results.values()
        if isinstance(r, dict) and r.get("status") == "PASS"
    )
    print(f"Validators: {validator_passed}/{len(VALIDATORS)} passed")

    algorithm_passed = sum(1 for r in algorithm_results.values() if r)
    print(f"Algorithms: {algorithm_passed}/{len(KEY_ALGORITHMS)} passed")

    print("=" * 80)

    return validator_passed == len(VALIDATORS) and algorithm_passed == len(
        KEY_ALGORITHMS
    )


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
