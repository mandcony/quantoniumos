#!/usr/bin/env python3
"""
Run All Core Validators
=======================
This script runs all validators in the 02_CORE_VALIDATORS package and generates
a comprehensive validation report for the entire QuantoniumOS system.
"""

import importlib
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))


# Custom JSON encoder to handle non-serializable objects
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)


def run_all_validators():
    """Run all validators in the 02_CORE_VALIDATORS package"""
    validators = [
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

    # First try to import key modules
    try:
        import bulletproof_quantum_kernel
        import topological_quantum_kernel
    except ImportError as e:
        print(f"[WARNING] Import failed: {e}")
        print("[ACTION] Using stub implementations for missing modules")

    try:
        import paper_compliant_rft_fixed
    except ImportError:
        print("[WARNING] Paper compliant RFT module not found")

    try:
        import quantonium_hpc_pipeline
    except ImportError:
        print("[WARNING] HPC pipeline module not found")

    results = {}

    for validator in validators:
        try:
            module_name = f"02_CORE_VALIDATORS.{validator}"
            module = importlib.import_module(module_name)
            if hasattr(module, "run_validation"):
                print(f"Running validator: {validator}")
                results[validator] = module.run_validation()
            elif hasattr(module, "main"):
                print(f"Running validator: {validator}")
                results[validator] = module.main()
            else:
                print(
                    f"Validator {validator} has no run_validation() or main() function"
                )
                results[validator] = {
                    "status": "ERROR",
                    "message": "No entry point found",
                }
        except Exception as e:
            print(f"Error running validator {validator}: {e}")
            results[validator] = {"status": "ERROR", "message": str(e)}

    return results


def main():
    """Main entry point for running all validators"""
    print("\n" + "=" * 80)
    print(" QuantoniumOS Complete Validation Suite ".center(80, "="))
    print("=" * 80 + "\n")

    print("Starting validation of all QuantoniumOS components...")
    start_time = time.time()

    # Run all validators
    results = run_all_validators()

    # Calculate overall status
    total = len(results)
    passed = sum(
        1 for v in results.values() if isinstance(v, dict) and v.get("status") == "PASS"
    )
    failed = sum(
        1 for v in results.values() if isinstance(v, dict) and v.get("status") == "FAIL"
    )
    errors = sum(
        1
        for v in results.values()
        if isinstance(v, dict) and v.get("status") == "ERROR"
    )

    # Generate summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"Validation Summary: {passed}/{total} tests passed")
    print(f"Time elapsed: {elapsed:.2f} seconds")
    print("-" * 80)
    print(f"PASS: {passed}")
    print(f"FAIL: {failed}")
    print(f"ERROR: {errors}")
    print("=" * 80)

    # Save detailed report to file
    report_path = project_root / "validation_results" / "full_validation_report.json"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(
            {
                "timestamp": time.time(),
                "summary": {
                    "total": total,
                    "passed": passed,
                    "failed": failed,
                    "errors": errors,
                    "elapsed": elapsed,
                },
                "results": results,
            },
            f,
            indent=2,
            cls=CustomJSONEncoder,
        )
    print(f"Detailed report saved to {report_path}")

    return 0 if failed == 0 and errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
