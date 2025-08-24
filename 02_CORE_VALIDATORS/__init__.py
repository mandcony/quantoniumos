"""
02_CORE_VALIDATORS package for QuantoniumOS.
This package contains core validation modules for verifying the correctness,
performance, and compliance of QuantoniumOS components.
"""

import sys
from pathlib import Path

# Define validator modules
__all__ = [
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

# Add parent directory to path for importing from project root
sys.path.append(str(Path(__file__).parent.parent))


def run_all_validators():
    """Run all validators in this package and return results dictionary"""
    import importlib

    results = {}

    for validator in __all__:
        try:
            module = importlib.import_module(f"02_CORE_VALIDATORS.{validator}")
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
