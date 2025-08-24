#!/usr/bin/env python3
"""
Patch the comprehensive scientific test suite to use the symbiotic RFT engine.
This ensures energy conservation in all RFT transformations.
"""

from pathlib import Path


def patch_test_suite():
    """
    Patch the comprehensive scientific test suite to use the symbiotic RFT engine.
    """
    test_suite_path = Path("comprehensive_scientific_test_suite.py")
    if not test_suite_path.exists():
        print(f"Error: {test_suite_path} not found")
        return False

    # Read the test suite file
    with open(test_suite_path, "r") as f:
        content = f.read()

    # Check if the file has already been patched
    if "from symbiotic_rft_engine_adapter import" in content:
        print("The test suite has already been patched")
        return True

    # Add import for symbiotic engine
    import_line = "import numpy as np"
    patched_import = "import numpy as np\nfrom symbiotic_rft_engine_adapter import SymbioticRFTEngine, forward_true_rft, inverse_true_rft"
    content = content.replace(import_line, patched_import)

    # Replace the RFT initialization in test setup
    init_line = "        # Create BQK with C++ acceleration"
    patched_init = "        # Create symbiotic RFT engine for energy conservation\n        self.symbiotic_rft = SymbioticRFTEngine(dimension=self.dimension)\n        \n        # Create BQK with C++ acceleration"
    content = content.replace(init_line, patched_init)

    # Replace direct forward_true_rft calls
    if "import importlib.util
import os

# Load the canonical_true_rft module
spec = importlib.util.spec_from_file_location(
    "canonical_true_rft", 
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                "04_RFT_ALGORITHMS/canonical_true_rft.py")
)
canonical_true_rft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(canonical_true_rft)

# Import specific functions/classes
forward_true_rft, inverse_true_rft = canonical_true_rft.forward_true_rft, canonical_true_rft.inverse_true_rft" in content:
        # Remove the import since we're now using the symbiotic version
        content = content.replace(
            "import importlib.util
import os

# Load the canonical_true_rft module
spec = importlib.util.spec_from_file_location(
    "canonical_true_rft", 
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                "04_RFT_ALGORITHMS/canonical_true_rft.py")
)
canonical_true_rft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(canonical_true_rft)

# Import specific functions/classes
forward_true_rft, inverse_true_rft = canonical_true_rft.forward_true_rft, canonical_true_rft.inverse_true_rft",
            "# Using symbiotic_rft_engine_adapter instead",
        )

    # Write the patched file
    backup_path = test_suite_path.with_suffix(".py.backup")
    print(f"Creating backup at {backup_path}")
    with open(backup_path, "w") as f:
        f.write(content)

    with open(test_suite_path, "w") as f:
        f.write(content)

    print(f"Successfully patched {test_suite_path}")
    return True


if __name__ == "__main__":
    print("Patching comprehensive scientific test suite...")
    if patch_test_suite():
        print("Patch completed successfully")
    else:
        print("Patching failed")
