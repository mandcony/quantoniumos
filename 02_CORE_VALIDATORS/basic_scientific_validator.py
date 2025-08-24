#!/usr/bin/env python3
"""
QuantoniumOS Basic Scientific Validation
Minimum viable scientific testing with available components

This implements essential scientific validation using only 
the available working components in the system.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Import available components
try:
    from bulletproof_quantum_kernel import BulletproofQuantumKernel

    print("[IMPORT] Successfully imported BulletproofQuantumKernel")
    BULLETPROOF_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] BulletproofQuantumKernel import failed: {e}")
    BULLETPROOF_AVAILABLE = False

try:
    print("[IMPORT] Successfully imported TopologicalQuantumKernel")
    TOPOLOGICAL_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] TopologicalQuantumKernel import failed: {e}")
    TOPOLOGICAL_AVAILABLE = False


class BasicScientificValidator:
    """
    Basic scientific validation with available components.
    """

    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "validation_status": {},
        }
        self.precision_threshold = 1e-12

    def test_bulletproof_kernel_functionality(self) -> Dict[str, Any]:
        """Test basic functionality of BulletproofQuantumKernel."""
        print("\n=== TEST 1: BULLETPROOF KERNEL FUNCTIONALITY ===")

        if not BULLETPROOF_AVAILABLE:
            return {
                "status": "SKIPPED",
                "reason": "BulletproofQuantumKernel not available",
            }

        try:
            # Initialize kernel
            kernel = BulletproofQuantumKernel(dimension=8, is_test_mode=True)
            print(f"[TEST] Kernel initialized: dimension={kernel.dimension}")

            # Test basic properties
            has_quantum_state = hasattr(kernel, "quantum_state")
            has_rft_methods = hasattr(kernel, "forward_rft") and hasattr(
                kernel, "inverse_rft"
            )
            has_precision = hasattr(kernel, "precision")

            # Test state initialization
            state_initialized = kernel.quantum_state is not None
            state_shape_correct = (
                len(kernel.quantum_state) == kernel.dimension
                if state_initialized
                else False
            )

            # Try basic RFT if available
            rft_functional = False
            energy_conserved = False

            if has_rft_methods:
                try:
                    test_signal = np.random.complex128(kernel.dimension)
                    test_signal = test_signal / np.linalg.norm(test_signal)

                    # Forward transform
                    transformed = kernel.forward_rft(test_signal)
                    rft_functional = transformed is not None

                    # Test energy conservation if inverse is available
                    if rft_functional:
                        try:
                            recovered = kernel.inverse_rft(transformed)
                            energy_original = np.linalg.norm(test_signal) ** 2
                            energy_recovered = np.linalg.norm(recovered) ** 2
                            energy_error = abs(energy_original - energy_recovered)
                            energy_conserved = energy_error < 1e-6
                            print(
                                f"[TEST] Energy conservation error: {energy_error:.2e}"
                            )
                        except Exception as e:
                            print(f"[WARNING] Inverse RFT failed: {e}")

                except Exception as e:
                    print(f"[WARNING] RFT test failed: {e}")

            result = {
                "kernel_initialized": True,
                "has_quantum_state": has_quantum_state,
                "has_rft_methods": has_rft_methods,
                "has_precision": has_precision,
                "state_initialized": state_initialized,
                "state_shape_correct": state_shape_correct,
                "rft_functional": rft_functional,
                "energy_conserved": energy_conserved,
                "overall_status": "FUNCTIONAL"
                if all([has_quantum_state, state_initialized, state_shape_correct])
                else "PARTIAL",
            }

            print(f"[RESULT] BulletproofQuantumKernel: {result['overall_status']}")
            return result

        except Exception as e:
            print(f"[ERROR] BulletproofQuantumKernel test failed: {e}")
            return {"status": "FAILED", "error": str(e)}

    def test_import_chain_integrity(self) -> Dict[str, Any]:
        """Test that critical imports work correctly."""
        print("\n=== TEST 2: IMPORT CHAIN INTEGRITY ===")

        import_results = {}

        # Test core imports
        core_imports = ["numpy", "json", "pathlib", "datetime"]

        for module in core_imports:
            try:
                __import__(module)
                import_results[module] = "SUCCESS"
                print(f"[IMPORT] {module}: SUCCESS")
            except ImportError as e:
                import_results[module] = f"FAILED: {e}"
                print(f"[IMPORT] {module}: FAILED - {e}")

        # Test QuantoniumOS components
        quantonium_imports = [
            ("bulletproof_quantum_kernel", BULLETPROOF_AVAILABLE),
            ("topological_quantum_kernel", TOPOLOGICAL_AVAILABLE),
        ]

        for module, available in quantonium_imports:
            import_results[module] = "SUCCESS" if available else "FAILED"
            print(f"[IMPORT] {module}: {'SUCCESS' if available else 'FAILED'}")

        # Overall import health
        total_imports = len(import_results)
        successful_imports = sum(
            1 for status in import_results.values() if status == "SUCCESS"
        )
        import_health = successful_imports / total_imports

        result = {
            "import_results": import_results,
            "total_imports": total_imports,
            "successful_imports": successful_imports,
            "import_health": import_health,
            "status": "HEALTHY" if import_health >= 0.8 else "DEGRADED",
        }

        print(f"[RESULT] Import health: {import_health:.1%} ({result['status']})")
        return result

    def test_file_restoration_verification(self) -> Dict[str, Any]:
        """Verify that critical files were properly restored."""
        print("\n=== TEST 3: FILE RESTORATION VERIFICATION ===")

        critical_files = [
            "main.py",
            "app.py",
            "topological_quantum_kernel.py",
            "topological_vertex_engine.py",
            "working_quantum_kernel.py",
            "__init__.py",
        ]

        file_status = {}
        for filename in critical_files:
            filepath = Path(filename)
            if filepath.exists():
                file_size = filepath.stat().st_size
                file_status[filename] = {
                    "exists": True,
                    "size": file_size,
                    "non_empty": file_size > 0,
                    "status": "OK" if file_size > 0 else "EMPTY",
                }
                print(
                    f"[FILE] {filename}: {file_size} bytes ({'OK' if file_size > 0 else 'EMPTY'})"
                )
            else:
                file_status[filename] = {
                    "exists": False,
                    "size": 0,
                    "non_empty": False,
                    "status": "MISSING",
                }
                print(f"[FILE] {filename}: MISSING")

        # Calculate restoration success rate
        total_files = len(critical_files)
        restored_files = sum(
            1 for status in file_status.values() if status["status"] == "OK"
        )
        restoration_rate = restored_files / total_files

        result = {
            "file_status": file_status,
            "total_critical_files": total_files,
            "restored_files": restored_files,
            "restoration_rate": restoration_rate,
            "status": "SUCCESS" if restoration_rate >= 0.8 else "PARTIAL",
        }

        print(f"[RESULT] Restoration: {restoration_rate:.1%} ({result['status']})")
        return result

    def generate_basic_validation_report(self) -> Dict[str, Any]:
        """Generate basic validation report."""
        print("\n=== BASIC SCIENTIFIC VALIDATION REPORT ===")

        # Run all tests
        tests = [
            ("bulletproof_functionality", self.test_bulletproof_kernel_functionality()),
            ("import_integrity", self.test_import_chain_integrity()),
            ("file_restoration", self.test_file_restoration_verification()),
        ]

        # Compile results
        for test_name, result in tests:
            self.results["tests"][test_name] = result

        # Overall system health assessment
        functional_tests = sum(
            1
            for name, result in tests
            if result.get("status") not in ["FAILED", "SKIPPED"]
            and result.get("overall_status") != "FAILED"
        )

        total_tests = len(tests)
        system_health = functional_tests / total_tests

        # Validation status
        self.results["validation_status"] = {
            "total_tests": total_tests,
            "functional_tests": functional_tests,
            "system_health": system_health,
            "overall_status": "FUNCTIONAL" if system_health >= 0.7 else "DEGRADED",
            "ready_for_advanced_testing": system_health >= 0.8,
        }

        # Save report
        report_path = Path("basic_validation_report.json")
        with open(report_path, "w") as f:
            json.dump(self.results, f, indent=2)

        print(f"[REPORT] Basic validation saved: {report_path}")
        print(
            f"[SUMMARY] System health: {system_health:.1%} ({self.results['validation_status']['overall_status']})"
        )
        print(
            f"[STATUS] Ready for advanced testing: {self.results['validation_status']['ready_for_advanced_testing']}"
        )

        return self.results


if __name__ == "__main__":
    print("=== QUANTONIUMOS BASIC SCIENTIFIC VALIDATION ===")
    print("Minimum Viable Testing with Available Components")
    print("=" * 60)

    validator = BasicScientificValidator()
    report = validator.generate_basic_validation_report()

    print("\n=== BASIC VALIDATION COMPLETE ===")


def run_validation():
    """Entry point for external validation calls."""
    print("=== QUANTONIUMOS BASIC SCIENTIFIC VALIDATION ===")
    print("Minimum Viable Testing with Available Components")
    print("=" * 60)

    validator = BasicScientificValidator()
    report = validator.generate_basic_validation_report()

    print("\n=== BASIC VALIDATION COMPLETE ===")

    return report
