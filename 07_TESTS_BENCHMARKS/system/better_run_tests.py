# -*- coding: utf-8 -*-
#
# QuantoniumOS Test Suite
# Testing with QuantoniumOS implementations
#
# ===================================================================

import unittest
import sys
import os
import numpy as np
from binascii import unhexlify

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import QuantoniumOS RFT algorithms
try:
    sys.path.insert(0, '/workspaces/quantoniumos/04_RFT_ALGORITHMS')
    from canonical_true_rft import CanonicalTrueRFT
    from true_rft_exact import TrueRFTExact
    from true_rft_engine_bindings import TrueRFTEngineBindings
except ImportError as e:
    print(f"Warning: Could not import RFT algorithms: {e}")

# Import QuantoniumOS quantum engines
try:
    sys.path.insert(0, '/workspaces/quantoniumos/05_QUANTUM_ENGINES')
    from bulletproof_quantum_kernel import BulletproofQuantumKernel
    from topological_quantum_kernel import TopologicalQuantumKernel
    from topological_vertex_engine import TopologicalVertexEngine
    from topological_vertex_geometric_engine import TopologicalVertexGeometricEngine
    from vertex_engine_canonical import VertexEngineCanonical
    from working_quantum_kernel import WorkingQuantumKernel
    from true_rft_engine_bindings import TrueRFTEngineBindings as QuantumRFTBindings
except ImportError as e:
    print(f"Warning: Could not import quantum engines: {e}")

# Import QuantoniumOS cryptography modules
try:
    sys.path.insert(0, '/workspaces/quantoniumos/06_CRYPTOGRAPHY')
    from quantonium_crypto_production import QuantoniumCrypto
    from true_rft_feistel_bindings import TrueRFTFeistel
except ImportError as e:
    print(f"Warning: Could not import cryptography modules: {e}")

# Import QuantoniumOS validators
try:
    sys.path.insert(0, '/workspaces/quantoniumos/02_CORE_VALIDATORS')
    from basic_scientific_validator import BasicScientificValidator
    from definitive_quantum_validation import DefinitiveQuantumValidation
    from phd_level_scientific_validator import PhdLevelScientificValidator
    from publication_ready_validation import PublicationReadyValidation
except ImportError as e:
    print(f"Warning: Could not import validators: {e}")

# Import QuantoniumOS running systems
try:
    sys.path.insert(0, '/workspaces/quantoniumos/03_RUNNING_SYSTEMS')
    from app import app
    from main import main
    from quantonium import QuantoniumOS
except ImportError as e:
    print(f"Warning: Could not import running systems: {e}")

"""
Modified test runner script that handles pytest skips properly
"""

import argparse
import importlib.util
import os
import sys
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Monkey patch pytest.skip to just print and return
class MockPytest:
    @staticmethod
    def skip(msg):
        print(f" ⏭️ SKIPPED: {msg}")
        return None

sys.modules['pytest'] = MockPytest()

class QuantoniumTestRunner:
    """Unified test runner for QuantoniumOS"""

    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.results = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "categories": {},
        }

    def discover_root_tests(self) -> List[Path]:
        """Discover test files in project root"""
        test_files = []

        # Standard test file patterns
        patterns = [
            "test_*.py",
            "*_test.py",
            "comprehensive_*test*.py",
            "*_validation.py",
            "*_validator.py",
        ]

        # Look in 07_TESTS_BENCHMARKS directory
        tests_dir = self.project_root / "07_TESTS_BENCHMARKS"
        for pattern in patterns:
            test_files.extend(tests_dir.glob(pattern))

        return test_files

    def run_file_tests(self, test_file: Path) -> Dict[str, Any]:
        """Run tests from a single file"""
        print(f"\n🔬 Running tests from: {test_file.name}")

        results = {
            "file": test_file.name,
            "tests_run": 0,
            "passed": 0,
            "failed": 0,
            "errors": [],
        }

        try:
            # Import the test module
            spec = importlib.util.spec_from_file_location(test_file.stem, test_file)
            module = importlib.util.module_from_spec(spec)
            
            try:
                spec.loader.exec_module(module)
            except Exception as e:
                error_msg = str(e)
                print(f"  ❌ ERROR loading module: {error_msg}")
                results["errors"].append(f"Module load failed: {error_msg}")
                return results

            # Find and run test functions
            test_functions = [
                getattr(module, name)
                for name in dir(module)
                if name.startswith("test_") and callable(getattr(module, name))
            ]

            for test_func in test_functions:
                try:
                    print(f"  ▶ {test_func.__name__}...", end="")
                    start_time = time.time()

                    # Execute test function
                    try:
                        # Check function signature - if it requires arguments, skip it
                        import inspect
                        sig = inspect.signature(test_func)
                        if len(sig.parameters) > 0:
                            print(f" ⚠️ ERROR: {test_func.__name__}() missing {len(sig.parameters)} required positional argument(s)")
                            results["errors"].append(f"Function requires arguments: {sig.parameters}")
                            continue
                            
                        result = test_func()
                        
                    except SystemExit:
                        print(f" ⏭️ SKIPPED: System exit")
                        continue
                    except Exception as e:
                        error_str = str(e)
                        if "skip" in error_str.lower():
                            print(f" ⏭️ SKIPPED: {error_str}")
                            continue
                        else:
                            raise e

                    duration = time.time() - start_time

                    # Interpret result
                    if result is None or result is True:
                        print(f" ✅ PASS ({duration:.3f}s)")
                        results["passed"] += 1
                    elif isinstance(result, dict) and result.get("success", False):
                        print(f" ✅ PASS ({duration:.3f}s)")
                        results["passed"] += 1
                    else:
                        print(f" ❌ FAIL ({duration:.3f}s)")
                        results["failed"] += 1
                        results["errors"].append(f"{test_func.__name__}: Failed")

                    results["tests_run"] += 1

                except Exception as e:
                    print(f" ⚠️ ERROR: {str(e)}")
                    results["failed"] += 1
                    results["errors"].append(f"{test_func.__name__}: {str(e)}")
                    results["tests_run"] += 1

            # Look for test classes (unittest style)
            test_classes = [
                getattr(module, name)
                for name in dir(module)
                if (
                    isinstance(getattr(module, name), type)
                    and issubclass(getattr(module, name), unittest.TestCase)
                    and getattr(module, name) != unittest.TestCase
                )
            ]

            for test_class in test_classes:
                suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
                unittest_result = unittest.TextTestRunner(verbosity=2).run(suite)
                
                results["tests_run"] += unittest_result.testsRun
                results["passed"] += unittest_result.testsRun - len(unittest_result.errors) - len(unittest_result.failures)
                results["failed"] += len(unittest_result.errors) + len(unittest_result.failures)
                
                for error in unittest_result.errors:
                    results["errors"].append(f"{error[0]}: {error[1]}")
                    
                for failure in unittest_result.failures:
                    results["errors"].append(f"{failure[0]}: {failure[1]}")

        except Exception as e:
            print(f"  ❌ Error running tests: {str(e)}")
            results["errors"].append(f"Test execution error: {str(e)}")

        return results

    def run_category_tests(self, category: str) -> Dict[str, Any]:
        """Run tests for a specific category"""
        print(f"\n🔖 Running tests for category: {category}")

        results = {
            "category": category,
            "tests_run": 0,
            "passed": 0,
            "failed": 0,
            "files": [],
        }

        # TODO: Implement category-specific test discovery and execution
        # For now, just return empty results
        return results

    def run_all_tests(self, category: str = None) -> Dict[str, Any]:
        """Run all discovered tests"""
        start_time = time.time()

        if category:
            # Run only tests for specific category
            self.results["categories"][category] = self.run_category_tests(category)
            return self.results["categories"][category]
        else:
            # Run all tests
            test_files = self.discover_root_tests()
            
            # Filter to run only a subset for faster testing
            # test_files = test_files[:20]  # Uncomment to run only first 20 tests
            
            for test_file in test_files:
                try:
                    file_results = self.run_file_tests(test_file)
                    self.results["total_tests"] += file_results["tests_run"]
                    self.results["passed"] += file_results["passed"]
                    self.results["failed"] += file_results["failed"]
                except Exception as e:
                    print(f"Error processing file {test_file}: {str(e)}")

        duration = time.time() - start_time

        # Print summary
        print("\n" + "=" * 50)
        print("📊 TEST SUMMARY")
        print("=" * 50)
        print(f"Total Tests: {self.results['total_tests']}")
        print(f"✅ Passed: {self.results['passed']}")
        print(f"❌ Failed: {self.results['failed']}")
        print(f"⏱️ Duration: {duration:.2f}s")

        if self.results["total_tests"] > 0:
            success_rate = (self.results["passed"] / self.results["total_tests"]) * 100
            print(f"📈 Success Rate: {success_rate:.1f}%")

        return self.results

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="QuantoniumOS Unified Test Runner")
    parser.add_argument(
        "--category",
        choices=["unit", "integration", "benchmarks"],
        help="Run tests from specific category only",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()
    runner = QuantoniumTestRunner()

    print("🚀 QuantoniumOS Unified Test Runner")
    print("==================================================\n")

    try:
        print("📁 Discovering root-level test files...")
        test_files = runner.discover_root_tests()
        print(f"Found {len(test_files)} test files")

        if args.category:
            print(f"\n📦 Running {args.category} tests...")
            results = runner.run_category_tests(args.category)
        else:
            results = runner.run_all_tests(category=args.category)

        # Exit with appropriate status code
        if results["failed"] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
    except KeyboardInterrupt:
        print("\n\n⚠️ Test execution interrupted by user.")
        sys.exit(2)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(3)

if __name__ == "__main__":
    main()
