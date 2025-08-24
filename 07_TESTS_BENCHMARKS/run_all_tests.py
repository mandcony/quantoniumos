#!/usr/bin/env python3
"""
QuantoniumOS Unified Test Runner

Consolidates all test execution from scattered test files into a
unified, discoverable test suite.

Features:
- Automatic test discovery
- Categorized test execution  
- Results reporting
- Integration with existing test files

Usage:
    python tests/run_all_tests.py
    python tests/run_all_tests.py --category unit
    python tests/run_all_tests.py --category integration
    python tests/run_all_tests.py --category benchmarks
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

        for pattern in patterns:
            test_files.extend(self.project_root.glob(pattern))

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
            spec.loader.exec_module(module)

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
                    result = test_func()

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
                )
            ]

            for test_class in test_classes:
                suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
                runner = unittest.TextTestRunner(
                    verbosity=0, stream=open(os.devnull, "w")
                )
                result = runner.run(suite)

                results["tests_run"] += result.testsRun
                results["passed"] += (
                    result.testsRun - len(result.failures) - len(result.errors)
                )
                results["failed"] += len(result.failures) + len(result.errors)

                for failure in result.failures:
                    results["errors"].append(f"{failure[0]}: {failure[1]}")
                for error in result.errors:
                    results["errors"].append(f"{error[0]}: {error[1]}")

        except Exception as e:
            print(f"  ❌ ERROR loading module: {e}")
            results["errors"].append(f"Module load error: {e}")

        return results

    def run_category_tests(self, category: str) -> Dict[str, Any]:
        """Run tests from a specific category"""
        print(f"\n📂 Running {category} tests...")

        category_dir = self.project_root / "tests" / category
        if not category_dir.exists():
            print(f"  ⚠️ Category directory not found: {category_dir}")
            return {"tests_run": 0, "passed": 0, "failed": 0}

        test_files = list(category_dir.glob("test_*.py"))
        if not test_files:
            print(f"  ℹ️ No test files found in {category}")
            return {"tests_run": 0, "passed": 0, "failed": 0}

        category_results = {"tests_run": 0, "passed": 0, "failed": 0, "files": []}

        for test_file in test_files:
            file_results = self.run_file_tests(test_file)
            category_results["tests_run"] += file_results["tests_run"]
            category_results["passed"] += file_results["passed"]
            category_results["failed"] += file_results["failed"]
            category_results["files"].append(file_results)

        return category_results

    def run_all_tests(self, category: str = None) -> Dict[str, Any]:
        """Run all tests or tests from specific category"""
        print("🚀 QuantoniumOS Unified Test Runner")
        print("=" * 50)

        start_time = time.time()

        if category:
            # Run specific category
            category_results = self.run_category_tests(category)
            self.results["categories"][category] = category_results
            self.results["total_tests"] = category_results["tests_run"]
            self.results["passed"] = category_results["passed"]
            self.results["failed"] = category_results["failed"]
        else:
            # Run root-level tests
            print("\n📁 Discovering root-level test files...")
            root_tests = self.discover_root_tests()
            print(f"Found {len(root_tests)} test files")

            for test_file in root_tests:
                file_results = self.run_file_tests(test_file)
                self.results["total_tests"] += file_results["tests_run"]
                self.results["passed"] += file_results["passed"]
                self.results["failed"] += file_results["failed"]

            # Run categorized tests
            for cat in ["unit", "integration", "benchmarks"]:
                category_results = self.run_category_tests(cat)
                self.results["categories"][cat] = category_results
                self.results["total_tests"] += category_results["tests_run"]
                self.results["passed"] += category_results["passed"]
                self.results["failed"] += category_results["failed"]

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
    results = runner.run_all_tests(category=args.category)

    # Exit with error code if tests failed
    if results["failed"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
