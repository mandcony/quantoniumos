#!/usr/bin/env python3
"""
QUANTONIUM OS COMPREHENSIVE TEST RUNNER
======================================
Runs all tests in order and generates detailed reports
"""

import sys
import os
import time
import subprocess
import json
from pathlib import Path
from datetime import datetime
import traceback

# Add QuantoniumOS paths
sys.path.append('/workspaces/quantoniumos')
sys.path.append('/workspaces/quantoniumos/04_RFT_ALGORITHMS')
sys.path.append('/workspaces/quantoniumos/05_QUANTUM_ENGINES')
sys.path.append('/workspaces/quantoniumos/06_CRYPTOGRAPHY')

class QuantoniumTestRunner:
    """Comprehensive test runner for QuantoniumOS"""
    
    def __init__(self, base_path="/workspaces/quantoniumos/07_TESTS_BENCHMARKS"):
        self.base_path = Path(base_path)
        self.results = {}
        self.start_time = datetime.now()
        self.test_order = [
            "unit",           # Basic functionality first
            "cryptography",   # Core crypto validation
            "rft",           # RFT algorithms
            "quantum",       # Quantum engines
            "mathematical",  # Mathematical validation
            "scientific",    # Scientific compliance
            "performance",   # Performance benchmarks
            "system",        # System integration
            "integration",   # Full integration tests
            "utilities",     # Utility functions
            "benchmarks",    # Final benchmarks
            "legacy"         # Legacy compatibility
        ]
        
    def discover_test_files(self, category_path):
        """Discover all test files in a category"""
        test_files = []
        category_dir = self.base_path / category_path
        
        if not category_dir.exists():
            return test_files
            
        for file_path in category_dir.rglob("test_*.py"):
            if file_path.is_file():
                test_files.append(file_path)
                
        return sorted(test_files)
    
    def run_pytest_on_file(self, test_file):
        """Run pytest on a specific test file"""
        try:
            cmd = [
                sys.executable, "-m", "pytest", 
                str(test_file), 
                "-v", "--tb=short", "--no-header"
            ]
            
            start_time = time.time()
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=self.base_path.parent,
                timeout=300  # 5 minute timeout per file
            )
            end_time = time.time()
            
            return {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": end_time - start_time,
                "success": result.returncode == 0
            }
            
        except subprocess.TimeoutExpired:
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": "Test timed out after 5 minutes",
                "duration": 300,
                "success": False
            }
        except Exception as e:
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": str(e),
                "duration": 0,
                "success": False
            }
    
    def run_python_file_directly(self, test_file):
        """Run Python test file directly (fallback)"""
        try:
            cmd = [sys.executable, str(test_file)]
            
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.base_path.parent,
                timeout=300
            )
            end_time = time.time()
            
            return {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": end_time - start_time,
                "success": result.returncode == 0
            }
            
        except Exception as e:
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": str(e),
                "duration": 0,
                "success": False
            }
    
    def run_category_tests(self, category):
        """Run all tests in a category"""
        print(f"\n{'='*80}")
        print(f"🧪 TESTING CATEGORY: {category.upper()}")
        print(f"{'='*80}")
        
        test_files = self.discover_test_files(category)
        category_results = {
            "total_files": len(test_files),
            "passed_files": 0,
            "failed_files": 0,
            "file_results": {},
            "total_duration": 0,
            "start_time": datetime.now().isoformat()
        }
        
        if not test_files:
            print(f"📁 No test files found in {category}")
            category_results["status"] = "NO_TESTS"
            return category_results
        
        print(f"📁 Found {len(test_files)} test files in {category}")
        
        for i, test_file in enumerate(test_files, 1):
            relative_path = test_file.relative_to(self.base_path)
            print(f"\n🔍 [{i}/{len(test_files)}] Running: {relative_path}")
            
            # Try pytest first, then direct execution
            result = self.run_pytest_on_file(test_file)
            if not result["success"] and "not found" in result["stderr"].lower():
                print("   ⚠️  Pytest failed, trying direct execution...")
                result = self.run_python_file_directly(test_file)
            
            # Store results
            file_key = str(relative_path)
            category_results["file_results"][file_key] = result
            category_results["total_duration"] += result["duration"]
            
            if result["success"]:
                category_results["passed_files"] += 1
                print(f"   ✅ PASSED ({result['duration']:.2f}s)")
            else:
                category_results["failed_files"] += 1
                print(f"   ❌ FAILED ({result['duration']:.2f}s)")
                if result["stderr"]:
                    print(f"   💥 Error: {result['stderr'][:200]}...")
        
        # Category summary
        category_results["end_time"] = datetime.now().isoformat()
        category_results["status"] = "COMPLETED"
        
        print(f"\n📊 CATEGORY SUMMARY: {category}")
        print(f"   ✅ Passed: {category_results['passed_files']}")
        print(f"   ❌ Failed: {category_results['failed_files']}")
        print(f"   ⏱️  Duration: {category_results['total_duration']:.2f}s")
        
        return category_results
    
    def generate_report(self):
        """Generate comprehensive test report"""
        total_duration = (datetime.now() - self.start_time).total_seconds()
        
        # Calculate totals
        total_categories = len(self.results)
        total_files = sum(r.get("total_files", 0) for r in self.results.values())
        total_passed = sum(r.get("passed_files", 0) for r in self.results.values())
        total_failed = sum(r.get("failed_files", 0) for r in self.results.values())
        
        report = {
            "test_run_info": {
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "total_duration": total_duration,
                "test_runner_version": "1.0.0",
                "quantonium_os_path": str(self.base_path.parent)
            },
            "summary": {
                "total_categories": total_categories,
                "total_test_files": total_files,
                "total_passed": total_passed,
                "total_failed": total_failed,
                "success_rate": (total_passed / total_files * 100) if total_files > 0 else 0
            },
            "category_results": self.results,
            "recommendations": self.generate_recommendations()
        }
        
        return report
    
    def generate_recommendations(self):
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Check critical categories
        critical_categories = ["unit", "cryptography", "quantum", "rft"]
        for category in critical_categories:
            if category in self.results:
                result = self.results[category]
                if result.get("failed_files", 0) > 0:
                    recommendations.append(f"🚨 CRITICAL: Fix failures in {category} category")
        
        # Check overall success rate
        total_files = sum(r.get("total_files", 0) for r in self.results.values())
        total_passed = sum(r.get("passed_files", 0) for r in self.results.values())
        
        if total_files > 0:
            success_rate = total_passed / total_files * 100
            if success_rate < 80:
                recommendations.append(f"⚠️  Overall success rate is {success_rate:.1f}% - aim for >90%")
            elif success_rate >= 95:
                recommendations.append("🎯 Excellent test coverage and success rate!")
        
        # Check for missing categories
        expected_categories = ["unit", "quantum", "cryptography", "rft"]
        missing = [cat for cat in expected_categories if cat not in self.results]
        if missing:
            recommendations.append(f"📝 Consider adding tests for: {', '.join(missing)}")
        
        return recommendations
    
    def save_report(self, report, filename="quantonium_test_report.json"):
        """Save report to file"""
        report_path = self.base_path / filename
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        return report_path
    
    def print_final_report(self, report):
        """Print human-readable final report"""
        print(f"\n\n{'='*100}")
        print("🏆 QUANTONIUM OS COMPREHENSIVE TEST REPORT")
        print(f"{'='*100}")
        
        # Test run info
        summary = report["summary"]
        print(f"📊 OVERALL RESULTS:")
        print(f"   🗂️  Categories tested: {summary['total_categories']}")
        print(f"   📄 Total test files: {summary['total_test_files']}")
        print(f"   ✅ Passed: {summary['total_passed']}")
        print(f"   ❌ Failed: {summary['total_failed']}")
        print(f"   📈 Success rate: {summary['success_rate']:.1f}%")
        print(f"   ⏱️  Total duration: {report['test_run_info']['total_duration']:.1f}s")
        
        # Category breakdown
        print(f"\n📋 CATEGORY BREAKDOWN:")
        for category, result in report["category_results"].items():
            status_emoji = "✅" if result.get("failed_files", 0) == 0 else "❌"
            print(f"   {status_emoji} {category:15} | Files: {result.get('total_files', 0):2d} | "
                  f"Passed: {result.get('passed_files', 0):2d} | "
                  f"Failed: {result.get('failed_files', 0):2d} | "
                  f"Duration: {result.get('total_duration', 0):6.1f}s")
        
        # Recommendations
        if report["recommendations"]:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in report["recommendations"]:
                print(f"   {rec}")
        
        # Final status
        if summary["success_rate"] >= 90:
            print(f"\n🎉 EXCELLENT: QuantoniumOS test suite is in great shape!")
        elif summary["success_rate"] >= 70:
            print(f"\n👍 GOOD: QuantoniumOS test suite is mostly working, minor fixes needed.")
        else:
            print(f"\n⚠️  NEEDS ATTENTION: Several test failures need to be addressed.")
    
    def run_all_tests(self):
        """Run all tests in the defined order"""
        print("🚀 QUANTONIUM OS COMPREHENSIVE TEST SUITE")
        print(f"📁 Base path: {self.base_path}")
        print(f"🕒 Started at: {self.start_time}")
        print(f"📋 Test order: {' → '.join(self.test_order)}")
        
        # Run tests in order
        for category in self.test_order:
            try:
                self.results[category] = self.run_category_tests(category)
            except Exception as e:
                print(f"💥 CRITICAL ERROR in {category}: {e}")
                self.results[category] = {
                    "status": "ERROR",
                    "error": str(e),
                    "total_files": 0,
                    "passed_files": 0,
                    "failed_files": 0,
                    "total_duration": 0
                }
        
        # Generate and save report
        report = self.generate_report()
        report_path = self.save_report(report)
        
        # Print final report
        self.print_final_report(report)
        
        print(f"\n📄 Detailed report saved to: {report_path}")
        return report

def main():
    """Main function to run all tests"""
    runner = QuantoniumTestRunner()
    report = runner.run_all_tests()
    
    # Return exit code based on results
    success_rate = report["summary"]["success_rate"]
    if success_rate >= 90:
        return 0  # Success
    elif success_rate >= 70:
        return 1  # Warning
    else:
        return 2  # Error

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
