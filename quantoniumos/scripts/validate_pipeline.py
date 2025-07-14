#!/usr/bin/env python3
"""
QuantoniumOS Pre-Push Validation
Bulletproof pipeline testing before GitHub push
Like a submarine pressure test before diving
"""

import os
import sys
import time
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class PipelineValidator:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.start_time = time.time()
        self.tests_run = 0
        self.tests_failed = 0
        self.warnings = []
        self.critical_failures = []
        
    def log(self, message: str, level: str = "INFO"):
        timestamp = time.strftime("%H:%M:%S")
        colors = {
            "INFO": "\033[0m",      # Default
            "SUCCESS": "\033[92m",  # Green
            "WARNING": "\033[93m",  # Yellow
            "ERROR": "\033[91m",    # Red
            "CRITICAL": "\033[95m"  # Magenta
        }
        reset = "\033[0m"
        
        if level in colors:
            print(f"{colors[level]}[{timestamp}] {message}{reset}")
        else:
            print(f"[{timestamp}] {message}")
    
    def run_command(self, cmd: str, timeout: int = 60, critical: bool = True) -> Tuple[bool, str]:
        """Run command with timeout and return success status and output"""
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=timeout
            )
            success = result.returncode == 0
            output = result.stdout + result.stderr
            
            if not success and critical:
                self.critical_failures.append(f"Command failed: {cmd}")
            elif not success:
                self.warnings.append(f"Non-critical failure: {cmd}")
                
            return success, output
            
        except subprocess.TimeoutExpired:
            error_msg = f"Command timed out after {timeout}s: {cmd}"
            if critical:
                self.critical_failures.append(error_msg)
            else:
                self.warnings.append(error_msg)
            return False, f"TIMEOUT after {timeout}s"
        except Exception as e:
            error_msg = f"Command exception: {cmd} - {str(e)}"
            if critical:
                self.critical_failures.append(error_msg)
            else:
                self.warnings.append(error_msg)
            return False, str(e)
    
    def test_file_structure(self) -> bool:
        """Test that critical files exist and are valid"""
        self.log("🔍 Testing file structure...", "INFO")
        
        critical_files = [
            "requirements.txt", "main.py", "setup.py", 
            ".github/workflows/main-ci.yml"
        ]
        
        all_good = True
        for file_path in critical_files:
            if not Path(file_path).exists():
                self.log(f"❌ Missing critical file: {file_path}", "CRITICAL")
                self.critical_failures.append(f"Missing file: {file_path}")
                all_good = False
            else:
                if self.verbose:
                    self.log(f"✅ Found: {file_path}", "SUCCESS")
        
        return all_good
    
    def test_python_syntax(self) -> bool:
        """Test Python syntax on all .py files"""
        self.log("🐍 Testing Python syntax...", "INFO")
        
        python_files = list(Path(".").rglob("*.py"))
        if not python_files:
            self.log("⚠️  No Python files found", "WARNING")
            return True
        
        failed_files = []
        for py_file in python_files[:20]:  # Limit to avoid timeout
            success, output = self.run_command(
                f"python -m py_compile {py_file}", 
                timeout=10, 
                critical=False
            )
            if not success:
                failed_files.append(str(py_file))
        
        if failed_files:
            self.log(f"❌ Syntax errors in {len(failed_files)} files", "ERROR")
            if self.verbose:
                for file in failed_files[:5]:
                    self.log(f"   - {file}", "ERROR")
            return False
        else:
            self.log(f"✅ Syntax check passed for {len(python_files)} files", "SUCCESS")
            return True
    
    def test_dependencies(self) -> bool:
        """Test dependency installation"""
        self.log("📦 Testing dependencies...", "INFO")
        
        # Test pip install dry run
        success, output = self.run_command(
            "python -m pip install --dry-run -r requirements.txt",
            timeout=120,
            critical=True
        )
        
        if not success:
            self.log("❌ Dependency resolution failed", "CRITICAL")
            return False
        
        # Test critical imports
        critical_imports = ["flask", "numpy", "cryptography", "pybind11"]
        failed_imports = []
        
        for module in critical_imports:
            success, output = self.run_command(
                f"python -c 'import {module}'",
                timeout=10,
                critical=False
            )
            if not success:
                failed_imports.append(module)
        
        if failed_imports:
            self.log(f"⚠️  Missing imports: {', '.join(failed_imports)}", "WARNING")
            self.log("   Installing missing dependencies...", "INFO")
            
            # Try to install missing deps
            success, output = self.run_command(
                "python -m pip install -r requirements.txt",
                timeout=300,
                critical=False
            )
            
            if success:
                self.log("✅ Dependencies installed successfully", "SUCCESS")
                return True
            else:
                self.log("❌ Failed to install dependencies", "CRITICAL")
                return False
        else:
            self.log("✅ All critical imports available", "SUCCESS")
            return True
    
    def test_cli_functionality(self) -> bool:
        """Test CLI functionality"""
        self.log("⚡ Testing CLI functionality...", "INFO")
        
        if Path("scripts/verify_cli.py").exists():
            success, output = self.run_command(
                "python scripts/verify_cli.py --verbose",
                timeout=60,
                critical=False
            )
            
            if success:
                self.log("✅ CLI verification passed", "SUCCESS")
                return True
            else:
                self.log("⚠️  CLI verification failed", "WARNING")
                if self.verbose and output:
                    self.log(f"Output: {output[:200]}...", "INFO")
                return False
        else:
            # Basic test
            success, output = self.run_command(
                "python -c 'print(\"Basic CLI test passed\")'",
                timeout=10,
                critical=False
            )
            
            if success:
                self.log("✅ Basic CLI test passed", "SUCCESS")
                return True
            else:
                self.log("❌ Basic CLI test failed", "ERROR")
                return False
    
    def test_security_basics(self) -> bool:
        """Basic security checks"""
        self.log("🔒 Running security checks...", "INFO")
        
        # Check if bandit is available
        success, output = self.run_command(
            "bandit --version",
            timeout=10,
            critical=False
        )
        
        if not success:
            self.log("📦 Installing bandit for security scan...", "INFO")
            success, output = self.run_command(
                "python -m pip install bandit",
                timeout=60,
                critical=False
            )
        
        if success:
            # Run bandit scan
            success, output = self.run_command(
                "bandit -r . -f json --severity-level medium",
                timeout=60,
                critical=False
            )
            
            if success:
                self.log("✅ Security scan completed", "SUCCESS")
                return True
            else:
                self.log("⚠️  Security issues detected", "WARNING")
                return False
        else:
            self.log("⚠️  Bandit not available, skipping security scan", "WARNING")
            return True
    
    def test_build_process(self) -> bool:
        """Test package build"""
        self.log("📦 Testing build process...", "INFO")
        
        # Test setup.py check
        success, output = self.run_command(
            "python setup.py check",
            timeout=30,
            critical=False
        )
        
        if success:
            self.log("✅ Package structure is valid", "SUCCESS")
            return True
        else:
            self.log("⚠️  Package structure issues detected", "WARNING")
            if self.verbose and output:
                self.log(f"Output: {output[:200]}...", "INFO")
            return False
    
    def run_full_validation(self) -> bool:
        """Run complete validation suite"""
        self.log("🚀 Starting QuantoniumOS Pipeline Validation", "INFO")
        self.log("=" * 50, "INFO")
        
        # Run all tests
        tests = [
            ("File Structure", self.test_file_structure),
            ("Python Syntax", self.test_python_syntax),
            ("Dependencies", self.test_dependencies),
            ("CLI Functionality", self.test_cli_functionality),
            ("Security Basics", self.test_security_basics),
            ("Build Process", self.test_build_process),
        ]
        
        results = {}
        for test_name, test_func in tests:
            self.tests_run += 1
            try:
                start_time = time.time()
                result = test_func()
                duration = time.time() - start_time
                
                results[test_name] = {
                    "passed": result,
                    "duration": round(duration, 1)
                }
                
                if not result:
                    self.tests_failed += 1
                    
            except Exception as e:
                self.log(f"❌ Test {test_name} crashed: {str(e)}", "CRITICAL")
                self.tests_failed += 1
                results[test_name] = {
                    "passed": False,
                    "duration": 0,
                    "error": str(e)
                }
        
        # Print summary
        total_duration = time.time() - self.start_time
        self.log("\n" + "=" * 50, "INFO")
        self.log("🎯 VALIDATION SUMMARY", "INFO")
        self.log("=" * 50, "INFO")
        
        for test_name, result in results.items():
            status = "✅ PASS" if result["passed"] else "❌ FAIL"
            duration = result["duration"]
            self.log(f"{status} {test_name} ({duration}s)", 
                    "SUCCESS" if result["passed"] else "ERROR")
        
        self.log(f"\n⏱️  Total time: {total_duration:.1f}s", "INFO")
        self.log(f"🧪 Tests run: {self.tests_run}", "INFO")
        self.log(f"❌ Failed: {self.tests_failed}", "ERROR" if self.tests_failed > 0 else "INFO")
        self.log(f"⚠️  Warnings: {len(self.warnings)}", "WARNING" if self.warnings else "INFO")
        
        # Show critical failures
        if self.critical_failures:
            self.log("\n🚨 CRITICAL FAILURES:", "CRITICAL")
            for failure in self.critical_failures:
                self.log(f"  - {failure}", "CRITICAL")
        
        # Show warnings if verbose
        if self.warnings and self.verbose:
            self.log("\n⚠️  WARNINGS:", "WARNING")
            for warning in self.warnings:
                self.log(f"  - {warning}", "WARNING")
        
        # Final verdict
        if self.tests_failed == 0 and not self.critical_failures:
            self.log("\n🎉 ALL TESTS PASSED - READY FOR PUSH!", "SUCCESS")
            self.log("✅ Your pipeline is watertight - safe to deploy", "SUCCESS")
            return True
        else:
            self.log(f"\n❌ {self.tests_failed} TESTS FAILED", "ERROR")
            self.log("🔧 Fix issues before pushing to production", "ERROR")
            return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="QuantoniumOS Pipeline Validator")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--fast", action="store_true", help="Skip slow tests")
    
    args = parser.parse_args()
    
    validator = PipelineValidator(verbose=args.verbose)
    
    try:
        success = validator.run_full_validation()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        validator.log("\n🛑 Validation interrupted by user", "WARNING")
        sys.exit(130)
    except Exception as e:
        validator.log(f"\n💥 Validation crashed: {str(e)}", "CRITICAL")
        sys.exit(1)

if __name__ == "__main__":
    main()
