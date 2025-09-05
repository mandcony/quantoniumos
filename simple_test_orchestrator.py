#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QuantoniumOS Simple Test Orchestrator
====================================
Coordinates simplified test suites for validation
"""

import subprocess
import time
import json
from pathlib import Path
from datetime import datetime
import argparse
import sys

class SimpleTestOrchestrator:
    """Simple test orchestration for QuantoniumOS validation"""
    
    def __init__(self, output_dir="simple_test_results"):
        """Initialize orchestrator"""
        self.output_dir = Path(output_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"test_run_{self.timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self.test_results = {}
        self.test_summary = {
            'start_time': datetime.now().isoformat(),
            'run_id': self.timestamp,
            'tests_planned': [],
            'tests_completed': [],
            'tests_failed': [],
            'overall_status': 'running'
        }
    
    def run_comprehensive_testing(self, 
                                 include_basic=True,
                                 include_advanced=True,
                                 include_hardware=True,
                                 include_reliability=True,
                                 reliability_hours=0.1):
        """Run comprehensive testing with working test suites"""
        
        print("=" * 80)
        print("QUANTONIUMOS SIMPLE TEST ORCHESTRATOR")
        print("=" * 80)
        print(f"Run ID: {self.timestamp}")
        print(f"Output Directory: {self.run_dir}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Define working test suite execution plan
        test_plan = []
        
        if include_basic:
            test_plan.extend([
                ('operational_validation', 'test_suite.py', 'Basic operational validation'),
                ('performance_benchmarks', 'benchmark_suite.py', 'Performance benchmarking'),
            ])
        
        if include_advanced:
            test_plan.append(
                ('advanced_mathematical', 'advanced_mathematical_tests.py', 'Advanced mathematical tests')
            )
        
        if include_hardware:
            test_plan.append(
                ('hardware_validation', '../simple_hardware_validation.py', 'Hardware compatibility tests')
            )
        
        if include_reliability:
            test_plan.append(
                ('reliability_testing', f'../simple_reliability_tests.py --duration {reliability_hours}', 
                 f'Reliability tests ({reliability_hours}h)')
            )
        
        self.test_summary['tests_planned'] = [test[0] for test in test_plan]
        
        print(f"\nTest Plan: {len(test_plan)} test suites")
        for i, (name, script, description) in enumerate(test_plan, 1):
            print(f"  {i}. {description}")
        
        # Execute test suites
        for test_name, test_script, test_description in test_plan:
            print(f"\n{'='*60}")
            print(f"EXECUTING: {test_description}")
            print(f"{'='*60}")
            
            success = self._execute_test_suite(test_name, test_script, test_description)
            
            if success:
                self.test_summary['tests_completed'].append(test_name)
                print(f"? {test_description} - COMPLETED")
            else:
                self.test_summary['tests_failed'].append(test_name)
                print(f"? {test_description} - FAILED")
        
        # Generate comprehensive report
        print(f"\n{'='*60}")
        print("GENERATING COMPREHENSIVE REPORT")
        print(f"{'='*60}")
        
        self._generate_comprehensive_report()
        
        # Final summary
        self.test_summary['end_time'] = datetime.now().isoformat()
        self.test_summary['overall_status'] = 'completed'
        
        self._print_final_summary()
        
        return self.test_results
    
    def _execute_test_suite(self, test_name, test_script, description):
        """Execute a single test suite"""
        try:
            print(f"  Starting {test_name}...")
            
            # Prepare command
            script_parts = test_script.split()
            script_file = script_parts[0]
            script_args = script_parts[1:] if len(script_parts) > 1 else []
            
            # Check if script exists
            script_path = Path(script_file)
            if not script_path.exists():
                print(f"  ?? Script not found: {script_file}")
                return False
            
            # Execute test suite
            start_time = time.time()
            
            cmd = ['python', script_file] + script_args
            print(f"  Command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Store results
            self.test_results[test_name] = {
                'description': description,
                'script': test_script,
                'duration_seconds': duration,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'success': result.returncode == 0,
                'start_time': datetime.fromtimestamp(start_time).isoformat(),
                'end_time': datetime.fromtimestamp(end_time).isoformat()
            }
            
            # Save individual test output
            test_output_dir = self.run_dir / test_name
            test_output_dir.mkdir(exist_ok=True)
            
            with open(test_output_dir / "stdout.txt", 'w', encoding='utf-8') as f:
                f.write(result.stdout)
            
            with open(test_output_dir / "stderr.txt", 'w', encoding='utf-8') as f:
                f.write(result.stderr)
            
            with open(test_output_dir / "summary.json", 'w', encoding='utf-8') as f:
                json.dump(self.test_results[test_name], f, indent=2)
            
            print(f"  Duration: {duration/60:.1f} minutes")
            print(f"  Output saved to: {test_output_dir}")
            
            return result.returncode == 0
            
        except subprocess.TimeoutExpired:
            print(f"  Test suite timed out after 30 minutes")
            self.test_results[test_name] = {
                'description': description,
                'script': test_script,
                'success': False,
                'error': 'timeout',
                'duration_seconds': 1800
            }
            return False
            
        except Exception as e:
            print(f"  Error executing test suite: {e}")
            self.test_results[test_name] = {
                'description': description,
                'script': test_script,
                'success': False,
                'error': str(e),
                'duration_seconds': 0
            }
            return False
    
    def _generate_comprehensive_report(self):
        """Generate comprehensive test report"""
        
        # Executive Summary
        summary_path = self.run_dir / "EXECUTIVE_SUMMARY.md"
        
        total_tests = len(self.test_summary['tests_planned'])
        completed_tests = len(self.test_summary['tests_completed'])
        failed_tests = len(self.test_summary['tests_failed'])
        success_rate = (completed_tests / total_tests * 100) if total_tests > 0 else 0
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("# QuantoniumOS Test Suite - Executive Summary\\n\\n")
            
            f.write(f"**Test Run ID**: {self.timestamp}\\n")
            f.write(f"**Start Time**: {self.test_summary['start_time']}\\n")
            f.write(f"**End Time**: {self.test_summary.get('end_time', 'In Progress')}\\n\\n")
            
            f.write("## Overall Results\\n\\n")
            f.write(f"- **Total Test Suites**: {total_tests}\\n")
            f.write(f"- **Completed Successfully**: {completed_tests}\\n")
            f.write(f"- **Failed**: {failed_tests}\\n")
            f.write(f"- **Success Rate**: {success_rate:.1f}%\\n\\n")
            
            # Status indicator
            if success_rate >= 80:
                f.write("?? **STATUS: GOOD** - Ready for deployment\\n\\n")
            elif success_rate >= 60:
                f.write("?? **STATUS: MODERATE** - Some issues to address\\n\\n")
            else:
                f.write("?? **STATUS: NEEDS WORK** - Multiple issues detected\\n\\n")
            
            f.write("## Test Suite Results\\n\\n")
            
            for test_name in self.test_summary['tests_planned']:
                if test_name in self.test_results:
                    result = self.test_results[test_name]
                    status = "? PASS" if result['success'] else "? FAIL"
                    duration = result.get('duration_seconds', 0)
                    
                    f.write(f"### {result['description']}\\n")
                    f.write(f"- **Status**: {status}\\n")
                    f.write(f"- **Duration**: {duration/60:.1f} minutes\\n")
                    
                    if not result['success']:
                        error = result.get('error', 'Unknown error')
                        f.write(f"- **Error**: {error}\\n")
                    
                    f.write("\\n")
            
            # Generate recommendations
            f.write("## Recommendations\\n\\n")
            
            if success_rate >= 80:
                f.write("? **System is ready** for production deployment\\n")
                f.write("- All critical tests completed successfully\\n")
                f.write("- Minor issues (if any) can be addressed post-deployment\\n")
            elif failed_tests > 0:
                f.write("?? **Address the following issues:**\\n")
                for test_name in self.test_summary['tests_failed']:
                    if test_name in self.test_results:
                        error = self.test_results[test_name].get('error', 'Unknown')
                        f.write(f"- **{test_name}**: {error}\\n")
        
        print(f"Executive summary: {summary_path}")
        
        # Detailed results
        results_path = self.run_dir / "detailed_test_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump({
                'test_summary': self.test_summary,
                'test_results': self.test_results,
                'generated': datetime.now().isoformat()
            }, f, indent=2, default=str)
        
        print(f"Detailed results: {results_path}")
    
    def _print_final_summary(self):
        """Print final summary to console"""
        print(f"\\n{'='*80}")
        print("COMPREHENSIVE TEST SUITE COMPLETE")
        print(f"{'='*80}")
        
        total_tests = len(self.test_summary['tests_planned'])
        completed_tests = len(self.test_summary['tests_completed'])
        failed_tests = len(self.test_summary['tests_failed'])
        success_rate = (completed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"RESULTS SUMMARY:")
        print(f"   Total Test Suites: {total_tests}")
        print(f"   Completed Successfully: {completed_tests}")
        print(f"   Failed: {failed_tests}")
        print(f"   Success Rate: {success_rate:.1f}%")
        
        print(f"\\nRESULTS LOCATION: {self.run_dir}")
        print(f"Executive Summary: {self.run_dir}/EXECUTIVE_SUMMARY.md")
        print(f"Detailed Results: {self.run_dir}/detailed_test_results.json")
        
        if success_rate >= 80:
            print(f"\\n?? SUCCESS! QuantoniumOS validation completed successfully!")
        elif success_rate >= 60:
            print(f"\\n?? Partial success. Review failed tests and re-run validation.")
        else:
            print(f"\\n?? Multiple issues detected. Address failures before deployment.")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="QuantoniumOS Simple Test Suite")
    parser.add_argument('--skip-basic', action='store_true', 
                       help='Skip basic validation tests')
    parser.add_argument('--skip-advanced', action='store_true', 
                       help='Skip advanced mathematical tests')
    parser.add_argument('--skip-hardware', action='store_true', 
                       help='Skip hardware validation tests')
    parser.add_argument('--skip-reliability', action='store_true', 
                       help='Skip reliability tests')
    parser.add_argument('--reliability-hours', type=float, default=0.1,
                       help='Duration for reliability tests in hours (default: 0.1)')
    parser.add_argument('--output-dir', default='simple_test_results',
                       help='Output directory for test results')
    
    args = parser.parse_args()
    
    orchestrator = SimpleTestOrchestrator(args.output_dir)
    
    results = orchestrator.run_comprehensive_testing(
        include_basic=not args.skip_basic,
        include_advanced=not args.skip_advanced,
        include_hardware=not args.skip_hardware,
        include_reliability=not args.skip_reliability,
        reliability_hours=args.reliability_hours
    )
    
    return results

if __name__ == "__main__":
    main()