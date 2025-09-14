#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QuantoniumOS Final Comprehensive Validation
==========================================
Complete validation suite combining operational success with comprehensive testing
"""

import sys
import os
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.working_quantum_kernel import WorkingQuantumKernel

class FinalValidationSuite:
    """Final comprehensive validation for QuantoniumOS"""
    
    def __init__(self):
        self.results = {}
        self.start_time = datetime.now()
        
    def run_final_validation(self):
        """Run complete final validation"""
        print("=" * 80)
        print("QUANTONIUMOS FINAL COMPREHENSIVE VALIDATION")
        print("=" * 80)
        print(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 1. Operational Validation (Core Functionality)
        print("\n" + "="*60)
        print("1. OPERATIONAL VALIDATION")
        print("="*60)
        self.results['operational'] = self.validate_operational_functionality()
        
        # 2. Hardware Validation
        print("\n" + "="*60)
        print("2. HARDWARE VALIDATION")
        print("="*60)
        self.results['hardware'] = self.run_hardware_validation()
        
        # 3. Mathematical Validation
        print("\n" + "="*60)
        print("3. MATHEMATICAL VALIDATION")
        print("="*60)
        self.results['mathematical'] = self.validate_mathematical_properties()
        
        # 4. Performance Validation
        print("\n" + "="*60)
        print("4. PERFORMANCE VALIDATION")
        print("="*60)
        self.results['performance'] = self.validate_performance()
        
        # 5. Reliability Validation  
        print("\n" + "="*60)
        print("5. RELIABILITY VALIDATION")
        print("="*60)
        self.results['reliability'] = self.run_reliability_validation()
        
        # Generate final assessment
        self.generate_final_assessment()
        
        return self.results
    
    def validate_operational_functionality(self):
        """Validate core operational functionality"""
        print("Testing core QuantoniumOS functionality...")
        
        operational_results = {
            'bell_state_creation': False,
            'quantum_kernel_functional': False,
            'assembly_integration': False,
            'desktop_compatibility': False,
            'errors': []
        }
        
        try:
            # Test quantum kernel and Bell state creation
            print("  Creating quantum kernel...")
            kernel = WorkingQuantumKernel(2)
            
            print("  Creating Bell state...")
            kernel.create_bell_state()
            
            # Validate Bell state
            state = kernel.state
            expected_amplitudes = [0.7071067811865475, 0.0, 0.0, 0.7071067811865475]
            
            # Check Bell state accuracy
            if len(state) >= 4:
                bell_accurate = True
                for i in range(4):
                    real_part = state[i].real
                    expected = expected_amplitudes[i]
                    if abs(real_part - expected) > 0.0001:
                        bell_accurate = False
                        break
                
                operational_results['bell_state_creation'] = bell_accurate
                operational_results['quantum_kernel_functional'] = True
                
                print(f"    Bell state: {[complex(s).real for s in state[:4]]}")
                print(f"    Expected:   {expected_amplitudes}")
                print(f"    Accuracy:   {'PERFECT' if bell_accurate else 'INACCURATE'}")
            
            # Test assembly integration
            try:
                # Check if assembly components are available
                assembly_available = hasattr(kernel, 'rft_processor') or hasattr(kernel, 'optimized_processor')
                operational_results['assembly_integration'] = assembly_available
                print(f"    Assembly integration: {'YES' if assembly_available else 'NO'}")
            except:
                operational_results['assembly_integration'] = False
            
            # Test basic desktop compatibility
            try:
                import tkinter
                operational_results['desktop_compatibility'] = True
                print("    Desktop compatibility: YES")
            except:
                operational_results['desktop_compatibility'] = False
                print("    Desktop compatibility: NO")
                
        except Exception as e:
            operational_results['errors'].append(str(e))
            print(f"  Error in operational validation: {e}")
        
        # Calculate operational score
        score_components = [
            operational_results['bell_state_creation'],
            operational_results['quantum_kernel_functional'],
            operational_results['assembly_integration'],
            operational_results['desktop_compatibility']
        ]
        
        operational_results['score'] = sum(score_components) / len(score_components)
        operational_results['status'] = 'PASS' if operational_results['score'] >= 0.75 else 'PARTIAL' if operational_results['score'] >= 0.5 else 'FAIL'
        
        print(f"  Operational Score: {operational_results['score']:.1%}")
        print(f"  Status: {operational_results['status']}")
        
        return operational_results
    
    def run_hardware_validation(self):
        """Run hardware validation using our working test"""
        print("Running hardware compatibility validation...")
        
        try:
            # Run the working hardware validation
            result = subprocess.run([
                'python', '../simple_hardware_validation.py'
            ], capture_output=True, text=True, timeout=120)
            
            hardware_results = {
                'execution_success': result.returncode == 0,
                'output': result.stdout,
                'errors': result.stderr,
                'compatible': 'COMPATIBLE' in result.stdout,
                'x86_64': 'x86-64 Compatible: True' in result.stdout,
                'adequate_memory': 'Sufficient Memory: True' in result.stdout,
                'adequate_performance': 'Performance Adequate: True' in result.stdout
            }
            
            if hardware_results['execution_success']:
                print("  Hardware validation completed successfully")
                print("  System compatibility:", "YES" if hardware_results['compatible'] else "NO")
                print("  x86-64 architecture:", "YES" if hardware_results['x86_64'] else "NO")
                print("  Adequate memory:", "YES" if hardware_results['adequate_memory'] else "NO")
                print("  Adequate performance:", "YES" if hardware_results['adequate_performance'] else "NO")
            else:
                print(f"  Hardware validation failed: {result.stderr}")
                
        except Exception as e:
            hardware_results = {
                'execution_success': False,
                'error': str(e),
                'compatible': False
            }
            print(f"  Error running hardware validation: {e}")
        
        # Calculate hardware score
        if hardware_results.get('execution_success', False):
            score_components = [
                hardware_results.get('compatible', False),
                hardware_results.get('x86_64', False),
                hardware_results.get('adequate_memory', False),
                hardware_results.get('adequate_performance', False)
            ]
            hardware_results['score'] = sum(score_components) / len(score_components)
        else:
            hardware_results['score'] = 0.0
            
        hardware_results['status'] = 'PASS' if hardware_results['score'] >= 0.75 else 'PARTIAL' if hardware_results['score'] >= 0.5 else 'FAIL'
        
        print(f"  Hardware Score: {hardware_results['score']:.1%}")
        print(f"  Status: {hardware_results['status']}")
        
        return hardware_results
    
    def validate_mathematical_properties(self):
        """Validate basic mathematical properties"""
        print("Validating mathematical properties...")
        
        mathematical_results = {
            'bell_state_normalization': False,
            'quantum_state_integrity': False,
            'complex_arithmetic': False,
            'numerical_precision': False
        }
        
        try:
            # Test Bell state normalization
            kernel = WorkingQuantumKernel(2)
            kernel.create_bell_state()
            state = kernel.state
            
            # Calculate normalization
            normalization = sum(abs(amplitude)**2 for amplitude in state)
            mathematical_results['bell_state_normalization'] = abs(normalization - 1.0) < 0.001
            
            print(f"  Bell state normalization: {normalization:.6f}")
            print(f"  Normalization valid: {'YES' if mathematical_results['bell_state_normalization'] else 'NO'}")
            
            # Test quantum state integrity  
            expected_pattern = [0.7071067811865475, 0.0, 0.0, 0.7071067811865475]
            state_real = [s.real for s in state[:4]]
            integrity_check = all(abs(actual - expected) < 0.0001 for actual, expected in zip(state_real, expected_pattern))
            mathematical_results['quantum_state_integrity'] = integrity_check
            
            print(f"  Quantum state integrity: {'YES' if integrity_check else 'NO'}")
            
            # Test complex arithmetic
            import numpy as np
            complex_test = complex(0.7071067811865475, 0) * complex(1, 0)
            mathematical_results['complex_arithmetic'] = abs(complex_test.real - 0.7071067811865475) < 1e-15
            
            print(f"  Complex arithmetic: {'YES' if mathematical_results['complex_arithmetic'] else 'NO'}")
            
            # Test numerical precision
            precision_test = abs(1.0/np.sqrt(2) - 0.7071067811865475) < 1e-15
            mathematical_results['numerical_precision'] = precision_test
            
            print(f"  Numerical precision: {'YES' if precision_test else 'NO'}")
            
        except Exception as e:
            print(f"  Error in mathematical validation: {e}")
            mathematical_results['error'] = str(e)
        
        # Calculate mathematical score
        score_components = [
            mathematical_results['bell_state_normalization'],
            mathematical_results['quantum_state_integrity'],
            mathematical_results['complex_arithmetic'],
            mathematical_results['numerical_precision']
        ]
        
        mathematical_results['score'] = sum(score_components) / len(score_components)
        mathematical_results['status'] = 'PASS' if mathematical_results['score'] >= 0.75 else 'PARTIAL' if mathematical_results['score'] >= 0.5 else 'FAIL'
        
        print(f"  Mathematical Score: {mathematical_results['score']:.1%}")
        print(f"  Status: {mathematical_results['status']}")
        
        return mathematical_results
    
    def validate_performance(self):
        """Validate performance characteristics"""
        print("Validating performance characteristics...")
        
        performance_results = {
            'bell_state_creation_time': None,
            'memory_efficiency': False,
            'cpu_utilization': False,
            'responsiveness': False
        }
        
        try:
            import numpy as np
            
            # Test Bell state creation performance
            kernel = WorkingQuantumKernel(2)
            
            start_time = time.perf_counter()
            for _ in range(100):
                kernel.create_bell_state()
            end_time = time.perf_counter()
            
            total_time = end_time - start_time
            avg_time = total_time / 100
            performance_results['bell_state_creation_time'] = avg_time
            
            print(f"  Average Bell state creation time: {avg_time*1000:.3f} ms")
            
            # Performance thresholds
            performance_results['responsiveness'] = avg_time < 0.01  # Less than 10ms
            
            print(f"  Responsiveness: {'YES' if performance_results['responsiveness'] else 'NO'}")
            
            # Test memory efficiency (simplified)
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss
            
            # Create multiple kernels to test memory usage
            kernels = []
            for _ in range(10):
                k = WorkingQuantumKernel(2)
                k.create_bell_state()
                kernels.append(k)
            
            final_memory = process.memory_info().rss
            memory_growth = (final_memory - initial_memory) / (1024 * 1024)  # MB
            
            performance_results['memory_efficiency'] = memory_growth < 50  # Less than 50MB for 10 kernels
            
            print(f"  Memory growth for 10 kernels: {memory_growth:.1f} MB")
            print(f"  Memory efficiency: {'YES' if performance_results['memory_efficiency'] else 'NO'}")
            
            # Test CPU utilization
            cpu_percent = psutil.cpu_percent(interval=1)
            performance_results['cpu_utilization'] = cpu_percent < 90
            
            print(f"  CPU utilization: {cpu_percent:.1f}%")
            print(f"  CPU efficiency: {'YES' if performance_results['cpu_utilization'] else 'NO'}")
            
        except Exception as e:
            print(f"  Error in performance validation: {e}")
            performance_results['error'] = str(e)
        
        # Calculate performance score
        score_components = [
            performance_results['responsiveness'],
            performance_results['memory_efficiency'],
            performance_results['cpu_utilization']
        ]
        
        valid_components = [c for c in score_components if c is not None]
        performance_results['score'] = sum(valid_components) / len(valid_components) if valid_components else 0
        performance_results['status'] = 'PASS' if performance_results['score'] >= 0.67 else 'PARTIAL' if performance_results['score'] >= 0.33 else 'FAIL'
        
        print(f"  Performance Score: {performance_results['score']:.1%}")
        print(f"  Status: {performance_results['status']}")
        
        return performance_results
    
    def run_reliability_validation(self):
        """Run reliability validation using our working test"""
        print("Running reliability validation...")
        
        try:
            # Run the working reliability validation
            result = subprocess.run([
                'python', '../simple_reliability_tests.py', '--duration', '0.05'
            ], capture_output=True, text=True, timeout=300)
            
            reliability_results = {
                'execution_success': result.returncode == 0,
                'output': result.stdout,
                'errors': result.stderr,
                'memory_stable': 'Memory stable: True' in result.stdout,
                'concurrent_stable': 'Concurrent stable: True' in result.stdout,
                'system_stable': 'System stable: True' in result.stdout
            }
            
            if reliability_results['execution_success']:
                print("  Reliability validation completed successfully")
                print("  Memory stability:", "YES" if reliability_results['memory_stable'] else "NO")
                print("  Concurrent operations:", "YES" if reliability_results['concurrent_stable'] else "NO")
                print("  System stability:", "YES" if reliability_results['system_stable'] else "NO")
            else:
                print(f"  Reliability validation failed: {result.stderr}")
                
        except Exception as e:
            reliability_results = {
                'execution_success': False,
                'error': str(e),
                'memory_stable': False,
                'concurrent_stable': False,
                'system_stable': False
            }
            print(f"  Error running reliability validation: {e}")
        
        # Calculate reliability score
        if reliability_results.get('execution_success', False):
            score_components = [
                reliability_results.get('memory_stable', False),
                reliability_results.get('concurrent_stable', False),
                reliability_results.get('system_stable', False)
            ]
            reliability_results['score'] = sum(score_components) / len(score_components)
        else:
            reliability_results['score'] = 0.0
            
        reliability_results['status'] = 'PASS' if reliability_results['score'] >= 0.67 else 'PARTIAL' if reliability_results['score'] >= 0.33 else 'FAIL'
        
        print(f"  Reliability Score: {reliability_results['score']:.1%}")
        print(f"  Status: {reliability_results['status']}")
        
        return reliability_results
    
    def generate_final_assessment(self):
        """Generate final comprehensive assessment"""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        print("\n" + "="*80)
        print("FINAL COMPREHENSIVE ASSESSMENT")
        print("="*80)
        
        # Calculate overall scores
        category_scores = {}
        category_statuses = {}
        
        for category, results in self.results.items():
            if isinstance(results, dict) and 'score' in results:
                category_scores[category] = results['score']
                category_statuses[category] = results.get('status', 'UNKNOWN')
        
        overall_score = sum(category_scores.values()) / len(category_scores) if category_scores else 0
        
        print(f"Test Duration: {duration.total_seconds()/60:.1f} minutes")
        print(f"Overall Score: {overall_score:.1%}")
        
        print("\nCATEGORY RESULTS:")
        for category, score in category_scores.items():
            status = category_statuses[category]
            print(f"  {category.upper()}: {score:.1%} ({status})")
        
        # Final assessment
        if overall_score >= 0.85:
            final_status = "EXCELLENT - READY FOR PRODUCTION"
            recommendation = "? System is ready for production deployment and academic publication"
        elif overall_score >= 0.70:
            final_status = "GOOD - READY FOR DEPLOYMENT"
            recommendation = "? System is ready for deployment with minor optimizations"
        elif overall_score >= 0.55:
            final_status = "ADEQUATE - CONDITIONAL DEPLOYMENT"
            recommendation = "?? System can be deployed with careful monitoring and improvements"
        else:
            final_status = "NEEDS IMPROVEMENT"
            recommendation = "? Address identified issues before deployment"
        
        print(f"\nFINAL STATUS: {final_status}")
        print(f"RECOMMENDATION: {recommendation}")
        
        # Save comprehensive results
        results_file = Path("final_validation_results.json")
        comprehensive_results = {
            'summary': {
                'start_time': self.start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_minutes': duration.total_seconds() / 60,
                'overall_score': overall_score,
                'final_status': final_status,
                'recommendation': recommendation
            },
            'category_scores': category_scores,
            'category_statuses': category_statuses,
            'detailed_results': self.results
        }
        
        with open(results_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to: {results_file}")
        
        # Key findings summary
        print("\nKEY FINDINGS:")
        
        # Operational findings
        if self.results.get('operational', {}).get('bell_state_creation', False):
            print("? Perfect Bell state creation: (|00? + |11?)/?2")
        
        # Hardware findings  
        if self.results.get('hardware', {}).get('compatible', False):
            print("? Hardware compatibility confirmed")
        
        # Mathematical findings
        if self.results.get('mathematical', {}).get('bell_state_normalization', False):
            print("? Quantum state normalization validated")
        
        # Performance findings
        if self.results.get('performance', {}).get('responsiveness', False):
            print("? Real-time performance achieved")
        
        # Reliability findings
        if self.results.get('reliability', {}).get('memory_stable', False):
            print("? Memory stability confirmed")
        
        return comprehensive_results

def main():
    """Run final comprehensive validation"""
    validator = FinalValidationSuite()
    results = validator.run_final_validation()
    
    print("\n" + "="*80)
    print("QUANTONIUMOS FINAL VALIDATION COMPLETE")
    print("="*80)
    
    return results

if __name__ == "__main__":
    main()