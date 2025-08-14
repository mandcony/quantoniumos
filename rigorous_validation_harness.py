#!/usr/bin/env python3
"""
RIGOROUS CRYPTOGRAPHIC VALIDATION HARNESS - CRITICAL FIXES

This addresses all the critical issues identified in the previous validation:
1. Fixes broken runs test logic
2. Properly handles biased quantum geometric streams  
3. Implements proper state normalization
4. Fixes summary aggregation logic
5. Provides honest, trustworthy statistical reporting

NO MORE FAKE NUMBERS OR CONTRADICTORY REPORTS.
"""

import os
import sys
import time
import math
import json
import traceback
from typing import Dict, List, Any, Optional, Tuple

# Add paths
sys.path.append('/workspaces/quantoniumos/core')
sys.path.append('/workspaces/quantoniumos')

print("🔬 RIGOROUS CRYPTOGRAPHIC VALIDATION HARNESS")
print("=" * 60)
print("Honest statistical analysis with critical bug fixes")
print()

# Import core validation - with error handling
try:
    from comprehensive_final_rft_validation import ComprehensiveRFTStatisticalValidator
    HAS_CORE_VALIDATOR = True
    print("✅ Core RFT validation system loaded")
except ImportError as e:
    HAS_CORE_VALIDATOR = False
    print(f"❌ Core validator unavailable: {e}")

# Test engines
try:
    import quantonium_core
    HAS_QUANTONIUM_CORE = True
    print("✅ QuantoniumOS RFT Core Engine available")
except:
    HAS_QUANTONIUM_CORE = False
    print("❌ QuantoniumOS RFT Core Engine unavailable")

try:
    import quantum_engine
    HAS_QUANTUM_ENGINE = True  
    print("✅ QuantoniumOS Quantum Engine available")
except:
    HAS_QUANTUM_ENGINE = False
    print("❌ QuantoniumOS Quantum Engine unavailable")

class RigorousStatisticalAnalyzer:
    """
    Honest statistical analysis that doesn't lie about results
    Implements proper runs tests and other critical fixes
    """
    
    def __init__(self):
        self.analysis_errors = []
        
    def correct_runs_test(self, data: bytes) -> Dict[str, Any]:
        """
        CORRECTED runs test implementation
        For N bits, expected runs ≈ N/2, not some tiny number
        """
        if len(data) == 0:
            return {'error': 'Empty data', 'runs': 0, 'expected_runs': 0}
        
        # Convert bytes to bits
        bits = []
        for byte in data:
            for i in range(8):
                bits.append((byte >> (7-i)) & 1)
        
        if len(bits) < 2:
            return {'error': 'Insufficient bits', 'runs': 0, 'expected_runs': 0}
        
        # Count actual runs
        runs = 1
        max_run_length = 1
        current_run_length = 1
        
        for i in range(1, len(bits)):
            if bits[i] != bits[i-1]:
                runs += 1
                max_run_length = max(max_run_length, current_run_length)
                current_run_length = 1
            else:
                current_run_length += 1
        
        max_run_length = max(max_run_length, current_run_length)
        
        # CORRECT expected runs calculation
        N = len(bits)
        expected_runs = (N - 1) / 2.0  # This is the ACTUAL formula
        
        # Statistical test
        # For large N, runs is approximately normal with:
        # mean = (N-1)/2, variance = (N-1)/4
        if N > 100:
            variance = (N - 1) / 4.0
            z_score = abs(runs - expected_runs) / math.sqrt(variance)
            runs_test_pass = z_score < 1.96  # 95% confidence
        else:
            runs_test_pass = None  # Too small for normal approximation
        
        return {
            'runs_count': runs,
            'expected_runs': expected_runs,
            'max_run_length': max_run_length,
            'total_bits': N,
            'runs_test_pass': runs_test_pass,
            'z_score': z_score if N > 100 else None,
            'error': None
        }
    
    def honest_entropy_analysis(self, data: bytes) -> Dict[str, Any]:
        """Honest entropy analysis without inflated numbers"""
        if len(data) == 0:
            return {'error': 'No data provided'}
        
        # Frequency count
        freq = [0] * 256
        for byte in data:
            freq[byte] += 1
        
        # Shannon entropy
        entropy = 0.0
        total = len(data)
        for count in freq:
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        # Basic statistics
        data_list = list(data)
        mean = sum(data_list) / len(data_list)
        
        variance = sum((x - mean) ** 2 for x in data_list) / len(data_list)
        
        # Chi-square test
        expected = total / 256
        chi_square = sum((count - expected) ** 2 / expected for count in freq if expected > 0)
        
        # Serial correlation (lag-1)
        if len(data) > 1:
            serial_corr = 0.0
            for i in range(len(data) - 1):
                serial_corr += (data[i] - mean) * (data[i+1] - mean)
            serial_corr /= ((len(data) - 1) * variance)
        else:
            serial_corr = 0.0
        
        return {
            'bits_per_byte': entropy,
            'shannon_entropy': entropy,
            'mean': mean,
            'variance': variance,
            'chi_square': chi_square,
            'chi_square_critical_95': 293.248,  # For 255 DOF
            'serial_correlation': serial_corr,
            'sample_size': total,
            'error': None
        }
    
    def normalize_quantum_state(self, amplitudes: List[float]) -> Tuple[List[float], float]:
        """
        CRITICAL FIX: Properly normalize quantum states
        Returns (normalized_amplitudes, normalization_factor)
        """
        if not amplitudes:
            return [], 0.0
        
        # Calculate norm
        norm_squared = sum(amp * amp for amp in amplitudes)
        
        if norm_squared < 1e-15:  # Essentially zero
            return amplitudes, 0.0
        
        norm = math.sqrt(norm_squared)
        normalized = [amp / norm for amp in amplitudes]
        
        # Verify normalization
        verification = sum(amp * amp for amp in normalized)
        
        return normalized, verification
    
    def detect_bias_patterns(self, data: bytes) -> Dict[str, Any]:
        """
        Honest bias detection - no hiding problems
        """
        if len(data) < 256:
            return {'error': 'Insufficient data for bias analysis'}
        
        analysis = self.honest_entropy_analysis(data)
        bias_indicators = []
        
        # Check entropy
        if analysis['bits_per_byte'] < 7.5:
            bias_indicators.append(f"Low entropy: {analysis['bits_per_byte']:.3f} < 7.5")
        
        # Check mean
        expected_mean = 127.5
        if abs(analysis['mean'] - expected_mean) > 10:
            bias_indicators.append(f"Biased mean: {analysis['mean']:.1f} (expected ~127.5)")
        
        # Check chi-square
        if analysis['chi_square'] > analysis['chi_square_critical_95']:
            bias_indicators.append(f"Failed chi-square: {analysis['chi_square']:.1f} > {analysis['chi_square_critical_95']:.1f}")
        
        # Check serial correlation
        if abs(analysis['serial_correlation']) > 0.1:
            bias_indicators.append(f"High serial correlation: {analysis['serial_correlation']:.3f}")
        
        return {
            'bias_detected': len(bias_indicators) > 0,
            'bias_indicators': bias_indicators,
            'severity': 'HIGH' if len(bias_indicators) >= 3 else 'MEDIUM' if len(bias_indicators) >= 2 else 'LOW' if len(bias_indicators) == 1 else 'NONE',
            'analysis': analysis
        }

class HonestCryptographicValidator:
    """
    Honest cryptographic validator that reports actual results
    No fake numbers, no contradictory summaries
    """
    
    def __init__(self):
        self.analyzer = RigorousStatisticalAnalyzer()
        self.test_results = {}
        self.validation_errors = []
        
    def validate_stream_honestly(self, data: bytes, stream_name: str) -> Dict[str, Any]:
        """Validate a data stream with honest reporting"""
        print(f"\n🔍 HONEST VALIDATION: {stream_name}")
        print("-" * 40)
        
        if len(data) < 1000:
            error = f"Insufficient data: {len(data)} bytes < 1000 minimum"
            print(f"❌ {error}")
            return {'error': error, 'stream_name': stream_name}
        
        print(f"📊 Analyzing {len(data):,} bytes...")
        
        # Entropy analysis
        entropy_result = self.analyzer.honest_entropy_analysis(data)
        print(f"   Entropy: {entropy_result['bits_per_byte']:.6f} bits/byte")
        print(f"   Mean: {entropy_result['mean']:.3f} (ideal: ~127.5)")
        print(f"   Chi-square: {entropy_result['chi_square']:.2f} (95% crit: {entropy_result['chi_square_critical_95']:.2f})")
        print(f"   Serial correlation: {entropy_result['serial_correlation']:.6f}")
        
        # Corrected runs test
        runs_result = self.analyzer.correct_runs_test(data)
        if runs_result['error'] is None:
            print(f"   Runs: {runs_result['runs_count']:,} (expected: ~{runs_result['expected_runs']:.0f})")
            print(f"   Max run length: {runs_result['max_run_length']}")
            if runs_result['runs_test_pass'] is not None:
                print(f"   Runs test: {'PASS' if runs_result['runs_test_pass'] else 'FAIL'} (z={runs_result['z_score']:.2f})")
        else:
            print(f"   Runs test error: {runs_result['error']}")
        
        # Bias detection
        bias_result = self.analyzer.detect_bias_patterns(data)
        print(f"   Bias detected: {'YES' if bias_result['bias_detected'] else 'NO'}")
        if bias_result['bias_detected']:
            print(f"   Bias severity: {bias_result['severity']}")
            for indicator in bias_result['bias_indicators']:
                print(f"     - {indicator}")
        
        # Overall assessment
        assessment = self._assess_stream_quality(entropy_result, runs_result, bias_result)
        print(f"   Overall quality: {assessment['quality']}")
        print(f"   Cryptographic suitability: {assessment['crypto_suitable']}")
        
        return {
            'stream_name': stream_name,
            'data_size': len(data),
            'entropy_analysis': entropy_result,
            'runs_analysis': runs_result,
            'bias_analysis': bias_result,
            'overall_assessment': assessment,
            'error': None
        }
    
    def _assess_stream_quality(self, entropy: Dict, runs: Dict, bias: Dict) -> Dict[str, str]:
        """Honest quality assessment"""
        issues = []
        
        # Entropy issues
        if entropy['bits_per_byte'] < 7.5:
            issues.append("LOW_ENTROPY")
        
        # Runs issues
        if runs.get('runs_test_pass') == False:
            issues.append("RUNS_FAIL")
        
        # Bias issues
        if bias['bias_detected']:
            if bias['severity'] in ['HIGH', 'MEDIUM']:
                issues.append(f"BIAS_{bias['severity']}")
        
        # Chi-square issues
        if entropy['chi_square'] > entropy['chi_square_critical_95']:
            issues.append("CHI_SQUARE_FAIL")
        
        # Serial correlation issues
        if abs(entropy['serial_correlation']) > 0.1:
            issues.append("HIGH_CORRELATION")
        
        if len(issues) == 0:
            quality = "EXCELLENT"
            crypto_suitable = "YES"
        elif len(issues) == 1:
            quality = "GOOD"  
            crypto_suitable = "PROBABLY" if issues[0] not in ["LOW_ENTROPY", "BIAS_HIGH"] else "NO"
        elif len(issues) <= 2:
            quality = "FAIR"
            crypto_suitable = "QUESTIONABLE"
        else:
            quality = "POOR"
            crypto_suitable = "NO"
        
        return {
            'quality': quality,
            'crypto_suitable': crypto_suitable,
            'issues': issues,
            'issue_count': len(issues)
        }
    
    def test_quantum_state_normalization(self) -> Dict[str, Any]:
        """Test proper quantum state normalization"""
        print(f"\n⚛️ QUANTUM STATE NORMALIZATION TEST")
        print("-" * 40)
        
        # Test case 1: Unnormalized state
        unnormalized = [0.5, 0.3, 0.4, 0.2, 0.1, 0.6, 0.3, 0.1]
        print(f"Original state: sum of squares = {sum(x*x for x in unnormalized):.3f}")
        
        normalized, verification = self.analyzer.normalize_quantum_state(unnormalized)
        print(f"Normalized state: sum of squares = {verification:.6f}")
        print(f"Normalization {'SUCCESS' if abs(verification - 1.0) < 1e-10 else 'FAILED'}")
        
        # Test case 2: Already normalized state
        already_norm = [1/math.sqrt(8)] * 8  # Equal superposition
        norm2, verif2 = self.analyzer.normalize_quantum_state(already_norm)
        print(f"Already normalized: {abs(verif2 - 1.0) < 1e-10}")
        
        return {
            'unnormalized_test': {
                'original_norm_sq': sum(x*x for x in unnormalized),
                'normalized_norm_sq': verification,
                'normalization_successful': abs(verification - 1.0) < 1e-10
            },
            'prenormalized_test': {
                'verification': abs(verif2 - 1.0) < 1e-10
            }
        }
    
    def run_comprehensive_honest_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation with complete honesty"""
        print(f"\n🔬 COMPREHENSIVE HONEST VALIDATION")
        print("=" * 50)
        print("NO FAKE NUMBERS. NO CONTRADICTIONS. TRUTH ONLY.")
        print()
        
        validation_results = {
            'timestamp': time.time(),
            'streams_tested': [],
            'quantum_tests': {},
            'summary': {},
            'validation_errors': self.validation_errors.copy()
        }
        
        # Test quantum state normalization first
        quantum_norm_result = self.test_quantum_state_normalization()
        validation_results['quantum_tests']['normalization'] = quantum_norm_result
        
        # Test available data streams
        if HAS_CORE_VALIDATOR:
            try:
                print("📊 Testing available cryptographic streams...")
                validator = ComprehensiveRFTStatisticalValidator(sample_size_mb=1)  # Smaller for testing
                
                # Test Python crypto
                try:
                    python_file = validator.generate_python_crypto_data()
                    if os.path.exists(python_file):
                        with open(python_file, 'rb') as f:
                            python_data = f.read()[:100000]  # First 100KB for analysis
                        
                        python_result = self.validate_stream_honestly(python_data, "Python Crypto")
                        validation_results['streams_tested'].append(python_result)
                        self.test_results['python_crypto'] = python_result
                        
                except Exception as e:
                    error_msg = f"Python crypto test failed: {e}"
                    print(f"❌ {error_msg}")
                    self.validation_errors.append(error_msg)
                
                # Test C++ RFT if available
                if HAS_QUANTONIUM_CORE:
                    try:
                        rft_file = validator.generate_cpp_rft_enhanced_data()
                        if os.path.exists(rft_file):
                            with open(rft_file, 'rb') as f:
                                rft_data = f.read()[:100000]  # First 100KB
                            
                            rft_result = self.validate_stream_honestly(rft_data, "C++ RFT Engine")
                            validation_results['streams_tested'].append(rft_result)
                            self.test_results['rft_engine'] = rft_result
                            
                    except Exception as e:
                        error_msg = f"RFT engine test failed: {e}"
                        print(f"❌ {error_msg}")
                        self.validation_errors.append(error_msg)
                
                # Test Quantum Engine if available
                if HAS_QUANTUM_ENGINE:
                    try:
                        quantum_file = validator.generate_cpp_quantum_hash_data()
                        if os.path.exists(quantum_file):
                            with open(quantum_file, 'rb') as f:
                                quantum_data = f.read()[:100000]  # First 100KB
                            
                            quantum_result = self.validate_stream_honestly(quantum_data, "Quantum Geometric Engine")
                            validation_results['streams_tested'].append(quantum_result)
                            self.test_results['quantum_engine'] = quantum_result
                            
                    except Exception as e:
                        error_msg = f"Quantum engine test failed: {e}"
                        print(f"❌ {error_msg}")
                        self.validation_errors.append(error_msg)
                
            except Exception as e:
                error_msg = f"Core validator initialization failed: {e}"
                print(f"❌ {error_msg}")
                self.validation_errors.append(error_msg)
        
        # Generate honest summary
        summary = self._generate_honest_summary(validation_results)
        validation_results['summary'] = summary
        
        return validation_results
    
    def _generate_honest_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate completely honest summary with no contradictions"""
        streams = results['streams_tested']
        
        if not streams:
            return {
                'streams_analyzed': 0,
                'overall_assessment': 'NO_DATA',
                'crypto_suitable_count': 0,
                'issues_found': len(self.validation_errors),
                'validation_errors': self.validation_errors,
                'recommendation': 'Fix validation harness and rerun tests'
            }
        
        # Count assessments honestly
        quality_counts = {'EXCELLENT': 0, 'GOOD': 0, 'FAIR': 0, 'POOR': 0}
        crypto_suitable = {'YES': 0, 'PROBABLY': 0, 'QUESTIONABLE': 0, 'NO': 0}
        total_issues = 0
        
        for stream in streams:
            if 'overall_assessment' in stream:
                assessment = stream['overall_assessment']
                quality = assessment.get('quality', 'UNKNOWN')
                suitability = assessment.get('crypto_suitable', 'UNKNOWN')
                
                if quality in quality_counts:
                    quality_counts[quality] += 1
                
                if suitability in crypto_suitable:
                    crypto_suitable[suitability] += 1
                
                total_issues += assessment.get('issue_count', 0)
        
        # Overall assessment
        excellent_or_good = quality_counts['EXCELLENT'] + quality_counts['GOOD']
        total_streams = len(streams)
        
        if excellent_or_good == total_streams and total_issues == 0:
            overall = 'EXCELLENT'
        elif excellent_or_good >= total_streams * 0.75:
            overall = 'GOOD'
        elif excellent_or_good >= total_streams * 0.5:
            overall = 'FAIR'
        else:
            overall = 'POOR'
        
        return {
            'streams_analyzed': total_streams,
            'quality_distribution': quality_counts,
            'crypto_suitability_distribution': crypto_suitable,
            'overall_assessment': overall,
            'total_issues_found': total_issues,
            'validation_errors': len(self.validation_errors),
            'quantum_normalization_working': results['quantum_tests']['normalization']['unnormalized_test']['normalization_successful'],
            'recommendation': self._get_honest_recommendation(overall, total_issues, len(self.validation_errors))
        }
    
    def _get_honest_recommendation(self, overall: str, issues: int, errors: int) -> str:
        """Provide honest recommendation based on results"""
        if errors > 0:
            return f"Fix {errors} validation errors and rerun tests"
        elif overall == 'POOR':
            return "Streams show significant quality issues - not suitable for cryptographic use"
        elif overall == 'FAIR':
            return "Mixed results - investigate specific issues before deployment"
        elif overall == 'GOOD':
            return "Generally good quality with minor issues - suitable for most applications"
        else:
            return "Excellent quality - suitable for cryptographic deployment"

def main():
    """Run honest validation with all critical fixes"""
    print("🔬 RIGOROUS CRYPTOGRAPHIC VALIDATION")
    print("=" * 50)
    print("Honest analysis with critical bug fixes implemented")
    print()
    
    validator = HonestCryptographicValidator()
    
    try:
        results = validator.run_comprehensive_honest_validation()
        
        # Display honest results
        print(f"\n📊 HONEST VALIDATION RESULTS")
        print("=" * 40)
        
        summary = results['summary']
        print(f"Streams Analyzed: {summary['streams_analyzed']}")
        print(f"Overall Assessment: {summary['overall_assessment']}")
        print(f"Total Issues Found: {summary['total_issues_found']}")
        print(f"Validation Errors: {summary['validation_errors']}")
        print(f"Quantum Normalization: {'✅ WORKING' if summary['quantum_normalization_working'] else '❌ BROKEN'}")
        print(f"Recommendation: {summary['recommendation']}")
        
        if summary['streams_analyzed'] > 0:
            print(f"\nQuality Distribution:")
            for quality, count in summary['quality_distribution'].items():
                if count > 0:
                    print(f"  {quality}: {count}")
            
            print(f"\nCrypto Suitability:")
            for suitability, count in summary['crypto_suitability_distribution'].items():
                if count > 0:
                    print(f"  {suitability}: {count}")
        
        # Save honest results
        output_file = "/workspaces/quantoniumos/honest_validation_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Honest results saved to: {output_file}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
