"""
QuantoniumOS - Side-Channel & Constant-Time Audit

This module provides tools for auditing the C++ core of QuantoniumOS for
side-channel vulnerabilities and ensuring constant-time execution of
security-critical operations.

The module includes static analysis, runtime verification, and measurement
tools for detecting timing, cache, power, and electromagnetic side-channel
leaks in the C++ implementation.
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import json
import time
import ctypes
from datetime import datetime

class CppSourceFile:
    """Class representing aC++ source file for analysis"""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.content = None
        self.functions = {}
        self.security_critical_functions = set()
        self.branching_on_secret = []
        self.memory_access_on_secret = []

    def load(self):
        """Load and parse the source file"""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.content = f.read()

        self._parse_functions()
        return self

    def _parse_functions(self):
        """Parse functions from the source file"""
        # This is a simplified function parser for demonstration
        # A real implementation would use a proper C++ parser

        # Simple regex to match function definitions
        function_pattern = r'(\w+(?:\s*::\s*\w+)?)\s+(\w+)\s*\(([^)]*)\)\s*(?:const|noexcept|override|final)?\s*(?:{\s*|$)'

        matches = re.finditer(function_pattern, self.content)

        for match in matches:
            return_type = match.group(1)
            function_name = match.group(2)
            parameters = match.group(3)

            start_pos = match.end()
            # Find the matching closing brace (naive implementation)
            brace_count = 1
            end_pos = start_pos

            while brace_count > 0 and end_pos < len(self.content):
                if self.content[end_pos] == '{':
                    brace_count += 1
                elif self.content[end_pos] == '}':
                    brace_count -= 1
                end_pos += 1

            function_body = self.content[start_pos:end_pos]

            self.functions[function_name] = {
                'name': function_name,
                'return_type': return_type,
                'parameters': parameters,
                'body': function_body,
                'start_pos': start_pos,
                'end_pos': end_pos
            }

    def mark_security_critical(self, function_names: List[str]):
        """Mark functions as security-critical"""
        for name in function_names:
            if name in self.functions:
                self.security_critical_functions.add(name)

    def analyze_constant_time(self):
        """Analyze functions for constant-time violations"""
        for name in self.security_critical_functions:
            if name not in self.functions:
                continue

            function = self.functions[name]
            body = function['body']

            # Check for branching on secret data
            # This is a simplified check - a real implementation would be more sophisticated
            if re.search(r'if\s*\(.*(secret|key|private|password|token).*\)', body, re.IGNORECASE):
                self.branching_on_secret.append({
                    'function': name,
                    'type': 'branch_on_secret',
                    'snippet': self._find_snippet(body, r'if\s*\(.*(secret|key|private|password|token).*\)')
                })

            # Check for variable-time operations
            if re.search(r'(^|\W)(strlen|strcmp|memcmp|strncmp)(\W|$)', body):
                self.branching_on_secret.append({
                    'function': name,
                    'type': 'variable_time_operation',
                    'snippet': self._find_snippet(body, r'(^|\W)(strlen|strcmp|memcmp|strncmp)(\W|$)')
                })

            # Check for memory access patterns that depend on secrets
            if re.search(r'(secret|key|private|password|token)\w*\s*\[\s*\w+\s*\]', body, re.IGNORECASE):
                self.memory_access_on_secret.append({
                    'function': name,
                    'type': 'secret_dependent_memory_access',
                    'snippet': self._find_snippet(body, r'(secret|key|private|password|token)\w*\s*\[\s*\w+\s*\]')
                })

    def _find_snippet(self, body: str, pattern: str) -> str:
        """Extract code snippet containing the pattern"""
        match = re.search(pattern, body)
        if not match:
            return ""

        start = max(0, match.start() - 50)
        end = min(len(body), match.end() + 50)

        # Try to find line boundaries
        while start > 0 and body[start] != '\n':
            start -= 1

        while end < len(body) - 1 and body[end] != '\n':
            end += 1

        return body[start:end].strip()

    def get_violations(self) -> Dict:
        """Get all constant-time violations"""
        return {
            'file': self.file_path,
            'branching_on_secret': self.branching_on_secret,
            'memory_access_on_secret': self.memory_access_on_secret,
            'total_violations': len(self.branching_on_secret) + len(self.memory_access_on_secret)
        }

class StaticAnalyzer:
    """Static analyzer for side-channel vulnerabilities"""

    def __init__(self, cpp_directory: str):
        self.cpp_directory = cpp_directory
        self.cpp_files = []
        self.security_critical_patterns = [
            r'crypt',
            r'encrypt',
            r'decrypt',
            r'hash',
            r'sign',
            r'verify',
            r'mac',
            r'auth',
            r'password',
            r'secret',
            r'key',
            r'token'
        ]

    def scan_directory(self):
        """Scan directory for C++ files"""
        for root, _, files in os.walk(self.cpp_directory):
            for file in files:
                if file.endswith(('.cpp', '.cc', '.cxx', '.h', '.hpp')):
                    file_path = os.path.join(root, file)
                    self.cpp_files.append(file_path)

    def identify_security_critical_functions(self, cpp_file: CppSourceFile) -> List[str]:
        """Identify security-critical functions in a file"""
        security_critical = []

        for name, function in cpp_file.functions.items():
            # Check if function name or body contains security-critical patterns
            for pattern in self.security_critical_patterns:
                if re.search(pattern, name, re.IGNORECASE) or \
                   re.search(pattern, function['body'], re.IGNORECASE):
                    security_critical.append(name)
                    break

        return security_critical

    def analyze_files(self) -> Dict:
        """Analyze all C++ files for side-channel vulnerabilities"""
        results = {
            'total_files': len(self.cpp_files),
            'files_with_violations': 0,
            'total_violations': 0,
            'violations_by_type': {},
            'violations_by_file': {},
            'security_critical_functions': 0
        }

        for file_path in self.cpp_files:
            try:
                cpp_file = CppSourceFile(file_path).load()

                # Identify security-critical functions
                security_critical = self.identify_security_critical_functions(cpp_file)
                cpp_file.mark_security_critical(security_critical)
                results['security_critical_functions'] += len(security_critical)

                # Analyze for constant-time violations
                cpp_file.analyze_constant_time()

                violations = cpp_file.get_violations()
                if violations['total_violations'] > 0:
                    results['files_with_violations'] += 1
                    results['total_violations'] += violations['total_violations']

                    # Track violations by file
                    rel_path = os.path.relpath(file_path, self.cpp_directory)
                    results['violations_by_file'][rel_path] = violations

                    # Track violations by type
                    for violation in violations['branching_on_secret']:
                        vtype = violation['type']
                        results['violations_by_type'][vtype] = results['violations_by_type'].get(vtype, 0) + 1

                    for violation in violations['memory_access_on_secret']:
                        vtype = violation['type']
                        results['violations_by_type'][vtype] = results['violations_by_type'].get(vtype, 0) + 1

            except Exception as e:
                print(f"Error analyzing {file_path}: {e}")

        return results

class RuntimeVerifier:
    """Runtime verification of constant-time execution"""

    def __init__(self, lib_path: str):
        self.lib_path = lib_path
        self.lib = None

        # Try to load the library
        try:
            self.lib = ctypes.CDLL(lib_path)
            print(f"Successfully loaded library: {lib_path}")
        except Exception as e:
            print(f"Failed to load library: {e}")

    def time_function(self, function_name: str, input_variations: List[bytes],
                     num_trials: int = 1000) -> Dict:
        """
        Time a function with different inputs to detect timing side-channels

        Args:
            function_name: Name of the function to test
            input_variations: List of input byte arrays to test
            num_trials: Number of trials for each input

        Returns:
            Dictionary of timing results
        """
        if not self.lib:
            return {"error": "Library not loaded"}

        try:
            # Get function from library
            func = getattr(self.lib, function_name)

            # Prepare result containers
            timing_results = []

            for input_data in input_variations:
                trials = []
                for _ in range(num_trials):
                    # Create ctypes buffer for input
                    input_buffer = ctypes.create_string_buffer(input_data)
                    input_size = len(input_data)

                    # Time the function call
                    start = time.perf_counter_ns()
                    func(input_buffer, input_size)
                    end = time.perf_counter_ns()

                    # Store execution time in nanoseconds
                    trials.append(end - start)

                # Calculate statistics
                timing_results.append({
                    'input_size': len(input_data),
                    'input_prefix': input_data[:10].hex() + '...' if len(input_data) > 10 else input_data.hex(),
                    'min_time': min(trials),
                    'max_time': max(trials),
                    'mean_time': np.mean(trials),
                    'median_time': np.median(trials),
                    'std_dev': np.std(trials),
                    'trials': trials
                })

            return {
                'function': function_name,
                'num_trials': num_trials,
                'timing_results': timing_results
            }

        except Exception as e:
            return {"error": f"Error timing function {function_name}: {e}"}

    def detect_timing_leak(self, timing_results: Dict) -> Dict:
        """
        Analyze timing results to detect potential leaks

        Args:
            timing_results: Results from time_function

        Returns:
            Dictionary with leak detection results
        """
        if 'error' in timing_results:
            return {'error': timing_results['error'], 'leak_detected': None}

        timing_data = timing_results['timing_results']
        function_name = timing_results['function']

        # Calculate statistics
        means = [data['mean_time'] for data in timing_data]
        std_devs = [data['std_dev'] for data in timing_data]

        # Calculate coefficient of variation (CV) for each input
        cvs = [std_dev / mean if mean > 0 else 0 for mean, std_dev in zip(means, std_devs)]

        # Check if mean execution times vary significantly
        # Use t-test for each pair of timing distributions
        from scipy import stats
        t_tests = []

        for i in range(len(timing_data)):
            for j in range(i + 1, len(timing_data)):
                trials_i = timing_data[i]['trials']
                trials_j = timing_data[j]['trials']

                t_stat, p_value = stats.ttest_ind(trials_i, trials_j)

                # Record the t-test results
                t_tests.append({
                    'input_i': timing_data[i]['input_prefix'],
                    'input_j': timing_data[j]['input_prefix'],
                    't_stat': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.01  # Significance threshold
                })

        # Determine if there'sa likely timing leak significant_tests = sum(1 for test in t_tests if test['significant']) leak_likelihood = significant_tests / len(t_tests) if t_tests else 0 leak_detected = leak_likelihood > 0.3 # More than 30% of tests show significant differences return { 'function': function_name, 'leak_detected': leak_detected, 'leak_likelihood': leak_likelihood, 'significant_tests': significant_tests, 'total_tests': len(t_tests), 'coefficient_variation': cvs, 'detailed_tests': t_tests } def plot_timing_distributions(self, timing_results: Dict, output_file: str = None): """ Plot timing distributions to visualize potential leaks Args: timing_results: Results from time_function output_file: Path to save the plot image """ if 'error' in timing_results: print(f"Cannot plot: {timing_results['error']}") return timing_data = timing_results['timing_results'] function_name = timing_results['function'] plt.figure(figsize=(12, 8)) # Plot histogram for each input variation for i, data in enumerate(timing_data): trials = data['trials'] plt.hist(trials, bins=30, alpha=0.5, label=f"Input {i+1}: {data['input_prefix']}") plt.title(f"Timing Distribution: {function_name}") plt.xlabel("Execution Time (ns)") plt.ylabel("Frequency") plt.legend() plt.grid(True, alpha=0.3) if output_file: plt.savefig(output_file) plt.close() else: plt.show() class CacheSideChannelAnalyzer: """Analyzer for cache-based side-channel vulnerabilities""" def __init__(self, lib_path: str): self.lib_path = lib_path self.lib = None try: self.lib = ctypes.CDLL(lib_path) except Exception as e: print(f"Failed to load library: {e}") def flush_cache(self): """Flush the CPU cache before measurement""" # This is a simplistic approach - in a real implementation this would # use more sophisticated techniques like CLFLUSH or large memory traversal # Allocate a large array and access it to flush cache array_size = 16 * 1024 * 1024 # 16 MB array = bytearray(array_size) # Access the array to flush cache for i in range(0, array_size, 64): # 64 bytes is a common cache line size array[i] = 1 def measure_cache_access_pattern(self, function_name: str, input_variations: List[bytes], num_trials: int = 100) -> Dict: """ Measure cache access patterns to detect cache-based side channels This is a simplified simulation - real cache side-channel detection would use hardware performance counters or techniques like Flush+Reload Args: function_name: Name of the function to test input_variations: List of input byte arrays to test num_trials: Number of trials for each input Returns: Dictionary of cache measurement results """ if not self.lib: return {"error": "Library not loaded"} try: # Get function from library func = getattr(self.lib, function_name) # Prepare result containers cache_results = [] for input_data in input_variations: trials = [] for _ in range(num_trials): # Create ctypes buffer for input input_buffer = ctypes.create_string_buffer(input_data) input_size = len(input_data) # Flush cache before measurement self.flush_cache() # Measure cache misses (simulated here) # In a real implementation, we would use perf events or similar start = time.perf_counter_ns() func(input_buffer, input_size) end = time.perf_counter_ns() # We're using time as a proxy for cache behavior here
                    # In a real implementation, we would use actual cache miss counts
                    trials.append(end - start)

                # Calculate statistics
                cache_results.append({
                    'input_size': len(input_data),
                    'input_prefix': input_data[:10].hex() + '...' if len(input_data) > 10 else input_data.hex(),
                    'min_time': min(trials),
                    'max_time': max(trials),
                    'mean_time': np.mean(trials),
                    'median_time': np.median(trials),
                    'std_dev': np.std(trials),
                    'trials': trials
                })

            return {
                'function': function_name,
                'num_trials': num_trials,
                'cache_results': cache_results
            }

        except Exception as e:
            return {"error": f"Error measuring cache for function {function_name}: {e}"}

    def detect_cache_leak(self, cache_results: Dict) -> Dict:
        """
        Analyze cache measurement results to detect potential leaks

        Args:
            cache_results: Results from measure_cache_access_pattern

        Returns:
            Dictionary with leak detection results
        """
        # Similar to timing leak detection, but focused on cache patterns
        if 'error' in cache_results:
            return {'error': cache_results['error'], 'leak_detected': None}

        cache_data = cache_results['cache_results']
        function_name = cache_results['function']

        # Calculate statistics
        means = [data['mean_time'] for data in cache_data]
        std_devs = [data['std_dev'] for data in cache_data]

        # Calculate coefficient of variation (CV) for each input
        cvs = [std_dev / mean if mean > 0 else 0 for mean, std_dev in zip(means, std_devs)]

        # Check if cache patterns vary significantly between inputs
        from scipy import stats
        t_tests = []

        for i in range(len(cache_data)):
            for j in range(i + 1, len(cache_data)):
                trials_i = cache_data[i]['trials']
                trials_j = cache_data[j]['trials']

                t_stat, p_value = stats.ttest_ind(trials_i, trials_j)

                # Record the t-test results
                t_tests.append({
                    'input_i': cache_data[i]['input_prefix'],
                    'input_j': cache_data[j]['input_prefix'],
                    't_stat': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.01  # Significance threshold
                })

        # Determine if there'sa likely cache leak significant_tests = sum(1 for test in t_tests if test['significant']) leak_likelihood = significant_tests / len(t_tests) if t_tests else 0 leak_detected = leak_likelihood > 0.3 # More than 30% of tests show significant differences return { 'function': function_name, 'leak_detected': leak_detected, 'leak_likelihood': leak_likelihood, 'significant_tests': significant_tests, 'total_tests': len(t_tests), 'coefficient_variation': cvs, 'detailed_tests': t_tests } class ConstantTimeAudit: """Comprehensive constant-time audit for C++ code""" def __init__(self, cpp_directory: str, output_directory: str = "audit_results"): self.cpp_directory = cpp_directory self.output_directory = output_directory self.lib_path = self._find_library_path() # Create output directory os.makedirs(output_directory, exist_ok=True) # Initialize analyzers self.static_analyzer = StaticAnalyzer(cpp_directory) if self.lib_path: self.runtime_verifier = RuntimeVerifier(self.lib_path) self.cache_analyzer = CacheSideChannelAnalyzer(self.lib_path) else: self.runtime_verifier = None self.cache_analyzer = None def _find_library_path(self) -> Optional[str]: """Find the path to the compiled library""" # Look for .so, .dll, or .dylib files lib_extensions = ['.so', '.dll', '.dylib'] for root, _, files in os.walk(self.cpp_directory): for file in files: if any(file.endswith(ext) for ext in lib_extensions): return os.path.join(root, file) # Also check in standard directories like 'build', 'lib', 'bin' for dir_name in ['build', 'lib', 'bin', 'out', 'output']: check_dir = os.path.join(self.cpp_directory, dir_name) if os.path.exists(check_dir): for root, _, files in os.walk(check_dir): for file in files: if any(file.endswith(ext) for ext in lib_extensions): return os.path.join(root, file) return None def run_static_analysis(self) -> Dict: """Run static analysis""" print("Running static analysis for side-channel vulnerabilities...") # Scan the directory for C++ files self.static_analyzer.scan_directory() # Analyze the files static_results = self.static_analyzer.analyze_files() # Save results self._save_json(static_results, "static_analysis_results.json") return static_results def run_runtime_verification(self, functions_to_test: Dict[str, List[bytes]]) -> Dict: """ Run runtime verification Args: functions_to_test: Dictionary mapping function names to lists of input variations Returns: Dictionary of verification results """ if not self.runtime_verifier: return {"error": "No library found for runtime verification"} print("Running runtime verification for constant-time execution...") runtime_results = {} for function_name, input_variations in functions_to_test.items(): print(f"Testing function: {function_name}") # Measure timing timing_results = self.runtime_verifier.time_function( function_name, input_variations, num_trials=1000 ) # Detect timing leaks leak_results = self.runtime_verifier.detect_timing_leak(timing_results) # Plot timing distributions plot_file = os.path.join(self.output_directory, f"timing_{function_name}.png") self.runtime_verifier.plot_timing_distributions(timing_results, plot_file) runtime_results[function_name] = { 'timing_results': timing_results, 'leak_analysis': leak_results, 'plot_file': plot_file } # Save results self._save_json(runtime_results, "runtime_verification_results.json") return runtime_results def run_cache_analysis(self, functions_to_test: Dict[str, List[bytes]]) -> Dict: """ Run cache side-channel analysis Args: functions_to_test: Dictionary mapping function names to lists of input variations Returns: Dictionary of cache analysis results """ if not self.cache_analyzer: return {"error": "No library found for cache analysis"} print("Running cache side-channel analysis...") cache_results = {} for function_name, input_variations in functions_to_test.items(): print(f"Analyzing cache patterns for function: {function_name}") # Measure cache patterns measurements = self.cache_analyzer.measure_cache_access_pattern( function_name, input_variations, num_trials=100 ) # Detect cache leaks leak_results = self.cache_analyzer.detect_cache_leak(measurements) cache_results[function_name] = { 'cache_measurements': measurements, 'leak_analysis': leak_results } # Save results self._save_json(cache_results, "cache_analysis_results.json") return cache_results def run_comprehensive_audit(self, functions_to_test: Dict[str, List[bytes]]) -> Dict: """ Run comprehensive constant-time audit Args: functions_to_test: Dictionary mapping function names to lists of input variations Returns: Dictionary with all audit results """ timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") print(f"Starting comprehensive constant-time audit at {timestamp}") # Run all analyses static_results = self.run_static_analysis() runtime_results = self.run_runtime_verification(functions_to_test) cache_results = self.run_cache_analysis(functions_to_test) # Combine results comprehensive_results = { 'timestamp': timestamp, 'cpp_directory': self.cpp_directory, 'lib_path': self.lib_path, 'static_analysis': static_results, 'runtime_verification': runtime_results, 'cache_analysis': cache_results } # Save comprehensive results self._save_json(comprehensive_results, "comprehensive_audit_results.json") # Generate comprehensive report self._generate_report(comprehensive_results) return comprehensive_results def _save_json(self, data: Dict, filename: str): """Save data to JSON file""" file_path = os.path.join(self.output_directory, filename) # Remove non-serializable data data_copy = self._prepare_for_json(data) with open(file_path, 'w') as f: json.dump(data_copy, f, indent=2) def _prepare_for_json(self, data): """Prepare data for JSON serialization by removing non-serializable elements""" if isinstance(data, dict): return {k: self._prepare_for_json(v) for k, v in data.items() if k not in ['lib', 'trials']} elif isinstance(data, list): return [self._prepare_for_json(item) for item in data] elif isinstance(data, np.ndarray): return data.tolist() elif hasattr(data, 'tolist'): return data.tolist() elif isinstance(data, (str, int, float, bool, type(None))): return data else: # Convert other types to string return str(data) def _generate_report(self, results: Dict): """Generate comprehensive HTML report""" timestamp = results['timestamp'] report_file = os.path.join(self.output_directory, f"audit_report_{timestamp}.html") static_results = results.get('static_analysis', {}) runtime_results = results.get('runtime_verification', {}) cache_results = results.get('cache_analysis', {}) # Count violations and leaks total_static_violations = static_results.get('total_violations', 0) runtime_leaks = sum(1 for func in runtime_results.values() if isinstance(func, dict) and func.get('leak_analysis', {}).get('leak_detected')) cache_leaks = sum(1 for func in cache_results.values() if isinstance(func, dict) and func.get('leak_analysis', {}).get('leak_detected')) with open(report_file, 'w') as f: f.write(f""" <!DOCTYPE html> <html> <head> <title>Constant-Time Audit Report: {timestamp}</title> <style> body {{ font-family: Arial, sans-serif; margin: 20px; }} h1 {{ color: #2c3e50; }} h2 {{ color: #3498db; }} h3 {{ color: #2980b9; }} table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }} th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }} th {{ background-color: #f2f2f2; }} tr:nth-child(even) {{ background-color: #f9f9f9; }} .pass {{ color: green; }} .fail {{ color: red; }} .warning {{ color: orange; }} .summary {{ background-color: #eef; padding: 10px; border-radius: 5px; margin-bottom: 20px; }} pre {{ background-color: #f5f5f5; padding: 10px; overflow-x: auto; }} img {{ max-width: 100%; height: auto; margin: 10px 0; }} </style> </head> <body> <h1>QuantoniumOS Constant-Time Audit Report</h1> <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p> <div class="summary"> <h2>Executive Summary</h2> <p><b>C++ Directory:</b> {results['cpp_directory']}</p> <p><b>Library:</b> {results['lib_path'] or 'Not found'}</p> <p><b>Static Analysis Violations:</b> <span class="{'pass' if total_static_violations == 0 else 'fail'}"> {total_static_violations} </span> </p> <p><b>Runtime Timing Leaks:</b> <span class="{'pass' if runtime_leaks == 0 else 'fail'}"> {runtime_leaks} </span> </p> <p><b>Cache Side-Channel Leaks:</b> <span class="{'pass' if cache_leaks == 0 else 'fail'}"> {cache_leaks} </span> </p> <p><b>Overall Assessment:</b> <span class="{'pass' if total_static_violations == 0 and runtime_leaks == 0 and cache_leaks == 0 else 'fail'}"> {'PASS' if total_static_violations == 0 and runtime_leaks == 0 and cache_leaks == 0 else 'FAIL'} </span> </p> </div> """) # Static Analysis Section f.write(""" <h2>1. Static Analysis Results</h2> """) if isinstance(static_results, dict) and 'total_files' in static_results: f.write(f""" <p>Analyzed {static_results['total_files']} C++ files.</p> <p>Found {static_results['files_with_violations']} files with violations.</p> <p>Total security-critical functions: {static_results['security_critical_functions']}</p> <h3>Violations by Type:</h3> <table> <tr> <th>Violation Type</th> <th>Count</th> </tr> """) for vtype, count in static_results.get('violations_by_type', {}).items(): f.write(f""" <tr> <td>{vtype}</td> <td>{count}</td> </tr> """) f.write("</table>") # Files with violations if static_results.get('files_with_violations', 0) > 0: f.write(""" <h3>Files with Violations:</h3> """) for file_path, file_results in static_results.get('violations_by_file', {}).items(): f.write(f""" <h4>{file_path}</h4> <p>Total violations: {file_results['total_violations']}</p> """) if file_results.get('branching_on_secret'): f.write(""" <h5>Branching on Secret Data:</h5> <table> <tr> <th>Function</th> <th>Type</th> <th>Code Snippet</th> </tr> """) for violation in file_results['branching_on_secret']: f.write(f""" <tr> <td>{violation['function']}</td> <td>{violation['type']}</td> <td><pre>{violation['snippet']}</pre></td> </tr> """) f.write("</table>") if file_results.get('memory_access_on_secret'): f.write(""" <h5>Secret-Dependent Memory Access:</h5> <table> <tr> <th>Function</th> <th>Type</th> <th>Code Snippet</th> </tr> """) for violation in file_results['memory_access_on_secret']: f.write(f""" <tr> <td>{violation['function']}</td> <td>{violation['type']}</td> <td><pre>{violation['snippet']}</pre></td> </tr> """) f.write("</table>") else: f.write("<p>No static analysis results available.</p>") # Runtime Verification Section f.write(""" <h2>2. Runtime Verification Results</h2> """) if isinstance(runtime_results, dict) and not runtime_results.get('error'): for function_name, func_results in runtime_results.items(): if not isinstance(func_results, dict): continue leak_analysis = func_results.get('leak_analysis', {}) leak_detected = leak_analysis.get('leak_detected') f.write(f""" <h3>Function: {function_name}</h3> <p><b>Timing Leak Detected:</b> <span class="{'fail' if leak_detected else 'pass'}"> {leak_detected} </span> </p> """) if leak_detected: f.write(f""" <p><b>Leak Likelihood:</b> {leak_analysis.get('leak_likelihood', 0):.2f}</p> <p><b>Significant Test Results:</b> {leak_analysis.get('significant_tests', 0)} / {leak_analysis.get('total_tests', 0)}</p> """) # Show timing distribution plot plot_file = func_results.get('plot_file') if plot_file: plot_filename = os.path.basename(plot_file) f.write(f""" <h4>Timing Distribution:</h4> <img src="{plot_filename}" alt="Timing Distribution"> """) else: error_msg = runtime_results.get('error', "Unknown error") f.write(f"<p>Runtime verification could not be performed: {error_msg}</p>") # Cache Analysis Section f.write(""" <h2>3. Cache Side-Channel Analysis</h2> """) if isinstance(cache_results, dict) and not cache_results.get('error'): for function_name, func_results in cache_results.items(): if not isinstance(func_results, dict): continue leak_analysis = func_results.get('leak_analysis', {}) leak_detected = leak_analysis.get('leak_detected') f.write(f""" <h3>Function: {function_name}</h3> <p><b>Cache Leak Detected:</b> <span class="{'fail' if leak_detected else 'pass'}"> {leak_detected} </span> </p> """) if leak_detected: f.write(f""" <p><b>Leak Likelihood:</b> {leak_analysis.get('leak_likelihood', 0):.2f}</p> <p><b>Significant Test Results:</b> {leak_analysis.get('significant_tests', 0)} / {leak_analysis.get('total_tests', 0)}</p> """) else: error_msg = cache_results.get('error', "Unknown error") f.write(f"<p>Cache analysis could not be performed: {error_msg}</p>") # Recommendations Section f.write(""" <h2>4. Recommendations</h2> <ul> """) if total_static_violations > 0: f.write(""" <li>Fix static violations by implementing constant-time alternatives: <ul> <li>Replace secret-dependent branching with constant-time conditional operations</li> <li>Replace variable-time functions like strcmp with constant-time equivalents</li> <li>Ensure memory access patterns don't depend on secret values</li>
                    </ul>
                </li>
                """)

            if runtime_leaks > 0:
                f.write("""
                <li>Address timing side channels:
                    <ul>
                        <li>Identify and fix operations that take variable time based on input values</li>
                        <li>Implement constant-time comparison for sensitive data</li>
                        <li>Add timing noise or enforce minimum execution time for critical operations</li>
                    </ul>
                </li>
                """)

            if cache_leaks > 0:
                f.write("""
                <li>Mitigate cache side channels:
                    <ul>
                        <li>Use cache-resistant programming techniques like data-independent memory access</li>
                        <li>Implement cache prefetching or cache line padding</li>
                        <li>Consider using protection mechanisms like memory access obfuscation</li>
                    </ul>
                </li>
                """)

            if total_static_violations == 0 and runtime_leaks == 0 and cache_leaks == 0:
                f.write("""
                <li>Continue best practices for side-channel resistant code:
                    <ul>
                        <li>Maintain constant-time implementations for all security-critical operations</li>
                        <li>Regularly audit new code for side-channel vulnerabilities</li>
                        <li>Consider formal verification of critical components</li>
                    </ul>
                </li>
                """)

            f.write("""
                </ul>

                <h2>5. Conclusion</h2>
                <p>
                    This audit has evaluated the QuantoniumOS C++ core for side-channel vulnerabilities
                    including timing side channels, cache-based side channels, and violations of constant-time
                    programming principles.
                </p>
                <p>
                    Based on the findings, the codebase is assessed to be
                    <span class="{'pass' if total_static_violations == 0 and runtime_leaks == 0 and cache_leaks == 0 else 'fail'}">
                    {'' if total_static_violations == 0 and runtime_leaks == 0 and cache_leaks == 0 else 'NOT '}
                    RESISTANT
                    </span>
                    to side-channel attacks.
                </p>
                <p>
                    The recommendations provided should be addressed to improve the security posture
                    of the QuantoniumOS implementation against side-channel adversaries.
                </p>
            </body>
            </html>
            """)

        print(f"Comprehensive report generated: {report_file}")

def run_side_channel_audit():
    """Run the side-channel and constant-time audit for QuantoniumOS C++ core"""

    # Path to C++ directory
    cpp_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "core", "cpp")

    # Create audit tool
    audit = ConstantTimeAudit(cpp_directory, output_directory="audit_results/side_channel")

    # Example functions to test (would be actual functions from the library in a real audit)
    # Each function is tested with different input variations to detect potential leaks
    functions_to_test = {
        "resonance_encrypt": [
            # Different input lengths
            os.urandom(16),
            os.urandom(32),
            os.urandom(64),
            os.urandom(128),
            # Same length, different content
            b"\x00" * 32,
            b"\xFF" * 32,
            b"\xAA" * 32,
            os.urandom(32)
        ],
        "waveform_hash": [
            # Different input lengths
            os.urandom(16),
            os.urandom(32),
            os.urandom(64),
            # Same length, different content
            b"\x00" * 32,
            b"\xFF" * 32,
            b"\xAA" * 32,
            os.urandom(32)
        ],
        "constant_time_compare": [
            # Different combinations of equal/unequal inputs
            b"\x00" * 32,
            b"\x00" * 31 + b"\x01",
            b"\x01" + b"\x00" * 31,
            os.urandom(32)
        ]
    }

    # Run the comprehensive audit
    results = audit.run_comprehensive_audit(functions_to_test)

    return results

if __name__ == "__main__":
    # Run the side-channel and constant-time audit
    run_side_channel_audit()
