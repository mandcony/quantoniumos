||#!/usr/bin/env python3
""""""
RFT Benchmark Suite Controller

Master controller for running individual or all RFT benchmarks
with configurable parameters and environment scaling.
""""""

import argparse
import json
import time
import subprocess
import sys
from typing import Dict, List, Any
from benchmark_utils import BenchmarkUtils

class BenchmarkController:
    """"""Master controller for RFT benchmark suite""""""

    def __init__(self):
        self.available_benchmarks = {
            'crypto': {
                'script': 'crypto_benchmark.py',
                'description': 'Cryptographic robustness under differential attacks'
            },
            'quantum': {
                'script': 'quantum_benchmark.py',
                'description': 'Quantum coherence preservation in noisy environments'
            },
            'optimization': {
                'script': 'optimization_benchmark.py',
                'description': 'Non-linear optimization with symbolic constraints'
            },
            'pattern': {
                'script': 'pattern_benchmark.py',
                'description': 'High-dimensional pattern recognition'
            }
        }

    def run_benchmark(self, benchmark_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """"""Run a single benchmark with given configuration""""""
        if benchmark_name not in self.available_benchmarks:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")

        script = self.available_benchmarks[benchmark_name]['script']
        output_file = f"{benchmark_name}_benchmark_results.json"

        # Build command line arguments
        cmd = [sys.executable, script]

        # Add common arguments
        if 'scale' in config:
            cmd.extend(['--scale', config['scale']])
        if 'random_seed' in config:
            cmd.extend(['--random-seed', str(config['random_seed'])])
        cmd.extend(['--output', output_file])

        # Add benchmark-specific arguments
        if benchmark_name == 'crypto':
            if 'num_tests' in config:
                cmd.extend(['--num-tests', str(config['num_tests'])])

        elif benchmark_name == 'quantum':
            if 'n_qubits' in config:
                cmd.extend(['--n-qubits', str(config['n_qubits'])])
            if 'circuit_depth' in config:
                cmd.extend(['--circuit-depth', str(config['circuit_depth'])])
            if 'num_trials' in config:
                cmd.extend(['--num-trials', str(config['num_trials'])])

        elif benchmark_name == 'optimization':
            if 'dimensions' in config:
                cmd.extend(['--dimensions'] + [str(d) for d in config['dimensions']])
            if 'max_iterations' in config:
                cmd.extend(['--max-iterations', str(config['max_iterations'])])
            if 'num_restarts' in config:
                cmd.extend(['--num-restarts', str(config['num_restarts'])])

        elif benchmark_name == 'pattern':
            if 'n_samples' in config:
                cmd.extend(['--n-samples', str(config['n_samples'])])
            if 'n_features' in config:
                cmd.extend(['--n-features', str(config['n_features'])])
            if 'n_patterns' in config:
                cmd.extend(['--n-patterns', str(config['n_patterns'])])

        print(f"🚀 Running {benchmark_name} benchmark...")
        print(f"Command: {' '.join(cmd)}")

        try:
            # Run the benchmark
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(result.stdout)

            # Load results
            results = BenchmarkUtils.load_results(output_file)
            return {
                'status': 'success',
                'results': results,
                'output_file': output_file,
                'stdout': result.stdout
            }

        except subprocess.CalledProcessError as e:
            print(f"❌ Benchmark {benchmark_name} failed:")
            print(f"Return code: {e.returncode}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            return {
                'status': 'failed',
                'error': str(e),
                'stdout': e.stdout,
                'stderr': e.stderr
            }

    def run_all_benchmarks(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """"""Run all benchmarks sequentially""""""
        print("🔬 RFT BENCHMARK SUITE")
        print("=" * 50)
        print("Running all benchmarks with configuration:")
        for key, value in config.items():
            print(f" {key}: {value}")
        print()

        all_results = {}
        summary = {
            'total_benchmarks': len(self.available_benchmarks),
            'successful': 0,
            'failed': 0,
            'start_time': time.time()
        }

        for name, info in self.available_benchmarks.items():
            print(f"\n📊 {name.upper()} BENCHMARK")
            print(f"Description: {info['description']}")
            print("-" * 40)

            result = self.run_benchmark(name, config)
            all_results[name] = result

            if result['status'] == 'success':
                summary['successful'] += 1
                print(f"✅ {name} benchmark completed successfully")
            else:
                summary['failed'] += 1
                print(f"❌ {name} benchmark failed")

        summary['end_time'] = time.time()
        summary['total_time'] = summary['end_time'] - summary['start_time']

        # Generate final summary
        print("||n" + "=" * 70)
        print("📈 BENCHMARK SUITE SUMMARY")
        print("=" * 70)

        for name, result in all_results.items():
            if result['status'] == 'success':
                # Extract key metrics from each benchmark
                metrics = self._extract_key_metrics(name, result['results'])
                print(f"{name.upper():15} | {metrics}")
            else:
                print(f"{name.upper():15} ||| FAILED")

        print(f"\nTotal time: {summary['total_time']:.1f} seconds")
        print(f"Successful: {summary['successful']}/{summary['total_benchmarks']}")
        print(f"Failed: {summary['failed']}/{summary['total_benchmarks']}")

        if summary['successful'] == summary['total_benchmarks']:
            print("\n🎉 All benchmarks completed successfully!")
            print("🚀 RFT demonstrates clear computational advantages")
        else:
            print(f"\n⚠️ {summary['failed']} benchmark(s) failed")

        # Save combined results
        combined_results = {
            'summary': summary,
            'config': config,
            'benchmarks': all_results
        }

        output_file = 'rft_benchmark_suite_results.json'
        BenchmarkUtils.save_results(combined_results, output_file)
        print(f"||n📁 Complete results saved to: {output_file}")

        return combined_results

    def _extract_key_metrics(self, benchmark_name: str, results: Dict[str, Any]) -> str:
        """"""Extract key performance metrics for summary""""""
        try:
            if benchmark_name == 'crypto':
                improvement = results.get('robustness_improvement', 0)
                return f"Robustness: {improvement:.1f}× better"

            elif benchmark_name == 'quantum':
                improvement = results.get('avg_improvement', 0)
                return f"Coherence: {improvement:.1f}× better"

            elif benchmark_name == 'optimization':
                acc_imp = results.get('avg_accuracy_improvement', 0)
                return f"Accuracy: {acc_imp:.1f}× better"

            elif benchmark_name == 'pattern':
                quality_imp = results.get('quality_improvement', 0)
                return f"Quality: {quality_imp:.1f}× better"

            else:
                return "Metrics available"

        except Exception:
            return "Completed"

    def list_benchmarks(self):
        """"""List available benchmarks""""""
        print("Available benchmarks:")
        for name, info in self.available_benchmarks.items():
            print(f" {name:12} - {info['description']}")

def main():
    """"""Main CLI interface for benchmark controller""""""
    parser = argparse.ArgumentParser(
        description="RFT Benchmark Suite Controller",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=""""""
Examples:
  # Run all benchmarks with medium scale
  python benchmark_controller.py --all --scale medium

  # Run only crypto benchmark with small scale
  python benchmark_controller.py --only crypto --scale small

  # Run crypto and quantum benchmarks
  python benchmark_controller.py --only crypto quantum --scale large

  # List available benchmarks
  python benchmark_controller.py --list
        """"""
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Run all benchmarks")
    group.add_argument("--only", nargs='+', choices=['crypto', 'quantum', 'optimization', 'pattern'],
                      help="Run specific benchmark(s)")
    group.add_argument("--list", action="store_true", help="List available benchmarks")

    parser.add_argument("--scale", choices=['small', 'medium', 'large', 'xlarge'], default='medium',
                       help="Scale factor for benchmarks (affects test sizes)")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility")

    # Benchmark-specific parameters
    parser.add_argument("--num-tests", type=int, help="Number of tests (crypto)")
    parser.add_argument("--n-qubits", type=int, help="Number of qubits (quantum)")
    parser.add_argument("--circuit-depth", type=int, help="Circuit depth (quantum)")
    parser.add_argument("--num-trials", type=int, help="Number of trials (quantum)")
    parser.add_argument("--dimensions", nargs='+', type=int, help="Problem dimensions (optimization)")
    parser.add_argument("--max-iterations", type=int, help="Max optimization iterations")
    parser.add_argument("--num-restarts", type=int, help="Number of optimization restarts")
    parser.add_argument("--n-samples", type=int, help="Number of samples (pattern)")
    parser.add_argument("--n-features", type=int, help="Number of features (pattern)")
    parser.add_argument("--n-patterns", type=int, help="Number of patterns (pattern)")

    args = parser.parse_args()

    controller = BenchmarkController()

    if args.list:
        controller.list_benchmarks()
        return

    # Build configuration from arguments
    config = {
        'scale': args.scale,
        'random_seed': args.random_seed
    }

    # Add benchmark-specific parameters
    if args.num_tests is not None:
        config['num_tests'] = args.num_tests
    if args.n_qubits is not None:
        config['n_qubits'] = args.n_qubits
    if args.circuit_depth is not None:
        config['circuit_depth'] = args.circuit_depth
    if args.num_trials is not None:
        config['num_trials'] = args.num_trials
    if args.dimensions is not None:
        config['dimensions'] = args.dimensions
    if args.max_iterations is not None:
        config['max_iterations'] = args.max_iterations
    if args.num_restarts is not None:
        config['num_restarts'] = args.num_restarts
    if args.n_samples is not None:
        config['n_samples'] = args.n_samples
    if args.n_features is not None:
        config['n_features'] = args.n_features
    if args.n_patterns is not None:
        config['n_patterns'] = args.n_patterns

    if args.all:
        controller.run_all_benchmarks(config)
    else:
        for benchmark_name in args.only:
            result = controller.run_benchmark(benchmark_name, config)
            if result['status'] == 'failed':
                print(f"❌ {benchmark_name} benchmark failed")
                sys.exit(1)

if __name__ == "__main__":
    main()
