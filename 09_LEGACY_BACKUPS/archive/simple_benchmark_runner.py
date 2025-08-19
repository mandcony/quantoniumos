#!/usr/bin/env python3
"""
Simple RFT Benchmark Runner - No Fluff Edition Run individual benchmarks to validate RFT paradigm with fair comparisons: - Crypto: Avalanche/differential score (RFT vs AES) - Quantum: Coherence ratio vs baseline - Optimization: Rosenbrock minima (RFT vs standard) - Pattern: Silhouette score difference on resonance patterns
"""
"""
import subprocess
import sys
import json from pathlib
import Path
def run_benchmark(name, scale="small"): """
        Run a single benchmark and
        return the key validation metric
"""
        """ benchmarks = { "crypto": "crypto_benchmark.py", "quantum": "quantum_benchmark.py", "optimization": "optimization_benchmark.py", "pattern": "pattern_benchmark.py" }
        if name not in benchmarks:
        print(f"Unknown benchmark: {name}")
        print(f"Available: {list(benchmarks.keys())}")
        return None script = benchmarks[name]
        print(f"🔬 Running {name} benchmark (scale: {scale})")
        try:

        # Simple subprocess call - no complex argument parsing result = subprocess.run([ sys.executable, script, "--scale", scale, "--output", f"{name}_results.json" ], capture_output=True, text=True, check=True)
        print(result.stdout)

        # Load and extract the key metric results_file = f"{name}_results.json"
        if Path(results_file).exists(): with open(results_file) as f: data = json.load(f)
        return extract_key_metric(name, data) except subprocess.CalledProcessError as e:
        print(f"❌ {name} failed: {e}")
        print(f"Stderr: {e.stderr}")
        return None
def extract_key_metric(benchmark_name, results): """
        Extract the single metric that proves RFT advantage
"""
"""
        if benchmark_name == "crypto":

        # Avalanche/differential score comparison
        return results.get("avalanche_improvement", 0)
        el
        if benchmark_name == "quantum":

        # Coherence ratio vs baseline
        return results.get("coherence_ratio", 0)
        el
        if benchmark_name == "optimization":

        # How much lower minima RFT finds vs standard
        return results.get("minima_improvement", 0)
        el
        if benchmark_name == "pattern":

        # Silhouette score difference
        return results.get("silhouette_improvement", 0)
        return 0
def validate_rft_paradigm(): """
        Run all benchmarks and check
        if RFT shows consistent improvement
"""
"""
        print(" RFT PARADIGM VALIDATION")
        print("=" * 40) benchmarks = ["crypto", "quantum", "optimization", "pattern"] results = {}
        for benchmark in benchmarks: metric = run_benchmark(benchmark, scale="small") results[benchmark] = metric
        if metric and metric > 1.0:
        print(f"✅ {benchmark}: {metric:.2f}× improvement")
        else:
        print(f"❌ {benchmark}: No improvement shown")
        print()

        # Final validation successful = sum(1
        for m in results.values()
        if m and m > 1.0)
        print("=" * 40)
        if successful >= 3: # 3 out of 4 showing improvement
        print(f"🎉 RFT PARADIGM VALIDATED ({successful}/4 benchmarks)")
        print(" Symbolic Resonance Computing shows measurable advantages")
        else:
        print(f"⚠️ RFT needs work ({successful}/4 benchmarks)")
        print("🔧 Focus on failing benchmarks")
        return results

if __name__ == "__main__":
if len(sys.argv) > 1:

# Run single benchmark benchmark = sys.argv[1] scale = sys.argv[2]
if len(sys.argv) > 2 else "small" run_benchmark(benchmark, scale)
else:

# Run validation suite validate_rft_paradigm()