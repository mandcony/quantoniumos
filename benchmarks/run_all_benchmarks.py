#!/usr/bin/env python3
# Copyright (c) 2025 QuantoniumOS
#
# This file is part of QuantoniumOS.
#
# QuantoniumOS is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# QuantoniumOS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with QuantoniumOS.  If not, see <https://www.gnu.org/licenses/>.

"""
QuantoniumOS Formal Benchmark Suite
====================================

Master runner for all benchmark classes.

Classes:
  A - Quantum Symbolic Simulation (QSC vs Qiskit/Cirq)
  B - Transform & DSP (Φ-RFT vs FFT ecosystem)
  C - Compression (RFTMW vs zstd/brotli/lzma)
  D - Cryptography (RFT-SIS + Feistel vs OpenSSL/liboqs)
  E - Audio & DAW (Audio engine performance)

VARIANT COVERAGE:
  - All 14 Φ-RFT variants benchmarked per class
  - All 17 hybrids tested where applicable
  - Use --variants flag to run full variant/hybrid benchmarks

Usage:
    python run_all_benchmarks.py           # Run all classes
    python run_all_benchmarks.py A B       # Run specific classes
    python run_all_benchmarks.py --variants # Include variant/hybrid benchmarks
    python run_all_benchmarks.py --install # Install missing dependencies
"""

import sys
import os
import json
import argparse
import subprocess
import time
from datetime import datetime

# Change to project root
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '..')


def check_dependencies():
    """Check and report available dependencies"""
    deps = {
        'core': {},
        'class_a': {},
        'class_b': {},
        'class_c': {},
        'class_d': {},
        'class_e': {}
    }
    
    # Core dependencies
    try:
        import numpy
        deps['core']['numpy'] = numpy.__version__
    except ImportError:
        deps['core']['numpy'] = None
    
    # Class A - Quantum
    try:
        import qiskit
        deps['class_a']['qiskit'] = qiskit.__version__
    except ImportError:
        deps['class_a']['qiskit'] = None
    
    try:
        import cirq
        deps['class_a']['cirq'] = cirq.__version__
    except ImportError:
        deps['class_a']['cirq'] = None
    
    # Class B - Transform
    try:
        import scipy
        deps['class_b']['scipy'] = scipy.__version__
    except ImportError:
        deps['class_b']['scipy'] = None
    
    try:
        import pyfftw
        deps['class_b']['pyfftw'] = pyfftw.__version__
    except (ImportError, AttributeError):
        deps['class_b']['pyfftw'] = None
    
    # Class C - Compression
    try:
        import zstandard
        deps['class_c']['zstandard'] = zstandard.__version__
    except ImportError:
        deps['class_c']['zstandard'] = None
    
    try:
        import brotli
        deps['class_c']['brotli'] = True
    except ImportError:
        deps['class_c']['brotli'] = None
    
    try:
        import lz4
        deps['class_c']['lz4'] = lz4.__version__
    except ImportError:
        deps['class_c']['lz4'] = None
    
    # Class D - Crypto
    try:
        from cryptography import __version__ as crypto_ver
        deps['class_d']['cryptography'] = crypto_ver
    except ImportError:
        deps['class_d']['cryptography'] = None
    
    try:
        import nacl
        deps['class_d']['pynacl'] = nacl.__version__
    except ImportError:
        deps['class_d']['pynacl'] = None
    
    try:
        import oqs
        deps['class_d']['liboqs'] = True
    except ImportError:
        deps['class_d']['liboqs'] = None
    
    # Class E - Audio
    try:
        import sounddevice
        deps['class_e']['sounddevice'] = sounddevice.__version__
    except ImportError:
        deps['class_e']['sounddevice'] = None
    
    try:
        import librosa
        deps['class_e']['librosa'] = librosa.__version__
    except ImportError:
        deps['class_e']['librosa'] = None
    
    # Native module
    try:
        sys.path.insert(0, '../src/rftmw_native/build')
        import rftmw_native
        deps['core']['rftmw_native'] = True
    except ImportError:
        deps['core']['rftmw_native'] = None
    
    return deps


def install_dependencies():
    """Install recommended dependencies"""
    print("Installing benchmark dependencies...")
    print()
    
    packages = [
        # Core
        'numpy',
        'scipy',
        
        # Class A - Quantum
        'qiskit',
        'cirq',
        
        # Class B - Transform
        'pyfftw',
        
        # Class C - Compression  
        'zstandard',
        'brotli',
        'lz4',
        
        # Class D - Crypto
        'cryptography',
        'pynacl',
        
        # Class E - Audio
        'sounddevice',
        'librosa',
    ]
    
    for pkg in packages:
        print(f"  Installing {pkg}...")
        try:
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '-q', pkg],
                check=True,
                capture_output=True
            )
            print(f"    ✓ {pkg} installed")
        except subprocess.CalledProcessError:
            print(f"    ✗ {pkg} failed (non-critical)")
    
    print()
    print("Note: liboqs requires manual installation:")
    print("  https://github.com/open-quantum-safe/liboqs-python")
    print()


def run_class_a():
    """Run Class A benchmark"""
    try:
        from class_a_quantum_simulation import run_class_a_benchmark, run_variant_quantum_benchmark, run_hybrid_quantum_benchmark
        result = run_class_a_benchmark()
        return {'main': result}
    except Exception as e:
        print(f"  Error: {e}")
        return None


def run_class_a_variants():
    """Run Class A variant benchmarks"""
    try:
        from class_a_quantum_simulation import run_variant_quantum_benchmark, run_hybrid_quantum_benchmark
        v = run_variant_quantum_benchmark()
        h = run_hybrid_quantum_benchmark()
        return {'variants': len(v), 'hybrids': len(h)}
    except Exception as e:
        print(f"  Variant Error: {e}")
        return None


def run_class_b():
    """Run Class B benchmark"""
    try:
        from class_b_transform_dsp import run_class_b_benchmark
        result = run_class_b_benchmark()
        
        # Add Phi Frame Benchmark (Gram & Asymptotics)
        print("\n" + "━"*75)
        print("  RUNNING PHI FRAME BENCHMARK (Gram & Asymptotics)")
        print("━"*75)
        try:
            from rft_phi_frame_benchmark import main as run_frame
            run_frame()
        except Exception as e:
            print(f"  Frame Benchmark Error: {e}")
            
        return result
    except Exception as e:
        print(f"  Error: {e}")
        return None


def run_class_b_variants():
    """Run Class B variant benchmarks"""
    try:
        from class_b_transform_dsp import run_variant_dsp_benchmark, run_hybrid_dsp_benchmark
        v = run_variant_dsp_benchmark()
        h = run_hybrid_dsp_benchmark()
        return {'variants': len(v), 'hybrids': len(h)}
    except Exception as e:
        print(f"  Variant Error: {e}")
        return None


def run_class_c():
    """Run Class C benchmark"""
    try:
        from class_c_compression import run_class_c_benchmark
        return run_class_c_benchmark()
    except Exception as e:
        print(f"  Error: {e}")
        return None


def run_class_c_variants():
    """Run Class C variant benchmarks"""
    try:
        from class_c_compression import run_variant_compression_benchmark, run_hybrid_compression_benchmark
        v = run_variant_compression_benchmark()
        h = run_hybrid_compression_benchmark()
        return {'variants': len(v), 'hybrids': len(h)}
    except Exception as e:
        print(f"  Variant Error: {e}")
        return None


def run_class_d():
    """Run Class D benchmark"""
    try:
        from class_d_crypto import run_class_d_benchmark
        return run_class_d_benchmark()
    except Exception as e:
        print(f"  Error: {e}")
        return None


def run_class_d_variants():
    """Run Class D variant benchmarks"""
    try:
        from class_d_crypto import run_variant_crypto_benchmark
        v = run_variant_crypto_benchmark()
        return {'variants': len(v)}
    except Exception as e:
        print(f"  Variant Error: {e}")
        return None


def run_class_e():
    """Run Class E benchmark"""
    try:
        from class_e_audio_daw import run_class_e_benchmark
        return run_class_e_benchmark()
    except Exception as e:
        print(f"  Error: {e}")
        return None


def run_class_e_variants():
    """Run Class E variant benchmarks"""
    try:
        from class_e_audio_daw import run_variant_audio_benchmark, run_hybrid_audio_benchmark
        v = run_variant_audio_benchmark()
        h = run_hybrid_audio_benchmark()
        return {'variants': len(v), 'hybrids': len(h)}
    except Exception as e:
        print(f"  Variant Error: {e}")
        return None


def print_summary(results):
    """Print overall benchmark summary"""
    print()
    print("=" * 75)
    print("  QUANTONIUMOS BENCHMARK SUMMARY")
    print("=" * 75)
    print()
    
    summary = []
    
    if 'A' in results:
        summary.append("  CLASS A (Quantum):      QSC achieves O(n) symbolic compression (Not full statevector)")
    if 'B' in results:
        summary.append("  CLASS B (Transform):    Φ-RFT provides golden-ratio decorrelation (Slower than FFT)")
    if 'C' in results:
        summary.append("  CLASS C (Compression):  RFTMW exploits entropy gap (Lower ratio than zstd)")
    if 'D' in results:
        summary.append("  CLASS D (Crypto):       RFT-SIS implements lattice primitives (Research Prototype)")
    if 'E' in results:
        summary.append("  CLASS E (Audio):        Φ-RFT spectral analysis (High latency, offline only)")
    
    for s in summary:
        print(s)
    
    print()
    print("  HONEST FRAMING ACROSS ALL CLASSES:")
    print("  ┌─────────────────────────────────────────────────────────────────────┐")
    print("  │  SCIENTIFIC CONTEXT:                                              │")
    print("  │  QuantoniumOS is a specialized 'microscope' for aperiodic order.   │")
    print("  │  It is NOT a general-purpose replacement for FFT, gzip, or AES.    │")
    print("  │  It excels at analyzing quasi-periodic structures (Class A/B)      │")
    print("  │  but trades computational efficiency for structural precision.     │")
    print("  └─────────────────────────────────────────────────────────────────────┘")
    print()


def main():
    parser = argparse.ArgumentParser(description='QuantoniumOS Benchmark Suite')
    parser.add_argument('classes', nargs='*', default=['A', 'B', 'C', 'D', 'E'],
                        help='Classes to run (A, B, C, D, E)')
    parser.add_argument('--install', action='store_true',
                        help='Install missing dependencies')
    parser.add_argument('--deps', action='store_true',
                        help='Check dependencies only')
    parser.add_argument('--variants', action='store_true',
                        help='Include full variant/hybrid benchmarks')
    parser.add_argument('--json', type=str,
                        help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Dependency check
    if args.deps:
        deps = check_dependencies()
        print("Dependency Status:")
        for category, packages in deps.items():
            print(f"  {category}:")
            for pkg, version in packages.items():
                status = '✓' if version else '✗'
                ver = f" ({version})" if version and version is not True else ""
                print(f"    {status} {pkg}{ver}")
        return
    
    # Install if requested
    if args.install:
        install_dependencies()
        return
    
    print()
    print("╔═══════════════════════════════════════════════════════════════════════╗")
    print("║         QUANTONIUMOS FORMAL BENCHMARK SUITE                           ║")
    print("║         Competitive Analysis Against Industry Standards               ║")
    print("╚═══════════════════════════════════════════════════════════════════════╝")
    print()
    print(f"  Date: {datetime.now().isoformat()}")
    print(f"  Classes: {', '.join(args.classes)}")
    print()
    
    # Check dependencies first
    deps = check_dependencies()
    print("  Dependency Status:")
    for pkg, ver in deps['core'].items():
        print(f"    {pkg}: {'✓' if ver else '✗'}")
    print()
    
    results = {}
    
    class_runners = {
        'A': ('Quantum Symbolic Simulation', run_class_a, run_class_a_variants),
        'B': ('Transform & DSP', run_class_b, run_class_b_variants),
        'C': ('Compression', run_class_c, run_class_c_variants),
        'D': ('Cryptography & Post-Quantum', run_class_d, run_class_d_variants),
        'E': ('Audio & DAW', run_class_e, run_class_e_variants),
    }
    
    for cls in args.classes:
        cls = cls.upper()
        if cls in class_runners:
            name, runner, variant_runner = class_runners[cls]
            print()
            print(f"{'─' * 75}")
            print(f"  Running CLASS {cls}: {name}")
            print(f"{'─' * 75}")
            print()
            
            start = time.perf_counter()
            result = runner()
            elapsed = time.perf_counter() - start
            
            results[cls] = {
                'name': name,
                'elapsed_s': elapsed,
                'data': result
            }
            
            print()
            print(f"  CLASS {cls} completed in {elapsed:.2f}s")
            
            # Run variant benchmarks if requested
            if args.variants and variant_runner:
                print()
                print(f"  Running CLASS {cls} VARIANT BENCHMARKS...")
                variant_start = time.perf_counter()
                variant_result = variant_runner()
                variant_elapsed = time.perf_counter() - variant_start
                
                results[cls]['variant_data'] = variant_result
                results[cls]['variant_elapsed_s'] = variant_elapsed
                print(f"  CLASS {cls} variants completed in {variant_elapsed:.2f}s")
    
    # Summary
    print_summary(results)
    
    # Save results
    if args.json:
        output = {
            'timestamp': datetime.now().isoformat(),
            'classes': results
        }
        with open(args.json, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"  Results saved to: {args.json}")
    
    print()
    return results


if __name__ == "__main__":
    main()
