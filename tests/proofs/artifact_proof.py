#!/usr/bin/env python3
"""
ARTIFACT PROOF - REPRODUCIBILITY DOCUMENTATION
==============================================
Complete environment and reproduction information for reviewers.
"""

import platform
import sys
import os
import hashlib
import subprocess
import time

def get_system_info():
    """Collect detailed system information"""
    print("SYSTEM HARDWARE & ENVIRONMENT")
    print("=" * 50)
    
    # Basic system info
    print(f"OS: {platform.system()} {platform.release()} {platform.version()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    
    # Python info
    print(f"Python: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    # CPU info (Windows specific)
    try:
        import psutil
        cpu_count = psutil.cpu_count(logical=False)
        cpu_count_logical = psutil.cpu_count(logical=True)
        memory_gb = psutil.virtual_memory().total / (1024**3)
        print(f"CPU cores (physical): {cpu_count}")
        print(f"CPU cores (logical): {cpu_count_logical}")
        print(f"Total RAM: {memory_gb:.1f} GB")
    except ImportError:
        print("psutil not available - install for detailed CPU/memory info")
    
    # Windows build info
    if platform.system() == "Windows":
        try:
            result = subprocess.run(['wmic', 'os', 'get', 'BuildNumber'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                build_lines = result.stdout.strip().split('\n')
                if len(build_lines) > 1:
                    build_number = build_lines[1].strip()
                    print(f"Windows Build: {build_number}")
        except:
            pass

def get_dll_info():
    """Get DLL binary provenance information"""
    print("\nBINARY PROVENANCE")
    print("=" * 50)
    
    dll_path = os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'assembly', 'compiled', 'libquantum_symbolic.dll')
    
    if os.path.exists(dll_path):
        # File stats
        stat = os.stat(dll_path)
        size_kb = stat.st_size / 1024
        build_time = time.ctime(stat.st_mtime)
        
        print(f"Library: libquantum_symbolic.dll")
        print(f"Size: {size_kb:.1f} KB")
        print(f"Build timestamp: {build_time}")
        
        # SHA-256 hash
        with open(dll_path, 'rb') as f:
            dll_bytes = f.read()
            sha256_hash = hashlib.sha256(dll_bytes).hexdigest()
            print(f"SHA-256: {sha256_hash}")
        
        # Try to get embedded build info
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'assembly', 'python_bindings'))
            from quantum_symbolic_engine import QuantumSymbolicEngine
            engine = QuantumSymbolicEngine()
            print(f"DLL loaded successfully: YES")
            print(f"Assembly backend: {'ENABLED' if engine.use_assembly else 'DISABLED'}")
            print(f"Compression size: {engine.compression_size}")
        except Exception as e:
            print(f"DLL load test: FAILED ({e})")
    else:
        print(f"DLL not found at: {dll_path}")

def get_environment_lock():
    """Generate environment lock information"""
    print("\nENVIRONMENT LOCK")
    print("=" * 50)
    
    # Try pip freeze
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            packages = result.stdout.strip().split('\n')
            key_packages = [p for p in packages if any(key in p.lower() for key in 
                          ['numpy', 'scipy', 'ctypes', 'psutil'])]
            print("Key packages:")
            for pkg in key_packages:
                print(f"  {pkg}")
            
            # Save full freeze
            freeze_file = os.path.join(os.path.dirname(__file__), 'requirements_freeze.txt')
            with open(freeze_file, 'w') as f:
                f.write(result.stdout)
            print(f"Full freeze saved to: {freeze_file}")
        else:
            print("pip freeze failed")
    except Exception as e:
        print(f"Environment lock failed: {e}")

def document_test_seeds():
    """Document random seeds used in tests"""
    print("\nRANDOM SEEDS USED")
    print("=" * 50)
    
    print("Deterministic seeds for reproducibility:")
    print("  timing_curves.py: No explicit seed (time-based)")
    print("  scaling_analysis.py: No explicit seed (time-based)")
    print("  parity_test.py: seed=42")
    print("  roundtrip_integrity.py: seed=12345")
    print("  test_corrected_assembly.py: No explicit seed")
    print("")
    print("Note: For full reproducibility, add explicit seeds to all tests")

def reproduction_commands():
    """Provide exact reproduction commands"""
    print("\nREPRODUCTION COMMANDS")
    print("=" * 50)
    
    print("To reproduce all validation results:")
    print("")
    print("1. Core mathematical validation:")
    print("   python run_comprehensive_validation.py")
    print("   Expected: 7/7 tests PASS")
    print("")
    print("2. Timing curves analysis:")
    print("   python test_timing_curves.py")
    print("   Expected: 0.732 → 0.057 μs/qubit scaling improvement")
    print("")
    print("3. Scaling exponent analysis:")
    print("   python ci_scaling_analysis.py") 
    print("   Expected: α ≈ 0.16, R² ≈ 0.22 (overhead-dominated small-n)")
    print("")
    print("4. Assembly engine validation:")
    print("   python test_corrected_assembly.py")
    print("   Expected: 3/3 tests PASS, 100k symbolic qubits in ~5ms")
    print("")
    print("5. Complete summary:")
    print("   python final_summary.py")
    print("   Expected: All test data summary with corrected metrics")

def main():
    """Generate complete artifact proof documentation"""
    print("ARTIFACT PROOF - COMPLETE REPRODUCIBILITY DOCUMENTATION")
    print("=" * 80)
    print("Generated for independent verification by reviewers")
    print("=" * 80)
    
    get_system_info()
    get_dll_info()
    get_environment_lock()
    document_test_seeds()
    reproduction_commands()
    
    print("\n" + "=" * 80)
    print("ARTIFACT PROOF COMPLETE")
    print("=" * 80)
    print("All information provided for independent reproduction")
    print("Contact: Provide through paper submission system")
    print("License: As specified in project LICENSE.md")

if __name__ == "__main__":
    main()
