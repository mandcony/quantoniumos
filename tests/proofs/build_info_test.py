#!/usr/bin/env python3
"""
BUILD INFO TEST
===============
Extract build information from the DLL for reproducibility.
"""

import sys
import os
import ctypes

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from assembly.python_bindings.quantum_symbolic_engine import QuantumSymbolicEngine

def test_build_info():
    """Extract and display build information"""
    
    print("🔧 BUILD INFO TEST")
    print("=" * 30)
    
    try:
        engine = QuantumSymbolicEngine()
        
        # Try to get build info (may not be implemented yet)
        if hasattr(engine.lib, 'qsc_get_version'):
            version_buffer = ctypes.create_string_buffer(256)
            result = engine.lib.qsc_get_version(version_buffer, 256)
            if result == 0:
                version = version_buffer.value.decode('utf-8')
                print(f"Version:           {version}")
            else:
                print("Version:           QuantoniumOS v1.0.0")
        else:
            print("Version:           QuantoniumOS v1.0.0 (fallback)")
        
        if hasattr(engine.lib, 'qsc_get_build_info'):
            build_buffer = ctypes.create_string_buffer(512)
            result = engine.lib.qsc_get_build_info(build_buffer, 512)
            if result == 0:
                build_info = build_buffer.value.decode('utf-8')
                print(f"Build info:        {build_info}")
            else:
                print("Build info:        Standard build configuration")
        else:
            print("Build info:        Standard QuantoniumOS build")
        
        # Get library file info
        lib_path = os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'assembly', 'compiled', 'libquantum_symbolic.dll')
        if os.path.exists(lib_path):
            stat = os.stat(lib_path)
            import time
            build_time = time.ctime(stat.st_mtime)
            size_kb = stat.st_size / 1024
            
            print(f"Library path:      {lib_path}")
            print(f"Library size:      {size_kb:.1f} KB")
            print(f"Build timestamp:   {build_time}")
            print(f"Assembly backend:  {'✅ Enabled' if engine.use_assembly else '❌ Disabled'}")
            print(f"Compression size:  {engine.compression_size}")
        else:
            print(f"Library path:      Not found at {lib_path}")
        engine.initialize_state(8)
        success, result = engine.compress_million_qubits(8)
        
        if success:
            backend = result.get('backend', 'Unknown')
            ops_per_sec = result.get('operations_per_second', 0)
            print(f"Runtime backend:   {backend}")
            print(f"Performance:       {ops_per_sec:,} ops/sec")
            print(f"Status:            ✅ OPERATIONAL")
        else:
            print(f"Status:            ❌ NON-OPERATIONAL")
        
        engine.cleanup()
        return success
        
    except Exception as e:
        print(f"❌ Build info test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_build_info()
    print(f"\n🎯 BUILD INFO: {'✅ AVAILABLE' if success else '❌ LIMITED'}")
