#!/usr/bin/env python3
"""
QuantoniumOS Main Launcher
=========================
Precision-engineered launcher for your quantum OS
"""

import sys
import os

# Ensure correct paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
sys.path.insert(0, FRONTEND_DIR)

def check_dependencies():
    """Check if all required dependencies are available"""
    missing_deps = []
    
    try:
        import PyQt5
    except ImportError:
        missing_deps.append("PyQt5")
    
    try:
        import qtawesome
    except ImportError:
        missing_deps.append("qtawesome")
    
    try:
        import pytz
    except ImportError:
        missing_deps.append("pytz")
    
    try:
        import psutil
    except ImportError:
        missing_deps.append("psutil")
    
    return missing_deps

def check_rft_assembly():
    """Check if RFT assembly is available"""
    assembly_path = os.path.join(BASE_DIR, "ASSEMBLY", "python_bindings")
    rft_dll = os.path.join(BASE_DIR, "ASSEMBLY", "compiled", "librftkernel.dll")
    
    assembly_available = os.path.exists(assembly_path)
    dll_available = os.path.exists(rft_dll)
    
    return assembly_available, dll_available

def print_system_status():
    """Print system status"""
    print("=" * 60)
    print("QuantoniumOS - Quantum Computing Operating System")
    print("=" * 60)
    
    # Check dependencies
    missing_deps = check_dependencies()
    if missing_deps:
        print("❌ Missing dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\nInstall with: pip install PyQt5 qtawesome pytz psutil")
        return False
    else:
        print("✅ All Python dependencies available")
    
    # Check RFT assembly
    assembly_available, dll_available = check_rft_assembly()
    if assembly_available and dll_available:
        print("✅ RFT Assembly available")
        print(f"   - Python bindings: {assembly_available}")
        print(f"   - Compiled kernel: {dll_available}")
    else:
        print("⚠️ RFT Assembly status:")
        print(f"   - Python bindings: {assembly_available}")
        print(f"   - Compiled kernel: {dll_available}")
    
    # Check quantum engines
    engines_dir = os.path.join(BASE_DIR, "engines")
    core_dir = os.path.join(BASE_DIR, "core")
    
    if os.path.exists(engines_dir) and os.path.exists(core_dir):
        print("✅ Quantum engines available")
    else:
        print("❌ Quantum engines missing")
    
    print("=" * 60)
    return True

def launch_quantonium_os():
    """Launch the main QuantoniumOS interface"""
    if not print_system_status():
        print("Cannot launch QuantoniumOS due to missing dependencies")
        return
    
    print("🚀 Launching QuantoniumOS Desktop...")
    
    try:
        from quantonium_desktop import main as desktop_main
        desktop_main()
    except ImportError as e:
        print(f"❌ Failed to import desktop module: {e}")
    except Exception as e:
        print(f"❌ Failed to launch QuantoniumOS: {e}")

if __name__ == "__main__":
    launch_quantonium_os()
