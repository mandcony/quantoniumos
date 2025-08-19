||#!/usr/bin/env python3
from canonical_true_rft
import forward_true_rft, inverse_true_rft """"""
QuantoniumOS C++ Engine Build Script This script builds all C++ engines using pybind11 and makes them available to Python with automatic fallback handling.
"""
"""

import os
import sys
import subprocess
import platform from pathlib
import Path
import shutil
def check_dependencies():
"""
"""
        Check
        if required dependencies are available.
"""
"""
        print(" Checking build dependencies...") required = []

        # Check for cmake
        try: result = subprocess.run(['cmake', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
        print("✓ CMake found:", result.stdout.split('\n')[0])
        else: required.append('cmake')
        except FileNotFoundError: required.append('cmake')

        # Check for C++ compiler compilers = ['g++', 'clang++', 'cl'] cpp_found = False
        for compiler in compilers:
        try: result = subprocess.run([compiler, '--version'], capture_output=True, text=True)
        if result.returncode == 0:
        print(f"✓ C++ compiler found: {compiler}") cpp_found = True break
        except FileNotFoundError: continue
        if not cpp_found: required.append('C++ compiler (g++/clang++/MSVC)')

        # Check for pybind11
        try:
import pybind11
        print(f"✓ pybind11 found: {pybind11.__version__}")
        except ImportError: required.append('pybind11')

        # Check for numpy
        try:
import numpy
        print(f"✓ numpy found: {numpy.__version__}")
        except ImportError: required.append('numpy')

        # Check for Eigen3 eigen_paths = [ '/usr/include/eigen3', '/usr/local/include/eigen3', '/opt/homebrew/include/eigen3', 'third_party/eigen', Path.cwd() / 'third_party' / 'eigen' ] eigen_found = False
        for path in eigen_paths:
        if Path(path).exists() and (Path(path) / 'Eigen').exists():
        print(f"✓ Eigen3 found: {path}") eigen_found = True break
        if not eigen_found: required.append('eigen3')
        if required:
        print("\n❌ Missing dependencies:")
        for dep in required:
        print(f" - {dep}")
        print("\nInstall missing dependencies:")
        print("Ubuntu/Debian: sudo apt-get install cmake build-essential libeigen3-dev python3-pybind11")
        print("macOS: brew install cmake eigen pybind11")
        print("Python packages: pip install pybind11 numpy")
        return False
        print("✅ All dependencies satisfied")
        return True
def setup_build_directory(): """"""
        Set up build directory.
"""
"""
        build_dir = Path.cwd() / 'build' build_dir.mkdir(exist_ok=True)
        return build_dir
def build_with_cmake():
"""
"""
        Build using CMake.
"""
"""
        print("\n🔨 Building with CMake...") build_dir = setup_build_directory() os.chdir(build_dir)

        # Configure cmake_args = [ 'cmake', '..', '-DCMAKE_BUILD_TYPE=Release', '-DPYTHON_EXECUTABLE=' + sys.executable ]

        # Add platform-specific optimizations
        if platform.system() == 'Darwin': # macOS cmake_args.extend([ '-DCMAKE_OSX_ARCHITECTURES=arm64;x86_64'

        # Universal binary ])
        print("Configuring build...") result = subprocess.run(cmake_args, capture_output=True, text=True)
        if result.returncode != 0:
        print("❌ CMake configuration failed:")
        print(result.stderr)
        return False
        print("✓ Configuration successful")

        # Build
        print("Building C++ modules...") build_result = subprocess.run(['cmake', '--build', '.', '--config', 'Release'], capture_output=True, text=True)
        if build_result.returncode != 0:
        print("❌ Build failed:")
        print(build_result.stderr)
        return False
        print("✅ Build successful")
        return True
def build_with_setuptools_fallback(): """"""
        Fallback build using setuptools and pybind11.
"""
"""
        print("\n🔨 Building with setuptools fallback...")
        try: from pybind11.setup_helpers
import Pybind11Extension, build_ext from setuptools
import setup, Extension
import pybind11
        except ImportError:
        print("❌ pybind11 not available for fallback build")
        return False

        # Define extensions extensions = [ Pybind11Extension( "quantonium_core", ["core/pybind_interface.cpp", "core/engine_core.cpp", "core/symbolic_eigenvector.cpp"], include_dirs=[ pybind11.get_cmake_dir(), "core/include", "third_party/eigen" ], cxx_std=17, define_macros=[("VERSION_INFO", '"dev"')], ), Pybind11Extension( "resonance_engine", ["core/resonance_engine_bindings.cpp", "core/engine_core.cpp", "core/symbolic_eigenvector.cpp"], include_dirs=[ pybind11.get_cmake_dir(), "core/include", "third_party/eigen" ], cxx_std=17, define_macros=[("VERSION_INFO", '"dev"')], ), Pybind11Extension( "quantum_engine", ["core/quantum_engine_bindings.cpp", "core/engine_core.cpp", "core/symbolic_eigenvector.cpp"], include_dirs=[ pybind11.get_cmake_dir(), "core/include", "third_party/eigen" ], cxx_std=17, define_macros=[("VERSION_INFO", '"dev"')], ), ]

        # Build in place setup( ext_modules=extensions, cmdclass={"build_ext": build_ext}, script_name="setup.py", script_args=["build_ext", "--inplace"] )
        print("✅ Setuptools build successful")
        return True
def install_modules(): """"""
        Install/copy built modules to the correct locations.
"""
"""
        print("\n📦 Installing modules...")

        # Look for built modules possible_extensions = ['.so', '.pyd', '.dll'] modules = ['quantonium_core', 'resonance_engine', 'quantum_engine'] installed_count = 0
        for module in modules: found = False
        for ext in possible_extensions:

        # Check in build directory build_path = Path('build') / f"{module}{ext}"
        if build_path.exists():

        # Copy to core directory dest_path = Path('core') / f"{module}{ext}" shutil.copy2(build_path, dest_path)
        print(f"✓ Installed {module}{ext}") found = True installed_count += 1 break

        # Check in current directory (setuptools build) current_path = Path(f"{module}{ext}")
        if current_path.exists():

        # Copy to core directory dest_path = Path('core') / f"{module}{ext}" shutil.copy2(current_path, dest_path)
        print(f"✓ Installed {module}{ext}") found = True installed_count += 1 break
        if not found:
        print(f"⚠️ Module {module} not found")
        return installed_count > 0
def test_engines(): """"""
        Test the built engines.
"""
"""
        print("\n🧪 Testing engines...")

        # Add core directory to path sys.path.insert(0, str(Path.cwd() / 'core')) test_results = {}

        # Test resonance engine
        try:
import resonance_engine engine = resonance_engine.ResonanceFourierEngine() result = engine.forward_true_rft([1.0, 0.5, -0.3, 0.8]) test_results['resonance_engine'] = f"✓ Working ({len(result)} coefficients)" except Exception as e: test_results['resonance_engine'] = f"❌ Failed: {e}"

        # Test quantum engine
        try:
import quantum_engine qe = quantum_engine.QuantumEntropyEngine() entropy = qe.generate_quantum_entropy(10) test_results['quantum_engine'] = f"✓ Working ({len(entropy)} entropy values)" except Exception as e: test_results['quantum_engine'] = f"❌ Failed: {e}"

        # Test core engine
        try:
import quantonium_core rft = quantonium_core.ResonanceFourierTransform([1.0, 0.5, -0.3, 0.8]) freq = rft.forward_transform() test_results['quantonium_core'] = f"✓ Working ({len(freq)} frequencies)" except Exception as e: test_results['quantonium_core'] = f"❌ Failed: {e}"
        print("Test Results:") for engine, result in test_results.items():
        print(f" {engine}: {result}") working_engines = sum(1
        for result in test_results.values()
        if result.startswith('✓'))
        return working_engines > 0
def main(): """"""
        Main build process.
"""
"""
        print(" QuantoniumOS C++ Engine Build System")
        print("=" * 50)

        # Check dependencies
        if not check_dependencies(): sys.exit(1)

        # Try CMake build first build_success = False
        if shutil.which('cmake'): build_success = build_with_cmake()
        if build_success: os.chdir('..')

        # Return to root directory

        # Fallback to setuptools
        if CMake failed
        if not build_success:
        print("CMake build failed, trying setuptools fallback...")
        try: build_success = build_with_setuptools_fallback() except Exception as e:
        print(f"❌ Setuptools build also failed: {e}")
        if not build_success:
        print("\n❌ All build methods failed!")
        print("The system will fall back to Python implementations.") sys.exit(1)

        # Install modules
        if install_modules():
        print("✅ Modules installed successfully")
        else:
        print("⚠️ Module installation issues")

        # Test engines
        if test_engines():
        print("\n🎉 Build successful! C++ engines are ready.")
        print("\nNext steps:")
        print("1. Run: python -c \"from core.high_performance_engine
import get_engine_status;
        print(get_engine_status())\"")
        print("2. Your Python code will now automatically use high-performance C++ implementations")
        else:
        print("||n⚠️ Build completed but engines have issues.")
        print("The system will fall back to Python implementations.")

if __name__ == "__main__": main()