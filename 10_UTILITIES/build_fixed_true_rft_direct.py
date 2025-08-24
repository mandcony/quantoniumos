"""
Build the fixed True RFT Engine bindings with proper normalization.
This script builds the C++ extension using direct compilation commands.
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    print("Building fixed True RFT Engine bindings with proper normalization...")

    # Get the current directory
    current_dir = Path(os.getcwd())

    # Source file path
    source_file = current_dir / "true_rft_engine_bindings_fixed.cpp"

    # Ensure the source file exists
    if not source_file.exists():
        print(f"Error: Source file {source_file} not found!")
        return 1

    # Detect the operating system
    is_windows = platform.system() == "Windows"

    # Setup include paths
    include_paths = [
        current_dir / "core" / "include",
        current_dir / "core" / "cpp" / "engines",
        # Add Python include paths
        Path(sys.prefix) / "include",
        Path(sys.exec_prefix) / "include",
    ]

    # Add numpy include path
    try:
        import numpy

        include_paths.append(Path(numpy.get_include()))
    except ImportError:
        print("Warning: NumPy not found. NumPy headers will not be included.")

    # Add pybind11 include path
    try:
        import pybind11

        include_paths.append(Path(pybind11.get_include()))
        include_paths.append(Path(pybind11.get_include(user=True)))
    except ImportError:
        print("Warning: pybind11 not found. Please install it: pip install pybind11")
        return 1

    # Output file name
    output_ext = ".pyd" if is_windows else ".so"
    output_file = current_dir / f"true_rft_engine_bindings{output_ext}"

    # Prepare the command based on the platform
    if is_windows:
        # Windows/MSVC build command
        cmd = ["cl", "/EHsc", "/std:c++14", "/LD", "/O2", "/DNDEBUG"]

        # Add include paths
        for path in include_paths:
            if path.exists():
                cmd.append(f"/I{path}")

        # Add Python library path
        python_lib = Path(sys.exec_prefix) / "libs"
        if python_lib.exists():
            cmd.append(f"/LIBPATH:{python_lib}")

        # Add source file and output file
        cmd.append(str(source_file))
        cmd.append(f"/Fe:{output_file}")

        # Link with engine core if available
        engine_lib = current_dir / "core" / "lib" / "engine_core.lib"
        if engine_lib.exists():
            cmd.append(str(engine_lib))

    else:
        # Unix build command
        cmd = ["g++", "-O3", "-Wall", "-shared", "-std=c++14", "-fPIC"]

        # Add include paths
        for path in include_paths:
            if path.exists():
                cmd.append(f"-I{path}")

        # Add source file and output file
        cmd.append(str(source_file))
        cmd.append("-o")
        cmd.append(str(output_file))

        # Link with engine core if available
        engine_lib = current_dir / "core" / "lib" / "libengine_core.so"
        if engine_lib.exists():
            cmd.append(f"-L{current_dir / 'core' / 'lib'}")
            cmd.append("-lengine_core")

    # Run the build command
    print(f"Running build command: {' '.join(map(str, cmd))}")
    try:
        result = subprocess.run(cmd, check=True)
        if result.returncode == 0:
            print(f"Successfully built {output_file}")
            return 0
        else:
            print(f"Build failed with error code {result.returncode}")
            return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Build failed: {e}")
        return e.returncode
    except Exception as e:
        print(f"Error during build: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
