"""
Fixed build script for Windows environment
"""
import os
import platform
import subprocess
import sys
from pathlib import Path


def main():
    print("Building fixed True RFT Engine bindings with proper normalization...")

    # Import required libraries for includes
    try:
        import numpy

        numpy_include = Path(numpy.get_include())
        print(f"NumPy include: {numpy_include}")
    except ImportError:
        print("NumPy not found. Please install it: pip install numpy")
        return 1

    try:
        import pybind11

        pybind11_include = Path(pybind11.get_include())
        print(f"pybind11 include: {pybind11_include}")
    except ImportError:
        print("pybind11 not found. Please install it: pip install pybind11")
        return 1

    # Get Python information
    python_version = f"{sys.version_info.major}{sys.version_info.minor}"
    python_include = Path(sys.prefix) / "include"
    python_libs = Path(sys.prefix) / "libs"

    print(f"Python include: {python_include}")
    print(f"Python libs: {python_libs}")

    # Source file
    source_file = Path("true_rft_engine_bindings_fixed.cpp")
    if not source_file.exists():
        print(f"Error: Source file {source_file} not found!")
        return 1

    # Output file
    output_file = Path(
        "true_rft_engine_bindings.cp{}-win_amd64.pyd".format(python_version)
    )

    # Core includes and libraries
    core_include = Path("core/include")
    core_lib = Path("core/lib")

    # Check for MSVC cl.exe
    try:
        subprocess.run(["cl", "/?"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Found MSVC compiler")
    except FileNotFoundError:
        print(
            "MSVC compiler (cl.exe) not found in PATH. Please run this from an MSVC developer command prompt."
        )
        print(
            "If using VS Code, you can open a Developer PowerShell from the Terminal menu."
        )
        return 1

    # Collect compile flags
    compile_flags = [
        "/EHsc",  # Enable C++ exceptions
        "/std:c++14",  # Use C++14 standard
        "/LD",  # Create a DLL
        "/O2",  # Optimize for speed
        "/DNDEBUG",  # Define NDEBUG
        f"/I{python_include}",
        f"/I{numpy_include}",
        f"/I{pybind11_include}",
    ]

    # Add core includes if they exist
    if core_include.exists():
        compile_flags.append(f"/I{core_include}")

    # Link libraries
    link_flags = [
        f"/LIBPATH:{python_libs}",
        f"python{python_version}.lib",
    ]

    # Add core libs if they exist
    if core_lib.exists():
        engine_lib = core_lib / "engine_core.lib"
        if engine_lib.exists():
            link_flags.append(f"/LIBPATH:{core_lib}")
            link_flags.append("engine_core.lib")

    # Build the command
    cmd = (
        ["cl"]
        + compile_flags
        + [str(source_file)]
        + link_flags
        + [f"/Fe:{output_file}"]
    )

    # Print the command
    print(f"Running: {' '.join(map(str, cmd))}")

    # Run the build
    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully built {output_file}")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Build failed with error code {e.returncode}")
        return e.returncode


if __name__ == "__main__":
    sys.exit(main())
