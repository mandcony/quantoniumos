||#!/usr/bin/env python3
""""""
Simple C++ engine builder using pybind11
""""""

import subprocess
import sys
import os
from pathlib import Path

def get_python_cmd():
    """"""Get the correct Python command.""""""
    return "/home/codespace/.python/current/bin/python"

def build_module(name, sources, includes=None):
    """"""Build a single pybind11 module.""""""
    if includes is None:
        includes = []

    cmd = [
        get_python_cmd(), "-c",
        f""""""
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
import pybind11

ext = Pybind11Extension(
    "{name}",
    {sources},
    include_dirs=[
        pybind11.get_cmake_dir(),
        "core/include",
        "/usr/include/eigen3",
        {includes}
    ],
    cxx_std=17,
    define_macros=[("VERSION_INFO", '"dev"')],
)

setup(
    ext_modules=[ext],
    cmdclass={{"build_ext": build_ext}},
    script_args=["build_ext", "--inplace"]
)
""""""
    ]

    print(f"Building {name}...")
    result = subprocess.run(cmd, shell=False)
    return result.returncode == 0

def main():
    print("🔨 Simple C++ Engine Builder")

    # Change to project root
    os.chdir('/workspaces/quantoniumos')

    success_count = 0

    # Build modules individually with specific sources
    modules = [
        ("quantonium_core", '["core/pybind_interface.cpp", "core/engine_core.cpp", "core/symbolic_eigenvector.cpp"]'),
        ("resonance_engine", '["core/resonance_engine_bindings.cpp", "core/engine_core.cpp", "core/symbolic_eigenvector.cpp"]'),
        ("quantum_engine", '["core/quantum_engine_bindings.cpp", "core/engine_core.cpp", "core/symbolic_eigenvector.cpp"]')
    ]

    for name, sources in modules:
        try:
            if build_module(name, sources):
                print(f"✓ {name} built successfully")
                success_count += 1
            else:
                print(f"❌ {name} build failed")
        except Exception as e:
            print(f"❌ {name} build error: {e}")

    print(f"\n🎯 Built {success_count}/{len(modules)} modules")

    # Test what we built
    if success_count > 0:
        print("||n🧪 Testing built modules...")
        for name, _ in modules:
            try:
                cmd = [get_python_cmd(), "-c", f"import {name}; print('✓ {name} imports successfully')"]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print(result.stdout.strip())
                else:
                    print(f"❌ {name} import failed: {result.stderr}")
            except Exception as e:
                print(f"❌ {name} test error: {e}")

if __name__ == "__main__":
    main()
