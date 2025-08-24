#!/usr/bin/env python3
"""
Setup script for QuantoniumOS C++ extensions
Minimal working build to verify RFT algorithm
"""
import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

# Define a single, minimal C++ extension for testing
ext_modules = [
    # Minimal test module to verify build system
    Pybind11Extension(
        "quantonium_test",
        ["core/minimal_test.cpp"],  # We'll create this simple file
        include_dirs=["core/include", pybind11.get_cmake_dir()],
        cxx_std=17,
        define_macros=[("VERSION_INFO", '"0.3.0"')],
    ),
]

if __name__ == "__main__":
    setup(
        name="quantonium-test",
        ext_modules=ext_modules,
        cmdclass={"build_ext": build_ext},
        zip_safe=False,
    )
