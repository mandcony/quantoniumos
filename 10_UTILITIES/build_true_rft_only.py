#!/usr/bin/env python3
"""
Quick build script for just the true RFT engine
"""
import os
import sys

import pybind11
import setuptools
from pybind11 import get_cmake_dir
from pybind11.setup_helpers import Pybind11Extension, build_ext


def build_true_rft():
    print("🔧 Building true_rft_engine_canonical")

    # Define the extension
    ext = Pybind11Extension(
        "true_rft_engine_canonical",
        [
            "04_RFT_ALGORITHMS/true_rft_engine_bindings.cpp",
            "04_RFT_ALGORITHMS/true_rft_engine.cpp",
            "core/engine_core_simple.cpp",
        ],
        include_dirs=[
            "core/",
            "core/include/",
            "04_RFT_ALGORITHMS/",
            "06_CRYPTOGRAPHY/",
        ],
        cxx_std=17,
        define_macros=[("VERSION_INFO", '"dev"')],
    )

    # Build it
    setuptools.setup(
        ext_modules=[ext],
        cmdclass={"build_ext": build_ext},
        zip_safe=False,
        script_args=["build_ext", "--inplace"],
    )


if __name__ == "__main__":
    build_true_rft()
