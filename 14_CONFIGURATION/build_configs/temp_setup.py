import pybind11
from pybind11 import get_cmake_dir
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import Extension, setup

ext_modules = [
    Pybind11Extension(
        "enhanced_rft_crypto_bindings",
        [
            "enhanced_rft_crypto_wrapper.cpp",
        ],
        include_dirs=[
            pybind11.get_include(),
        ],
        language="c++",
        cxx_std=17,
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
