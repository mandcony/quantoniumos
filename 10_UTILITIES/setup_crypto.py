from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "enhanced_rft_crypto_bindings_fixed",
        [
            "paper_compliant_crypto_bindings.cpp",
        ],
        include_dirs=[
            # Path to pybind11 headers
        ],
        language="c++",
        cxx_std="14",
    ),
]

setup(
    name="enhanced_rft_crypto_bindings_fixed",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.6",
)
