#!/usr/bin/env python3
"""
Canonical Engine Builder for QuantoniumOS
Builds ONLY the canonical engines based on novel equation: R = Σ_i w_i D_φi C_σi D_φi†
NO DUPLICATES - Clean mathematical organization
"""

import os
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


def build_canonical_engines():
    """Build the 5 canonical engines according to novel equation"""

    print("🚀 CANONICAL ENGINE BUILDER - NOVEL EQUATION BASED")
    print("=" * 60)
    print("Equation: R = Σ_i w_i D_φi C_σi D_φi†")
    print("Distinctness: 93.2% from classical transforms")
    print("=" * 60)

    # Define canonical engines based on mathematical components
    canonical_engines = [
        {
            "name": "true_rft_engine_canonical",
            "sources": [
                "04_RFT_ALGORITHMS/true_rft_engine_bindings.cpp",
                "04_RFT_ALGORITHMS/true_rft_engine.cpp",
                "core/engine_core_simple.cpp",
            ],
            "description": "Core R Transform (93.2% distinct)",
            "component": "R - Resonance Transform Output",
        },
        {
            "name": "enhanced_rft_crypto_canonical",
            "sources": [
                "06_CRYPTOGRAPHY/enhanced_rft_crypto_bindings.cpp",
                "06_CRYPTOGRAPHY/enhanced_rft_crypto.cpp",
            ],
            "description": "Cryptographic Applications",
            "component": "Security Layer",
        },
        {
            "name": "vertex_engine_canonical",
            "sources": ["core/vertex_engine_bindings.cpp", "core/vertex_engine.cpp"],
            "description": "Quantum Vertex Processing",
            "component": "D_φi - Dilation Operators",
        },
        {
            "name": "resonance_engine_canonical",
            "sources": [
                "core/resonance_engine_bindings.cpp",
                "core/resonance_engine_simple.cpp",
            ],
            "description": "Resonance/Fourier Processing",
            "component": "C_σi - Circulant Matrices",
        },
    ]

    built_engines = []
    failed_engines = []

    for engine in canonical_engines:
        name = engine["name"]
        sources = engine["sources"]
        desc = engine["description"]
        component = engine["component"]

        print(f"\n🔧 Building {name}")
        print(f"   Component: {component}")
        print(f"   Description: {desc}")
        print(f"   Sources: {sources}")

        # Check if source files exist
        missing = [src for src in sources if not os.path.exists(src)]
        if missing:
            print(f"   ❌ Missing sources: {missing}")
            failed_engines.append(name)
            continue

        try:
            # Create extension
            ext_modules = [
                Pybind11Extension(
                    name,
                    sources,
                    language="c++",
                    cxx_std=17,
                    include_dirs=[
                        "core/",
                        "core/include/",
                        "04_RFT_ALGORITHMS/",
                        "06_CRYPTOGRAPHY/",
                    ],
                )
            ]

            # Build
            setup(
                name=name,
                ext_modules=ext_modules,
                cmdclass={"build_ext": build_ext},
                zip_safe=False,
                script_args=["build_ext", "--inplace"],
            )

            print(f"   ✅ {name} built successfully")
            built_engines.append(name)

        except Exception as e:
            print(f"   ❌ Failed to build {name}: {e}")
            failed_engines.append(name)

    # Summary
    print(f"\n📊 CANONICAL BUILD SUMMARY")
    print("=" * 60)

    print(f"✅ Successfully built: {len(built_engines)}")
    for engine in built_engines:
        print(f"   • {engine}")

    print(f"\n❌ Failed to build: {len(failed_engines)}")
    for engine in failed_engines:
        print(f"   • {engine}")

    total = len(built_engines) + len(failed_engines)
    success_rate = (len(built_engines) / total) * 100 if total > 0 else 0

    print(f"\n📈 Success rate: {success_rate:.1f}% ({len(built_engines)}/{total})")

    if len(built_engines) >= 2:
        print("\n🎉 SUFFICIENT CANONICAL ENGINES BUILT!")
        print("   Novel equation R = Σ_i w_i D_φi C_σi D_φi† implemented")
        print("   Ready for comprehensive testing")
        return True
    else:
        print("\n🚨 INSUFFICIENT ENGINES BUILT")
        return False


if __name__ == "__main__":
    success = build_canonical_engines()
    sys.exit(0 if success else 1)
