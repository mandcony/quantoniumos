#!/usr/bin/env python3
"""
Comprehensive C++ Engine Builder for QuantoniumOS
Builds all missing C++ engines that the comprehensive test requires
"""

import os
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import Extension, setup


def build_engine(engine_name, source_files, include_dirs=None, dependencies=None):
    """Build a single C++ engine"""

    if include_dirs is None:
        include_dirs = []

    # Add core directory to include paths
    include_dirs.extend(["core/", "src/", "./"])

    print(f"🔧 Building {engine_name}...")

    try:
        ext_modules = [
            Pybind11Extension(
                engine_name,
                source_files,
                language="c++",
                cxx_std=17,
                include_dirs=include_dirs,
            )
        ]

        # Create a minimal setup for this engine
        setup(
            name=engine_name,
            ext_modules=ext_modules,
            cmdclass={"build_ext": build_ext},
            zip_safe=False,
            script_args=["build_ext", "--inplace"],
        )

        print(f"✅ {engine_name} built successfully")
        return True

    except Exception as e:
        print(f"❌ Failed to build {engine_name}: {e}")
        return False


def main():
    """Build all required C++ engines"""

    print("🚀 COMPREHENSIVE C++ ENGINE BUILDER")
    print("=" * 50)

    # Define all engines to build
    engines_to_build = [
        {
            "name": "resonance_engine",
            "sources": ["core/resonance_engine_bindings.cpp"],
            "includes": ["core/"],
        },
        {
            "name": "vertex_engine",
            "sources": ["core/vertex_engine_bindings.cpp"],
            "includes": ["core/"],
        },
        {
            "name": "quantonium_core",
            "sources": ["core/engine_core.cpp"],  # Will check if exists
            "includes": ["core/"],
        },
        {
            "name": "quantum_engine",
            "sources": ["core/quantum_engine_bindings.cpp"],
            "includes": ["core/"],
        },
    ]

    built_engines = []
    failed_engines = []

    for engine_config in engines_to_build:
        name = engine_config["name"]
        sources = engine_config["sources"]
        includes = engine_config.get("includes", [])

        # Check if source files exist
        missing_sources = []
        for source in sources:
            if not os.path.exists(source):
                missing_sources.append(source)

        if missing_sources:
            print(f"⚠️ Skipping {name}: Missing sources {missing_sources}")
            failed_engines.append(name)
            continue

        # Try to build
        if build_engine(name, sources, includes):
            built_engines.append(name)
        else:
            failed_engines.append(name)

    # Summary
    print("\n📊 BUILD SUMMARY")
    print("=" * 50)

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
        print("\n🎉 SUFFICIENT ENGINES BUILT FOR COMPREHENSIVE TESTING!")
        return True
    else:
        print("\n🚨 INSUFFICIENT ENGINES BUILT")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
