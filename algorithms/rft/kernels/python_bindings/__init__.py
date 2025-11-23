#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
QuantoniumOS Assembly Bindings - Robust Import System
======================================================

This module provides robust loading of C/Assembly libraries with fallback handling
for different platforms and build configurations.
"""

import os
import sys
import ctypes
import platform
from typing import Optional, List, Tuple

# Global flag for assembly availability
ASSEMBLY_AVAILABLE = False
ASSEMBLY_PATH = None

def _find_library_paths() -> List[str]:
    """Find all possible library paths for the current platform with comprehensive search."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    system = platform.system().lower()

    # Highest priority: explicit override via environment variable
    env_lib = os.environ.get('RFT_KERNEL_LIB')
    if env_lib and os.path.exists(env_lib):
        return [env_lib]

    # Base library names with comprehensive variants
    lib_names = []
    if system == 'windows':
        lib_names = [
            'quantum_symbolic.dll',
            'libquantum_symbolic.dll',
            'rft_kernel.dll',
            'librft_kernel.dll',
            'unitary_rft.dll',
            'libunitary_rft.dll'
        ]
    elif system == 'linux':
        lib_names = [
            'libquantum_symbolic.so',
            'libquantum_symbolic.so.1',
            'librft_kernel.so',
            'librft_kernel.so.1',
            'libunitary_rft.so',
            'libunitary_rft.so.1'
        ]
    elif system == 'darwin':  # macOS
        lib_names = [
            'libquantum_symbolic.dylib',
            'libquantum_symbolic.1.dylib',
            'librft_kernel.dylib',
            'librft_kernel.1.dylib',
            'libunitary_rft.dylib',
            'libunitary_rft.1.dylib'
        ]
    else:
        # Generic fallback
        lib_names = ['libquantum_symbolic.so', 'libquantum_symbolic.dylib', 'quantum_symbolic.dll']

    # Search paths in priority order - comprehensive coverage
    search_paths = [
        # Local compiled libraries (highest priority)
        os.path.join(script_dir, '..', 'compiled'),
        os.path.join(script_dir, '..', 'build'),
        os.path.join(script_dir, '..', 'Release'),
        os.path.join(script_dir, '..', 'Debug'),
        os.path.join(script_dir, '..', 'bin'),
        os.path.join(script_dir, '..', 'lib'),

        # Development builds
        os.path.join(script_dir, '..', '..', '..', 'build'),
        os.path.join(script_dir, '..', '..', '..', 'compiled'),
        os.path.join(script_dir, '..', '..', '..', 'Release'),
        os.path.join(script_dir, '..', '..', '..', 'Debug'),
        os.path.join(script_dir, '..', '..', '..', 'bin'),
        os.path.join(script_dir, '..', '..', '..', 'lib'),

    # Repository root builds
        os.path.join(script_dir, '..', '..', '..', '..', 'build'),
        os.path.join(script_dir, '..', '..', '..', '..', 'compiled'),
        os.path.join(script_dir, '..', '..', '..', '..', 'Release'),
        os.path.join(script_dir, '..', '..', '..', '..', 'Debug'),

    # QuantoniumOS common Windows build output locations
    os.path.join(script_dir, '..', '..', '..', '..', 'src', 'assembly', 'compiled'),
    os.path.join(script_dir, '..', '..', '..', '..', 'system', 'assembly', 'assembly', 'compiled'),

        # System paths
        '/usr/local/lib',
        '/usr/lib',
        '/usr/lib64',
        '/opt/local/lib',
        '/lib',
        '/lib64',

        # Windows-specific paths
        'C:\\Program Files\\QuantoniumOS\\lib',
        'C:\\Program Files (x86)\\QuantoniumOS\\lib',
        os.path.join(os.environ.get('PROGRAMFILES', 'C:\\Program Files'), 'QuantoniumOS', 'lib'),
        os.path.join(os.environ.get('PROGRAMFILES(X86)', 'C:\\Program Files (x86)'), 'QuantoniumOS', 'lib'),

        # Current and script directory
        '.',
        script_dir,
        os.getcwd(),
    ]

    # Add Python site-packages paths
    try:
        import site
        search_paths.extend(site.getsitepackages())
    except:
        pass

    # Add PATH environment variable directories
    path_env = os.environ.get('PATH', '')
    if path_env:
        search_paths.extend(path_env.split(os.pathsep))

    # Remove duplicates while preserving order
    seen = set()
    unique_paths = []
    for path in search_paths:
        if path and path not in seen:
            seen.add(path)
            unique_paths.append(path)

    library_paths = []
    for base_path in unique_paths:
        if not os.path.exists(base_path):
            continue
        for lib_name in lib_names:
            full_path = os.path.join(base_path, lib_name)
            if os.path.exists(full_path):
                library_paths.append(full_path)

    return library_paths

def _load_assembly_library() -> Optional[ctypes.CDLL]:
    """Load the assembly library with robust error handling and diagnostics."""
    global ASSEMBLY_AVAILABLE, ASSEMBLY_PATH

    library_paths = _find_library_paths()

    if not library_paths:
        # Keep quiet here; higher-level loaders may use env overrides
        return None

    print(f"INFO: Found {len(library_paths)} potential library paths")

    for lib_path in library_paths:
        try:
            print(f"INFO: Attempting to load: {lib_path}")
            lib = ctypes.CDLL(lib_path)

            # Test basic functionality
            if hasattr(lib, 'rft_init') or hasattr(lib, 'quantum_symbolic_init'):
                ASSEMBLY_AVAILABLE = True
                ASSEMBLY_PATH = lib_path
                print(f"SUCCESS: Assembly library loaded from {lib_path}")
                return lib
            else:
                print(f"WARNING: Library loaded but missing expected functions: {lib_path}")
                continue

        except (OSError, AttributeError) as e:
            print(f"INFO: Failed to load {lib_path}: {e}")
            continue
        except Exception as e:
            print(f"ERROR: Unexpected error loading {lib_path}: {e}")
            continue

    print("WARNING: No compatible assembly library found - using fallback mode")
    return None

def _create_mock_library():
    """Create a mock library for fallback when assembly is not available."""
    class MockLibrary:
        def __getattr__(self, name):
            def mock_function(*args, **kwargs):
                raise RuntimeError(f"Assembly function '{name}' not available - library not compiled")
            return mock_function

    return MockLibrary()

# Load the library
_library = _load_assembly_library()
if _library is None:
    _library = _create_mock_library()

# Export the library and status
library = _library
ASSEMBLY_AVAILABLE = ASSEMBLY_AVAILABLE
ASSEMBLY_PATH = ASSEMBLY_PATH

# Try to import Python wrappers with fallbacks
try:
    from .unitary_rft import UnitaryRFT, RFT_FLAG_DEFAULT, RFT_FLAG_UNITARY
    RFT_AVAILABLE = True
except ImportError:
    RFT_AVAILABLE = False
    # Create mock classes for fallback
    class UnitaryRFT:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("RFT Assembly not available - library not compiled")

    RFT_FLAG_DEFAULT = 0x00000000
    RFT_FLAG_UNITARY = 0x00000008

try:
    from .optimized_rft import OptimizedRFT, RFT_OPT_AVX2, RFT_OPT_AVX512
    OPTIMIZED_RFT_AVAILABLE = True
except ImportError:
    OPTIMIZED_RFT_AVAILABLE = False
    # Create mock classes for fallback
    class OptimizedRFT:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Optimized RFT Assembly not available - library not compiled")

    RFT_OPT_AVX2 = 0x00000001
    RFT_OPT_AVX512 = 0x00000002

try:
    from .quantum_symbolic_engine import QuantumSymbolicEngine
    QUANTUM_SYMBOLIC_AVAILABLE = True
except ImportError:
    QUANTUM_SYMBOLIC_AVAILABLE = False
    # Create mock classes for fallback
    class QuantumSymbolicEngine:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Quantum Symbolic Engine not available - library not compiled")

# RFT Constants (fallback definitions when assembly not available)
RFT_FLAG_DEFAULT = 0x00
RFT_FLAG_UNITARY = 0x01
RFT_FLAG_QUANTUM_SAFE = 0x02
RFT_FLAG_USE_RESONANCE = 0x04

RFT_OPT_AVX2 = 0x10
RFT_OPT_AVX512 = 0x20

# Export all available components
__all__ = [
    'library',
    'ASSEMBLY_AVAILABLE',
    'ASSEMBLY_PATH',
    'RFT_AVAILABLE',
    'OPTIMIZED_RFT_AVAILABLE',
    'QUANTUM_SYMBOLIC_AVAILABLE',
    'UnitaryRFT',
    'OptimizedRFT',
    'QuantumSymbolicEngine',
    'RFT_FLAG_DEFAULT',
    'RFT_FLAG_UNITARY',
    'RFT_OPT_AVX2',
    'RFT_OPT_AVX512',
]