# -*- coding: utf-8 -*-
#
# QuantoniumOS Cryptography Tests
# Testing with QuantoniumOS crypto implementations
#
# ===================================================================

import unittest
import sys
import os
from binascii import unhexlify

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import QuantoniumOS cryptography modules
try:
    from quantonium_crypto_production import QuantoniumCrypto
    from true_rft_feistel_bindings import TrueRFTFeistel
    from paper_compliant_crypto_bindings import PaperCompliantCrypto
except ImportError:
    # Fallback imports if modules are in different locations
    sys.path.insert(0, '/workspaces/quantoniumos/06_CRYPTOGRAPHY')
    from quantonium_crypto_production import QuantoniumCrypto
    from true_rft_feistel_bindings import TrueRFTFeistel

# Import QuantoniumOS RFT algorithms
try:
    sys.path.insert(0, '/workspaces/quantoniumos/04_RFT_ALGORITHMS')
    from canonical_true_rft import CanonicalTrueRFT
    from true_rft_exact import TrueRFTExact
    from true_rft_engine_bindings import TrueRFTEngineBindings
except ImportError:
    pass

# Import QuantoniumOS quantum engines
try:
    sys.path.insert(0, '/workspaces/quantoniumos/05_QUANTUM_ENGINES')
    from bulletproof_quantum_kernel import BulletproofQuantumKernel
    from topological_quantum_kernel import TopologicalQuantumKernel
    from working_quantum_kernel import WorkingQuantumKernel
except ImportError:
    pass

import subprocess
import sys
import os

import pytest

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# QuantoniumOS public modules
PUBLIC_MODULES = [
    '04_RFT_ALGORITHMS.canonical_true_rft',
    '04_RFT_ALGORITHMS.true_rft_engine_bindings', 
    '04_RFT_ALGORITHMS.true_rft_exact',
    '05_QUANTUM_ENGINES.bulletproof_quantum_kernel',
    '05_QUANTUM_ENGINES.topological_quantum_kernel',
    '05_QUANTUM_ENGINES.topological_vertex_engine',
    '05_QUANTUM_ENGINES.topological_vertex_geometric_engine',
    '05_QUANTUM_ENGINES.vertex_engine_canonical',
    '05_QUANTUM_ENGINES.working_quantum_kernel',
    '06_CRYPTOGRAPHY.quantonium_crypto_production',
    '06_CRYPTOGRAPHY.true_rft_feistel_bindings',
]

# Test for import cycles in QuantoniumOS modules
# Check that all modules are importable in a new Python process.
# This is not necessarily true if there are import cycles present.

@pytest.mark.fail_slow(40)
@pytest.mark.slow
@pytest.mark.thread_unsafe
def test_public_modules_importable():
    """Test that all QuantoniumOS public modules can be imported without cycles."""
    pids = [
        subprocess.Popen([sys.executable, "-c", f"import sys; sys.path.insert(0, '/workspaces/quantoniumos'); import {module}"])
        for module in PUBLIC_MODULES
    ]
    for i, pid in enumerate(pids):
        assert pid.wait() == 0, f"Failed to import {PUBLIC_MODULES[i]}"
