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

import networkx as nx
import pytest

np = pytest.importorskip("numpy")

@pytest.mark.parametrize(
    "k, weight, expected",
    [
        (None, None, 7.21),  # infers 3 communities
        (2, None, 11.7),
        (None, "weight", 25.45),
        (2, "weight", 38.8),
    ],
)
def test_non_randomness(k, weight, expected):
    G = nx.karate_club_graph()
    np.testing.assert_almost_equal(
        nx.non_randomness(G, k, weight)[0], expected, decimal=2
    )

def test_non_connected():
    G = nx.Graph([(1, 2)])
    G.add_node(3)
    with pytest.raises(nx.NetworkXException, match="Non connected"):
        nx.non_randomness(G)

def test_self_loops():
    G = nx.Graph()
    G.add_edge(1, 2)
    G.add_edge(1, 1)
    with pytest.raises(nx.NetworkXError, match="Graph must not contain self-loops"):
        nx.non_randomness(G)

def test_empty_graph():
    G = nx.empty_graph(1)
    with pytest.raises(nx.NetworkXError, match=".*not applicable to empty graphs"):
        nx.non_randomness(G)
