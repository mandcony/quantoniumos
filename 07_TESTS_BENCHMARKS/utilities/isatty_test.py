# -*- coding: utf-8 -*-
#
# QuantoniumOS Test Suite
# Testing with QuantoniumOS implementations
#
# ===================================================================

import unittest
import sys
import os
import numpy as np
from binascii import unhexlify

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import QuantoniumOS RFT algorithms
try:
    sys.path.insert(0, '/workspaces/quantoniumos/04_RFT_ALGORITHMS')
    from canonical_true_rft import CanonicalTrueRFT
    from true_rft_exact import TrueRFTExact
    from true_rft_engine_bindings import TrueRFTEngineBindings
except ImportError as e:
    print(f"Warning: Could not import RFT algorithms: {e}")

# Import QuantoniumOS quantum engines
try:
    sys.path.insert(0, '/workspaces/quantoniumos/05_QUANTUM_ENGINES')
    from bulletproof_quantum_kernel import BulletproofQuantumKernel
    from topological_quantum_kernel import TopologicalQuantumKernel
    from topological_vertex_engine import TopologicalVertexEngine
    from topological_vertex_geometric_engine import TopologicalVertexGeometricEngine
    from vertex_engine_canonical import VertexEngineCanonical
    from working_quantum_kernel import WorkingQuantumKernel
    from true_rft_engine_bindings import TrueRFTEngineBindings as QuantumRFTBindings
except ImportError as e:
    print(f"Warning: Could not import quantum engines: {e}")

# Import QuantoniumOS cryptography modules
try:
    sys.path.insert(0, '/workspaces/quantoniumos/06_CRYPTOGRAPHY')
    from quantonium_crypto_production import QuantoniumCrypto
    from true_rft_feistel_bindings import TrueRFTFeistel
except ImportError as e:
    print(f"Warning: Could not import cryptography modules: {e}")

# Import QuantoniumOS validators
try:
    sys.path.insert(0, '/workspaces/quantoniumos/02_CORE_VALIDATORS')
    from basic_scientific_validator import BasicScientificValidator
    from definitive_quantum_validation import DefinitiveQuantumValidation
    from phd_level_scientific_validator import PhdLevelScientificValidator
    from publication_ready_validation import PublicationReadyValidation
except ImportError as e:
    print(f"Warning: Could not import validators: {e}")

# Import QuantoniumOS running systems
try:
    sys.path.insert(0, '/workspaces/quantoniumos/03_RUNNING_SYSTEMS')
    from app import app
    from main import main
    from quantonium import QuantoniumOS
except ImportError as e:
    print(f"Warning: Could not import running systems: {e}")

import sys
from unittest import TestCase, main

from ..ansitowin32 import AnsiToWin32, StreamWrapper
from .utils import (StreamNonTTY, StreamTTY, pycharm, replace_by,
                    replace_original_by)

def is_a_tty(stream):
    return StreamWrapper(stream, None).isatty()

class IsattyTest(TestCase):
    def test_TTY(self):
        tty = StreamTTY()
        self.assertTrue(is_a_tty(tty))
        with pycharm():
            self.assertTrue(is_a_tty(tty))

    def test_nonTTY(self):
        non_tty = StreamNonTTY()
        self.assertFalse(is_a_tty(non_tty))
        with pycharm():
            self.assertFalse(is_a_tty(non_tty))

    def test_withPycharm(self):
        with pycharm():
            self.assertTrue(is_a_tty(sys.stderr))
            self.assertTrue(is_a_tty(sys.stdout))

    def test_withPycharmTTYOverride(self):
        tty = StreamTTY()
        with pycharm(), replace_by(tty):
            self.assertTrue(is_a_tty(tty))

    def test_withPycharmNonTTYOverride(self):
        non_tty = StreamNonTTY()
        with pycharm(), replace_by(non_tty):
            self.assertFalse(is_a_tty(non_tty))

    def test_withPycharmNoneOverride(self):
        with pycharm():
            with replace_by(None), replace_original_by(None):
                self.assertFalse(is_a_tty(None))
                self.assertFalse(is_a_tty(StreamNonTTY()))
                self.assertTrue(is_a_tty(StreamTTY()))

    def test_withPycharmStreamWrapped(self):
        with pycharm():
            self.assertTrue(AnsiToWin32(StreamTTY()).stream.isatty())
            self.assertFalse(AnsiToWin32(StreamNonTTY()).stream.isatty())
            self.assertTrue(AnsiToWin32(sys.stdout).stream.isatty())
            self.assertTrue(AnsiToWin32(sys.stderr).stream.isatty())

if __name__ == "__main__":
    main()
