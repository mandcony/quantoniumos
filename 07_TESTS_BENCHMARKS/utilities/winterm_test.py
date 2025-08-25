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
from unittest import TestCase, main, skipUnless

try:
    from unittest.mock import Mock, patch
except ImportError:
    from mock import Mock, patch

from ..winterm import WinColor, WinStyle, WinTerm

class WinTermTest(TestCase):
    @patch("colorama.winterm.win32")
    def testInit(self, mockWin32):
        mockAttr = Mock()
        mockAttr.wAttributes = 7 + 6 * 16 + 8
        mockWin32.GetConsoleScreenBufferInfo.return_value = mockAttr
        term = WinTerm()
        self.assertEqual(term._fore, 7)
        self.assertEqual(term._back, 6)
        self.assertEqual(term._style, 8)

    @skipUnless(sys.platform.startswith("win"), "requires Windows")
    def testGetAttrs(self):
        term = WinTerm()

        term._fore = 0
        term._back = 0
        term._style = 0
        self.assertEqual(term.get_attrs(), 0)

        term._fore = WinColor.YELLOW
        self.assertEqual(term.get_attrs(), WinColor.YELLOW)

        term._back = WinColor.MAGENTA
        self.assertEqual(term.get_attrs(), WinColor.YELLOW + WinColor.MAGENTA * 16)

        term._style = WinStyle.BRIGHT
        self.assertEqual(
            term.get_attrs(), WinColor.YELLOW + WinColor.MAGENTA * 16 + WinStyle.BRIGHT
        )

    @patch("colorama.winterm.win32")
    def testResetAll(self, mockWin32):
        mockAttr = Mock()
        mockAttr.wAttributes = 1 + 2 * 16 + 8
        mockWin32.GetConsoleScreenBufferInfo.return_value = mockAttr
        term = WinTerm()

        term.set_console = Mock()
        term._fore = -1
        term._back = -1
        term._style = -1

        term.reset_all()

        self.assertEqual(term._fore, 1)
        self.assertEqual(term._back, 2)
        self.assertEqual(term._style, 8)
        self.assertEqual(term.set_console.called, True)

    @skipUnless(sys.platform.startswith("win"), "requires Windows")
    def testFore(self):
        term = WinTerm()
        term.set_console = Mock()
        term._fore = 0

        term.fore(5)

        self.assertEqual(term._fore, 5)
        self.assertEqual(term.set_console.called, True)

    @skipUnless(sys.platform.startswith("win"), "requires Windows")
    def testBack(self):
        term = WinTerm()
        term.set_console = Mock()
        term._back = 0

        term.back(5)

        self.assertEqual(term._back, 5)
        self.assertEqual(term.set_console.called, True)

    @skipUnless(sys.platform.startswith("win"), "requires Windows")
    def testStyle(self):
        term = WinTerm()
        term.set_console = Mock()
        term._style = 0

        term.style(22)

        self.assertEqual(term._style, 22)
        self.assertEqual(term.set_console.called, True)

    @patch("colorama.winterm.win32")
    def testSetConsole(self, mockWin32):
        mockAttr = Mock()
        mockAttr.wAttributes = 0
        mockWin32.GetConsoleScreenBufferInfo.return_value = mockAttr
        term = WinTerm()
        term.windll = Mock()

        term.set_console()

        self.assertEqual(
            mockWin32.SetConsoleTextAttribute.call_args,
            ((mockWin32.STDOUT, term.get_attrs()), {}),
        )

    @patch("colorama.winterm.win32")
    def testSetConsoleOnStderr(self, mockWin32):
        mockAttr = Mock()
        mockAttr.wAttributes = 0
        mockWin32.GetConsoleScreenBufferInfo.return_value = mockAttr
        term = WinTerm()
        term.windll = Mock()

        term.set_console(on_stderr=True)

        self.assertEqual(
            mockWin32.SetConsoleTextAttribute.call_args,
            ((mockWin32.STDERR, term.get_attrs()), {}),
        )

if __name__ == "__main__":
    main()
