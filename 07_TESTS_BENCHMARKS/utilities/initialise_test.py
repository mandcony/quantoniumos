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
    from unittest.mock import patch
except ImportError:
    from mock import patch

from ..ansitowin32 import StreamWrapper
from ..initialise import _wipe_internal_state_for_tests, init
from .utils import osname, replace_by

orig_stdout = sys.stdout
orig_stderr = sys.stderr

class InitTest(TestCase):
    @skipUnless(sys.stdout.isatty(), "sys.stdout is not a tty")
    def setUp(self):
        # sanity check
        self.assertNotWrapped()

    def tearDown(self):
        _wipe_internal_state_for_tests()
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr

    def assertWrapped(self):
        self.assertIsNot(sys.stdout, orig_stdout, "stdout should be wrapped")
        self.assertIsNot(sys.stderr, orig_stderr, "stderr should be wrapped")
        self.assertTrue(isinstance(sys.stdout, StreamWrapper), "bad stdout wrapper")
        self.assertTrue(isinstance(sys.stderr, StreamWrapper), "bad stderr wrapper")

    def assertNotWrapped(self):
        self.assertIs(sys.stdout, orig_stdout, "stdout should not be wrapped")
        self.assertIs(sys.stderr, orig_stderr, "stderr should not be wrapped")

    @patch("colorama.initialise.reset_all")
    @patch("colorama.ansitowin32.winapi_test", lambda *_: True)
    @patch("colorama.ansitowin32.enable_vt_processing", lambda *_: False)
    def testInitWrapsOnWindows(self, _):
        with osname("nt"):
            init()
            self.assertWrapped()

    @patch("colorama.initialise.reset_all")
    @patch("colorama.ansitowin32.winapi_test", lambda *_: False)
    def testInitDoesntWrapOnEmulatedWindows(self, _):
        with osname("nt"):
            init()
            self.assertNotWrapped()

    def testInitDoesntWrapOnNonWindows(self):
        with osname("posix"):
            init()
            self.assertNotWrapped()

    def testInitDoesntWrapIfNone(self):
        with replace_by(None):
            init()
            # We can't use assertNotWrapped here because replace_by(None) # changes stdout/stderr already. self.assertIsNone(sys.stdout) self.assertIsNone(sys.stderr) def testInitAutoresetOnWrapsOnAllPlatforms(self): with osname("posix"): init(autoreset=True) self.assertWrapped() def testInitWrapOffDoesntWrapOnWindows(self): with osname("nt"): init(wrap=False) self.assertNotWrapped() def testInitWrapOffIncompatibleWithAutoresetOn(self): self.assertRaises(ValueError, lambda: init(autoreset=True, wrap=False)) @patch('colorama.win32.SetConsoleTextAttribute') @patch('colorama.initialise.AnsiToWin32') def testAutoResetPassedOn(self, mockATW32, _): with osname("nt"): init(autoreset=True) self.assertEqual(len(mockATW32.call_args_list), 2) self.assertEqual(mockATW32.call_args_list[1][1]['autoreset'], True) self.assertEqual(mockATW32.call_args_list[0][1]['autoreset'], True) @patch('colorama.initialise.AnsiToWin32') def testAutoResetChangeable(self, mockATW32): with osname("nt"): init() init(autoreset=True) self.assertEqual(len(mockATW32.call_args_list), 4) self.assertEqual(mockATW32.call_args_list[2][1]['autoreset'], True) self.assertEqual(mockATW32.call_args_list[3][1]['autoreset'], True) init() self.assertEqual(len(mockATW32.call_args_list), 6) self.assertEqual( mockATW32.call_args_list[4][1]['autoreset'], False) self.assertEqual( mockATW32.call_args_list[5][1]['autoreset'], False) @patch('colorama.initialise.atexit.register') def testAtexitRegisteredOnlyOnce(self, mockRegister): init() self.assertTrue(mockRegister.called) mockRegister.reset_mock() init() self.assertFalse(mockRegister.called) class JustFixWindowsConsoleTest(TestCase): def _reset(self): _wipe_internal_state_for_tests() sys.stdout = orig_stdout sys.stderr = orig_stderr def tearDown(self): self._reset() @patch("colorama.ansitowin32.winapi_test", lambda: True) def testJustFixWindowsConsole(self): if sys.platform != "win32": # just_fix_windows_console should be a no-op just_fix_windows_console() self.assertIs(sys.stdout, orig_stdout) self.assertIs(sys.stderr, orig_stderr) else: def fake_std(): # Emulate stdout=not a tty, stderr=tty # to check that we handle both cases correctly stdout = Mock() stdout.closed = False stdout.isatty.return_value = False stdout.fileno.return_value = 1 sys.stdout = stdout stderr = Mock() stderr.closed = False stderr.isatty.return_value = True stderr.fileno.return_value = 2 sys.stderr = stderr for native_ansi in [False, True]: with patch( 'colorama.ansitowin32.enable_vt_processing', lambda *_: native_ansi ): self._reset() fake_std() # Regular single-call test prev_stdout = sys.stdout prev_stderr = sys.stderr just_fix_windows_console() self.assertIs(sys.stdout, prev_stdout) if native_ansi: self.assertIs(sys.stderr, prev_stderr) else: self.assertIsNot(sys.stderr, prev_stderr) # second call without resetting is always a no-op prev_stdout = sys.stdout prev_stderr = sys.stderr just_fix_windows_console() self.assertIs(sys.stdout, prev_stdout) self.assertIs(sys.stderr, prev_stderr) self._reset() fake_std() # If init() runs first, just_fix_windows_console should be a no-op init() prev_stdout = sys.stdout prev_stderr = sys.stderr just_fix_windows_console() self.assertIs(prev_stdout, sys.stdout) self.assertIs(prev_stderr, sys.stderr) if __name__ == '__main__':

    main()
