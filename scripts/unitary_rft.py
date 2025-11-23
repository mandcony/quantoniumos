# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
Top-level shim for UnitaryRFT bindings.

This allows `from unitary_rft import UnitaryRFT, RFT_FLAG_QUANTUM_SAFE, ...`
to work regardless of environment-specific binding paths.

It simply re-exports the implementation from
`algorithms.rft.kernels.python_bindings.unitary_rft`.
"""

# Re-export all public symbols
from algorithms.rft.kernels.python_bindings.unitary_rft import *  # noqa: F401,F403
