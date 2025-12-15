# SPDX-License-Identifier: AGPL-3.0-or-later

"""φ-phase FFT baseline (deprecated) — transform-only.

This module is a compatibility re-export of the legacy implementation.
It lives in `transform_core/` so theory/validation code can import a
transform-only reference without pulling application modules.
"""

from algorithms.rft.core.phi_phase_fft import *  # noqa: F401,F403
