# SPDX-License-Identifier: AGPL-3.0-or-later

"""CanonicalTrueRFT (legacy wrapper) — transform-only.

This module re-exports the historical `CanonicalTrueRFT` API from
`algorithms.rft.core.canonical_true_rft` so callers can reference it under the
new proof-first layout.
"""

from algorithms.rft.core.canonical_true_rft import CanonicalTrueRFT

__all__ = ["CanonicalTrueRFT"]
