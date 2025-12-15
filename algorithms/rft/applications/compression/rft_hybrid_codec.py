# SPDX-License-Identifier: AGPL-3.0-or-later

"""Compression application: Hybrid codec (compatibility shim).

The canonical module currently lives at `algorithms.rft.hybrids.rft_hybrid_codec`.
This shim exists to support the proof-first repo layout while keeping backward
compatibility.
"""

from algorithms.rft.hybrids.rft_hybrid_codec import *  # noqa: F401,F403
