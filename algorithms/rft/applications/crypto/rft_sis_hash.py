# SPDX-License-Identifier: AGPL-3.0-or-later

"""Crypto application: RFT-SIS hash (compatibility shim).

The reference implementation currently lives in
`algorithms.rft.core.resonant_fourier_transform`.
"""

from algorithms.rft.core.resonant_fourier_transform import RFTSISHash

__all__ = ["RFTSISHash"]
