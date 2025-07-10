"""
Quantonium OS - Symbolic Projection Layer

Projects symbolic states onto new basis vectors for advanced quantum-like computations.
"""

from typing import List, Tuple, Union

import numpy as np


def project_symbolic_state(
    current_state: Union[List[float], List[Tuple[float, complex]]],
) -> List[float]:
    """
    Project a symbolic state onto a new basis.
    If input is from a Fourier transform, it handles the frequency-value pairs.
    """
    # Check if we have a Fourier transform output (list of tuples)
    if (
        isinstance(current_state, list)
        and current_state
        and isinstance(current_state[0], tuple)
    ):
        # Extract just the magnitudes from frequency-value pairs
        magnitudes = [abs(val[1]) for val in current_state]
        # Normalize
        if any(magnitudes):
            norm_factor = max(magnitudes)
            return [mag / norm_factor for mag in magnitudes]
        return magnitudes

    # Regular vector projection
    vector = np.array(current_state)
    # Simple normalization
    norm = np.linalg.norm(vector)
    if norm > 0:
        return (vector / norm).tolist()
    return vector.tolist()
