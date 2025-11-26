# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.
"""Bell inequality validation utilities used by the test suite."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence

import numpy as np


@dataclass(frozen=True)
class ProtocolResult:
    """Container for protocol evaluation results."""

    value: float
    passed: bool
    details: dict[str, Any]


class BellTestProtocol:
    """Simple CHSH Bell inequality validator for vertex engines."""

    CLASSICAL_BOUND = 2.0
    TARGET_THRESHOLD = 2.7

    def validate(self, engine: Any, vertices: Sequence[int] | None = None) -> ProtocolResult:
        """Evaluate CHSH violation for the provided engine.

        Parameters
        ----------
        engine:
            Instance of `EntangledVertexEngine` or compatible object providing
            `assemble_entangled_state`.
        vertices:
            Optional sequence of vertex indices to highlight. Currently the
            implementation assumes two vertices and uses the global state that
            the engine assembles.
        """
        state = engine.assemble_entangled_state(entanglement_level=1.0)
        if state.ndim != 1:
            state = state.reshape(-1)

        if state.size < 4:
            raise ValueError("BellTestProtocol expects at least a two-qubit state")

        if state.size > 4:
            state = state[:4]
            norm = np.linalg.norm(state)
            if norm > 0:
                state = state / norm

        chsh_value = self._compute_chsh(state)
        violation_strength = max(0.0, chsh_value - self.CLASSICAL_BOUND)
        passed = chsh_value >= self.TARGET_THRESHOLD

        return ProtocolResult(
            value=chsh_value,
            passed=passed,
            details={
                "violation_strength": violation_strength,
                "vertices": list(vertices) if vertices is not None else [0, 1],
            },
        )

    @staticmethod
    def _compute_chsh(state: np.ndarray) -> float:
        """Compute the CHSH correlator for a two-qubit pure state."""
        if state.ndim != 1:
            state = state.reshape(-1)

        def pauli_measurement(angle: float) -> np.ndarray:
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            return np.array([[cos_a, sin_a], [sin_a, -cos_a]], dtype=complex)

        angles_A = [0.0, np.pi / 2]  # Alice: 0°, 90°
        angles_B = [np.pi / 4, -np.pi / 4]  # Bob: 45°, -45°

        chsh_sum = 0.0
        for i, angle_A in enumerate(angles_A):
            for j, angle_B in enumerate(angles_B):
                A = pauli_measurement(angle_A)
                B = pauli_measurement(angle_B)
                AB = np.kron(A, B)
                correlation = float(np.real(np.conjugate(state) @ (AB @ state)))
                if (i, j) == (1, 1):
                    chsh_sum -= correlation
                else:
                    chsh_sum += correlation

        return chsh_sum

class EntanglementValidationSuite:
    """Collection of entanglement validation protocols."""

    def __init__(self, protocols: Iterable[BellTestProtocol] | None = None) -> None:
        self.protocols: List[BellTestProtocol] = list(protocols) if protocols is not None else [BellTestProtocol()]

    def run_all(self, engine: Any, vertices: Sequence[int] | None = None) -> List[ProtocolResult]:
        return [protocol.validate(engine, vertices=vertices) for protocol in self.protocols]


__all__ = ["BellTestProtocol", "EntanglementValidationSuite", "ProtocolResult"]
