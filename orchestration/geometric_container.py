"""
Quantonium OS - Geometric Container Module

Implements geometric containers for secure data storage with waveform-based access controls.
"""

import hashlib
import json
from pathlib import Path

import numpy as np


class GeometricContainer:
    def __init__(self, label):
        """
        Initialize a Geometric Container.
        """
        self.label = label
        self.locked = False
        self.waveform_hash = None
        self.data = {}
        self.A = None
        self.phi = None

    def seal(self, filepath):
        """
        Seal the container with data from a file.
        """
        try:
            content = Path(filepath).read_text(encoding="utf-8")
            self.A = sum(ord(c) for c in content) % 256
            self.phi = sum(ord(c) ** 2 for c in content) % 360
            self.waveform_hash = hashlib.sha256(
                f"{self.A}{self.phi}".encode()
            ).hexdigest()
            self.locked = True
            return True
        except Exception as e:
            print(f"Error sealing container: {str(e)}")
            return False

    def unlock(self, A_val, phi_val):
        """
        Attempt to unlock the container with amplitude and phase values.
        """
        if not self.locked:
            return False
        tolerance = 0.01
        A_match = abs(self.A - A_val) < tolerance
        phi_match = abs(self.phi - phi_val) < tolerance
        return A_match and phi_match

    def to_json(self):
        """
        Convert the container to JSON.
        """
        return json.dumps(
            {
                "label": self.label,
                "locked": self.locked,
                "hash": self.waveform_hash,
                "A": self.A,
                "phi": self.phi,
            },
            indent=4,
        )
