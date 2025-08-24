#!/usr/bin/env python3
"""
Patch for BulletproofQuantumKernel to use the energy-conserving RFT implementation.
This script modifies the BulletproofQuantumKernel class to use our adapter.
"""

import os
import re
import shutil
import sys
from pathlib import Path


def backup_file(file_path):
    """Create a backup of the file."""
    backup_path = f"{file_path}.backup"
    if not os.path.exists(backup_path):
        shutil.copy2(file_path, backup_path)
        print(f"Created backup at {backup_path}")
    return backup_path


def patch_bulletproof_quantum_kernel():
    """
    Patch the BulletproofQuantumKernel to use the energy-conserving RFT adapter.
    """
    # Path to the kernel file
    kernel_path = Path("bulletproof_quantum_kernel.py")

    if not kernel_path.exists():
        print(f"Error: {kernel_path} not found.")
        return False

    # Create backup
    backup_path = backup_file(kernel_path)

    # Read the file
    with open(kernel_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Add import for our adapter
    import_pattern = r"import numpy as np"
    adapter_import = "import numpy as np\nfrom energy_conserving_rft_adapter import EnergyConservingRFTEngine"

    content = re.sub(import_pattern, adapter_import, content)

    # Replace forward_rft method
    forward_rft_pattern = r"def forward_rft\(self, signal: np\.ndarray\) -> np\.ndarray:.*?return self\.rft_basis\.conj\(\)\.T @ signal"
    forward_rft_replacement = '''def forward_rft(self, signal: np.ndarray) -> np.ndarray:
        """
        Forward RFT transform: X = Ψ† x
        Uses energy-conserving implementation for core computation.
        
        Args:
            signal: Input signal vector
            
        Returns:
            RFT spectrum
        """
        # Use the energy-conserving adapter
        adapter = EnergyConservingRFTEngine(dimension=self.dimension)
        return adapter.forward_true_rft(signal)'''

    content = re.sub(
        forward_rft_pattern, forward_rft_replacement, content, flags=re.DOTALL
    )

    # Replace inverse_rft method
    inverse_rft_pattern = r"def inverse_rft\(self, spectrum: np\.ndarray\) -> np\.ndarray:.*?return self\.rft_basis @ spectrum"
    inverse_rft_replacement = '''def inverse_rft(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Inverse RFT transform: x = Ψ X
        Uses energy-conserving implementation for core computation.
        
        Args:
            spectrum: RFT spectrum
            
        Returns:
            Reconstructed signal
        """
        # Use the energy-conserving adapter
        adapter = EnergyConservingRFTEngine(dimension=self.dimension)
        return adapter.inverse_true_rft(spectrum)'''

    content = re.sub(
        inverse_rft_pattern, inverse_rft_replacement, content, flags=re.DOTALL
    )

    # Write the modified content
    with open(kernel_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Successfully patched {kernel_path}")
    print("The BulletproofQuantumKernel now uses the energy-conserving RFT adapter.")
    return True


if __name__ == "__main__":
    print("Patching BulletproofQuantumKernel to use energy-conserving RFT...")
    if patch_bulletproof_quantum_kernel():
        print("Patch completed successfully.")
    else:
        print("Patch failed.")
