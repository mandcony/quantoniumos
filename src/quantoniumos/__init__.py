"""
QuantoniumOS: Quantum-Enhanced Operating System

A quantum-enhanced operating system featuring Resonance Fourier Transform
algorithms, quantum cryptography, and advanced quantum computing capabilities.

Core Features:
- Resonance Fourier Transform (RFT) algorithms
- Quantum-enhanced cryptography
- Quantum process management
- Advanced quantum state manipulation
- Production-ready quantum kernels

Example:
    >>> import core.quantoniumos as quantoniumosas qos
    >>> kernel = qos.QuantumKernel()
    >>> result = kernel.run_quantum_algorithm(data)
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0+unknown"

from .algorithms.rft import ResonanceFourierTransform
# Core imports for public API
from .core.kernel import QuantumKernel
from .crypto.quantum_cipher import QuantumCipher
from .engines.bulletproof import BulletproofQuantumEngine

__all__ = [
    "__version__",
    "QuantumKernel",
    "ResonanceFourierTransform",
    "QuantumCipher",
    "BulletproofQuantumEngine",
]

# Package metadata
__title__ = "quantoniumos"
__description__ = "Quantum-Enhanced Operating System with RFT Algorithms"
__author__ = "QuantoniumOS Team"
__email__ = "team@quantoniumos.org"
__license__ = "MIT"
__url__ = "https://github.com/mandcony/quantoniumos"
