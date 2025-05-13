# apps/resonance_process.py

from system_resonance_manager import Process
from apps.wave_primitives import WaveNumber

class ResonanceProcess(Process):
    """
    Extends the original Process by integrating a WaveNumber-based priority and
    wave-based state (for amplitude or other resonance info).
    """
    def __init__(self, id: int, vertices, initial_amplitude=1.0, initial_phase=0.0, complex_amplitude=complex(1.0, 0.0)):
        """
        :param id: process ID
        :param vertices: geometric or symbolic structure (required by Process)
        :param initial_amplitude: the wave amplitude for scheduling priority
        :param initial_phase: the wave phase for scheduling
        :param complex_amplitude: the quantum-like amplitude for memory or resource usage
        """
        super().__init__(id=id, vertices=vertices, priority=0.0, amplitude=complex_amplitude)
        # Instead of a float 'priority', we store a WaveNumber:
        self.wave_priority = WaveNumber(initial_amplitude, initial_phase)

    def __repr__(self):
        return (f"ResonanceProcess(id={self.id}, wave_priority={self.wave_priority}, "
                f"amp={self.amplitude}, resonance={self.resonance})")