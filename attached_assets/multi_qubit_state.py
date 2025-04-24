import math
from geometric_container import GeometricContainer
import config

class QubitState(GeometricContainer):
    def __init__(self, id: int, amplitude: complex):
        super().__init__(amplitude)
        self.id = id

def monitor_multi_qubit(states, dt):
    cfg = config.Config()
    freq = cfg.data.get("resonance_frequency", 1.0)
    print(f"Monitoring multi-qubit states with frequency {freq}...")
    for s in states:
        s.resonance = abs(s.amplitude.real) * freq * dt
    return states