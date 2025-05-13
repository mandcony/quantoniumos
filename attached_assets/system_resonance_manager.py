from apps.geometric_container import GeometricContainer
from apps.wave_primitives import WaveNumber
import apps.config as config
import math
import random

class Process(GeometricContainer):
    def __init__(self, id: int, priority, amplitude: complex, vertices):
        if isinstance(priority, (int, float)):
            priority = WaveNumber(amplitude=float(priority), phase=0.0)
        # Pass the required arguments to GeometricContainer
        super().__init__(id, vertices=vertices)
        self.id = id
        self.priority = priority  # WaveNumber object
        self.amplitude = amplitude  # Complex number
        self.resonance = 0.0  # Float
        self.time = 0.0
        self.priority_phase = random.uniform(0, 2 * math.pi)
        self.amplitude_phase = random.uniform(0, 2 * math.pi)
        self.resonance_phase = random.uniform(0, 2 * math.pi)

    def __repr__(self):
        return (f"Process(id={self.id}, "
                f"priority={self.priority.amplitude:.1f}, "
                f"amplitude={abs(self.amplitude.real):.1f}, "
                f"resonance={self.resonance:.1f})")

def monitor_resonance_states(processes, dt):
    cfg = config.Config()
    freq_val = cfg.data.get("resonance_frequency", 1.0)
    system_freq = WaveNumber(freq_val, phase=0.0)
    
    print(f"[Resonance Manager] freq={freq_val}, dt={dt}")
    for p in processes:
        # Increment time for oscillation
        p.time += dt
        # Simulate fluctuating priority (CPU)
        if hasattr(p.priority, "scale_amplitude"):
            load_variation = math.sin(freq_val * p.time + p.priority_phase)
            damping_factor = 0.5 + 0.5 * load_variation  # Oscillates between 0 and 1
            p.priority.scale_amplitude(damping_factor)
            p.priority.amplitude = min(max(p.priority.amplitude, 0), 100)  # Clamp to 0-100%
        
        # Simulate fluctuating amplitude (Memory)
        p.amplitude = complex(50 + 50 * math.sin(freq_val * p.time * 0.8 + p.amplitude_phase), 0)
        p.amplitude = complex(min(max(p.amplitude.real, 0), 100), 0)  # Clamp real part to 0-100%
        
        # Simulate fluctuating resonance (Disk)
        p.resonance = 50 + 50 * math.sin(freq_val * p.time * 0.6 + p.resonance_phase)
        p.resonance = min(max(p.resonance, 0), 100)  # Clamp to 0-100%
        
    return processes