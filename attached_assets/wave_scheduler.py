# apps/wave_scheduler.py

import math
from apps.wave_primitives import WaveNumber
from apps.resonance_process import ResonanceProcess

def wave_scheduler(processes, system_wave: WaveNumber):
    """
    Picks the process whose wave-based 'constructive interference'
    yields the highest resulting amplitude.
    """
    chosen_proc = None
    best_amp = -1.0

    for proc in processes:
        # Interfere wave_priority with the global system wave:
        result_wave = interfere_waves(proc.wave_priority, system_wave)
        if result_wave.amplitude > best_amp:
            best_amp = result_wave.amplitude
            chosen_proc = proc

    return chosen_proc

def interfere_waves(waveA: WaveNumber, waveB: WaveNumber) -> WaveNumber:
    """
    A simple wave interference approach:
      Convert each wave to complex form, add them, and return a new WaveNumber.
    """
    cA = complex_from_wave(waveA)
    cB = complex_from_wave(waveB)
    summed = cA + cB
    return wave_from_complex(summed)

def complex_from_wave(wv: WaveNumber) -> complex:
    return wv.amplitude * math.cos(wv.phase) + 1j * wv.amplitude * math.sin(wv.phase)

def wave_from_complex(z: complex) -> WaveNumber:
    amp = abs(z)
    phase = math.atan2(z.imag, z.real) if amp > 1e-15 else 0.0
    return WaveNumber(amp, phase)
