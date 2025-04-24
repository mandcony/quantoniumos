# apps/resonance_fourier.py

import math
import cmath
from apps.wave_primitives import WaveNumber

def resonance_fourier_transform(wave_samples):
    """
    Takes a list of WaveNumber samples, each representing amplitude+phase in 'time'.
    Produces a list of WaveNumber in 'frequency domain' (toy example).
    """
    N = len(wave_samples)
    output = []
    for k in range(N):
        re_sum = 0.0
        im_sum = 0.0
        for n in range(N):
            # Convert wave to complex:
            z = wave_to_complex(wave_samples[n])
            angle = -2.0 * math.pi * k * n / N
            re_sum += (z.real * math.cos(angle) - z.imag * math.sin(angle))
            im_sum += (z.real * math.sin(angle) + z.imag * math.cos(angle))
        # Create a wave from the sum:
        amp = math.sqrt(re_sum**2 + im_sum**2) / N
        phase = math.atan2(im_sum, re_sum)
        output.append(WaveNumber(amp, phase))
    return output

def wave_to_complex(wv: WaveNumber) -> complex:
    return wv.amplitude * math.cos(wv.phase) + 1j * wv.amplitude * math.sin(wv.phase)
