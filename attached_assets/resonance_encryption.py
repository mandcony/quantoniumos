# C:\quantonium_os\apps\resonance_encryption.py
import base64
import math
from .wave_primitives import WaveNumber

def _coerce_to_wave(obj) -> WaveNumber:
    """
    Internal helper: If 'obj' is already WaveNumber, return it as is.
    If it's a float/int, wrap it in WaveNumber(amplitude=obj, phase=0.0).
    """
    if isinstance(obj, WaveNumber):
        return obj
    elif isinstance(obj, (float, int)):
        return WaveNumber(amplitude=float(obj), phase=0.0)
    else:
        raise TypeError(f"resonance_encrypt expects WaveNumber or float, got {type(obj)}")

def resonance_encrypt(message: str, wave_key) -> str:
    """
    Example encryption using wave_key's amplitude & phase.
    wave_key can be either a WaveNumber or a float.
    """
    wave_key = _coerce_to_wave(wave_key)
    amplitude_scaled = int(abs(wave_key.amplitude) * 127)
    phase_offset = int((wave_key.phase / math.pi) * 50)

    encrypted_bytes = []
    for i, c in enumerate(message):
        # Use amplitude_scaled + i*phase_offset as a pseudo-key
        val = ord(c) ^ ((amplitude_scaled + i * phase_offset) % 256)
        encrypted_bytes.append(val)

    return base64.b64encode(bytes(encrypted_bytes)).decode()

def resonance_decrypt(encrypted_message: str, wave_key) -> str:
    """
    Reverse of resonance_encrypt. wave_key can be WaveNumber or float.
    """
    wave_key = _coerce_to_wave(wave_key)
    amplitude_scaled = int(abs(wave_key.amplitude) * 127)
    phase_offset = int((wave_key.phase / math.pi) * 50)

    encrypted_bytes = base64.b64decode(encrypted_message)
    decrypted_chars = []

    for i, b in enumerate(encrypted_bytes):
        val = b ^ ((amplitude_scaled + i * phase_offset) % 256)
        decrypted_chars.append(chr(val))

    return "".join(decrypted_chars)
