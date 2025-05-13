import base64
import math

from attached_assets.wave_primitives import WaveNumber

def _coerce_to_wave(obj) -> WaveNumber:
    """
    Internal helper: If 'obj' is already a WaveNumber, return it as is.
    If it's a float/int, wrap it in a WaveNumber with phase 0.
    """
    if isinstance(obj, WaveNumber):
        return obj
    elif isinstance(obj, (float, int)):
        return WaveNumber(float(obj), 0.0)
    else:
        raise TypeError(f"Cannot convert {type(obj)} to WaveNumber")

def resonance_encrypt(plaintext: str, wave_key) -> str:
    """
    Encrypts a string using resonance-based encryption with a WaveNumber key.
    
    This is a simplified example of our wave-based encryption technique
    that combines basic XOR with amplitude and phase information. The actual
    resonance encryption is much more complex and uses quantum-inspired 
    transformations that are beyond the scope of this demo.
    
    Args:
        plaintext: text to encrypt
        wave_key: a WaveNumber object or float value to use as key
        
    Returns:
        Base64-encoded encrypted text with resonance signatures
    """
    # Ensure key is a WaveNumber
    wave_key = _coerce_to_wave(wave_key)
    
    # Convert plaintext to bytes if it's a string
    if isinstance(plaintext, str):
        plaintext = plaintext.encode('utf-8')
    
    # Basic transformation based on the key's amplitude and phase
    result = bytearray()
    for i, byte in enumerate(plaintext):
        # Use both amplitude and phase, with position-dependent modulation
        amplitude_factor = wave_key.amplitude * (1 + 0.1 * math.sin(i / 5))
        phase_factor = wave_key.phase * (1 + 0.1 * math.cos(i / 7))
        
        # Combine factors
        factor = (amplitude_factor + phase_factor) % 256
        
        # Apply the transformation
        transformed = (byte + int(factor)) % 256
        result.append(transformed)
    
    # Encode as base64
    encoded = base64.b64encode(result).decode('ascii')
    return encoded

def resonance_decrypt(ciphertext: str, wave_key) -> str:
    """
    Decrypts a string encrypted using resonance-based encryption with a WaveNumber key.
    
    Args:
        ciphertext: Base64-encoded encrypted text
        wave_key: the same WaveNumber object or float value used for encryption
        
    Returns:
        Decrypted text
    """
    # Ensure key is a WaveNumber
    wave_key = _coerce_to_wave(wave_key)
    
    # Decode from base64
    encrypted_data = base64.b64decode(ciphertext)
    
    # Reverse the transformation
    result = bytearray()
    for i, byte in enumerate(encrypted_data):
        # Reconstruct the same factors used in encryption
        amplitude_factor = wave_key.amplitude * (1 + 0.1 * math.sin(i / 5))
        phase_factor = wave_key.phase * (1 + 0.1 * math.cos(i / 7))
        
        # Combine factors
        factor = (amplitude_factor + phase_factor) % 256
        
        # Reverse the transformation
        original = (byte - int(factor)) % 256
        result.append(original)
    
    # Convert back to string
    return result.decode('utf-8')

def generate_wave_from_text(text: str) -> WaveNumber:
    """
    Generate a WaveNumber from a text key.
    
    This is a simplified key derivation for demonstration.
    For real applications, use proper key derivation functions (KDFs).
    """
    try:
        from attached_assets.geometric_waveform_hash import geometric_waveform_hash
        
        # Use our special hash function to generate wave parameters
        hash_value = geometric_waveform_hash(text.encode('utf-8'))
        
        # Extract amplitude and phase from the hash
        parts = hash_value.split('-')
        if len(parts) >= 4:
            try:
                amplitude = float(parts[2][1:])  # Extract number after 'A'
                phase = float(parts[3][1:])      # Extract number after 'P'
                return WaveNumber(amplitude, phase)
            except (ValueError, IndexError):
                pass
    except ImportError:
        pass
    
    # Fallback method if geometric hash isn't available
    hash_value = sum(ord(c) for c in text)
    amplitude = (hash_value % 100) / 10.0
    phase = (hash_value % 314) / 100.0
    
    return WaveNumber(amplitude, phase)

def sign_message(message: str, key: str) -> str:
    """
    Sign a message using wave-based resonance
    
    Args:
        message: The message to sign
        key: The key to use for signing
        
    Returns:
        A base64-encoded signature
    """
    try:
        from attached_assets.geometric_waveform_hash import geometric_waveform_hash
        
        # Generate the key
        wave_key = generate_wave_from_text(key)
        
        # Combine message and key
        data = (message + str(wave_key.amplitude) + str(wave_key.phase)).encode('utf-8')
        
        # Use our geometric hash
        hash_value = geometric_waveform_hash(data)
        
        # Create and return the signature
        return base64.b64encode(hash_value.encode('utf-8')).decode('ascii')
    except ImportError:
        # Fallback to simple method if geometric_waveform_hash isn't available
        wave_key = generate_wave_from_text(key)
        combined = message + str(wave_key.amplitude) + str(wave_key.phase)
        signature = str(sum(ord(c) * (i + 1) for i, c in enumerate(combined)))
        return base64.b64encode(signature.encode('utf-8')).decode('ascii')

def verify_signature(message: str, signature: str, key: str) -> bool:
    """
    Verify a signature using wave-based resonance
    
    Args:
        message: The original message
        signature: The base64-encoded signature
        key: The key used for signing
        
    Returns:
        True if signature is valid, False otherwise
    """
    # Generate the expected signature
    expected = sign_message(message, key)
    
    # Compare with provided signature
    return signature == expected
