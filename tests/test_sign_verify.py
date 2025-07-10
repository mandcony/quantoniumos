import os
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_QOS = _ROOT / "quantoniumos"
if _QOS.exists() and str(_QOS) not in sys.path:
    sys.path.insert(0, str(_QOS))

from encryption.resonance_encrypt import verify_wave_hmac, wave_hmac


def test_wave_hmac_sign_verify():
    """Test the wave_hmac signature and verification"""

    # Generate random test data
    message = os.urandom(128)
    key = os.urandom(32)

    # Sign the message
    signature = wave_hmac(message, key)

    # Verify the signature
    assert verify_wave_hmac(message, signature, key), "Signature verification failed"

    # Test tampering resistance
    tampered_message = bytearray(message)
    if len(tampered_message) > 0:
        tampered_message[0] ^= 0x01  # Flip one bit

    # Verify the tampered message should fail
    assert not verify_wave_hmac(
        tampered_message, signature, key
    ), "Tampered message was incorrectly verified"

    # Test with phase_info=False (pure HMAC)
    signature_pure = wave_hmac(message, key, phase_info=False)
    assert verify_wave_hmac(
        message, signature_pure, key, phase_info=False
    ), "Pure HMAC verification failed"

    # Verify signatures are different
    assert signature != signature_pure, "Phase and pure signatures should differ"
