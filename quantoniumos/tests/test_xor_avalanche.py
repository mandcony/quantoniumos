import sys, pathlib, os
_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_QOS = _ROOT / "quantoniumos"
if _QOS.exists() and str(_QOS) not in sys.path:
    sys.path.insert(0, str(_QOS))

from encryption.resonance_encrypt import encrypt_symbolic

def _hex32() -> str:
    return os.urandom(16).hex()

def _bit_flip(hex_str: str, bit_index: int) -> str:
    b = bytearray.fromhex(hex_str)
    b[bit_index // 8] ^= 1 << (bit_index % 8)
    return b.hex()

def test_xor_avalanche_effect():
    pt = _hex32()
    key = _hex32()
    baseline = encrypt_symbolic(pt, key)["ciphertext"]

    changed = 0
    for bit in range(16):          # flip first 16 bits
        pt_mut = _bit_flip(pt, bit)
        ct = encrypt_symbolic(pt_mut, key)["ciphertext"]
        if ct != baseline:
            changed += 1

    # Require at least half of the flips to alter the ciphertext
    assert changed >= 8, f"weak avalanche: {changed}/16 flips changed output"