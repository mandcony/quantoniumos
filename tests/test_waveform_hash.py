import sys, pathlib, os
_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_QOS = _ROOT / "quantoniumos"
if _QOS.exists() and str(_QOS) not in sys.path:
    sys.path.insert(0, str(_QOS))

from encryption.geometric_waveform_hash import (
    wave_hash,
    extract_wave_parameters,
)

def _rand_bytes(n: int = 256) -> bytes:
    return os.urandom(n)

def test_wave_hash_length_and_extract():
    payload = _rand_bytes()
    h = wave_hash(payload)

    # basic structural checks
    assert isinstance(h, str), "hash should be str"
    assert len(h) == 64, "hash must be 64 hex‐chars"

    # deterministic for same input
    assert h == wave_hash(payload), "hash not deterministic"

    # parameter extraction
    waves, thresh = extract_wave_parameters(h)
    assert waves, "no waves extracted"
    assert 0.6 <= thresh <= 0.8, "coherence threshold out of range"