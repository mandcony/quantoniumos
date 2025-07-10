import os
import pathlib
import sys
import tempfile

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_QOS = _ROOT / "quantoniumos"
if _QOS.exists() and str(_QOS) not in sys.path:
    sys.path.insert(0, str(_QOS))

from orchestration.symbolic_container import SymbolicContainer, hash_file


def test_symbolic_container_seal_unlock():
    # create temporary validation file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(os.urandom(128))
        validation_path = tmp.name

    key_id, A, phi = hash_file(validation_path)
    payload = "TEST_PAYLOAD_12345"

    c = SymbolicContainer(payload, (key_id, A, phi), validation_file=validation_path)
    assert c.seal(), "seal failed"

    recovered = c.unlock(A, phi)
    assert recovered == payload, "payload mismatch after unlock"

    os.unlink(validation_path)
