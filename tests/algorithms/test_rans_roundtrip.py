import os
import random
import pytest
from algorithms.rft.compression.lossless.rans_stream import ans_encode, ans_decode

@pytest.mark.parametrize("alphabet", [2, 16, 256])
@pytest.mark.parametrize("precision", [8, 12])
def test_rans_roundtrip_small(alphabet, precision):
    data = [random.randrange(alphabet) for _ in range(1000)]
    blob = ans_encode(data, alphabet, precision)
    out = ans_decode(blob)
    assert out == data
