import os
import pytest

@pytest.mark.slow
def test_nist_sts_placeholder():
    # Placeholder: replace with actual STS harness invoking a dataset
    # This ensures CI wiring recognizes the slow marker
    assert os.getenv("RUN_STS_PLACEHOLDER", "0") in {"0", "1"}
