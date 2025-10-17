# Ensure repository root is importable so tests can import `core.*` shims
import os
import sys
import random
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Standardize RNG from environment for reproducibility
_seed_env = os.getenv("QOS_TEST_SEED")
if _seed_env:
    try:
        seed = int(_seed_env, 16) if all(c in '0123456789abcdefABCDEF' for c in _seed_env) else int(_seed_env)
    except Exception:
        seed = abs(hash(_seed_env)) % (2**32)
else:
    seed = 123456789

random.seed(seed)
np.random.seed(seed % (2**32))
