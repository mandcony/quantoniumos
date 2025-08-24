# sitecustomize.py
# Auto-register the pytest shim for any test file executed from this repo.
import importlib, sys
try:
    import test_patch_layer  # installs sys.modules['pytest']
except Exception as e:
    sys.stderr.write(f"[sitecustomize] Warning: pytest shim not loaded: {e}\n")
