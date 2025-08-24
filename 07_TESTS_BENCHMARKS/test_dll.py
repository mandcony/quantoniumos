import ctypes
import os

# Add project root to path for imports

DLL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin"
)
dll_path = os.path.join(DLL_DIR, "engine_core.dll")


def test_dll_loading():
    try:
        if not os.path.exists(dll_path):
            raise FileNotFoundError(f"Missing DLL: {dll_path}")
        ctypes.CDLL(dll_path)
        print(f"✅ DLL loaded successfully from: {dll_path}")
        return True
    except Exception as e:
        print(f"❌ DLL load failed: {e}")
        return False


if __name__ == "__main__":
    test_dll_loading()
