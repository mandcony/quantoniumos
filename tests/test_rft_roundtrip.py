import numpy as np, pytest, sys, os  
sys.path.append(os.path.abspath("."))               # add root to path  
from api.symbolic_interface import run_rft, inverse_rft # thin wrappers to DLL

VECTORS = [  
    np.random.rand(64).tolist(),  
    [0.0]*63 + [1.0],                                # impulse  
    [np.sin(2*np.pi*k/64) for k in range(64)]        # smooth wave  
]

@pytest.mark.parametrize("vec", VECTORS)
def test_roundtrip(vec):
    F = run_rft(vec)
    x_hat = inverse_rft(F)
    assert np.allclose(x_hat, vec, rtol=1e-7, atol=1e-9)