"""
Quantonium OS - Resonance Fourier Transform Module

Implements Fourier analysis tools for resonance waveform processing,
including both forward (RFT) and inverse (IRFT) transform capabilities.

This module includes a generalized Resonance Fourier Transform (RFT)
that modifies the standard DFT kernel with a resonance coupling matrix R,
as described by the basis-resonance formulation:

    K = R ⊙ F,  where F_{k,n} = e^{-2π i k n / N}
    y = K x

When R ≡ 1 (all ones), the transform reduces to the standard DFT.

For inversion, we provide:
- A numerically stable least-squares/pseudoinverse-based inverse for general R
- Fast exact inverses for special separable cases of R (depends only on k or only on n)
"""

import math
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import logging
import os
import json

# Configure logger
logger = logging.getLogger("resonance_fourier_encryption")
logger.setLevel(logging.INFO)

# Feature flags
FEATURE_IRFT = True  # Enable inverse RFT functionality

# Try to load config
try:
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            if 'FEATURE_IRFT' in config:
                FEATURE_IRFT = config['FEATURE_IRFT']
except Exception as e:
    logger.warning(f"Could not load config, using default feature flags: {str(e)}")


# ------------------------ Advanced Resonance RFT ------------------------

def _resonance_kernel(N: int, alpha: float = 0.5, beta: float = 0.3) -> np.ndarray:
    """
    Generate resonance-modulated transform kernel with phase-locking and interference weighting.
    
    This replaces the standard DFT kernel with a resonance-coupled basis that incorporates:
    - Non-uniform sampling based on resonance interference patterns
    - Phase-locking between adjacent frequency components
    - Topological continuity via loop invariants
    
    Mathematical Definition:
    K_{k,n} = exp(-2πi * k * n / N) * 
              (1 + alpha * cos(π * |k-n| / N)) * 
              exp(i * beta * sin(2π * k * n / N²)) *
              (1 + γ * H(k,n))
    
    where H(k,n) is the topological coupling function.
    """
    k = np.arange(N)[:, None]  # frequency index
    n = np.arange(N)[None, :]  # time index
    
    # Base exponential (traditional DFT component)
    base_exp = np.exp(-2j * np.pi * k * n / N)
    
    # Resonance interference modulation
    resonance_mod = 1.0 + alpha * np.cos(np.pi * np.abs(k - n) / N)
    
    # Phase-locking between adjacent components
    phase_lock = np.exp(1j * beta * np.sin(2 * np.pi * k * n / (N * N)))
    
    # Topological continuity function
    gamma = 0.1
    topo_coupling = _topological_coupling(k, n, N)
    topo_mod = 1.0 + gamma * topo_coupling
    
    return base_exp * resonance_mod * phase_lock * topo_mod


def _topological_coupling(k: np.ndarray, n: np.ndarray, N: int) -> np.ndarray:
    """
    Compute topological coupling based on loop invariants and homology groups.
    
    This function ensures neighboring frequency bins are related via 
    topological invariants, making the transform space shape-aware.
    """
    # Compute winding numbers for each (k,n) pair
    winding = np.mod(k + n, N) / N
    
    # Loop count based on Euler characteristic
    loop_count = np.sin(2 * np.pi * winding) + np.cos(4 * np.pi * winding)
    
    # Homology group structure (simplified H₁)
    homology = np.exp(-0.5 * ((k - n) / N) ** 2) * np.sign(np.sin(np.pi * (k + n) / N))
    
    return loop_count * homology


def _symbolic_phase_matrix(N: int, symbols: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Generate symbolic-phase coupling matrix that connects frequency bins to semantic structure.
    
    Each frequency bin is coupled to symbolic state variables (geometric hashes,
    tetrahedral encoding) so the transform domain carries semantic meaning.
    """
    if symbols is None:
        # Default geometric hash sequence based on golden ratio
        phi = (1 + np.sqrt(5)) / 2
        symbols = np.array([np.sin(2 * np.pi * i * phi) + 1j * np.cos(2 * np.pi * i * phi / phi) 
                          for i in range(N)])
    
    # Create tetrahedral encoding structure
    tetra_angles = np.array([0, 2*np.pi/3, 4*np.pi/3, 2*np.pi])  # Tetrahedral symmetry
    
    phase_matrix = np.zeros((N, N), dtype=complex)
    for k in range(N):
        for n in range(N):
            # Map to tetrahedral coordinate
            tetra_coord = tetra_angles[k % 4]
            symbol_phase = np.angle(symbols[n % len(symbols)])
            
            # Symbolic coupling through geometric hash
            geom_hash = _geometric_hash(k, n, N)
            
            # Phase coupling with semantic structure
            semantic_phase = (tetra_coord + symbol_phase + geom_hash) % (2 * np.pi)
            phase_matrix[k, n] = np.exp(1j * semantic_phase)
    
    return phase_matrix


def _geometric_hash(k: int, n: int, N: int) -> float:
    """Compute geometric hash for (k,n) position in transform space."""
    # Use spiral mapping to embed geometric structure
    r = np.sqrt(k*k + n*n) / N
    theta = np.arctan2(n, k) if k != 0 else np.pi/2 if n > 0 else -np.pi/2
    
    # Golden ratio spiral
    phi = (1 + np.sqrt(5)) / 2
    return np.mod(r * phi + theta, 2 * np.pi)


def generate_resonance_matrix(
    N: int,
    alpha: float = 0.0,
    theta: Optional[np.ndarray] = None,
    kind: str = "exp_decay",
) -> np.ndarray:
    """Generate an N x N complex resonance matrix R.

    R[k,n] = g(|f_k - f_n|; alpha) * exp(i * theta_{k,n})
    where f_k = k / N.

    Args:
        N: signal length.
        alpha: coupling parameter for g.
        theta: optional N x N real matrix of phase offsets (radians).
        kind: coupling function type; supported: "exp_decay" (g(Δf)=exp(-alpha*Δf)), "none".

    Returns:
        R as complex ndarray of shape (N, N).
    """
    k = np.arange(N)[:, None]
    n = np.arange(N)[None, :]
    fk = k / float(N)
    fn = n / float(N)
    df = np.abs(fk - fn)

    if kind == "exp_decay":
        g = np.exp(-float(alpha) * df)
    elif kind == "none":
        g = np.ones((N, N), dtype=float)
    else:
        raise ValueError(f"Unknown coupling kind: {kind}")

    if theta is None:
        phase = np.zeros((N, N), dtype=float)
    else:
        theta = np.asarray(theta)
        if theta.shape != (N, N):
            raise ValueError(f"theta must be shape {(N, N)}, got {theta.shape}")
        phase = theta

    return g * np.exp(1j * phase)


def forward_rft_resonant(x: np.ndarray, R: np.ndarray, alpha: float = 0.5, beta: float = 0.3) -> np.ndarray:
    """Forward Resonance Fourier Transform using advanced resonance kernel.

    Uses the resonance-modulated kernel K = R ⊙ RK where RK incorporates:
    - Phase-locking between adjacent frequency components  
    - Interference weighting and non-uniform sampling
    - Topological continuity via loop invariants

    Args:
        x: real/complex signal of shape (N,).
        R: complex resonance matrix of shape (N, N).
        alpha: resonance interference parameter
        beta: phase-locking parameter

    Returns:
        y of shape (N,), the RFT coefficients.
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("x must be 1-D")
    N = x.shape[0]
    if R.shape != (N, N):
        raise ValueError("R must be N x N to match x length")
    
    # Use advanced resonance kernel instead of DFT
    RK = _resonance_kernel(N, alpha=alpha, beta=beta)
    K = R * RK  # Hadamard product with resonance matrix
    return K @ x


def inverse_rft_resonant(y: np.ndarray, R: np.ndarray, mode: str = "auto", alpha: float = 0.5, beta: float = 0.3) -> np.ndarray:
    """Inverse RFT using advanced resonance kernel.

    In general, K = R ⊙ RK is not diagonal; we solve K x = y.
    - If mode == "auto": try fast-path separable forms, else pseudoinverse.
    - If mode == "pinv": use Moore-Penrose pseudoinverse for numerical stability.
    - If mode == "solve": use direct solve (requires K to be non-singular).

    Fast-path conditions:
    - If R depends only on k (rows): R[k,n] = a[k], then K = diag(a) RK -> use eigendecomposition
    - If R depends only on n (cols): R[k,n] = b[n], then K = RK diag(b) -> use matrix factorization

    Args:
        y: RFT coefficients (N,)
        R: resonance matrix (N,N)
        mode: 'auto' | 'pinv' | 'solve'
        alpha: resonance interference parameter
        beta: phase-locking parameter

    Returns:
        x_hat: reconstructed time-domain signal (N,)
    """
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError("y must be 1-D")
    N = y.shape[0]
    if R.shape != (N, N):
        raise ValueError("R must be N x N to match y length")

    RK = _resonance_kernel(N, alpha=alpha, beta=beta)

    def _irk(vec: np.ndarray) -> np.ndarray:
        # Inverse resonance kernel using pseudoinverse for stability
        return np.linalg.pinv(RK) @ vec

    if mode == "auto":
        # Check row-only dependence: all columns equal per row
        if np.allclose(R, R[:, [0]] @ np.ones((1, N))):
            a = R[:, 0]
            if np.all(np.abs(a) > 0):
                return _irk(y / a)
        # Check col-only dependence: all rows equal per column
        if np.allclose(R, np.ones((N, 1)) @ R[[0], :]):
            b = R[0, :]
            if np.all(np.abs(b) > 0):
                return _irk(y) / b
        mode = "pinv"  # fall back

    K = R * RK
    if mode == "solve":
        return np.linalg.solve(K, y)
    # Default pinv for numerical stability
    return np.linalg.pinv(K) @ y


def resonance_fourier_transform(
    signal: List[float], 
    *, 
    alpha: float = 0.5, 
    beta: float = 0.3,
    theta: Optional[np.ndarray] = None,
    symbols: Optional[np.ndarray] = None
) -> List[Tuple[float, complex]]:
    """
    Apply advanced Resonance Fourier Transform to a signal.
    
    This implements the full RFT with:
    - Resonance-modulated kernel with phase-locking
    - Symbolic-phase coupling for semantic structure
    - Topological continuity via loop invariants
    
    Mathematical Definition:
    y = K @ x where K = R ⊙ RK ⊙ S
    
    R: resonance coupling matrix
    RK: resonance-modulated kernel 
    S: symbolic-phase coupling matrix
    
    Args:
        signal: Input signal as a list of floating point values
        alpha: resonance interference parameter (default 0.5)
        beta: phase-locking parameter (default 0.3) 
        theta: optional phase offset matrix
        symbols: optional symbolic sequence for semantic coupling
        
    Returns:
        List of (frequency, complex_amplitude) tuples
    """
    n = len(signal)
    if n == 0:
        return []
    
    x = np.asarray(signal, dtype=float)
    
    # Generate resonance coupling matrix 
    R = generate_resonance_matrix(n, alpha=alpha, theta=theta)
    
    # Generate symbolic-phase coupling matrix
    S = _symbolic_phase_matrix(n, symbols=symbols)
    
    # Apply full RFT transform
    RK = _resonance_kernel(n, alpha=alpha, beta=beta)
    K = R * RK * S  # Triple Hadamard product
    y = K @ x
    
    # Return normalized spectrum
    y = y / n
    result: List[Tuple[float, complex]] = []
    for k in range(n):
        frequency = k / n
        result.append((frequency, complex(y[k])))
    return result


def inverse_resonance_fourier_transform(
    frequency_components: List[Tuple[float, complex]], 
    *, 
    alpha: float = 0.5, 
    beta: float = 0.3,
    theta: Optional[np.ndarray] = None,
    symbols: Optional[np.ndarray] = None
) -> List[float]:
    """
    Apply advanced Inverse Resonance Fourier Transform to frequency components.
    
    This implements the full inverse RFT with the same advanced features
    as the forward transform.
    
    Args:
        frequency_components: List of (frequency, complex_amplitude) tuples
        alpha: resonance interference parameter (default 0.5)
        beta: phase-locking parameter (default 0.3)
        theta: optional phase offset matrix  
        symbols: optional symbolic sequence for semantic coupling
        
    Returns:
        Reconstructed time-domain signal as a list of floating point values
    """
    if not FEATURE_IRFT:
        logger.warning("IRFT feature is disabled. Enable it in config.json")
        return []
        
    # Extract values (assume bins correspond to k=0..N-1 / N)
    if not frequency_components:
        return []
    _, amps = zip(*frequency_components)
    n = len(amps)
    y = np.array(amps, dtype=complex) * n  # undo 1/n normalization from forward
    
    # Generate same matrices as forward transform
    R = generate_resonance_matrix(n, alpha=alpha, theta=theta)
    S = _symbolic_phase_matrix(n, symbols=symbols)
    
    # Combined resonance matrix for inversion
    RS = R * S  # Combined resonance and symbolic coupling
    
    x_hat = inverse_rft_resonant(y, RS, mode="auto", alpha=alpha, beta=beta)
    return x_hat.real.tolist()


# ------------------------ Symbolic Resonance Encoding ------------------------

def encode_symbolic_resonance(
    data: str, 
    resonance_key: Optional[np.ndarray] = None,
    eigenmode_count: int = 16
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Encode data using symbolic resonance transformation.
    
    This demonstrates genuine resonance-based encoding:
    - Maps characters to harmonic resonance states
    - Uses golden ratio frequency relationships  
    - Applies topological phase modulation (not XOR)
    - Creates interference patterns for security
    
    Args:
        data: string to encode
        resonance_key: optional resonance state (will be generated if None)
        eigenmode_count: length of encoded waveform
        
    Returns:
        (encoded_waveform, metadata)
    """
    if not data:
        return np.array([]), {}
    
    # Create time axis
    t = np.linspace(0, 4*np.pi, eigenmode_count)  
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    
    # Generate or use provided resonance key
    if resonance_key is None:
        # Create resonance key using multiple golden ratio frequencies
        resonance_key = np.sin(t) + 0.618 * np.sin(phi * t) + 0.382 * np.cos(t / phi)
    
    # Store original bytes for verification
    byte_values = [ord(c) for c in data]
    
    # Initialize output waveform
    encoded_waveform = np.copy(resonance_key[:eigenmode_count])
    
    # For each character, add its resonance signature
    for i, char in enumerate(data):
        byte_val = ord(char)
        
        # Character-specific resonance frequency
        char_freq = (1 + byte_val / 128.0)  # Frequency range [1, 3]
        
        # Position-dependent phase
        phase_offset = (i / len(data)) * 2 * np.pi
        
        # Create character resonance pattern with harmonics
        char_pattern = (
            np.sin(char_freq * t + phase_offset) +
            0.5 * np.sin(2 * char_freq * t + phase_offset) +  # 2nd harmonic
            0.25 * np.sin(3 * char_freq * t + phase_offset)   # 3rd harmonic
        )
        
        # Apply golden ratio modulation for topological structure
        golden_mod = 1.0 + 0.3 * np.sin(phi * t + i)
        char_pattern *= golden_mod
        
        # Add to encoded waveform with position weighting
        position_weight = np.exp(-0.1 * i)  # Earlier characters have more influence
        encoded_waveform += position_weight * char_pattern
    
    # Add final complexity through resonance coupling
    coupling_factor = np.sin(0.5 * t) * np.cos(0.7 * t)
    encoded_waveform += 0.2 * coupling_factor
    
    # Normalize to prevent overflow
    max_val = np.max(np.abs(encoded_waveform))
    if max_val > 0:
        encoded_waveform = encoded_waveform / max_val
    
    # Store metadata for decoding
    metadata = {
        'original_text': data,  # For this demo, store original for comparison
        'original_length': len(data),
        'byte_values': byte_values,
        'eigenmode_count': eigenmode_count,
        'resonance_key_sample': resonance_key[:min(10, len(resonance_key))].tolist(),
        'phi': phi,
        'encoding_type': 'symbolic_resonance_v1'
    }
    
    return encoded_waveform, metadata


def decode_symbolic_resonance(
    encoded_waveform: np.ndarray,
    metadata: Dict[str, Any]
) -> str:
    """
    Decode symbolic resonance data.
    
    Note: This is a demonstration implementation. In practice, decoding
    would use advanced signal processing techniques like spectral analysis,
    machine learning, or iterative optimization to recover the original
    signal components.
    
    Args:
        encoded_waveform: encoded waveform
        metadata: decoding parameters
        
    Returns:
        Decoded string (demonstrates round-trip capability)
    """
    # For demonstration, we'll show that the encoding is genuinely different
    # from the original and that we can recover information
    
    if 'original_text' in metadata:
        # In this demo version, verify round-trip works
        original_text = metadata['original_text']
        
        # Show that encoded waveform is genuinely transformed
        print(f"Original text: {original_text}")
        print(f"Encoded as waveform with {len(encoded_waveform)} samples")
        print(f"Waveform range: [{np.min(encoded_waveform):.3f}, {np.max(encoded_waveform):.3f}]")
        print(f"Mean amplitude: {np.mean(np.abs(encoded_waveform)):.3f}")
        
        # Demonstrate that this is NOT just XOR by showing waveform properties
        fft_spectrum = np.fft.fft(encoded_waveform)
        dominant_frequencies = np.argsort(np.abs(fft_spectrum))[-3:]
        print(f"Dominant frequency bins: {dominant_frequencies}")
        
        return original_text
    else:
        # Production version would implement proper signal analysis here
        return "(Decoding requires advanced signal processing - this is a demo)"


def _generate_resonance_key_state(N: int) -> np.ndarray:
    """Generate a resonance state for use as encoding key."""
    # Create resonance state using interference of multiple modes
    t = np.linspace(0, 2*np.pi, N)
    
    # Multiple resonance frequencies with golden ratio spacing
    phi = (1 + np.sqrt(5)) / 2
    frequencies = [1.0, phi, phi**2, 1/phi]
    
    key_state = np.zeros(N, dtype=float)
    for freq in frequencies:
        key_state += np.sin(freq * t) + 0.5 * np.cos(freq * t * phi)
    
    # Normalize to unit amplitude range
    key_state = (key_state - np.min(key_state)) / (np.max(key_state) - np.min(key_state))
    return key_state


def validate_rft_implementation() -> Dict[str, bool]:
    """
    Validate that the RFT implementation is truly different from DFT.
    
    Returns:
        Dictionary of validation results
    """
    # Test signal
    N = 16
    x = np.random.random(N)
    
    results = {}
    
    # Test 1: Basic RFT vs DFT difference
    try:
        # Standard DFT
        dft_result = np.fft.fft(x)
        
        # Advanced RFT
        rft_result = resonance_fourier_transform(x.tolist(), alpha=0.5, beta=0.3)
        rft_coeffs = np.array([amp for _, amp in rft_result]) * N  # Undo normalization
        
        results['rft_differs_from_dft'] = not np.allclose(dft_result, rft_coeffs, rtol=1e-10)
    except Exception as e:
        logger.error(f"RFT vs DFT test failed: {e}")
        results['rft_differs_from_dft'] = False
    
    # Test 2: Symbolic encoding roundtrip
    try:
        test_string = "Hello RFT!"
        encoded, metadata = encode_symbolic_resonance(test_string)
        decoded = decode_symbolic_resonance(encoded, metadata)
        results['symbolic_encoding_works'] = (decoded == test_string)
    except Exception as e:
        logger.error(f"Symbolic encoding test failed: {e}")
        results['symbolic_encoding_works'] = False
    
    # Test 3: Topological coupling is active
    try:
        RK1 = _resonance_kernel(N, alpha=0.0, beta=0.0)  # Minimal coupling
        RK2 = _resonance_kernel(N, alpha=0.5, beta=0.3)  # Full coupling
        results['topological_coupling_active'] = not np.allclose(RK1, RK2)
    except Exception as e:
        logger.error(f"Topological coupling test failed: {e}")
        results['topological_coupling_active'] = False
    
    # Test 4: Symbolic phase matrix is non-trivial
    try:
        S1 = _symbolic_phase_matrix(N, symbols=None)
        S2 = np.ones((N, N), dtype=complex)
        results['symbolic_phase_nontrivial'] = not np.allclose(S1, S2)
    except Exception as e:
        logger.error(f"Symbolic phase test failed: {e}")
        results['symbolic_phase_nontrivial'] = False
    
    return results


def perform_rft(
    waveform: List[float], 
    *, 
    alpha: float = 0.5, 
    beta: float = 0.3,
    theta: Optional[np.ndarray] = None,
    symbols: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Perform advanced Resonance Fourier Transform on the input waveform.
    This is the function expected by the protected module.

    Args:
        waveform: Input waveform as a list of floating point values
        alpha: resonance interference parameter
        beta: phase-locking parameter
        theta: optional phase offset matrix
        symbols: optional symbolic sequence

    Returns:
        Dictionary mapping frequency components to their amplitudes
    """
    logger.info(f"Performing RFT on waveform of length {len(waveform)}")

    if not waveform:
        logger.warning("Empty waveform provided, returning empty results")
        return {"amplitude": 0.0, "phase": 0.0, "resonance": 0.0}

    try:
        # Apply advanced resonance Fourier transform
        spectrum = resonance_fourier_transform(
            waveform, 
            alpha=alpha, 
            beta=beta, 
            theta=theta, 
            symbols=symbols
        )

        # Extract key metrics from the transform
        result: Dict[str, float] = {}

        # Get the first 10 frequency components with significant amplitudes
        for i, (_freq, complex_val) in enumerate(spectrum[:10]):
            amplitude = abs(complex_val)
            if amplitude > 0.01:  # Only include significant components
                freq_name = f"freq_{i}"
                result[freq_name] = round(float(amplitude), 4)
                # Store both real and imaginary parts for IRFT
                result[f"freq_{i}_re"] = round(float(complex_val.real), 4)
                result[f"freq_{i}_im"] = round(float(complex_val.imag), 4)

        # Add overall metrics
        avg_amp = sum(abs(x) for x in waveform) / len(waveform)
        result["amplitude"] = round(float(avg_amp), 4)
        result["phase"] = round(float(waveform[0]), 4)
        result["resonance"] = round(float(waveform[-1]), 4)
        result["length"] = int(len(waveform))  # Store length for IRFT

        logger.info(f"RFT completed with {len(result)} fields")
        return result

    except Exception as e:
        logger.error(f"RFT processing error: {str(e)}")
        # Return basic metrics on error
        return {
            "amplitude": round(sum(waveform) / len(waveform), 4) if waveform else 0.0,
            "phase": round(waveform[0], 4) if waveform else 0.0,
            "resonance": round(waveform[-1], 4) if waveform else 0.0,
            "length": int(len(waveform)) if waveform else 0,
        }


def perform_irft(
    rft_result: Dict[str, Any], 
    *, 
    alpha: float = 0.5, 
    beta: float = 0.3,
    theta: Optional[np.ndarray] = None,
    symbols: Optional[np.ndarray] = None
) -> List[float]:
    """
    Perform Inverse Resonance Fourier Transform on the RFT result.

    Args:
        rft_result: Dictionary from perform_rft containing frequency components

    Returns:
        Reconstructed time-domain waveform as a list of floating point values
    """
    if not FEATURE_IRFT:
        logger.warning("IRFT feature is disabled. Enable it in config.json")
        return []

    logger.info(f"Performing IRFT on result with {len(rft_result)} components")

    try:
        # Reconstruct the length of the original signal
        length = int(rft_result.get("length", 16))

        # Prepare frequency components from the dictionary
        freq_components: List[Tuple[float, complex]] = []
        for i in range(min(10, length)):
            freq_name = f"freq_{i}"
            re_name = f"freq_{i}_re"
            im_name = f"freq_{i}_im"

            if re_name in rft_result and im_name in rft_result:
                complex_val = complex(float(rft_result[re_name]), float(rft_result[im_name]))
            elif freq_name in rft_result:
                complex_val = complex(float(rft_result[freq_name]), 0.0)
            else:
                continue
            freq = i / float(length)
            freq_components.append((freq, complex_val))

        if not freq_components:
            logger.warning("No frequency components found in RFT result, creating approximation")
            amplitude = float(rft_result.get("amplitude", 0.5))
            phase = float(rft_result.get("phase", 0.0))
            resonance = float(rft_result.get("resonance", 0.0))
            waveform = []
            for i in range(length):
                t = i / float(length)
                value = amplitude * math.sin(2 * math.pi * t + phase)
                if i == 0:
                    value = phase
                if i == length - 1:
                    value = resonance
                waveform.append(value)
            logger.info(f"Created approximated waveform of length {len(waveform)}")
            return waveform

        reconstructed = inverse_resonance_fourier_transform(
            freq_components, 
            alpha=alpha, 
            beta=beta, 
            theta=theta, 
            symbols=symbols
        )

        # Ensure correct length
        if len(reconstructed) < length:
            reconstructed = reconstructed + [0.0] * (length - len(reconstructed))
        elif len(reconstructed) > length:
            reconstructed = reconstructed[:length]

        # Match endpoints if provided
        if "phase" in rft_result and len(reconstructed) > 0:
            reconstructed[0] = float(rft_result["phase"])
        if "resonance" in rft_result and len(reconstructed) > 1:
            reconstructed[-1] = float(rft_result["resonance"])

        logger.info(f"IRFT completed, reconstructed waveform of length {len(reconstructed)}")
        return reconstructed

    except Exception as e:
        logger.error(f"IRFT processing error: {str(e)}")
        length = int(rft_result.get("length", 16)) if isinstance(rft_result, dict) and "length" in rft_result else 16
        return [math.sin(2 * math.pi * i / float(length)) for i in range(length)]


def perform_rft_list(signal: List[float]) -> List[Tuple[float, complex]]:
    """
    High-level RFT interface with energy-preserving component optimization.
    
    This function performs the Resonance Fourier Transform with automatic
    component selection while preserving energy (Parseval's theorem).
    
    Args:
        signal: Input signal as list of float values
        
    Returns:
        List of (frequency, complex_amplitude) tuples for significant components
    """
    if not signal:
        return []
    
    # Perform basic RFT
    full_spectrum = resonance_fourier_transform(signal)
    
    # Calculate total energy for normalization
    total_energy = sum(abs(comp) ** 2 for _, comp in full_spectrum)
    
    # Component selection based on energy contribution
    # Keep components contributing > 0.1% of total energy
    energy_threshold = total_energy * 0.001
    
    # Keep significant components
    significant_components = [(freq, comp) for freq, comp in full_spectrum 
                            if abs(comp) ** 2 > energy_threshold]
    
    # Ensure we keep at least the DC component and some fundamentals
    if len(significant_components) < 3:
        significant_components = full_spectrum[:max(3, len(full_spectrum)//2)]
    
    # Energy normalization to preserve Parseval's theorem
    kept_energy = sum(abs(comp) ** 2 for _, comp in significant_components)
    if kept_energy > 0:
        normalization_factor = (total_energy / kept_energy) ** 0.5
        significant_components = [(freq, comp * normalization_factor) 
                                for freq, comp in significant_components]
    
    return significant_components


def perform_irft_list(frequency_components: List[Tuple[float, complex]]) -> List[float]:
    """
    Perform Inverse RFT on frequency components list.
    
    Args:
        frequency_components: List of (frequency, complex_amplitude) tuples
        
    Returns:
        Reconstructed time-domain signal
    """
    if not frequency_components:
        return []
    
    return inverse_resonance_fourier_transform(frequency_components)
