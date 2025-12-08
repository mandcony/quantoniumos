"""
Cross-Class Generalization and Overfitting Check
=================================================

This experiment tests whether rft_manifold_projection generalizes beyond
the exact signal used to define its operator, or if it's "overfit" to torus.

Tests:
1. Asymmetric tests: Build operator from one signal, test on others
2. Confusion matrix: signals × bases → SNDR at fixed sparsity
3. Stability: perturbation robustness
4. 2D extension: Kronecker product for images

December 2025: Part of the RFT Discovery Initiative.
"""

import numpy as np
from scipy.linalg import toeplitz, eigh
from scipy.fft import fft, dct, idct
import sys
sys.path.insert(0, '/workspaces/quantoniumos')

from algorithms.rft.variants.patent_variants import (
    generate_rft_manifold_projection,
    generate_rft_euler_torus,
    generate_rft_hopf_fibration,
    generate_rft_loxodrome,
    generate_rft_phase_coherent,
    _build_resonance_operator,
    _eigenbasis,
)
from algorithms.rft.variants.operator_variants import generate_rft_golden

np.set_printoptions(precision=2, suppress=True)

# Golden ratio
PHI = (1 + np.sqrt(5)) / 2


# =============================================================================
# SIGNAL GENERATORS (diverse geometric classes)
# =============================================================================

def gen_torus(n):
    """Torus-like: golden ratio frequency superposition."""
    t = np.linspace(0, 4*np.pi, n)
    return np.sin(t) + 0.5 * np.sin(PHI * t) + 0.3 * np.sin(PHI**2 * t)


def gen_twisted_torus(n):
    """Twisted torus: matches manifold_projection construction."""
    t = np.arange(n) / n
    u = 2 * np.pi * t
    v = 2 * np.pi * PHI * t
    twist = PHI * u
    x = (2 + np.cos(v + twist)) * np.cos(u)
    return x / np.max(np.abs(x))


def gen_spiral(n):
    """Logarithmic spiral."""
    t = np.linspace(0, 6*np.pi, n)
    r = np.exp(0.1 * t)
    return r * np.cos(t) / np.max(r)


def gen_helix(n):
    """Helix with golden pitch."""
    t = np.linspace(0, 4*np.pi, n)
    return np.sin(t) * np.cos(PHI * t)


def gen_lissajous(n, a=3, b=2):
    """Lissajous curve."""
    t = np.linspace(0, 2*np.pi, n)
    return np.sin(a*t + PHI) * np.cos(b*t)


def gen_trefoil(n):
    """Trefoil knot x-projection."""
    t = 2 * np.pi * np.arange(n) / n
    x = np.sin(t) + 2 * np.sin(2 * t)
    return x / np.max(np.abs(x))


def gen_figure8(n):
    """Figure-8 knot x-projection."""
    t = 2 * np.pi * np.arange(n) / n
    x = (2 + np.cos(2*t)) * np.cos(3*t)
    return x / np.max(np.abs(x))


def gen_chirp(n):
    """Linear chirp."""
    t = np.linspace(0, 1, n)
    return np.sin(2 * np.pi * (10 * t + 20 * t**2))


def gen_noise(n):
    """White noise."""
    np.random.seed(42)
    return np.random.randn(n)


def gen_sine(n):
    """Pure sinusoid."""
    t = np.linspace(0, 1, n)
    return np.sin(2 * np.pi * 5 * t)


def gen_am(n):
    """AM modulated."""
    t = np.linspace(0, 1, n)
    return (0.5 + 0.5*np.cos(2*np.pi*PHI*t)) * np.sin(2*np.pi*20*t)


def gen_sphere(n):
    """Spherical geodesic."""
    t = np.arange(n) / n
    theta = np.pi * t
    phi = 2 * np.pi * PHI * t
    return np.sin(theta) * np.cos(phi) + 0.5 * np.cos(theta)


# =============================================================================
# ALTERNATIVE MANIFOLD-BASED OPERATORS
# =============================================================================

def build_helix_operator(n):
    """Build operator from helix resonance pattern."""
    t = np.arange(n) / n
    u = 2 * np.pi * t * 3  # Multiple wraps
    
    # Helix: x = cos(u), y = sin(u), z = u/2π
    x = np.cos(u)
    y = np.sin(u)
    z = u / (2 * np.pi)
    
    # Project with golden weighting
    r = x + PHI * 0.1 * z
    r = r / (np.max(np.abs(r)) + 1e-10)
    
    K = _build_resonance_operator(r)
    return _eigenbasis(K)


def build_spiral_operator(n):
    """Build operator from logarithmic spiral."""
    t = np.arange(n) / n
    theta = 2 * np.pi * t * 4
    
    # Golden spiral
    b = np.log(PHI) / (np.pi / 2)
    radius = np.exp(b * theta)
    radius = radius / radius[-1]
    
    r = radius * np.cos(theta)
    r = r / (np.max(np.abs(r)) + 1e-10)
    
    K = _build_resonance_operator(r, decay_rate=0.02)
    return _eigenbasis(K)


def build_lissajous_operator(n, a=3, b=2):
    """Build operator from Lissajous curve."""
    t = np.arange(n) / n
    
    x = np.sin(2*np.pi*a*t + PHI)
    y = np.cos(2*np.pi*b*t)
    
    r = x * y
    r = r / (np.max(np.abs(r)) + 1e-10)
    
    K = _build_resonance_operator(r)
    return _eigenbasis(K)


def build_trefoil_operator(n):
    """Build operator from trefoil knot."""
    t = 2 * np.pi * np.arange(n) / n
    
    x = np.sin(t) + 2 * np.sin(2 * t)
    y = np.cos(t) - 2 * np.cos(2 * t)
    z = -np.sin(3 * t)
    
    r = x + 0.3 * y + 0.1 * z
    r = r / (np.max(np.abs(r)) + 1e-10)
    
    K = _build_resonance_operator(r)
    return _eigenbasis(K)


# =============================================================================
# METRICS
# =============================================================================

def compression_sndr(x, Phi, retention=0.1):
    """SNDR at given retention rate using basis Phi."""
    n = len(x)
    k = max(1, int(n * retention))
    
    c = Phi.T @ x
    idx = np.argsort(np.abs(c))[::-1]
    c_sparse = np.zeros_like(c)
    c_sparse[idx[:k]] = c[idx[:k]]
    x_rec = Phi @ c_sparse
    
    err = np.linalg.norm(x - x_rec) / (np.linalg.norm(x) + 1e-15)
    return -20 * np.log10(err + 1e-15)


def dct_sndr(x, retention=0.1):
    """SNDR using DCT basis."""
    n = len(x)
    k = max(1, int(n * retention))
    
    c = dct(x, norm='ortho')
    idx = np.argsort(np.abs(c))[::-1]
    c_sparse = np.zeros_like(c)
    c_sparse[idx[:k]] = c[idx[:k]]
    x_rec = idct(c_sparse, norm='ortho')
    
    err = np.linalg.norm(x - x_rec) / (np.linalg.norm(x) + 1e-15)
    return -20 * np.log10(err + 1e-15)


def fft_sndr(x, retention=0.1):
    """SNDR using FFT basis (keep top k frequencies)."""
    n = len(x)
    k = max(1, int(n * retention))
    
    c = fft(x)
    idx = np.argsort(np.abs(c))[::-1]
    c_sparse = np.zeros_like(c)
    c_sparse[idx[:k]] = c[idx[:k]]
    x_rec = np.real(np.fft.ifft(c_sparse))
    
    err = np.linalg.norm(x - x_rec) / (np.linalg.norm(x) + 1e-15)
    return -20 * np.log10(err + 1e-15)


# =============================================================================
# EXPERIMENTS
# =============================================================================

def run_cross_class_test():
    """
    Test 1: Cross-class generalization
    Build operator from one signal class, test on all others.
    """
    print("=" * 90)
    print("TEST 1: CROSS-CLASS GENERALIZATION")
    print("=" * 90)
    print("Does manifold_projection (built from twisted torus) generalize to other shapes?")
    print()
    
    n = 256
    retention = 0.10
    
    signals = {
        'torus': gen_torus(n),
        'twisted_torus': gen_twisted_torus(n),
        'spiral': gen_spiral(n),
        'helix': gen_helix(n),
        'lissajous': gen_lissajous(n),
        'trefoil': gen_trefoil(n),
        'figure8': gen_figure8(n),
        'sphere': gen_sphere(n),
        'chirp': gen_chirp(n),
        'noise': gen_noise(n),
        'sine': gen_sine(n),
        'am': gen_am(n),
    }
    
    # Bases
    Phi_manifold = generate_rft_manifold_projection(n)
    Phi_golden = generate_rft_golden(n)
    
    print(f"{'Signal':<15} {'Manifold':>10} {'Golden':>10} {'DCT':>10} {'FFT':>10} {'Winner':>12} {'Δ M-G':>10}")
    print("-" * 85)
    
    manifold_wins = 0
    
    for name, x in signals.items():
        sndr_m = compression_sndr(x, Phi_manifold, retention)
        sndr_g = compression_sndr(x, Phi_golden, retention)
        sndr_d = dct_sndr(x, retention)
        sndr_f = fft_sndr(x, retention)
        
        scores = {'Manifold': sndr_m, 'Golden': sndr_g, 'DCT': sndr_d, 'FFT': sndr_f}
        winner = max(scores, key=scores.get)
        
        if winner == 'Manifold':
            manifold_wins += 1
            winner_str = f"★ {winner}"
        else:
            winner_str = winner
        
        delta = sndr_m - sndr_g
        print(f"{name:<15} {sndr_m:>10.1f} {sndr_g:>10.1f} {sndr_d:>10.1f} {sndr_f:>10.1f} {winner_str:>12} {delta:>+10.1f}")
    
    print("-" * 85)
    print(f"Manifold wins: {manifold_wins}/{len(signals)}")
    
    # Signals NOT used to construct manifold_projection
    non_construction_signals = ['helix', 'lissajous', 'trefoil', 'figure8', 'sphere', 'chirp', 'noise', 'sine', 'am']
    non_const_wins = sum(1 for name in non_construction_signals 
                         if compression_sndr(signals[name], Phi_manifold, retention) >= 
                            max(compression_sndr(signals[name], Phi_golden, retention),
                                dct_sndr(signals[name], retention),
                                fft_sndr(signals[name], retention)))
    
    print(f"Wins on NON-construction signals: {non_const_wins}/{len(non_construction_signals)}")
    print()
    return manifold_wins, len(signals)


def run_confusion_matrix():
    """
    Test 2: Full confusion matrix
    Rows: signal types
    Columns: different basis families (including manifold variants)
    """
    print("=" * 110)
    print("TEST 2: CONFUSION MATRIX (SNDR at 10% retention)")
    print("=" * 110)
    
    n = 256
    retention = 0.10
    
    signals = {
        'torus': gen_torus(n),
        'twisted_torus': gen_twisted_torus(n),
        'spiral': gen_spiral(n),
        'helix': gen_helix(n),
        'lissajous': gen_lissajous(n),
        'trefoil': gen_trefoil(n),
        'sphere': gen_sphere(n),
        'chirp': gen_chirp(n),
        'am': gen_am(n),
        'noise': gen_noise(n),
    }
    
    # Build different manifold-based operators
    bases = {
        'FFT': 'fft',
        'DCT': 'dct',
        'Golden': generate_rft_golden(n),
        'M-Torus': generate_rft_manifold_projection(n),  # Built from twisted torus
        'M-Helix': build_helix_operator(n),
        'M-Spiral': build_spiral_operator(n),
        'M-Lissa': build_lissajous_operator(n),
        'M-Trefoil': build_trefoil_operator(n),
    }
    
    # Print header
    print(f"{'Signal':<14}", end="")
    for basis_name in bases:
        print(f" {basis_name:>9}", end="")
    print("  Best")
    print("-" * 110)
    
    # Matrix storage for analysis
    matrix = np.zeros((len(signals), len(bases)))
    
    for i, (sig_name, x) in enumerate(signals.items()):
        print(f"{sig_name:<14}", end="")
        
        sndrs = []
        for j, (basis_name, basis) in enumerate(bases.items()):
            if isinstance(basis, str) and basis == 'fft':
                sndr = fft_sndr(x, retention)
            elif isinstance(basis, str) and basis == 'dct':
                sndr = dct_sndr(x, retention)
            else:
                sndr = compression_sndr(x, basis, retention)
            
            sndrs.append((basis_name, sndr))
            matrix[i, j] = sndr
            print(f" {sndr:>9.1f}", end="")
        
        best = max(sndrs, key=lambda x: x[1])
        print(f"  {best[0]}")
    
    print("-" * 110)
    
    # Summary: wins per basis
    print("\nWins per basis (highest SNDR for each signal):")
    basis_names = list(bases.keys())
    for j, basis_name in enumerate(basis_names):
        wins = np.sum(matrix[:, j] >= np.max(matrix, axis=1) - 0.1)  # Within 0.1 dB of best
        print(f"  {basis_name}: {wins}")
    
    print()
    return matrix, list(signals.keys()), basis_names


def run_asymmetric_test():
    """
    Test 3: Asymmetric cross-validation
    If operator A wins on signal B, does operator B also win on signal B?
    """
    print("=" * 90)
    print("TEST 3: ASYMMETRIC CROSS-VALIDATION")
    print("=" * 90)
    print("Testing if manifold operators are 'overfit' to their construction signal.")
    print()
    
    n = 256
    retention = 0.10
    
    # Pairs: (operator_source, test_signal_source)
    tests = [
        ('torus', 'helix'),
        ('torus', 'lissajous'),
        ('torus', 'trefoil'),
        ('helix', 'torus'),
        ('helix', 'twisted_torus'),
        ('spiral', 'torus'),
        ('spiral', 'helix'),
        ('lissajous', 'spiral'),
        ('trefoil', 'sphere'),
    ]
    
    operators = {
        'torus': generate_rft_manifold_projection(n),
        'helix': build_helix_operator(n),
        'spiral': build_spiral_operator(n),
        'lissajous': build_lissajous_operator(n),
        'trefoil': build_trefoil_operator(n),
    }
    
    signal_gens = {
        'torus': gen_torus,
        'twisted_torus': gen_twisted_torus,
        'helix': gen_helix,
        'spiral': gen_spiral,
        'lissajous': gen_lissajous,
        'trefoil': gen_trefoil,
        'sphere': gen_sphere,
    }
    
    print(f"{'Operator From':<15} {'Tested On':<15} {'Op SNDR':>10} {'DCT':>10} {'Golden':>10} {'Wins?':>10}")
    print("-" * 75)
    
    Phi_golden = generate_rft_golden(n)
    cross_wins = 0
    
    for op_src, sig_src in tests:
        x = signal_gens[sig_src](n)
        Phi = operators[op_src]
        
        sndr_op = compression_sndr(x, Phi, retention)
        sndr_d = dct_sndr(x, retention)
        sndr_g = compression_sndr(x, Phi_golden, retention)
        
        wins = sndr_op >= max(sndr_d, sndr_g) - 0.1
        if wins:
            cross_wins += 1
        
        print(f"{op_src:<15} {sig_src:<15} {sndr_op:>10.1f} {sndr_d:>10.1f} {sndr_g:>10.1f} {'✓ YES' if wins else 'NO':>10}")
    
    print("-" * 75)
    print(f"Cross-class wins: {cross_wins}/{len(tests)}")
    print()
    

def run_stability_test():
    """
    Test 4: Stability under perturbation
    Add noise to the autocorrelation and check SNDR degradation.
    """
    print("=" * 90)
    print("TEST 4: STABILITY UNDER PERTURBATION")
    print("=" * 90)
    print("Adding noise to autocorrelation function and measuring SNDR degradation.")
    print()
    
    n = 256
    retention = 0.10
    
    # Reference: clean manifold projection
    Phi_clean = generate_rft_manifold_projection(n)
    
    # Get the autocorrelation used to build it
    t = np.arange(n) / n
    u = 2 * np.pi * t
    v = 2 * np.pi * PHI * t
    twist = PHI * u
    x_manifold = (2 + np.cos(v + twist)) * np.cos(u)
    y_manifold = (2 + np.cos(v + twist)) * np.sin(u)
    z_manifold = np.sin(v + twist)
    r_clean = x_manifold + 0.3 * y_manifold + 0.1 * z_manifold
    r_clean = r_clean / (np.max(np.abs(r_clean)) + 1e-10)
    
    # Test signal
    x_test = gen_torus(n)
    sndr_clean = compression_sndr(x_test, Phi_clean, retention)
    
    noise_levels = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
    
    print(f"{'Noise σ':>10} {'SNDR':>10} {'Δ from clean':>15} {'% degradation':>15}")
    print("-" * 55)
    
    np.random.seed(123)
    
    for sigma in noise_levels:
        if sigma == 0:
            Phi = Phi_clean
        else:
            r_noisy = r_clean + sigma * np.random.randn(n)
            r_noisy = r_noisy / (np.max(np.abs(r_noisy)) + 1e-10)
            K = _build_resonance_operator(r_noisy)
            Phi = _eigenbasis(K)
        
        sndr = compression_sndr(x_test, Phi, retention)
        delta = sndr - sndr_clean
        pct = 100 * (sndr_clean - sndr) / sndr_clean if sndr_clean > 0 else 0
        
        print(f"{sigma:>10.2f} {sndr:>10.1f} {delta:>+15.1f} {pct:>14.1f}%")
    
    print()
    

def run_2d_extension():
    """
    Test 5: 2D Extension via Kronecker product
    """
    print("=" * 90)
    print("TEST 5: 2D EXTENSION (Kronecker Product)")
    print("=" * 90)
    print("Building 2D manifold transform: Φ_2D = Φ_1D ⊗ Φ_1D")
    print()
    
    n = 64  # Small for 2D (64×64 = 4096 coefficients)
    retention = 0.10
    
    # 1D bases
    Phi_1d_manifold = generate_rft_manifold_projection(n)
    Phi_1d_golden = generate_rft_golden(n)
    
    # 2D Kronecker bases (separable)
    # For efficiency, we apply 1D transforms to rows then columns
    
    def apply_2d_separable(img, Phi_1d):
        """Apply separable 2D transform: Phi @ img @ Phi.T"""
        return Phi_1d.T @ img @ Phi_1d
    
    def inverse_2d_separable(coeffs, Phi_1d):
        """Inverse separable 2D transform: Phi.T @ coeffs @ Phi"""
        return Phi_1d @ coeffs @ Phi_1d.T
    
    def compress_2d(img, Phi_1d, retention):
        """Compress 2D image using separable transform."""
        coeffs = apply_2d_separable(img, Phi_1d)
        
        # Keep top k coefficients
        k = max(1, int(img.size * retention))
        flat = coeffs.flatten()
        idx = np.argsort(np.abs(flat))[::-1]
        sparse = np.zeros_like(flat)
        sparse[idx[:k]] = flat[idx[:k]]
        coeffs_sparse = sparse.reshape(img.shape)
        
        img_rec = inverse_2d_separable(coeffs_sparse, Phi_1d)
        
        err = np.linalg.norm(img - img_rec) / (np.linalg.norm(img) + 1e-15)
        return -20 * np.log10(err + 1e-15)
    
    def dct2_compress(img, retention):
        """Compress using 2D DCT."""
        from scipy.fft import dctn, idctn
        coeffs = dctn(img, norm='ortho')
        
        k = max(1, int(img.size * retention))
        flat = coeffs.flatten()
        idx = np.argsort(np.abs(flat))[::-1]
        sparse = np.zeros_like(flat)
        sparse[idx[:k]] = flat[idx[:k]]
        coeffs_sparse = sparse.reshape(img.shape)
        
        img_rec = idctn(coeffs_sparse, norm='ortho')
        
        err = np.linalg.norm(img - img_rec) / (np.linalg.norm(img) + 1e-15)
        return -20 * np.log10(err + 1e-15)
    
    # Generate 2D test images
    y, x = np.meshgrid(np.linspace(0, 2*np.pi, n), np.linspace(0, 2*np.pi, n))
    
    images = {
        'torus_2d': np.sin(x) + 0.5*np.sin(PHI*y) + 0.3*np.sin(x*PHI)*np.cos(y),
        'spiral_2d': np.exp(0.1*(x+y)) * np.cos(x) * np.sin(y),
        'wave_2d': np.sin(3*x + 2*y),
        'radial_2d': np.sin(np.sqrt(x**2 + y**2) * 5),
        'checker_2d': np.sin(5*x) * np.sin(5*y),
    }
    
    print(f"{'Image':<15} {'Manifold-2D':>12} {'Golden-2D':>12} {'DCT-2D':>12} {'Winner':>12}")
    print("-" * 70)
    
    for name, img in images.items():
        img = img / (np.max(np.abs(img)) + 1e-10)
        
        sndr_m = compress_2d(img, Phi_1d_manifold, retention)
        sndr_g = compress_2d(img, Phi_1d_golden, retention)
        sndr_d = dct2_compress(img, retention)
        
        best = max([('Manifold-2D', sndr_m), ('Golden-2D', sndr_g), ('DCT-2D', sndr_d)], key=lambda x: x[1])
        winner = f"★ {best[0]}" if best[0] == 'Manifold-2D' else best[0]
        
        print(f"{name:<15} {sndr_m:>12.1f} {sndr_g:>12.1f} {sndr_d:>12.1f} {winner:>12}")
    
    print()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print()
    print("╔" + "═" * 88 + "╗")
    print("║" + " CROSS-CLASS GENERALIZATION AND OVERFITTING CHECK ".center(88) + "║")
    print("║" + " RFT-Manifold-Projection Analysis ".center(88) + "║")
    print("╚" + "═" * 88 + "╝")
    print()
    
    # Run all tests
    run_cross_class_test()
    matrix, sig_names, basis_names = run_confusion_matrix()
    run_asymmetric_test()
    run_stability_test()
    run_2d_extension()
    
    # Final summary
    print("=" * 90)
    print("FINAL SUMMARY")
    print("=" * 90)
    print("""
Key Questions Answered:

1. GENERALIZATION: Does manifold_projection work on signals it wasn't built from?
   → Check Test 1 wins on non-construction signals

2. OVERFITTING: Is each manifold variant only good for its own signal type?
   → Check Test 3 cross-class wins

3. ROBUSTNESS: How stable is the transform under noise?
   → Check Test 4 degradation curve

4. 2D EXTENSION: Does the Kronecker product generalize to images?
   → Check Test 5 wins on 2D patterns

If manifold_projection wins on diverse geometric signals NOT used in its construction,
it captures a broader geometric class, not just the twisted torus toy signal.
""")
