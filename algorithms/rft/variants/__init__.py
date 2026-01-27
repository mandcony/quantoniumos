# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""Φ-RFT variant generators and registry."""

from .registry import (
	PHI,
	VARIANTS,
	VariantInfo,
	generate_adaptive_phi,
	generate_chaotic_mix,
	generate_convex_mixed_phi_rft,
	generate_exact_golden_ratio_unitary,
	generate_fibonacci_tilt,
	generate_geometric_lattice,
	generate_harmonic_phase,
	generate_hyperbolic_phase,
	generate_log_periodic_phi_rft,
	generate_original_phi_rft,
	generate_phi_chaotic_hybrid,
	generate_adaptive_split_variant,
	generate_dct_basis,
	generate_hybrid_dct_rft,
)
from .golden_ratio_unitary import GoldenRatioUnitary
from .symbolic_unitary import SymbolicUnitary
from .entropic_unitary import EntropicUnitary

# Patent-aligned variants (US Patent Application 19/169,399)
from .patent_variants import (
	PATENT_VARIANTS,
	get_patent_variant,
	list_patent_variants,
	# Top performers:
	generate_rft_manifold_projection,  # 4 wins, best on torus/spiral
	generate_rft_sphere_parametric,    # 1 win, best on phyllotaxis  
	generate_rft_phase_coherent,       # 1 win, best on chirp
	generate_rft_entropy_modulated,    # 1 win, best on noise
	generate_rft_loxodrome,            # +12 dB vs golden on sine
	generate_rft_polar_golden,         # Claim 3: polar-Cartesian
	generate_rft_spiral_golden,        # Claim 3: golden spiral
	generate_rft_complex_exp,          # Claim 3: complex exponential
	generate_rft_winding,              # Claim 3: winding numbers
	generate_rft_torus_parametric,     # Claim 3: torus parametric
	generate_rft_hopf_fibration,       # Claim 3: Hopf fibration
	generate_rft_bloom_hash,           # Claim 2: Bloom filters
	generate_rft_trefoil_knot,         # Claim 3: knot invariants
)

__all__ = [
	"PHI",
	"VARIANTS",
	"VariantInfo",
	"generate_original_phi_rft",
	"generate_harmonic_phase",
	"generate_fibonacci_tilt",
	"generate_chaotic_mix",
	"generate_geometric_lattice",
	"generate_phi_chaotic_hybrid",
	"generate_adaptive_phi",
	"generate_hyperbolic_phase",
	"generate_log_periodic_phi_rft",
	"generate_convex_mixed_phi_rft",
	"generate_exact_golden_ratio_unitary",
	"generate_adaptive_split_variant",
	"generate_dct_basis",
	"generate_hybrid_dct_rft",
	"GoldenRatioUnitary",
	"SymbolicUnitary",
	"EntropicUnitary",
	# Patent variants (USPTO 19/169,399)
	"PATENT_VARIANTS",
	"get_patent_variant",
	"list_patent_variants",
	"generate_rft_manifold_projection",
	"generate_rft_sphere_parametric",
	"generate_rft_phase_coherent",
	"generate_rft_entropy_modulated",
	"generate_rft_loxodrome",
	"generate_rft_polar_golden",
	"generate_rft_spiral_golden",
	"generate_rft_complex_exp",
	"generate_rft_winding",
	"generate_rft_torus_parametric",
	"generate_rft_hopf_fibration",
	"generate_rft_bloom_hash",
	"generate_rft_trefoil_knot",
]
