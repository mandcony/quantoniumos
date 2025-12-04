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
]
