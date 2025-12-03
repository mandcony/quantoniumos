"""Shared RFT variant manifest used by runtime and tests."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, Sequence, Tuple

from .registry import VARIANTS, VariantInfo

try:  # Optional: assembly bindings might not build in every environment
    from algorithms.rft.kernels.python_bindings import unitary_rft as _bindings  # type: ignore
except Exception:  # pragma: no cover - bindings unavailable in pure-Python setups
    _bindings = None


@dataclass(frozen=True)
class VariantEntry:
    """Single source of truth for a Φ-RFT variant."""

    code: str  # Short human readable label (STANDARD, HARMONIC, ...)
    registry_key: str  # Key inside algorithms.rft.variants.registry.VARIANTS
    kernel_symbol: Optional[str]  # Name of the enum exported by the native bindings
    kernel_id: Optional[int]  # Integer enum (None if bindings missing)
    info: VariantInfo  # Rich metadata (use cases, generator, etc.)
    experimental: bool = False

    @property
    def label(self) -> str:
        return self.info.name


def _resolve_kernel_value(symbol: Optional[str]) -> Optional[int]:
    if not symbol or _bindings is None:
        return None
    return getattr(_bindings, symbol, None)


# Order matters – matches README/architecture docs and GUI exposure
_BASE_VARIANT_ROWS: Sequence[Tuple[str, str, Optional[str], bool]] = (
    ("STANDARD", "original", "RFT_VARIANT_STANDARD", False),
    ("HARMONIC", "harmonic_phase", "RFT_VARIANT_HARMONIC", False),
    ("FIBONACCI", "fibonacci_tilt", "RFT_VARIANT_FIBONACCI", False),
    ("CHAOTIC", "chaotic_mix", "RFT_VARIANT_CHAOTIC", False),
    ("GEOMETRIC", "geometric_lattice", "RFT_VARIANT_GEOMETRIC", False),
    ("PHI_CHAOTIC", "phi_chaotic_hybrid", "RFT_VARIANT_PHI_CHAOTIC", True),
    ("HYPERBOLIC", "hyperbolic_phase", "RFT_VARIANT_HYPERBOLIC", True),
    ("LOG_PERIODIC", "log_periodic", None, True),
    ("CONVEX_MIX", "convex_mix", None, True),
    ("GOLDEN_EXACT", "golden_ratio_exact", None, True),
    ("CASCADE", "h3_cascade", "RFT_VARIANT_CASCADE", False),
    ("ADAPTIVE_SPLIT", "adaptive_split", "RFT_VARIANT_ADAPTIVE_SPLIT", False),
    ("ENTROPY_GUIDED", "fh5_entropy", "RFT_VARIANT_ENTROPY_GUIDED", False),
    ("DICTIONARY", "h6_dictionary", "RFT_VARIANT_DICTIONARY", False),
)


VARIANT_MANIFEST: Tuple[VariantEntry, ...] = tuple(
    VariantEntry(
        code=code,
        registry_key=registry_key,
        kernel_symbol=kernel_symbol,
        kernel_id=_resolve_kernel_value(kernel_symbol),
        info=VARIANTS[registry_key],
        experimental=experimental,
    )
    for code, registry_key, kernel_symbol, experimental in _BASE_VARIANT_ROWS
    if registry_key in VARIANTS
)


def iter_variants(*, include_experimental: bool = True, require_kernel_constant: bool = False) -> Iterator[VariantEntry]:
    """Yield the requested subset of variants."""

    for entry in VARIANT_MANIFEST:
        if not include_experimental and entry.experimental:
            continue
        if require_kernel_constant and entry.kernel_id is None:
            continue
        yield entry


def get_variant_codes(**kwargs) -> Iterable[str]:
    """Return ordered variant codes (helper for CLI/debug output)."""

    return [entry.code for entry in iter_variants(**kwargs)]


__all__ = ["VariantEntry", "VARIANT_MANIFEST", "iter_variants", "get_variant_codes"]
