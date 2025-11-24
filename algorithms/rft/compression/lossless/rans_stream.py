# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""Simple rANS encoder/decoder for integer symbol streams.

The implementation follows a single-state range Asymmetric Numeral Systems (rANS)
with configurable precision. Frequencies are derived from the symbol histogram
with Laplace smoothing so every symbol remains decodable.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Iterable, List

RANS_PRECISION_DEFAULT = 12  # Total = 4096 histogram buckets
RANS_L = 1 << 23  # Lower renormalisation bound
RANS_BASE_MASK = (1 << 32) - 1


@dataclass
class _RansTable:
    precision: int
    freq: List[int]
    cum: List[int]
    lookup: List[int]

    @property
    def total(self) -> int:
        return 1 << self.precision


def _normalise_frequencies(counts: List[int], precision: int) -> List[int]:
    total = sum(counts)
    if total == 0:
        raise ValueError("Counts must not all be zero")
    target = 1 << precision
    freqs = [max(1, (count * target) // total) for count in counts]
    current = sum(freqs)
    if current == target:
        return freqs

    length = len(freqs)
    if current < target:
        diff = target - current
        idx = 0
        while diff > 0:
            freqs[idx] += 1
            diff -= 1
            idx = (idx + 1) % length
        return freqs

    deficit = current - target
    order = sorted(range(length), key=freqs.__getitem__, reverse=True)
    while deficit > 0:
        changed = False
        for idx in order:
            if freqs[idx] > 1:
                freqs[idx] -= 1
                deficit -= 1
                changed = True
                if deficit == 0:
                    break
        if not changed:
            raise ValueError("Unable to normalise frequencies with given precision")
    return freqs


def _build_table(symbols: Iterable[int], alphabet_size: int, precision: int) -> _RansTable:
    if alphabet_size <= 0:
        raise ValueError("alphabet_size must be positive")
    if alphabet_size > (1 << precision):
        raise ValueError("alphabet_size cannot exceed 2^precision")
    counts = [1] * alphabet_size  # Laplace smoothing
    for sym in symbols:
        if sym < 0 or sym >= alphabet_size:
            raise ValueError(f"symbol {sym} outside alphabet size {alphabet_size}")
        counts[sym] += 1
    freqs = _normalise_frequencies(counts, precision)
    cum = [0] * alphabet_size
    acc = 0
    for i, f in enumerate(freqs):
        cum[i] = acc
        acc += f
    if acc != 1 << precision:
        raise AssertionError("frequency normalisation failed")
    lookup = [0] * (1 << precision)
    for sym, f in enumerate(freqs):
        start = cum[sym]
        for offset in range(f):
            lookup[start + offset] = sym
    return _RansTable(precision=precision, freq=freqs, cum=cum, lookup=lookup)


def ans_encode(symbols: Iterable[int], alphabet_size: int, precision: int = RANS_PRECISION_DEFAULT) -> bytes:
    """Encode a sequence of non-negative integers into bytes via rANS."""
    symbols_list = list(symbols)
    table = _build_table(symbols_list, alphabet_size, precision)
    state = RANS_L
    out = bytearray()
    mask = (1 << precision) - 1
    for sym in reversed(symbols_list):
        freq = table.freq[sym]
        cum = table.cum[sym]
        while state >= (freq << (32 - precision)):
            out.append(state & 0xFF)
            state >>= 8
        state = ((state // freq) << precision) + (state % freq) + cum
    payload = bytes(out)
    header = struct.pack("<BHI", precision, alphabet_size, len(symbols_list))
    freq_bytes = b"".join(struct.pack("<H", f) for f in table.freq)
    return header + freq_bytes + struct.pack("<Q", state) + payload


def ans_decode(blob: bytes) -> List[int]:
    """Decode a byte stream produced by :func:`ans_encode`."""
    if len(blob) < struct.calcsize("<BHI") + 8:
        raise ValueError("Encoded payload too small")
    precision, alphabet_size, length = struct.unpack("<BHI", blob[: struct.calcsize("<BHI")])
    if precision <= 0 or precision > 16:
        raise ValueError("Unsupported precision")
    if alphabet_size <= 0 or alphabet_size > (1 << precision):
        raise ValueError("Invalid alphabet size in payload")
    freq_bytes_offset = struct.calcsize("<BHI")
    freq_bytes_len = alphabet_size * 2
    freq_data = blob[freq_bytes_offset : freq_bytes_offset + freq_bytes_len]
    freqs = [struct.unpack_from("<H", freq_data, 2 * i)[0] for i in range(alphabet_size)]
    state_offset = freq_bytes_offset + freq_bytes_len
    state = struct.unpack_from("<Q", blob, state_offset)[0]
    if state < RANS_L:
        raise ValueError("Invalid initial state in payload")
    payload = blob[state_offset + 8 :]
    table = _RansTable(
        precision=precision,
        freq=freqs,
        cum=[0] * alphabet_size,
        lookup=[0] * (1 << precision),
    )
    acc = 0
    for i, f in enumerate(freqs):
        table.cum[i] = acc
        acc += f
    if acc != 1 << precision:
        raise ValueError("Invalid frequency table in payload")
    mask = (1 << precision) - 1
    for sym, f in enumerate(freqs):
        start = table.cum[sym]
        for offset in range(f):
            table.lookup[start + offset] = sym
    buf = payload
    ptr = len(buf) - 1
    result: List[int] = []
    for _ in range(length):
        x = state & mask
        sym = table.lookup[x]
        freq = table.freq[sym]
        cum = table.cum[sym]
        state = freq * (state >> precision) + (x - cum)
        while state < RANS_L and ptr >= 0:
            state = (state << 8) | buf[ptr]
            ptr -= 1
        result.append(sym)
    if ptr >= 0:
        raise ValueError("Malformed ANS payload: remaining bytes after decode")
    result.reverse()
    return result