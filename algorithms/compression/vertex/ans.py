# algorithms/compression/vertex/ans.py
"""
Asymmetric Numeral Systems (ANS) Coder
======================================

This module provides a basic implementation of ranged Asymmetric Numeral Systems (rANS),
a modern entropy coder that offers performance close to arithmetic coding but with
higher throughput. It is used by the RFT vertex codec for near-lossless compression
of quantized spectral coefficients.

This implementation is based on the principles described by Jarek Duda and is
designed to be simple, correct, and self-contained for integration into the
QuantoniumOS compression pipeline.

Key Components:
- `RansEncoder`: Encodes a sequence of symbols given their frequency distribution.
- `RansDecoder`: Decodes the original symbols from the compressed state.
- `build_cumulative_freq_table`: A helper to create the necessary data structures
  for the coder from a frequency distribution.

The coder operates on a state `x`, which is a large integer. Encoding a symbol
updates this state, and decoding a symbol reverses the update. The final state
is serialized as the compressed data.

This implementation is for educational and research purposes and may not be as
optimized as production-level ANS coders found in libraries like CRAM or Zstandard.

Author: Based on public domain rANS principles by Jarek Duda.
License: MIT
"""

from collections import Counter
import numpy as np

# --- Constants ---
RANS_L_BASE = 2**16  # Lower bound of the rANS state interval
RANS_PRECISION_DEFAULT = 12  # Default bit precision for frequencies

# --- Frequency Table Builder ---

def build_cumulative_freq_table(frequencies, precision):
    """
    Builds a cumulative frequency table required for the rANS coder.

    Args:
        frequencies (dict or Counter): A map from symbol to its frequency count.
        precision (int): The number of bits for the total frequency count.

    Returns:
        tuple: A tuple containing:
            - symbols (list): A list of unique symbols.
            - cumulative_freqs (np.ndarray): Cumulative frequencies for each symbol.
            - total_freq (int): The total frequency count (2**precision).
    """
    if not frequencies:
        raise ValueError("Frequency distribution cannot be empty.")

    total = sum(frequencies.values())
    total_freq = 1 << precision
    
    symbols = sorted(frequencies.keys())
    symbol_map = {s: i for i, s in enumerate(symbols)}
    
    norm_freqs = np.zeros(len(symbols), dtype=np.uint32)
    
    for i, s in enumerate(symbols):
        # Normalize frequencies to the precision range
        norm_freqs[i] = max(1, int(frequencies[s] * total_freq / total))

    # Adjust frequencies to sum exactly to total_freq
    current_sum = norm_freqs.sum()
    if current_sum < total_freq:
        norm_freqs[np.argmax(norm_freqs)] += total_freq - current_sum
    elif current_sum > total_freq:
        while norm_freqs.sum() > total_freq:
            norm_freqs[np.argmax(norm_freqs)] -= 1

    assert norm_freqs.sum() == total_freq, "Normalized frequencies do not sum to total_freq"

    # Create cumulative frequency table
    cumulative_freqs = np.zeros(len(symbols) + 1, dtype=np.uint32)
    cumulative_freqs[1:] = np.cumsum(norm_freqs)
    
    return symbols, symbol_map, cumulative_freqs, total_freq


# --- rANS Encoder ---

class RansEncoder:
    def __init__(self, precision=RANS_PRECISION_DEFAULT):
        self.state = RANS_L_BASE
        self.precision = precision
        self.encoded_data = []

    def encode_symbol(self, symbol, symbol_map, cumulative_freqs, total_freq):
        """Encodes a single symbol."""
        sym_idx = symbol_map[symbol]
        start_freq = cumulative_freqs[sym_idx]
        freq = cumulative_freqs[sym_idx + 1] - start_freq

        # Renormalize state if it's too large
        if self.state >= RANS_L_BASE * freq:
            self.encoded_data.append(self.state & 0xFFFF)
            self.state >>= 16

        # Update state
        self.state = ((self.state // freq) << self.precision) + (self.state % freq) + start_freq

    def get_encoded_data(self):
        """Finalizes encoding and returns the compressed data."""
        # Flush the final state into the data buffer
        final_state = self.state
        while final_state > 0:
            self.encoded_data.append(final_state & 0xFFFF)
            final_state >>= 16
        return np.array(self.encoded_data, dtype=np.uint16)


# --- rANS Decoder ---

class RansDecoder:
    def __init__(self, encoded_data):
        self.encoded_data = list(encoded_data)
        # Reconstruct the initial state from the end of the data
        self.state = 0
        while self.state < RANS_L_BASE and self.encoded_data:
            self.state = (self.state << 16) | self.encoded_data.pop()

    def decode_symbol(self, symbols, cumulative_freqs, total_freq):
        """Decodes a single symbol."""
        # Find symbol from state
        slot = self.state & (total_freq - 1)
        sym_idx = np.searchsorted(cumulative_freqs, slot, side='right') - 1
        symbol = symbols[sym_idx]

        start_freq = cumulative_freqs[sym_idx]
        freq = cumulative_freqs[sym_idx + 1] - start_freq

        # Update state
        self.state = freq * (self.state >> RANS_PRECISION_DEFAULT) + slot - start_freq

        # Renormalize state if it's too small
        if self.state < RANS_L_BASE and self.encoded_data:
            self.state = (self.state << 16) | self.encoded_data.pop()
            
        return symbol


# --- High-level API ---

def ans_encode(data_to_encode, precision=RANS_PRECISION_DEFAULT):
    """
    Encodes a sequence of symbols using rANS.

    Args:
        data_to_encode (list or np.ndarray): The sequence of symbols to encode.
        precision (int): The precision for frequency distribution.

    Returns:
        tuple: A tuple containing:
            - encoded_data (np.ndarray): The compressed data.
            - freq_data (dict): The frequency distribution needed for decoding.
    """
    if len(data_to_encode) == 0:
        return np.array([], dtype=np.uint16), {}

    frequencies = Counter(data_to_encode)
    symbols, symbol_map, cumulative_freqs, total_freq = build_cumulative_freq_table(frequencies, precision)
    
    encoder = RansEncoder(precision)
    
    # Encode in reverse order
    for symbol in reversed(data_to_encode):
        encoder.encode_symbol(symbol, symbol_map, cumulative_freqs, total_freq)
        
    encoded_data = encoder.get_encoded_data()
    
    # Prepare frequency data for decoder
    freq_data = {
        "frequencies": dict(frequencies),
        "precision": precision
    }
    
    return encoded_data, freq_data


def ans_decode(encoded_data, freq_data, num_symbols):
    """
    Decodes a sequence of symbols from rANS compressed data.

    Args:
        encoded_data (np.ndarray): The compressed data.
        freq_data (dict): The frequency distribution used during encoding.
        num_symbols (int): The number of symbols to decode.

    Returns:
        list: The decoded sequence of symbols.
    """
    if num_symbols == 0:
        return []

    frequencies = freq_data["frequencies"]
    precision = freq_data["precision"]
    
    symbols, _, cumulative_freqs, total_freq = build_cumulative_freq_table(frequencies, precision)
    
    decoder = RansDecoder(encoded_data)
    
    decoded_data = []
    for _ in range(num_symbols):
        decoded_data.append(decoder.decode_symbol(symbols, cumulative_freqs, total_freq))
        
    return decoded_data

__all__ = ["ans_encode", "ans_decode", "RANS_PRECISION_DEFAULT"]
