"""
QuantoniumOS - Cryptographic Diffusion and Avalanche Testing

This module implements diffusion functions to ensure strong avalanche effects
for the QuantoniumOS cryptographic primitives. It also provides testing functions
to verify avalanche properties.
"""

import logging
import secrets
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

# Configure logger
logger = logging.getLogger("crypto_diffusion")
logger.setLevel(logging.INFO)

# Constants for MDS matrix (Maximum Distance Separable)
# 4×4 MDS matrix with branch number 5 (optimal for 4×4)
MDS_MATRIX = np.array(
    [[2, 3, 1, 1], [1, 2, 3, 1], [1, 1, 2, 3], [3, 1, 1, 2]], dtype=np.uint8
)

# Inverse MDS matrix for decryption
MDS_MATRIX_INV = np.array(
    [[12, 1, 13, 8], [8, 12, 1, 13], [13, 8, 12, 1], [1, 13, 8, 12]], dtype=np.uint8
)


def bytes_to_matrix(data: bytes) -> np.ndarray:
    """
    Convert bytes to a 4×4 matrix (column-major order)

    Args:
        data: Input bytes (must be exactly 16 bytes)

    Returns:
        4×4 numpy array
    """
    if len(data) != 16:
        raise ValueError(f"Data must be exactly 16 bytes (got {len(data)})")

    # Convert to numpy array and reshape to 4×4
    array = np.frombuffer(data, dtype=np.uint8)
    return array.reshape(4, 4)


def matrix_to_bytes(matrix: np.ndarray) -> bytes:
    """
    Convert a 4×4 matrix back to bytes (column-major order)

    Args:
        matrix: 4×4 numpy array

    Returns:
        16-byte bytes object
    """
    if matrix.shape != (4, 4):
        raise ValueError(f"Matrix must be 4×4 (got {matrix.shape})")

    # Flatten and convert to bytes
    return matrix.flatten().tobytes()


def galois_multiply(a: int, b: int) -> int:
    """
    Multiply two numbers in the Galois Field GF(2^8)

    Args:
        a: First byte
        b: Second byte

    Returns:
        Result of multiplication in GF(2^8)
    """
    p = 0
    for _ in range(8):
        if b & 1:
            p ^= a
        high_bit_set = a & 0x80
        a <<= 1
        if high_bit_set:
            a ^= 0x1B  # Irreducible polynomial x^8 + x^4 + x^3 + x + 1
        b >>= 1

    return p & 0xFF


def mix_columns(state: np.ndarray) -> np.ndarray:
    """
    Apply MDS matrix to each column of the state matrix for diffusion

    Args:
        state: 4×4 state matrix

    Returns:
        4×4 state matrix after mixing columns
    """
    result = np.zeros((4, 4), dtype=np.uint8)

    for col in range(4):
        for row in range(4):
            value = 0
            for i in range(4):
                value ^= galois_multiply(MDS_MATRIX[row, i], state[i, col])
            result[row, col] = value

    return result


def inv_mix_columns(state: np.ndarray) -> np.ndarray:
    """
    Apply inverse MDS matrix to each column of the state matrix

    Args:
        state: 4×4 state matrix

    Returns:
        4×4 state matrix after inverse mixing columns
    """
    result = np.zeros((4, 4), dtype=np.uint8)

    for col in range(4):
        for row in range(4):
            value = 0
            for i in range(4):
                value ^= galois_multiply(MDS_MATRIX_INV[row, i], state[i, col])
            result[row, col] = value

    return result


def shift_rows(state: np.ndarray) -> np.ndarray:
    """
    Shift rows of the state matrix for diffusion
    Row 0: No shift
    Row 1: Shift left by 1
    Row 2: Shift left by 2
    Row 3: Shift left by 3

    Args:
        state: 4×4 state matrix

    Returns:
        4×4 state matrix after shifting rows
    """
    result = state.copy()

    # Row 0: No shift
    # Row 1: Shift left by 1
    result[1] = np.roll(result[1], -1)
    # Row 2: Shift left by 2
    result[2] = np.roll(result[2], -2)
    # Row 3: Shift left by 3
    result[3] = np.roll(result[3], -3)

    return result


def inv_shift_rows(state: np.ndarray) -> np.ndarray:
    """
    Inverse of shift_rows

    Args:
        state: 4×4 state matrix

    Returns:
        4×4 state matrix after inverse shifting rows
    """
    result = state.copy()

    # Row 0: No shift
    # Row 1: Shift right by 1
    result[1] = np.roll(result[1], 1)
    # Row 2: Shift right by 2
    result[2] = np.roll(result[2], 2)
    # Row 3: Shift right by 3
    result[3] = np.roll(result[3], 3)

    return result


def add_diffusion_layer(block: bytes) -> bytes:
    """
    Add a strong diffusion layer to a 16-byte block

    This function:
    1. Converts the block to a 4×4 matrix
    2. Applies ShiftRows operation
    3. Applies MixColumns operation using MDS matrix
    4. Converts back to bytes

    Args:
        block: 16-byte input block

    Returns:
        16-byte output block with diffusion applied
    """
    # Convert to matrix
    state = bytes_to_matrix(block)

    # Apply ShiftRows
    state = shift_rows(state)

    # Apply MixColumns
    state = mix_columns(state)

    # Convert back to bytes
    return matrix_to_bytes(state)


def remove_diffusion_layer(block: bytes) -> bytes:
    """
    Reverse the diffusion layer for a 16-byte block (for decryption)

    This function:
    1. Converts the block to a 4×4 matrix
    2. Applies inverse MixColumns operation
    3. Applies inverse ShiftRows operation
    4. Converts back to bytes

    Args:
        block: 16-byte input block

    Returns:
        16-byte output block with diffusion reversed
    """
    # Convert to matrix
    state = bytes_to_matrix(block)

    # Inverse operations in reverse order
    state = inv_mix_columns(state)
    state = inv_shift_rows(state)

    # Convert back to bytes
    return matrix_to_bytes(state)


def add_multi_round_diffusion(data: bytes, rounds: int = 3) -> bytes:
    """
    Apply multiple rounds of diffusion to ensure strong avalanche effect

    Args:
        data: Input data bytes
        rounds: Number of diffusion rounds (3 recommended minimum)

    Returns:
        Data with diffusion applied
    """
    result = bytearray(data)

    # Apply diffusion to each 16-byte block
    for i in range(0, len(result), 16):
        block = bytes(result[i : i + 16])

        # Pad block if needed
        if len(block) < 16:
            block = block + b"\x00" * (16 - len(block))

        # Apply multiple rounds of diffusion
        for _ in range(rounds):
            block = add_diffusion_layer(block)

        # Copy back to result
        result[i : i + 16] = block[: len(result[i : i + 16])]

    return bytes(result)


def remove_multi_round_diffusion(data: bytes, rounds: int = 3) -> bytes:
    """
    Remove multiple rounds of diffusion (for decryption)

    Args:
        data: Input data bytes
        rounds: Number of diffusion rounds (must match encryption)

    Returns:
        Original data with diffusion removed
    """
    result = bytearray(data)

    # Remove diffusion from each 16-byte block
    for i in range(0, len(result), 16):
        block = bytes(result[i : i + 16])

        # Pad block if needed
        if len(block) < 16:
            block = block + b"\x00" * (16 - len(block))

        # Apply multiple rounds of inverse diffusion
        for _ in range(rounds):
            block = remove_diffusion_layer(block)

        # Copy back to result
        result[i : i + 16] = block[: len(result[i : i + 16])]

    return bytes(result)


def test_avalanche_effect(
    data: bytes, rounds: int = 3, bits_to_flip: int = 1
) -> Dict[str, Any]:
    """
    Test avalanche effect by flipping bits and measuring diffusion

    Args:
        data: Input data bytes
        rounds: Number of diffusion rounds
        bits_to_flip: Number of bits to flip for testing

    Returns:
        Dictionary with avalanche statistics
    """
    if len(data) < 16:
        data = data + b"\x00" * (16 - len(data))

    # Original diffused output
    original_output = add_multi_round_diffusion(data, rounds)

    # Stats
    total_bits = len(original_output) * 8
    bit_changes = []

    # Test flipping each bit
    for bit_pos in range(
        min(total_bits, 128)
    ):  # Limit to first 128 bits for large data
        # Create modified input with one bit flipped
        modified = bytearray(data)
        byte_pos = bit_pos // 8
        bit_in_byte = bit_pos % 8
        modified[byte_pos] ^= 1 << bit_in_byte

        # Apply diffusion
        modified_output = add_multi_round_diffusion(bytes(modified), rounds)

        # Count differing bits
        diff_bits = 0
        for a, b in zip(original_output, modified_output):
            xor_byte = a ^ b
            for i in range(8):
                if (xor_byte >> i) & 1:
                    diff_bits += 1

        # Calculate percentage of bits changed
        percentage = (diff_bits / total_bits) * 100
        bit_changes.append(percentage)

    # Calculate statistics
    avg_change = np.mean(bit_changes)
    std_dev = np.std(bit_changes)
    min_change = np.min(bit_changes)
    max_change = np.max(bit_changes)

    return {
        "average_bit_change_percent": avg_change,
        "std_deviation": std_dev,
        "min_bit_change_percent": min_change,
        "max_bit_change_percent": max_change,
        "total_bits_in_output": total_bits,
        "diffusion_rounds": rounds,
        "bits_tested": len(bit_changes),
        "ideal_bit_change": 50.0,
        "deviation_from_ideal": abs(50.0 - avg_change),
    }


def plot_avalanche_distribution(
    data: bytes, rounds: list = [1, 2, 3, 4], output_file: str = None
):
    """
    Plot avalanche effect distribution for different numbers of rounds

    Args:
        data: Input data bytes
        rounds: List of round counts to test
        output_file: Path to save the plot
    """
    plt.figure(figsize=(12, 8))

    for r in rounds:
        # Test avalanche effect
        result = test_avalanche_effect(data, rounds=r)

        # Create histogram data from individual bit changes
        if len(data) < 16:
            data_padded = data + b"\x00" * (16 - len(data))
        else:
            data_padded = data[:16]  # Use first 16 bytes for testing

        original_output = add_multi_round_diffusion(data_padded, r)

        bit_changes = []
        for bit_pos in range(128):  # Test first 128 bits
            # Create modified input with one bit flipped
            modified = bytearray(data_padded)
            byte_pos = bit_pos // 8
            bit_in_byte = bit_pos % 8
            modified[byte_pos] ^= 1 << bit_in_byte

            # Apply diffusion
            modified_output = add_multi_round_diffusion(bytes(modified), r)

            # Count differing bits
            diff_bits = 0
            for a, b in zip(original_output, modified_output):
                xor_byte = a ^ b
                for i in range(8):
                    if (xor_byte >> i) & 1:
                        diff_bits += 1

            # Calculate percentage of bits changed
            percentage = (diff_bits / (len(original_output) * 8)) * 100
            bit_changes.append(percentage)

        # Plot histogram
        plt.hist(
            bit_changes,
            bins=20,
            alpha=0.5,
            label=f"{r} rounds (avg: {result['average_bit_change_percent']:.2f}%)",
        )

    plt.axvline(x=50, color="r", linestyle="--", label="Ideal (50%)")
    plt.xlabel("Percentage of Bits Changed")
    plt.ylabel("Frequency")
    plt.title("Avalanche Effect Distribution by Number of Diffusion Rounds")
    plt.legend()
    plt.grid(True)

    if output_file:
        plt.savefig(output_file)
    else:
        plt.savefig("docs/avalanche_distribution.png")

    plt.close()


def generate_test_vectors(count: int = 10, key_size: int = 32) -> List[Dict[str, Any]]:
    """
    Generate test vectors for diffusion layer validation

    Args:
        count: Number of test vectors to generate
        key_size: Size of key in bytes

    Returns:
        List of test vector dictionaries
    """
    test_vectors = []

    for i in range(count):
        # Generate random plaintext and key
        plaintext = secrets.token_bytes(16)  # 16-byte block
        key = secrets.token_bytes(key_size)

        # Apply diffusion
        ciphertext = add_multi_round_diffusion(plaintext)

        # Create test vector
        vector = {
            "test_number": i + 1,
            "plaintext": plaintext.hex(),
            "key": key.hex(),
            "ciphertext": ciphertext.hex(),
            "diffusion_rounds": 3,
        }

        test_vectors.append(vector)

    return test_vectors


def validate_diffusion_implementation():
    """
    Validate that diffusion layer correctly implements inverse operations

    Returns:
        True if validation passes, False otherwise
    """
    # Generate test data
    test_data = secrets.token_bytes(64)

    # Test diffusion and inverse for various round counts
    for rounds in [1, 2, 3, 4]:
        # Apply diffusion
        diffused = add_multi_round_diffusion(test_data, rounds)

        # Remove diffusion
        original = remove_multi_round_diffusion(diffused, rounds)

        # Check if original data is recovered
        if original != test_data:
            logger.error(f"Diffusion validation failed for {rounds} rounds")
            return False

    # Test individual block operations
    test_block = secrets.token_bytes(16)

    # Test ShiftRows and inverse
    matrix = bytes_to_matrix(test_block)
    shifted = shift_rows(matrix)
    unshifted = inv_shift_rows(shifted)

    if not np.array_equal(matrix, unshifted):
        logger.error("ShiftRows validation failed")
        return False

    # Test MixColumns and inverse
    mixed = mix_columns(matrix)
    unmixed = inv_mix_columns(mixed)

    if not np.array_equal(matrix, unmixed):
        logger.error("MixColumns validation failed")
        return False

    logger.info("All diffusion operations validated successfully")
    return True


if __name__ == "__main__":
    # Run validation tests
    if validate_diffusion_implementation():
        print("Diffusion layer implementation validated successfully")
    else:
        print("Diffusion layer validation failed")

    # Generate test data
    test_data = secrets.token_bytes(64)

    # Test and plot avalanche effect
    for rounds in [1, 2, 3, 4]:
        result = test_avalanche_effect(test_data, rounds)
        print(f"Avalanche effect with {rounds} rounds:")
        print(f" Average bit change: {result['average_bit_change_percent']:.2f}%")
        print(f" Deviation from ideal: {result['deviation_from_ideal']:.2f}%")
        print(f" Standard deviation: {result['std_deviation']:.2f}%")

    # Plot avalanche distribution
    plot_avalanche_distribution(test_data)

    # Generate and print test vectors
    test_vectors = generate_test_vectors(5)
    print("\nTest Vectors:")
    for i, vector in enumerate(test_vectors):
        print(f"Vector {i+1}:")
        print(f" Plaintext: {vector['plaintext']}")
        print(f" Key: {vector['key']}")
        print(f" Ciphertext: {vector['ciphertext']}")
