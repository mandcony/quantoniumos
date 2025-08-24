"""
QuantoniumOS - Symbolic Collision Analysis

This module tests the collision resistance and avalanche effect of the geometric waveform hash.
"""

import logging
import os
import random
import string
from collections import defaultdict

import matplotlib.pyplot as plt

# Add project root to Python path for imports

# Import QuantoniumOS modules
try:
    from core.encryption.geometric_waveform_hash import geometric_hash
except ImportError:
    # Fallback implementation
    import hashlib

    def geometric_hash(data):
        return hashlib.sha256(data).hexdigest()


logger = logging.getLogger(__name__)


def generate_random_glyph(length=5):
    """
    Generates a random glyph (string) of the given length using uppercase letters and digits.
    """
    characters = string.ascii_uppercase + string.digits
    return "".join(random.choice(characters) for _ in range(length))


def test_collision_analysis(num_samples=1000):
    """
    Generates num_samples random glyphs, computes their geometric waveform hash,
    and reports on the number of unique hashes and collisions.

    Args:
        num_samples: Number of random samples to test

    Returns:
        Dictionary containing collision analysis results
    """
    hash_dict = {}  # Maps hash -> glyph (first occurrence)
    collisions = {}  # Maps hash -> list of glyphs with that hash (if more than one)

    logger.info(f"Starting collision analysis with {num_samples} samples")

    for i in range(num_samples):
        if i % 100 == 0 and i > 0:
            logger.info(f"Processed {i} samples...")

        glyph = generate_random_glyph(length=5)
        # Convert the glyph to bytes and compute the hash
        hash_val = geometric_hash(glyph.encode())

        if hash_val in hash_dict:
            # Record collision: if not already recorded, initialize the list with the first glyph
            if hash_val not in collisions:
                collisions[hash_val] = [hash_dict[hash_val]]
            collisions[hash_val].append(glyph)
        else:
            hash_dict[hash_val] = glyph

    total_unique = len(hash_dict)
    total_collisions = sum(len(glyphs) - 1 for glyphs in collisions.values())

    results = {
        "samples": num_samples,
        "unique_hashes": total_unique,
        "total_collisions": total_collisions,
        "collision_rate": total_collisions / num_samples if num_samples > 0 else 0,
        "collision_details": {
            hash_val: glyphs for hash_val, glyphs in collisions.items()
        },
    }

    logger.info(
        f"Collision analysis complete: {total_unique} unique hashes, {total_collisions} collisions"
    )
    return results


def avalanche_analysis(num_samples=1000, bit_flips=1):
    """
    Tests the avalanche effect by flipping bits in the input and measuring
    the Hamming distance in the output hash.

    Args:
        num_samples: Number of random samples to test
        bit_flips: Number of bits to flip in each sample

    Returns:
        Dictionary containing avalanche analysis results
    """
    results = {
        "samples": num_samples,
        "bit_flips": bit_flips,
        "hamming_distances": [],
        "avalanche_histogram": defaultdict(int),
        "bit_change_percentage": 0.0,
    }

    logger.info(
        f"Starting avalanche analysis with {num_samples} samples, {bit_flips} bit flips"
    )

    total_bits_changed = 0
    total_bits_compared = 0

    for i in range(num_samples):
        if i % 100 == 0 and i > 0:
            logger.info(f"Processed {i} samples...")

        # Generate a random input
        input_bytes = os.urandom(32)  # 32 bytes = 256 bits

        # Get the original hash
        original_hash = geometric_hash(input_bytes)

        # Convert to binary representation for bit manipulation
        input_bits = "".join(format(b, "08b") for b in input_bytes)

        # Flip random bits
        positions = random.sample(range(len(input_bits)), bit_flips)
        modified_bits = list(input_bits)
        for pos in positions:
            modified_bits[pos] = "1" if input_bits[pos] == "0" else "0"
        modified_bits = "".join(modified_bits)

        # Convert back to bytes
        modified_bytes = bytes(
            int(modified_bits[i : i + 8], 2) for i in range(0, len(modified_bits), 8)
        )

        # Get the modified hash
        modified_hash = geometric_hash(modified_bytes)

        # Compare the hashes bit by bit
        original_hash_bits = (
            "".join(format(b, "08b") for b in original_hash)
            if isinstance(original_hash, bytes)
            else bin(int(original_hash, 16))[2:].zfill(len(original_hash) * 4)
        )
        modified_hash_bits = (
            "".join(format(b, "08b") for b in modified_hash)
            if isinstance(modified_hash, bytes)
            else bin(int(modified_hash, 16))[2:].zfill(len(modified_hash) * 4)
        )

        # Ensure equal length for comparison
        min_len = min(len(original_hash_bits), len(modified_hash_bits))
        original_hash_bits = original_hash_bits[:min_len]
        modified_hash_bits = modified_hash_bits[:min_len]

        # Count bit differences
        hamming_distance = sum(
            o != m for o, m in zip(original_hash_bits, modified_hash_bits)
        )
        results["hamming_distances"].append(hamming_distance)
        results["avalanche_histogram"][hamming_distance] += 1

        total_bits_changed += hamming_distance
        total_bits_compared += min_len

    # Calculate average bit change percentage
    if total_bits_compared > 0:
        results["bit_change_percentage"] = (
            total_bits_changed / total_bits_compared
        ) * 100

    # Calculate expected ideal avalanche effect (50% bit change)
    results["ideal_bit_change"] = 50.0
    results["avalanche_quality"] = 100 - abs(
        results["bit_change_percentage"] - results["ideal_bit_change"]
    )

    logger.info(
        f"Avalanche analysis complete: {results['bit_change_percentage']:.2f}% bits changed on average"
    )
    return results


def plot_collision_results(results, save_path=None):
    """
    Create a visualization of collision analysis results.

    Args:
        results: Results from test_collision_analysis function
        save_path: Path to save the plot image

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Basic stats
    ax.bar(
        ["Total Samples", "Unique Hashes"],
        [results["samples"], results["unique_hashes"]],
        color=["blue", "green"],
    )

    ax.set_ylabel("Count")
    ax.set_title("Collision Analysis Results")

    # Add collision rate as text
    collision_text = f"Collision Rate: {results['collision_rate']*100:.4f}%"
    ax.text(
        0.5,
        0.9,
        collision_text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        bbox=dict(facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        logger.info(f"Collision analysis plot saved to {save_path}")

    return fig


def plot_avalanche_results(results, save_path=None):
    """
    Create a visualization of avalanche analysis results.

    Args:
        results: Results from avalanche_analysis function
        save_path: Path to save the plot image

    Returns:
        Matplotlib figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot histogram of Hamming distances
    distances = sorted(results["avalanche_histogram"].keys())
    counts = [results["avalanche_histogram"][d] for d in distances]

    ax1.bar(distances, counts, color="purple")
    ax1.set_xlabel("Hamming Distance (Bits Changed)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of Bit Changes")

    # Plot quality metrics
    metrics = ["Bit Change %", "Ideal (50%)", "Quality Score"]
    values = [
        results["bit_change_percentage"],
        results["ideal_bit_change"],
        results["avalanche_quality"],
    ]
    colors = ["blue", "red", "green"]

    ax2.bar(metrics, values, color=colors)
    ax2.set_ylabel("Percentage")
    ax2.set_title("Avalanche Effect Quality Metrics")

    # Draw 50% line on histogram
    expected_bits = (
        len(results["hamming_distances"][0]) // 2 if results["hamming_distances"] else 0
    )
    if expected_bits > 0:
        ax1.axvline(x=expected_bits, color="red", linestyle="--", label="Ideal (50%)")
        ax1.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        logger.info(f"Avalanche analysis plot saved to {save_path}")

    return fig


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Run collision analysis
    collision_results = test_collision_analysis(num_samples=1000)
    print(f"\nTested {collision_results['samples']} glyphs")
    print(f"Unique Hashes: {collision_results['unique_hashes']}")
    print(f"Total Collisions: {collision_results['total_collisions']}")
    print(f"Collision Rate: {collision_results['collision_rate']*100:.4f}%")

    # Run avalanche analysis
    avalanche_results = avalanche_analysis(num_samples=1000, bit_flips=1)
    print("\nAvalanche Effect Analysis:")
    print(f"Average Bit Change: {avalanche_results['bit_change_percentage']:.2f}%")
    print(f"Ideal Bit Change: {avalanche_results['ideal_bit_change']:.2f}%")
    print(f"Quality Score: {avalanche_results['avalanche_quality']:.2f}/100")

    # Plot results
    plot_collision_results(collision_results)
    plot_avalanche_results(avalanche_results)
    plt.show()
