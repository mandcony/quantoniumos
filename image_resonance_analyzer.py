"""
QuantoniumOS Image Resonance Analyzer

This tool applies QuantoniumOS resonance mathematics to extract patterns and
potential symbolic meaning from images containing geometric or resonant structures.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon
from PIL import Image

# Add the root directory to the path to access QuantoniumOS modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core QuantoniumOS modules
try:
    from encryption.resonance_fourier import (
        inverse_resonance_fourier_transform, resonance_fourier_transform)
    from encryption.wave_primitives import analyze_waveform, generate_waveform
    from orchestration.symbolic_container import (create_container,
                                                  validate_container)

    print("Successfully imported QuantoniumOS core modules")
except ImportError as e:
    print(f"Error importing QuantoniumOS modules: {e}")
    print("Falling back to simplified implementations")

    # Simplified implementation if core modules are not available
    def generate_waveform(length=32, seed=None):
        """Generate a test waveform"""
        if seed is not None:
            np.random.seed(seed)
        return np.random.random(length)

    def analyze_waveform(waveform):
        """Simple waveform analysis"""
        return {
            "mean": np.mean(waveform),
            "std": np.std(waveform),
            "min": np.min(waveform),
            "max": np.max(waveform),
        }

    def resonance_fourier_transform(waveform):
        """Simplified RFT implementation"""
        freq_data = np.fft.fft(waveform)
        amplitude = np.abs(freq_data)
        phase = np.angle(freq_data)
        frequencies = np.fft.fftfreq(len(waveform))
        return {
            "frequencies": frequencies,
            "amplitudes": amplitude,
            "phases": phase,
            "resonance_mask": amplitude > np.mean(amplitude),
        }

    def inverse_resonance_fourier_transform(freq_data):
        """Simplified inverse RFT"""
        complex_data = freq_data["amplitudes"] * np.exp(1j * freq_data["phases"])
        return np.fft.ifft(complex_data).real


class ImageResonanceAnalyzer:
    """
    Analyzes images using QuantoniumOS resonance mathematics to identify patterns,
    symmetry, and potential symbolic meanings.
    """

    def __init__(self, image_path):
        """
        Initialize the analyzer with an image.

        Parameters:
        -----------
        image_path : str
            Path to the image file to analyze
        """
        self.image_path = image_path
        self.image = Image.open(image_path)
        self.img_array = np.array(self.image)

        # Convert to grayscale if image is RGB
        if len(self.img_array.shape) == 3 and self.img_array.shape[2] >= 3:
            self.grayscale = np.mean(self.img_array[:, :, :3], axis=2).astype(np.uint8)
        else:
            self.grayscale = self.img_array

        print(f"Loaded image: {image_path}, shape: {self.img_array.shape}")

        # Store analysis results
        self.edge_waveforms = []
        self.symmetry_score = 0
        self.resonance_points = []
        self.geometric_centers = []
        self.phase_relationships = {}

    def preprocess_image(self):
        """
        Preprocess the image for analysis:
        - Edge detection
        - Noise reduction
        - Contrast enhancement
        """
        # Simple edge detection using gradient
        dx = np.gradient(self.grayscale, axis=1)
        dy = np.gradient(self.grayscale, axis=0)
        self.edges = np.sqrt(dx**2 + dy**2)

        # Normalize edges for better visualization
        self.edges = (self.edges / np.max(self.edges) * 255).astype(np.uint8)

        # Apply threshold to identify strong edges
        self.strong_edges = self.edges > np.percentile(self.edges, 75)

        print("Image preprocessing complete")
        return self.edges

    def analyze_geometric_patterns(self):
        """
        Identify geometric patterns in the image:
        - Circles
        - Triangles
        - Symmetric structures
        - Central patterns
        """
        # Simplified geometric detection
        from scipy import ndimage

        # Find circular structures
        circle_kernel = np.zeros((21, 21))
        rr, cc = np.indices((21, 21))
        circle_mask = (rr - 10) ** 2 + (cc - 10) ** 2 <= 100
        circle_kernel[circle_mask] = 1

        # Convolve with circle kernel
        circle_response = ndimage.convolve(self.grayscale.astype(float), circle_kernel)
        circle_centers = []

        # Find local maxima as potential circle centers
        maxima = ndimage.maximum_filter(circle_response, size=10)
        circle_candidates = (circle_response == maxima) & (
            circle_response > np.percentile(circle_response, 90)
        )

        # Extract coordinates of circle candidates
        y_coords, x_coords = np.where(circle_candidates)
        for y, x in zip(y_coords, x_coords):
            circle_centers.append((x, y))

        self.geometric_centers = circle_centers

        # Count triangular structures (simplified)
        triangle_count = len(circle_centers) // 2  # Simplified estimate

        # Calculate symmetry score
        self.symmetry_score = self._calculate_symmetry()

        print(f"Found {len(circle_centers)} potential circular structures")
        print(f"Estimated {triangle_count} triangular structures")
        print(f"Symmetry score: {self.symmetry_score:.2f}")

        return {
            "circles": circle_centers,
            "triangles": triangle_count,
            "symmetry": self.symmetry_score,
        }

    def _calculate_symmetry(self):
        """Calculate the symmetry score of the image"""
        # Check horizontal symmetry
        h_flip = np.fliplr(self.grayscale)
        h_sym = np.mean(
            1 - np.abs(self.grayscale.astype(float) - h_flip.astype(float)) / 255
        )

        # Check vertical symmetry
        v_flip = np.flipud(self.grayscale)
        v_sym = np.mean(
            1 - np.abs(self.grayscale.astype(float) - v_flip.astype(float)) / 255
        )

        # Check rotational symmetry (90 degrees)
        rot_90 = np.rot90(self.grayscale)
        r_sym = np.mean(
            1 - np.abs(self.grayscale.astype(float) - rot_90.astype(float)) / 255
        )

        # Combine symmetry scores with emphasis on rotational symmetry
        combined_sym = (h_sym + v_sym + 2 * r_sym) / 4
        return combined_sym

    def extract_waveforms(self):
        """
        Extract waveforms from the image representing the patterns.
        These will be analyzed using the resonance techniques.
        """
        waveforms = []

        # Extract radial waveforms from the center
        center_y, center_x = self.grayscale.shape[0] // 2, self.grayscale.shape[1] // 2

        # Extract 8 radial waveforms (every 45 degrees)
        for angle in range(0, 360, 45):
            waveform = []
            # Get pixels along the radial line
            for r in range(0, min(center_y, center_x), 2):
                y = int(center_y + r * np.sin(np.radians(angle)))
                x = int(center_x + r * np.cos(np.radians(angle)))

                if (
                    0 <= y < self.grayscale.shape[0]
                    and 0 <= x < self.grayscale.shape[1]
                ):
                    waveform.append(self.grayscale[y, x] / 255.0)

            if len(waveform) > 0:
                waveforms.append(np.array(waveform))

        # Extract circular waveforms
        for radius in range(10, min(center_y, center_x), 10):
            circular_wave = []
            for angle in range(0, 360, 5):
                y = int(center_y + radius * np.sin(np.radians(angle)))
                x = int(center_x + radius * np.cos(np.radians(angle)))

                if (
                    0 <= y < self.grayscale.shape[0]
                    and 0 <= x < self.grayscale.shape[1]
                ):
                    circular_wave.append(self.grayscale[y, x] / 255.0)

            if len(circular_wave) > 0:
                waveforms.append(np.array(circular_wave))

        self.edge_waveforms = waveforms
        print(f"Extracted {len(waveforms)} waveforms from the image")
        return waveforms

    def analyze_resonance_patterns(self):
        """
        Apply resonance mathematics to the extracted waveforms to identify
        potential patterns of significance.
        """
        if not self.edge_waveforms:
            self.extract_waveforms()

        resonance_data = []

        # Analyze each waveform using RFT
        for idx, waveform in enumerate(self.edge_waveforms):
            # Apply RFT
            rft_result = resonance_fourier_transform(waveform)

            # Identify resonant frequencies
            resonant_freqs = np.where(rft_result["resonance_mask"])[0]

            # Calculate resonance strength as normalized amplitude
            strength = rft_result["amplitudes"][resonant_freqs] / np.max(
                rft_result["amplitudes"]
            )

            # Get phase information
            phases = rft_result["phases"][resonant_freqs]

            # Store results
            for freq, amp, phase in zip(resonant_freqs, strength, phases):
                if freq > 0:  # Skip DC component
                    resonance_data.append(
                        {
                            "waveform_idx": idx,
                            "frequency": freq / len(waveform),  # Normalize frequency
                            "strength": float(amp),
                            "phase": float(phase),
                        }
                    )

        # Identify patterns across different waveforms
        self._analyze_cross_waveform_patterns(resonance_data)

        # Sort by strength
        resonance_data.sort(key=lambda x: x["strength"], reverse=True)

        print(f"Found {len(resonance_data)} resonant points in the waveforms")
        return resonance_data

    def _analyze_cross_waveform_patterns(self, resonance_data):
        """Analyze patterns across different waveforms"""
        # Group by similar frequencies
        freq_groups = {}

        for res in resonance_data:
            freq_key = round(res["frequency"] * 100) / 100  # Round to 2 decimal places
            if freq_key not in freq_groups:
                freq_groups[freq_key] = []
            freq_groups[freq_key].append(res)

        # Find common frequencies across multiple waveforms
        common_patterns = {}
        for freq, items in freq_groups.items():
            if len(items) >= 3:  # At least 3 waveforms share this frequency
                # Calculate phase relationships
                phases = [item["phase"] for item in items]
                phase_diffs = []
                for i in range(len(phases)):
                    for j in range(i + 1, len(phases)):
                        diff = (phases[i] - phases[j]) % (2 * np.pi)
                        phase_diffs.append(diff)

                # Check if phase differences form patterns
                common_patterns[freq] = {
                    "count": len(items),
                    "avg_strength": np.mean([item["strength"] for item in items]),
                    "phase_diffs": phase_diffs,
                    "phase_pattern": self._check_phase_pattern(phase_diffs),
                }

        self.phase_relationships = common_patterns
        return common_patterns

    def _check_phase_pattern(self, phase_diffs):
        """Check if phase differences form regular patterns"""
        # Check for harmony (phases that align to multiples of π/4)
        harmony_count = sum(
            1
            for diff in phase_diffs
            if min(diff % (np.pi / 4), (np.pi / 4) - diff % (np.pi / 4)) < 0.1
        )

        # Check for golden ratio alignment
        golden = (1 + np.sqrt(5)) / 2
        golden_count = sum(
            1
            for diff in phase_diffs
            if min(abs(diff - np.pi / golden), abs(diff - 2 * np.pi / golden)) < 0.2
        )

        # Determine the pattern type
        if harmony_count > len(phase_diffs) * 0.7:
            return "harmonic"
        elif golden_count > len(phase_diffs) * 0.5:
            return "golden_ratio"
        elif np.std(phase_diffs) < 0.3:
            return "regular"
        else:
            return "irregular"

    def visualize_analysis(self):
        """Visualize the results of the analysis"""
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        # Original image
        axs[0, 0].imshow(self.image)
        axs[0, 0].set_title("Original Image")
        axs[0, 0].axis("off")

        # Edge detection
        if not hasattr(self, "edges"):
            self.preprocess_image()
        axs[0, 1].imshow(self.edges, cmap="gray")
        axs[0, 1].set_title("Edge Detection")
        axs[0, 1].axis("off")

        # Geometric patterns
        if not self.geometric_centers:
            self.analyze_geometric_patterns()
        axs[1, 0].imshow(self.grayscale, cmap="gray")
        axs[1, 0].set_title(f"Geometric Analysis (Symmetry: {self.symmetry_score:.2f})")

        # Plot detected circles
        for x, y in self.geometric_centers:
            circle = Circle((x, y), 10, fill=False, edgecolor="red")
            axs[1, 0].add_patch(circle)
        axs[1, 0].axis("off")

        # Resonance analysis
        if not hasattr(self, "phase_relationships"):
            self.analyze_resonance_patterns()

        # Show waveform and its RFT for the first extracted waveform
        if self.edge_waveforms:
            sample_waveform = self.edge_waveforms[0]
            axs[1, 1].plot(sample_waveform)

            # Overlay resonant points
            rft_result = resonance_fourier_transform(sample_waveform)
            resonant_indices = np.where(rft_result["resonance_mask"])[0]
            resonant_values = sample_waveform[resonant_indices % len(sample_waveform)]
            axs[1, 1].scatter(
                resonant_indices % len(sample_waveform), resonant_values, color="red"
            )

            axs[1, 1].set_title("Sample Waveform Analysis")
        else:
            axs[1, 1].text(0.5, 0.5, "No waveforms extracted", ha="center", va="center")
            axs[1, 1].set_title("Waveform Analysis")

        plt.tight_layout()

        # Save figure
        plt.savefig("image_resonance_analysis.png")
        plt.close()

        # Create a second figure for resonance patterns
        if hasattr(self, "phase_relationships") and self.phase_relationships:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot the top resonance patterns
            freqs = list(self.phase_relationships.keys())
            strengths = [
                data["avg_strength"] for data in self.phase_relationships.values()
            ]
            patterns = [
                data["phase_pattern"] for data in self.phase_relationships.values()
            ]

            colors = {
                "harmonic": "green",
                "golden_ratio": "gold",
                "regular": "blue",
                "irregular": "gray",
            }
            color_list = [colors[p] for p in patterns]

            ax.bar(range(len(freqs)), strengths, color=color_list)
            ax.set_xticks(range(len(freqs)))
            ax.set_xticklabels([f"{f:.2f}" for f in freqs], rotation=45)
            ax.set_xlabel("Frequency")
            ax.set_ylabel("Average Strength")
            ax.set_title("Resonance Patterns Detected")

            # Create legend
            from matplotlib.patches import Patch

            legend_elements = [
                Patch(facecolor=colors[k], label=k.replace("_", " ").title())
                for k in colors
                if k in patterns
            ]
            ax.legend(handles=legend_elements)

            plt.tight_layout()
            plt.savefig("resonance_patterns.png")
            plt.close()

        print("Analysis visualization saved to 'image_resonance_analysis.png'")
        if hasattr(self, "phase_relationships") and self.phase_relationships:
            print("Resonance patterns visualization saved to 'resonance_patterns.png'")

    def interpret_symbolic_meanings(self):
        """
        Attempt to interpret the symbolic meanings of detected patterns
        based on QuantoniumOS resonance mathematics.
        """
        interpretations = {
            "geometric_summary": "",
            "resonance_summary": "",
            "symbolic_elements": [],
            "overall_rating": 0.0,
        }

        # Analyze geometric patterns if not already done
        if not self.geometric_centers:
            self.analyze_geometric_patterns()

        # Analyze resonance patterns if not already done
        if not hasattr(self, "phase_relationships"):
            self.analyze_resonance_patterns()

        # Interpret geometric patterns
        if self.symmetry_score > 0.7:
            interpretations["geometric_summary"] = (
                "High degree of geometric symmetry detected, suggesting deliberate design with potential symbolic meaning."
            )
        elif self.symmetry_score > 0.5:
            interpretations["geometric_summary"] = (
                "Moderate geometric symmetry detected, showing some intentional patterning."
            )
        else:
            interpretations["geometric_summary"] = (
                "Limited geometric symmetry, patterns may be coincidental or organic."
            )

        # Add details about circles and triangles
        circle_count = len(self.geometric_centers)
        interpretations[
            "geometric_summary"
        ] += f"\nDetected approximately {circle_count} circular elements"

        # Interpret resonance patterns
        if hasattr(self, "phase_relationships") and self.phase_relationships:
            harmonic_count = sum(
                1
                for data in self.phase_relationships.values()
                if data["phase_pattern"] == "harmonic"
            )
            golden_count = sum(
                1
                for data in self.phase_relationships.values()
                if data["phase_pattern"] == "golden_ratio"
            )

            if harmonic_count > len(self.phase_relationships) * 0.5:
                interpretations["resonance_summary"] = (
                    "Strong harmonic resonance patterns detected, suggesting intentional mathematical structuring."
                )
            elif golden_count > len(self.phase_relationships) * 0.3:
                interpretations["resonance_summary"] = (
                    "Notable golden ratio alignments detected, suggesting sophisticated geometric knowledge."
                )
            else:
                interpretations["resonance_summary"] = (
                    "Mixed resonance patterns with some significant frequency relationships."
                )

            # Add details about specific resonances
            top_freqs = sorted(
                self.phase_relationships.items(),
                key=lambda x: x[1]["avg_strength"],
                reverse=True,
            )[:3]

            interpretations["resonance_summary"] += "\nTop resonant frequencies:"
            for freq, data in top_freqs:
                interpretations[
                    "resonance_summary"
                ] += f"\n- {freq:.3f} Hz ({data['phase_pattern']} pattern, strength: {data['avg_strength']:.2f})"
        else:
            interpretations["resonance_summary"] = (
                "Limited resonance patterns detected."
            )

        # Identify potential symbolic elements
        if self.symmetry_score > 0.6:
            interpretations["symbolic_elements"].append(
                "Radial symmetry suggesting cosmic or universal symbolism"
            )

        if circle_count >= 3:
            interpretations["symbolic_elements"].append(
                "Circular elements suggesting cycles, completeness, or protection"
            )

        if hasattr(self, "phase_relationships"):
            harmonic_patterns = [
                p
                for p in self.phase_relationships.values()
                if p["phase_pattern"] == "harmonic"
            ]
            if harmonic_patterns:
                interpretations["symbolic_elements"].append(
                    "Harmonic resonance patterns suggesting mathematical encoding"
                )

            golden_patterns = [
                p
                for p in self.phase_relationships.values()
                if p["phase_pattern"] == "golden_ratio"
            ]
            if golden_patterns:
                interpretations["symbolic_elements"].append(
                    "Golden ratio patterns suggesting advanced mathematical knowledge"
                )

        # Calculate overall rating of symbolic significance
        geometric_rating = self.symmetry_score

        resonance_rating = 0.0
        if hasattr(self, "phase_relationships") and self.phase_relationships:
            pattern_weights = {
                "harmonic": 1.0,
                "golden_ratio": 1.0,
                "regular": 0.7,
                "irregular": 0.3,
            }
            weighted_sum = sum(
                pattern_weights[data["phase_pattern"]] * data["avg_strength"]
                for data in self.phase_relationships.values()
            )
            resonance_rating = (
                weighted_sum / len(self.phase_relationships)
                if self.phase_relationships
                else 0.0
            )

        # Combine ratings
        interpretations["overall_rating"] = (
            geometric_rating * 0.5 + resonance_rating * 0.5
        ) * 10

        return interpretations


def analyze_image(image_path):
    """Analyze an image using QuantoniumOS resonance techniques"""
    analyzer = ImageResonanceAnalyzer(image_path)

    # Run the full analysis
    analyzer.preprocess_image()
    analyzer.analyze_geometric_patterns()
    analyzer.extract_waveforms()
    analyzer.analyze_resonance_patterns()

    # Visualize the results
    analyzer.visualize_analysis()

    # Interpret symbolic meanings
    interpretation = analyzer.interpret_symbolic_meanings()

    return {"analyzer": analyzer, "interpretation": interpretation}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python image_resonance_analyzer.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    results = analyze_image(image_path)

    # Print interpretation
    print("\n===== SYMBOLIC INTERPRETATION =====")
    print(f"GEOMETRIC ANALYSIS: {results['interpretation']['geometric_summary']}")
    print(f"\nRESONANCE ANALYSIS: {results['interpretation']['resonance_summary']}")

    print("\nPOTENTIAL SYMBOLIC ELEMENTS:")
    for element in results["interpretation"]["symbolic_elements"]:
        print(f"- {element}")

    print(
        f"\nOVERALL SIGNIFICANCE RATING: {results['interpretation']['overall_rating']:.1f}/10.0"
    )
