"""
Resonance Analyzer Adapter Module

This module provides a bridge between the web platform and the Resonance Analyzer system,
allowing the web interface to interact with image analysis components while
maintaining proper isolation and security.
"""

import base64
import json
import logging
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("resonance_analyzer_adapter")

# Find the attached_assets directory
ASSETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "attached_assets",
)
sys.path.append(ASSETS_DIR)

# Import necessary components from the original resonance analyzer system
try:
    # This import includes numpy for arrays
    import numpy as np

    # Try to import the actual resonance analyzer components
    try:
        from image_resonance_analyzer import (ImageResonanceAnalyzer,
                                              analyze_image)

        ANALYZER_AVAILABLE = True
        logger.info("Successfully imported Image Resonance Analyzer components")
    except ImportError as e:
        logger.warning(f"Could not import Image Resonance Analyzer: {str(e)}")
        ANALYZER_AVAILABLE = False

        # Define a simple fallback implementation for testing
        class ImageResonanceAnalyzer:
            def __init__(self, image_path):
                self.image_path = image_path
                logger.warning(
                    f"Using fallback ImageResonanceAnalyzer with image: {image_path}"
                )

            def preprocess_image(self):
                return {"status": "Image preprocessing simulated (fallback)"}

            def analyze_geometric_patterns(self):
                return {
                    "circles": 2,
                    "triangles": 3,
                    "rectangles": 1,
                    "symmetric_axes": 1,
                    "fallback": True,
                }

            def extract_waveforms(self):
                # Generate some random waveforms for testing
                return {
                    "horizontal": np.random.random(32).tolist(),
                    "vertical": np.random.random(32).tolist(),
                    "diagonal": np.random.random(32).tolist(),
                    "fallback": True,
                }

            def analyze_resonance_patterns(self):
                return {
                    "resonance_score": 0.75,
                    "dominant_frequencies": [0.1, 0.3, 0.5],
                    "phase_coherence": 0.65,
                    "fallback": True,
                }

            def visualize_analysis(self):
                # This would normally return an image, but we just return a message for the fallback
                return "Visualization generated (fallback)"

            def interpret_symbolic_meanings(self):
                return {
                    "symbolic_score": 0.82,
                    "potential_meanings": [
                        "Structural balance",
                        "Harmonic relationships",
                        "Temporal coherence",
                    ],
                    "confidence": 0.7,
                    "fallback": True,
                }

            def _calculate_symmetry(self):
                return 0.8

            def _analyze_cross_waveform_patterns(self, resonance_data):
                return {
                    "cross_correlation": 0.6,
                    "phase_alignment": 0.7,
                    "fallback": True,
                }

            def _check_phase_pattern(self, phase_diffs):
                return {
                    "regular_pattern": True,
                    "pattern_type": "linear",
                    "fallback": True,
                }

        def analyze_image(image_path):
            """Analyze an image using the fallback implementation"""
            analyzer = ImageResonanceAnalyzer(image_path)

            # Run the various analysis steps
            preprocessing = analyzer.preprocess_image()
            geometric = analyzer.analyze_geometric_patterns()
            waveforms = analyzer.extract_waveforms()
            resonance = analyzer.analyze_resonance_patterns()
            symbolic = analyzer.interpret_symbolic_meanings()

            # Combine the results
            return {
                "image_path": image_path,
                "geometric_patterns": geometric,
                "waveforms": waveforms,
                "resonance_patterns": resonance,
                "symbolic_interpretation": symbolic,
                "fallback": True,
            }

except ImportError as e:
    logger.error(f"Could not import numpy: {str(e)}")
    ANALYZER_AVAILABLE = False

    # Define minimal fallback if even numpy is not available
    class ImageResonanceAnalyzer:
        def __init__(self, image_path):
            self.image_path = image_path
            logger.warning(
                f"Using minimal fallback ImageResonanceAnalyzer with image: {image_path}"
            )

        def preprocess_image(self):
            return {"status": "Image preprocessing simulated (minimal fallback)"}

        def analyze_geometric_patterns(self):
            return {"fallback": True, "minimal": True}

        def extract_waveforms(self):
            return {"fallback": True, "minimal": True}

        def analyze_resonance_patterns(self):
            return {"fallback": True, "minimal": True}

        def visualize_analysis(self):
            return "Visualization not available (minimal fallback)"

        def interpret_symbolic_meanings(self):
            return {"fallback": True, "minimal": True}

    def analyze_image(image_path):
        """Minimal fallback for analyze_image"""
        return {"fallback": True, "minimal": True}


class ResonanceAnalyzerAdapter:
    """
    Adapter class for interacting with the Resonance Analyzer from the web platform.
    Provides methods for analyzing images using resonance techniques with proper isolation.
    """

    def __init__(self):
        """Initialize the Resonance Analyzer Adapter."""
        self.temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
        os.makedirs(self.temp_dir, exist_ok=True)
        logger.info(
            f"Initialized Resonance Analyzer Adapter (temp dir: {self.temp_dir})"
        )

    def _save_base64_image(self, base64_data: str, file_name: str = None) -> str:
        """
        Save base64-encoded image data to a file.

        Args:
            base64_data: Base64-encoded image data
            file_name: Optional file name (without path)

        Returns:
            Path to the saved image file
        """
        try:
            # Strip the base64 prefix if present
            if "," in base64_data:
                base64_data = base64_data.split(",", 1)[1]

            # Decode the base64 data
            image_data = base64.b64decode(base64_data)

            # Generate a file name if not provided
            if not file_name:
                file_name = f"image_{int(time.time())}.png"

            # Save the image to a file
            file_path = os.path.join(self.temp_dir, file_name)
            with open(file_path, "wb") as f:
                f.write(image_data)

            logger.info(f"Saved base64 image to {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"Error saving base64 image: {str(e)}")
            raise ValueError(f"Invalid base64 image data: {str(e)}")

    def analyze_image_file(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze an image file using resonance techniques.

        Args:
            image_path: Path to the image file

        Returns:
            Dict with analysis results
        """
        try:
            # Check if the file exists
            if not os.path.exists(image_path):
                return {
                    "success": False,
                    "error": f"Image file not found: {image_path}",
                }

            # Run the analysis
            logger.info(f"Analyzing image: {image_path}")
            results = analyze_image(image_path)

            # Convert numpy arrays to lists for JSON serialization
            return {
                "success": True,
                "image_path": image_path,
                "results": self._serialize_results(results),
            }

        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            return {"success": False, "error": str(e)}

    def analyze_base64_image(self, base64_data: str) -> Dict[str, Any]:
        """
        Analyze a base64-encoded image using resonance techniques.

        Args:
            base64_data: Base64-encoded image data

        Returns:
            Dict with analysis results
        """
        try:
            # Save the base64 data to a file
            image_path = self._save_base64_image(base64_data)

            # Analyze the saved image
            return self.analyze_image_file(image_path)

        except Exception as e:
            logger.error(f"Error analyzing base64 image: {str(e)}")
            return {"success": False, "error": str(e)}

    def _serialize_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert analysis results to a JSON-serializable format.

        Args:
            results: Analysis results

        Returns:
            JSON-serializable version of the results
        """
        serialized = {}

        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serialized[key] = value.tolist()
            elif isinstance(value, dict):
                serialized[key] = self._serialize_results(value)
            elif isinstance(value, list):
                serialized[key] = [
                    item.tolist() if isinstance(item, np.ndarray) else item
                    for item in value
                ]
            else:
                serialized[key] = value

        return serialized

    def cleanup_temp_files(self):
        """Clean up temporary image files."""
        try:
            for file_name in os.listdir(self.temp_dir):
                file_path = os.path.join(self.temp_dir, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            logger.info(f"Cleaned up temporary files in {self.temp_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up temporary files: {str(e)}")


# Singleton instance for use in API endpoints
_adapter_instance = None


def get_adapter() -> ResonanceAnalyzerAdapter:
    """
    Get the singleton adapter instance, creating it if necessary.

    Returns:
        ResonanceAnalyzerAdapter instance
    """
    global _adapter_instance

    if _adapter_instance is None:
        _adapter_instance = ResonanceAnalyzerAdapter()

    return _adapter_instance


# Test the adapter if run directly
if __name__ == "__main__":
    import time

    adapter = ResonanceAnalyzerAdapter()

    # Create a simple test image for analysis
    try:
        import numpy as np
        from PIL import Image, ImageDraw

        # Create a test image
        img = Image.new("RGB", (200, 200), color="black")
        draw = ImageDraw.Draw(img)

        # Draw a circle
        draw.ellipse((50, 50, 150, 150), fill="white")

        # Save the image
        test_image_path = os.path.join(adapter.temp_dir, "test_circle.png")
        img.save(test_image_path)

        # Analyze the image
        print("Analyzing test image...")
        results = adapter.analyze_image_file(test_image_path)
        print(f"Analysis results: {json.dumps(results, indent=2)}")

        # Convert to base64 and test base64 analysis
        with open(test_image_path, "rb") as f:
            base64_data = base64.b64encode(f.read()).decode("utf-8")

        print("Analyzing base64 image...")
        results = adapter.analyze_base64_image(base64_data)
        print(f"Base64 analysis results: {json.dumps(results, indent=2)}")

    except ImportError:
        print("PIL library not available for generating test images")

        # Use fallback
        test_image_path = os.path.join(adapter.temp_dir, "test_fallback.txt")
        with open(test_image_path, "w") as f:
            f.write("This is not an image, but we'll test the fallback functionality")

        results = adapter.analyze_image_file(test_image_path)
        print(f"Fallback analysis results: {json.dumps(results, indent=2)}")

    # Clean up
    adapter.cleanup_temp_files()
