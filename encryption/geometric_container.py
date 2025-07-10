"""
QuantoniumOS - Geometric Container

This module implements a geometric container for spatial hash mapping
and waveform analysis, providing a quantum-inspired approach to data
organization.
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


class GeometricContainer:
    """
    Geometric container for spatial hash mapping.

    Attributes:
        size: Size of the container
        points: List of points in the container
        distance_matrix: Matrix of distances between points
    """

    def __init__(self, size: int) -> None:
        """
        Initialize a new geometric container.

        Args:
            size: Size of the container (will be a square)
        """
        self.size = size
        self.points = []
        self.distance_matrix = None

    def add_point(self, x: float, y: float) -> None:
        """
        Add a point to the container.

        Args:
            x: X coordinate
            y: Y coordinate
        """
        self.points.append((x, y))

        # Clear the cached distance matrix
        self.distance_matrix = None

    def remove_point(self, x: float, y: float) -> bool:
        """
        Remove a point from the container.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            True if the point was removed, False otherwise
        """
        try:
            self.points.remove((x, y))
            self.distance_matrix = None
            return True
        except ValueError:
            return False

    def clear(self) -> None:
        """Clear all points from the container."""
        self.points = []
        self.distance_matrix = None

    def calculate_distance_matrix(self) -> np.ndarray:
        """
        Calculate the distance matrix between all points.

        Returns:
            Distance matrix as numpy array
        """
        n = len(self.points)
        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                # Calculate Euclidean distance
                x1, y1 = self.points[i]
                x2, y2 = self.points[j]

                dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                matrix[i, j] = dist
                matrix[j, i] = dist

        self.distance_matrix = matrix
        return matrix

    def get_nearest_neighbors(self, point_index: int, k: int) -> List[int]:
        """
        Get the k nearest neighbors of a point.

        Args:
            point_index: Index of the point
            k: Number of neighbors to return

        Returns:
            List of indices of the nearest neighbors
        """
        if self.distance_matrix is None:
            self.calculate_distance_matrix()

        distances = self.distance_matrix[point_index]

        # Sort indices by distance (excluding self)
        sorted_indices = np.argsort(distances)

        # Remove self (distance 0)
        nearest = [idx for idx in sorted_indices if idx != point_index]

        # Return up to k neighbors
        return nearest[:k]

    def calculate_density(self, sigma: float = 1.0) -> List[float]:
        """
        Calculate the density of each point using a Gaussian kernel.

        Args:
            sigma: Bandwidth parameter for the Gaussian kernel

        Returns:
            List of density values for each point
        """
        if self.distance_matrix is None:
            self.calculate_distance_matrix()

        n = len(self.points)
        densities = np.zeros(n)

        for i in range(n):
            # Sum of Gaussian similarities
            for j in range(n):
                if i != j:
                    d = self.distance_matrix[i, j]
                    densities[i] += math.exp(-(d**2) / (2 * sigma**2))

        return densities.tolist()

    def generate_waveform(self, num_samples: int = 64) -> List[float]:
        """
        Generate a waveform based on the container's geometry.

        Args:
            num_samples: Number of samples in the waveform

        Returns:
            List of waveform values
        """
        if not self.points:
            return [0.0] * num_samples

        # Calculate densities as basis for the waveform
        densities = self.calculate_density()

        # Map densities to time domain
        waveform = [0.0] * num_samples

        for i, density in enumerate(densities):
            # Distribute the influence of each point across the waveform
            x, y = self.points[i]

            # Normalize coordinates
            x_norm = x / self.size
            y_norm = y / self.size

            # Position in the waveform
            center = int(x_norm * num_samples)

            # Width of the influence
            width = max(1, int(y_norm * num_samples / 8))

            # Add influence with Gaussian weighting
            for j in range(num_samples):
                distance = abs(j - center)
                influence = density * math.exp(-(distance**2) / (2 * width**2))
                waveform[j] += influence

        # Normalize the waveform
        max_val = max(waveform) if max(waveform) > 0 else 1.0
        return [w / max_val for w in waveform]

    def get_stats(self) -> Dict[str, Union[int, float]]:
        """
        Get statistics about the container.

        Returns:
            Dictionary of statistics
        """
        if not self.points:
            return {
                "count": 0,
                "avg_distance": 0.0,
                "max_distance": 0.0,
                "area": 0.0,
                "density": 0.0,
            }

        if self.distance_matrix is None:
            self.calculate_distance_matrix()

        n = len(self.points)

        # Calculate average and max distance
        distances = self.distance_matrix.flatten()
        avg_distance = sum(distances) / (n * n - n)  # Exclude zeros on diagonal
        max_distance = max(distances)

        # Calculate approximate area using convex hull
        if n >= 3:
            # Simple approximation based on bounding box
            x_vals = [p[0] for p in self.points]
            y_vals = [p[1] for p in self.points]

            area = (max(x_vals) - min(x_vals)) * (max(y_vals) - min(y_vals))
        else:
            area = 0.0

        # Calculate density
        density = n / (self.size * self.size)

        return {
            "count": n,
            "avg_distance": avg_distance,
            "max_distance": max_distance,
            "area": area,
            "density": density,
        }

    def serialize(self) -> List[List[float]]:
        """
        Serialize the container to a list of points.

        Returns:
            List of points as [x, y] pairs
        """
        return [[x, y] for x, y in self.points]

    @classmethod
    def deserialize(cls, data: List[List[float]], size: int) -> "GeometricContainer":
        """
        Deserialize a container from a list of points.

        Args:
            data: List of points as [x, y] pairs
            size: Size of the container

        Returns:
            New GeometricContainer
        """
        container = cls(size)

        for point in data:
            if len(point) == 2:
                container.add_point(point[0], point[1])

        return container
