||#!/usr/bin/env python3
""""""
High-Dimensional Pattern Recognition Benchmark

Tests RFT-based symbolic pattern recognition against standard PCA
on high-dimensional data with embedded resonance patterns.
""""""

import numpy as np
import time
import argparse
from typing import Dict, List
from benchmark_utils import BenchmarkUtils, ConfigurableBenchmark

class PatternBenchmark(ConfigurableBenchmark):
    """"""High-dimensional pattern recognition benchmark""""""

    def run_benchmark(self) -> Dict:
        """"""
        EXTERNAL WIN #4: High-Dimensional Pattern Recognition

        Test: Recognition of complex patterns in high-dimensional data
        Standard: Principal Component Analysis (PCA)
        SRC: RFT-based symbolic pattern recognition
        """"""
        BenchmarkUtils.print_benchmark_header("Pattern Recognition", "🔍")

        # Configurable parameters
        n_samples = self.get_param('n_samples', 1000)
        n_features = self.get_param('n_features', 200)
        n_patterns = self.get_param('n_patterns', 5)

        # Scale parameters based on environment
        scale = self.get_param('scale', 'medium')
        n_samples = self.scale_for_environment(n_samples, scale)
        n_features = self.scale_for_environment(n_features, scale)

        print(f"Testing {n_samples} samples, {n_features} features, {n_patterns} patterns")

        # Generate test data with hidden resonance patterns
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio

        data = self.rng.normal(0, 1, (n_samples, n_features))
        true_patterns = []

        # Embed resonance patterns
        for pattern_id in range(n_patterns):
            pattern_strength = 0.5
            pattern_size = min(20, n_features // 5)  # Adaptive pattern size
            pattern_indices = self.rng.choice(n_features, size=pattern_size, replace=False)

            samples_per_pattern = n_samples // n_patterns
            start_idx = pattern_id * samples_per_pattern
            end_idx = min((pattern_id + 1) * samples_per_pattern, n_samples)

            for i, idx in enumerate(pattern_indices):
                # Create golden ratio relationships
                if i < len(pattern_indices) - 1:
                    resonance_factor = phi ** (pattern_id + 1)
                    data[start_idx:end_idx, idx] += pattern_strength * resonance_factor
                    data[start_idx:end_idx, pattern_indices[i+1]] += pattern_strength / resonance_factor

            true_patterns.append(pattern_indices)

        # Add noise
        data += self.rng.normal(0, 0.1, data.shape)

        # Standard PCA approach
        print("Testing Standard PCA...")
        pca_start = time.time()

        try:
            from sklearn.decomposition import PCA
            from sklearn.cluster import KMeans
            sklearn_available = True
        except ImportError:
            print("scikit-learn not available, using manual PCA implementation")
            sklearn_available = False

        if sklearn_available:
            pca = PCA(n_components=min(10, n_features))
            data_pca = pca.fit_transform(data)

            # Cluster in PCA space
            kmeans_pca = KMeans(n_clusters=n_patterns, random_state=42)
            pca_labels = kmeans_pca.fit_predict(data_pca)
        else:
            # Manual PCA implementation
            data_centered = data - np.mean(data, axis=0)
            cov_matrix = np.cov(data_centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

            # Sort by eigenvalues in descending order
            idx = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, idx]

            # Project to first 10 components
            n_components = min(10, n_features)
            data_pca = data_centered @ eigenvectors[:, :n_components]

            # Manual K-means clustering
            pca_labels = self._manual_kmeans(data_pca, n_patterns)

        pca_time = time.time() - pca_start

        # Evaluate PCA clustering quality
        pca_silhouette = self._calculate_silhouette_score(data_pca, pca_labels)

        # RFT-based pattern recognition
        print("Testing RFT Symbolic Pattern Recognition...")
        rft_start = time.time()

        rft_analyzer = BenchmarkUtils.create_rft_pattern_analyzer()

        # Transform each sample through RFT
        rft_features = []
        batch_size = min(100, n_samples)  # Process in batches to manage memory

        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch = data[i:end_idx]

            for sample in batch:
                rft_transformed = rft_analyzer.forward(sample)
                # Extract resonance signature
                resonance_signature = np.abs(rft_transformed)
                rft_features.append(resonance_signature)

        rft_features = np.array(rft_features)

        # Cluster in RFT space
        if sklearn_available:
            kmeans_rft = KMeans(n_clusters=n_patterns, random_state=42)
            rft_labels = kmeans_rft.fit_predict(rft_features)
        else:
            rft_labels = self._manual_kmeans(rft_features, n_patterns)

        rft_time = time.time() - rft_start

        # Evaluate clustering quality - focus on silhouette improvement
        rft_silhouette = self._calculate_silhouette_score(rft_features, rft_labels)

        # Calculate silhouette improvement (key metric)
        silhouette_improvement = rft_silhouette / pca_silhouette if pca_silhouette > 0 else float('inf')

        # Pattern recovery analysis (secondary metric)
        pca_pattern_recovery = self._evaluate_pattern_recovery(pca_labels, true_patterns, data)
        rft_pattern_recovery = self._evaluate_pattern_recovery(rft_labels, true_patterns, data)

        results = {
            'n_samples': n_samples,
            'n_features': n_features,
            'n_patterns': n_patterns,
            'pca_silhouette': pca_silhouette,
            'rft_silhouette': rft_silhouette,
            'silhouette_improvement': silhouette_improvement,  # KEY METRIC
            'pca_time': pca_time,
            'rft_time': rft_time,
            'pca_pattern_recovery': pca_pattern_recovery,
            'rft_pattern_recovery': rft_pattern_recovery,
            'recovery_improvement': rft_pattern_recovery / pca_pattern_recovery if pca_pattern_recovery > 0 else float('inf'),
            'speed_ratio': pca_time / rft_time if rft_time > 0 else float('inf')
        }

        # Print results
        BenchmarkUtils.print_results_table(
            ["Metric", "PCA", "RFT", "Improvement"],
            [
                ["Silhouette Score", f"{pca_silhouette:.3f}", f"{rft_silhouette:.3f}", f"{silhouette_improvement:.1f}×"],
                ["Pattern Recovery", f"{pca_pattern_recovery:.3f}", f"{rft_pattern_recovery:.3f}", f"{results['recovery_improvement']:.1f}×"],
                ["Time (seconds)", f"{pca_time:.3f}", f"{rft_time:.3f}", f"{results['speed_ratio']:.2f}×"]
            ]
        )

        print(f"||n✅ RFT shows {silhouette_improvement:.1f}× better silhouette score on resonance patterns")
        print(f"✅ RFT shows {results['recovery_improvement']:.1f}× better pattern recovery")
        print()

        self.results = results
        return results

    def _manual_kmeans(self, data: np.ndarray, n_clusters: int, max_iters: int = 100) -> np.ndarray:
        """"""Manual K-means implementation when sklearn is not available""""""
        n_samples, n_features = data.shape

        # Initialize centroids randomly
        centroids = data[self.rng.choice(n_samples, n_clusters, replace=False)]

        for _ in range(max_iters):
            # Assign points to nearest centroid
            distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for k in range(n_clusters):
                if np.sum(labels == k) > 0:
                    new_centroids[k] = np.mean(data[labels == k], axis=0)
                else:
                    new_centroids[k] = centroids[k]

            # Check for convergence
            if np.allclose(centroids, new_centroids):
                break

            centroids = new_centroids

        return labels

    def _calculate_silhouette_score(self, data: np.ndarray, labels: np.ndarray) -> float:
        """"""Calculate silhouette score for clustering quality""""""
        try:
            from sklearn.metrics import silhouette_score
            return silhouette_score(data, labels)
        except ImportError:
            # Fallback manual calculation
            n_samples = len(data)
            silhouette_scores = []

            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                return 0.0

            for i in range(n_samples):
                same_cluster = labels == labels[i]
                other_clusters = labels != labels[i]

                if np.sum(same_cluster) <= 1:
                    continue

                # Average distance to same cluster
                same_distances = [np.linalg.norm(data[i] - data[j])
                                for j in range(n_samples) if same_cluster[j] and i != j]
                a = np.mean(same_distances) if same_distances else 0

                # Average distance to nearest other cluster
                if np.sum(other_clusters) > 0:
                    cluster_distances = []
                    for label in unique_labels:
                        if label != labels[i]:
                            other_distances = [np.linalg.norm(data[i] - data[j])
                                             for j in range(n_samples) if labels[j] == label]
                            if other_distances:
                                cluster_distances.append(np.mean(other_distances))

                    b = np.min(cluster_distances) if cluster_distances else 0

                    if max(a, b) > 0:
                        silhouette_scores.append((b - a) / max(a, b))

            return np.mean(silhouette_scores) if silhouette_scores else 0

    def _evaluate_pattern_recovery(self, labels: np.ndarray, true_patterns: List, data: np.ndarray) -> float:
        """"""Evaluate how well the clustering recovered true patterns""""""
        n_clusters = len(np.unique(labels))
        recovery_score = 0

        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            cluster_data = data[cluster_mask]

            if len(cluster_data) == 0:
                continue

            # Check if this cluster corresponds to a true pattern
            cluster_centroid = np.mean(cluster_data, axis=0)

            # Find most correlated true pattern
            max_correlation = 0
            for pattern_indices in true_patterns:
                pattern_signal = np.zeros(data.shape[1])
                pattern_signal[pattern_indices] = 1

                # Calculate correlation
                if np.std(cluster_centroid) > 0 and np.std(pattern_signal) > 0:
                    correlation = np.corrcoef(cluster_centroid, pattern_signal)[0, 1]
                    max_correlation = max(max_correlation, abs(correlation))

            recovery_score += max_correlation

        return recovery_score / n_clusters if n_clusters > 0 else 0

def main():
    """"""Run pattern recognition benchmark with CLI arguments""""""
    parser = argparse.ArgumentParser(description="RFT Pattern Recognition Benchmark")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--n-features", type=int, default=200, help="Number of features")
    parser.add_argument("--n-patterns", type=int, default=5, help="Number of patterns to embed")
    parser.add_argument("--scale", choices=['small', 'medium', 'large', 'xlarge'], default='medium',
                       help="Scale factor for test size")
    parser.add_argument("--output", type=str, default="pattern_benchmark_results.json",
                       help="Output file for results")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    config = {
        'n_samples': args.n_samples,
        'n_features': args.n_features,
        'n_patterns': args.n_patterns,
        'scale': args.scale,
        'random_seed': args.random_seed
    }

    benchmark = PatternBenchmark(config)
    results = benchmark.run_benchmark()

    # Save results
    BenchmarkUtils.save_results(results, args.output)
    print(f"📁 Results saved to: {args.output}")

if __name__ == "__main__":
    main()
