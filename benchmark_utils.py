#!/usr/bin/env python3
""""""
Shared utilities for RFT benchmark suite
""""""

import numpy as np
import json
import sys
import os
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corrected_rft_quantum import HighPerformanceRFTQuantum
from minimal_true_rft import MinimalTrueRFT

class BenchmarkUtils:
    """"""Shared utilities for all benchmarks""""""

    @staticmethod
    def save_results(results: Dict[str, Any], filename: str) -> None:
        """"""Save benchmark results to JSON file""""""
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj

        with open(filename, 'w') as f:
            json.dump(convert_numpy(results), f, indent=2)

    @staticmethod
    def load_results(filename: str) -> Dict[str, Any]:
        """"""Load benchmark results from JSON file""""""
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    @staticmethod
    def create_rft_crypto() -> MinimalTrueRFT:
        """"""Create standard RFT crypto configuration""""""
        return MinimalTrueRFT(
            weights=[0.7, 0.2, 0.1],
            theta0_values=[0.0, np.pi/4, np.pi/2],
            omega_values=[1.0, 1.618, 2.618],  # Golden ratio harmonics
            sigma0=1.5,
            gamma=0.3
        )

    @staticmethod
    def create_rft_quantum(num_qubits: int = 4) -> HighPerformanceRFTQuantum:
        """"""Create standard RFT quantum configuration""""""
        return HighPerformanceRFTQuantum(num_qubits=num_qubits)

    @staticmethod
    def create_rft_optimizer() -> MinimalTrueRFT:
        """"""Create RFT configuration for optimization""""""
        return MinimalTrueRFT(
            weights=[0.6, 0.3, 0.1],
            theta0_values=[0.0, np.pi/3, 2*np.pi/3],
            omega_values=[1.0, 1.618, 2.618],
            sigma0=2.0,
            gamma=0.2
        )

    @staticmethod
    def create_rft_pattern_analyzer() -> MinimalTrueRFT:
        """"""Create RFT configuration for pattern recognition""""""
        phi = (1 + np.sqrt(5)) / 2
        return MinimalTrueRFT(
            weights=[0.5, 0.3, 0.2],
            theta0_values=[0.0, np.pi/4, np.pi/2],
            omega_values=[1.0, phi, phi**2],
            sigma0=1.8,
            gamma=0.15
        )

    @staticmethod
    def print_benchmark_header(title: str, emoji: str = "🔬") -> None:
        """"""Print standardized benchmark header""""""
        print(f"{emoji} BENCHMARK: {title}")
        print("=" * 50)

    @staticmethod
    def print_results_table(headers: list, rows: list) -> None:
        """"""Print formatted results table""""""
        # Calculate column widths
        widths = [len(str(header)) for header in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(str(cell)))

        # Print header
        header_str = " | ".join(f"{headers[i]:<{widths[i]}}" for i in range(len(headers)))
        print(header_str)
        print("-" * len(header_str))

        # Print rows
        for row in rows:
            row_str = " | ".join(f"{str(row[i]):<{widths[i]}}" for i in range(len(row)))
            print(row_str)

class ConfigurableBenchmark:
    """"""Base class for configurable benchmarks""""""

    def __init__(self, config: Dict[str, Any] = None):
        """"""Initialize with optional configuration""""""
        self.config = config or {}
        self.rng = np.random.RandomState(self.config.get('random_seed', 42))
        self.results = {}

    def get_param(self, key: str, default: Any) -> Any:
        """"""Get parameter with fallback to default""""""
        return self.config.get(key, default)

    def scale_for_environment(self, base_value: int, scale_factor: str = 'medium') -> int:
        """"""Scale parameters based on environment""""""
        scale_factors = {
            'small': 0.1,
            'medium': 0.5,
            'large': 1.0,
            'xlarge': 2.0
        }
        factor = scale_factors.get(scale_factor, 1.0)
        return max(1, int(base_value * factor))
