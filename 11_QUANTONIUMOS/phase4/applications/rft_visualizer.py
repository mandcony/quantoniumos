"""
QuantoniumOS Phase 4: RFT Transform Visualizer
Advanced visualization tool for Reverse Fourier Transform analysis
"""

import json
import logging
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# 3D visualization
try:
    from mpl_toolkits.mplot3d import Axes3D

    HAS_3D = True
except ImportError:
    HAS_3D = False


class RFTTransformVisualizer:
    """
    Advanced RFT Transform Visualizer with real-time analysis
    """

    def __init__(self, parent=None):
        self.logger = logging.getLogger(__name__)

        # Create main window
        if parent:
            self.root = tk.Toplevel(parent)
        else:
            self.root = tk.Tk()

        self.root.title("QuantoniumOS - RFT Transform Visualizer")
        self.root.geometry("1400x900")
        self.root.configure(bg="#1e1e1e")

        # Data storage
        self.input_data = None
        self.rft_result = None
        self.analysis_results = {}
        self.visualization_params = {
            "show_magnitude": True,
            "show_phase": True,
            "show_real": False,
            "show_imaginary": False,
            "colormap": "viridis",
            "log_scale": False,
            "normalize": True,
        }

        # Real-time update
        self.auto_update = False
        self.update_interval = 100  # ms
        self.last_update = time.time()

        self.setup_ui()
        self.setup_plots()

        # Start update loop
        self.update_loop()

        self.logger.info("RFT Transform Visualizer initialized")

    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="RFT Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # Data input controls
        input_frame = ttk.Frame(control_frame)
        input_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(input_frame, text="Data Input:").pack(side=tk.LEFT)
        ttk.Button(input_frame, text="Load File", command=self.load_data_file).pack(
            side=tk.LEFT, padx=(10, 5)
        )
        ttk.Button(
            input_frame, text="Generate Synthetic", command=self.generate_synthetic_data
        ).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(input_frame, text="Clear", command=self.clear_data).pack(
            side=tk.LEFT, padx=(0, 10)
        )

        # Transform controls
        transform_frame = ttk.Frame(control_frame)
        transform_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(transform_frame, text="Transform:").pack(side=tk.LEFT)
        ttk.Button(transform_frame, text="Compute RFT", command=self.compute_rft).pack(
            side=tk.LEFT, padx=(10, 5)
        )
        ttk.Button(
            transform_frame, text="Inverse RFT", command=self.compute_inverse_rft
        ).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(
            transform_frame, text="Compare FFT", command=self.compare_with_fft
        ).pack(side=tk.LEFT, padx=(0, 10))

        # Visualization controls
        viz_frame = ttk.Frame(control_frame)
        viz_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(viz_frame, text="Display:").pack(side=tk.LEFT)

        self.show_magnitude_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            viz_frame,
            text="Magnitude",
            variable=self.show_magnitude_var,
            command=self.update_visualization,
        ).pack(side=tk.LEFT, padx=(10, 5))

        self.show_phase_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            viz_frame,
            text="Phase",
            variable=self.show_phase_var,
            command=self.update_visualization,
        ).pack(side=tk.LEFT, padx=(0, 5))

        self.show_real_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            viz_frame,
            text="Real",
            variable=self.show_real_var,
            command=self.update_visualization,
        ).pack(side=tk.LEFT, padx=(0, 5))

        self.show_imaginary_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            viz_frame,
            text="Imaginary",
            variable=self.show_imaginary_var,
            command=self.update_visualization,
        ).pack(side=tk.LEFT, padx=(0, 10))

        # Scale controls
        scale_frame = ttk.Frame(control_frame)
        scale_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(scale_frame, text="Scale:").pack(side=tk.LEFT)

        self.log_scale_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            scale_frame,
            text="Log Scale",
            variable=self.log_scale_var,
            command=self.update_visualization,
        ).pack(side=tk.LEFT, padx=(10, 5))

        self.normalize_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            scale_frame,
            text="Normalize",
            variable=self.normalize_var,
            command=self.update_visualization,
        ).pack(side=tk.LEFT, padx=(0, 5))

        # Colormap selection
        ttk.Label(scale_frame, text="Colormap:").pack(side=tk.LEFT, padx=(20, 5))
        self.colormap_var = tk.StringVar(value="viridis")
        colormap_combo = ttk.Combobox(
            scale_frame,
            textvariable=self.colormap_var,
            values=["viridis", "plasma", "inferno", "magma", "jet", "rainbow"],
            state="readonly",
            width=10,
        )
        colormap_combo.pack(side=tk.LEFT, padx=(0, 10))
        colormap_combo.bind(
            "<<ComboboxSelected>>", lambda e: self.update_visualization()
        )

        # Real-time controls
        realtime_frame = ttk.Frame(control_frame)
        realtime_frame.pack(fill=tk.X)

        self.auto_update_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            realtime_frame,
            text="Auto Update",
            variable=self.auto_update_var,
            command=self.toggle_auto_update,
        ).pack(side=tk.LEFT)

        ttk.Label(realtime_frame, text="Interval (ms):").pack(
            side=tk.LEFT, padx=(20, 5)
        )
        self.interval_var = tk.StringVar(value="100")
        interval_entry = ttk.Entry(
            realtime_frame, textvariable=self.interval_var, width=8
        )
        interval_entry.pack(side=tk.LEFT, padx=(0, 10))
        interval_entry.bind("<Return>", self.update_interval_setting)

        # Plot area
        self.plot_frame = ttk.Frame(main_frame)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(
            main_frame, textvariable=self.status_var, relief=tk.SUNKEN
        )
        status_bar.pack(fill=tk.X, pady=(10, 0))

    def setup_plots(self):
        """Setup matplotlib plots"""
        # Create figure with subplots
        self.fig = Figure(figsize=(14, 8), facecolor="#2e2e2e")
        self.fig.patch.set_facecolor("#2e2e2e")

        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initialize subplots
        self.setup_subplots()

    def setup_subplots(self):
        """Setup subplot layout"""
        self.fig.clear()

        # 2x3 grid layout
        self.ax_input = self.fig.add_subplot(2, 3, 1)
        self.ax_magnitude = self.fig.add_subplot(2, 3, 2)
        self.ax_phase = self.fig.add_subplot(2, 3, 3)
        self.ax_real = self.fig.add_subplot(2, 3, 4)
        self.ax_imaginary = self.fig.add_subplot(2, 3, 5)
        self.ax_3d = self.fig.add_subplot(2, 3, 6, projection="3d" if HAS_3D else None)

        # Style subplots
        for ax in [
            self.ax_input,
            self.ax_magnitude,
            self.ax_phase,
            self.ax_real,
            self.ax_imaginary,
        ]:
            ax.set_facecolor("#3e3e3e")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")

        if HAS_3D:
            self.ax_3d.set_facecolor("#3e3e3e")
            self.ax_3d.tick_params(colors="white")
            self.ax_3d.xaxis.label.set_color("white")
            self.ax_3d.yaxis.label.set_color("white")
            self.ax_3d.zaxis.label.set_color("white")
            self.ax_3d.title.set_color("white")

        # Set titles
        self.ax_input.set_title("Input Data")
        self.ax_magnitude.set_title("RFT Magnitude")
        self.ax_phase.set_title("RFT Phase")
        self.ax_real.set_title("RFT Real Part")
        self.ax_imaginary.set_title("RFT Imaginary Part")
        self.ax_3d.set_title("3D Visualization")

        self.fig.tight_layout()
        self.canvas.draw()

    def load_data_file(self):
        """Load data from file"""
        try:
            file_path = filedialog.askopenfilename(
                title="Load Data File",
                filetypes=[
                    ("JSON files", "*.json"),
                    ("CSV files", "*.csv"),
                    ("Text files", "*.txt"),
                    ("All files", "*.*"),
                ],
            )

            if not file_path:
                return

            if file_path.endswith(".json"):
                with open(file_path, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "data" in data:
                    self.input_data = np.array(data["data"])
                else:
                    self.input_data = np.array(data)
            elif file_path.endswith(".csv"):
                self.input_data = np.loadtxt(file_path, delimiter=",")
            else:
                self.input_data = np.loadtxt(file_path)

            self.status_var.set(f"Loaded data: {self.input_data.shape}")
            self.plot_input_data()
            self.logger.info(f"Loaded data from {file_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {e}")
            self.logger.error(f"Failed to load data: {e}")

    def generate_synthetic_data(self):
        """Generate synthetic test data"""
        try:
            # Create dialog for parameters
            dialog = tk.Toplevel(self.root)
            dialog.title("Generate Synthetic Data")
            dialog.geometry("400x300")
            dialog.configure(bg="#2e2e2e")

            # Parameters
            params = {}

            ttk.Label(dialog, text="Data Type:").pack(pady=5)
            data_type_var = tk.StringVar(value="sine_wave")
            data_type_combo = ttk.Combobox(
                dialog,
                textvariable=data_type_var,
                values=[
                    "sine_wave",
                    "chirp",
                    "gaussian_pulse",
                    "noise",
                    "complex_signal",
                ],
                state="readonly",
            )
            data_type_combo.pack(pady=5)

            ttk.Label(dialog, text="Number of samples:").pack(pady=5)
            samples_var = tk.StringVar(value="1024")
            ttk.Entry(dialog, textvariable=samples_var).pack(pady=5)

            ttk.Label(dialog, text="Frequency (Hz):").pack(pady=5)
            freq_var = tk.StringVar(value="50")
            ttk.Entry(dialog, textvariable=freq_var).pack(pady=5)

            ttk.Label(dialog, text="Sampling Rate (Hz):").pack(pady=5)
            fs_var = tk.StringVar(value="1000")
            ttk.Entry(dialog, textvariable=fs_var).pack(pady=5)

            ttk.Label(dialog, text="Noise Level:").pack(pady=5)
            noise_var = tk.StringVar(value="0.1")
            ttk.Entry(dialog, textvariable=noise_var).pack(pady=5)

            def generate():
                try:
                    data_type = data_type_var.get()
                    n_samples = int(samples_var.get())
                    frequency = float(freq_var.get())
                    fs = float(fs_var.get())
                    noise_level = float(noise_var.get())

                    t = np.linspace(0, n_samples / fs, n_samples)

                    if data_type == "sine_wave":
                        signal = np.sin(2 * np.pi * frequency * t)
                    elif data_type == "chirp":
                        signal = np.sin(2 * np.pi * frequency * t * t)
                    elif data_type == "gaussian_pulse":
                        signal = np.exp(
                            -((t - n_samples / (2 * fs)) ** 2) / (2 * (0.1) ** 2)
                        ) * np.sin(2 * np.pi * frequency * t)
                    elif data_type == "noise":
                        signal = np.random.randn(n_samples)
                    elif data_type == "complex_signal":
                        signal = np.sin(2 * np.pi * frequency * t) + 1j * np.cos(
                            2 * np.pi * frequency * t
                        )
                    else:
                        signal = np.sin(2 * np.pi * frequency * t)

                    # Add noise
                    if noise_level > 0:
                        if np.iscomplexobj(signal):
                            noise = noise_level * (
                                np.random.randn(n_samples)
                                + 1j * np.random.randn(n_samples)
                            )
                        else:
                            noise = noise_level * np.random.randn(n_samples)
                        signal += noise

                    self.input_data = signal
                    self.status_var.set(f"Generated {data_type}: {len(signal)} samples")
                    self.plot_input_data()
                    dialog.destroy()

                except Exception as e:
                    messagebox.showerror("Error", f"Failed to generate data: {e}")

            ttk.Button(dialog, text="Generate", command=generate).pack(pady=20)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to create generator: {e}")

    def clear_data(self):
        """Clear all data and plots"""
        self.input_data = None
        self.rft_result = None
        self.analysis_results = {}
        self.setup_subplots()
        self.status_var.set("Data cleared")

    def compute_rft(self):
        """Compute RFT transform"""
        if self.input_data is None:
            messagebox.showwarning("Warning", "No input data available")
            return

        try:
            self.status_var.set("Computing RFT...")

            # Use numpy FFT as RFT implementation (placeholder)
            # In real implementation, this would call the actual RFT algorithm
            self.rft_result = np.fft.fft(self.input_data)

            # Store analysis results
            self.analysis_results = {
                "magnitude": np.abs(self.rft_result),
                "phase": np.angle(self.rft_result),
                "real": np.real(self.rft_result),
                "imaginary": np.imag(self.rft_result),
                "power": np.abs(self.rft_result) ** 2,
                "energy": np.sum(np.abs(self.rft_result) ** 2),
                "peak_frequency": np.argmax(np.abs(self.rft_result)),
                "computed_at": datetime.now().isoformat(),
            }

            self.update_visualization()
            self.status_var.set(f"RFT computed: {len(self.rft_result)} coefficients")
            self.logger.info("RFT transform computed")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to compute RFT: {e}")
            self.logger.error(f"Failed to compute RFT: {e}")

    def compute_inverse_rft(self):
        """Compute inverse RFT"""
        if self.rft_result is None:
            messagebox.showwarning("Warning", "No RFT result available")
            return

        try:
            self.status_var.set("Computing inverse RFT...")

            # Compute inverse
            inverse_result = np.fft.ifft(self.rft_result)

            # Plot comparison
            self.ax_input.clear()
            self.ax_input.set_facecolor("#3e3e3e")
            self.ax_input.tick_params(colors="white")
            self.ax_input.title.set_color("white")
            self.ax_input.set_title("Original vs Reconstructed")

            if self.input_data is not None:
                self.ax_input.plot(
                    np.real(self.input_data), "b-", label="Original", alpha=0.7
                )
            self.ax_input.plot(
                np.real(inverse_result), "r--", label="Reconstructed", alpha=0.7
            )
            self.ax_input.legend()
            self.ax_input.grid(True, alpha=0.3)

            self.canvas.draw()
            self.status_var.set("Inverse RFT computed")
            self.logger.info("Inverse RFT computed")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to compute inverse RFT: {e}")
            self.logger.error(f"Failed to compute inverse RFT: {e}")

    def compare_with_fft(self):
        """Compare RFT with standard FFT"""
        if self.input_data is None:
            messagebox.showwarning("Warning", "No input data available")
            return

        try:
            self.status_var.set("Comparing with FFT...")

            # Compute FFT
            fft_result = np.fft.fft(self.input_data)

            # Create comparison window
            compare_window = tk.Toplevel(self.root)
            compare_window.title("RFT vs FFT Comparison")
            compare_window.geometry("800x600")

            # Create comparison plots
            fig_compare = Figure(figsize=(8, 6), facecolor="#2e2e2e")
            canvas_compare = FigureCanvasTkAgg(fig_compare, master=compare_window)
            canvas_compare.draw()
            canvas_compare.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Plot comparison
            ax1 = fig_compare.add_subplot(2, 2, 1)
            ax2 = fig_compare.add_subplot(2, 2, 2)
            ax3 = fig_compare.add_subplot(2, 2, 3)
            ax4 = fig_compare.add_subplot(2, 2, 4)

            # Style
            for ax in [ax1, ax2, ax3, ax4]:
                ax.set_facecolor("#3e3e3e")
                ax.tick_params(colors="white")
                ax.title.set_color("white")

            # Magnitude comparison
            ax1.plot(np.abs(self.rft_result), "b-", label="RFT", alpha=0.7)
            ax1.plot(np.abs(fft_result), "r--", label="FFT", alpha=0.7)
            ax1.set_title("Magnitude Comparison")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Phase comparison
            ax2.plot(np.angle(self.rft_result), "b-", label="RFT", alpha=0.7)
            ax2.plot(np.angle(fft_result), "r--", label="FFT", alpha=0.7)
            ax2.set_title("Phase Comparison")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Difference
            diff_mag = np.abs(self.rft_result) - np.abs(fft_result)
            ax3.plot(diff_mag, "g-", alpha=0.7)
            ax3.set_title("Magnitude Difference")
            ax3.grid(True, alpha=0.3)

            # Statistics
            ax4.axis("off")
            stats_text = f"""
Comparison Statistics:
Max Magnitude Diff: {np.max(np.abs(diff_mag)):.2e}
Mean Magnitude Diff: {np.mean(np.abs(diff_mag)):.2e}
RMS Error: {np.sqrt(np.mean(diff_mag**2)):.2e}
Correlation: {np.corrcoef(np.abs(self.rft_result), np.abs(fft_result))[0,1]:.4f}
            """
            ax4.text(
                0.1,
                0.5,
                stats_text,
                transform=ax4.transAxes,
                fontsize=12,
                color="white",
                verticalalignment="center",
            )

            fig_compare.tight_layout()
            canvas_compare.draw()

            self.status_var.set("FFT comparison completed")
            self.logger.info("FFT comparison completed")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to compare with FFT: {e}")
            self.logger.error(f"Failed to compare with FFT: {e}")

    def plot_input_data(self):
        """Plot input data"""
        if self.input_data is None:
            return

        self.ax_input.clear()
        self.ax_input.set_facecolor("#3e3e3e")
        self.ax_input.tick_params(colors="white")
        self.ax_input.title.set_color("white")
        self.ax_input.set_title("Input Data")

        if np.iscomplexobj(self.input_data):
            self.ax_input.plot(np.real(self.input_data), "b-", label="Real", alpha=0.7)
            self.ax_input.plot(
                np.imag(self.input_data), "r-", label="Imaginary", alpha=0.7
            )
            self.ax_input.legend()
        else:
            self.ax_input.plot(self.input_data, "b-", alpha=0.7)

        self.ax_input.grid(True, alpha=0.3)
        self.canvas.draw()

    def update_visualization(self):
        """Update all visualization plots"""
        if self.rft_result is None:
            return

        try:
            # Update visualization parameters
            self.visualization_params.update(
                {
                    "show_magnitude": self.show_magnitude_var.get(),
                    "show_phase": self.show_phase_var.get(),
                    "show_real": self.show_real_var.get(),
                    "show_imaginary": self.show_imaginary_var.get(),
                    "colormap": self.colormap_var.get(),
                    "log_scale": self.log_scale_var.get(),
                    "normalize": self.normalize_var.get(),
                }
            )

            # Clear and redraw plots
            if self.visualization_params["show_magnitude"]:
                self.plot_magnitude()

            if self.visualization_params["show_phase"]:
                self.plot_phase()

            if self.visualization_params["show_real"]:
                self.plot_real()

            if self.visualization_params["show_imaginary"]:
                self.plot_imaginary()

            self.plot_3d_visualization()

            self.canvas.draw()

        except Exception as e:
            self.logger.error(f"Failed to update visualization: {e}")

    def plot_magnitude(self):
        """Plot magnitude spectrum"""
        magnitude = self.analysis_results["magnitude"]

        if self.visualization_params["normalize"]:
            magnitude = magnitude / np.max(magnitude)

        self.ax_magnitude.clear()
        self.ax_magnitude.set_facecolor("#3e3e3e")
        self.ax_magnitude.tick_params(colors="white")
        self.ax_magnitude.title.set_color("white")
        self.ax_magnitude.set_title("RFT Magnitude")

        if self.visualization_params["log_scale"]:
            magnitude = np.log10(magnitude + 1e-10)
            self.ax_magnitude.set_ylabel("Log Magnitude", color="white")
        else:
            self.ax_magnitude.set_ylabel("Magnitude", color="white")

        self.ax_magnitude.plot(magnitude, color="cyan", alpha=0.8)
        self.ax_magnitude.grid(True, alpha=0.3)

    def plot_phase(self):
        """Plot phase spectrum"""
        phase = self.analysis_results["phase"]

        self.ax_phase.clear()
        self.ax_phase.set_facecolor("#3e3e3e")
        self.ax_phase.tick_params(colors="white")
        self.ax_phase.title.set_color("white")
        self.ax_phase.set_title("RFT Phase")
        self.ax_phase.set_ylabel("Phase (radians)", color="white")

        self.ax_phase.plot(phase, color="yellow", alpha=0.8)
        self.ax_phase.grid(True, alpha=0.3)

    def plot_real(self):
        """Plot real part"""
        real_part = self.analysis_results["real"]

        if self.visualization_params["normalize"]:
            real_part = real_part / np.max(np.abs(real_part))

        self.ax_real.clear()
        self.ax_real.set_facecolor("#3e3e3e")
        self.ax_real.tick_params(colors="white")
        self.ax_real.title.set_color("white")
        self.ax_real.set_title("RFT Real Part")
        self.ax_real.set_ylabel("Real", color="white")

        self.ax_real.plot(real_part, color="green", alpha=0.8)
        self.ax_real.grid(True, alpha=0.3)

    def plot_imaginary(self):
        """Plot imaginary part"""
        imag_part = self.analysis_results["imaginary"]

        if self.visualization_params["normalize"]:
            imag_part = imag_part / np.max(np.abs(imag_part))

        self.ax_imaginary.clear()
        self.ax_imaginary.set_facecolor("#3e3e3e")
        self.ax_imaginary.tick_params(colors="white")
        self.ax_imaginary.title.set_color("white")
        self.ax_imaginary.set_title("RFT Imaginary Part")
        self.ax_imaginary.set_ylabel("Imaginary", color="white")

        self.ax_imaginary.plot(imag_part, color="red", alpha=0.8)
        self.ax_imaginary.grid(True, alpha=0.3)

    def plot_3d_visualization(self):
        """Plot 3D visualization"""
        if not HAS_3D:
            return

        self.ax_3d.clear()
        self.ax_3d.set_facecolor("#3e3e3e")
        self.ax_3d.tick_params(colors="white")
        self.ax_3d.title.set_color("white")
        self.ax_3d.set_title("3D Visualization")

        # Create 3D surface plot
        n = len(self.rft_result)
        x = np.arange(n)
        y = np.arange(n)
        X, Y = np.meshgrid(x[: min(50, n)], y[: min(50, n)])

        # Use magnitude for Z values
        magnitude = self.analysis_results["magnitude"][: min(50, n)]
        Z = np.outer(magnitude, magnitude)

        surf = self.ax_3d.plot_surface(
            X, Y, Z, cmap=self.visualization_params["colormap"], alpha=0.8
        )

        self.ax_3d.set_xlabel("Frequency Bin", color="white")
        self.ax_3d.set_ylabel("Frequency Bin", color="white")
        self.ax_3d.set_zlabel("Magnitude", color="white")

    def toggle_auto_update(self):
        """Toggle auto update mode"""
        self.auto_update = self.auto_update_var.get()
        if self.auto_update:
            self.status_var.set("Auto update enabled")
        else:
            self.status_var.set("Auto update disabled")

    def update_interval_setting(self, event=None):
        """Update the update interval"""
        try:
            new_interval = int(self.interval_var.get())
            if new_interval > 0:
                self.update_interval = new_interval
                self.status_var.set(f"Update interval set to {new_interval}ms")
        except ValueError:
            messagebox.showerror("Error", "Invalid interval value")

    def update_loop(self):
        """Main update loop"""
        try:
            if self.auto_update and self.rft_result is not None:
                current_time = time.time()
                if (current_time - self.last_update) * 1000 >= self.update_interval:
                    # Add some random variation to simulate real-time data
                    if self.input_data is not None:
                        noise = 0.01 * np.random.randn(len(self.input_data))
                        self.input_data += noise
                        self.compute_rft()
                    self.last_update = current_time

            # Schedule next update
            self.root.after(50, self.update_loop)

        except Exception as e:
            self.logger.error(f"Update loop error: {e}")
            self.root.after(1000, self.update_loop)  # Retry after 1 second

    def run(self):
        """Run the visualizer"""
        self.root.mainloop()


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    visualizer = RFTTransformVisualizer()
    visualizer.run()
