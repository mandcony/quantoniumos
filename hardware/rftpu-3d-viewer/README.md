# RFTPU-64 Interactive 3D Viewer

This is a high-fidelity, interactive 3D visualization of the QuantoniumOS RFTPU-64 accelerator chip. It is designed to explain the unique "Golden Ratio" architecture to both technical and non-technical audiences.

## Features

- **Interactive Guided Tour**: A step-by-step narrated walkthrough of the chip's internal components.
- **"Why It Matters" Analysis**: Explains technical specs alongside real-world analogies and benefits.
- **Exploded View**: Interactively dissect the chip layers (Heat Spreader, Die, NoC, Substrate).
- **Visual Data Flow**: Animated particles showing the Network-on-Chip (NoC) traffic.
- **Cyberpunk Aesthetic**: High-contrast visual style matching the QuantoniumOS brand.

## How to Run

1. Open a terminal in this folder:
   ```bash
   cd hardware/rftpu-3d-viewer
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the viewer:
   ```bash
   npm run dev
   ```

4. Open the link shown in the terminal (usually `http://localhost:5173`).
