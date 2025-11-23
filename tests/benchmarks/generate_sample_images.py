#!/usr/bin/env python3
"""Generate small synthetic images for benchmarking.

Creates a handful of 128x128 PNG images with gradients, noise, and pattern mixes
under `tests/data/images/`. Safe to run multiple times; will overwrite.
"""
from pathlib import Path
import numpy as np
from PIL import Image

def make_gradient(size=128):
    x = np.linspace(0, 255, size, dtype=np.uint8)
    img = np.tile(x, (size, 1))
    return np.stack([img, img.T, img[::-1]], axis=2)

def make_noise(size=128, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)

def make_checker(size=128, blocks=8):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    bs = size // blocks
    for i in range(blocks):
        for j in range(blocks):
            if (i + j) % 2 == 0:
                img[i*bs:(i+1)*bs, j*bs:(j+1)*bs] = 255
    return img

def save(arr, path):
    Image.fromarray(arr).save(path)

def main():
    out_dir = Path("tests/data/images")
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = {
        "gradient.png": make_gradient(),
        "noise.png": make_noise(seed=1234),
        "checker.png": make_checker(),
        "gradient_rot.png": np.rot90(make_gradient(), k=1),
    }
    for name, arr in samples.items():
        save(arr, out_dir / name)
    print(f"Generated {len(samples)} images in {out_dir}")

if __name__ == "__main__":
    main()
