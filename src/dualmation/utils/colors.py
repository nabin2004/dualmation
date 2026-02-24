"""
Utilities for color extraction and theme synchronization.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def extract_palette(image: Image.Image, num_colors: int = 5) -> list[str]:
    """Extract a dominant color palette from an image using k-means.

    Args:
        image: The source PIL image.
        num_colors: Number of dominant colors to extract.

    Returns:
        List of hex color strings (e.g., ["#FFFFFF", "#000000"]).
    """
    # Resize for faster processing
    img = image.copy()
    img.thumbnail((150, 150))
    
    # Convert to RGB and then to numpy array
    img_rgb = img.convert("RGB")
    data = np.asarray(img_rgb).reshape(-1, 3).astype(float)
    
    if len(data) == 0:
        return ["#FFFFFF"] * num_colors

    # Simple k-means implementation
    centroids = data[np.random.choice(data.shape[0], num_colors, replace=False)]
    
    for _ in range(10): # 10 iterations is usually enough for a palette
        # Compute distances to centroids
        distances = np.sqrt(((data[:, np.newaxis, :] - centroids)**2).sum(axis=2))
        # Assign to nearest centroid
        labels = np.argmin(distances, axis=1)
        
        # Update centroids
        new_centroids = np.array([
            data[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
            for i in range(num_colors)
        ])
        
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    # Convert to hex
    hex_colors = []
    for c in centroids:
        rgb = tuple(np.clip(c, 0, 255).astype(int))
        hex_colors.append(f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}")
    
    return hex_colors


def palette_to_manim_constants(palette: list[str]) -> str:
    """Convert a list of hex colors to Manim variable definitions.

    Args:
        palette: List of hex color strings.

    Returns:
        String of Python code defining THEME_COLOR_X variables.
    """
    code_lines = []
    names = ["PRIMARY", "SECONDARY", "ACCENT", "BG_DARK", "BG_LIGHT"]
    
    for i, color in enumerate(palette):
        name = names[i] if i < len(names) else f"EXTRA_{i}"
        code_lines.append(f"THEME_{name} = '{color}'")
        
    return "\n".join(code_lines)
