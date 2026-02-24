"""
Unit tests for color palette extraction and theme sync.
"""

from __future__ import annotations

import unittest
from PIL import Image
import numpy as np

from dualmation.utils.colors import extract_palette, palette_to_manim_constants


class TestColorSync(unittest.TestCase):
    """Test suite for color extraction and theme synchronization logic."""

    def test_palette_extraction(self):
        """Test that k-means extracts consistent colors from a simple image."""
        # Create a red and blue image
        data = np.zeros((100, 100, 3), dtype=np.uint8)
        data[:50, :, :] = [255, 0, 0]   # Red top half
        data[50:, :, :] = [0, 0, 255]   # Blue bottom half
        img = Image.fromarray(data)
        
        palette = extract_palette(img, num_colors=2)
        
        self.assertEqual(len(palette), 2)
        # Check if #ff0000 and #0000ff are in palette (approximate)
        self.assertTrue(any(c.lower() == "#ff0000" for c in palette))
        self.assertTrue(any(c.lower() == "#0000ff" for c in palette))

    def test_palette_to_constants(self):
        """Test conversion of palette list to Manim-ready code strings."""
        palette = ["#ff0000", "#0000ff"]
        code = palette_to_manim_constants(palette)
        
        self.assertIn("THEME_PRIMARY = '#ff0000'", code)
        self.assertIn("THEME_SECONDARY = '#0000ff'", code)


if __name__ == "__main__":
    unittest.main()
