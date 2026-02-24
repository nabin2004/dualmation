"""
Tests for the alpha compositing module.

Uses synthetic images to test compositing without real Manim/diffusion outputs.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from dualmation.compositor.compositor import AlphaCompositor, CompositeConfig


def _make_solid_image(color: tuple[int, ...], size: tuple[int, int] = (200, 100)) -> Image.Image:
    """Create a solid color RGBA image."""
    img = Image.new("RGBA", size, color)
    return img


def _make_gradient_image(size: tuple[int, int] = (200, 100)) -> Image.Image:
    """Create a horizontal gradient image."""
    arr = np.zeros((*size[::-1], 4), dtype=np.uint8)
    for x in range(size[0]):
        val = int(255 * x / size[0])
        arr[:, x, :3] = val
        arr[:, x, 3] = 255
    return Image.fromarray(arr)


class TestAlphaCompositor:
    """Tests for the AlphaCompositor."""

    def test_composite_dimensions(self):
        """Output should match configured dimensions."""
        config = CompositeConfig(output_width=640, output_height=480)
        compositor = AlphaCompositor(config)

        fg = _make_solid_image((255, 0, 0, 128), size=(100, 100))
        bg = _make_solid_image((0, 0, 255, 255), size=(300, 200))

        result = compositor.composite(fg, bg)

        assert result.size == (640, 480)
        assert result.mode == "RGBA"

    def test_composite_opaque_foreground(self):
        """Fully opaque foreground should dominate."""
        config = CompositeConfig(output_width=100, output_height=100)
        compositor = AlphaCompositor(config)

        fg = _make_solid_image((255, 0, 0, 255), size=(100, 100))
        bg = _make_solid_image((0, 0, 255, 255), size=(100, 100))

        result = compositor.composite(fg, bg)
        result_arr = np.array(result)

        # The foreground is red and fully opaque → result should be mostly red
        # (alpha enhancement may modify some pixels in the black detection step)
        avg_red = result_arr[:, :, 0].mean()
        assert avg_red > 200, f"Expected mostly red, got avg_red={avg_red}"

    def test_composite_transparent_foreground(self):
        """Fully transparent foreground should show background."""
        config = CompositeConfig(output_width=100, output_height=100)
        compositor = AlphaCompositor(config)

        fg = _make_solid_image((255, 0, 0, 0), size=(100, 100))
        bg = _make_solid_image((0, 0, 255, 255), size=(100, 100))

        result = compositor.composite(fg, bg)
        result_arr = np.array(result)

        # Transparent foreground → result should be the blue background
        avg_blue = result_arr[:, :, 2].mean()
        assert avg_blue > 200, f"Expected mostly blue, got avg_blue={avg_blue}"

    def test_composite_sequence(self):
        """Sequence compositing should produce correct number of frames."""
        config = CompositeConfig(output_width=100, output_height=100)
        compositor = AlphaCompositor(config)

        fg_frames = [_make_solid_image((255, 0, 0, 128)) for _ in range(5)]
        bg_frames = [_make_solid_image((0, 0, 255, 255)) for _ in range(3)]

        results = compositor.composite_sequence(fg_frames, bg_frames)

        assert len(results) == 5  # Should match foreground count
        for frame in results:
            assert frame.size == (100, 100)

    def test_explicit_alpha_mask(self):
        """Should use explicit alpha mask when provided."""
        config = CompositeConfig(output_width=100, output_height=100)
        compositor = AlphaCompositor(config)

        fg = _make_solid_image((255, 0, 0, 255), size=(100, 100))
        bg = _make_solid_image((0, 0, 255, 255), size=(100, 100))
        mask = Image.new("L", (100, 100), 128)  # 50% alpha

        result = compositor.composite(fg, bg, alpha_mask=mask)
        assert result.size == (100, 100)

    def test_screen_blend_mode(self):
        """Screen blend mode should produce valid output."""
        config = CompositeConfig(output_width=100, output_height=100, blend_mode="screen")
        compositor = AlphaCompositor(config)

        fg = _make_solid_image((128, 0, 0, 255), size=(100, 100))
        bg = _make_solid_image((0, 0, 128, 255), size=(100, 100))

        result = compositor.composite(fg, bg)
        assert result.size == (100, 100)

    def test_multiply_blend_mode(self):
        """Multiply blend mode should produce valid output."""
        config = CompositeConfig(output_width=100, output_height=100, blend_mode="multiply")
        compositor = AlphaCompositor(config)

        fg = _make_solid_image((200, 200, 200, 255), size=(100, 100))
        bg = _make_solid_image((100, 100, 100, 255), size=(100, 100))

        result = compositor.composite(fg, bg)
        assert result.size == (100, 100)

    def test_save_sequence(self, tmp_path):
        """Save sequence should write files to disk."""
        compositor = AlphaCompositor()
        frames = [_make_solid_image((255, 0, 0, 255)) for _ in range(3)]

        paths = compositor.save_sequence(frames, tmp_path, prefix="test")

        assert len(paths) == 3
        for p in paths:
            assert p.exists()
            assert p.suffix == ".png"
