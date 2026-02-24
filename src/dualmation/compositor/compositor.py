"""
Alpha compositing engine for merging Manim foreground with diffusion background.

Composites mathematically precise Manim-rendered foreground elements over
diffusion-generated aesthetic backgrounds using alpha blending.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class CompositeConfig:
    """Configuration for the compositing engine."""

    output_width: int = 1920
    output_height: int = 1080
    background_opacity: float = 1.0
    foreground_opacity: float = 1.0
    blend_mode: str = "alpha"  # "alpha", "screen", "multiply"


class AlphaCompositor:
    """Composites Manim foreground frames over diffusion background frames.

    Supports multiple blend modes and batch processing of frame sequences.
    Manim typically renders with a transparent or solid background — this
    module extracts the foreground content and layers it over the diffusion
    background.

    Args:
        config: Composite configuration. Uses defaults if None.
    """

    def __init__(self, config: CompositeConfig | None = None) -> None:
        self.config = config or CompositeConfig()

    def composite(
        self,
        foreground: Image.Image,
        background: Image.Image,
        alpha_mask: Image.Image | None = None,
    ) -> Image.Image:
        """Composite a single foreground frame over a background.

        Args:
            foreground: Manim-rendered foreground image (RGBA or RGB).
            background: Diffusion-generated background image.
            alpha_mask: Optional explicit alpha mask. If None, uses the foreground's
                alpha channel (if RGBA), or generates one automatically.

        Returns:
            Composited PIL Image.
        """
        # Resize both to target dimensions
        target_size = (self.config.output_width, self.config.output_height)
        bg = background.convert("RGBA").resize(target_size, Image.Resampling.LANCZOS)
        fg = foreground.convert("RGBA").resize(target_size, Image.Resampling.LANCZOS)

        # Extract or generate alpha mask
        if alpha_mask is not None:
            mask = alpha_mask.convert("L").resize(target_size, Image.Resampling.LANCZOS)
        else:
            # Use foreground alpha channel
            mask = fg.split()[3]
            # Auto-detect if foreground has a solid black background (common for Manim)
            mask = self._enhance_alpha(fg, mask)

        # Apply opacity adjustments
        if self.config.foreground_opacity < 1.0:
            mask_np = np.array(mask).astype(np.float32)
            mask_np *= self.config.foreground_opacity
            mask = Image.fromarray(mask_np.astype(np.uint8))

        if self.config.background_opacity < 1.0:
            bg_np = np.array(bg).astype(np.float32)
            bg_np[..., 3] *= self.config.background_opacity
            bg = Image.fromarray(bg_np.astype(np.uint8))

        # Perform compositing based on blend mode
        if self.config.blend_mode == "alpha":
            result = self._alpha_blend(fg, bg, mask)
        elif self.config.blend_mode == "screen":
            result = self._screen_blend(fg, bg, mask)
        elif self.config.blend_mode == "multiply":
            result = self._multiply_blend(fg, bg, mask)
        else:
            raise ValueError(f"Unknown blend mode: {self.config.blend_mode}")

        return result

    def composite_sequence(
        self,
        foreground_frames: list[Image.Image],
        background_frames: list[Image.Image],
        alpha_masks: list[Image.Image] | None = None,
    ) -> list[Image.Image]:
        """Composite a sequence of frames for video output.

        If the background sequence is shorter than the foreground, the last
        background frame is repeated for remaining foreground frames.

        Args:
            foreground_frames: List of Manim-rendered foreground frames.
            background_frames: List of diffusion-generated backgrounds.
            alpha_masks: Optional list of alpha masks per frame.

        Returns:
            List of composited PIL Images.
        """
        results = []
        for i, fg in enumerate(foreground_frames):
            # Repeat last background if we run out
            bg_idx = min(i, len(background_frames) - 1)
            bg = background_frames[bg_idx]

            mask = None
            if alpha_masks and i < len(alpha_masks):
                mask = alpha_masks[i]

            composited = self.composite(fg, bg, alpha_mask=mask)
            results.append(composited)

        logger.info("Composited %d frames", len(results))
        return results

    def _enhance_alpha(self, fg: Image.Image, alpha: Image.Image) -> Image.Image:
        """Enhance alpha mask by detecting solid black regions as background.

        Manim often renders on a pure black background. This method detects
        near-black pixels and marks them as transparent.

        Args:
            fg: Foreground RGBA image.
            alpha: Original alpha channel.

        Returns:
            Enhanced alpha mask.
        """
        fg_np = np.array(fg).astype(np.float32)
        alpha_np = np.array(alpha).astype(np.float32)

        # Detect near-black pixels (Manim default background)
        rgb = fg_np[..., :3]
        brightness = rgb.mean(axis=-1)

        # Where brightness is very low AND alpha is high → likely background
        background_mask = (brightness < 15.0) & (alpha_np > 200.0)
        alpha_np[background_mask] = 0.0

        return Image.fromarray(alpha_np.astype(np.uint8))

    def _alpha_blend(
        self, fg: Image.Image, bg: Image.Image, mask: Image.Image
    ) -> Image.Image:
        """Standard alpha blending: result = fg * alpha + bg * (1 - alpha)."""
        return Image.composite(fg, bg, mask)

    def _screen_blend(
        self, fg: Image.Image, bg: Image.Image, mask: Image.Image
    ) -> Image.Image:
        """Screen blend mode: lighter result, good for glowing elements."""
        fg_np = np.array(fg).astype(np.float32) / 255.0
        bg_np = np.array(bg).astype(np.float32) / 255.0
        mask_np = np.array(mask).astype(np.float32) / 255.0
        mask_4ch = np.stack([mask_np] * 4, axis=-1)

        # Screen: 1 - (1 - fg) * (1 - bg)
        screened = 1.0 - (1.0 - fg_np) * (1.0 - bg_np)
        result = screened * mask_4ch + bg_np * (1.0 - mask_4ch)
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(result)

    def _multiply_blend(
        self, fg: Image.Image, bg: Image.Image, mask: Image.Image
    ) -> Image.Image:
        """Multiply blend mode: darker result, good for shadows."""
        fg_np = np.array(fg).astype(np.float32) / 255.0
        bg_np = np.array(bg).astype(np.float32) / 255.0
        mask_np = np.array(mask).astype(np.float32) / 255.0
        mask_4ch = np.stack([mask_np] * 4, axis=-1)

        multiplied = fg_np * bg_np
        result = multiplied * mask_4ch + bg_np * (1.0 - mask_4ch)
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(result)

    def save_sequence(
        self, frames: list[Image.Image], output_dir: str | Path, prefix: str = "frame"
    ) -> list[Path]:
        """Save composited frames to disk.

        Args:
            frames: List of composited images.
            output_dir: Output directory.
            prefix: Filename prefix.

        Returns:
            List of saved file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = []
        for i, frame in enumerate(frames):
            path = output_dir / f"{prefix}_{i:04d}.png"
            frame.convert("RGB").save(path)
            paths.append(path)

        logger.info("Saved %d frames to %s", len(paths), output_dir)
        return paths
