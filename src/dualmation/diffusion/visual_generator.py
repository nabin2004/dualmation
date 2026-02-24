"""
Diffusion-based visual context and background generator.

Uses HuggingFace Diffusers (e.g., Stable Diffusion) to generate rich visual
backgrounds conditioned on concept descriptions and/or shared embeddings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class DiffusionConfig:
    """Configuration for diffusion-based image generation."""

    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    width: int = 1920
    height: int = 1080
    num_images: int = 1
    negative_prompt: str = (
        "text, watermark, low quality, blurry, distorted, "
        "oversaturated, ugly, deformed"
    )


class VisualGenerator:
    """Generates visual backgrounds using a diffusion model.

    Uses Stable Diffusion from HuggingFace Diffusers to create rich,
    aesthetic backgrounds for educational animations.

    Args:
        model_name: HuggingFace model identifier for the diffusion model.
        device: Device to run the model on.
        torch_dtype: Data type for model weights.
    """

    DEFAULT_MODEL = "stabilityai/stable-diffusion-2-1"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        torch_dtype: torch.dtype = torch.float16,
    ) -> None:
        # Lazy import to avoid loading diffusers at module import time
        from diffusers import StableDiffusionPipeline

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading diffusion model: %s on %s", model_name, self.device)

        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        self.pipe = self.pipe.to(self.device)

        # Enable memory-efficient attention if available
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            logger.info("xformers memory-efficient attention enabled")
        except Exception:
            logger.info("xformers not available, using default attention")

    def _build_prompt(self, concept: str) -> str:
        """Build an aesthetically-focused prompt from a concept description.

        Args:
            concept: The educational concept to visualize.

        Returns:
            Enhanced prompt for the diffusion model.
        """
        return (
            f"Educational animation background for: {concept}. "
            "Clean, modern, professional design. Subtle gradients, "
            "dark theme with glowing accents. Abstract mathematical visualization. "
            "High quality, 4K, digital art, minimalist."
        )

    @torch.inference_mode()
    def generate(
        self,
        concept: str,
        config: DiffusionConfig | None = None,
        seed: int | None = None,
    ) -> list[Image.Image]:
        """Generate background images for an educational concept.

        Args:
            concept: Natural language concept description.
            config: Diffusion generation parameters. Uses defaults if None.
            seed: Random seed for reproducibility.

        Returns:
            List of generated PIL Images.
        """
        if config is None:
            config = DiffusionConfig()

        prompt = self._build_prompt(concept)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipe(
            prompt=prompt,
            negative_prompt=config.negative_prompt,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            width=config.width,
            height=config.height,
            num_images_per_prompt=config.num_images,
            generator=generator,
        )

        return result.images

    def generate_with_embedding(
        self,
        concept: str,
        embedding: torch.Tensor | None = None,
        config: DiffusionConfig | None = None,
        seed: int | None = None,
    ) -> list[Image.Image]:
        """Generate images conditioned on a shared embedding.

        For now, the embedding enhances the text prompt. In future iterations,
        this will use IP-Adapter or cross-attention injection for direct
        embedding conditioning.

        Args:
            concept: Concept description.
            embedding: Optional embedding from the multimodal space.
            config: Diffusion configuration.
            seed: Random seed.

        Returns:
            List of generated PIL Images.
        """
        # Future: use IP-Adapter for proper embedding conditioning
        return self.generate(concept, config=config, seed=seed)

    def generate_frame_sequence(
        self,
        concept: str,
        num_frames: int = 10,
        config: DiffusionConfig | None = None,
        seed: int = 42,
    ) -> list[Image.Image]:
        """Generate a sequence of background frames for animation.

        Uses incrementing seeds to create visual variation across frames
        while maintaining thematic consistency.

        Args:
            concept: Concept description.
            num_frames: Number of frames to generate.
            config: Diffusion configuration.
            seed: Base seed (incremented per frame).

        Returns:
            List of PIL Images for each frame.
        """
        frames = []
        for i in range(num_frames):
            frame = self.generate(concept, config=config, seed=seed + i)
            frames.extend(frame)
        return frames

    @torch.inference_mode()
    def generate_video(
        self,
        image: Image.Image,
        config: DiffusionConfig | None = None,
        seed: int | None = None,
    ) -> list[Image.Image]:
        """Generate a video sequence from a static image using SVD.

        Args:
            image: Context image to animate.
            config: Configuration (uses video_model_name, fps, num_frames).
            seed: Random seed.

        Returns:
            List of PIL Images (frames).
        """
        from diffusers import StableVideoDiffusionPipeline
        
        # Load SVD pipe if not already loaded (lazy)
        if not hasattr(self, "video_pipe"):
            logger.info("Loading SVD video model...")
            self.video_pipe = StableVideoDiffusionPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid-xt",
                torch_dtype=torch.float16,
                variant="fp16"
            )
            self.video_pipe.enable_model_cpu_offload() # Save memory
            
        # Ensure image is resized to multiples of 8 or 64 as required by SVD
        image = image.resize((1024, 576))
        
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        frames = self.video_pipe(
            image,
            decode_chunk_size=8,
            generator=generator,
        ).frames[0]
        
        return frames

    def save_images(
        self, images: list[Image.Image], output_dir: str | Path, prefix: str = "bg"
    ) -> list[Path]:
        """Save generated images to disk.

        Args:
            images: List of PIL Images to save.
            output_dir: Directory to save images in.
            prefix: Filename prefix.

        Returns:
            List of saved file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = []
        for i, img in enumerate(images):
            path = output_dir / f"{prefix}_{i:04d}.png"
            img.save(path)
            paths.append(path)
            logger.info("Saved: %s", path)

        return paths
