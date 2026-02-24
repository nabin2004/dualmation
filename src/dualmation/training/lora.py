"""
LoRA (Low-Rank Adaptation) Trainer for Stable Diffusion.

Enables fine-tuning Stable Diffusion on specialized aesthetics like
"flat, educational illustration" with minimal compute overhead.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import UNet2DConditionModel, StableDiffusionPipeline
from peft import LoraConfig, get_peft_model, PeftModel

from dualmation.experiment.config import DiffusionConfig, TrainingConfig

logger = logging.getLogger(__name__)


class DiffusionLoRATrainer:
    """Trainer for Stable Diffusion LoRA fine-tuning.

    Args:
        diffusion_config: Configuration for the diffusion model.
        training_config: Configuration for training.
        device: Compute device.
    """

    def __init__(
        self,
        diffusion_config: DiffusionConfig,
        training_config: TrainingConfig,
        device: str = "auto",
    ) -> None:
        self.diffusion_config = diffusion_config
        self.training_config = training_config
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.accelerator = Accelerator(
            gradient_accumulation_steps=1,
            mixed_precision="fp16" if torch.cuda.is_available() else "no",
        )

        # Load UNet for training
        self.unet = UNet2DConditionModel.from_pretrained(
            diffusion_config.model_name,
            subfolder="unet",
            torch_dtype=torch.float32,
        )

        # Inject LoRA layers
        if diffusion_config.use_lora:
            lora_config = LoraConfig(
                r=diffusion_config.lora.r,
                lora_alpha=diffusion_config.lora.alpha,
                target_modules=diffusion_config.lora.target_modules,
                lora_dropout=diffusion_config.lora.dropout,
                bias=diffusion_config.lora.bias,
            )
            self.unet = get_peft_model(self.unet, lora_config)
            logger.info("LoRA layers injected into UNet (rank=%d)", diffusion_config.lora.r)

        self.unet.to(self.accelerator.device)

    def train_step(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Perform a single training step (Denoising Diffusion loss).
        
        Args:
            pixel_values: RGB image tensors (B, 3, H, W).
            input_ids: Tokenized text prompts.
            
        Returns:
            Scalar loss tensor.
        """
        # This is a simplified version of the training step
        # In a full implementation, we would need:
        # 1. VAE to encode images to latents
        # 2. Text Encoder to get prompt embeddings
        # 3. Noise Scheduler to add noise to latents
        # 4. UNet to predict noise
        
        # For now, we return a dummy loss to verify the flow
        # In a real scenario, this would be the MSE between noise and predicted noise
        return torch.tensor(0.0, requires_grad=True, device=self.accelerator.device)

    def save_lora_weights(self, path: str | Path) -> None:
        """Save the LoRA weights only."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        if isinstance(self.unet, PeftModel):
            self.unet.save_pretrained(path)
            logger.info("Saved LoRA weights to %s", path)
        else:
            logger.warning("Model is not a PeftModel, nothing to save as LoRA.")

    @classmethod
    def load_lora_into_pipeline(
        cls, 
        pipeline: StableDiffusionPipeline, 
        lora_path: str | Path
    ) -> StableDiffusionPipeline:
        """Utility to load trained LoRA weights into an inference pipeline."""
        pipeline.load_lora_weights(str(lora_path))
        logger.info("LoRA weights loaded from %s", lora_path)
        return pipeline
