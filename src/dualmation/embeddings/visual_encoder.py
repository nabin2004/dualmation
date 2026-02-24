"""
ViT-based visual encoder for projecting images into the shared embedding space.

Uses `google/vit-base-patch16-224` as the backbone, with a learned linear projection
head that maps the CLS token representation into the multimodal embedding space.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModel, ViTImageProcessor


class VisualEncoder(nn.Module):
    """Encodes images into the shared multimodal embedding space.

    Architecture:
        ViT (frozen or fine-tuned) → CLS token → Linear → LayerNorm → embedding

    Args:
        model_name: HuggingFace model identifier for the visual encoder backbone.
        embedding_dim: Dimension of the shared embedding space.
        freeze_backbone: If True, freeze the backbone weights (only train the projection head).
    """

    DEFAULT_MODEL = "google/vit-base-patch16-224"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        embedding_dim: int = 512,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()

        # Load backbone
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Projection head: backbone hidden → shared embedding space
        backbone_dim = self.backbone.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Forward pass with pre-processed pixel values.

        Args:
            pixel_values: Preprocessed image tensor of shape (batch, channels, height, width).

        Returns:
            Embeddings of shape (batch_size, embedding_dim), L2-normalized.
        """
        outputs = self.backbone(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]  # CLS token
        embedding = self.projection(cls_token)
        # L2 normalize for cosine similarity in contrastive loss
        embedding = nn.functional.normalize(embedding, p=2, dim=-1)
        return embedding

    def encode(
        self, images: Image.Image | list[Image.Image], device: torch.device | None = None
    ) -> torch.Tensor:
        """Convenience method to encode raw PIL image(s) into embeddings.

        Args:
            images: A single PIL image or list of PIL images.
            device: Device to place tensors on. Defaults to model device.

        Returns:
            Embeddings of shape (batch_size, embedding_dim).
        """
        if isinstance(images, Image.Image):
            images = [images]

        if device is None:
            device = next(self.parameters()).device

        inputs = self.processor(images=images, return_tensors="pt").to(device)

        with torch.no_grad():
            return self.forward(inputs["pixel_values"])
