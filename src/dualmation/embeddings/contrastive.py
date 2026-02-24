"""
Contrastive learning module using InfoNCE loss for aligning code and visual embeddings.

Learns to pull code-visual pairs together while pushing non-matching pairs apart
in the shared multimodal embedding space.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss for code-visual alignment.

    Given a batch of (code_emb, visual_emb) pairs, computes symmetric cross-entropy
    loss over cosine similarity. Uses temperature scaling to control the sharpness
    of the probability distribution.

    Args:
        temperature: Temperature parameter for scaling logits. Lower temperatures
            create sharper distributions (harder negatives matter more).
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = nn.Parameter(
            torch.tensor(temperature).log(), requires_grad=True
        )

    @property
    def current_temperature(self) -> float:
        """Current temperature value (exponentiated from log-space)."""
        return self.temperature.exp().item()

    def forward(
        self,
        code_embeddings: torch.Tensor,
        visual_embeddings: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute symmetric InfoNCE loss.

        Args:
            code_embeddings: L2-normalized code embeddings of shape (batch_size, embed_dim).
            visual_embeddings: L2-normalized visual embeddings of shape (batch_size, embed_dim).

        Returns:
            Dictionary with keys:
                - "loss": scalar total loss (average of code→visual and visual→code)
                - "loss_c2v": code-to-visual loss
                - "loss_v2c": visual-to-code loss
                - "accuracy_c2v": code-to-visual retrieval accuracy
                - "accuracy_v2c": visual-to-code retrieval accuracy
        """
        # Temperature-scaled cosine similarity matrix
        temp = self.temperature.exp().clamp(min=1e-4)
        logits = (code_embeddings @ visual_embeddings.T) / temp  # (B, B)

        # Labels: diagonal elements are positives
        batch_size = logits.shape[0]
        labels = torch.arange(batch_size, device=logits.device)

        # Symmetric cross-entropy
        loss_c2v = F.cross_entropy(logits, labels)
        loss_v2c = F.cross_entropy(logits.T, labels)
        loss = (loss_c2v + loss_v2c) / 2.0

        # Accuracy metrics
        with torch.no_grad():
            acc_c2v = (logits.argmax(dim=1) == labels).float().mean()
            acc_v2c = (logits.T.argmax(dim=1) == labels).float().mean()

        return {
            "loss": loss,
            "loss_c2v": loss_c2v,
            "loss_v2c": loss_v2c,
            "accuracy_c2v": acc_c2v,
            "accuracy_v2c": acc_v2c,
        }


class ContrastiveEmbeddingModel(nn.Module):
    """Wrapper combining both encoders with the contrastive loss.

    Manages the full contrastive training loop: encode code → encode images →
    compute InfoNCE loss.

    Args:
        code_encoder: The CodeEncoder module.
        visual_encoder: The VisualEncoder module.
        temperature: Temperature for InfoNCE loss.
    """

    def __init__(
        self,
        code_encoder: nn.Module,
        visual_encoder: nn.Module,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.code_encoder = code_encoder
        self.visual_encoder = visual_encoder
        self.criterion = InfoNCELoss(temperature=temperature)

    def forward(
        self,
        code_input_ids: torch.Tensor,
        code_attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through both encoders and contrastive loss.

        Args:
            code_input_ids: Tokenized code input IDs.
            code_attention_mask: Attention mask for code tokens.
            pixel_values: Preprocessed image pixel values.

        Returns:
            Dictionary with loss and accuracy metrics from InfoNCE.
        """
        code_emb = self.code_encoder(code_input_ids, code_attention_mask)
        visual_emb = self.visual_encoder(pixel_values)
        return self.criterion(code_emb, visual_emb)
