"""
CodeBERT-based code encoder for projecting source code into the shared embedding space.

Uses `microsoft/codebert-base` as the backbone, with a learned linear projection head
that maps the CLS token representation into the multimodal embedding space.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class CodeEncoder(nn.Module):
    """Encodes source code strings into the shared multimodal embedding space.

    Architecture:
        CodeBERT (frozen or fine-tuned) → CLS token → Linear → LayerNorm → embedding

    Args:
        model_name: HuggingFace model identifier for the code encoder backbone.
        embedding_dim: Dimension of the shared embedding space.
        freeze_backbone: If True, freeze the backbone weights (only train the projection head).
        max_length: Maximum token sequence length for the tokenizer.
    """

    DEFAULT_MODEL = "microsoft/codebert-base"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        embedding_dim: int = 512,
        freeze_backbone: bool = True,
        max_length: int = 512,
    ) -> None:
        super().__init__()
        self.max_length = max_length

        # Load backbone
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
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

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass with pre-tokenized inputs.

        Args:
            input_ids: Tokenized input IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, seq_len).

        Returns:
            Embeddings of shape (batch_size, embedding_dim), L2-normalized.
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]  # CLS token
        embedding = self.projection(cls_token)
        # L2 normalize for cosine similarity in contrastive loss
        embedding = nn.functional.normalize(embedding, p=2, dim=-1)
        return embedding

    def encode(self, code: str | list[str], device: torch.device | None = None) -> torch.Tensor:
        """Convenience method to encode raw code string(s) into embeddings.

        Args:
            code: A single code string or list of code strings.
            device: Device to place tensors on. Defaults to model device.

        Returns:
            Embeddings of shape (batch_size, embedding_dim).
        """
        if isinstance(code, str):
            code = [code]

        if device is None:
            device = next(self.parameters()).device

        tokens = self.tokenizer(
            code,
            return_tensors="pt",
            max_length=self.max_length,
            padding=True,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            return self.forward(tokens["input_ids"], tokens["attention_mask"])
