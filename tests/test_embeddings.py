"""
Tests for the multimodal embedding module.

Uses mock/lightweight data to test without requiring GPU or model downloads.
"""

from __future__ import annotations

import pytest
import torch
import numpy as np
from PIL import Image

from dualmation.embeddings.contrastive import InfoNCELoss


class TestInfoNCELoss:
    """Tests for the InfoNCE contrastive loss."""

    def test_loss_shape(self):
        """Loss should return a scalar tensor."""
        loss_fn = InfoNCELoss(temperature=0.07)
        # Random normalized embeddings
        code_emb = torch.randn(8, 512)
        code_emb = torch.nn.functional.normalize(code_emb, p=2, dim=-1)
        vis_emb = torch.randn(8, 512)
        vis_emb = torch.nn.functional.normalize(vis_emb, p=2, dim=-1)

        result = loss_fn(code_emb, vis_emb)

        assert "loss" in result
        assert result["loss"].dim() == 0  # scalar
        assert result["loss"].item() > 0  # loss is positive

    def test_symmetric_loss(self):
        """c2v and v2c losses should be roughly similar for random data."""
        loss_fn = InfoNCELoss(temperature=0.07)
        code_emb = torch.nn.functional.normalize(torch.randn(16, 256), p=2, dim=-1)
        vis_emb = torch.nn.functional.normalize(torch.randn(16, 256), p=2, dim=-1)

        result = loss_fn(code_emb, vis_emb)

        # Both losses should be in a reasonable range
        assert result["loss_c2v"].item() > 0
        assert result["loss_v2c"].item() > 0
        assert result["loss"].item() == pytest.approx(
            (result["loss_c2v"].item() + result["loss_v2c"].item()) / 2, rel=1e-5
        )

    def test_perfect_alignment(self):
        """Identical embeddings should have high accuracy."""
        loss_fn = InfoNCELoss(temperature=0.5)
        emb = torch.nn.functional.normalize(torch.randn(4, 64), p=2, dim=-1)

        result = loss_fn(emb, emb)

        # Perfect alignment → accuracy should be 1.0
        assert result["accuracy_c2v"].item() == 1.0
        assert result["accuracy_v2c"].item() == 1.0

    def test_temperature_parameter(self):
        """Temperature should be a learnable parameter."""
        loss_fn = InfoNCELoss(temperature=0.1)
        assert loss_fn.temperature.requires_grad is True
        temp_val = loss_fn.current_temperature
        assert abs(temp_val - 0.1) < 0.01

    def test_batch_size_one(self):
        """Should work with batch size 1 (degenerate case)."""
        loss_fn = InfoNCELoss()
        code_emb = torch.nn.functional.normalize(torch.randn(1, 128), p=2, dim=-1)
        vis_emb = torch.nn.functional.normalize(torch.randn(1, 128), p=2, dim=-1)

        result = loss_fn(code_emb, vis_emb)
        assert result["loss"].dim() == 0
        # With batch_size=1, accuracy is always 1.0 (only one choice)
        assert result["accuracy_c2v"].item() == 1.0
