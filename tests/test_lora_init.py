"""
Unit tests for Diffusion LoRA initialization.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import torch
from dualmation.experiment.config import DiffusionConfig, TrainingConfig, LoRAConfig
from dualmation.training.lora import DiffusionLoRATrainer


class TestLoRATrainer(unittest.TestCase):
    """Test suite for DiffusionLoRATrainer initialization."""

    def setUp(self):
        self.diffusion_config = DiffusionConfig(
            model_name="test-diffusion-model",
            use_lora=True,
            lora=LoRAConfig(r=4)
        )
        self.training_config = TrainingConfig()

    @patch("dualmation.training.lora.UNet2DConditionModel")
    @patch("dualmation.training.lora.get_peft_model")
    @patch("dualmation.training.lora.Accelerator")
    def test_lora_injection(self, mock_accelerator, mock_get_peft_model, mock_unet):
        """Test that LoRA layers are injected into the UNet."""
        # Setup mocks
        mock_unet_inst = MagicMock()
        mock_unet.from_pretrained.return_value = mock_unet_inst
        mock_get_peft_model.return_value = MagicMock()
        
        trainer = DiffusionLoRATrainer(
            diffusion_config=self.diffusion_config,
            training_config=self.training_config,
            device="cpu"
        )
        
        mock_unet.from_pretrained.assert_called_once()
        mock_get_peft_model.assert_called_once()
        logger_name = "dualmation.training.lora"
        # We can't easily check logging here without more setup, 
        # but the mock calls confirm the logic path.

    @patch("dualmation.training.lora.UNet2DConditionModel")
    @patch("dualmation.training.lora.get_peft_model")
    @patch("dualmation.training.lora.Accelerator")
    def test_no_lora_init(self, mock_accelerator, mock_get_peft_model, mock_unet):
        """Test that LoRA is NOT injected if use_lora is False."""
        self.diffusion_config.use_lora = False
        mock_unet.from_pretrained.return_value = MagicMock()
        
        trainer = DiffusionLoRATrainer(
            diffusion_config=self.diffusion_config,
            training_config=self.training_config,
            device="cpu"
        )
        
        mock_get_peft_model.assert_not_called()

if __name__ == "__main__":
    unittest.main()
