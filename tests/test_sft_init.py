"""
Unit tests for LLM SFT initialization.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import torch
from dualmation.experiment.config import LLMConfig, TrainingConfig, LoRAConfig
from dualmation.training.sft import LLMSFTTrainer


class TestSFTTrainer(unittest.TestCase):
    """Test suite for LLMSFTTrainer initialization."""

    def setUp(self):
        self.llm_config = LLMConfig(
            model_name="test-llm-model",
            use_lora=True,
            lora=LoRAConfig(r=8)
        )
        self.training_config = TrainingConfig()

    @patch("dualmation.training.sft.AutoModelForCausalLM")
    @patch("dualmation.training.sft.AutoTokenizer")
    @patch("dualmation.training.sft.SFTTrainer")
    def test_sft_init(self, mock_sft_trainer, mock_tokenizer, mock_model):
        """Test that the SFT trainer initializes correctly with LoRA."""
        # Setup mocks
        mock_model_inst = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_inst
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        
        trainer = LLMSFTTrainer(
            llm_config=self.llm_config,
            training_config=self.training_config,
            device="cpu"
        )
        
        self.assertIsNotNone(trainer.peft_config)
        self.assertEqual(trainer.peft_config.r, 8)
        mock_model.from_pretrained.assert_called_once()


    @patch("dualmation.training.sft.AutoModelForCausalLM")
    @patch("dualmation.training.sft.AutoTokenizer")
    @patch("dualmation.training.sft.SFTTrainer")
    @patch("dualmation.training.sft.TrainingArguments")
    def test_train_call(self, mock_train_args, mock_sft_trainer, mock_tokenizer, mock_model):
        """Test that trainer.train can be called (though it delegates to trl)."""
        mock_model.from_pretrained.return_value = MagicMock()
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_trainer_inst = mock_sft_trainer.return_value
        
        trainer = LLMSFTTrainer(
            llm_config=self.llm_config,
            training_config=self.training_config,
            device="cpu"
        )
        
        mock_dataset = MagicMock()
        trainer.train(mock_dataset, output_dir="test_sft_out")
        
        mock_sft_trainer.assert_called_once()
        mock_trainer_inst.train.assert_called_once()


if __name__ == "__main__":
    unittest.main()
