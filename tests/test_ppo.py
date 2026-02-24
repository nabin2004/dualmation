"""
Unit tests for the PPO Trainer.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import torch
from dualmation.experiment.config import TrainingConfig
from dualmation.training.ppo import DualAnimatePPOTrainer


class TestPPOTrainer(unittest.TestCase):
    """Test suite for DualAnimatePPOTrainer."""

    def setUp(self):
        self.config = TrainingConfig(
            learning_rate=1e-5,
            batch_size=4,
            rl_kl_coeff=0.1,
        )
        self.model_name = "test-model"

    @patch("dualmation.training.ppo.AutoModelForCausalLMWithValueHead")
    @patch("dualmation.training.ppo.AutoTokenizer")
    @patch("dualmation.training.ppo.PPOTrainer")
    def test_trainer_init(self, mock_ppo_trainer, mock_tokenizer, mock_model):
        """Test that the trainer initializes correctly with mocks."""
        # Setup mocks
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()
        
        trainer = DualAnimatePPOTrainer(
            config=self.config,
            model_name=self.model_name,
            device="cpu",
        )
        
        self.assertIsNotNone(trainer.ppo_trainer)
        self.assertIsNotNone(trainer.tokenizer)
        mock_ppo_trainer.assert_called_once()

    @patch("dualmation.training.ppo.AutoModelForCausalLMWithValueHead")
    @patch("dualmation.training.ppo.AutoTokenizer")
    @patch("dualmation.training.ppo.PPOTrainer")
    def test_train_step(self, mock_ppo_trainer, mock_tokenizer, mock_model):
        """Test the training step logic."""
        # Setup mocks
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()
        mock_instance = mock_ppo_trainer.return_value
        mock_instance.step.return_value = {"ppo/loss/total": 0.5}

        trainer = DualAnimatePPOTrainer(
            config=self.config,
            model_name=self.model_name,
            device="cpu",
        )
        
        query = [torch.tensor([1, 2, 3])]
        response = [torch.tensor([4, 5, 6])]
        rewards = [torch.tensor(1.0)]
        
        stats = trainer.train_step(query, response, rewards)
        
        self.assertEqual(stats["ppo/loss/total"], 0.5)
        mock_instance.step.assert_called_with(query, response, rewards)

if __name__ == "__main__":
    unittest.main()
