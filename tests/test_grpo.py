"""
Unit tests for the GRPO Trainer.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import torch
from dualmation.reward.reward_model import RewardConfig
from dualmation.training.grpo import DualAnimateGRPOTrainer, create_reward_functions


class TestGRPOTrainer(unittest.TestCase):
    """Test suite for DualAnimateGRPOTrainer."""

    def setUp(self):
        self.reward_config = RewardConfig()
        self.training_config = MagicMock()
        self.training_config.learning_rate = 1e-5
        self.training_config.batch_size = 8
        self.training_config.num_generations = 2
        self.training_config.num_epochs = 1
        self.training_config.output_dir = "test_grpo"
        self.model_name = "test-model"

    def test_reward_functions_adapter(self):
        """Test that reward models are correctly adapted for GRPO."""
        mock_reward_model = MagicMock()
        mock_reward_model.score.return_value = MagicMock(
            compilation_success=1.0,
            concept_alignment=0.8
        )
        
        funcs = create_reward_functions(mock_reward_model)
        self.assertEqual(len(funcs), 2)
        
        prompts = ["Explain gravity"]
        completions = ["from manim import * ..."]
        
        # Test compilation reward
        comp_scores = funcs[0](prompts, completions)
        self.assertEqual(comp_scores, [1.0])
        
        # Test alignment reward
        align_scores = funcs[1](prompts, completions)
        self.assertEqual(align_scores, [0.8])

    @patch("dualmation.training.grpo.AutoModelForCausalLM")
    @patch("dualmation.training.grpo.AutoTokenizer")
    @patch("dualmation.training.grpo.GRPOTrainer")
    def test_trainer_init(self, mock_grpo_trainer, mock_tokenizer, mock_model):
        """Test that the trainer initializes correctly with mocks."""
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()
        
        trainer = DualAnimateGRPOTrainer(
            model_name=self.model_name,
            reward_config=self.reward_config,
            training_config=self.training_config,
            device="cpu",
        )
        
        self.assertIsNotNone(trainer.grpo_config)
        self.assertEqual(len(trainer.reward_funcs), 2)
        self.assertEqual(trainer.grpo_config.learning_rate, 1e-5)

    @patch("dualmation.training.grpo.AutoModelForCausalLM")
    @patch("dualmation.training.grpo.AutoTokenizer")
    @patch("dualmation.training.grpo.GRPOTrainer")
    def test_train_call(self, mock_grpo_trainer, mock_tokenizer, mock_model):
        """Test that trainer.train calls TRL's train method."""
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()
        mock_instance = mock_grpo_trainer.return_value
        
        trainer = DualAnimateGRPOTrainer(
            model_name=self.model_name,
            reward_config=self.reward_config,
            training_config=self.training_config,
            device="cpu",
        )
        
        mock_dataset = MagicMock()
        trainer.train(mock_dataset)
        
        mock_instance.train.assert_called_once()


if __name__ == "__main__":
    unittest.main()
