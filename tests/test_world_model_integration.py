"""
Unit tests for ManimWorldModel integration in RewardModel.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
import torch

from dualmation.reward.reward_model import RewardModel, RewardConfig


class TestWorldModelIntegration(unittest.TestCase):
    """Test suite for World Model fast-pass logic in RewardModel."""

    def setUp(self):
        self.config = RewardConfig(
            world_model_path="dummy_path.pt",
            world_model_threshold=0.5
        )
        
        # Patch the model loading to avoid actual weight loading
        with patch('dualmation.reward.reward_model.RewardModel._load_world_model'):
            self.reward_model = RewardModel(self.config)
            self.reward_model._world_model = MagicMock()
            self.reward_model._world_tokenizer = MagicMock()

    def test_world_model_skip_on_low_prob(self):
        """Test that subprocess is skipped if World Model predicts low probability."""
        # Mock low compilability
        self.reward_model._world_model.predict_code.return_value = 0.1
        
        with patch('subprocess.run') as mock_run:
            score, output = self.reward_model._score_compilation("print('hello')")
            
            # Subprocess should NOT be called
            mock_run.assert_not_called()
            self.assertEqual(score, 0.05) # 0.1 * 0.5 per logic
            self.assertIn("Skipped", output)

    def test_world_model_pass_on_high_prob(self):
        """Test that subprocess IS called if World Model predicts high probability."""
        # Mock high compilability
        self.reward_model._world_model.predict_code.return_value = 0.9
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            score, output = self.reward_model._score_compilation("from manim import *")
            
            # Subprocess SHOULD be called (twice in our current implementation: syntax + manim)
            self.assertGreaterEqual(mock_run.call_count, 1)
            self.assertEqual(score, 1.0)
            self.assertIn("successful", output)

    def test_use_world_model_only(self):
        """Test the ultra-fast proxy mode."""
        self.reward_model.config.use_world_model_only = True
        self.reward_model._world_model.predict_code.return_value = 0.75
        
        with patch('subprocess.run') as mock_run:
            score, output = self.reward_model._score_compilation("test code")
            mock_run.assert_not_called()
            self.assertEqual(score, 0.75)
            self.assertIn("Predicted by World Model", output)

if __name__ == "__main__":
    unittest.main()
