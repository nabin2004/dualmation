"""
Unit tests for the LLM self-correction "Compiler Loop".
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from dualmation.pipeline import DualAnimatePipeline, PipelineResult
from dualmation.experiment.config import ExperimentConfig
from dualmation.reward.reward_model import RewardScore


class TestSelfCorrection(unittest.TestCase):
    """Test suite for the self-correction loop in DualAnimatePipeline."""

    def setUp(self):
        self.config = ExperimentConfig()
        self.config.llm.enable_self_correction = True
        self.config.llm.max_correction_turns = 2
        
        # Mock modules
        self.pipeline = DualAnimatePipeline(self.config)
        self.pipeline._code_encoder = MagicMock()
        self.pipeline._code_generator = MagicMock()
        self.pipeline._visual_generator = MagicMock()
        self.pipeline._reward_model = MagicMock()

    def test_self_correction_loop_success(self):
        """Test that the loop retries on failure and stops on success."""
        # Turn 0: Failure
        reward_fail = RewardScore(
            total=0.3, 
            concept_alignment=0.5, 
            visual_quality=0.5, 
            compilation_success=0.0,
            compilation_output="NameError: name 'MathText' is not defined"
        )
        # Turn 1: Success
        reward_success = RewardScore(
            total=0.9, 
            concept_alignment=0.9, 
            visual_quality=0.9, 
            compilation_success=1.0
        )
        
        self.pipeline._reward_model.score.side_effect = [reward_fail, reward_success]
        self.pipeline._code_generator.generate_with_embedding.return_value = "invalid code"
        self.pipeline._code_generator.generate_correction.return_value = "fixed code"
        self.pipeline._visual_generator.generate_with_embedding.return_value = []

        result = self.pipeline.run("test concept")
        
        # Check calls
        self.assertEqual(self.pipeline._reward_model.score.call_count, 2)
        self.assertEqual(self.pipeline._code_generator.generate_correction.call_count, 1)
        self.assertEqual(result.generated_code, "fixed code")
        self.assertEqual(result.reward.compilation_success, 1.0)

    def test_self_correction_max_turns(self):
        """Test that the loop stops after reaching max_turns."""
        reward_fail = RewardScore(
            total=0.1, 
            concept_alignment=0.1, 
            visual_quality=0.1, 
            compilation_success=0.0,
            compilation_output="Persistent Error"
        )
        
        # max_correction_turns is 2, so it should call score 3 times (Turn 0, 1, 2)
        self.pipeline._reward_model.score.return_value = reward_fail
        self.pipeline._code_generator.generate_with_embedding.return_value = "code 0"
        self.pipeline._code_generator.generate_correction.side_effect = ["code 1", "code 2", "code 3"]
        self.pipeline._visual_generator.generate_with_embedding.return_value = []

        result = self.pipeline.run("test concept")
        
        self.assertEqual(self.pipeline._reward_model.score.call_count, 3) # Turn 0, 1, 2
        self.assertEqual(self.pipeline._code_generator.generate_correction.call_count, 2)
        self.assertEqual(result.generated_code, "code 2")
        self.assertEqual(result.reward.compilation_success, 0.0)

if __name__ == "__main__":
    unittest.main()
