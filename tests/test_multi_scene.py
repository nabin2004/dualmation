"""
Unit tests for multi-scene animation support.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from dualmation.pipeline import DualAnimatePipeline
from dualmation.experiment.config import ExperimentConfig
from dualmation.reward.reward_model import RewardScore


class TestMultiSceneAnimation(unittest.TestCase):
    """Test suite for multi-scene animation support in DualAnimatePipeline."""

    def setUp(self):
        self.config = ExperimentConfig()
        self.config.llm.enable_multi_scene = True
        self.config.llm.max_scenes = 3
        
        # Mock modules
        self.pipeline = DualAnimatePipeline(self.config)
        self.pipeline._code_encoder = MagicMock()
        self.pipeline._code_generator = MagicMock()
        self.pipeline._visual_generator = MagicMock()
        self.pipeline._reward_model = MagicMock()

    def test_multi_scene_workflow(self):
        """Test the decomposition and sequential generation loop."""
        scene_descriptions = ["Intro", "Middle", "End"]
        self.pipeline._code_generator.decompose_concept.return_value = scene_descriptions
        self.pipeline._code_generator.generate_scene_with_context.side_effect = [
            "code 1", "code 2", "code 3"
        ]
        self.pipeline._visual_generator.generate_with_embedding.return_value = []
        reward = RewardScore(
            total=0.8, 
            concept_alignment=0.8, 
            visual_quality=0.8, 
            compilation_success=1.0,
            weights={"alignment": 0.4, "visual": 0.3, "compilation": 0.3}
        )
        self.pipeline._reward_model.score.return_value = reward

        result = self.pipeline.run("test complex concept")
        
        # Verify calls
        self.pipeline._code_generator.decompose_concept.assert_called_once_with(
            "test complex concept", max_scenes=3
        )
        self.assertEqual(self.pipeline._code_generator.generate_scene_with_context.call_count, 3)
        self.assertEqual(len(result.generated_scenes), 3)
        self.assertIn("CHAPTER 1: Intro", result.generated_code)
        self.assertIn("code 1", result.generated_code)
        self.assertIn("CHAPTER 3: End", result.generated_code)
        self.assertIn("code 3", result.generated_code)

    def test_multi_scene_decomposition_failure(self):
        """Test that the pipeline handles decomposition failure gracefully."""
        self.pipeline._code_generator.decompose_concept.side_effect = Exception("LLM OOM")
        
        result = self.pipeline.run("test concept")
        
        self.assertEqual(result.generated_code, "")
        self.assertIsNone(result.reward)
        self.assertEqual(self.pipeline._reward_model.score.call_count, 0)

if __name__ == "__main__":
    unittest.main()
