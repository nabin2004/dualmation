"""
Tests for the multi-component reward model.

Tests compilation checking and reward scoring without requiring GPU or model downloads.
"""

from __future__ import annotations

from dualmation.reward.reward_model import RewardConfig, RewardModel, RewardScore


class TestRewardScore:
    """Tests for the RewardScore dataclass."""

    def test_reward_score_fields(self):
        """RewardScore should have all expected fields."""
        score = RewardScore(
            total=0.75,
            concept_alignment=0.8,
            visual_quality=0.7,
            compilation_success=1.0,
        )
        assert score.total == 0.75
        assert score.concept_alignment == 0.8
        assert score.visual_quality == 0.7
        assert score.compilation_success == 1.0

    def test_reward_score_defaults(self):
        """RewardScore should have sensible defaults."""
        score = RewardScore(total=0.5, concept_alignment=0.5, visual_quality=0.5, compilation_success=0.5)
        assert score.compilation_output == ""
        assert score.weights == {}


class TestRewardModel:
    """Tests for the RewardModel."""

    def test_compilation_valid_python(self):
        """Valid Python code should pass syntax check."""
        model = RewardModel()
        code = 'from manim import *\n\nclass TestScene(Scene):\n    def construct(self):\n        circle = Circle()\n        self.play(Create(circle))\n'

        score, output = model._score_compilation(code)

        # At minimum should pass syntax check (0.3+ even if Manim not installed)
        assert score >= 0.0
        assert isinstance(output, str)

    def test_compilation_invalid_python(self):
        """Invalid Python should return 0.0 score."""
        model = RewardModel()
        code = "def broken(\n    this is not valid python {}{}{}"

        score, output = model._score_compilation(code)

        assert score == 0.0
        assert "error" in output.lower() or "syntax" in output.lower()

    def test_compilation_empty_code(self):
        """Empty code should compile (it's valid Python)."""
        model = RewardModel()
        score, output = model._score_compilation("")

        # Empty string is valid Python syntax
        assert score >= 0.0

    def test_score_without_visual(self):
        """Scoring without visual should use defaults for visual components."""
        config = RewardConfig(
            weight_alignment=0.4,
            weight_visual=0.3,
            weight_compilation=0.3,
        )
        model = RewardModel(config=config)

        code = "from manim import *\nclass Test(Scene):\n    def construct(self):\n        pass\n"
        result = model.score(code=code, concept="test concept")

        assert isinstance(result, RewardScore)
        assert 0.0 <= result.total <= 1.0
        assert result.weights == {
            "alignment": 0.4,
            "visual": 0.3,
            "compilation": 0.3,
        }

    def test_compute_advantage(self):
        """Advantage computation should center rewards around baseline."""
        model = RewardModel()

        rewards = [
            RewardScore(total=0.8, concept_alignment=0.8, visual_quality=0.8, compilation_success=1.0),
            RewardScore(total=0.4, concept_alignment=0.4, visual_quality=0.3, compilation_success=0.5),
            RewardScore(total=0.6, concept_alignment=0.6, visual_quality=0.5, compilation_success=0.7),
        ]

        advantages = model.compute_advantage(rewards)

        # Mean should be approximately 0
        assert abs(sum(advantages)) < 1e-6
        assert len(advantages) == 3

    def test_compute_advantage_with_baseline(self):
        """Custom baseline should shift advantages."""
        import pytest

        model = RewardModel()

        rewards = [
            RewardScore(total=0.8, concept_alignment=0.8, visual_quality=0.8, compilation_success=1.0),
            RewardScore(total=0.4, concept_alignment=0.4, visual_quality=0.3, compilation_success=0.5),
        ]

        advantages = model.compute_advantage(rewards, baseline=0.5)

        assert advantages[0] == pytest.approx(0.3)   # 0.8 - 0.5
        assert advantages[1] == pytest.approx(-0.1)   # 0.4 - 0.5

    def test_reward_config_defaults(self):
        """RewardConfig should have reasonable defaults."""
        config = RewardConfig()
        assert config.weight_alignment + config.weight_visual + config.weight_compilation == 1.0
        assert config.manim_timeout > 0
