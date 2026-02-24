"""
Multi-component reward model for RL-based pipeline feedback.

Scores generated animations on three axes:
1. Concept Alignment — cosine similarity between concept embedding and visual output
2. Visual Quality — CLIP-based aesthetic scoring
3. Compilation Success — whether the generated Manim code compiles and renders
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class RewardScore:
    """Composite reward score from all components.

    Attributes:
        total: Weighted composite score in [0, 1].
        concept_alignment: Cosine similarity between concept and output embeddings.
        visual_quality: CLIP-based aesthetic quality score.
        compilation_success: 1.0 if Manim code compiles, 0.0 otherwise.
        compilation_output: stdout/stderr from Manim compilation attempt.
        weights: Weights used for each component.
    """

    total: float
    concept_alignment: float
    visual_quality: float
    compilation_success: float
    compilation_output: str = ""
    weights: dict[str, float] = field(default_factory=dict)


@dataclass
class RewardConfig:
    """Configuration for the reward model."""

    weight_alignment: float = 0.4
    weight_visual: float = 0.3
    weight_compilation: float = 0.3
    manim_timeout: int = 60  # seconds
    clip_model_name: str = "openai/clip-vit-base-patch32"


class RewardModel:
    """Multi-component reward model for scoring generated animations.

    Combines concept alignment, visual quality, and compilation success
    into a single scalar reward for RL training (PPO/GRPO).

    Args:
        config: Reward configuration.
        device: Compute device.
    """

    def __init__(
        self,
        config: RewardConfig | None = None,
        device: str | None = None,
    ) -> None:
        self.config = config or RewardConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._clip_model = None
        self._clip_processor = None

    def _load_clip(self) -> None:
        """Lazy-load CLIP model for visual quality scoring."""
        if self._clip_model is not None:
            return

        from transformers import CLIPModel, CLIPProcessor

        logger.info("Loading CLIP model: %s", self.config.clip_model_name)
        self._clip_processor = CLIPProcessor.from_pretrained(self.config.clip_model_name)
        self._clip_model = CLIPModel.from_pretrained(self.config.clip_model_name).to(self.device)
        self._clip_model.eval()

    def score(
        self,
        code: str,
        visual: Image.Image | None = None,
        concept: str = "",
        concept_embedding: torch.Tensor | None = None,
        output_embedding: torch.Tensor | None = None,
    ) -> RewardScore:
        """Compute composite reward for a generation result.

        Args:
            code: Generated Manim Python code.
            visual: Optional generated/rendered visual output.
            concept: Original concept description (for CLIP scoring).
            concept_embedding: Optional pre-computed concept embedding.
            output_embedding: Optional pre-computed output embedding.

        Returns:
            RewardScore with component and composite scores.
        """
        # 1. Concept alignment
        alignment = self._score_alignment(concept, visual, concept_embedding, output_embedding)

        # 2. Visual quality (CLIP aesthetic score)
        quality = self._score_visual_quality(concept, visual)

        # 3. Compilation success
        compilation, comp_output = self._score_compilation(code)

        # Weighted composite
        weights = {
            "alignment": self.config.weight_alignment,
            "visual": self.config.weight_visual,
            "compilation": self.config.weight_compilation,
        }
        total = (
            weights["alignment"] * alignment
            + weights["visual"] * quality
            + weights["compilation"] * compilation
        )

        return RewardScore(
            total=total,
            concept_alignment=alignment,
            visual_quality=quality,
            compilation_success=compilation,
            compilation_output=comp_output,
            weights=weights,
        )

    def _score_alignment(
        self,
        concept: str,
        visual: Image.Image | None,
        concept_emb: torch.Tensor | None,
        output_emb: torch.Tensor | None,
    ) -> float:
        """Score concept-output alignment via cosine similarity.

        Uses pre-computed embeddings if available, otherwise falls back to CLIP.
        """
        # If we have pre-computed embeddings, use direct cosine similarity
        if concept_emb is not None and output_emb is not None:
            sim = F.cosine_similarity(concept_emb.flatten(), output_emb.flatten(), dim=0)
            return max(0.0, sim.item())

        # Fallback: use CLIP text-image similarity
        if visual is not None and concept:
            return self._clip_similarity(concept, visual)

        return 0.5  # Default neutral score if no scoring method available

    @torch.inference_mode()
    def _clip_similarity(self, text: str, image: Image.Image) -> float:
        """Compute CLIP text-image similarity score."""
        self._load_clip()

        inputs = self._clip_processor(
            text=[text], images=[image], return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self._clip_model(**inputs)
        # Normalized to [0, 1]
        logits = outputs.logits_per_image.squeeze()
        score = torch.sigmoid(logits / 100.0)  # Scale and sigmoid
        return score.item()

    @torch.inference_mode()
    def _score_visual_quality(self, concept: str, visual: Image.Image | None) -> float:
        """Score visual quality using CLIP alignment with quality descriptors."""
        if visual is None:
            return 0.0

        self._load_clip()

        quality_prompts = [
            "a high quality, professional educational animation",
            "a beautiful, clear, well-designed mathematical visualization",
            "an aesthetic, modern, visually appealing digital illustration",
        ]

        scores = []
        for prompt in quality_prompts:
            score = self._clip_similarity(prompt, visual)
            scores.append(score)

        return sum(scores) / len(scores)

    def _score_compilation(self, code: str) -> tuple[float, str]:
        """Test whether the generated Manim code compiles successfully.

        Writes the code to a temp file and attempts to render it with Manim.
        Returns 1.0 for success, 0.0 for failure.

        Args:
            code: Manim Python code to test.

        Returns:
            Tuple of (score, compilation_output).
        """
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, prefix="manim_test_"
        ) as f:
            f.write(code)
            temp_path = f.name

        try:
            # Dry run: just check if the code is valid Python (syntax check)
            result = subprocess.run(
                ["python", "-c", f"import ast; ast.parse(open('{temp_path}').read())"],
                capture_output=True,
                text=True,
                timeout=self.config.manim_timeout,
            )

            if result.returncode == 0:
                # Try Manim render (dry-run with -ql for low quality, fast)
                manim_result = subprocess.run(
                    ["python", "-m", "manim", "render", "-ql", "--dry_run", temp_path],
                    capture_output=True,
                    text=True,
                    timeout=self.config.manim_timeout,
                )
                if manim_result.returncode == 0:
                    return 1.0, "Compilation successful"
                else:
                    return 0.3, f"Syntax OK but Manim render failed: {manim_result.stderr[:500]}"
            else:
                return 0.0, f"Syntax error: {result.stderr[:500]}"

        except subprocess.TimeoutExpired:
            return 0.1, "Compilation timed out"
        except FileNotFoundError:
            # Manim not installed — just do syntax check
            logger.warning("Manim not found, falling back to syntax check only")
            try:
                compile(code, "<string>", "exec")
                return 0.7, "Syntax valid (Manim not available for full check)"
            except SyntaxError as e:
                return 0.0, f"Syntax error: {e}"
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def compute_advantage(
        self, rewards: list[RewardScore], baseline: float | None = None
    ) -> list[float]:
        """Compute advantages for PPO training.

        Args:
            rewards: List of RewardScores from a batch.
            baseline: Optional baseline value. If None, uses batch mean.

        Returns:
            List of advantage values (reward - baseline).
        """
        totals = [r.total for r in rewards]
        if baseline is None:
            baseline = sum(totals) / len(totals)
        return [t - baseline for t in totals]
