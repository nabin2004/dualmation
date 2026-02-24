"""
Group Relative Policy Optimization (GRPO) Trainer for DualAnimate.

GRPO improves stability in logic-heavy tasks like Manim code generation
by comparing multiple completions for the same prompt and using
relative rewards, eliminating the need for a separate value head.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import torch
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

from dualmation.reward.reward_model import RewardModel, RewardConfig

logger = logging.getLogger(__name__)


def create_reward_functions(reward_model: RewardModel) -> list[Callable]:
    """Create a list of reward functions for GRPOTrainer from a RewardModel.
    
    The functional interface for trl.GRPOTrainer reward functions is:
    def reward_func(prompts: list[str], completions: list[str], **kwargs) -> list[float]
    """
    
    def compilation_reward(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        scores = []
        for code in completions:
            # We use a dummy visual and concept for the compile check component
            res = reward_model.score(code=code, concept="", visual=None)
            scores.append(res.compilation_success)
        return scores

    def alignment_reward(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        scores = []
        for prompt, code in zip(prompts, completions):
            # Alignment usually requires visual, so here we score concept-code alignment if possible
            # or use the reward model's internal alignment logic
            res = reward_model.score(code=code, concept=prompt, visual=None)
            scores.append(res.concept_alignment)
        return scores
        
    return [compilation_reward, alignment_reward]


class DualAnimateGRPOTrainer:
    """Wrapper for TRL's GRPOTrainer tailored for DualAnimate.

    Args:
        model_name: Name of the LLM to train.
        reward_config: Configuration for the RewardModel.
        training_config: Configuration for training.
        tokenizer: Optional tokenizer.
        device: Compute device.
    """

    def __init__(
        self,
        model_name: str,
        reward_config: RewardConfig,
        training_config: Any, # ExperimentConfig.training
        tokenizer: Any = None,
        device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model (GRPO usually doesn't need CausalLMWithValueHead)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize Reward Model and adapters
        self.reward_model = RewardModel(reward_config, device=self.device)
        self.reward_funcs = create_reward_functions(self.reward_model)

        num_generations = getattr(training_config, "num_generations", 8)

        # GRPO Configuration
        self.grpo_config = GRPOConfig(
            output_dir=training_config.output_dir if hasattr(training_config, 'output_dir') else "experiments/grpo",
            learning_rate=training_config.learning_rate,
            per_device_train_batch_size=training_config.batch_size,
            gradient_accumulation_steps=1,
            num_train_epochs=training_config.num_epochs,
            # GRPO Specifics
            num_generations=num_generations,
            max_completion_length=1024,
            use_vllm=False,
            use_cpu=(self.device == "cpu"),
        )

    def train(self, dataset: Any) -> Any:
        """Execute the GRPO training process.
        
        Args:
            dataset: A dataset (e.g., HuggingFace Dataset) containing 'prompt' column.
        """
        trainer = GRPOTrainer(
            model=self.model,
            reward_funcs=self.reward_funcs,
            args=self.grpo_config,
            train_dataset=dataset,
            processing_class=self.tokenizer,
        )
        
        logger.info("Starting GRPO training...")
        return trainer.train()
