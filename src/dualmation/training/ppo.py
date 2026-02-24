"""
PPO Trainer implementation for DualAnimate LLM.

Uses the `trl` library to perform Proximal Policy Optimization
on the Manim code generation model based on rewards from the RewardModel.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from trl.experimental.ppo import PPOTrainer, PPOConfig
from trl import AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer

from dualmation.experiment.config import TrainingConfig

logger = logging.getLogger(__name__)


class DualAnimatePPOTrainer:
    """Wrapper for TRL's PPOTrainer tailored for DualAnimate.

    Args:
        config: Training configuration.
        model_name: Name of the LLM to train.
        tokenizer: Tokenizer for the LLM.
        device: Compute device.
    """

    def __init__(
        self,
        config: TrainingConfig,
        model_name: str,
        tokenizer: AutoTokenizer | None = None,
        device: str = "auto",
    ) -> None:
        self.config = config
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model with value head for PPO
        # In a real scenario, we might want to load in 8-bit or use PEFT (LoRA)
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name,
            device_map=None, # Disable device_map for tests/CPU
            torch_dtype=torch.float32,
        )
        
        # Reference model for KL divergence (cloned from model at start)
        self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name,
            device_map=None,
            torch_dtype=torch.float32,
        )

        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # PPO Configuration
        ppo_config = PPOConfig(
            learning_rate=config.learning_rate,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=1,
            seed=42,
            use_cpu=(self.device == "cpu"),
        )

        self.ppo_trainer = PPOTrainer(
            ppo_config,
            self.model,
            self.ref_model,
            self.tokenizer,
        )

    def train_step(
        self,
        query_tensors: list[torch.Tensor],
        response_tensors: list[torch.Tensor],
        rewards: list[torch.Tensor],
    ) -> dict[str, Any]:
        """Perform a single PPO optimization step.

        Args:
            query_tensors: List of input (concept) tokens.
            response_tensors: List of output (generated code) tokens.
            rewards: List of scalar rewards for each (query, response) pair.

        Returns:
            Dictionary of training statistics.
        """
        # PPO step
        stats = self.ppo_trainer.step(query_tensors, response_tensors, rewards)
        
        # Log basic stats
        logger.info(
            "PPO Step: Loss=%.4f, KL=%.4f, Reward=%.4f",
            stats.get("ppo/loss/total", 0),
            stats.get("ppo/policy/approx_kl", 0),
            torch.stack(rewards).mean().item(),
        )
        
        return stats

    def save_checkpoint(self, path: str | Path) -> None:
        """Save the model and tokenizer."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.ppo_trainer.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info("Saved PPO checkpoint to %s", path)
