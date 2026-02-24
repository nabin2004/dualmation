"""
OmegaConf-based experiment configuration management.

Provides hierarchical, merge-able configuration for reproducible experiments.
Supports YAML config files, CLI overrides, and automatic config snapshotting.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for the embedding module."""
    code_model: str = "microsoft/codebert-base"
    visual_model: str = "google/vit-base-patch16-224"
    embedding_dim: int = 512
    freeze_backbone: bool = True
    contrastive_temperature: float = 0.07


@dataclass
class LLMConfig:
    """Configuration for the LLM code generation module."""
    model_name: str = "codellama/CodeLlama-7b-hf"
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    load_in_8bit: bool = False


@dataclass
class LoRAConfig:
    """Configuration for Low-Rank Adaptation (LoRA)."""
    r: int = 8
    alpha: int = 16
    target_modules: list[str] = field(default_factory=lambda: ["to_q", "to_k", "to_v", "to_out.0"])
    dropout: float = 0.05
    bias: str = "none"


@dataclass
class DiffusionConfig:
    """Configuration for the diffusion visual generation module."""
    model_name: str = "stabilityai/stable-diffusion-2-1"
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    width: int = 1920
    height: int = 1080
    use_lora: bool = False
    lora: LoRAConfig = field(default_factory=LoRAConfig)


@dataclass
class CompositorConfig:
    """Configuration for the compositor."""
    output_width: int = 1920
    output_height: int = 1080
    blend_mode: str = "alpha"
    background_opacity: float = 1.0
    foreground_opacity: float = 1.0


@dataclass
class RewardModelConfig:
    """Configuration for the reward model."""
    weight_alignment: float = 0.4
    weight_visual: float = 0.3
    weight_compilation: float = 0.3
    clip_model_name: str = "openai/clip-vit-base-patch32"
    manim_timeout: int = 60


@dataclass
class TrainingConfig:
    """Configuration for training."""
    learning_rate: float = 1e-4
    batch_size: int = 8
    num_epochs: int = 100
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    eval_interval: int = 100
    save_interval: int = 500
    rl_algorithm: str = "ppo"  # ppo | grpo
    rl_clip_range: float = 0.2
    rl_kl_coeff: float = 0.1


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration.

    Hierarchical config that can be loaded from YAML, overridden from CLI,
    and snapshotted for reproducibility.
    """
    # Experiment metadata
    experiment_name: str = "dualmation_default"
    run_name: str | None = None
    description: str = ""
    tags: list[str] = field(default_factory=list)
    seed: int = 42

    # Module configs
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    compositor: CompositorConfig = field(default_factory=CompositorConfig)
    reward: RewardModelConfig = field(default_factory=RewardModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Paths
    output_dir: str = "experiments"
    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints"

    # Tracking
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = "dualmation"
    wandb_entity: str | None = None

    # Compute
    device: str = "auto"  # auto | cuda | cpu
    num_workers: int = 4
    mixed_precision: bool = True


def load_config(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> ExperimentConfig:
    """Load experiment configuration from YAML file with optional overrides.

    Supports three levels of configuration (in priority order):
    1. Programmatic overrides (highest)
    2. YAML config file
    3. Dataclass defaults (lowest)

    Args:
        config_path: Path to a YAML config file.
        overrides: Dictionary of overrides (supports dot notation keys like
            "training.learning_rate").

    Returns:
        Merged ExperimentConfig.
    """
    config = ExperimentConfig()

    if config_path is not None:
        config_path = Path(config_path)
        if config_path.exists():
            try:
                from omegaconf import OmegaConf, DictConfig
                yaml_conf = OmegaConf.load(config_path)
                base_conf = OmegaConf.structured(config)
                merged = OmegaConf.merge(base_conf, yaml_conf)
                if overrides:
                    override_conf = OmegaConf.create(overrides)
                    merged = OmegaConf.merge(merged, override_conf)
                config = OmegaConf.to_object(merged)
                logger.info("Config loaded from: %s", config_path)
            except ImportError:
                logger.warning("OmegaConf not installed, loading YAML with PyYAML fallback")
                import yaml
                with open(config_path) as f:
                    yaml_data = yaml.safe_load(f)
                config = _merge_dict_into_config(config, yaml_data)
                if overrides:
                    config = _merge_dict_into_config(config, overrides)
        else:
            logger.warning("Config file not found: %s, using defaults", config_path)

    elif overrides:
        try:
            from omegaconf import OmegaConf
            base_conf = OmegaConf.structured(config)
            override_conf = OmegaConf.create(overrides)
            merged = OmegaConf.merge(base_conf, override_conf)
            config = OmegaConf.to_object(merged)
        except ImportError:
            config = _merge_dict_into_config(config, overrides)

    return config


def save_config(config: ExperimentConfig, path: str | Path) -> Path:
    """Save experiment configuration to YAML file.

    Args:
        config: Configuration to save.
        path: Output file path.

    Returns:
        Path to saved config file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from omegaconf import OmegaConf
        conf = OmegaConf.structured(config)
        with open(path, "w") as f:
            OmegaConf.save(conf, f)
    except ImportError:
        # Fallback: save as JSON
        path = path.with_suffix(".json")
        with open(path, "w") as f:
            json.dump(asdict(config), f, indent=2, default=str)

    logger.info("Config saved to: %s", path)
    return path


def snapshot_config(config: ExperimentConfig, run_dir: str | Path) -> Path:
    """Create a timestamped snapshot of the config for reproducibility.

    Saves both YAML and JSON formats, plus a human-readable summary.

    Args:
        config: Configuration to snapshot.
        run_dir: Run directory to save the snapshot in.

    Returns:
        Path to the snapshot directory.
    """
    run_dir = Path(run_dir)
    snapshot_dir = run_dir / "config_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    save_config(config, snapshot_dir / "config.yaml")

    # Also save as JSON for easy programmatic access
    json_path = snapshot_dir / "config.json"
    with open(json_path, "w") as f:
        json.dump(asdict(config), f, indent=2, default=str)

    # Save human-readable summary
    summary_path = snapshot_dir / "config_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Experiment: {config.experiment_name}\n")
        f.write(f"Run: {config.run_name}\n")
        f.write(f"Description: {config.description}\n")
        f.write(f"Seed: {config.seed}\n")
        f.write(f"Timestamp: {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"\n{'='*60}\n")
        f.write(json.dumps(asdict(config), indent=2, default=str))

    logger.info("Config snapshot saved: %s", snapshot_dir)
    return snapshot_dir


def _merge_dict_into_config(config: ExperimentConfig, data: dict[str, Any]) -> ExperimentConfig:
    """Recursively merge a dictionary into the config dataclass."""
    config_dict = asdict(config)

    def _deep_merge(base: dict, override: dict) -> dict:
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                _deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    merged = _deep_merge(config_dict, data)

    # Reconstruct from merged dict
    return ExperimentConfig(
        experiment_name=merged.get("experiment_name", config.experiment_name),
        run_name=merged.get("run_name", config.run_name),
        description=merged.get("description", config.description),
        tags=merged.get("tags", []),
        seed=merged.get("seed", config.seed),
        embedding=EmbeddingConfig(**merged.get("embedding", {})),
        llm=LLMConfig(**merged.get("llm", {})),
        diffusion=DiffusionConfig(**merged.get("diffusion", {})),
        compositor=CompositorConfig(**merged.get("compositor", {})),
        reward=RewardModelConfig(**merged.get("reward", {})),
        training=TrainingConfig(**merged.get("training", {})),
        output_dir=merged.get("output_dir", config.output_dir),
        data_dir=merged.get("data_dir", config.data_dir),
        checkpoint_dir=merged.get("checkpoint_dir", config.checkpoint_dir),
        use_tensorboard=merged.get("use_tensorboard", config.use_tensorboard),
        use_wandb=merged.get("use_wandb", config.use_wandb),
        wandb_project=merged.get("wandb_project", config.wandb_project),
        wandb_entity=merged.get("wandb_entity", config.wandb_entity),
        device=merged.get("device", config.device),
        num_workers=merged.get("num_workers", config.num_workers),
        mixed_precision=merged.get("mixed_precision", config.mixed_precision),
    )
