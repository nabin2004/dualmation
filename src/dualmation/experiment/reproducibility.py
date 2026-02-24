"""
Reproducibility utilities for DualAnimate experiments.

Provides deterministic seed setting, system info capture, and
environment snapshotting for fully reproducible experiments.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set all random seeds for full reproducibility.

    Sets seeds for: Python random, NumPy, PyTorch (CPU + CUDA),
    and enables deterministic algorithms where possible.

    Args:
        seed: Integer seed value.
        deterministic: If True, enable deterministic CUDA algorithms
            (may reduce performance slightly).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # PyTorch 2.0+ deterministic mode
            if hasattr(torch, "use_deterministic_algorithms"):
                try:
                    torch.use_deterministic_algorithms(True, warn_only=True)
                except Exception:
                    pass
    except ImportError:
        pass

    logger.info("Random seed set: %d (deterministic=%s)", seed, deterministic)


def get_system_info() -> dict[str, Any]:
    """Capture comprehensive system information for reproducibility.

    Returns a dictionary with Python version, OS, GPU info, package versions,
    and git state — everything needed to reproduce an experiment environment.

    Returns:
        Dictionary with system info suitable for JSON serialization.
    """
    info: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": sys.platform,
        },
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
    }

    # GPU info
    try:
        import torch
        info["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        if torch.cuda.is_available():
            gpus = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpus.append({
                    "name": props.name,
                    "total_memory_gb": round(props.total_mem / 1e9, 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                })
            info["torch"]["gpus"] = gpus
    except ImportError:
        info["torch"] = {"installed": False}

    # Key package versions
    packages = [
        "transformers", "diffusers", "accelerate", "manim",
        "numpy", "Pillow", "trl", "omegaconf", "tensorboard", "wandb",
    ]
    info["packages"] = {}
    for pkg in packages:
        try:
            mod = __import__(pkg)
            info["packages"][pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            info["packages"][pkg] = "not_installed"

    # Git state
    info["git"] = _get_git_info()

    return info


def _get_git_info() -> dict[str, str | None]:
    """Capture current git state (commit hash, branch, dirty status)."""
    git_info: dict[str, str | None] = {
        "commit": None,
        "branch": None,
        "dirty": None,
        "remote": None,
    }

    try:
        git_info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        git_info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        git_info["dirty"] = "yes" if status else "no"
        git_info["remote"] = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return git_info


def save_environment_snapshot(run_dir: str | Path) -> Path:
    """Save a complete environment snapshot for reproducibility.

    Includes system info, pip freeze output, and a hash of the environment
    for easy comparison between experiments.

    Args:
        run_dir: Directory to save the snapshot in.

    Returns:
        Path to the snapshot directory.
    """
    run_dir = Path(run_dir)
    env_dir = run_dir / "environment"
    env_dir.mkdir(parents=True, exist_ok=True)

    # System info
    sys_info = get_system_info()
    with open(env_dir / "system_info.json", "w") as f:
        json.dump(sys_info, f, indent=2, default=str)

    # pip freeze
    try:
        freeze_output = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            stderr=subprocess.DEVNULL, text=True
        )
        (env_dir / "requirements_frozen.txt").write_text(freeze_output)

        # Environment hash (for quick comparison)
        env_hash = hashlib.sha256(freeze_output.encode()).hexdigest()[:12]
        (env_dir / "env_hash.txt").write_text(env_hash)
    except subprocess.CalledProcessError:
        pass

    logger.info("Environment snapshot saved: %s", env_dir)
    return env_dir
