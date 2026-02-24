"""
Multi-backend experiment tracker.

Supports simultaneous logging to:
- TensorBoard (for training curves, histograms, images)
- Weights & Biases (for cloud experiment tracking — optional)
- CSV/JSON files (for paper-ready data export)

The tracker uses a unified API so all backends receive the same metrics.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TrackerConfig:
    """Configuration for the experiment tracker.

    Attributes:
        experiment_name: Human-readable experiment name (used in all backends).
        run_name: Unique run identifier. Auto-generated if None.
        base_dir: Root directory for all experiment artifacts.
        use_tensorboard: Enable TensorBoard logging.
        use_wandb: Enable Weights & Biases logging.
        use_csv: Enable CSV metric export.
        use_json: Enable JSON metric export.
        wandb_project: W&B project name.
        wandb_entity: W&B team/user entity.
        wandb_tags: Tags for the W&B run.
        log_interval: Log metrics every N steps to reduce I/O.
    """

    experiment_name: str = "dualmation"
    run_name: str | None = None
    base_dir: str = "experiments"
    use_tensorboard: bool = True
    use_wandb: bool = False
    use_csv: bool = True
    use_json: bool = True
    wandb_project: str = "dualmation"
    wandb_entity: str | None = None
    wandb_tags: list[str] = field(default_factory=list)
    log_interval: int = 1


class ExperimentTracker:
    """Unified multi-backend experiment tracker for research workflows.

    Provides a single API that simultaneously logs to TensorBoard, W&B,
    and local CSV/JSON files. Designed for reproducibility and easy
    inclusion of results in research papers.

    Usage:
        ```python
        tracker = ExperimentTracker(TrackerConfig(experiment_name="ablation_v1"))
        tracker.start()

        for step in range(1000):
            loss = train_step()
            tracker.log_scalar("train/loss", loss, step)
            tracker.log_scalars("train", {"lr": lr, "grad_norm": gn}, step)

        tracker.finish()
        ```

    Args:
        config: Tracker configuration.
    """

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self._step = 0
        self._start_time: float | None = None
        self._tb_writer = None
        self._wandb_run = None
        self._csv_writers: dict[str, csv.DictWriter] = {}
        self._csv_files: dict[str, Any] = {}
        self._json_metrics: list[dict[str, Any]] = []
        self._run_dir: Path | None = None
        self._is_active = False

    @property
    def run_dir(self) -> Path:
        """Directory for this specific run's artifacts."""
        if self._run_dir is None:
            raise RuntimeError("Tracker not started. Call tracker.start() first.")
        return self._run_dir

    def start(self) -> ExperimentTracker:
        """Initialize all backends and create the run directory.

        Returns:
            self (for method chaining).
        """
        # Generate run name
        if self.config.run_name is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            self.config.run_name = f"{self.config.experiment_name}_{timestamp}"

        # Create run directory
        self._run_dir = Path(self.config.base_dir) / self.config.run_name
        self._run_dir.mkdir(parents=True, exist_ok=True)
        (self._run_dir / "checkpoints").mkdir(exist_ok=True)
        (self._run_dir / "figures").mkdir(exist_ok=True)
        (self._run_dir / "exports").mkdir(exist_ok=True)

        self._start_time = time.time()
        self._is_active = True

        # Initialize TensorBoard
        if self.config.use_tensorboard:
            self._init_tensorboard()

        # Initialize W&B
        if self.config.use_wandb:
            self._init_wandb()

        logger.info("Experiment started: %s → %s", self.config.run_name, self._run_dir)
        return self

    def _init_tensorboard(self) -> None:
        """Initialize TensorBoard SummaryWriter."""
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_dir = self._run_dir / "tensorboard"
            self._tb_writer = SummaryWriter(log_dir=str(tb_dir))
            logger.info("TensorBoard: %s", tb_dir)
        except ImportError:
            logger.warning("TensorBoard not installed. pip install tensorboard")
            self.config.use_tensorboard = False

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases run."""
        try:
            import wandb

            self._wandb_run = wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=self.config.run_name,
                tags=self.config.wandb_tags,
                dir=str(self._run_dir),
                reinit=True,
            )
            logger.info("W&B run: %s", self._wandb_run.url)
        except ImportError:
            logger.warning("wandb not installed. pip install wandb")
            self.config.use_wandb = False

    # ── Scalar Logging ──────────────────────────────────────────────

    def log_scalar(self, tag: str, value: float, step: int | None = None) -> None:
        """Log a single scalar metric to all backends.

        Args:
            tag: Metric name (e.g., "train/loss", "eval/accuracy").
            value: Scalar value.
            step: Global step. Auto-incremented if None.
        """
        if step is None:
            step = self._step
            self._step += 1

        # TensorBoard
        if self._tb_writer is not None:
            self._tb_writer.add_scalar(tag, value, step)

        # W&B
        if self._wandb_run is not None:
            import wandb
            wandb.log({tag: value}, step=step)

        # JSON buffer
        if self.config.use_json:
            self._json_metrics.append({
                "step": step,
                "tag": tag,
                "value": value,
                "timestamp": time.time(),
            })

        # CSV
        if self.config.use_csv:
            self._write_csv(tag, {"step": step, "value": value})

    def log_scalars(
        self, main_tag: str, scalars: dict[str, float], step: int | None = None
    ) -> None:
        """Log multiple related scalars at once.

        Args:
            main_tag: Group name (e.g., "train", "reward").
            scalars: Dictionary of metric_name → value.
            step: Global step.
        """
        if step is None:
            step = self._step
            self._step += 1

        for name, value in scalars.items():
            full_tag = f"{main_tag}/{name}"
            self.log_scalar(full_tag, value, step)

    # ── Hyperparameter Logging ──────────────────────────────────────

    def log_hyperparams(self, hparams: dict[str, Any], metrics: dict[str, float] | None = None) -> None:
        """Log hyperparameters and their associated metrics.

        Essential for experiment comparison in TensorBoard's HParams plugin
        and for W&B sweep analysis.

        Args:
            hparams: Dictionary of hyperparameter names and values.
            metrics: Optional final metrics associated with these hparams.
        """
        if self._tb_writer is not None:
            from torch.utils.tensorboard.summary import hparams as tb_hparams
            # TensorBoard HParams
            self._tb_writer.add_hparams(
                hparam_dict=hparams,
                metric_dict=metrics or {},
            )

        if self._wandb_run is not None:
            import wandb
            wandb.config.update(hparams)
            if metrics:
                wandb.log(metrics)

        # Save to JSON for paper reference
        hparams_path = self.run_dir / "hyperparams.json"
        data = {"hparams": hparams, "metrics": metrics or {}}
        with open(hparams_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info("Hyperparams logged: %d params, %d metrics", len(hparams), len(metrics or {}))

    # ── Image / Figure Logging ──────────────────────────────────────

    def log_image(self, tag: str, image: Any, step: int | None = None) -> None:
        """Log an image to TensorBoard and W&B.

        Args:
            tag: Image tag name.
            image: PIL Image, numpy array, or torch Tensor.
            step: Global step.
        """
        import numpy as np

        if step is None:
            step = self._step

        # Convert PIL to numpy if needed
        if hasattr(image, "convert"):  # PIL Image
            img_np = np.array(image)
        elif hasattr(image, "numpy"):  # torch Tensor
            img_np = image.detach().cpu().numpy()
        else:
            img_np = np.array(image)

        # Ensure HWC → CHW for TensorBoard
        if self._tb_writer is not None:
            if img_np.ndim == 3 and img_np.shape[-1] in (1, 3, 4):
                tb_img = np.transpose(img_np, (2, 0, 1))  # HWC → CHW
            else:
                tb_img = img_np
            self._tb_writer.add_image(tag, tb_img, step)

        if self._wandb_run is not None:
            import wandb
            wandb.log({tag: wandb.Image(img_np)}, step=step)

    def log_figure(self, tag: str, figure: Any, step: int | None = None, save: bool = True) -> Path | None:
        """Log a matplotlib figure to all backends and optionally save to disk.

        Args:
            tag: Figure tag name.
            figure: matplotlib Figure object.
            step: Global step.
            save: If True, save as PNG and PDF (for papers).

        Returns:
            Path to saved figure, or None.
        """
        if step is None:
            step = self._step

        if self._tb_writer is not None:
            self._tb_writer.add_figure(tag, figure, step)

        if self._wandb_run is not None:
            import wandb
            wandb.log({tag: wandb.Image(figure)}, step=step)

        saved_path = None
        if save and self._run_dir:
            fig_dir = self._run_dir / "figures"
            # Save both PNG (for quick review) and PDF (for papers)
            safe_tag = tag.replace("/", "_").replace(" ", "_")
            png_path = fig_dir / f"{safe_tag}_step{step}.png"
            pdf_path = fig_dir / f"{safe_tag}_step{step}.pdf"
            figure.savefig(png_path, dpi=300, bbox_inches="tight")
            figure.savefig(pdf_path, bbox_inches="tight")
            saved_path = png_path
            logger.info("Figure saved: %s (.png + .pdf)", safe_tag)

        return saved_path

    # ── Text / Artifact Logging ─────────────────────────────────────

    def log_text(self, tag: str, text: str, step: int | None = None) -> None:
        """Log text content (e.g., generated code, prompts).

        Args:
            tag: Text tag name.
            text: Text content.
            step: Global step.
        """
        if step is None:
            step = self._step

        if self._tb_writer is not None:
            self._tb_writer.add_text(tag, text, step)

        if self._wandb_run is not None:
            import wandb
            wandb.log({tag: wandb.Html(f"<pre>{text}</pre>")}, step=step)

    def log_artifact(self, name: str, path: str | Path, artifact_type: str = "model") -> None:
        """Log a file artifact (model checkpoint, dataset, etc.).

        Args:
            name: Artifact name.
            path: Path to the artifact file or directory.
            artifact_type: Type of artifact (model, dataset, result).
        """
        if self._wandb_run is not None:
            import wandb
            artifact = wandb.Artifact(name, type=artifact_type)
            path = Path(path)
            if path.is_dir():
                artifact.add_dir(str(path))
            else:
                artifact.add_file(str(path))
            self._wandb_run.log_artifact(artifact)
            logger.info("W&B artifact logged: %s", name)

    # ── Histogram / Distribution Logging ────────────────────────────

    def log_histogram(self, tag: str, values: Any, step: int | None = None, bins: int = 64) -> None:
        """Log a histogram of values (weights, activations, gradients).

        Args:
            tag: Histogram tag.
            values: Array-like of values.
            step: Global step.
            bins: Number of bins.
        """
        if step is None:
            step = self._step

        if self._tb_writer is not None:
            import torch
            if hasattr(values, "detach"):
                values = values.detach().cpu()
            self._tb_writer.add_histogram(tag, values, step, bins=bins)

    # ── CSV Export ──────────────────────────────────────────────────

    def _write_csv(self, tag: str, row: dict[str, Any]) -> None:
        """Write a row to a tag-specific CSV file."""
        safe_tag = tag.replace("/", "_")
        csv_path = self.run_dir / "exports" / f"{safe_tag}.csv"

        if safe_tag not in self._csv_writers:
            file_exists = csv_path.exists()
            f = open(csv_path, "a", newline="")
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            self._csv_writers[safe_tag] = writer
            self._csv_files[safe_tag] = f

        self._csv_writers[safe_tag].writerow(row)

    # ── Timer Context Manager ──────────────────────────────────────

    @contextmanager
    def timer(self, name: str, step: int | None = None):
        """Context manager for timing code blocks and logging duration.

        Usage:
            ```python
            with tracker.timer("train/epoch_time", step=epoch):
                train_one_epoch()
            ```
        """
        start = time.time()
        yield
        elapsed = time.time() - start
        self.log_scalar(f"timing/{name}", elapsed, step)

    # ── Summary & Export ────────────────────────────────────────────

    def export_metrics_json(self, path: str | Path | None = None) -> Path:
        """Export all buffered metrics to a single JSON file.

        Args:
            path: Output path. Defaults to run_dir/exports/all_metrics.json.

        Returns:
            Path to the exported JSON file.
        """
        if path is None:
            path = self.run_dir / "exports" / "all_metrics.json"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self._json_metrics, f, indent=2)

        logger.info("Exported %d metric entries to %s", len(self._json_metrics), path)
        return path

    def export_summary(self) -> dict[str, Any]:
        """Generate a run summary with key statistics.

        Returns:
            Dictionary with run metadata and aggregated metrics.
        """
        elapsed = time.time() - self._start_time if self._start_time else 0

        # Aggregate metrics by tag
        aggregated: dict[str, list[float]] = {}
        for entry in self._json_metrics:
            tag = entry["tag"]
            if tag not in aggregated:
                aggregated[tag] = []
            aggregated[tag].append(entry["value"])

        summary_stats = {}
        for tag, values in aggregated.items():
            summary_stats[tag] = {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "last": values[-1],
                "count": len(values),
            }

        summary = {
            "run_name": self.config.run_name,
            "experiment_name": self.config.experiment_name,
            "duration_seconds": elapsed,
            "total_steps": self._step,
            "metrics_summary": summary_stats,
        }

        # Save summary
        summary_path = self.run_dir / "run_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        return summary

    # ── Lifecycle ───────────────────────────────────────────────────

    def finish(self) -> dict[str, Any]:
        """Finalize the experiment run. Flushes all backends and exports data.

        Returns:
            Run summary dictionary.
        """
        summary = self.export_summary()
        self.export_metrics_json()

        # Flush TensorBoard
        if self._tb_writer is not None:
            self._tb_writer.flush()
            self._tb_writer.close()
            self._tb_writer = None

        # Finish W&B
        if self._wandb_run is not None:
            import wandb
            wandb.finish()
            self._wandb_run = None

        # Close CSV files
        for f in self._csv_files.values():
            f.close()
        self._csv_writers.clear()
        self._csv_files.clear()

        self._is_active = False
        logger.info("Experiment finished: %s (%.1fs)", self.config.run_name, summary["duration_seconds"])
        return summary

    def __enter__(self) -> ExperimentTracker:
        return self.start()

    def __exit__(self, *args) -> None:
        if self._is_active:
            self.finish()
