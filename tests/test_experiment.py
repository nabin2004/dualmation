"""
Tests for the experimentation framework.

Tests cover the experiment tracker, config management, reproducibility,
and metrics collection — all without requiring GPU or model downloads.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
import numpy as np

from dualmation.experiment.tracker import ExperimentTracker, TrackerConfig
from dualmation.experiment.config import (
    ExperimentConfig, load_config, save_config, snapshot_config,
    TrainingConfig, EmbeddingConfig,
)
from dualmation.experiment.reproducibility import set_seed, get_system_info
from dualmation.experiment.metrics import MetricsCollector


# ── Tracker Tests ───────────────────────────────────────────────


class TestExperimentTracker:
    """Tests for the multi-backend experiment tracker."""

    def test_tracker_lifecycle(self, tmp_path):
        """Tracker should create run dir, log metrics, and finish cleanly."""
        config = TrackerConfig(
            experiment_name="test",
            base_dir=str(tmp_path),
            use_tensorboard=False,
            use_wandb=False,
            use_csv=True,
            use_json=True,
        )
        tracker = ExperimentTracker(config)
        tracker.start()

        tracker.log_scalar("train/loss", 0.5, step=0)
        tracker.log_scalar("train/loss", 0.3, step=1)
        tracker.log_scalar("train/loss", 0.1, step=2)

        summary = tracker.finish()

        assert summary["total_steps"] >= 0
        assert "metrics_summary" in summary
        assert (tracker.run_dir / "run_summary.json").exists()
        assert (tracker.run_dir / "exports" / "all_metrics.json").exists()

    def test_tracker_context_manager(self, tmp_path):
        """Tracker should work as a context manager."""
        config = TrackerConfig(
            experiment_name="ctx_test",
            base_dir=str(tmp_path),
            use_tensorboard=False,
            use_wandb=False,
        )

        with ExperimentTracker(config) as tracker:
            tracker.log_scalar("test/metric", 1.0, step=0)
            assert tracker._is_active

        assert not tracker._is_active

    def test_log_scalars(self, tmp_path):
        """log_scalars should log multiple metrics under a group."""
        config = TrackerConfig(
            base_dir=str(tmp_path), use_tensorboard=False, use_wandb=False,
        )
        with ExperimentTracker(config) as tracker:
            tracker.log_scalars("reward", {"alignment": 0.8, "quality": 0.6, "compile": 1.0}, step=5)

        # Check JSON export has all three metrics
        json_path = tracker.run_dir / "exports" / "all_metrics.json"
        with open(json_path) as f:
            metrics = json.load(f)

        tags = {m["tag"] for m in metrics}
        assert "reward/alignment" in tags
        assert "reward/quality" in tags
        assert "reward/compile" in tags

    def test_log_hyperparams(self, tmp_path):
        """Hyperparameters should be saved to JSON."""
        config = TrackerConfig(
            base_dir=str(tmp_path), use_tensorboard=False, use_wandb=False,
        )
        with ExperimentTracker(config) as tracker:
            tracker.log_hyperparams(
                {"lr": 1e-4, "batch_size": 8},
                {"final_loss": 0.05},
            )

        hp_path = tracker.run_dir / "hyperparams.json"
        assert hp_path.exists()
        with open(hp_path) as f:
            data = json.load(f)
        assert data["hparams"]["lr"] == 1e-4
        assert data["metrics"]["final_loss"] == 0.05

    def test_timer(self, tmp_path):
        """Timer context manager should log elapsed time."""
        config = TrackerConfig(
            base_dir=str(tmp_path), use_tensorboard=False, use_wandb=False, use_json=True,
        )
        with ExperimentTracker(config) as tracker:
            with tracker.timer("test_op", step=0):
                _ = sum(range(1000))

        json_path = tracker.run_dir / "exports" / "all_metrics.json"
        with open(json_path) as f:
            metrics = json.load(f)
        assert any(m["tag"] == "timing/test_op" for m in metrics)

    def test_csv_export(self, tmp_path):
        """CSV files should be created per metric tag."""
        config = TrackerConfig(
            base_dir=str(tmp_path), use_tensorboard=False, use_wandb=False, use_csv=True,
        )
        with ExperimentTracker(config) as tracker:
            for i in range(5):
                tracker.log_scalar("train/loss", 1.0 / (i + 1), step=i)

        csv_path = tracker.run_dir / "exports" / "train_loss.csv"
        assert csv_path.exists()


# ── Config Tests ────────────────────────────────────────────────


class TestExperimentConfig:
    """Tests for the configuration management system."""

    def test_default_config(self):
        """Default config should have sensible values."""
        config = ExperimentConfig()
        assert config.seed == 42
        assert config.training.learning_rate == 1e-4
        assert config.embedding.embedding_dim == 512
        assert config.reward.weight_alignment + config.reward.weight_visual + config.reward.weight_compilation == pytest.approx(1.0)

    def test_config_with_overrides(self):
        """Config should accept programmatic overrides."""
        config = load_config(overrides={"seed": 123, "training": {"learning_rate": 3e-5}})
        assert config.seed == 123
        assert config.training.learning_rate == 3e-5

    def test_save_and_load_config(self, tmp_path):
        """Config should round-trip through save/load."""
        config = ExperimentConfig(experiment_name="roundtrip_test", seed=99)
        path = save_config(config, tmp_path / "config.yaml")

        loaded = load_config(path)
        assert loaded.experiment_name == "roundtrip_test"
        assert loaded.seed == 99

    def test_snapshot_config(self, tmp_path):
        """Config snapshot should create multiple files."""
        config = ExperimentConfig(experiment_name="snapshot_test")
        snapshot_dir = snapshot_config(config, tmp_path)

        assert (snapshot_dir / "config.json").exists()
        assert (snapshot_dir / "config_summary.txt").exists()


# ── Reproducibility Tests ───────────────────────────────────────


class TestReproducibility:
    """Tests for reproducibility utilities."""

    def test_set_seed_deterministic(self):
        """Setting the same seed should produce identical random sequences."""
        set_seed(42)
        seq1 = [random.random() for _ in range(10)]

        set_seed(42)
        seq2 = [random.random() for _ in range(10)]

        assert seq1 == seq2

    def test_set_seed_numpy(self):
        """Numpy should also be deterministic after set_seed."""
        set_seed(42)
        arr1 = np.random.rand(5).tolist()

        set_seed(42)
        arr2 = np.random.rand(5).tolist()

        assert arr1 == arr2

    def test_system_info(self):
        """System info should capture key fields."""
        info = get_system_info()
        assert "python" in info
        assert "os" in info
        assert "packages" in info
        assert "git" in info
        assert info["python"]["version"] is not None


# ── Metrics Tests ───────────────────────────────────────────────


class TestMetricsCollector:
    """Tests for the metrics collection and export system."""

    def test_add_and_retrieve(self):
        """Should store and retrieve metrics correctly."""
        collector = MetricsCollector()
        collector.add("loss", step=0, value=1.0)
        collector.add("loss", step=1, value=0.5)
        collector.add("loss", step=2, value=0.25)

        assert len(collector.get("loss")) == 3
        assert collector.get_values("loss") == [1.0, 0.5, 0.25]
        assert collector.get_steps("loss") == [0, 1, 2]

    def test_summary(self):
        """Summary should compute correct statistics."""
        collector = MetricsCollector()
        for i in range(100):
            collector.add("train/loss", step=i, value=1.0 / (i + 1))

        summary = collector.summary("train/loss")
        assert summary["min"] == pytest.approx(0.01, abs=0.001)
        assert summary["max"] == pytest.approx(1.0)
        assert summary["count"] == 100
        assert summary["last"] == pytest.approx(0.01, abs=0.001)

    def test_smoothing(self):
        """EMA smoothing should produce valid output."""
        collector = MetricsCollector()
        for i in range(50):
            collector.add("noisy", step=i, value=float(i) + random.random())

        steps, smoothed = collector.smooth("noisy", window=10)
        assert len(steps) == 50
        assert len(smoothed) == 50

    def test_csv_roundtrip(self, tmp_path):
        """Metrics should round-trip through CSV export/import."""
        collector = MetricsCollector()
        collector.add("a", step=0, value=1.0, phase="train")
        collector.add("a", step=1, value=2.0, phase="eval")
        collector.add("b", step=0, value=3.0)

        path = collector.to_csv(tmp_path / "metrics.csv")
        loaded = MetricsCollector.from_csv(path)

        assert loaded.get_values("a") == [1.0, 2.0]
        assert loaded.get_values("b") == [3.0]

    def test_json_roundtrip(self, tmp_path):
        """Metrics should round-trip through JSON export/import."""
        collector = MetricsCollector()
        for i in range(10):
            collector.add("loss", step=i, value=float(i))

        path = collector.to_json(tmp_path / "metrics.json")
        loaded = MetricsCollector.from_json(path)

        assert loaded.get_values("loss") == collector.get_values("loss")

    def test_latex_table(self):
        """LaTeX table generation should produce valid output."""
        collector = MetricsCollector()
        for i in range(20):
            collector.add("loss", step=i, value=1.0 / (i + 1))
            collector.add("accuracy", step=i, value=i / 20.0)

        latex = collector.to_latex_table()
        assert "\\begin{table}" in latex
        assert "\\toprule" in latex
        assert "loss" in latex
        assert "accuracy" in latex
        assert "\\end{table}" in latex

    def test_markdown_table(self):
        """Markdown table should produce valid output."""
        collector = MetricsCollector()
        collector.add("loss", step=0, value=0.5)
        collector.add("loss", step=1, value=0.3)

        md = collector.to_markdown_table()
        assert "| Metric |" in md
        assert "loss" in md

    def test_metric_names(self):
        """metric_names should return sorted unique names."""
        collector = MetricsCollector()
        collector.add("b", step=0, value=1.0)
        collector.add("a", step=0, value=1.0)
        collector.add("c", step=0, value=1.0)
        collector.add("a", step=1, value=2.0)

        assert collector.metric_names == ["a", "b", "c"]
