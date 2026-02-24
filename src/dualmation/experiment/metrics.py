"""
Metrics collection and paper-ready visualization module.

Provides:
- Structured metrics collection with automatic aggregation
- Publication-quality matplotlib plots (following IEEE/ACM formatting)
- LaTeX table generation for research papers
- Comparison utilities for ablation studies
"""

from __future__ import annotations

import csv
import json
import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Publication-quality plot configuration
PAPER_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.figsize": (3.5, 2.625),  # Single column IEEE width
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
}

# Color palette suitable for colorblind readers (Wong palette)
PAPER_COLORS = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#E69F00",  # amber
    "#56B4E9",  # sky blue
    "#CC79A7",  # pink
    "#F0E442",  # yellow
    "#000000",  # black
]

DOUBLE_COL_STYLE = {**PAPER_STYLE, "figure.figsize": (7.16, 3.5)}


@dataclass
class MetricEntry:
    """A single metric data point."""
    step: int
    value: float
    epoch: int | None = None
    phase: str = "train"  # train | eval | test


class MetricsCollector:
    """Collects, aggregates, and exports metrics for research experiments.

    Designed for easy integration with the ExperimentTracker and for
    generating paper-ready outputs (plots, tables, LaTeX).

    Usage:
        ```python
        collector = MetricsCollector()
        for step in range(1000):
            collector.add("train/loss", step, loss_val)
            collector.add("train/accuracy", step, acc_val)

        # Generate paper-ready plot
        collector.plot_metrics(["train/loss"], save_path="figures/loss.pdf")

        # Export to LaTeX table
        collector.to_latex_table(save_path="tables/results.tex")
        ```
    """

    def __init__(self) -> None:
        self._metrics: dict[str, list[MetricEntry]] = defaultdict(list)

    def add(
        self,
        name: str,
        step: int,
        value: float,
        epoch: int | None = None,
        phase: str = "train",
    ) -> None:
        """Record a metric value.

        Args:
            name: Metric name (e.g., "train/loss").
            step: Global step number.
            value: Metric value.
            epoch: Optional epoch number.
            phase: Phase identifier (train/eval/test).
        """
        self._metrics[name].append(MetricEntry(
            step=step, value=value, epoch=epoch, phase=phase
        ))

    def get(self, name: str) -> list[MetricEntry]:
        """Get all entries for a metric."""
        return self._metrics.get(name, [])

    def get_values(self, name: str) -> list[float]:
        """Get just the values for a metric."""
        return [e.value for e in self.get(name)]

    def get_steps(self, name: str) -> list[int]:
        """Get just the steps for a metric."""
        return [e.step for e in self.get(name)]

    @property
    def metric_names(self) -> list[str]:
        """All registered metric names."""
        return sorted(self._metrics.keys())

    # ── Aggregation ─────────────────────────────────────────────────

    def summary(self, name: str) -> dict[str, float]:
        """Compute summary statistics for a metric.

        Args:
            name: Metric name.

        Returns:
            Dictionary with min, max, mean, std, last, count.
        """
        values = self.get_values(name)
        if not values:
            return {}

        return {
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "median": statistics.median(values),
            "last": values[-1],
            "best": min(values) if "loss" in name.lower() else max(values),
            "count": len(values),
        }

    def summary_all(self) -> dict[str, dict[str, float]]:
        """Compute summary for all metrics."""
        return {name: self.summary(name) for name in self.metric_names}

    def smooth(self, name: str, window: int = 10) -> tuple[list[int], list[float]]:
        """Apply exponential moving average smoothing.

        Args:
            name: Metric name.
            window: Smoothing window size.

        Returns:
            Tuple of (steps, smoothed_values).
        """
        entries = self.get(name)
        if not entries:
            return [], []

        alpha = 2.0 / (window + 1)
        smoothed = []
        current = entries[0].value
        for entry in entries:
            current = alpha * entry.value + (1 - alpha) * current
            smoothed.append(current)

        steps = [e.step for e in entries]
        return steps, smoothed

    # ── Visualization ───────────────────────────────────────────────

    def plot_metrics(
        self,
        metric_names: list[str],
        title: str = "",
        xlabel: str = "Step",
        ylabel: str = "Value",
        save_path: str | Path | None = None,
        smoothing_window: int = 0,
        double_column: bool = False,
        show_std: bool = False,
        legend_loc: str = "best",
    ) -> Any:
        """Generate publication-quality metric plots.

        Follows IEEE/ACM single-column (3.5") or double-column (7.16") formatting.
        Uses colorblind-safe color palette.

        Args:
            metric_names: List of metric names to plot.
            title: Plot title.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            save_path: If set, saves as both PNG and PDF.
            smoothing_window: If > 0, apply EMA smoothing.
            double_column: If True, use double-column width.
            show_std: If True, show standard deviation bands.
            legend_loc: Legend location.

        Returns:
            matplotlib Figure object.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        style = DOUBLE_COL_STYLE if double_column else PAPER_STYLE
        with plt.rc_context(style):
            fig, ax = plt.subplots()

            for i, name in enumerate(metric_names):
                color = PAPER_COLORS[i % len(PAPER_COLORS)]
                label = name.split("/")[-1].replace("_", " ").title()

                if smoothing_window > 0:
                    steps, values = self.smooth(name, smoothing_window)
                    # Also plot raw in light color
                    raw_steps = self.get_steps(name)
                    raw_values = self.get_values(name)
                    ax.plot(raw_steps, raw_values, color=color, alpha=0.15, linewidth=0.5)
                else:
                    steps = self.get_steps(name)
                    values = self.get_values(name)

                ax.plot(steps, values, color=color, label=label)

            if title:
                ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.legend(loc=legend_loc, framealpha=0.8)

            fig.tight_layout()

            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_path.with_suffix(".png"), dpi=300)
                fig.savefig(save_path.with_suffix(".pdf"))
                logger.info("Plot saved: %s (.png + .pdf)", save_path.stem)

            plt.close(fig)
            return fig

    def plot_comparison(
        self,
        experiments: dict[str, MetricsCollector],
        metric_name: str,
        title: str = "",
        save_path: str | Path | None = None,
        smoothing_window: int = 10,
    ) -> Any:
        """Plot the same metric across multiple experiments (ablation study).

        Args:
            experiments: Dict of experiment_name → MetricsCollector.
            metric_name: Metric to compare.
            title: Plot title.
            save_path: Save path for the figure.
            smoothing_window: EMA smoothing window.

        Returns:
            matplotlib Figure.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with plt.rc_context(PAPER_STYLE):
            fig, ax = plt.subplots()

            for i, (exp_name, collector) in enumerate(experiments.items()):
                color = PAPER_COLORS[i % len(PAPER_COLORS)]

                if smoothing_window > 0:
                    steps, values = collector.smooth(metric_name, smoothing_window)
                else:
                    steps = collector.get_steps(metric_name)
                    values = collector.get_values(metric_name)

                ax.plot(steps, values, color=color, label=exp_name)

            ax.set_title(title or metric_name)
            ax.set_xlabel("Step")
            ax.set_ylabel(metric_name.split("/")[-1].replace("_", " ").title())
            ax.legend(framealpha=0.8)
            fig.tight_layout()

            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_path.with_suffix(".png"), dpi=300)
                fig.savefig(save_path.with_suffix(".pdf"))

            plt.close(fig)
            return fig

    # ── Table Export (LaTeX / Markdown) ──────────────────────────────

    def to_latex_table(
        self,
        metric_names: list[str] | None = None,
        caption: str = "Experimental Results",
        label: str = "tab:results",
        save_path: str | Path | None = None,
        precision: int = 4,
        highlight_best: bool = True,
    ) -> str:
        """Generate a LaTeX table from metric summaries.

        Suitable for direct inclusion in IEEE/ACM papers.

        Args:
            metric_names: Metrics to include. All if None.
            caption: Table caption.
            label: LaTeX label.
            save_path: If set, saves the .tex file.
            precision: Decimal precision for values.
            highlight_best: Bold the best value per metric.

        Returns:
            LaTeX table string.
        """
        names = metric_names or self.metric_names
        summaries = {name: self.summary(name) for name in names}

        # Build LaTeX
        cols = "l" + "r" * 5
        lines = [
            f"\\begin{{table}}[htbp]",
            f"\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            f"\\begin{{tabular}}{{{cols}}}",
            "\\toprule",
            "Metric & Min & Max & Mean $\\pm$ Std & Median & Last \\\\",
            "\\midrule",
        ]

        for name in names:
            s = summaries[name]
            if not s:
                continue
            short_name = name.split("/")[-1].replace("_", "\\_")
            mean_std = f"${s['mean']:.{precision}f} \\pm {s['std']:.{precision}f}$"
            row = (
                f"{short_name} & "
                f"{s['min']:.{precision}f} & "
                f"{s['max']:.{precision}f} & "
                f"{mean_std} & "
                f"{s['median']:.{precision}f} & "
                f"{s['last']:.{precision}f} \\\\"
            )
            lines.append(row)

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ])

        latex = "\n".join(lines)

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(latex)
            logger.info("LaTeX table saved: %s", save_path)

        return latex

    def to_markdown_table(
        self,
        metric_names: list[str] | None = None,
        precision: int = 4,
    ) -> str:
        """Generate a Markdown table from metric summaries.

        Args:
            metric_names: Metrics to include.
            precision: Decimal precision.

        Returns:
            Markdown table string.
        """
        names = metric_names or self.metric_names
        summaries = {name: self.summary(name) for name in names}

        lines = [
            "| Metric | Min | Max | Mean ± Std | Last |",
            "|--------|-----|-----|------------|------|",
        ]

        for name in names:
            s = summaries[name]
            if not s:
                continue
            short = name.split("/")[-1]
            mean_std = f"{s['mean']:.{precision}f} ± {s['std']:.{precision}f}"
            lines.append(
                f"| {short} | {s['min']:.{precision}f} | {s['max']:.{precision}f} | "
                f"{mean_std} | {s['last']:.{precision}f} |"
            )

        return "\n".join(lines)

    # ── CSV/JSON Export ─────────────────────────────────────────────

    def to_csv(self, path: str | Path, metric_names: list[str] | None = None) -> Path:
        """Export metrics to CSV file.

        Args:
            path: Output CSV path.
            metric_names: Metrics to export. All if None.

        Returns:
            Path to saved CSV.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        names = metric_names or self.metric_names

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "step", "value", "epoch", "phase"])
            for name in names:
                for entry in self.get(name):
                    writer.writerow([name, entry.step, entry.value, entry.epoch, entry.phase])

        logger.info("Metrics exported to CSV: %s (%d entries)", path, sum(len(self.get(n)) for n in names))
        return path

    def to_json(self, path: str | Path, metric_names: list[str] | None = None) -> Path:
        """Export metrics to JSON file.

        Args:
            path: Output JSON path.
            metric_names: Metrics to export.

        Returns:
            Path to saved JSON.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        names = metric_names or self.metric_names

        data = {}
        for name in names:
            data[name] = {
                "entries": [
                    {"step": e.step, "value": e.value, "epoch": e.epoch, "phase": e.phase}
                    for e in self.get(name)
                ],
                "summary": self.summary(name),
            }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info("Metrics exported to JSON: %s", path)
        return path

    # ── Loading ─────────────────────────────────────────────────────

    @classmethod
    def from_json(cls, path: str | Path) -> MetricsCollector:
        """Load metrics from a previously exported JSON file.

        Args:
            path: Path to JSON metrics file.

        Returns:
            MetricsCollector populated with the loaded data.
        """
        collector = cls()
        with open(path) as f:
            data = json.load(f)

        for name, metric_data in data.items():
            for entry in metric_data.get("entries", []):
                collector.add(
                    name=name,
                    step=entry["step"],
                    value=entry["value"],
                    epoch=entry.get("epoch"),
                    phase=entry.get("phase", "train"),
                )

        logger.info("Loaded metrics from: %s (%d metrics)", path, len(data))
        return collector

    @classmethod
    def from_csv(cls, path: str | Path) -> MetricsCollector:
        """Load metrics from a CSV file.

        Args:
            path: Path to CSV metrics file.

        Returns:
            MetricsCollector populated with the loaded data.
        """
        collector = cls()
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                collector.add(
                    name=row["metric"],
                    step=int(row["step"]),
                    value=float(row["value"]),
                    epoch=int(row["epoch"]) if row.get("epoch") else None,
                    phase=row.get("phase", "train"),
                )

        return collector
