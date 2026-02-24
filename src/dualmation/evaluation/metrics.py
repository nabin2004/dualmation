"""
Metrics collection and aggregation for pipeline evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class EvalResult:
    """Result of a single concept evaluation."""
    concept: str
    reward_total: float
    alignment: float
    visual_quality: float
    compilation_success: float
    num_turns: int = 0


class MetricsCollector:
    """Collects and aggregates metrics from multiple pipeline runs."""

    def __init__(self) -> None:
        self.results: list[EvalResult] = []

    def add_result(self, result: EvalResult) -> None:
        """Add a single evaluation result."""
        self.results.append(result)

    def get_summary(self) -> dict[str, float]:
        """Compute aggregate statistics over all collected results."""
        if not self.results:
            return {}

        totals = [r.reward_total for r in self.results]
        alignments = [r.alignment for r in self.results]
        visuals = [r.visual_quality for r in self.results]
        compilations = [r.compilation_success for r in self.results]
        turns = [r.num_turns for r in self.results]

        return {
            "mean_reward": float(np.mean(totals)),
            "mean_alignment": float(np.mean(alignments)),
            "mean_visual_quality": float(np.mean(visuals)),
            "pass_rate": float(np.mean(compilations)),
            "mean_turns": float(np.mean(turns)),
            "count": len(self.results),
        }

    def save_json(self, path: str | Path) -> None:
        """Save raw results to a JSON file."""
        data = [asdict(r) for r in self.results]
        Path(path).write_text(json.dumps(data, indent=2))

    def generate_latex_table(self) -> str:
        """Generate a LaTeX table summary of the results."""
        summary = self.get_summary()
        if not summary:
            return "No results."

        latex = r"""
\begin{table}[h]
\centering
\begin{tabular}{|l|c|}
\hline
\textbf{Metric} & \textbf{Score} \\
\hline
Mean Reward & """ + f"{summary['mean_reward']:.3f}" + r""" \\
Concept Alignment & """ + f"{summary['mean_alignment']:.3f}" + r""" \\
Visual Quality & """ + f"{summary['mean_visual_quality']:.3f}" + r""" \\
Pass Rate (Compilation) & """ + f"{summary['pass_rate']:.1%}" + r""" \\
Avg Correction Turns & """ + f"{summary['mean_turns']:.2f}" + r""" \\
\hline
Total Concepts & """ + str(summary['count']) + r""" \\
\hline
\end{tabular}
\caption{DualAnimate Batch Evaluation Summary}
\end{table}
"""
        return latex.strip()
