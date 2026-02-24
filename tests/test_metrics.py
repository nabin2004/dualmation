"""
Unit tests for MetricsCollector and report generation.
"""

from __future__ import annotations

import unittest
from pathlib import Path
import tempfile
import shutil

from dualmation.evaluation.metrics import MetricsCollector, EvalResult


class TestMetrics(unittest.TestCase):
    """Test suite for metrics aggregation and reporting."""

    def test_metrics_aggregation(self):
        """Test that MetricsCollector computes correct averages."""
        collector = MetricsCollector()
        collector.add_result(EvalResult("c1", 0.8, 0.9, 0.7, 1.0, 1))
        collector.add_result(EvalResult("c2", 0.4, 0.5, 0.3, 0.0, 3))
        
        summary = collector.get_summary()
        self.assertEqual(summary["count"], 2)
        self.assertAlmostEqual(summary["mean_reward"], 0.6)
        self.assertAlmostEqual(summary["pass_rate"], 0.5)
        self.assertAlmostEqual(summary["mean_turns"], 2.0)

    def test_latex_generation(self):
        """Test that LaTeX table generation produces expected strings."""
        collector = MetricsCollector()
        collector.add_result(EvalResult("c1", 0.8, 0.9, 0.7, 1.0, 1))
        
        latex = collector.generate_latex_table()
        self.assertIn("\\begin{table}", latex)
        self.assertIn("Mean Reward & 0.800", latex)
        self.assertIn("Total Concepts & 1", latex)

    def test_save_json(self):
        """Test saving results to JSON."""
        collector = MetricsCollector()
        collector.add_result(EvalResult("c1", 0.8, 0.9, 0.7, 1.0, 1))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "results.json"
            collector.save_json(json_path)
            self.assertTrue(json_path.exists())
            self.assertIn('"concept": "c1"', json_path.read_text())


if __name__ == "__main__":
    unittest.main()
