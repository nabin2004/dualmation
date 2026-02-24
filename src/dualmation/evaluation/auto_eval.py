"""
Automated evaluation script for batch-processing benchmarks.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dualmation.pipeline import DualAnimatePipeline
from dualmation.experiment.config import load_config
from dualmation.evaluation.metrics import MetricsCollector, EvalResult

logger = logging.getLogger(__name__)


def run_eval(config_path: str, benchmark_path: str, output_dir: str):
    """Run the evaluation pipeline on a benchmark dataset."""
    config = load_config(config_path)
    pipeline = DualAnimatePipeline(config)
    collector = MetricsCollector()
    
    benchmark_data = json.loads(Path(benchmark_path).read_text())
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("🚀 Starting batch evaluation on %d concepts", len(benchmark_data))

    for i, item in enumerate(benchmark_data):
        concept = item["concept"]
        logger.info("[%d/%d] Evaluating: %s", i+1, len(benchmark_data), concept)
        
        try:
            result = pipeline.run(concept)
            
            eval_res = EvalResult(
                concept=concept,
                reward_total=result.reward.total if result.reward else 0.0,
                alignment=result.reward.concept_alignment if result.reward else 0.0,
                visual_quality=result.reward.visual_quality if result.reward else 0.0,
                compilation_success=result.reward.compilation_success if result.reward else 0.0,
                num_turns=0 # Turn tracking needs plumbing in pipeline.run for exact count
            )
            collector.add_result(eval_res)
        except Exception as e:
            logger.error("Failed to evaluate %s: %s", concept, e)

    # Save results
    collector.save_json(output_path / "eval_results.json")
    latex_report = collector.generate_latex_table()
    (output_path / "evaluation_report.tex").write_text(latex_report)
    
    logger.info("✅ Evaluation complete. Results saved to %s", output_dir)
    print("\n" + "="*20)
    print("EVALUATION SUMMARY")
    print("="*20)
    summary = collector.get_summary()
    for k, v in summary.items():
        print(f"{k:20}: {v:.4f}")
    print("="*20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DualAnimate Auto-Evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config")
    parser.add_argument("--benchmark", type=str, default="data/benchmarks/gold_standard.json", help="Path to benchmark JSON")
    parser.add_argument("--output", type=str, default="outputs/eval_report", help="Output directory")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    run_eval(args.config, args.benchmark, args.output)
