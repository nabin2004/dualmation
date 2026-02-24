"""
Script to collect (code, compilability) pairs for training the World Model.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from dualmation.pipeline import DualAnimatePipeline
from dualmation.experiment.config import load_config
from dualmation.reward.reward_model import RewardModel

logger = logging.getLogger(__name__)


def collect_data(config_path: str, output_path: str, num_samples: int = 50):
    """Collect compilation data by running the code generator on varying prompts."""
    config = load_config(config_path)
    pipeline = DualAnimatePipeline(config)
    reward_model = RewardModel() # Used for the compilation check
    
    concepts = [
        "Fourier Series visualization",
        "Neural Network layers",
        "Sorting algorithms",
        "Calculus derivatives",
        "Graph theory nodes",
        "Linear algebra transformations",
        "Physics gravity simulation",
        "Chemistry molecule bonds",
        "History timeline animation",
        "Music frequency wave"
    ]
    
    dataset = []
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info("📡 Starting data collection for %d samples", num_samples)

    for i in range(num_samples):
        concept = concepts[i % len(concepts)]
        logger.info("[%d/%d] Generating for: %s", i+1, num_samples, concept)
        
        try:
            # Generate code without the full pipeline overhead if possible
            # But pipeline.run handles the turn-based logic which is useful
            result = pipeline.run(concept)
            code = result.generated_code
            
            # We already have the compilation status in result.reward if reward model ran
            is_valid = 0.0
            if result.reward:
                is_valid = result.reward.compilation_success
            
            dataset.append({"code": code, "label": is_valid})
            
            # Incremental save
            with open(output_file, "a") as f:
                f.write(json.dumps({"code": code, "label": is_valid}) + "\n")
                
        except Exception as e:
            logger.error("Failed to collect sample: %s", e)

    logger.info("✅ Collection complete. Saved to %s", output_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, default="data/training/compilability_data.jsonl")
    parser.add_argument("--num_samples", type=int, default=20)
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    collect_data(args.config, args.output, args.num_samples)
