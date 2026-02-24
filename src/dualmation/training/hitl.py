"""
Human-in-the-loop (HITL) annotation utilities for DualAnimate.

Allows humans to rank or prefer between multiple generated candidates
per concept. This data is used to bootstrap reward models via RLHF.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class HumanAnnotator:
    """CLI tool for human ranking of animation candidates.

    Args:
        annotation_file: Path to save annotations (JSONL).
        experiment_dir: Directory containing experiment candidates to rank.
    """

    def __init__(
        self,
        annotation_file: str | Path = "data/human_annotations.jsonl",
        experiment_dir: str | Path = "experiments",
    ) -> None:
        self.annotation_file = Path(annotation_file)
        self.experiment_dir = Path(experiment_dir)
        self.annotation_file.parent.mkdir(parents=True, exist_ok=True)

    def _save_annotation(self, concept: str, candidates: list[dict], preference: list[int]) -> None:
        """Save a single comparison annotation."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "concept": concept,
            "candidates": candidates,
            "ranking": preference,  # Indices in candidates list, sorted by preference
            "best_idx": preference[0] if preference else None,
        }
        with open(self.annotation_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        logger.info("Annotation saved to %s", self.annotation_file)

    def run_interactive(self, sessions: list[dict[str, Any]]) -> None:
        """Run an interactive CLI session to rank candidates.

        Args:
            sessions: List of dicts, each with 'concept' and 'candidates' (list of code strings).
        """
        print("\n" + "=" * 60)
        print("🎨 DualAnimate — Human-in-the-Loop Annotation")
        print("Rank candidates from BEST to WORST (e.g., '1 0' if 1 is better than 0)")
        print("=" * 60 + "\n")

        for i, session in enumerate(sessions):
            concept = session["concept"]
            candidates = session["candidates"]

            print(f"[{i+1}/{len(sessions)}] Concept: {concept}")
            print("-" * 40)

            for idx, cand in enumerate(candidates):
                code_snippet = cand.get("code", "")[:200] + "..."
                print(f"\n[Candidate {idx}]")
                print(f"Code Preview:\n{code_snippet}")

            while True:
                try:
                    choice = input(f"\nRanking for {len(candidates)} candidates (space separated indices): ").strip()
                    if not choice:
                        print("Skipping...")
                        break
                    
                    indices = [int(x) for x in choice.split()]
                    if len(indices) != len(candidates) or any(x >= len(candidates) for x in indices) or len(set(indices)) != len(indices):
                        print(f"Please provide exactly {len(candidates)} unique indices from 0 to {len(candidates)-1}")
                        continue
                    
                    self._save_annotation(concept, candidates, indices)
                    print("✅ Recorded.")
                    break
                except ValueError:
                    print("Invalid input. Use integers separated by spaces.")

        print("\n🏁 Session complete!")

def collect_from_experiments(exp_dir: str | Path) -> list[dict]:
    """Automagically find candidates to rank from an experiment directory.
    
    Looks for subdirectories with generated_scene.py and groups them by experiment/concept.
    """
    exp_path = Path(exp_dir)
    results = {} # concept -> list of code
    
    # This is a heuristic; actual implementation depends on output_dir structure
    for p in exp_path.glob("**/generated_scene.py"):
        # Assume directory name or a metadata file contains the concept
        folder = p.parent
        concept = folder.name # Simple heuristic
        code = p.read_text()
        
        if concept not in results:
            results[concept] = []
        results[concept].append({"code": code, "path": str(p)})
        
    return [{"concept": c, "candidates": cds} for c, cds in results.items() if len(cds) > 1]
