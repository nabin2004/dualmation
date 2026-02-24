"""
DualAnimate CLI — command-line interface for running experiments.

Provides subcommands:
    dualmation run       — Run the full pipeline
    dualmation train     — Start training loop
    dualmation evaluate  — Evaluate a checkpoint
    dualmation config    — Manage experiment configs
    dualmation annotate  — Human-in-the-loop ranking tool
    dualmation info      — System and environment info

All commands respect YAML configs from configs/ and support CLI overrides.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="dualmation",
        description="DualAnimate — Hybrid LLM-Diffusion Framework for Educational Animation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  dualmation run --concept "Explain gradient descent"
  dualmation run --config configs/default.yaml --concept "Linear algebra basics"
  dualmation train --config configs/training.yaml --epochs 50
  dualmation annotate --sessions my_sessions.json
  dualmation config show configs/default.yaml
  dualmation info
        """,
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug-level logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── run ──────────────────────────────────────────────────────
    run_parser = subparsers.add_parser("run", help="Run the full pipeline on a concept")
    run_parser.add_argument(
        "--concept",
        type=str,
        help="Educational concept to animate",
    )
    run_parser.add_argument(
        "--output-dir", "-o",
        type=str,
        help="Output directory for generated artifacts",
    )
    run_parser.add_argument(
        "--seed",
        type=int,
        help="Random seed (overrides config)",
    )
    run_parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        help="Compute device",
    )
    run_parser.add_argument(
        "--experiment-name",
        type=str,
        help="Experiment name for tracking",
    )
    run_parser.add_argument(
        "--no-tracking",
        action="store_true",
        help="Disable experiment tracking",
    )

    # ── train ────────────────────────────────────────────────────
    train_parser = subparsers.add_parser("train", help="Start a training run")
    train_parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs (overrides config)",
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate (overrides config)",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size (overrides config)",
    )
    train_parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume from",
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        help="Random seed (overrides config)",
    )

    # ── evaluate ─────────────────────────────────────────────────
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a model checkpoint")
    eval_parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to model checkpoint",
    )
    eval_parser.add_argument(
        "--concepts",
        type=str,
        nargs="+",
        help="Concepts to evaluate on",
    )
    eval_parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="eval_results",
        help="Output directory for evaluation results",
    )

    # ── config ───────────────────────────────────────────────────
    config_parser = subparsers.add_parser("config", help="Manage experiment configurations")
    config_subparsers = config_parser.add_subparsers(dest="config_command")

    show_parser = config_subparsers.add_parser("show", help="Display a config file")
    show_parser.add_argument("config_file", type=str, nargs="?", default="configs/default.yaml")

    init_parser = config_subparsers.add_parser("init", help="Create a new config from defaults")
    init_parser.add_argument("output_path", type=str, help="Output path for new config")
    init_parser.add_argument("--experiment-name", type=str, default="my_experiment")

    diff_parser = config_subparsers.add_parser("diff", help="Compare two config files")
    diff_parser.add_argument("config_a", type=str)
    diff_parser.add_argument("config_b", type=str)

    # ── annotate ──────────────────────────────────────────────────
    annotate_parser = subparsers.add_parser("annotate", help="Human-in-the-loop candidate ranking")
    annotate_parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/human_annotations.jsonl",
        help="Path to save annotations",
    )
    annotate_parser.add_argument(
        "--sessions", "-s",
        type=str,
        help="Path to JSON file containing sessions to rank (optional)",
    )
    annotate_parser.add_argument(
        "--exp-dir",
        type=str,
        default="experiments",
        help="Directory to scan for candidates to rank",
    )

    # ── info ─────────────────────────────────────────────────────
    subparsers.add_parser("info", help="Show system and environment information")

    return parser


def cmd_run(args: argparse.Namespace) -> None:
    """Execute the 'run' subcommand."""
    from dualmation.experiment.config import ExperimentConfig, load_config
    from dualmation.experiment.logging_setup import setup_logging
    from dualmation.experiment.reproducibility import set_seed
    from dualmation.experiment.tracker import ExperimentTracker, TrackerConfig
    from dualmation.pipeline import DualAnimatePipeline

    # Build overrides from CLI args
    overrides = {}
    if args.concept:
        overrides["experiment_name"] = args.concept
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.device:
        overrides["device"] = args.device
    if args.experiment_name:
        overrides["experiment_name"] = args.experiment_name

    config = load_config(args.config, overrides=overrides if overrides else None)
    set_seed(config.seed)

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_dir=config.output_dir, level=log_level)

    # Create pipeline from config
    pipeline = DualAnimatePipeline(config)

    # Attach tracker if enabled
    if not args.no_tracking and config.use_tensorboard:
        tracker_config = TrackerConfig(
            experiment_name=config.experiment_name,
            base_dir=config.output_dir,
            use_tensorboard=config.use_tensorboard,
            use_wandb=config.use_wandb,
            wandb_project=config.wandb_project,
        )
        tracker = ExperimentTracker(tracker_config)
        tracker.start()
        pipeline.attach_tracker(tracker)

    # Run pipeline
    concept = args.concept or config.experiment_name
    result = pipeline.run(concept)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  DualAnimate — Run Complete")
    print(f"{'='*60}")
    print(f"  Concept:     {result.concept}")
    print(f"  Code length: {len(result.generated_code)} chars")
    print(f"  Backgrounds: {len(result.background_images)} images")
    if result.reward:
        print(f"  Reward:      {result.reward.total:.4f}")
        print(f"    ├─ Alignment:   {result.reward.concept_alignment:.4f}")
        print(f"    ├─ Quality:     {result.reward.visual_quality:.4f}")
        print(f"    └─ Compilation: {result.reward.compilation_success:.4f}")
    print(f"  Output dir:  {config.output_dir}")
    print(f"{'='*60}\n")

    # Finish tracker
    if not args.no_tracking and config.use_tensorboard:
        tracker.finish()


def cmd_train(args: argparse.Namespace) -> None:
    """Execute the 'train' subcommand."""
    from dualmation.experiment.config import load_config
    from dualmation.experiment.logging_setup import setup_logging
    from dualmation.experiment.reproducibility import set_seed, save_environment_snapshot

    overrides = {}
    if args.epochs is not None:
        overrides.setdefault("training", {})["num_epochs"] = args.epochs
    if args.lr is not None:
        overrides.setdefault("training", {})["learning_rate"] = args.lr
    if args.batch_size is not None:
        overrides.setdefault("training", {})["batch_size"] = args.batch_size
    if args.seed is not None:
        overrides["seed"] = args.seed

    config = load_config(args.config, overrides=overrides if overrides else None)
    set_seed(config.seed)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_dir=config.output_dir, level=log_level)

    # Save environment snapshot for reproducibility
    save_environment_snapshot(config.output_dir)

    print(f"\n{'='*60}")
    print(f"  DualAnimate — Training")
    print(f"{'='*60}")
    print(f"  Epochs:    {config.training.num_epochs}")
    print(f"  LR:        {config.training.learning_rate}")
    print(f"  Batch:     {config.training.batch_size}")
    print(f"  Algorithm: {config.training.rl_algorithm}")
    print(f"  Seed:      {config.seed}")
    print(f"  Device:    {config.device}")
    if args.resume:
        print(f"  Resume:    {args.resume}")
    print(f"{'='*60}")
    print("\n  ⚠️  Training loop not yet implemented.")
    print("  The config is loaded and validated. Implement the training")
    print("  loop in src/dualmation/training/ when ready.\n")


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Execute the 'evaluate' subcommand."""
    from dualmation.experiment.config import load_config

    config = load_config(args.config)

    print(f"\n{'='*60}")
    print(f"  DualAnimate — Evaluation")
    print(f"{'='*60}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Concepts:   {args.concepts or ['(from dataset)']}")
    print(f"  Output dir: {args.output_dir}")
    print(f"{'='*60}")
    print("\n  ⚠️  Evaluation not yet implemented.")
    print("  Checkpoint loading and batch evaluation coming soon.\n")


def cmd_config(args: argparse.Namespace) -> None:
    """Execute the 'config' subcommand."""
    from dualmation.experiment.config import ExperimentConfig, load_config, save_config
    from dataclasses import asdict

    if args.config_command == "show":
        config = load_config(args.config_file)
        print(json.dumps(asdict(config), indent=2, default=str))

    elif args.config_command == "init":
        config = ExperimentConfig(experiment_name=args.experiment_name)
        path = save_config(config, args.output_path)
        print(f"✅ Config initialized: {path}")

    elif args.config_command == "diff":
        from dataclasses import asdict
        config_a = asdict(load_config(args.config_a))
        config_b = asdict(load_config(args.config_b))

        diffs = _dict_diff(config_a, config_b)
        if diffs:
            print(f"\nDifferences between {args.config_a} and {args.config_b}:\n")
            for key, (val_a, val_b) in diffs.items():
                print(f"  {key}:")
                print(f"    - {val_a}")
                print(f"    + {val_b}")
        else:
            print("Configs are identical.")

    else:
        print("Usage: dualmation config {show|init|diff}")


def cmd_annotate(args: argparse.Namespace) -> None:
    """Execute the 'annotate' subcommand."""
    from dualmation.training.hitl import HumanAnnotator, collect_from_experiments

    annotator = HumanAnnotator(annotation_file=args.output)
    
    if args.sessions:
        with open(args.sessions, "r") as f:
            sessions = json.load(f)
    else:
        print(f"🔍 Scanning {args.exp_dir} for candidates to rank...")
        sessions = collect_from_experiments(args.exp_dir)
    
    if not sessions:
        print(f"❌ No candidates found to rank in {args.exp_dir}.")
        return

    annotator.run_interactive(sessions)


def cmd_info(args: argparse.Namespace) -> None:
    """Execute the 'info' subcommand."""
    from dualmation.experiment.reproducibility import get_system_info

    info = get_system_info()
    print(f"\n{'='*60}")
    print(f"  DualAnimate — System Info")
    print(f"{'='*60}")
    print(f"\n  Python:  {info['python']['version'].split()[0]}")
    print(f"  OS:      {info['os']['system']} {info['os']['release']}")
    print(f"  Machine: {info['os']['machine']}")

    torch_info = info.get("torch", {})
    if torch_info.get("version"):
        print(f"\n  PyTorch: {torch_info['version']}")
        print(f"  CUDA:    {torch_info.get('cuda_version', 'N/A')}")
        print(f"  GPUs:    {torch_info.get('gpu_count', 0)}")
        for gpu in torch_info.get("gpus", []):
            print(f"    └─ {gpu['name']} ({gpu['total_memory_gb']} GB)")

    print(f"\n  Packages:")
    for pkg, ver in sorted(info.get("packages", {}).items()):
        status = "✅" if ver != "not_installed" else "❌"
        print(f"    {status} {pkg}: {ver}")

    git = info.get("git", {})
    if git.get("commit"):
        print(f"\n  Git:     {git['branch']} @ {git['commit'][:8]}")
        print(f"  Dirty:   {git.get('dirty', 'unknown')}")

    print()


def _dict_diff(a: dict, b: dict, prefix: str = "") -> dict:
    """Recursively diff two dictionaries."""
    diffs = {}
    all_keys = set(a.keys()) | set(b.keys())
    for key in sorted(all_keys):
        full_key = f"{prefix}.{key}" if prefix else key
        val_a = a.get(key)
        val_b = b.get(key)
        if isinstance(val_a, dict) and isinstance(val_b, dict):
            diffs.update(_dict_diff(val_a, val_b, full_key))
        elif val_a != val_b:
            diffs[full_key] = (val_a, val_b)
    return diffs


def main() -> None:
    """CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "run": cmd_run,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "config": cmd_config,
        "annotate": cmd_annotate,
        "info": cmd_info,
    }

    handler = commands.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
