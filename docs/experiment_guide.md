# Experiment Guide

This guide covers how to set up, run, and analyze experiments with DualAnimate's research infrastructure.

## Experiment Workflow

```
1. Configure  →  2. Run  →  3. Track  →  4. Analyze  →  5. Report
   (YAML)           (CLI)     (TB/W&B)     (metrics)      (LaTeX)
```

## 1. Configuration

### Config Hierarchy

DualAnimate uses a layered config system (lowest to highest priority):

1. **Dataclass defaults** — sensible baselines baked into the code
2. **YAML config file** — `configs/default.yaml` or experiment-specific
3. **CLI overrides** — command-line flags like `--seed 123`

### Creating an Experiment Config

```bash
# Start from defaults
make config-init OUTPUT=configs/ablation_lr.yaml NAME="lr_ablation"
```

Then edit the YAML to set your experiment parameters:

```yaml
# configs/ablation_lr.yaml
experiment_name: lr_ablation
seed: 42

training:
  learning_rate: 1e-5    # Changed from default 1e-4
  num_epochs: 50
  batch_size: 16

use_tensorboard: true
use_wandb: true           # Enable cloud tracking
```

### Comparing Configs

```bash
make config-diff CONFIG_A=configs/default.yaml CONFIG_B=configs/ablation_lr.yaml
```

## 2. Running Experiments

### Single Run

```bash
make run CONFIG=configs/ablation_lr.yaml CONCEPT="Explain eigenvalues"
```

### Sweep (Manual)

```bash
for lr in 1e-3 1e-4 1e-5; do
  make train CONFIG=configs/default.yaml LR=$lr SEED=42
done
```

### Reproducibility

Every run automatically:
- Sets deterministic seeds (Python, NumPy, PyTorch, CUDA)
- Snapshots the full config to `run_dir/config_snapshot/`
- Captures system info and pip freeze to `run_dir/environment/`
- Records git commit hash and dirty status

To reproduce a past experiment:
```bash
# 1. Find the config snapshot
cat experiments/<run_name>/config_snapshot/config.json

# 2. Replay with the same config
make run CONFIG=experiments/<run_name>/config_snapshot/config.yaml
```

## 3. Experiment Tracking

### TensorBoard (Default)

```bash
# Start TensorBoard
make tensorboard

# Custom port
make tensorboard TB_PORT=8080

# Specific experiment directory
make tensorboard LOGDIR=experiments/my_run/tensorboard
```

Metrics logged:
- `train/loss`, `train/accuracy`, `eval/loss`
- `reward/total`, `reward/alignment`, `reward/quality`, `reward/compilation`
- `pipeline/code_length`, `pipeline/embedding_norm`
- `timing/*` — per-component timing

### Weights & Biases (Optional)

```bash
pip install wandb
wandb login

# Enable in your config YAML:
# use_wandb: true
# wandb_project: dualmation
```

### Programmatic Tracking

```python
from dualmation.experiment import ExperimentTracker
from dualmation.experiment.tracker import TrackerConfig

config = TrackerConfig(
    experiment_name="my_experiment",
    use_tensorboard=True,
    use_wandb=False,
    use_csv=True,
)

with ExperimentTracker(config) as tracker:
    for step in range(1000):
        loss = train_step()

        # Scalars
        tracker.log_scalar("train/loss", loss, step)

        # Multiple at once
        tracker.log_scalars("reward", {
            "alignment": 0.8,
            "quality": 0.6,
            "compilation": 1.0,
        }, step)

        # Timing
        with tracker.timer("forward_pass", step):
            output = model(input)

    # Log hyperparams + final metrics
    tracker.log_hyperparams(
        {"lr": 1e-4, "batch_size": 8},
        {"final_loss": 0.01, "best_reward": 0.95},
    )
```

## 4. Analysis & Visualization

### MetricsCollector

```python
from dualmation.experiment import MetricsCollector

collector = MetricsCollector()
for step in range(1000):
    collector.add("train/loss", step, loss_values[step])
    collector.add("eval/loss", step, eval_values[step])

# Summary stats
print(collector.summary("train/loss"))
# {'min': 0.01, 'max': 1.0, 'mean': 0.35, 'std': 0.28, ...}
```

### Paper-Ready Plots

```python
# Single-column IEEE format (3.5" wide)
collector.plot_metrics(
    ["train/loss", "eval/loss"],
    title="Training Convergence",
    xlabel="Step",
    ylabel="Loss",
    smoothing_window=20,
    save_path="figures/convergence.pdf",  # Auto-saves PNG + PDF
)

# Double-column format (7.16" wide)
collector.plot_metrics(
    ["reward/total", "reward/alignment", "reward/quality"],
    title="Reward Components",
    save_path="figures/rewards.pdf",
    double_column=True,
)
```

### Ablation Comparison

```python
# Load metrics from multiple experiments
exp_a = MetricsCollector.from_json("experiments/run_a/exports/metrics.json")
exp_b = MetricsCollector.from_json("experiments/run_b/exports/metrics.json")

collector.plot_comparison(
    {"baseline": exp_a, "improved": exp_b},
    metric_name="train/loss",
    title="Ablation: Learning Rate",
    save_path="figures/lr_ablation.pdf",
)
```

### LaTeX Tables

```python
# Generate a booktabs-style table for your paper
latex = collector.to_latex_table(
    metric_names=["train/loss", "eval/loss", "reward/total"],
    caption="Main Results",
    label="tab:main_results",
    save_path="tables/main_results.tex",
    precision=4,
)
```

Output:
```latex
\begin{table}[htbp]
\centering
\caption{Main Results}
\label{tab:main_results}
\begin{tabular}{lrrrrr}
\toprule
Metric & Min & Max & Mean $\pm$ Std & Median & Last \\
\midrule
loss & 0.0100 & 1.0000 & 0.3500 $\pm$ 0.2800 & 0.2500 & 0.0100 \\
\bottomrule
\end{tabular}
\end{table}
```

### Markdown Tables

```python
md = collector.to_markdown_table()
print(md)
```

## 5. Output Structure

Each experiment run creates:

```
experiments/<run_name>/
├── tensorboard/          # TensorBoard logs
├── checkpoints/          # Model checkpoints
├── figures/              # PNG + PDF plots
├── exports/
│   ├── all_metrics.json  # All logged metrics
│   ├── train_loss.csv    # Per-metric CSV files
│   └── ...
├── config_snapshot/
│   ├── config.yaml       # Frozen config
│   ├── config.json       # JSON copy
│   └── config_summary.txt
├── environment/
│   ├── system_info.json  # GPU, packages, git state
│   ├── requirements_frozen.txt
│   └── env_hash.txt      # Quick environment comparison
├── hyperparams.json      # Logged hyperparameters
├── run_summary.json      # Aggregated run statistics
├── experiment.log        # Human-readable log
└── experiment.jsonl      # Structured JSON log
```
