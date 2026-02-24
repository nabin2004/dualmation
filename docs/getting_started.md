# Getting Started

This guide walks you through installing DualAnimate, running your first experiment, and understanding the output.

## Prerequisites

- Python 3.10+
- pip (or conda)
- (Optional) CUDA-capable GPU for model inference

## Installation

### Quick Install (core only)

```bash
git clone https://github.com/nabin2004/dualmation.git
cd dualmation
make install
```

### Development Install

```bash
make install-dev
```

### Full Install (with W&B experiment tracking)

```bash
make install-all
```

## Project Structure

```
dualmation/
├── configs/              # YAML experiment configs
│   └── default.yaml      # Default configuration
├── docs/                 # Documentation
├── src/dualmation/       # Source code
│   ├── embeddings/       # CodeBERT + ViT encoders
│   ├── llm/              # LLM Manim code generator
│   ├── diffusion/        # Stable Diffusion backgrounds
│   ├── compositor/       # Alpha compositing
│   ├── reward/           # RL reward model
│   ├── experiment/       # Experiment tracking & config
│   ├── pipeline.py       # End-to-end orchestrator
│   └── cli.py            # Command-line interface
├── tests/                # Test suite
├── Makefile              # Common commands
└── pyproject.toml        # Project metadata + deps
```

## Your First Run

### 1. Check your environment

```bash
make info
```

This shows your Python version, GPU availability, and installed packages.

### 2. View the default config

```bash
make config-show
```

### 3. Run the pipeline

```bash
# Using Make
make run CONCEPT="Explain gradient descent visually"

# Or using the CLI directly
PYTHONPATH=src python -m dualmation.cli run \
    --config configs/default.yaml \
    --concept "Explain gradient descent visually"
```

### 4. View results

Results are saved to the `outputs/` directory:
- `generated_scene.py` — Manim code
- `background_*.png` — Diffusion backgrounds
- `reward_summary.txt` — Reward scores

## Configuration

All experiments are configured via YAML files in `configs/`:

```yaml
# configs/my_experiment.yaml
experiment_name: gradient_descent_v2
seed: 123

llm:
  temperature: 0.5
  max_new_tokens: 2048

training:
  learning_rate: 3e-5
  batch_size: 16
```

Create a new config from defaults:

```bash
make config-init OUTPUT=configs/my_experiment.yaml NAME="my_experiment"
```

Override config values from the command line:

```bash
make run CONFIG=configs/my_experiment.yaml SEED=99
```

## Running Tests

```bash
# All tests
make test

# With coverage
make test-cov

# Quick (no slow tests)
make test-fast
```

## Experiment Tracking

### TensorBoard

```bash
# Run an experiment (TensorBoard is enabled by default)
make run CONCEPT="Pythagorean theorem"

# Launch TensorBoard
make tensorboard
# Open http://localhost:6006
```

### Weights & Biases

```bash
# Install W&B
pip install wandb
wandb login

# Enable in config
# Set use_wandb: true in your YAML config
```

## Next Steps

- Read the [Architecture](architecture.md) doc for system design
- Check the [Experiment Guide](experiment_guide.md) for research workflows
- See the [API Reference](api_reference.md) for module details
