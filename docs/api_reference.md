# API Reference

Quick reference for all public modules and classes in DualAnimate.

---

## Pipeline

### `DualAnimatePipeline`

**Module**: `dualmation.pipeline`

End-to-end pipeline orchestrator.

```python
from dualmation.pipeline import DualAnimatePipeline
from dualmation.experiment.config import load_config

# From config file
config = load_config("configs/default.yaml")
pipeline = DualAnimatePipeline(config)

# Or with classmethod
pipeline = DualAnimatePipeline.from_config_file(
    "configs/default.yaml",
    overrides={"seed": 123}
)

result = pipeline.run("Explain gradient descent visually")
```

| Method | Description |
|--------|-------------|
| `__init__(config)` | Initialize with `ExperimentConfig` |
| `from_config_file(path, overrides)` | Create from YAML config |
| `attach_tracker(tracker)` | Attach experiment tracker |
| `run(concept)` | Execute full pipeline |

---

## Embeddings

### `CodeEncoder`

**Module**: `dualmation.embeddings.code_encoder`

```python
from dualmation.embeddings.code_encoder import CodeEncoder

encoder = CodeEncoder(model_name="microsoft/codebert-base", embedding_dim=512)
embedding = encoder.encode("def fibonacci(n): ...")  # → Tensor(1, 512)
```

### `VisualEncoder`

**Module**: `dualmation.embeddings.visual_encoder`

```python
from dualmation.embeddings.visual_encoder import VisualEncoder

encoder = VisualEncoder(model_name="google/vit-base-patch16-224", embedding_dim=512)
embedding = encoder.encode(pil_image)  # → Tensor(1, 512)
```

### `InfoNCELoss`

**Module**: `dualmation.embeddings.contrastive`

```python
from dualmation.embeddings.contrastive import InfoNCELoss

loss_fn = InfoNCELoss(temperature=0.07)
result = loss_fn(code_embeddings, visual_embeddings)
# result["loss"], result["loss_c2v"], result["loss_v2c"], result["accuracy"]
```

---

## LLM

### `ManimCodeGenerator`

**Module**: `dualmation.llm.code_generator`

```python
from dualmation.llm.code_generator import ManimCodeGenerator

gen = ManimCodeGenerator(model_name="codellama/CodeLlama-7b-hf")
code = gen.generate("Explain the Pythagorean theorem")
```

---

## Diffusion

### `VisualGenerator`

**Module**: `dualmation.diffusion.visual_generator`

```python
from dualmation.diffusion.visual_generator import VisualGenerator

gen = VisualGenerator(model_name="stabilityai/stable-diffusion-2-1")
images = gen.generate("Abstract mathematical landscape")
```

---

## Compositor

### `AlphaCompositor`

**Module**: `dualmation.compositor.compositor`

```python
from dualmation.compositor.compositor import AlphaCompositor, CompositeConfig

config = CompositeConfig(blend_mode="screen")
compositor = AlphaCompositor(config)
result = compositor.composite(foreground_img, background_img)
```

| Blend Mode | Description |
|------------|-------------|
| `alpha` | Standard alpha blending |
| `screen` | Lighter blend (good for glow effects) |
| `multiply` | Darker blend (good for shadows) |

---

## Reward

### `RewardModel`

**Module**: `dualmation.reward.reward_model`

```python
from dualmation.reward.reward_model import RewardModel, RewardConfig

config = RewardConfig(weight_alignment=0.4, weight_visual=0.3, weight_compilation=0.3)
model = RewardModel(config)
score = model.score(code="from manim import *...", visual=pil_image, concept="gradient descent")
# score.total, score.concept_alignment, score.visual_quality, score.compilation_success
```

---

## Experiment Framework

### `ExperimentTracker`

**Module**: `dualmation.experiment.tracker`

```python
from dualmation.experiment.tracker import ExperimentTracker, TrackerConfig

with ExperimentTracker(TrackerConfig(experiment_name="test")) as tracker:
    tracker.log_scalar("loss", 0.5, step=0)
    tracker.log_scalars("reward", {"align": 0.8}, step=0)
    tracker.log_hyperparams({"lr": 1e-4}, {"final_loss": 0.01})
    tracker.log_image("sample", pil_image, step=0)
    tracker.log_figure("plot", matplotlib_fig, step=0)
    with tracker.timer("train_step", step=0):
        ...
```

### `ExperimentConfig`

**Module**: `dualmation.experiment.config`

```python
from dualmation.experiment.config import ExperimentConfig, load_config, save_config

# Load from YAML
config = load_config("configs/default.yaml", overrides={"seed": 99})

# Save
save_config(config, "configs/my_run.yaml")
```

### `MetricsCollector`

**Module**: `dualmation.experiment.metrics`

```python
from dualmation.experiment.metrics import MetricsCollector

collector = MetricsCollector()
collector.add("loss", step=0, value=1.0)

# Outputs
collector.summary("loss")            # dict
collector.plot_metrics(["loss"])      # matplotlib Figure
collector.to_latex_table()            # str
collector.to_csv("metrics.csv")      # Path
collector.to_json("metrics.json")    # Path

# Round-trip
loaded = MetricsCollector.from_json("metrics.json")
```

### Reproducibility

**Module**: `dualmation.experiment.reproducibility`

```python
from dualmation.experiment.reproducibility import set_seed, get_system_info

set_seed(42, deterministic=True)
info = get_system_info()  # Python, OS, GPU, packages, git
```

### Logging

**Module**: `dualmation.experiment.logging_setup`

```python
from dualmation.experiment.logging_setup import setup_logging

logger = setup_logging(log_dir="logs/", level=logging.INFO)
# Creates: logs/experiment.log + logs/experiment.jsonl
```

---

## CLI

**Module**: `dualmation.cli`

```bash
# Run pipeline
dualmation run --config configs/default.yaml --concept "Gradient descent"

# Training
dualmation train --config configs/training.yaml --epochs 50 --lr 1e-5

# Config management
dualmation config show configs/default.yaml
dualmation config init configs/new.yaml --experiment-name my_exp
dualmation config diff configs/a.yaml configs/b.yaml

# System info
dualmation info
```

Or via Makefile:

```bash
make run CONCEPT="Gradient descent" CONFIG=configs/default.yaml
make train EPOCHS=50 LR=1e-5
make config-show
make info
make test
make tensorboard
```
