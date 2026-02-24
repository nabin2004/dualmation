# 🎬 DualAnimate

> **Hybrid LLM-Diffusion Framework for Educational Animation Generation with RL Feedback**

DualAnimate combines the mathematical precision of LLM-generated [Manim](https://www.manim.community/) code with the visual richness of diffusion-generated backgrounds, unified by self-supervised multimodal embeddings and reinforced through multi-component reward scoring.

---

## 🏗️ Architecture

```
                    ┌─────────────────────┐
                    │  Concept Description │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Multimodal Embedding │
                    │ CodeBERT + ViT (SSL) │
                    └────┬───────────┬────┘
                         ▼           ▼
              ┌──────────────┐ ┌──────────────┐
              │ 🧠 Brain 1   │ │ 🎨 Brain 2   │
              │ LLM Code Gen │ │ Diffusion Gen│
              │ (Manim Code) │ │ (Background) │
              └──────┬───────┘ └──────┬───────┘
                     ▼                ▼
              ┌───────────────────────────┐
              │  ⊕ Alpha Compositor       │
              │  Foreground + Background  │
              └────────────┬──────────────┘
                           ▼
              ┌───────────────────────────┐
              │  🎯 RL Reward Model       │
              │  Alignment · Quality ·    │
              │  Compilation Success      │
              └────────────┬──────────────┘
                           ▼
              ┌───────────────────────────┐
              │  🎬 Educational Animation │
              └───────────────────────────┘
                     ↑ RL Feedback Loop ↑
```

---

## 📦 Project Structure

```
dualmation/
├── src/dualmation/
│   ├── embeddings/        # CodeBERT + ViT + InfoNCE contrastive
│   ├── llm/               # LLM → Manim code generation
│   ├── diffusion/         # Diffusion → visual backgrounds
│   ├── compositor/        # Alpha compositing engine
│   ├── reward/            # Multi-component RL reward model
│   └── pipeline.py        # End-to-end orchestrator
├── tests/                 # pytest test suite
├── manim_scripts/         # Example Manim scenes
├── notebooks/             # Jupyter notebooks
├── docs/                  # Documentation
├── outputs/               # Generated samples
└── pyproject.toml
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/nabin2004/dualmation.git
cd dualmation

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"
```

### Usage

```python
from dualmation.pipeline import DualAnimatePipeline, PipelineConfig

config = PipelineConfig(
    concept="Explain gradient descent visually",
    llm_model="codellama/CodeLlama-7b-hf",
    diffusion_model="stabilityai/stable-diffusion-2-1",
)

pipeline = DualAnimatePipeline(config)
result = pipeline.run()
```

---

## 🧩 Modules

| Module | Description |
|--------|-------------|
| `embeddings` | Self-supervised multimodal embedding space (CodeBERT + ViT, InfoNCE loss) |
| `llm` | LLM-driven Manim Python code generation |
| `diffusion` | Diffusion-based visual context and background generation |
| `compositor` | Alpha compositing of Manim foreground + diffusion background |
| `reward` | Multi-component RL reward: concept alignment, visual quality, compilation success |
| `pipeline` | End-to-end orchestrator connecting all modules |

---

## 🧪 Testing

```bash
python -m pytest tests/ -v
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🗺️ Roadmap

- [x] Project structure & dependencies
- [ ] Self-supervised multimodal embeddings
- [ ] LLM code generation module
- [ ] Diffusion visual generation module
- [ ] Alpha compositor
- [ ] RL reward model
- [ ] End-to-end pipeline
- [ ] Example notebooks & sample outputs

---

*Built with ❤️ using PyTorch, HuggingFace Transformers, Diffusers, and Manim.*
