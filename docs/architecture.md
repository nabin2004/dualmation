# DualAnimate Architecture

This document describes the high-level architecture and design decisions behind DualAnimate.

## System Overview

DualAnimate is a hybrid LLM-Diffusion framework that generates educational animations by combining two "brains":

```
┌─────────────────────────────────────────────────────────────────┐
│                    DualAnimate Pipeline                         │
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────────┐ │
│  │ Concept   │───▶│  Embeddings  │───▶│  Shared Embedding     │ │
│  │ (text)    │    │  (CodeBERT + │    │  Space (512-dim)      │ │
│  │           │    │   ViT)       │    │                       │ │
│  └──────────┘    └──────────────┘    └───────┬───────────────┘ │
│                                              │                  │
│                    ┌─────────────────────────┤                  │
│                    │                         │                  │
│                    ▼                         ▼                  │
│  ┌─────────────────────┐  ┌─────────────────────────────┐      │
│  │ Brain 1: Logic       │  │ Brain 2: Aesthetics         │      │
│  │ (LLM → Manim code)  │  │ (Diffusion → backgrounds)   │      │
│  │ CodeLlama-7B         │  │ Stable Diffusion 2.1        │      │
│  └──────────┬──────────┘  └──────────────┬──────────────┘      │
│             │                            │                      │
│             ▼                            ▼                      │
│  ┌──────────────────────────────────────────────────────┐      │
│  │              Alpha Compositor                        │      │
│  │   Blend modes: alpha | screen | multiply             │      │
│  └──────────────────────┬───────────────────────────────┘      │
│                         │                                       │
│                         ▼                                       │
│  ┌──────────────────────────────────────────────────────┐      │
│  │              Reward Model (RL feedback)              │      │
│  │   • Concept alignment (CLIP)                         │      │
│  │   • Visual quality (aesthetic scoring)               │      │
│  │   • Compilation success (Manim runner)               │      │
│  └──────────────────────────────────────────────────────┘      │
│                                                                 │
│  ┌──────────────────────────────────────────────────────┐      │
│  │              Experiment Framework                    │      │
│  │   TensorBoard │ W&B │ CSV/JSON │ LaTeX tables        │      │
│  └──────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

## Module Descriptions

### Embeddings (`src/dualmation/embeddings/`)

Creates a shared multimodal embedding space where code and images can be compared.

| Component | Model | Purpose |
|-----------|-------|---------|
| `CodeEncoder` | CodeBERT | Encodes Manim source code into 512-d vectors |
| `VisualEncoder` | ViT-Base | Encodes images into 512-d vectors |
| `InfoNCELoss` | — | Contrastive loss for aligning code and visual embeddings |

### LLM Code Generator (`src/dualmation/llm/`)

Generates Manim Python code from natural language concept descriptions. Uses a Manim-specific system prompt with examples and best practices.

### Diffusion Visual Generator (`src/dualmation/diffusion/`)

Generates aesthetic backgrounds using Stable Diffusion. Prompts are automatically augmented for educational animation style.

### Compositor (`src/dualmation/compositor/`)

Merges Manim-rendered foregrounds with diffusion backgrounds using configurable blend modes (alpha, screen, multiply). Auto-detects Manim's black background for alpha mask generation.

### Reward Model (`src/dualmation/reward/`)

Multi-component scoring:
- **Concept alignment** (40%): CLIP similarity between concept and visual output
- **Visual quality** (30%): Aesthetic scoring via CLIP
- **Compilation success** (30%): Whether the Manim code compiles and runs

Supports PPO advantage computation for RL training.

### Experiment Framework (`src/dualmation/experiment/`)

Research-grade infrastructure:
- **Tracker**: TensorBoard + W&B + CSV/JSON unified logging
- **Config**: OmegaConf hierarchical configs from YAML files
- **Reproducibility**: Deterministic seeding, system info, env snapshots
- **Metrics**: IEEE-formatted plots, LaTeX tables, ablation comparisons
- **Logging**: Colored console + file + JSONL structured logs

## Design Principles

1. **Modularity**: Each component is independently testable and replaceable
2. **Lazy loading**: Heavy models (LLM, diffusion) load on-demand
3. **Config-driven**: All experiments parameterized via YAML configs
4. **Reproducible**: Seed management, config snapshots, environment capture
5. **Paper-ready**: LaTeX tables, IEEE-formatted plots, structured exports

## Data Flow

```
concept (str)
    │
    ├─▶ CodeEncoder.encode() ─────▶ embedding (512-d)
    │                                    │
    ├─▶ ManimCodeGenerator.generate() ◀──┤──▶ code (str)
    │                                    │        │
    ├─▶ VisualGenerator.generate() ◀─────┘──▶ backgrounds (PIL)
    │                                                │
    │    AlphaCompositor.composite() ◀───────────────┘──▶ frames (PIL)
    │                                                        │
    └─▶ RewardModel.score() ◀────────────────────────────────┘──▶ reward
```
