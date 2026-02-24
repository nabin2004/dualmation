# 📋 DualAnimate Development Checklist & Roadmap

This document serves as a guide for developers to understand the current state of the project, what needs immediate implementation, and the long-term research direction.

---

## 🏗️ Phase 1: Immediate Implementation (High Priority)

These items are necessary for the first end-to-end differentiable training run.

### 🔄 Training Loop (`src/dualmation/training/`)
- [ ] **Data Loaders**: Implement loaders for `(concept, code, visual)` triplets.
- [ ] **PPO Trainer**: Implement the Proximal Policy Optimization loop for the LLM.
- [ ] **GRPO Implementation**: Add Group Relative Policy Optimization (GRPO) for more stable RL training in math/logic tasks.
- [ ] **Differentiable Rewards**: Optimize the reward model to handle gradient pass-through where possible (e.g., embedding-based alignment).

### 🧪 Evaluation Suite (`src/dualmation/evaluation/`)
- [ ] **Benchmark Dataset**: Create a set of 100+ "Gold Standard" educational concepts with reference Manim code.
- [ ] **Auto-Evaluation Script**: Batch process concepts and generate a full comparison report (LaTeX/PDF) using `MetricsCollector`.
- [ ] **Human-in-the-loop (HITL)**: Basic UI/CLI for humans to rank animations to bootstrap the reward model.

---

## 🧠 Phase 2: Core Refinement

Improving the quality of individual "Brains".

### 🧠 Brain 1: LLM (Logic & Code)
- [ ] **Fine-tuning**: Fine-tune CodeLlama/DeepSeek-Coder on a curated Manim dataset.
- [ ] **Self-Correction**: Implement a multi-turn "Compiler Loop" where the LLM fixes its own code based on Manim traceback errors.
- [ ] **Multi-Scene Support**: Generate complex, multi-chapter animations with consistent variable naming.

### 🎨 Brain 2: Diffusion (Aesthetics)
- [ ] **LoRA Training**: Train a LoRA for Stable Diffusion specific to "flat, educational illustration" styles.
- [ ] **Video Consistency**: Use Video Diffusion (e.g., SVD) or ControlNet to make backgrounds dynamic rather than static images.
- [ ] **Color Palette Sync**: Automatically extract colors from the diffusion background and inject them as theme variables into the Manim code.

---

## 🔬 Phase 3: Research Roadmap (Future Direction)

Long-term goals for publishing in conferences (CVPR, NeurIPS, ICML).

### 📐 Multimodal Alignment
- [ ] **Contrastive SSL**: Scale up the shared embedding space with 10k+ paired samples.
- [ ] **Cross-Modal Retrieval**: Build a feature where a user can upload a sketch, and the system retrieves/generates related Manim code.

### 🤖 RL with World Models
- [ ] **World Model Integration**: Train a world model that predicts Manim "compilability" to speed up the RL loop (acting as a proxy for the slow subprocess call).
- [ ] **Multi-Reward Fusion**: Use Pareto optimization to balance "Aesthetics" vs "Accuracy" vs "Complexity".

---

## 🛠️ Developer Onboarding: Where to head next?

If you are new to the project, here is the suggested path:

1.  **Run the Pipeline**: Use `make run` to see the current inference flow.
2.  **Explore the Reward Model**: Look at `src/dualmation/reward/reward_model.py`. This is the "soul" of the project's feedback loop.
3.  **Implement the Simple Trainer**: Help us fill in `src/dualmation/training/`. Start by training the `CodeEncoder` to better match the `VisualEncoder`.
4.  **Add a Test Case**: Create a new complex scene in `manim_scripts/` and see if the CLIP reward accurately scores its alignment with a description.

---

## 📈 Metric Goals for v1.0
- [ ] Compilation Success Rate > 90%
- [ ] Concept Alignment (CLIP Score) > 0.35 (Average)
- [ ] Inference time under 60 seconds (7B model + SD 2.1)
