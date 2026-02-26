# Running DualAnimate on Kaggle

This guide explains how to run the DualAnimate pipeline on Kaggle, including environment setup and command usage.

## 1. Clone the Repository

Open a Kaggle notebook and run:

```bash
!git clone https://github.com/nabin2004/dualmation.git
cd dualmation
```

## 2. Install Dependencies

Install required packages using pip:

```bash
!pip install -e .
```

For development dependencies:

```bash
!pip install -e ".[dev]"
```

## 3. Set Up Environment Variables

Kaggle notebooks use Python 3.x. If needed, set environment variables:

```python
import os
os.environ["PYTHONPATH"] = "src"
```

## 4. Run the Pipeline

To run the pipeline, use the following command:

```bash
!python -m dualmation.cli --config configs/default.yaml run \
    --concept "Explain gradient descent visually" \
    --seed 42
```

- Place `--config` before the subcommand (`run`).
- You can change the concept and seed as needed.

## 5. Output

Results and artifacts will be saved in the `experiments/` and `outputs/` directories. You can download them from the notebook file browser.

## 6. Troubleshooting

- If you see errors about missing models or files, ensure you have internet access enabled in your notebook.
- For HuggingFace models, authentication may be required for private/gated repos. See [HuggingFace authentication guide](https://huggingface.co/docs/huggingface_hub/authentication).

## 7. Example Notebook

See `notebooks/` for example scripts and workflows.

---
For further help, see the [README.md](README.md) or open an issue on GitHub.
