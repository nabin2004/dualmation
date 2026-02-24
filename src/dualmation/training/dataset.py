"""
Dataset and DataLoader implementation for DualAnimate.

Handles triplets of (concept, code, visual) for training
multimodal embeddings and reinforcement learning rewards.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class AnimationDataset(Dataset):
    """Dataset for educational animation triplets.

    Expects a JSONL file where each line is a dictionary containing:
    - concept: str
    - code: str (Manim Python code)
    - visual_path: str (Optional, path to a reference image/frame)

    Args:
        data_path: Path to the JSONL data file.
        image_dir: Root directory for visual artifacts.
        transform: Optional torchvision transforms for images.
        tokenizer: Optional tokenizer for code (CodeBERT).
        processor: Optional processor for images (ViT).
    """

    def __init__(
        self,
        data_path: str | Path,
        image_dir: str | Path | None = None,
        transform: Any = None,
        tokenizer: Any = None,
        processor: Any = None,
    ) -> None:
        self.data_path = Path(data_path)
        self.image_dir = Path(image_dir) if image_dir else self.data_path.parent
        self.transform = transform
        self.tokenizer = tokenizer
        self.processor = processor

        self.samples = self._load_data()
        logger.info("Loaded %d samples from %s", len(self.samples), self.data_path)

    def _load_data(self) -> list[dict[str, Any]]:
        """Load samples from JSONL file."""
        samples = []
        if not self.data_path.exists():
            logger.warning("Data path %s does not exist. Returning empty dataset.", self.data_path)
            return []

        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Retrieve a sample by index."""
        sample = self.samples[idx]
        concept = sample.get("concept", "")
        code = sample.get("code", "")
        visual_path = sample.get("visual_path")

        item = {
            "concept": concept,
            "code": code,
        }

        # Load image if processor and path are available
        if self.processor and visual_path:
            full_path = self.image_dir / visual_path
            try:
                image = Image.open(full_path).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                
                # Use processor to get pixel values
                inputs = self.processor(images=image, return_tensors="pt")
                item["pixel_values"] = inputs["pixel_values"].squeeze(0)
            except Exception as e:
                logger.warning("Failed to load image %s: %s", full_path, e)

        # Tokenize code if tokenizer is available
        if self.tokenizer and code:
            tokens = self.tokenizer(
                code,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            item["input_ids"] = tokens["input_ids"].squeeze(0)
            item["attention_mask"] = tokens["attention_mask"].squeeze(0)

        return item


def create_dataloader(
    data_path: str | Path,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    **kwargs,
) -> DataLoader:
    """Utility to create a DataLoader for the AnimationDataset."""
    dataset = AnimationDataset(data_path, **kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
