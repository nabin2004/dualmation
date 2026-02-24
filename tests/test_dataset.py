"""
Unit tests for the training dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from dualmation.training.dataset import AnimationDataset


@pytest.fixture
def dummy_data(tmp_path):
    """Create a dummy JSONL dataset and images."""
    data_path = tmp_path / "train.jsonl"
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    # Create dummy images
    img = Image.new("RGB", (224, 224), color="red")
    img.save(image_dir / "img1.png")
    
    img2 = Image.new("RGB", (224, 224), color="blue")
    img2.save(image_dir / "img2.png")

    samples = [
        {"concept": "Concept 1", "code": "Code 1", "visual_path": "img1.png"},
        {"concept": "Concept 2", "code": "Code 2", "visual_path": "img2.png"},
        {"concept": "Concept 3", "code": "Code 3", "visual_path": None},
    ]

    with open(data_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    return data_path, image_dir


class MockTokenizer:
    def __call__(self, text, **kwargs):
        return {
            "input_ids": torch.zeros((1, 10), dtype=torch.long),
            "attention_mask": torch.ones((1, 10), dtype=torch.long),
        }


class MockProcessor:
    def __call__(self, images, **kwargs):
        return {"pixel_values": torch.zeros((1, 3, 224, 224))}


def test_dataset_loading(dummy_data):
    """Test that samples are loaded correctly."""
    data_path, image_dir = dummy_data
    dataset = AnimationDataset(data_path, image_dir=image_dir)
    
    assert len(dataset) == 3
    assert dataset[0]["concept"] == "Concept 1"
    assert dataset[1]["code"] == "Code 2"
    assert "pixel_values" not in dataset[0]  # No processor yet


def test_dataset_with_processor_and_tokenizer(dummy_data):
    """Test dataset with mock processor and tokenizer."""
    data_path, image_dir = dummy_data
    dataset = AnimationDataset(
        data_path,
        image_dir=image_dir,
        tokenizer=MockTokenizer(),
        processor=MockProcessor(),
    )
    
    item = dataset[0]
    assert "input_ids" in item
    assert "pixel_values" in item
    assert item["input_ids"].shape == (10,)
    assert item["pixel_values"].shape == (3, 224, 224)

    # Item 3 has no image
    item3 = dataset[2]
    assert "pixel_values" not in item3
    assert item3["concept"] == "Concept 3"
