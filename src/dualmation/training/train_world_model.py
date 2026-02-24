"""
Trainer for the Manim World Model.
"""

from __future__ import annotations

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from dualmation.reward.world_model import ManimWorldModel


class CompilabilityDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: AutoTokenizer, max_length: int = 512):
        self.samples = []
        with open(data_path, "r") as f:
            for line in f:
                self.samples.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        encoding = self.tokenizer(
            item["code"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor([float(item["label"])], dtype=torch.float)
        }


def train(data_path: str, model_path: str, epochs: int = 5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = ManimWorldModel().to(device)
    
    dataset = CompilabilityDataset(data_path, tokenizer)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-4)
    criterion = nn.BCELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to {model_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, default="models/world_model.pt")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    
    train(args.data, args.output, args.epochs)
