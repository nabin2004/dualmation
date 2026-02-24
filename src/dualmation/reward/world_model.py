"""
World Model for predicting Manim code compilability.

Acts as a fast proxy for the slow subprocess-based reward calculation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class ManimWorldModel(nn.Module):
    """Predicts the probability of Manim code compilation success.

    Uses a frozen or fine-tuned code encoder (e.g., CodeBERT) followed by
    a classification head.

    Args:
        encoder_name: HuggingFace model name for the base encoder.
        hidden_dim: Dimension of the classification head's hidden layer.
    """

    def __init__(self, encoder_name: str = "microsoft/codebert-base", hidden_dim: int = 256) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        
        # Freeze encoder by default to keep the world model lightweight/fast
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Predict compilability probability.

        Args:
            input_ids: Tokenized code indices.
            attention_mask: Mask for padding tokens.

        Returns:
            Tensor of shape (batch_size, 1) containing probabilities in [0, 1].
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use pooler output or mean pooling (CodeBERT typically uses CLS token)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.head(cls_output)

    def predict_code(self, code: str, tokenizer: AutoTokenizer, device: str = "cpu") -> float:
        """Convenience method to predict score for a single code snippet.

        Args:
            code: The Manim Python code to check.
            tokenizer: Tokenizer corresponding to the encoder.
            device: Device to run inference on.

        Returns:
            Float probability of success.
        """
        inputs = tokenizer(
            code, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(device)
        
        self.eval()
        with torch.no_grad():
            prob = self.forward(inputs["input_ids"], inputs["attention_mask"])
            
        return prob.item()
