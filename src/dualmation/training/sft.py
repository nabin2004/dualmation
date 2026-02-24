"""
Supervised Fine-Tuning (SFT) Trainer for DualAnimate LLM.

Uses the `trl` library to perform SFT on the code generation model
to improve its base Manim code generation capabilities.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from trl import SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig

from dualmation.experiment.config import LLMConfig, TrainingConfig

logger = logging.getLogger(__name__)


class LLMSFTTrainer:
    """Trainer for Supervised Fine-Tuning of the Manim code generator.

    Args:
        llm_config: Configuration for the LLM.
        training_config: Configuration for training.
        device: Compute device.
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        training_config: TrainingConfig,
        device: str = "auto",
    ) -> None:
        self.llm_config = llm_config
        self.training_config = training_config
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_config.model_name,
            load_in_8bit=llm_config.load_in_8bit,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(llm_config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # PEFT / LoRA setup
        self.peft_config = None
        if llm_config.use_lora:
            self.peft_config = LoraConfig(
                r=llm_config.lora.r,
                lora_alpha=llm_config.lora.alpha,
                target_modules=llm_config.lora.target_modules,
                lora_dropout=llm_config.lora.dropout,
                bias=llm_config.lora.bias,
                task_type="CAUSAL_LM",
            )
            logger.info("LoRA configuration initialized for SFT.")

    def train(self, dataset: Any, output_dir: str | Path) -> Any:
        """Execute the SFT process.
        
        Args:
            dataset: Dataset containing 'concept' and 'code' columns.
            output_dir: Directory to save the fine-tuned model.
        """
        output_dir = Path(output_dir)
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=self.training_config.batch_size,
            learning_rate=self.training_config.learning_rate,
            num_train_epochs=self.training_config.num_epochs,
            logging_steps=10,
            save_strategy="epoch",
            mixed_precision="bf16" if torch.cuda.is_available() else "no",
            use_cpu=(self.device == "cpu"),
            report_to="tensorboard",
        )

        # Dataset formatting for SFT
        def formatting_func(example):
            output_texts = []
            for i in range(len(example['concept'])):
                text = f"Concept: {example['concept'][i]}\nCode:\n{example['code'][i]}"
                output_texts.append(text)
            return output_texts

        trainer = SFTTrainer(
            model=self.model,
            train_dataset=dataset,
            peft_config=self.peft_config,
            dataset_text_field="text",  # We override this with formatting_func
            formatting_func=formatting_func,
            max_seq_length=self.llm_config.max_new_tokens,
            tokenizer=self.tokenizer,
            args=training_args,
        )

        logger.info("Starting Supervised Fine-Tuning...")
        return trainer.train()
