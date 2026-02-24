"""
LLM-based Manim code generator.

Uses a HuggingFace causal language model (e.g., CodeLlama) to generate
Manim Python code from natural language concept descriptions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

# System prompt that guides the LLM to produce valid Manim code
MANIM_SYSTEM_PROMPT = """\
You are an expert Manim animation programmer. Given a mathematical or scientific concept,
generate a complete, self-contained Manim Community Edition scene that visually explains the concept.

Rules:
1. Import from `manim` (Community Edition).
2. Create a single Scene subclass with a `construct` method.
3. Use precise mathematical objects (MathTex, Axes, NumberPlane, etc.).
4. Include smooth animations (Create, Write, Transform, FadeIn/Out).
5. Add brief text labels for clarity.
6. Output ONLY valid Python code — no markdown fences, no explanations.
"""


@dataclass
class GenerationConfig:
    """Configuration for code generation."""

    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.1
    stop_sequences: list[str] = field(default_factory=lambda: ["\n\n\n", "```"])


class ManimCodeGenerator:
    """Generates Manim Python code from concept descriptions using an LLM.

    Args:
        model_name: HuggingFace model identifier for the code generation LLM.
        device: Device to run the model on. Defaults to auto-detection.
        torch_dtype: Data type for model weights (e.g., torch.float16 for GPU).
        load_in_8bit: Whether to load the model in 8-bit quantization (requires bitsandbytes).
    """

    DEFAULT_MODEL = "codellama/CodeLlama-7b-hf"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        torch_dtype: torch.dtype = torch.float16,
        load_in_8bit: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading LLM: %s on %s", model_name, self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs: dict = {"torch_dtype": torch_dtype}
        if load_in_8bit:
            load_kwargs["load_in_8bit"] = True
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = self.device

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.model.eval()

    def _build_prompt(self, concept: str, additional_context: str = "") -> str:
        """Build the full prompt for Manim code generation.

        Args:
            concept: The natural language concept description.
            additional_context: Optional additional context or constraints.

        Returns:
            Formatted prompt string.
        """
        prompt_parts = [
            MANIM_SYSTEM_PROMPT,
            f"\nConcept: {concept}",
        ]
        if additional_context:
            prompt_parts.append(f"\nAdditional context: {additional_context}")
        prompt_parts.append("\n\n# Manim Scene Code:\nfrom manim import *\n")
        return "\n".join(prompt_parts)

    @torch.inference_mode()
    def generate(
        self,
        concept: str,
        config: GenerationConfig | None = None,
        additional_context: str = "",
    ) -> str:
        """Generate Manim code for a given concept.

        Args:
            concept: Natural language description of the concept to animate.
            config: Generation hyperparameters. Uses defaults if None.
            additional_context: Optional extra instructions or constraints.

        Returns:
            Generated Manim Python code as a string.
        """
        if config is None:
            config = GenerationConfig()

        prompt = self._build_prompt(concept, additional_context)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            do_sample=config.do_sample,
            repetition_penalty=config.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Decode only the generated tokens (not the prompt)
        generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
        code = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Clean up: stop at any stop sequences
        for stop_seq in config.stop_sequences:
            if stop_seq in code:
                code = code[: code.index(stop_seq)]

        # Prepend the standard import
        full_code = "from manim import *\n" + code.strip()
        return full_code

    def _build_correction_prompt(self, concept: str, original_code: str, error_message: str) -> str:
        """Build a prompt for correcting Manim code based on an error message.

        Args:
            concept: The original concept description.
            original_code: The code that failed to run.
            error_message: The traceback or error message from the Manim renderer.

        Returns:
            Formatted correction prompt.
        """
        prompt_parts = [
            MANIM_SYSTEM_PROMPT,
            f"\nOriginal Concept: {concept}",
            "\nPreviously generated code that failed:",
            "```python",
            original_code,
            "```",
            f"\nError encountered during rendering:\n{error_message}",
            "\nInstructions:",
            "1. Analyze the error above.",
            "2. Fix the bug in the Manim code.",
            "3. Ensure the scene still correctly visualizes the concept.",
            "4. Output ONLY the complete, fixed Python code.",
            "\nFixed Manim Scene Code:\nfrom manim import *\n",
        ]
        return "\n".join(prompt_parts)

    @torch.inference_mode()
    def generate_correction(
        self,
        concept: str,
        original_code: str,
        error_message: str,
        config: GenerationConfig | None = None,
    ) -> str:
        """Generate a corrected version of Manim code based on an error message.

        Args:
            concept: Original concept description.
            original_code: Previous code that failed.
            error_message: Traceback or error description.
            config: Generation parameters.

        Returns:
            Corrected Manim Python code.
        """
        if config is None:
            config = GenerationConfig()

        prompt = self._build_correction_prompt(concept, original_code, error_message)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            do_sample=config.do_sample,
            repetition_penalty=config.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
        code = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        for stop_seq in config.stop_sequences:
            if stop_seq in code:
                code = code[: code.index(stop_seq)]

        full_code = "from manim import *\n" + code.strip()
        return full_code

    def generate_with_embedding(
        self,
        concept: str,
        embedding: torch.Tensor | None = None,
        config: GenerationConfig | None = None,
    ) -> str:
        """Generate Manim code, optionally conditioned on a shared embedding.

        For now, the embedding is used to augment the prompt with a textual
        description. In future iterations, this will use embedding injection
        via adapter layers.

        Args:
            concept: Concept description.
            embedding: Optional embedding tensor from the multimodal space.
            config: Generation config.

        Returns:
            Generated Manim Python code.
        """
        context = ""
        if embedding is not None:
            # Future: inject embedding into model via cross-attention adapters
            # For now, we note the conditioning in the context
            context = f"[Embedding-conditioned generation, dim={embedding.shape[-1]}]"

        return self.generate(concept, config=config, additional_context=context)
