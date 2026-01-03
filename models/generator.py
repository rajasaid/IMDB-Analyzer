# models/generator.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ResponseGenerator:
    """
    Lightweight generative LLM wrapper.
    Uses Phi-2 (recommended) by default,
    but can load any causal language model.
    """

    def __init__(
        self,
        model_name: str = "microsoft/phi-2",
        max_new_tokens: int = 150,
        temperature: float = 0.7,
        top_p: float = 0.9,
        device: str = None
    ):
        """
        Initialize tokenizer and model.
        """

        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens

        # Select device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[ResponseGenerator] Using device: {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        # Load causal LLM
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=None
        ).to(self.device)

        self.model.eval()

    # ----------------------------------------------------------------------
    # GENERATION
    # ----------------------------------------------------------------------
    def generate(self, prompt: str) -> str:
        """
        Generate a text response from the LLM given a prompt.
        """

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode and clean response
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Remove the prompt portion
        response = response[len(prompt):].strip()

        return response
