"""
LLM Client for Qwen3 8B using Transformers library.
Provides a unified interface for text generation.
"""

import torch
import random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# Default seed for reproducibility
DEFAULT_SEED = 43


class LLMClient:
    """LLM Client that loads Qwen3 8B and provides generation interface."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", seed: int = DEFAULT_SEED):
        """
        Initialize the LLM client.
        
        Args:
            model_name: HuggingFace model name. Default is Qwen2.5-7B-Instruct.
            seed: Random seed for reproducible generation. Default is 43.
        """
        self.model_name = model_name
        self.seed = seed
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize call counter for deterministic seeding across runs
        self._call_counter = 0
        
        # Set initial seed
        self._set_seed_with_counter()
        
        print(f"Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        print(f"Model loaded on {self.device}")
    
    def _set_seed_with_counter(self):
        """Set seed based on call counter for reproducibility across runs but diversity within run."""
        # Use base seed + call counter so each call gets a unique but reproducible seed
        current_seed = self.seed + self._call_counter
        self._call_counter += 1
        
        set_seed(current_seed)
        torch.manual_seed(current_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(current_seed)
        np.random.seed(current_seed)
        random.seed(current_seed)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """
        Generate text from the given prompt.
        
        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling parameter
            do_sample: Whether to use sampling or greedy decoding
            
        Returns:
            Generated text response
        """
        # Set seed based on counter to ensure diversity + reproducibility
        self._set_seed_with_counter()
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Remove the input tokens from the output
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    
    def generate_with_system(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """
        Generate text with a system prompt.
        
        Args:
            system_prompt: System instruction
            user_prompt: User input prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling parameter
            do_sample: Whether to use sampling or greedy decoding
            
        Returns:
            Generated text response
        """
        # Set seed based on counter to ensure diversity + reproducibility
        self._set_seed_with_counter()
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response


if __name__ == "__main__":
    # Test the LLM client
    client = LLMClient()
    response = client.generate("Hello, how are you?")
    print(f"Response: {response}")
