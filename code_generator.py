"""
Code Generator module.
Uses LLM to generate PyTorch vision model code for CIFAR-10 classification.
"""

import torch
import random
import numpy as np
from typing import Optional
from llm_client import LLMClient

# Initial prompt template from CodeGenerator.md
INITIAL_PROMPT_TEMPLATE = """## Role

You are a visionary deep learning architect renowned for designing breakthrough neural networks by drawing inspiration from meta principles in diverse scientific domains.

## Task

Generate a vision model that maximizes the accuracy on the CIFAR-10 dataset for the image classification task.

## Requirements

- Don't use pre-trained models.
- Contain the implementation of the model, no other code.
- If reference code is provided, improve upon it based on the improvement suggestions.

## Reference Code (Best Implementation So Far)
{reference_code}

## Improvement Suggestions (from previous iteration)
{improvement_suggestions}

## Output format
```python
class Net(nn.Module):
    def __init__(self, parameters: dict):
        super(Net, self).__init__()
        
        self.xxx = xxx
        ...

    def forward(self, *args, **kwargs):
        pass
```"""

# For first iteration when there's no reference code
NO_REFERENCE_CODE = "No reference code available. This is the first iteration."

# For first iteration when there's no improvement suggestions
NO_IMPROVEMENT_SUGGESTIONS = "No improvement suggestions yet. This is the first iteration."


class CodeGenerator:
    """Generates vision model code using LLM."""
    
    def __init__(self, llm_client: LLMClient, initial_prompt_template: str = None):
        """
        Initialize the code generator.
        
        Args:
            llm_client: LLM client for text generation
            initial_prompt_template: Initial prompt template. If None, uses default.
        """
        self.llm_client = llm_client
        self.prompt_template = initial_prompt_template or INITIAL_PROMPT_TEMPLATE
        self.reference_code: Optional[str] = None
        self.improvement_suggestions: Optional[str] = None
        
    def generate(
        self, 
        prompt_template: str = None, 
        reference_code: str = None,
        improvement_suggestions: str = None
    ) -> str:
        """
        Generate vision model code.
        
        Args:
            prompt_template: Optional custom prompt template. If None, uses stored template.
            reference_code: Optional reference code from previous iteration.
            improvement_suggestions: Optional improvement suggestions from previous iteration.
            
        Returns:
            LLM response containing the generated code
        """
        template = prompt_template 
        ref_code = reference_code or NO_REFERENCE_CODE
        suggestions = improvement_suggestions or NO_IMPROVEMENT_SUGGESTIONS
        
        # Format the prompt with reference code and improvement suggestions
        current_prompt = template.format(
            reference_code=ref_code,
            improvement_suggestions=suggestions
        )
        
        response = self.llm_client.generate(
            current_prompt,
            max_new_tokens=2048,
            temperature=0.7
        )
        return response
    
    def update_prompt_template(self, new_template: str):
        """
        Update the generation prompt template.
        
        Args:
            new_template: New prompt template to use for code generation
        """
        self.prompt_template = new_template
    
    def update_reference_code(self, code: str):
        """
        Update the reference code for next generation.
        
        Args:
            code: Reference code from previous iteration
        """
        self.reference_code = code
    
    def update_improvement_suggestions(self, suggestions: str):
        """
        Update the improvement suggestions for next generation.
        
        Args:
            suggestions: Improvement suggestions from prompt improver
        """
        self.improvement_suggestions = suggestions
    
    def get_prompt_template(self) -> str:
        """Get the current prompt template."""
        return self.prompt_template
    
    def get_reference_code(self) -> Optional[str]:
        """Get the current reference code."""
        return self.reference_code
    
    def get_improvement_suggestions(self) -> Optional[str]:
        """Get the current improvement suggestions."""
        return self.improvement_suggestions


if __name__ == "__main__":
    # Test the code generator
    client = LLMClient()
    generator = CodeGenerator(client)
    
    print("Generating code (first iteration, no reference)...")
    response = generator.generate()
    print(f"Generated response:\n{response}")
