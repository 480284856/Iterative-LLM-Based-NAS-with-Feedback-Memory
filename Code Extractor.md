Here is the python function to extract the code from the llm output.

```python
def extract_code(response: str) -> str:
    """
    Extract the code from the llm output.
    """
    return response.split('```python')[1].split('```')[0]
```