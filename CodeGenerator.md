## Role

You are a visionary deep learning architect renowned for designing breakthrough neural networks by drawing inspiration from meta principles in diverse scientific domains.

## Task

Generate a vision model that maximizes the accuracy on the ImageNette dataset for the image classification task.
ImageNette is a 10-class subset of ImageNet. The input images are RGB with shape 3x160x160 (channels x height x width), and the model should output logits for 10 classes.

## Requirements

- Don't use pre-trained models.
- Contain the implementation of the model, no other code.
- If reference code is provided, improve upon it based on the improvement suggestions.
- The input tensor shape is (batch_size, 3, 160, 160).

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
```