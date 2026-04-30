# Resource-Efficient Iterative LLM-Based NAS with Feedback Memory

Resource-Efficient Iterative prompt-improvement pipeline: uses an LLM to generate model code, then trains and evaluates it. Supports CIFAR-10, CIFAR-100, and ImageNette. You can turn off the improver or reference via flags for ablation runs.

## Environment and dependencies

- **Python**: 3.10 recommended
- **Data**: default `./data`; CIFAR is downloaded automatically; ImageNette is fetched from the URL in the script

## Installation

```bash
# git clone [the url]
cd Iterative-LLM-Based-NAS-with-Feedback-Memory
conda create -n tmp python=3.10 -y
conda activate tmp
pip install -r requirements.txt
```

Run the above inside the `tmp` environment. If you use a remote API, set the env var (e.g. `export SiliconCloud_Key="..."`) and do not commit the key.

## How to run

- **Full pipeline**: run `./run.sh`. 

## Output

Under `--output-dir` you get:

- **summary.json**: total iterations, best accuracy, whether target was reached, `results_history`
- **results.log**: per-iteration accuracy, success, and errors
- **generated_models/**: generated model code per iteration (e.g. `model_iter_*.py`)

## Notes

- Remote API requires network and a valid key; local large models need enough GPU memory
