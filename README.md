# Iterative-LLM-Based-NAS-with-Feedback-Memory

Iterative prompt-improvement pipeline: uses an LLM to generate model code, then trains and evaluates it. Supports CIFAR-10, CIFAR-100, and ImageNette. You can turn off the improver or reference via flags for ablation runs.

## Environment and dependencies

- **Python**: 3.10 recommended
- **Conda**: use env `nngpt` (`conda activate nngpt`)
- **Dependencies**: `pip install -r requirements.txt`
- **Data**: default `./data`; CIFAR is downloaded automatically; ImageNette is fetched from the URL in the script

## Installation

```bash
git clone https://github.com/480284856/Iterative-LLM-Based-NAS-with-Feedback-Memory.git
cd Iterative-LLM-Based-NAS-with-Feedback-Memory
pip install -r requirements.txt
```

Run the above inside the `nngpt` environment. If you use a remote API, set the env var (e.g. `export SiliconCloud_Key="..."`) and do not commit the key.

## How to run

### Option 1: Call the pipeline directly

```bash
python pipeline.py --model <HuggingFace_model_name> --dataset <cifar10|cifar100|imagenette> \
  --max-iterations <N> --target-accuracy <0~1> --output-dir <output_directory>
```

- Add `--remote` when using a remote API
- Common args: `--epochs`, `--batch-size`, `--history-size`
- Ablation: `--no-improver`, `--no-reference`, `--no-history`

### Option 2: Use the scripts

- **Full pipeline**: run `./run.sh` (activates conda, creates `output` and `log`, runs `pipeline.py` per script config). Edit the `cd` path in the script for your machine.
- **Ablation (no improver, no reference)**: run `./run_ablation.sh` (default: DeepSeek-Coder-6.7B + CIFAR-100, 100 iterations). Results under `./output/*_ablation_*`, logs under `./log/*_ablation_*.log`. Edit paths in the script for your machine.

## Output

Under `--output-dir` you get:

- **summary.json**: total iterations, best accuracy, whether target was reached, `results_history`
- **results.log**: per-iteration accuracy, success, and errors
- **generated_models/**: generated model code per iteration (e.g. `model_iter_*.py`)

## Notes

- Remote API requires network and a valid key; local large models need enough GPU memory
- Update the paths in `run.sh` and `run_ablation.sh` for your machine
