# Prompt improvement pipeline MVP
_Exported on 2/1/2026 at 22:07:47 GMT+1 from Cursor (2.4.23)_

---

**User**

@prompt_improvement/Prompt.md @prompt_improvement 

根据prompt实现整个pipeline，代码放到 prompt_improvement里。实现一个MVP方案。不要用nn-gpt

---

**Cursor**

Let me first read all the prompt files in the `prompt_improvement` folder to understand the pipeline requirements.

I've reviewed all the prompt files. This is a prompt optimization pipeline for generating vision models. Before I create the plan, I have a few clarifying questions:

---

**User**

通过transformers库加载qwen3 8B

---

**Cursor**

明白了，使用 Transformers 库加载 Qwen3 8B 作为 LLM。现在我有足够的信息来创建计划。

---

**User**

Implement the plan as specified, it is attached for your reference. Do NOT edit the plan file itself.

To-do's from the plan have already been created. Do not create them again. Mark them as in_progress as you work, starting with the first one. Don't stop until you have completed all the to-dos.

---

**Cursor**

I'll implement the pipeline step by step. Let me start with the first component.

Let me verify the implementation and check for any linter errors.

All tasks completed. Here's a summary of the implemented MVP pipeline:

## Implemented Files

| File | Description |
|------|-------------|
| `llm_client.py` | Qwen3 8B model loader with `generate()` interface |
| `code_generator.py` | Uses LLM to generate PyTorch vision model code |
| `code_extractor.py` | Extracts and validates Python code from LLM output |
| `evaluator.py` | Orchestrates training evaluation via subprocess |
| `train_script.py` | Standalone training script for CIFAR-10 |
| `prompt_improver.py` | Improves prompts based on evaluation feedback |
| `pipeline.py` | Main orchestrator coordinating the entire cycle |
| `config.py` | Configuration parameters |
| `requirements.txt` | Dependencies |

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run with default settings
python pipeline.py

# Run with custom settings
python pipeline.py --target-accuracy 0.85 --max-iterations 15 --epochs 20

# Quick test run
python pipeline.py --test
```

## Pipeline Flow

```
┌─────────────────┐     ┌───────────────┐     ┌───────────┐
│ Code Generator  │ ──► │ Code Extractor│ ──► │ Evaluator │
│   (Qwen3 8B)    │     │               │     │ (CIFAR-10)│
└─────────────────┘     └───────────────┘     └─────┬─────┘
        ▲                                           │
        │                                           │
        │         ┌─────────────────┐               │
        └──────── │ Prompt Improver │ ◄─────────────┘
                  │   (feedback)    │
                  └─────────────────┘
```

Results are logged to `output/results.log` and prompt history to `output/prompts.log`.

---

**User**

我想更改@prompt_improvement/CodeGenerator.md 的prompt，让其拥有参考的code实现，也就是上一轮的实现。

然后Prompt Improvemer.md 那里则需要做相应的修改，让输出的prompt包含上一次迭代出来的code。

---

**Cursor**

我理解了，你想让每次迭代时把上一轮生成的代码作为参考传递给 Code Generator，这样可以增量改进而不是从头生成。我来修改相关文件。

修改完成。以下是主要变更：

## 修改内容

### 1. `CodeGenerator.md`
- 添加了 `## Reference Code (from previous iteration)` 部分
- 使用 `{reference_code}` 占位符

### 2. `Prompt Improver.md`
- 添加了要求：improved prompt 必须包含 reference code
- 输出格式新增 `improvement_suggestion` 字段
- 提示 prompt 需要保留 `{reference_code}` 占位符

### 3. `code_generator.py`
- 将 `INITIAL_PROMPT` 改为 `INITIAL_PROMPT_TEMPLATE`
- `generate()` 方法现在接受 `reference_code` 参数
- 新增 `update_reference_code()` 方法
- 第一轮迭代时显示 "No reference code available. This is the first iteration."

### 4. `prompt_improver.py`
- 更新模板说明，要求输出保留 `{reference_code}` 占位符
- 解析结果新增 `improvement_suggestion` 字段

### 5. `pipeline.py`
- 跟踪 `current_reference_code` 状态
- `run_iteration()` 返回生成的代码
- `improve_prompt()` 更新 reference code 用于下一轮迭代
- 每轮迭代会将上一轮的代码作为参考传递给 Code Generator

## 工作流程

```
第1轮: prompt_template + "No reference code" → 生成 code_v1
第2轮: improved_prompt_template + code_v1 → 生成 code_v2
第3轮: improved_prompt_template + code_v2 → 生成 code_v3
...
```

这样 LLM 可以基于上一轮的代码进行增量改进，而不是每次从头开始。

