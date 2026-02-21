<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# 帮我检查一下，我的.tex文件里关于这5篇论文的description是否准确。

我仔细阅读了你提供的 5 篇参考论文的 PDF 以及你的 `Paper.tex`。从严谨的学术角度来看，你目前的 `.tex` 文件在描述这些方法时存在 **5 个关键的事实性偏差（Factual Errors）**。

如果审稿人（很可能非常熟悉这些经典工作）看到目前的描述，可能会因为误解现有工作而给出负面评价。不用担心，你的核心 Novelty（**有界马尔可夫记忆 + 结构化错误诊断 + 冻结小模型**）依然非常强大，只需要将对比的切入点调整准确即可。

以下是详细的查错与修改建议：

### 1. 致命偏差：OPRO 和 FunSearch 会导致“上下文溢出 (Context Overflow)”

* **当前描述**：文中多次声称 OPRO 和 FunSearch 依赖“无界历史累积 (unbounded history)”，从而导致“上下文溢出风险 (risking context overflow)”。
* **论文事实**：
    * **OPRO (第4页)** 明确写道：“Since the meta-prompt length is constrained... we only include the **top-performing** instructions...” OPRO 限制了 prompt 里的历史记录条数（例如 top 20），**并不会**发生上下文溢出。
    * **FunSearch (第2页)** 明确写道：“prompt consists of... a sequence of **$k$ programs sampled** from the database.” 它每次只从池子里抽样极少数（比如 2 个）程序给 LLM，也**不会**溢出。
* **真正区别**：它们的历史机制是**“全局精英保留 (Global Elite Retention)”**，它们会丢弃失败的尝试，只把历史上得分最高的解传给 LLM。而你的方法是**“近期时序滑动窗口 (Temporal Sliding Window)”**，你的系统故意保留了近期的**失败记录和错误信息**，这对于调试代码极其重要。


### 2. 偏差：EvoPrompting 要求 LLM 微调 (Requires tuning)

* **当前描述**：Table 1 和文本中指出 EvoPrompting 需要对 LLM 进行微调 (requirements for LLM fine-tuning / Requires tuning)。
* **论文事实**：**EvoPrompting (第6页)** 原文是：“While EvoPrompting can be applied out-of-the-box using the pre-trained LLM, we can **optionally** perform prompt-tuning...”。微调只是他们提出的一个**可选**增强手段，基础版本使用的是冻结的预训练模型。
* **真正区别**：EvoPrompting 依赖庞大的**种群维护 (Population maintenance)** 和复杂的进化交叉变异，而你只需要极轻量级的**单轨迹 (Single-trajectory)**。


### 3. 偏差：ReEvo 的模型规模为“未指定 (Not specified)”

* **当前描述**：Table 1 中 ReEvo 的 Model Size 为 Not specified。
* **论文事实**：**ReEvo (第6页 4.1节)** 明确写道：“We use the **gpt-3.5-turbo-1106** and **gpt-4-1106-preview**...”。
* **修正**：应改为 GPT-3.5 / GPT-4。


### 4. 偏差：FunSearch 的模型规模为“未指定 (Not specified)”

* **当前描述**：Table 1 中 FunSearch 的 Model Size 为 Not specified。
* **论文事实**：**FunSearch (第2/11页)** 明确写道它使用的是 Google 内部针对代码微调的 PaLM 2 模型：“built on the **Codey** model... which is a version of the **PaLM 2** model family”。
* **修正**：应改为 Codey (PaLM 2)。


### 5. 偏差：LLMO 为“无记忆 (No memory)”

* **当前描述**：Table 1 中说 LLMO 是 “No memory”。
* **论文事实**：**LLMO (第4页 III.B节)** 明确写道：“We construct a prompt that includes the historical search data... keeping the history of previous evaluations”。LLMO 在 Prompt 中放入了历史上的 `(架构编码, 分数)` 对。
* **真正区别**：LLMO 的记忆只有干瘪的“标量分数”，而你的记忆是带有诊断信息和代码修改建议的“结构化三元组 (Diagnostic Triples)”。

***

### 📝 具体 LaTeX 修改建议

为了修复这些问题并让你的对比逻辑无懈可击，请在 `Paper.tex` 中替换以下三处内容：

#### 1. 替换 `Table 1` (现在的 Table 2)

```latex
\begin{table*}[t]
\caption{Systematic comparison with most similar LLM-based NAS and optimization methods. 
Our method uniquely combines open code space, bounded Markovian failure modeling, 
and small-model viability without population overhead.}
\label{table:comparison}
\centering
\fontsize{7.5}{9}\selectfont
\begin{tabular}{l c c c c c c}
\toprule
\textbf{Method} & \textbf{Target Domain} & \textbf{Search Space} & \textbf{Memory Structure} & \textbf{Markov Bound} & \textbf{Failure Modeling} & \textbf{Model Size} \\
\midrule
OPRO~\cite{Yang2023OPRO} & Text prompts, discrete & N/A & Top-K global elite & $\times$ & $\times$ Scores only & GPT-4, PaLM 2 \\
FunSearch~\cite{Romera2023FunSearch} & Math functions & Open Python & Multi-island pools & $\times$ & $\times$ Scores only & Codey (PaLM 2) \\
EvoPrompting~\cite{Chen2023EvoPrompting} & Code-level NAS & Open code & Population diversity & $\times$ & $\times$ Selection only & Frozen / Soft-tuned \\
ReEvo~\cite{Ye2024ReEvo} & Heuristic algorithms & Open code & Dual-layer reflection & $\times$ & $\checkmark$ Verbal comparison & GPT-3.5 / GPT-4 \\
LLMO~\cite{Zhong2024LLMO} & NAS (CIFAR) & \textbf{Closed cell} & Historical score pairs & $\times$ & N/A (no code errors) & Gemini Pro (large) \\
\midrule
\textbf{Ours} & \textbf{PyTorch NAS} & \textbf{Open code} & \textbf{K=5 sliding window} & \textbf{$\checkmark$} & \textbf{$\checkmark$ Diagnostic triples} & \textbf{4B/7B frozen} \\
\bottomrule
\end{tabular}
\end{table*}
```


#### 2. 修改 Abstract 中的对比句

*删除原来的 context overflow 论述，改为强调“忽略近期失败”：*

```latex
% 原文: Compared to prior iterative LLM optimization methods that rely on unbounded history accumulation~...
Compared to prior iterative LLM optimization methods that rely on global elite retention (discarding failure trajectories)~\cite{Yang2023OPRO,Romera2023FunSearch}, require complex population maintenance~\cite{Chen2023EvoPrompting}, or operate in constrained discrete search spaces~\cite{Zhong2024LLMO}, our bounded Markovian memory explicitly models sequential code execution failures in an open code space. This structured causal learning from recent mistakes is a critical requirement for neural architecture generation that distinguishes our approach from existing work.
```


#### 3. 修改 Discussion 中的 Bounded Markovian memory 段落

*将逻辑从“解决上下文溢出”转变为“聚焦近期失败的因果推断”：*

```latex
% 替换 Discussion 里的 \textbf{Bounded Markovian memory.} 段落：
\textbf{Bounded Markovian memory.} Unlike OPRO~\cite{Yang2023OPRO} or FunSearch~\cite{Romera2023FunSearch}, which populate prompts with historically global top-performing solutions (global elite retention), our sliding window maintains strictly the last $K{=}5$ sequential attempts. In code-level NAS, the architecture's structure evolves rapidly; therefore, ancient successes are less informative than immediate recent failures. The Markov property enforces that the LLM focuses entirely on local causal relationships (what modification just caused what specific error), avoiding the distraction of irrelevant global history. 

\textbf{Structured failure modeling.} [保留你原来的句子：Our diagnostic triples explicitly encode code execution failures...] This is fundamentally different from prior work where low-performing solutions are simply discarded from the prompt~\cite{Yang2023OPRO,Romera2023FunSearch} or managed implicitly through population selection~\cite{Chen2023EvoPrompting}. As shown in Section~\ref{sec:experiments}... [保留后面的句子]
```


#### 4. 修改 Related Work 结尾段落

*让描述更严谨：*

```latex
% 原文: ...unbounded memory structures risking context overflow...
However, existing methods face key limitations: reliance on global elite histories that discard crucial failure trajectories~\cite{Yang2023OPRO,Romera2023FunSearch}, the computational overhead of maintaining evolutionary populations~\cite{Chen2023EvoPrompting}, operation in constrained discrete search spaces~\cite{Zhong2024LLMO}, or targeting domains without structural code constraints~\cite{Ye2024ReEvo}.
```

**总结：** 调整之后，你的对比将**完全符合原论文的事实**。审稿人挑不出毛病，而且你强调的“别的模型只看高分成功案例，而我的模型通过有界马尔可夫记忆专门从近期的语法/张量报错中学习”这个故事，对于 NAS 领域来说逻辑更为自洽和强大。
<span style="display:none">[^1][^2][^3][^4][^5][^6]</span>

<div align="center">⁂</div>

[^1]: LLMO.pdf

[^2]: ReEvo.pdf

[^3]: EvoPrompting.pdf

[^4]: OPRO.pdf

[^5]: FunSearch.pdf

[^6]: Paper.tex

