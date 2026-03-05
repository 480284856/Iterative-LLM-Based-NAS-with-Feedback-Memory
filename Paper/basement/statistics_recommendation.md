# Statistics to Report for Publication

This document recommends which statistics to compute and report from your **accuracy-across-iterations** data (`output/acc.txt`) for a scientific paper on the prompt-improvement pipeline (iterative code generation → CIFAR-10 evaluation → prompt improvement).

---

## 1. Descriptive statistics

Report these so readers can compare with other methods and reproduce the scale.

| Statistic | What to report | Purpose |
|-----------|----------------|---------|
| **N** | Total number of iterations (e.g. 303) | Sample size |
| **Mean ± SD** | Overall mean and standard deviation of accuracy | Central tendency and variability |
| **Median** | Median accuracy | Robust to outliers |
| **Min / Max** | Minimum and maximum accuracy | Range of outcomes |
| **Final accuracy** | Accuracy at the last iteration | Primary end-point |
| **Best accuracy** | Maximum accuracy over all iterations (and at which iteration) | Best achievable by the pipeline |

Optionally, report **by phase** (e.g. first 50 vs middle vs last 50 iterations) to show improvement over time: mean ± SD for “early”, “mid”, “late”.

---

## 2. Improvement over iterations (trend)

Your design is a **single long run** of 300+ iterations, so “improvement” is a trend within that run, not a comparison across independent runs.

- **Correlation**: Report **Spearman’s ρ** (and p-value) between *iteration index* and *accuracy*. Spearman is robust to outliers and non-linearity.
- **Linear trend**: Fit **accuracy ~ iteration**. Report:
  - **Slope** (with 95% CI, e.g. via bootstrap): “accuracy increased by X per 100 iterations.”
  - **R²** of the linear model.
- **First vs last block**: Compare **mean accuracy in the first K iterations** vs **mean accuracy in the last K** (e.g. K = 50). Report:
  - **Difference in means** (last − first).
  - **95% CI** for that difference (bootstrap over the two blocks).
  - Optional: **relative improvement** = (mean_last − mean_first) / mean_first.

These give a clear, publishable story: “accuracy improved significantly over iterations (ρ = …, p &lt; …; slope … per 100 it., 95% CI […]); mean accuracy in the last 50 iterations was … points higher than in the first 50 (95% CI […]).”

---

## 3. Convergence and stability

- **Best accuracy and iteration**: Report the **maximum accuracy** and the **iteration** at which it was first reached.
- **Stability in late phase**: For the last K iterations (e.g. 50), report **mean ± SD** (and optionally **IQR**). This shows whether the pipeline has converged to a stable level.
- **Learning curve**: Keep (or add) a figure: **accuracy vs iteration** with a smoothed curve (e.g. moving average) and optionally a linear or LOWESS trend line. This is standard in iterative optimization papers.

---

## 4. Outliers and sensitivity

You have a few very low values (e.g. ~0.10, ~0.28, ~0.31). For transparency:

- **Report**: Number (and optionally list) of “failure” or outlier iterations (e.g. accuracy &lt; 0.35 or &lt; 0.40).
- **Sensitivity analysis**: Report key metrics **with and without** those outliers (e.g. mean, final accuracy, first-vs-last difference, Spearman ρ). This shows the main conclusions are not driven by a few bad runs.

---

## 5. What to avoid (single-run design)

- **Independent-samples t-test** between “early” and “late” as if they were independent groups: the data are sequential and autocorrelated, so standard t-test assumptions are violated. Prefer **bootstrap CI for the difference** and/or **regression/correlation** with iteration.
- **Reporting only final accuracy** without trend or variability: reviewers will want to see that improvement is systematic (trend) and that you report uncertainty (CI, SD, or sensitivity).

---

## 6. Suggested table for the paper

| Metric | Value |
|--------|--------|
| Iterations | 303 |
| Mean accuracy (all) | … ± … |
| Median accuracy | … |
| Best accuracy (iteration) | … (it. …) |
| Final accuracy | … |
| First 50 it. mean ± SD | … ± … |
| Last 50 it. mean ± SD | … ± … |
| Improvement (last − first 50) | … (95% CI: …, …) |
| Spearman ρ(iteration, accuracy) | …, p = … |
| Slope (acc. per 100 it.) | … (95% CI: …, …) |
| Outliers (acc &lt; 0.40) | n = … |

Run the script `Paper/compute_paper_statistics.py` to fill this table from `output/acc.txt`.
