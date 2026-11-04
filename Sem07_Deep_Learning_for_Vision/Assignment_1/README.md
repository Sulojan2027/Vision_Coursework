# EN4554 Assignment 1: LoRA for Vision-Language Models and Tool-Calling Agents

**Sulojan R. (220626V)** · **Praveen V.V.J. (220491B)**
Department of Electronic and Telecommunication Engineering, University of Moratuwa

The report is [`latex_report/main.pdf`](latex_report/main.pdf).

## Contents

| Path | Description |
|---|---|
| [`part1.ipynb`](part1.ipynb) | Part 1: AdaLoRA on a vision-language model (SmolVLM-256M) |
| [`part2.ipynb`](part2.ipynb) | Part 2: LoRA fine-tuning a small LLM for tool calling (Qwen2.5-0.5B) |
| [`latex_report/`](latex_report/) | Report source (`main.tex`), compiled PDF and figures |
| [`requirements.txt`](requirements.txt) | Python dependencies |
| [`EN4554_Assignment.pdf`](EN4554_Assignment.pdf) | Assignment brief |

## Part 1: Does AdaLoRA's rank allocation transfer to a VLM?

AdaLoRA (Zhang et al., ICLR 2023) allocates a fixed rank budget across weight matrices by importance. In NLP, it gives more rank to FFN matrices and to top layers. We apply AdaLoRA to SmolVLM-256M-Instruct, with one shared budget across the vision encoder, connector and language model. We compare a text-reading task (TextVQA) with an outside-knowledge task (OK-VQA), using 2,500 training and 500 evaluation questions per task.

| | TextVQA | OK-VQA |
|---|---|---|
| Zero-shot | 37.34 | 36.68 |
| LoRA, r = 8 | **46.62** | **42.92** |
| AdaLoRA, 12 → 8 (same budget) | 43.04 | 37.04 |

- **Allocation is task-invariant.** The vision encoder's share of the rank is about the same for both tasks (22.0% vs 24.1%). The two allocations are strongly correlated (Spearman ρ = 0.83).
- **FFN > attention transfers** to both the vision encoder and the language model.
- **Top layers > bottom layers holds only in the vision encoder.** In the language model the pattern reverses: its last ~30% of layers are pruned to almost zero.
- **AdaLoRA is significantly worse than LoRA** at this short, Colab-scale training schedule (paired-bootstrap 95% CIs exclude 0).
- **Bonus (exploratory).** Plain LoRA trained with AdaLoRA's discovered allocation beats uniform LoRA (+1.5 and +1.2 points).

## Part 2: Tool calling with a small LLM

A minimal agent loop (generate → parse the tool call → execute → feed back the result) around Qwen2.5-0.5B-Instruct with four tools: `calculator`, `unit_convert`, `get_weather` and `lookup_order`. The lookup tools use made-up data. We train with LoRA (r = 16) on 1,200 template-generated requests, with loss on assistant turns only. Both models are evaluated on 100 held-out requests whose phrasings differ from the training set.

| Metric | Baseline | LoRA |
|---|---|---|
| Call validity | 83.6% | **100.0%** |
| Call accuracy | 68.5% | **90.4%** |
| Abstention | 59.3% | **96.3%** |
| Task success | 55.0% | **92.0%** |

LoRA fixed 49 of the 56 baseline failures, left 7 and introduced 1. The remaining errors come from over-fitting to the training templates.

## Reproducing

1. Open a notebook in Google Colab with a T4 GPU, then run all cells.
2. Both notebooks use fp16 and seed 42.
3. `part1.ipynb` mounts Google Drive and caches the model, data subsets, results, adapters and figures under `MyDrive/EN4554_A1/`. A restarted session skips runs that have already finished. A full run takes about 45 minutes of T4 time.
4. To build the report, run `cd latex_report && latexmk -pdf main.tex`.
