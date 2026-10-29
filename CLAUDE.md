# EN4554 Assignment 1: Decisions and Plan

Spec: [EN4554_Assignment.pdf](Sem07_Deep_Learning_for_Vision/Assignment_1/EN4554_Assignment.pdf)
Code: `Sem07_Deep_Learning_for_Vision/Assignment_1/part1.ipynb`, `part2.ipynb`
Report: `Sem07_Deep_Learning_for_Vision/Assignment_1/latex_report/`

## Ownership

| Part | Owner | Deliverables |
|---|---|---|
| Part 1: MiLoRA on a VLM | **Sulojan** | `part1.ipynb`, Part 1 report (≤ 3 pages, `part1.tex`) |
| Part 2: Tool-calling agent | **Praveen** | `part2.ipynb`, Part 2 write-up (≤ 1 page, `part2.tex`) |
| Shared | Both | Final review and combined `main.tex` (pulls in both parts with `\input`) |

## Global constraints (from the spec)

- Everything must run on a free Colab T4 (16 GB). Use **fp16 and never bf16**. If you see NaNs, lower the learning rate.
- Use **seed 42 everywhere** and report it.
- Save checkpoints and adapters to the shared Google Drive folder, so a Colab disconnect doesn't lose work.
- Always use data subsets, and state the exact split and size in the report.

## Part 1: MiLoRA on SmolVLM (decided)

- **LoRA variant: MiLoRA** (Wang et al., NAACL 2025). The baselines are plain LoRA and PiSSA. PiSSA is built into `peft` (`init_lora_weights="pissa"`). MiLoRA uses a custom SVD initialisation on top of `peft`: it fills A and B from the *minor* r singular components and subtracts them from W.
- **VLM: `HuggingFaceTB/SmolVLM-256M-Instruct`.** Fall back to SmolVLM-500M only if the 256M baseline is too weak. Don't use Florence-2.
- **Hypothesis:** the principal subspace of a VLM's pretrained weights holds general visual-language knowledge. Fine-tuning on TextVQA through the minor components (MiLoRA) should therefore keep more zero-shot ability on a held-out VQA task than LoRA or PiSSA, while matching them on TextVQA.
  - TODO: pin the hypothesis to the specific table or figure in the MiLoRA paper that supports it (report item 2).
- **Datasets:**
  - Target task: TextVQA, about 3k training examples and 500 validation examples.
  - Forgetting probe: about 500 validation examples from VQAv2 or OK-VQA. It is never trained on; we only measure the accuracy drop.
  - Note in the report that SmolVLM likely saw TextVQA during pretraining.
- **Metrics tied to the claim:**
  1. VQA accuracy on TextVQA (target task).
  2. Δ accuracy on the probe task relative to the zero-shot baseline (forgetting).
  3. Subspace overlap of ΔW with the top-k singular vectors of W₀.
- **Runs:** zero-shot baseline, LoRA, PiSSA and MiLoRA, all with the same rank, learning rate, steps and seed.
- **Bonus:** apply MiLoRA separately to the vision encoder (SigLIP), the connector and the language model, and compare their singular-value spectra. The question is whether vision weights and text weights store knowledge differently.

## Part 2: Tool-calling agent (decided, open to edits)

| Choice | Pick |
|---|---|
| LLM | `Qwen/Qwen2.5-0.5B-Instruct` (tool template `<tool_call>{json}</tool_call>`) |
| Tools (4) | `calculator`, `unit_convert`, `get_weather(city)` with fictional cities, `lookup_order(order_id)` with a fake database |
| Data | Built from templates: about 1,000–1,500 training and 100 test examples, with test phrasings separate from training. About 20% need no tool and a few need two steps. |
| LoRA | r=16, α=32, all linear layers, lr 2e-4, 2–3 epochs, fp16, loss only on assistant turns |
| Evaluation | Call validity, call accuracy, abstention, task success. At least 10 failures per model, grouped into wrong format, wrong tool, wrong arguments, unnecessary call, and ignored tool result. |

## Collaboration workflow

- Code lives in this GitHub repo. Open the notebooks in Colab straight from GitHub.
- Each person edits only their own notebook and `.tex` file, to avoid merge conflicts.
