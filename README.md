# Decision models, meet the document pile

The deliverable is **`decisionrag_ollama_demo.ipynb`**, with implementation, explanations, original measured results and publication visuals inline. The companion **`post-plan.md`** follows the same main section headings and contains explicit placeholders, captions and readings for every graph/table. Both follow the supplied `post-guide.md`; the plan remains a separate outline. The independent critique and revision decisions are in `docs/post-plan-critique.md`; `graphs/score_resolution_audit.json` exposes the qualified tied-score diagnostic.

The completed companion article is **[`post.md`](post.md)**. It embeds all twelve numbered visuals, follows the notebook's main headings, and reports the measured findings with the critique's qualifications. Its local notebook link should be replaced with the verified public URL before publication.

## Rerun the delivered notebook

The current default is **`CFG.mode = 'replay'`**, **`CFG.study_profile = 'mini_3h'`**, **`CFG.run_budget_hours = 3.0`**. Restart the kernel and Run All to recompute the saved study's metrics, 10,000-draw paired dialogue bootstrap, figures and tables **without new model calls**. Replay runtime is never reported as inference speed. The notebook imports no project helper scripts.

Under **Setup**, edit the fields inside `class CFG`, then restart and Run All. `CFG.experiment` holds the model, retrieval, context and bootstrap settings; the other fields hold paths, prompts, display settings and asset specifications. Plotting uses `seaborn-v0_8` with `CFG.img_dim1` × `CFG.img_dim2` inches (9.5 × 6.5 by default), exported at `CFG.export_dpi` (200). No plotting function overrides that size, and no separate seaborn installation is needed.

All functions and helper classes are in **Utils**, grouped into boilerplate/evidence files, corpus preparation, judgment requests/local serving, ranking/navigation, experiment orchestration, metrics/uncertainty, and plotting/publication. Every code cell has an immediate explanation. The eleven narrative sections use top-level headings with the same titles and order as the post; Setup and Utils contain notebook-specific mechanics.

```bash
uv venv .venv --python python3
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python jupyterlab
.venv/bin/jupyter lab decisionrag_ollama_demo.ipynb
```

The corpus and tokenizer download separately into `data/` on first use and are SHA256-verified before reuse. Full inference evidence is external, in `results/gate.json`, `results/pilot.json` and `results/evaluation.json`. Include those files when sharing the measured notebook; the source data can be downloaded again. No corpus body or full inference traces are encoded into notebook source.

For a **fresh live study**, change `CFG.mode` to `'evaluate'`, start a compatible Ollama server at `http://127.0.0.1:11434`, and separately install the models:

```bash
ollama pull nimble:9b-q8_0
ollama pull qwen3.5:9b-q8_0
```

The three-hour live cap includes checks, warmup, the eight-query pilot, adapter diagnostics and twelve-query evaluation. The pilot must authorize evaluation with a 25% runtime reserve. Both active and wall deadlines apply; keep the laptop awake. The full profile remains available but is outside the measured restricted-corpus result.

## Numbered publication visuals

Every displayed graph and table is saved in **`./graphs`**. Numbering follows one sequence in notebook display order, including tables; the suffix `0111` is literal.

- Plots: `graph_<number>_0111.png`.
- Tables: `post_<number>_0111.png`.
- The first table is visual 3: `post_3_0111.png`.
- Twelve visuals total: eight plots and four tables, with JSON/CSV sidecars, `captions.md`, `manifest.json` and an offline `index.html` gallery.

The post outline links those exact files in the corresponding sections. The detailed historical evidence gallery remains under `artifacts/decisionrag/evaluation-20261030T222218Z/post_assets/`. The exact original live executed notebook is preserved under `artifacts/decisionrag/notebook-snapshots/v5.6-live-evaluation-20261030T222218Z.ipynb`.

## What the completed live study found

All four methods searched the same **32 pages / 329 complete passages**, with **eight training pilot queries** and **twelve validation queries from twelve distinct dialogues**, three per domain. All 32 pilot and 48 evaluation method-query combinations were valid. The whole notebook took **107.1 minutes**; evaluation took **60.2 minutes**. Timing-integrity and residency checks passed.

| Method | Recall@1 | Recall@10 | Median query seconds |
|---|---:|---:|---:|
| A: BM25 | 33.3% | 66.7% | 0.003 |
| B: BM25 + Nimble judging | 58.3% | 83.3% | 107.809 |
| C: Hierarchy + Nimble | 41.7% | 91.7% | 122.646 |
| D: Hierarchy + Qwen | 41.7% | 83.3% | 65.551 |

The expected A < B < C ordering did not hold at rank one. C/D had equal observed Recall@1 with a broad paired interval (−33.3 to +33.3 percentage points); their median-time ratio was 1.87 (95% bootstrap interval 1.80–2.07). Nimble's local hierarchy did not demonstrate an efficiency advantage. These twelve-query results are exploratory.

This implements the page→passage idea from *Retriever-Free Retrieval: Navigating Documents with Decision Models* ([paper](https://arxiv.org/abs/2610.35868v1)). It does not reproduce Jev's weights, shared-forward execution or hosted latency. Help-center routing, retrieval memory, a modern dense baseline and answer generation are absent. Scores describe this restricted task and these models/interfaces, not decision models in general.

## Authoring and validation

`notebook_src/` and `scripts/build_notebook.py` are authoring tools, not runtime notebook dependencies. The builder places code, narrative and visuals in the teaching order. Every displayed table is exported as a numbered image; diagnostic status uses prose to avoid untracked tables.

```bash
.venv/bin/python scripts/build_notebook.py
.venv/bin/python scripts/validate_delivery.py --write-executed
```

Acceptance runs a fresh isolated Jupyter kernel with networking and repository reads blocked, using copied data/results only. It independently recomputes recall and failures from raw rankings, checks the exact notebook/post heading alignment and visual order, verifies every post placeholder and filename, and compares inline PNG bytes with `./graphs`. Visual inspection is separate. The raw inference evidence remains unchanged by replay.

Data: [MultiDoc2Dial](https://doc2dial.github.io/multidoc2dial/), [IBM repository](https://github.com/IBM/multidoc2dial). Source archive SHA256: `f0c034c249663d7b3cb08b19cf2cc2c3d101372485be982621d4711931a1ce00`. Tokenizer: [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B), Apache 2.0. Archived administrative content is benchmark evidence, not current advice.
