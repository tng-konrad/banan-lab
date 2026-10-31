# Notebook import audit

Checked interpreter: `/Users/kbanachewicz/Documents/banan-lab/.venv/bin/python`; Python **3.12.8**.

All import statements, including imports inside functions, were extracted from the current notebook and tested. Availability applies to this project environment; a notebook using a different kernel may have different packages.

**Current blocking imports:** `rank_bm25` and `tokenizers` are missing. No package installation was performed.

## Python standard library

All imports below come with Python, require no separate package installation, and were successfully imported locally.

| Import | Role in the notebook | Available locally |
|---|---|---|
| `__future__` | `annotations`: postponed annotation evaluation; a Python language feature. | Yes |
| `copy` | Copy configurations and captured results without changing their originals. | Yes |
| `contextlib` | `contextmanager`: temporarily use a recorded configuration and restore the current one. | Yes |
| `collections` | `Counter`, `defaultdict`: count outcomes and group pages, passages and timings. | Yes |
| `dataclasses` | `dataclass`: imported but unused in the current notebook. | Yes |
| `datetime` | `datetime`, `timezone`: UTC timestamps for requests and runs. | Yes |
| `hashlib` | SHA256 checksums for downloaded data, frozen inputs and exported assets. | Yes |
| `html` | Escape text when generating the HTML evidence galleries. | Yes |
| `importlib.metadata` | Read installed package versions for reproducibility receipts. | Yes |
| `json` | Read/write datasets, schemas, request traces and result files. | Yes |
| `math` | Check numeric scores are finite and perform basic scalar calculations. | Yes |
| `pathlib` | `Path`: locate caches, result files and export directories. | Yes |
| `platform` | Record Python version and operating-system/architecture information. | Yes |
| `random` | Seeded ordering of methods in the experiment schedule. | Yes |
| `re` | Regular expressions for lexical tokenization and text/title processing. | Yes |
| `statistics` | Compute training-pilot mean timings for the runtime forecast. | Yes |
| `sys` | Interpreter/system access; imported but unused in the current notebook. | Yes |
| `time` | Monotonic and wall clocks, durations and request deadlines. | Yes |
| `tracemalloc` | Measure Python allocation peaks; this is not total RAM or GPU memory. | Yes |
| `unicodedata` | Normalize Unicode text for exact-text sensitivity checks. | Yes |
| `zipfile` | Read the separately downloaded dataset ZIP archive. | Yes |
| `textwrap` | Wrap text to fit exported figures and tables; imported inside a helper. | Yes |
| `io` | `BytesIO` for reading archive bytes; imported inside data preparation. | Yes |

## Third-party packages

These are not bundled with Python. Already-installed packages do not need reinstalling merely to import them; matching the original environment is a separate concern.

| Import | Installable package | Required pin | Local status/version | Role |
|---|---|---|---|---|
| `httpx` | `httpx` | 0.28.1 | Available: 0.28.1 | HTTP downloads and requests to the local Ollama server. |
| `matplotlib.pyplot` | `matplotlib` | 3.9.4 | Available: 3.11.2; differs from pin | `plt`: construct and save plots and rendered table figures. |
| `matplotlib.patches` | `matplotlib` | 3.9.4 | Available: 3.11.2; differs from pin | `FancyArrowPatch`, `FancyBboxPatch`: arrows and boxes in the methods diagram. |
| `numpy` | `numpy` | 2.0.2 | Available: 2.5.4; differs from pin | `np`: numeric arrays, bootstrap resampling and summary statistics. |
| `pandas` | `pandas` | 2.3.3 | Available: 3.1.0; differs from pin | `pd`: organize results and write tabular CSV sidecars. |
| `rank_bm25` | `rank-bm25` | 0.2.2 | **Missing** | `BM25Okapi`: the lexical retrieval baseline and candidate shortlist. |
| `tokenizers` | `tokenizers` | 0.22.2 | **Missing** | `Tokenizer`: count tokens using the separately downloaded pinned tokenizer. |
| `IPython.display` | `ipython` | Not directly pinned | Available: 9.18.0 | `display`, `Markdown`, `Image as NotebookImage`: inline text and visual output. |

`matplotlib.pyplot` and `matplotlib.patches` belong to the same `matplotlib` installation. `np`, `pd`, `plt` and `NotebookImage` are aliases, not extra packages. The pip name `rank-bm25` differs from the Python import `rank_bm25`. IPython is usually installed as a dependency of the notebook kernel.

## Notebook and execution tools

These support notebook execution but are not directly imported by notebook code. `nbformat` and `nbclient` are used by the project execution/validation scripts. JupyterLab is one possible frontend; VS Code or another compatible frontend can also run the notebook.

| Package | Local version | requirements.txt pin |
|---|---|---|
| `ipykernel` | 7.4.0 | 6.31.0 |
| `nbformat` | **Missing** | 5.10.4 |
| `nbclient` | **Missing** | 0.10.2 |
| `jupyterlab` | **Missing** | Not listed |

## Installation commands

To match the declared project dependencies, including the missing imports, run from the project directory:

```bash
uv pip install --python .venv/bin/python -r requirements.txt
```

That command may replace the newer local versions with the pinned versions. For only the two missing notebook imports:

```bash
uv pip install --python .venv/bin/python rank-bm25==0.2.2 tokenizers==0.22.2
```

If JupyterLab is the desired frontend:

```bash
uv pip install --python .venv/bin/python jupyterlab
```

`uv` is available locally. This virtual environment currently has no `pip` module, so `.venv/bin/python -m pip ...` will not work as-is.

Select this `.venv` as the notebook kernel. To check the interpreter inside Jupyter:

```python
import sys
print(sys.executable)
```

Ollama and the model weights are separate runtime prerequisites for live inference, not Python imports. No Python `ollama` SDK or local project module is imported by this notebook. Replay still needs the Python packages, external cached data and saved result files.

Machine-readable details, including module locations, notebook cell numbers and import failures, are in `results/import_audit.json`.
