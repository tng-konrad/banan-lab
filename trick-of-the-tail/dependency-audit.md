# Notebook import audit

Verified by importing every module in the isolated virtual environment `/Users/kbanachewicz/Documents/banan-lab/trick-of-the-tail/.venv/bin/python`, Python 3.12.8 on macOS arm64 (CPU `float64`). Full transitive pins are locked in [requirements.lock.txt](requirements.lock.txt); direct notebook and validation pins are in [requirements.txt](requirements.txt). No LLM server, external API, GPU accelerator or repository helper package is imported by the delivered notebook.

| Import | Actual use | Standard library? | Installable package and version pin | Import result and local version |
|---|---|---|---|---|
| `datetime` | Timestamp formatting | Yes | Python standard library | Passed; Python 3.12.8 |
| `pathlib` | Portable file and directory paths | Yes | Python standard library | Passed; Python 3.12.8 |
| `zoneinfo` | `Europe/Amsterdam` execution timezone | Yes | Python standard library | Passed; Python 3.12.8 |
| `hashlib` | SHA-256 verification of caches, calibration keys and evidence | Yes | Python standard library | Passed; Python 3.12.8 |
| `importlib.metadata` | Runtime package version inventory | Yes | Python standard library | Passed; Python 3.12.8 |
| `json` | Frozen manifests and full-precision JSON records | Yes | Python standard library | Passed; Python 3.12.8 |
| `os` | Execution mode (`TAILS_MODE`) and Matplotlib cache path | Yes | Python standard library | Passed; Python 3.12.8 |
| `platform` | Host architecture and OS provenance | Yes | Python standard library | Passed; Python 3.12.8 |
| `shutil` | Archive earlier same-named PNG renders | Yes | Python standard library | Passed; Python 3.12.8 |
| `sys` | Kernel Python version record | Yes | Python standard library | Passed; Python 3.12.8 |
| `tempfile` | Writable Matplotlib configuration directory | Yes | Python standard library | Passed; Python 3.12.8 |
| `time` | Monotonic elapsed-time guards and stage timings | Yes | Python standard library | Passed; Python 3.12.8 |
| `matplotlib` | Noninteractive `Agg` backend, `seaborn-v0_8` style and single-panel figures | No | `matplotlib==3.11.2` | Passed; 3.11.2 |
| `numpy` | Estimators, Bartlett HAC, trapezoidal quadrature, pivotal calibration and bootstraps | No | `numpy==2.5.4` | Passed; 2.5.4 |
| `pandas` | FRED CSV loading, tabular summaries and CSV sidecars | No | `pandas==2.3.3` | Passed; 2.3.3 |
| `psutil` | Process resident-memory (`RSS`) ceiling enforcement | No | `psutil==7.2.2` | Passed; 7.2.2 |
| `scipy.stats` | Chi-square survival function (`chi2.sf`) for Kupiec UC and Christoffersen IND diagnostics | No | `scipy==1.18.1` | Passed; 1.18.1 |
| `arch.univariate` | `ConstantMean`, `GARCH` (`o=0` and `o=1` GJR) and `StudentsT` volatility filtering | No | `arch==8.0.0` | Passed; 8.0.0 |
| `IPython.display` | Exact inline PNG and caption rendering (`Image`, `Markdown`, `display`) | No | `ipython==9.18.1` | Passed; 9.18.1 |

Aliases `np`, `pd` and `plt` refer to `numpy`, `pandas` and `matplotlib.pyplot`. Matplotlib bundles the `seaborn-v0_8` style without requiring the `seaborn` distribution. `statsmodels==0.15.0` is installed transitively by `arch==8.0.0` in [requirements.lock.txt](requirements.lock.txt) and is not imported directly in the notebook. All 20 AST import statements appear in the single first code cell, and every imported symbol is used.

Kernel and offline replay-validation tools, verified separately in `trick-of-the-tail/.venv`: `ipykernel==6.31.0`, `nbformat==5.10.4`, `nbclient==0.11.0`, `jupyter-client==8.10.0`, and `Pillow==13.0.0`. JupyterLab is an optional interactive front end and is not required for offline replay acceptance.

