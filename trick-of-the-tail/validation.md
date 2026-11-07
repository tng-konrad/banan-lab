# Execution and validation record

Status: complete local empirical study and companion publication draft; no external publication or messaging performed. Final delivery checks completed on 7 November 2026, `Europe/Amsterdam`.

## Scope and study identity

- **Notebook**: [trick_of_the_tail.ipynb](trick_of_the_tail.ipynb)
- **Frozen protocol**: [frozen-study.json](frozen-study.json) (reviewed before live execution in [plan.md](plan.md) and [plan-critique.md](plan-critique.md))
- **Completed live run**: `runs/20261107T155047630280+0100` (pointed to by [latest-run.json](latest-run.json))
- **Isolated environment**: `/Users/kbanachewicz/Documents/banan-lab/trick-of-the-tail/.venv` (`Python 3.12.8`, pinned in [requirements.lock.txt](requirements.lock.txt) and audited in [dependency-audit.md](dependency-audit.md))
- **Explicit user scope**: real FX empirical data only; Section 6's synthetic null-size and power Monte Carlo study from [trick-of-the-tail-proper-plan-v2.md](sources/trick-of-the-tail-proper-plan-v2.md) was omitted per user instruction, while Section 5's finite-sample `B = 5,000` IID unit-scale Pareto pivotal calibration (`gamma_0 = 0.3`) was retained solely to supply critical values and p-values for the empirical series.

## Pre-freeze fixtures and pilot admission

All nine deterministic correctness and anti-leakage fixtures in [fixture-checks.json](fixture-checks.json) passed prior to study freeze, verifying:
1. Analytic constant-scale Pareto threshold recovery,
2. Exact parity (`rtol = atol = 1e-12`) between vectorized and scalar textbook local common-threshold ratio Hill estimators,
3. Signed/zero/tie strict-exceedance (`X_t > b_hat`) invariance,
4. Exact multiplicative scale invariance under positive rescaling,
5. Explicit `NaN` propagation when nonpositive thresholds or empty exceedances occur,
6. Analytical Bartlett `0.05` variance-floor activation (`s_raw = 0.01 -> 0.05`) on an alternating sequence,
7. Asymmetric `GJR-GARCH(1,1)-t` leverage sign (`o = 1` activates on `r_{t-1} < 0`, i.e., positive USD foreign-currency losses) and machine-precision (`< 1e-12`) reconciliation with pure-NumPy variance recursion,
8. Strict zero-look-ahead forecasting invariance when future losses `s >= t` are perturbed by `+500`, and
9. Disjoint `PCG64` `SeedSequence` stream separation across pilot, pivotal calibration, cutoff bootstrap, per-series block bootstrap, and joint cross-FX block bootstrap.

Pilot timing and resource admission ([pilot-admission.json](pilot-admission.json)) projected `429.62s` conservative wall time (`3x` scaled pilot + `60s` reserve) and `235.0 MiB` RSS, well inside the `1,800s` CPU budget and `4 GiB` RSS ceiling.

## Live empirical execution and numerical accounting

- **Input integrity**: Raw FRED H.10 files ([DEXUSEU.csv](data/DEXUSEU.csv), [DEXUSUK.csv](data/DEXUSUK.csv), [DEXJPUS.csv](data/DEXJPUS.csv)) and source documents ([2611.06005v1.pdf](sources/2611.06005v1.pdf), [trick-of-the-tail-proper-plan-v2.md](sources/trick-of-the-tail-proper-plan-v2.md)) matched SHA-256 checksums in [source-manifest.json](data/source-manifest.json). Removing 273 `.` holiday rows without interpolation yielded `6,770` valid quotes and `6,769` signed percentage log losses per currency (`2,516` in `<= 2008-12-31` and `4,253` common scored dates in `2009-01-02..2025-12-31`).
- **Live runtime and memory**: Consolidated live run `runs/20261107T155047630280+0100` completed in **`1.82s`** reusing the seven SHA-256-verified `5,000`-draw Pareto calibration caches (`35,000` paths total; cold generation originally took `117.34s` / `119.13s` end-to-end, `0.01s` cache verification, `0.53s` retrospective inference and two-attempt SLSQP volatility filtering, and `1.15s` past-only forecasting across `2` windows × `3` currencies × `4,253` dates × `4` arms × `2` tail probabilities), with peak RSS **`238.3 MiB`** (`249,905,152` bytes), **`0` failed volatility filters**, and **`0` failed forecast origins**.
- **Volatility filters & HAC diagnostics** ([filter-hac-diagnostics.csv](filter-hac-diagnostics.csv)): All 6 full-sample filters (`GARCH(1,1)-t` and `GJR-GARCH(1,1)-t` across EUR, GBP, JPY) converged under both SLSQP initialization attempts, satisfied stationarity and `nu > 2.05`, and reconciled with pure-NumPy variance recursions within `2.66e-15`. Zero `0.05` Bartlett HAC floor activations occurred across any empirical or calibration path.
- **Headline inference outcome** ([baseline-stability-tests.csv](baseline-stability-tests.csv), [primary-comparison.json](primary-comparison.json)): **0/9** baseline slots rejected at nominal familywise `alpha = 0.05` after nine-slot Holm adjustment (`GBP Raw` raw `p = 0.0328`, Holm `p = 0.2623`; `GBP GJR |z|` raw `p = 0.0200`, Holm `p = 0.1800`; all EUR and JPY raw `p >= 0.2430`, Holm `p = 1.0000`). Across all six one-factor EUR perturbations ([eur-robustness-tests.csv](eur-robustness-tests.csv)), descriptive pivotal `p`-values stayed between `0.1390` and `0.4947`.
- **Headline prediction outcome** ([forecast-scores-summary.csv](forecast-scores-summary.csv), [forecast-paired-contrasts.csv](forecast-paired-contrasts.csv), [primary-comparison.json](primary-comparison.json)): On the predeclared primary contrast (`EUR/USD`, `p = 0.01`, expanding window, `4,253` dates), mean daily `loss(C) - loss(B)` is **`-0.000224`** (`-2.240e-4`), which is **inconclusive** (`L = 20` 95% moving-block CI `[-0.000678, +0.000204]`; `L = 10`: `[-0.000685, +0.000198]`; `L = 60`: `[-0.000763, +0.000262]`). By contrast, updating the local threshold while holding 2008 shape fixed (`B - A`) reduces mean daily pinball loss by **`-0.006673`** (`L = 20` CI `[-0.008087, -0.005061]`). Supporting windows (`Rolling 2,500`) and currencies (`GBP`, `JPY`, equal-weight 3-FX joint) are retained in full without cherry-picking.

## Fresh-kernel offline replay acceptance and visual QA

- **Fresh-kernel lockdown** ([fresh-kernel-validation.json](fresh-kernel-validation.json)): [validate_notebook.py](scripts/validate_notebook.py) copied only the standalone notebook, `data/`, `sources/`, `calibration/`, `runs/20261107T155047630280+0100/`, and JSON pointers into an isolated temporary directory (`/var/folders/.../trick-of-the-tail-portable-5f1kd9ui`), blocked `socket.socket.connect`, `ConstantMean.fit`, `run_study`, `calibrate_pivotal`, and `fit_volatility_filter`, and executed all `26` code cells in a fresh `Python 3.12.8` kernel in **`26.08s`**. Replay verified pure-NumPy volatility recursions (`rtol = atol = 1e-10`) and interior local tail/HAC estimators (`rtol = 1e-9, atol = 1e-11`) from raw inputs and recomputed all Holm p-values, pinball losses, coverage tests, block bootstraps, CSV sidecars, and `13` inline/disk PNGs without error.
- **Visual inspection** ([visual-qa.json](visual-qa.json), [asset-manifest.json](asset-manifest.json)): All 13 single-panel PNGs (`01`..`13`, dated `0711`) were visually inspected directly and verified to match inline notebook outputs byte-for-byte via SHA-256 digests.
- **Independent critiques & delivery audit**: Pre-freeze statistical review ([plan-critique.md](plan-critique.md)), post-outline review ([post-plan-review.md](post-plan-review.md)), and automated cross-artifact verification ([delivery-audit.json](delivery-audit.json)) confirm structural, numerical, and link integrity across the consolidated `trick-of-the-tail/` package (with the earlier exploratory package archived under [archive/v1-time-varying-tails/](archive/v1-time-varying-tails/README.md)).

