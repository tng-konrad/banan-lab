"""Build trick_of_the_tail.ipynb with all project helpers inline; optionally run pilot or live study."""
from pathlib import Path
import argparse
import json
import os

import nbformat as nbf

PROJECT = Path(__file__).resolve().parents[1]
UTILS = (PROJECT / "src/utils.py").read_text()

IMPORTS = '''from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import tempfile
import time

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "trick-of-the-tail-mpl"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from scipy.stats import chi2
from arch.univariate import ConstantMean, GARCH, StudentsT
from IPython.display import Image, Markdown, display
'''

CONFIG = '''class CFG:
    mode = os.environ.get("TAILS_MODE", "replay")
    data_folder = Path("data")
    sources_folder = Path("sources")
    calibration_folder = Path("calibration")
    graph_folder = Path("graphs")
    timezone = "Europe/Amsterdam"
    execution_started_at = datetime.now(ZoneInfo(timezone))
    execution_date = execution_started_at.strftime("%d%m")
    run_folder = Path(
        os.environ.get(
            "TAILS_RUN",
            json.loads(Path("latest-run.json").read_text())["path"]
            if Path("latest-run.json").exists()
            else "runs/missing",
        )
    )
    SEED = 20261107
    series_order = ["EUR", "GBP", "JPY"]
    date_bounds = ("1999-01-04", "2025-12-31")
    prefix_cutoff = "2008-12-31"
    scored_start = "2009-01-01"
    min_prefix_obs = 2000

    # Interior inference settings (Sections 4-5)
    m = 200
    h = 0.10
    J = 20
    hac_floor = 0.05
    grid_bounds = (0.15, 0.85)
    grid_points = 41
    grid = np.round(np.linspace(grid_bounds[0], grid_bounds[1], grid_points), 10)
    gamma_0 = 0.3
    calibration_reps = 5000
    calibration_chunk = 250
    cutoff_bootstrap_reps = 1000
    familywise_alpha = 0.05
    eur_robustness = [
        ("Baseline", 200, 0.10, 20),
        ("m=150", 150, 0.10, 20),
        ("m=250", 250, 0.10, 20),
        ("h=0.075", 200, 0.075, 20),
        ("h=0.125", 200, 0.125, 20),
        ("J=10", 200, 0.10, 10),
        ("J=30", 200, 0.10, 30),
    ]

    # Volatility filter settings (retrospective interior inference only)
    stationarity_margin = 1e-6
    slsqp_ftol = 1e-9
    slsqp_maxiter = 1000
    init_diag_obs = 250

    # Past-only chronological forecasting settings (Sections 7-8)
    forecast_m = 200
    forecast_h = 0.10
    rolling_window = 2500
    tail_probs = [0.01, 0.025]
    arms = ["A_static", "B_dyn_thresh", "C_dyn_both", "D_rect_hill"]
    arm_labels = {
        "A_static": "A: Static (2008)",
        "B_dyn_thresh": "B: Dynamic b, fixed gamma",
        "C_dyn_both": "C: Dynamic b + gamma",
        "D_rect_hill": "D: Rectangular Hill",
    }
    block_lengths = [20, 10, 60]
    forecast_bootstrap_reps = 5000
    violation_roll_window = 250

    # Resource guards and publication dimensions
    compute_budget_seconds = 1800
    rss_limit_bytes = 4 * 2**30
    img_dim1, img_dim2 = 10.8, 6.2
    table_dim1, table_dim2 = 11.8, 4.5
    export_dpi = 180
    colors = {
        "raw": "#17678e",
        "garch": "#2a7f62",
        "gjr": "#6a4c93",
        "projection": "#b85d19",
        "A": "#7a869a",
        "B": "#1f77b4",
        "C": "#c0392b",
        "D": "#d68910",
    }
    packages = [
        "numpy",
        "pandas",
        "scipy",
        "arch",
        "matplotlib",
        "psutil",
        "ipython",
        "nbformat",
        "nbclient",
        "ipykernel",
        "jupyter-client",
        "Pillow",
    ]


if CFG.mode not in ("live", "replay"):
    raise ValueError("CFG.mode must be 'live' or 'replay'")
plt.style.use("seaborn-v0_8")
plt.rcParams["figure.figsize"] = (CFG.img_dim1, CFG.img_dim2)
'''


def namespace():
    ns = {}
    exec(IMPORTS + "\n" + CONFIG + "\n" + UTILS, ns)
    return ns


def build():
    cells = []

    def md(text):
        cells.append(nbf.v4.new_markdown_cell(text))

    def code(text, explanation):
        cells.append(nbf.v4.new_code_cell(text))
        md(explanation)

    md("# Setup")

    code(
        IMPORTS.strip(),
        "All imports live in this opening cell so dependencies are explicit up front:\n\n"
        "- **Headless Matplotlib setup**: `MPLCONFIGDIR` points to a temporary folder (`trick-of-the-tail-mpl`) and `matplotlib.use(\"Agg\")` forces off-screen PNG rendering before `pyplot` loads, avoiding font-cache permission issues or GUI popups in batch runs.\n"
        "- **Numerical and econometric stack**: `numpy` and `pandas` handle array broadcasting and FRED CSV tables; `scipy.stats.chi2` provides survival probabilities (`chi2.sf`) for Kupiec and Christoffersen likelihood-ratio coverage tests; `arch.univariate` supplies `ConstantMean`, `GARCH` (symmetric `o=0` and asymmetric GJR `o=1`), and `StudentsT` innovations for full-sample volatility filtering.\n"
        "- **Runtime checks and inline display**: `hashlib`, `json`, `importlib.metadata`, `platform`, and `psutil` verify SHA-256 file hashes, log package versions, and enforce the memory ceiling; `IPython.display` renders the exported PNG bytes inline.",
    )

    code(
        CONFIG.strip(),
        "`CFG` keeps every path, seed, bandwidth, threshold count, optimizer tolerance, and backtest window in one place:\n\n"
        "- **Execution mode and run tracking**: `TAILS_MODE` selects `replay` (default; verifies saved `.npz` evidence and recomputes estimators and diagnostics without refitting `arch` or regenerating Monte Carlo draws) or `live` (executes a fresh study under `runs/`). `execution_date` (`ddmm` in `Europe/Amsterdam`) stamps exported asset filenames.\n"
        "- **Currency sample split (`1999-01-04` to `2025-12-31`)**: `EUR`, `GBP`, and `JPY` each provide `6,769` daily log losses. The prefix through `2008-12-31` (`2,516` observations; minimum `2,000`) anchors static baseline parameters; `2009-01-01` through `2025-12-31` (`4,253` common dates) is the out-of-sample forecasting window.\n"
        "- **Interior tail inference (`Sections 4–6` of Li & Peng)**: Local upper-tail estimation uses `m = 200` extreme order statistics ($q = m/n \\approx 2.95\\%$), Epanechnikov half-window bandwidth `h = 0.10` (effective local tail scale $mh = 20$), Bartlett HAC lag truncation `J = 20` with variance floor `0.05`, and a 41-point interior grid $u \\in [0.15, 0.85]$. `eur_robustness` perturbs $m \\in \\{150, 250\\}$, $h \\in \\{0.075, 0.125\\}$, and $J \\in \\{10, 30\\}$ one factor at a time.\n"
        "- **Volatility filters and chronological forecasting (`Sections 7–8`)**: GARCH/GJR filters apply only to retrospective interior series (`stationarity_margin = 1e-6`, `slsqp_ftol = 1e-9`). Real-time forecasting evaluates one-sided endpoint estimates ($u = 1, m = 200, h = 0.10$) at tail probabilities $p \\in \\{0.01, 0.025\\}$ across four arms (`A_static`, `B_dyn_thresh`, `C_dyn_both`, `D_rect_hill`), comparing expanding and 2,500-day rolling windows with 5,000-draw paired moving-block bootstraps ($L \\in \\{20, 10, 60\\}$).",
    )

    md("# Utils")

    util_explanations = {
        "Evidence, checksums and resource guards": (
            "These helpers handle deterministic seeding, file checksums, runtime limits, and the frozen study manifest:\n\n"
            "- **Strict JSON and SHA-256 hashing**: `json_write` rejects `NaN`/`Inf` values (`allow_nan=False`) so bad floats fail immediately; `sha` hashes raw file bytes; and `sha_obj` hashes canonicalized JSON strings (`sort_keys=True`) so every calibration setting maps to a deterministic cache key.\n"
            "- **Resource circuit breaker (`guard`)**: Checks elapsed monotonic time and process RSS against `CFG.compute_budget_seconds` (`1,800s`) and `CFG.rss_limit_bytes` (`4 GiB`).\n"
            "- **Disjoint RNG streams (`rng_for`)**: Builds independent `PCG64` generators from `SeedSequence([20261107, phase, group, sub, rep])`, keeping pilot draws (`phase=10`), pivotal Pareto calibrations (`phase=20`), and bootstrap resamples (`phase=50`) completely separate.\n"
            "- **Study contract (`protocol`)**: Snapshots all source hashes, sign conventions, kernel/quantile rules, GARCH/GJR tie-breakers, 9-slot Holm rules, and forecast scoring formulas. Replay mode checks `frozen[\"configuration\"] == protocol()` before loading any run."
        ),
        "Currency data and exposure alignment": (
            "`load_fx_panel` verifies source hashes, aligns currency quote conventions, and partitions the prefix and scored windows:\n\n"
            "- **Checksum verification**: Confirms that the paper PDF, planning specification, and all three FRED H.10 CSVs (`DEXUSEU.csv`, `DEXUSUK.csv`, `DEXJPUS.csv`) match the SHA-256 digests pinned in `data/source-manifest.json`.\n"
            "- **Unified USD exposure sign**: Drops `273` non-trading `.` entries per series without interpolation (`6,770` valid quotes $\\rightarrow$ `6,769` daily log changes). `DEXUSEU` and `DEXUSUK` are published as USD per foreign unit, while `DEXJPUS` (yen per USD) is inverted first (`1.0 / quote`) so `loss_t = -100 * diff(log(USD_per_foreign_unit))` always measures positive percentage losses on a long foreign-currency position in USD (`returns = -losses`).\n"
            "- **Prefix/scored split and date intersection**: Splits each series into the `<= 2008-12-31` prefix (`2,516` losses) and `>= 2009-01-02` scored window (`4,253` losses), verifying that all three currencies share the exact same `4,253` scored dates before joint block bootstrapping."
        ),
        "Reference and vectorized local tail estimators": (
            "These functions implement the common-threshold kernel Hill estimator, local Bartlett HAC studentization, and vectorized batch evaluation (Equations 4.2, 4.10–4.11, 5.2–5.3, and 6.2–6.4):\n\n"
            "- **Weighted quantile without float drift (`weighted_quantile_1d`)**: Sorts observations once and takes the left-continuous inverse where cumulative positive weight reaches $1 - q$. When integer `(m_int, n_int)` is passed, the target weight is computed as `((n - m) * total_w) / n` in a single division, preventing `+1 ULP` threshold shifts at rational fractions such as $m/n = 200/3000$.\n"
            "- **Local common-threshold ratio Hill & Bartlett HAC (`fit_local_reference`)**:\n"
            "  - At each interior point $u \\in [0.15, 0.85]$, Epanechnikov weights $\\frac{3}{4h}(1 - \\text{dist}^2)_+$ on strict open support $|\\text{dist}| < 1$ (`dist = (n*u - t)/(n*h)`) select a single local $(1 - m/n)$ threshold $\\hat{b}(u)$.\n"
            "  - Averaging weighted log excesses $\\log(X_t / \\hat{b}(u))$ strictly above $\\hat{b}(u) > 0$ cancels constant local scale factors inside the window.\n"
            "  - Serial dependence among clustered exceedances enters via residualized exceedances $\\hat{Z}_t(u) = \\mathbf{1}\\{X_t > \\hat{b}(u)\\}(\\log(X_t/\\hat{b}(u)) - \\hat{\\gamma}(u))$ and the Bartlett lag window $\\hat{S}(u)$ out to lag $J = 20$ (floored at `0.05`).\n"
            "  - Precision-weighted trapezoidal integration produces the constant projection $\\tilde{\\gamma} = (\\int \\hat{\\gamma}(u)/\\hat{S}(u)\\,du)/(\\int 1/\\hat{S}(u)\\,du)$, integrated squared deviation $T_n$, and omnibus test statistic $\\Lambda_n = mh\\,T_n / (\\tilde{\\gamma}^2 \\cdot 3/5)$.\n"
            "- **Batch broadcasting and causal endpoint (`fit_local_fast_batch`, `fit_endpoint_one_sided`)**: `precompute_slices` caches the active window indices at each grid point so `fit_local_fast_batch` can evaluate `250` simulation paths simultaneously via NumPy broadcasting. `fit_endpoint_one_sided` evaluates $u = 1.0$ using only past observations $s \\le t$ for real-time forecasting."
        ),
        "Volatility filters and recursion reconciliation": (
            "These helpers fit full-sample `GARCH(1,1)-t` and `GJR-GARCH(1,1)-t` filters on signed returns and verify their conditional variance paths in pure NumPy:\n\n"
            "- **Exact `arch` variance recursion (`ewma_backcast`, `numpy_garch_recursion`)**: In `arch==8.0.0`, `ConstantMean` computes its `0.94`-decay EWMA backcast on sample-demeaned returns `r - mean(r)`, then iterates $\\sigma_t^2 = \\omega + (\\alpha + \\gamma_{\\text{lev}}\\mathbf{1}\\{\\varepsilon_{t-1}<0\\})\\varepsilon_{t-1}^2 + \\beta\\sigma_{t-1}^2$ at MLE residuals $\\varepsilon_t = r_t - \\hat{\\mu}$. Matching this exact convention lets replay mode reconstruct $\\hat{\\sigma}_t$ to machine precision (`< 1e-12`) without calling `arch`'s optimizer.\n"
            "- **Two-attempt SLSQP schedule (`fit_volatility_filter`)**: Fits `ConstantMean(rescale=False)` + `GARCH` (`o=0` or `o=1`) + `StudentsT` on signed returns $r_t = -\\text{loss}_t$ (so negative returns correspond to USD holding losses and activate the GJR indicator $\\varepsilon_{t-1} < 0$). Attempt 1 uses `arch`'s default starting values; Attempt 2 starts from `[mean(r), 0.03*var(r), ...]`. Attempt 2 is selected only if valid and its log-likelihood beats Attempt 1 by more than `1e-10`.\n"
            "- **Admissibility and ACF diagnostics (`sample_acf`)**: A fit is accepted only if `convergence_flag == 0`, parameters satisfy stationarity and variance positivity (`persistence < 1 - 1e-6`, `alpha + gamma_lev >= -1e-9`, `nu > 2.05`), and `numpy_garch_recursion` agrees with `arch` within `1e-10`. `sample_acf` and `init_vol_ratio` record lag-1 magnitude autocorrelation (`|r|` vs. `|z|`) and initial 250-day volatility burn-in."
        ),
        "Pivotal Pareto calibration and Holm multiplicity": (
            "This module manages the content-addressed `B = 5,000` pivotal Pareto caches, critical-value bootstraps, and nine-slot Holm step-down adjustments:\n\n"
            "- **Hashed calibration caches (`calibration_spec`, `load_calibration_cache`)**: Hashes `(n, m, h, J, grid, floor, gamma_0=0.3, reps=5000)` into a SHA-256 key (`short_key` = first 16 hex chars) and verifies both the JSON metadata and `.npz` digest before reusing saved draws.\n"
            "- **Pivotal simulation (`calibrate_pivotal`, Remarks 5.3 & 6.1)**: Simulates `5,000` IID unit-scale Pareto paths $X_t = (1 - U_t)^{-0.3}$ in chunks of `250`. Because the uniform band theorem (Eq. 5.7) centers around the true parameter while the stability test (Eqs. 6.2–6.4) centers around the estimated projection, each draw records two distinct quantities:\n"
            "  - `band_max`: $\\sup_{u \\in \\mathcal{U}} \\sqrt{mh}\\,|\\hat{\\gamma}^{(b)}(u) - \\gamma_0| / (\\gamma_0 \\sqrt{\\tfrac{3}{5}\\hat{S}^{(b)}(u)})$, whose 95th percentile (`method=\"higher\"`) yields the simultaneous band multiplier $c_{0.95}$.\n"
            "  - `lambda_stat`: $\\Lambda_n^{(b)}$ centered at that path's own precision-weighted projection $\\tilde{\\gamma}^{(b)}$, yielding $\\Lambda_{0.95}$ and empirical $p$-values $(1 + \\sum_{b=1}^B \\mathbf{1}\\{\\Lambda_n^{(b)} \\ge \\Lambda_n^{\\text{obs}}\\}) / (B + 1)$.\n"
            "- **Cutoff uncertainty and Holm control (`summarize_calibration`, `holm_step_down`)**: Resamples the `5,000` pivotal draws `1,000` times to put 95% confidence bounds around $c_{0.95}$, $\\Lambda_{0.95}$, and $p_{\\text{raw}}$, then applies nine-slot Holm step-down control across `{EUR, GBP, JPY} x {Raw, GARCH |z|, GJR |z|}` at $\\alpha = 0.05$."
        ),
        "Past-only forecasting, coverage tests and block bootstrap": (
            "These functions run the strictly past-only Weissman quantile backtest, evaluate coverage likelihood ratios, and compute paired moving-block bootstrap intervals:\n\n"
            "- **Pinball loss and LR coverage tests (`pinball_loss`, `coverage_diagnostics`)**: Evaluates asymmetric quantile loss $\\rho_\\tau(y_t - \\hat{Q}_t) = (\\tau - \\mathbf{1}\\{y_t < \\hat{Q}_t\\})(y_t - \\hat{Q}_t)$ at $\\tau = 1 - p$ with strict exceedance indicator $I_t = \\mathbf{1}\\{y_t > \\hat{Q}_t\\}$. Kupiec unconditional coverage ($\\text{LR}_{\\text{uc}} \\sim \\chi^2_1$) and Christoffersen Markov independence ($\\text{LR}_{\\text{ind}} \\sim \\chi^2_1$) use `0 * log(0) = 0` and return `None` (`N/A`) whenever any marginal transition count is zero.\n"
            "- **Paired circular moving-block bootstrap (`circular_block_bootstrap`)**: Resamples contiguous blocks of length $L \\in \\{20, 10, 60\\}$ (`5,000` draws with circular wrap-around) from daily loss differences $d_t = \\text{loss}_{C,t} - \\text{loss}_{B,t}$ to preserve serial dependence across volatile clusters.\n"
            "- **Four extreme-quantile arms (`run_past_only_forecasts`, Eq. 7.1)**: At each origin $t = 2,516, \\dots, 6,768$, forecasts tomorrow's $(1-p)$ loss quantile via $\\hat{Q}_t(1-p) = \\hat{b}_t (q_t / p)^{\\hat{\\gamma}_t}$ (`m = 200, h = 0.10, q_t = m / n_t`) using only past losses $X_{1:t}$:\n"
            "  - **Arm A (`A_static`)**: Freezes both $\\hat{b}_0$ and $\\hat{\\gamma}_0$ at the `2008-12-31` prefix endpoint.\n"
            "  - **Arm B (`B_dyn_thresh`)**: Updates the local kernel threshold $\\hat{b}_t(1)$ daily while holding $\\hat{\\gamma}_0$ fixed at its `2008` value.\n"
            "  - **Arm C (`C_dyn_both`)**: Updates both $\\hat{b}_t(1)$ and the one-sided Epanechnikov tail index $\\hat{\\gamma}_t(1)$ daily.\n"
            "  - **Arm D (`D_rect_hill`)**: Updates both threshold and tail index daily on the same trailing `10%` support using uniform (rectangular) weights."
        ),
        "Fixtures, pilot, live execution and replay verification": (
            "`core_checks`, `pilot`, `run_study`, and `analyze_run` tie together pre-execution testing, live estimation, and deterministic offline replay:\n\n"
            "- **Unit and anti-leakage fixtures (`core_checks`)**: Verifies 9 properties before any empirical run—analytic Pareto threshold recovery, scalar vs. vectorized estimator parity (`1e-12`), negative/zero/tie invariance, positive scale invariance, Bartlett `0.05` floor activation on an alternating series, GJR leverage sign and NumPy-vs-`arch` recursion parity (`< 1e-12`), strict future-perturbation forecast invariance (`+500` shock to $s \\ge t$), and `0*log(0)` coverage edge cases.\n"
            "- **Prefix-only pilot (`pilot`)**: Times 20-draw calibrations, `<= 2008-12-31` volatility fits, and 100 prefix forecast origins to verify that `3x` projected wall time plus a `60s` reserve fits well inside the `1,800s` CPU and `4 GiB` RSS limits without touching `2009–2025` observations.\n"
            "- **Live pipeline (`run_study`) and offline replay (`analyze_run`)**: `run_study` executes all 7 pivotal calibrations (`35,000` Pareto paths), 6 full-sample GARCH/GJR filters, and `25,518` currency-window forecast origins, saving `.npz` arrays and `run.json` hashes. `analyze_run` verifies all SHA-256 digests, recomputes interior estimators (`rtol=1e-9`) and pure-NumPy volatility recursions (`rtol=1e-10`) against saved outputs, and builds every summary table and bootstrap interval."
        ),
        "Single-panel publication helpers": (
            "`publish` exports each chart or table as a standalone, title-free single-panel PNG (`graphs/graph_<NN>_<ddmm>.png` or `graphs/tabler_<NN>_<ddmm>.png`, archiving any prior file under `render-archive/`), writes the full-precision DataFrame sidecar to a matching `.csv`, records the SHA-256 hash and caption metadata in `asset-manifest.json`, and displays the saved PNG and caption inline."
        ),
    }

    for chunk in UTILS.split("# %% "):
        if not chunk.strip():
            continue
        title, body = chunk.split("\n", 1)
        md(f"## {title}")
        code(body.strip(), util_explanations[title])

    # Section 1
    md("# A tail curve is not a finding")

    code(
        '''checks = core_checks()
panel, source_summary, common_scored_dates = load_fx_panel()
manifest = []
print("Numerical & leakage fixtures:", checks["status"], f"({len(checks['checks'])} checks)")
print(source_summary[["series", "fred_id", "invert_quote", "losses", "prefix_losses", "scored_losses", "zero_losses"]].to_string(index=False))
print("Common scored dates across EUR, GBP, JPY:", len(common_scored_dates), f"({common_scored_dates[0]} to {common_scored_dates[-1]})")''',
        "All 9 mathematical and zero-look-ahead fixtures pass, and the three FRED H.10 series align on `6,769` daily log losses each (`1999-01-05` to `2025-12-31`): `2,516` in the `1999–2008` prefix and `4,253` shared scored dates across `2009-01-02` to `2025-12-31`, with `48` to `61` exact-zero days retained in full.",
    )

    # Section 2
    md("# One threshold and dependent extremes")

    code(
        '''scope_table = pd.DataFrame(
    [
        ["Local Hill ratio", "Eqs 4.2, 4.10-4.11; Epanechnikov; strict >", "Exact mechanism; fast vs scalar verified"],
        ["Bartlett HAC", "Eqs 5.2-5.3; J=20; 0.05 floor logged", "Variance factor; not finite-threshold bias fix"],
        ["Uniform bands", "Eq 5.7; 41-pt [0.15, 0.85]; B=5,000 at gamma0=0.3", "Simultaneous on grid; no empirical truth"],
        ["Stability test", "Eqs 6.2-6.4; projection center; 9-slot Holm", "Nominal pivotal p; synthetic DGP screen skipped"],
        ["Volatility filters", "GARCH(1,1)-t & GJR(1,1)-t; 2-attempt SLSQP", "Retrospective interior only; distinct estimand"],
        ["Risk forecasts", "Eq 7.1 past-only u=1; Arms A/B/C/D; p=0.01, 0.025", "Chronological raw losses; no ES or band reuse"],
        ["Currency panel", "FRED EUR, GBP, inverted JPY (1999-2025)", "Correlated USD FX; temporal pseudo-holdout"],
    ],
    columns=["Component", "Implemented here", "Claim boundary"],
)
manifest = publish(
    "table",
    1,
    "Scope and claim boundaries in the empirical study",
    "One threshold and dependent extremes",
    "Paper-to-notebook correspondence and explicit claim boundaries under the real-data-only scope.",
    "Seven-row table mapping each paper component to its local implementation and supported claim boundary.",
    "Expected shortfall, joint risk confidence intervals and synthetic null-size/power Monte Carlo studies are intentionally omitted; empirical stability p-values are therefore reported as nominal paper-calibrated pivotal quantities.",
    manifest,
    table=scope_table,
    sidecar=scope_table,
    table_height=4.4,
    col_widths=[0.19, 0.41, 0.40],
    font_size=9.1,
)''',
        "Table 01 maps each section of Li & Peng (`arXiv:2611.06005v1`) to our implementation and states the claim boundaries upfront:\n\n"
        "- **Nominal pivotal $p$-values**: Because Section 6's synthetic Monte Carlo size and power experiments were omitted to focus on the empirical currency panel, stability test decisions rely on the paper's `B = 5,000` IID unit-scale Pareto calibration ($\\gamma_0 = 0.3$) and are reported as nominal pivotal quantities rather than simulation-certified 5% familywise error rates.\n"
        "- **Retrospective vs. causal separation**: Full-sample `GARCH(1,1)-t` and `GJR-GARCH(1,1)-t` filters are used strictly for interior tail-stability inference on `[0.15, 0.85]` and never enter the chronological quantile forecasts (`2009–2025`).",
    )

    # Section 3
    md("# Can the test control false alarms?")

    code(
        '''frozen = json.loads(Path("frozen-study.json").read_text())
if frozen["configuration"] != protocol():
    raise ValueError("Frozen protocol does not match current CFG")
if CFG.mode == "live":
    if json.loads(Path("pilot-admission.json").read_text())["status"] != "admitted":
        raise ValueError("Pilot admission gate has not passed")
    CFG.run_folder = run_study()
result = analyze_run(CFG.run_folder, verify_estimators=True)
base_cal = result["cal_df"].iloc[0]
print(
    f"Source run: {result['path']} | status={result['record']['status']} | "
    f"baseline c_0.95={base_cal['c_95']:.4f} (boot 95% [{base_cal['c_95_boot_low']:.4f}, {base_cal['c_95_boot_high']:.4f}]) | "
    f"Lambda_0.95={base_cal['lambda_95']:.4f} (boot 95% [{base_cal['lambda_95_boot_low']:.4f}, {base_cal['lambda_95_boot_high']:.4f}])"
)''',
        "After verifying that `CFG` matches `frozen-study.json`, `analyze_run(..., verify_estimators=True)` checks every `.npz` digest, reconciles saved GARCH/GJR volatilities and interior Hill/HAC paths against fresh NumPy calculations, and prints the baseline (`n = 6,769, m = 200, h = 0.10, J = 20`) 5,000-draw Pareto cutoffs: simultaneous band multiplier $c_{0.95} = 4.0345$ (`95%` bootstrap CI `[3.9355, 4.1499]`) and omnibus test critical value $\\Lambda_{0.95} = 1.2395$ (`[1.1984, 1.2792]`).",
    )

    # Section 4
    md("# Raw losses and standardized innovations")

    code(
        '''diag = result["diag_df"].copy()
diag_view = pd.DataFrame(
    {
        "Series": diag["series"],
        "Target": diag["target"],
        "Persist.": diag["persistence"].map(lambda v: "—" if pd.isna(v) else f"{v:.4f}"),
        "DoF nu": diag["nu"].map(lambda v: "—" if pd.isna(v) else f"{v:.2f}"),
        "|x| ACF(1)": diag["abs_acf1"].map(lambda v: f"{v:.3f}"),
        "Mean S": diag["mean_s"].map(lambda v: f"{v:.3f}"),
        "Min S_raw": diag["min_s_raw"].map(lambda v: f"{v:.3f}"),
        "Floors": diag["floor_activations"].map(lambda v: f"{int(v)}/41"),
    }
)
manifest = publish(
    "table",
    2,
    "Volatility filter convergence and local HAC diagnostics",
    "Raw losses and standardized innovations",
    "Full-sample GARCH(1,1)-t and GJR-GARCH(1,1)-t parameters, lag-1 absolute autocorrelation and local Bartlett HAC summaries across EUR, GBP and JPY.",
    "Nine-row diagnostic table reporting persistence, Student-t degrees of freedom, lag-1 absolute ACF, mean and minimum unfloored Bartlett HAC factors, and floor counts.",
    "All six volatility filters converged under both SLSQP initialization attempts with persistence between 0.9910 and 0.9991 and Student-t degrees of freedom between 5.35 and 8.83, sharply reducing lag-1 absolute autocorrelation while keeping finite-sample HAC factors visible.",
    manifest,
    table=diag_view,
    sidecar=diag,
    table_height=5.0,
    col_widths=[0.08, 0.25, 0.11, 0.10, 0.12, 0.11, 0.12, 0.11],
    font_size=9.0,
)''',
        "Table 02 summarizes the six full-sample volatility filters and their effect on exceedance dependence across `EUR`, `GBP`, and `JPY`:\n\n"
        "- **High persistence and heavy Student-$t$ tails**: All six models converged on both SLSQP initialization attempts with persistence between `0.9910` and `0.9991` and Student-$t$ degrees of freedom $\\hat{\\nu}$ between `5.35` (`JPY`) and `8.83` (`GBP`), confirming heavy innovation tails even after volatility standardization.\n"
        "- **Compression of serial dependence**: Dividing returns by $\\hat{\\sigma}_t$ cuts lag-1 magnitude autocorrelation (`|x| ACF(1)`) from `0.077–0.146` on raw losses down to `-0.033..+0.007` on $|z_t|$, and lowers mean local Bartlett HAC factors (`Mean S` drops from `1.208` to `0.909–0.916` on `EUR` and from `1.389` to `1.132–1.186` on `GBP`), with `0/41` variance-floor activations anywhere.",
    )

    # Section 5
    md("# Does tail shape stay constant?")

    code(
        '''eur_raw_df = result["eur_paths"]["raw"]
eur_raw_meta = result["baseline_df"].query("series == 'EUR' and target_key == 'raw'").iloc[0]

def plot_eur_raw(ax):
    ax.fill_between(eur_raw_df["approx_date"], eur_raw_df["band_low"], eur_raw_df["band_high"], color=CFG.colors["raw"], alpha=0.20, label="95% HAC simultaneous band")
    ax.plot(eur_raw_df["approx_date"], eur_raw_df["gamma_hat"], color=CFG.colors["raw"], linewidth=2.2, label="Local ratio Hill gamma_hat(u)")
    ax.plot(eur_raw_df["approx_date"], eur_raw_df["gamma_tilde"], color=CFG.colors["projection"], linestyle="--", linewidth=1.8, label=f"Precision projection gamma_tilde = {eur_raw_meta['gamma_tilde']:.3f}")
    if eur_raw_df["band_low"].min() < 0.0:
        ax.axhline(0.0, color="#555555", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Approximate quote date on interior grid u in [0.15, 0.85]")
    ax.set_ylabel("Extreme value index gamma")
    ax.legend(loc="upper right", frameon=True)

manifest = publish(
    "graph",
    3,
    "EUR/USD raw positive-loss tail index and 95% HAC band",
    "Does tail shape stay constant?",
    f"EUR/USD raw upper-loss tail path (n=6,769, m=200, h=0.10, J=20): Lambda_n={eur_raw_meta['lambda_stat']:.3f}, raw pivotal p={eur_raw_meta['p_raw']:.4f}, Holm p={eur_raw_meta['p_holm']:.4f}.",
    "Line plot of EUR/USD raw-loss local tail index from 2003 to 2021 with shaded 95 percent HAC uniform confidence band and dashed constant projection line.",
    "These two-sided retrospective estimates use observations before and after each date. The wide HAC band reflects local estimation variance and serial dependence; unclipped band endpoints are shown honestly.",
    manifest,
    plot=plot_eur_raw,
    sidecar=eur_raw_df,
)''',
        "On raw `EUR/USD` losses across the interior grid $u \\in [0.15, 0.85]$ (`2003-01-17` to `2021-12-08`), the local ratio Hill curve $\\hat{\\gamma}(u)$ ranges from `0.179` to `0.349` around the precision-weighted constant projection $\\tilde{\\gamma} = 0.239$. Once serial dependence (`Mean S = 1.208`) and 41-point simultaneous coverage ($c_{0.95} = 4.035$) are accounted for, the 95% HAC band encloses $\\tilde{\\gamma}$ across the entire sample ($\\Lambda_n = 0.737$, $p_{\\text{raw}} = 0.2430$, $p_{\\text{Holm}} = 1.0000$).",
    )

    code(
        '''eur_garch_df = result["eur_paths"]["garch"]
eur_garch_meta = result["baseline_df"].query("series == 'EUR' and target_key == 'garch'").iloc[0]

def plot_eur_garch(ax):
    ax.fill_between(eur_garch_df["approx_date"], eur_garch_df["band_low"], eur_garch_df["band_high"], color=CFG.colors["garch"], alpha=0.20, label="95% HAC simultaneous band")
    ax.plot(eur_garch_df["approx_date"], eur_garch_df["gamma_hat"], color=CFG.colors["garch"], linewidth=2.2, label="GARCH |z| local gamma_hat(u)")
    ax.plot(eur_garch_df["approx_date"], eur_garch_df["gamma_tilde"], color=CFG.colors["projection"], linestyle="--", linewidth=1.8, label=f"Precision projection gamma_tilde = {eur_garch_meta['gamma_tilde']:.3f}")
    ax.set_xlabel("Approximate quote date on interior grid u in [0.15, 0.85]")
    ax.set_ylabel("Innovation tail index gamma")
    ax.legend(loc="upper right", frameon=True)

manifest = publish(
    "graph",
    4,
    "EUR/USD GARCH(1,1)-t innovation tail index and 95% HAC band",
    "Does tail shape stay constant?",
    f"EUR/USD absolute standardized GARCH(1,1)-t innovations: Lambda_n={eur_garch_meta['lambda_stat']:.3f}, raw pivotal p={eur_garch_meta['p_raw']:.4f}, Holm p={eur_garch_meta['p_holm']:.4f}.",
    "Line plot of local tail-index estimates for EUR/USD GARCH absolute standardized residuals with shaded 95 percent HAC uniform band and dashed projection center.",
    "Filtering conditional variance shifts the target from raw one-sided currency losses to two-sided standardized shock magnitudes; a lower or smoother path estimates a distinct innovation tail rather than 'correcting' raw gamma.",
    manifest,
    plot=plot_eur_garch,
    sidecar=eur_garch_df,
)''',
        "Switching from one-sided raw currency losses $X_t$ to two-sided `GARCH(1,1)-t` standardized innovation magnitudes $|z_t|$ changes the estimand to the shock distribution's tail ($\\tilde{\\gamma} = 0.205$, consistent with $\\hat{\\nu} = 7.21$ where $1/\\hat{\\nu} \\approx 0.139$ plus finite-threshold second-order bias at $m/n \\approx 3\\%$). The trajectory narrows to `0.147–0.289` and shows no evidence against constancy ($\\Lambda_n = 0.725$, $p_{\\text{raw}} = 0.2521$, $p_{\\text{Holm}} = 1.0000$).",
    )

    code(
        '''eur_gjr_df = result["eur_paths"]["gjr"]
eur_gjr_meta = result["baseline_df"].query("series == 'EUR' and target_key == 'gjr'").iloc[0]

def plot_eur_gjr(ax):
    ax.fill_between(eur_gjr_df["approx_date"], eur_gjr_df["band_low"], eur_gjr_df["band_high"], color=CFG.colors["gjr"], alpha=0.20, label="95% HAC simultaneous band")
    ax.plot(eur_gjr_df["approx_date"], eur_gjr_df["gamma_hat"], color=CFG.colors["gjr"], linewidth=2.2, label="GJR-GARCH |z| local gamma_hat(u)")
    ax.plot(eur_gjr_df["approx_date"], eur_gjr_df["gamma_tilde"], color=CFG.colors["projection"], linestyle="--", linewidth=1.8, label=f"Precision projection gamma_tilde = {eur_gjr_meta['gamma_tilde']:.3f}")
    ax.set_xlabel("Approximate quote date on interior grid u in [0.15, 0.85]")
    ax.set_ylabel("Innovation tail index gamma")
    ax.legend(loc="upper right", frameon=True)

manifest = publish(
    "graph",
    5,
    "EUR/USD GJR-GARCH(1,1)-t innovation tail index and 95% HAC band",
    "Does tail shape stay constant?",
    f"EUR/USD absolute standardized GJR-GARCH(1,1)-t innovations: Lambda_n={eur_gjr_meta['lambda_stat']:.3f}, raw pivotal p={eur_gjr_meta['p_raw']:.4f}, Holm p={eur_gjr_meta['p_holm']:.4f}.",
    "Line plot of local tail-index estimates for EUR/USD GJR-GARCH absolute standardized residuals with shaded 95 percent HAC uniform band and dashed projection center.",
    "Allowing asymmetric variance responses to USD euro-holding losses leaves the standardized innovation tail path closely aligned with symmetric GARCH(1,1)-t on EUR/USD.",
    manifest,
    plot=plot_eur_gjr,
    sidecar=eur_gjr_df,
)''',
        "Adding asymmetric leverage to the variance equation (`GJR-GARCH(1,1)-t`, $\\hat{\\gamma}_{\\text{lev}} = +0.011$ on `EUR/USD`) leaves the standardized innovation tail path almost indistinguishable from symmetric GARCH ($\\tilde{\\gamma} = 0.201$, range `0.157–0.278`, $\\Lambda_n = 0.634$, $p_{\\text{raw}} = 0.3305$, $p_{\\text{Holm}} = 1.0000$).",
    )

    code(
        '''base_df = result["baseline_df"].copy()
base_view = pd.DataFrame(
    {
        "Series": base_df["series"],
        "Target": base_df["target"],
        "gamma_tilde": base_df["gamma_tilde"].map(lambda v: f"{v:.3f}"),
        "Mean S": base_df["mean_s"].map(lambda v: f"{v:.3f}"),
        "Lambda_n": base_df["lambda_stat"].map(lambda v: f"{v:.3f}"),
        "Raw p": base_df["p_raw"].map(lambda v: f"{v:.4f}"),
        "Holm p": base_df["p_holm"].map(lambda v: "N/A" if pd.isna(v) else f"{v:.4f}"),
        "Nominal status": base_df["decision"],
    }
)
manifest = publish(
    "table",
    6,
    "Baseline nine-test tail-stability family (nominal pivotal calibration)",
    "Does tail shape stay constant?",
    "Precision-weighted constant projection gamma_tilde, mean Bartlett HAC factor, omnibus statistic Lambda_n, raw pivotal p-value and nine-slot Holm-adjusted p-value (m=200, h=0.10, J=20).",
    "Nine-row table of baseline tail-stability tests across EUR, GBP and JPY for raw losses, GARCH innovations and GJR-GARCH innovations.",
    "All nine prespecified slots completed without filter failure. Decisions are nominal paper-calibrated pivotal outcomes at familywise alpha=0.05 (unscreened by synthetic dependent-null simulations); correlated USD exchange rates are not independent replications.",
    manifest,
    table=base_view,
    sidecar=base_df,
    table_height=5.0,
    col_widths=[0.08, 0.22, 0.12, 0.09, 0.11, 0.09, 0.09, 0.20],
    font_size=8.9,
)''',
        "Table 06 evaluates the full nine-test baseline family (`{EUR, GBP, JPY} x {Raw, GARCH |z|, GJR |z|}`):\n\n"
        "- **Zero familywise rejections**: **0 of 9** tests reject constant tail shape after nine-slot Holm adjustment at $\\alpha = 0.05$.\n"
        "- **Unadjusted `GBP/USD` signals**: Two `GBP` rows cross the single-test 5% cutoff ($\\Lambda_{0.95} = 1.2395$)—`GBP Raw` ($\\Lambda_n = 1.369, p_{\\text{raw}} = 0.0328$) and `GBP GJR |z|` ($\\Lambda_n = 1.549, p_{\\text{raw}} = 0.0200$, whereas symmetric `GBP GARCH |z|` gives $\\Lambda_n = 0.959, p_{\\text{raw}} = 0.1230$)—but neither survives multiplicity control ($p_{\\text{Holm}} = 0.2623$ and $0.1800$). `JPY/USD` is quiet across all three targets ($p_{\\text{raw}} \\in [0.4605, 0.6423]$, $p_{\\text{Holm}} = 1.0000$).",
    )

    # Section 6
    md("# Predict tomorrow using yesterday")

    code(
        '''v_roll = result["eur_violations_roll"]

def plot_eur_violations(ax):
    ax.plot(v_roll["date"], v_roll["A_roll250"], color=CFG.colors["A"], linewidth=1.6, alpha=0.9, label="A: Static (2008)")
    ax.plot(v_roll["date"], v_roll["B_roll250"], color=CFG.colors["B"], linewidth=1.8, label="B: Dynamic b, fixed gamma")
    ax.plot(v_roll["date"], v_roll["C_roll250"], color=CFG.colors["C"], linewidth=1.8, linestyle="-.", label="C: Dynamic b + gamma")
    ax.plot(v_roll["date"], v_roll["D_roll250"], color=CFG.colors["D"], linewidth=1.5, linestyle=":", label="D: Rectangular Hill")
    ax.axhline(2.5, color="black", linestyle="--", linewidth=1.2, label="Nominal 2.5 violations / 250 days (1%)")
    ax.set_xlabel("Scored forecast date (trailing 250-observation window)")
    ax.set_ylabel("Strict exceedances y_t > Q_hat in trailing 250 days")
    ax.legend(loc="upper right", frameon=True, fontsize=9.5)

manifest = publish(
    "graph",
    7,
    "EUR/USD rolling 250-day 1% quantile violations (2009-2025)",
    "Predict tomorrow using yesterday",
    "Trailing 250-observation strict violation counts on EUR/USD (p=0.01, expanding window) across Arms A, B, C and D against the nominal 2.5 benchmark.",
    "Time-series line chart of rolling 250-day 1 percent VaR exceedance counts from 2009 to 2025 for static Arm A and dynamic Arms B, C and D.",
    "Leaving the 2008 crisis-era threshold frozen (Arm A) starves violations during quiet regimes, whereas updating the local threshold (Arms B, C and D) tracks regime shifts closely; violation counts alone do not rank forecast accuracy.",
    manifest,
    plot=plot_eur_violations,
    sidecar=v_roll,
)''',
        "Trailing 250-day 1% violation counts on `EUR/USD` show why scale adaptation and shape adaptation must be separated:\n\n"
        "- **Static threshold starvation (`Arm A`)**: Because the `2008-12-31` prefix ended inside the global financial crisis, keeping $\\hat{b}_0$ frozen leaves the 1% quantile far too high during calm regimes (`0` violations in trailing 250-day windows across `2012–2014` and `2017–2021`, `24/4,253` overall vs. `42.5` expected).\n"
        "- **Dynamic threshold recovery (`Arms B, C, D`)**: Updating the local intermediate threshold $\\hat{b}_t(1)$ daily restores violation counts around the nominal `2.5` per `250` days (`48–49` total violations across `4,253` dates), with minimal separation between fixed-shape `Arm B` and dynamic-shape `Arms C` and `D`.",
    )

    # Section 7
    md("# What does updating shape add?")

    code(
        '''eur_daily = result["eur_daily_primary"]
prim = result["primary"]["prediction_primary"]

def plot_primary_cumdiff(ax):
    ax.plot(
        eur_daily["date"],
        eur_daily["cum_diff_C_minus_B"],
        color=CFG.colors["C"],
        linewidth=2.0,
        label=(
            f"Cumulative loss(C) - loss(B) | mean = {prim['mean_diff']:+.6f}/day "
            f"(L=20 95% CI [{prim['ci20_low']:+.6f}, {prim['ci20_high']:+.6f}])"
        ),
    )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.2, label="Zero cumulative difference (equal pinball loss)")
    ax.set_xlabel("Scored forecast date (2009-01-02 to 2025-12-31)")
    ax.set_ylabel("Cumulative pinball-loss difference (C minus B)")
    ax.legend(loc="best", frameon=True, fontsize=9.5)

manifest = publish(
    "graph",
    8,
    "Cumulative pinball-loss difference: Arm C minus Arm B (EUR, p=0.01)",
    "What does updating shape add?",
    f"Primary prediction contrast on EUR/USD (4,253 dates, p=0.01, expanding): mean daily loss(C)-loss(B) = {prim['mean_diff']:+.6f}, paired L=20 block-bootstrap 95% CI [{prim['ci20_low']:+.6f}, {prim['ci20_high']:+.6f}].",
    "Chronological curve of cumulative daily pinball-loss differences between dynamic-shape Arm C and fixed-shape Arm B from 2009 through 2025.",
    f"Primary verdict: {prim['verdict']}. Downward movements mark periods where dynamic tail shape beat fixed 2008 shape; abrupt steps correspond to rare 1%-tail exceedance days where slope 0.99 magnifies quantile differences.",
    manifest,
    plot=plot_primary_cumdiff,
    sidecar=eur_daily,
)''',
        "Plotting the cumulative daily pinball-loss difference $\\sum_{s \\le t} (\\text{loss}_{C,s} - \\text{loss}_{B,s})$ on the primary contrast (`EUR/USD`, `p = 0.01`, expanding window, `4,253` dates) exposes the regime trade-off behind the full-sample average:\n\n"
        "- **Calm-regime drift vs. tail-day steps**: Across quieter stretches (`2010–Aug 2014` and `2017–2019`), dynamic-shape `Arm C` slowly accumulates higher loss than fixed-shape `Arm B` (the curve drifts upward to `+1.06`) as estimation variance in $\\hat{\\gamma}_t(1)$ perturbs non-exceedance quantiles. That drift is offset by a few sharp downward steps (`Sep–Nov 2014`, early `2015`, `2020`, and `2022`) on rare 1% exceedance days where $y_t > \\hat{Q}_t$ and the pinball slope `0.99` magnifies small quantile differences.\n"
        "- **Primary verdict (`inconclusive`)**: Mean daily $\\text{loss}(C) - \\text{loss}(B) = -0.000224$, with a paired $L = 20$ moving-block 95% CI `[-0.000678, +0.000204]` that straddles zero.",
    )

    code(
        '''eur_scores = result["score_df"].query("series == 'EUR' and window == 'expanding'").copy()
eur_score_view = pd.DataFrame(
    {
        "Tail p": eur_scores["p"].map(lambda v: f"{v:.3f}"),
        "Arm": eur_scores["arm"],
        "Pinball loss": eur_scores["mean_pinball_loss"].map(lambda v: f"{v:.5f}"),
        "Violations": eur_scores.apply(lambda r: f"{int(r['violations'])}/{int(r['attempted'])} (exp {r['expected_violations']:.1f})", axis=1),
        "Rate (%)": (100.0 * eur_scores["violation_rate"]).map(lambda v: f"{v:.2f}%"),
        "Kupiec p": eur_scores["p_uc"].map(lambda v: f"{v:.3f}"),
        "Indep. p": eur_scores["p_ind"].map(lambda v: "N/A" if pd.isna(v) else f"{v:.3f}"),
    }
)
manifest = publish(
    "table",
    9,
    "EUR/USD expanding-window quantile forecast scores (2009-2025)",
    "What does updating shape add?",
    "Mean pinball loss, violation counts against nominal expectations, and asymptotic Kupiec unconditional-coverage and Christoffersen independence p-values on EUR/USD (4,253 origins).",
    "Eight-row score table comparing Arms A, B, C and D on EUR/USD at tail probabilities 0.01 and 0.025.",
    "Dynamic thresholding (Arm B vs Arm A) delivers the dominant reduction in pinball loss and restores realistic violation frequencies; differences among dynamic Arms B, C and D are an order of magnitude smaller.",
    manifest,
    table=eur_score_view,
    sidecar=eur_scores,
    table_height=4.8,
    col_widths=[0.08, 0.27, 0.13, 0.21, 0.10, 0.10, 0.11],
    font_size=9.0,
)''',
        "Table 09 compares all four arms on `EUR/USD` (`4,253` expanding-window origins) at $p = 0.01$ and $p = 0.025$:\n\n"
        "- **Scale adaptation dominates (`Arm B` vs. `Arm A`)**: Updating $\\hat{b}_t(1)$ while keeping $\\hat{\\gamma}_0$ fixed cuts mean pinball loss from `0.02738` to `0.02071` at $p = 0.01$ (`-0.00667`, 95% CI `[-0.00809, -0.00506]`) and from `0.05221` to `0.04198` at $p = 0.025$, lifting Kupiec unconditional-coverage $p$-values from `0.002` / `0.000` to `0.409` / `0.427`.\n"
        "- **Small spread across dynamic arms (`B, C, D`)**: Mean pinball losses across `Arms B, C, D` differ by at most `0.00022` at $p = 0.01$, and Christoffersen independence (`Indep. p <= 0.011`) rejects on all four arms because these unhedged raw-loss quantiles adapt at a smooth 10% kernel bandwidth rather than daily GARCH volatility.",
    )

    code(
        '''cb_p01 = result["contrast_df"].query("p == 0.01 and contrast_code == 'C_minus_B'").copy()
order_keys = [
    ("EUR", "expanding", "EUR (Expanding)"),
    ("EUR", f"rolling_{CFG.rolling_window}", "EUR (Rolling 2,500)"),
    ("GBP", "expanding", "GBP (Expanding)"),
    ("GBP", f"rolling_{CFG.rolling_window}", "GBP (Rolling 2,500)"),
    ("JPY", "expanding", "JPY (Expanding)"),
    ("JPY", f"rolling_{CFG.rolling_window}", "JPY (Rolling 2,500)"),
    ("JOINT_EQUAL", "expanding", "Joint Equal (Expanding)"),
    ("JOINT_EQUAL", f"rolling_{CFG.rolling_window}", "Joint Equal (Rolling 2,500)"),
]
rows_plot = []
for s_code, w_code, lbl in order_keys:
    r = cb_p01.loc[(cb_p01["series"] == s_code) & (cb_p01["window"] == w_code)].iloc[0]
    rows_plot.append(dict(label=lbl, **r.to_dict()))
cb_plot_df = pd.DataFrame(rows_plot)

def plot_cb_forest(ax):
    y_pos = np.arange(len(cb_plot_df))[::-1]
    scale = 1e4
    means = cb_plot_df["mean_diff"].to_numpy() * scale
    lo20 = cb_plot_df["ci20_low"].to_numpy() * scale
    hi20 = cb_plot_df["ci20_high"].to_numpy() * scale
    lo60 = cb_plot_df["ci60_low"].to_numpy() * scale
    hi60 = cb_plot_df["ci60_high"].to_numpy() * scale
    ax.hlines(y_pos, lo60, hi60, color="#94a3b8", linewidth=4.5, alpha=0.7, label="95% block-bootstrap CI (L=60)")
    ax.errorbar(
        means,
        y_pos,
        xerr=np.vstack([means - lo20, hi20 - means]),
        fmt="o",
        color=CFG.colors["raw"],
        ecolor=CFG.colors["raw"],
        elinewidth=2.0,
        capsize=4,
        markersize=6,
        label="Mean & 95% block-bootstrap CI (L=20)",
    )
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2, label="Equal loss (C = B)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cb_plot_df["label"])
    ax.set_xlabel("Mean daily pinball-loss difference: loss(C) - loss(B)  [x 10^-4 %-loss units; negative favors C]")
    ax.legend(loc="best", frameon=True, fontsize=9.2)

manifest = publish(
    "graph",
    10,
    "Updating tail shape (Arm C minus Arm B) across windows and currencies",
    "What does updating shape add?",
    "Mean daily pinball-loss difference loss(C)-loss(B) at p=0.01 across EUR, GBP, JPY and equal-weight common-date joint series for expanding and rolling-2,500 windows (scaled by 10^4).",
    "Horizontal interval plot showing Arm C minus Arm B mean pinball-loss differences and L=20 and L=60 moving-block bootstrap 95 percent intervals across eight currency-window combinations.",
    "Comparing expanding and rolling-2,500 windows alongside L=20 and L=60 block lengths exposes whether any incremental gain from updating gamma survives nonstationary history rules and wider dependence blocks.",
    manifest,
    plot=plot_cb_forest,
    sidecar=cb_plot_df,
)''',
        "Comparing $\\text{loss}(C) - \\text{loss}(B)$ (in units of $10^{-4}$ percentage-loss points) across all eight currency-window combinations at $p = 0.01$ shows that every 95% moving-block interval ($L = 20$ and $L = 60$) crosses zero:\n\n"
        "- **Expanding window**: Point differences are slightly negative on `EUR` (`-2.24e-4`), `GBP` (`-1.99e-4`), and `Joint Equal` (`-0.88e-4`), but positive on `JPY` (`+1.58e-4`, favoring fixed 2008 shape).\n"
        "- **Rolling 2,500-day window**: Capping the estimation memory at `2,500` days shrinks `EUR`'s point difference to near zero (`-0.015e-4`), while `GBP` (`-2.16e-4`) and `JPY` (`-1.26e-4`) stay well within their bootstrap uncertainty bands.",
    )

    code(
        '''sup_rows = []
for s_code in ("EUR", "GBP", "JPY", "JOINT_EQUAL"):
    for w_code, w_lbl in (("expanding", "Expanding"), (f"rolling_{CFG.rolling_window}", "Rolling 2,500")):
        r = cb_p01.loc[(cb_p01["series"] == s_code) & (cb_p01["window"] == w_code)].iloc[0]
        if s_code != "JOINT_EQUAL":
            sc_b = result["score_df"].query("series == @s_code and window == @w_code and p == 0.01 and arm_code == 'B_dyn_thresh'").iloc[0]
            sc_c = result["score_df"].query("series == @s_code and window == @w_code and p == 0.01 and arm_code == 'C_dyn_both'").iloc[0]
            sc_d = result["score_df"].query("series == @s_code and window == @w_code and p == 0.01 and arm_code == 'D_rect_hill'").iloc[0]
            b_c_d_str = f"{sc_b['mean_pinball_loss']:.4f} / {sc_c['mean_pinball_loss']:.4f} / {sc_d['mean_pinball_loss']:.4f}"
        else:
            b_c_d_str = "Equal-weight 3-FX"
        sup_rows.append(
            {
                "Series": "Joint 3-FX" if s_code == "JOINT_EQUAL" else s_code,
                "Window": w_lbl,
                "Loss B / C / D": b_c_d_str,
                "Mean C-B (e-4)": f"{r['mean_diff'] * 1e4:+.3f}",
                "CI L=20 (e-4)": f"[{r['ci20_low'] * 1e4:+.2f}, {r['ci20_high'] * 1e4:+.2f}]",
                "CI L=10 / L=60 (e-4)": f"[{r['ci10_low'] * 1e4:+.2f},{r['ci10_high'] * 1e4:+.2f}] / [{r['ci60_low'] * 1e4:+.2f},{r['ci60_high'] * 1e4:+.2f}]",
            }
        )
sup_view = pd.DataFrame(sup_rows)
manifest = publish(
    "table",
    11,
    "Supporting currency, window and block-length forecast summary (p=0.01)",
    "What does updating shape add?",
    "Pinball losses for Arms B, C and D and paired C-minus-B moving-block bootstrap intervals (L=20, 10, 60; 5,000 resamples) across EUR, GBP, JPY and the 4,253-date joint panel.",
    "Eight-row forecast comparison table for EUR, GBP, JPY and joint 3-FX across expanding and rolling-2,500 windows.",
    "All supporting series, window designs and block-length intervals are retained without cherry-picking; moving-block intervals remain approximate conditional summaries on correlated USD exchange rates.",
    manifest,
    table=sup_view,
    sidecar=result["contrast_df"],
    table_height=4.9,
    col_widths=[0.11, 0.13, 0.23, 0.13, 0.16, 0.24],
    font_size=8.6,
)''',
        "Table 11 lists the `Loss B / C / D` levels and paired `C - B` moving-block intervals across all three block lengths ($L = 20, 10, 60$). Varying $L$ from `10` to `60` days alters interval widths modestly without changing any conclusion, and Epanechnikov-weighted `Arm C` and rectangular `Arm D` perform nearly identically (`0.0205` vs. `0.0206` on `EUR`, `0.0226` vs. `0.0226` on `GBP`, `0.0233` vs. `0.0233` on `JPY`).",
    )

    # Section 8
    md("# Robustness and remaining uncertainty")

    code(
        '''rob = result["rob_df"]
rob_table_rows = []
for label, m_val, h_val, j_val in CFG.eur_robustness:
    sub = rob.loc[rob["setting"] == label].set_index("target_key")
    c95_val = sub.loc["raw", "c_95"]
    fl_str = f"{int(sub.loc['raw','floor_count'])}/{int(sub.loc['garch','floor_count'])}/{int(sub.loc['gjr','floor_count'])}"
    rob_table_rows.append(
        {
            "Setting": label,
            "mh / J": f"{m_val * h_val:.1f} / {j_val}",
            "Band c0.95": f"{c95_val:.3f}",
            "Raw Lambda (p)": f"{sub.loc['raw','lambda_stat']:.2f} ({sub.loc['raw','p_descriptive']:.3f})",
            "GARCH Lambda (p)": f"{sub.loc['garch','lambda_stat']:.2f} ({sub.loc['garch','p_descriptive']:.3f})",
            "GJR Lambda (p)": f"{sub.loc['gjr','lambda_stat']:.2f} ({sub.loc['gjr','p_descriptive']:.3f})",
            "Floors R/G/J": fl_str,
        }
    )
rob_view = pd.DataFrame(rob_table_rows)
manifest = publish(
    "table",
    12,
    "EUR/USD interior stability sensitivity across m, h and HAC lag J",
    "Robustness and remaining uncertainty",
    "Baseline and six one-factor EUR/USD perturbations on [0.15, 0.85], each recalibrated with 5,000 independent Pareto draws; p-values are descriptive outside the 9-test Holm family.",
    "Seven-row robustness table showing recalibrated band cutoffs, Lambda_n statistics, descriptive p-values and HAC floor counts across raw, GARCH and GJR targets on EUR/USD.",
    "Changing m, h or J shifts finite-sample studentization and critical values noticeably—especially when J=30 approaches or exceeds mh—confirming why single-bandwidth curves must not be over-interpreted.",
    manifest,
    table=rob_view,
    sidecar=rob,
    table_height=4.7,
    col_widths=[0.12, 0.11, 0.11, 0.18, 0.19, 0.18, 0.11],
    font_size=8.8,
)''',
        "Across all six one-factor perturbations of $(m, h, J)$ on `EUR/USD`—each recalibrated against its own `5,000`-draw Pareto cache—descriptive pivotal $p$-values stay between `0.139` and `0.495` across raw, GARCH, and GJR targets (`0/0/0` floor activations throughout). Notice how strongly the finite-sample simultaneous band multiplier $c_{0.95}$ varies with effective tail sample size $mh$ and Bartlett lag $J$ (`3.778` at `h = 0.125` up to `4.537` at `h = 0.075`), illustrating why asymptotic Gaussian critical values cannot substitute for finite-sample pivotal calibration.",
    )

    code(
        '''cal_df = result["cal_df"]
rec = result["record"]
cal_view = pd.DataFrame(
    {
        "Configuration": cal_df["label"],
        "(m, h, J)": cal_df.apply(lambda r: f"({int(r['m'])}, {r['h']:.3f}, {int(r['J'])})", axis=1),
        "Cache key": cal_df["short_key"].str.slice(0, 10),
        "Band c0.95": cal_df["c_95"].map(lambda v: f"{v:.3f}"),
        "c0.95 boot 95%": cal_df.apply(lambda r: f"[{r['c_95_boot_low']:.3f}, {r['c_95_boot_high']:.3f}]", axis=1),
        "Test Lam0.95": cal_df["lambda_95"].map(lambda v: f"{v:.3f}"),
        "Lam0.95 boot 95%": cal_df.apply(lambda r: f"[{r['lambda_95_boot_low']:.3f}, {r['lambda_95_boot_high']:.3f}]", axis=1),
    }
)
manifest = publish(
    "table",
    13,
    "Pivotal calibration cutoffs (B=5,000) and cutoff bootstrap uncertainty",
    "Robustness and remaining uncertainty",
    f"Seven hashed 5,000-draw Pareto calibration caches (35,000 paths total) with 1,000-bootstrap 95% cutoff intervals; live run completed in {rec['total_seconds']:.2f}s with {rec['failed_filters']} filter and {rec['failed_forecasts']} forecast failures.",
    "Seven-row table listing calibration configurations, short SHA-256 keys, band critical values c_0.95, omnibus critical values Lambda_0.95 and their 95 percent bootstrap intervals.",
    "With 5,000 pivotal draws per configuration, Monte Carlo uncertainty in c_0.95 and Lambda_0.95 is tight relative to empirical sampling noise; zero filter or forecast origins were dropped.",
    manifest,
    table=cal_view,
    sidecar=cal_df,
    table_height=4.7,
    col_widths=[0.21, 0.13, 0.12, 0.10, 0.16, 0.11, 0.17],
    font_size=8.6,
)
print(
    {
        "live_run": result["path"],
        "live_total_seconds": round(rec["total_seconds"], 3),
        "stage_seconds": {k: round(v, 3) for k, v in rec["stage_seconds"].items()},
        "max_rss_mib": round(rec["max_rss_bytes"] / (2**20), 1),
        "failed_filters": rec["failed_filters"],
        "failed_forecasts": rec["failed_forecasts"],
        "exported_assets": len(manifest),
        "mode": CFG.mode,
    }
)''',
        "Table 13 reports the 95% critical values ($c_{0.95}$ and $\\Lambda_{0.95}$) and their `1,000`-bootstrap confidence intervals across all seven `5,000`-draw Pareto calibration caches (`35,000` simulated paths total), followed by the runtime summary for `CFG.run_folder` (`0` failed volatility filters, `0` dropped forecast origins, all 13 assets exported).",
    )

    nb = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3 (trick-of-the-tail)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12.8"},
        },
    )
    out_path = PROJECT / "trick_of_the_tail.ipynb"
    nbf.write(nb, out_path)
    print(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    os.chdir(PROJECT)
    ns = namespace()
    if args.pilot:
        print("Running core_checks()...")
        chk = ns["core_checks"]()
        print(chk)
        ns["json_write"]("fixture-checks.json", chk)
        _, summary_df, common_dates = ns["load_fx_panel"]()
        print(summary_df.to_dict(orient="records"))
        print("Common scored dates:", len(common_dates))
        print("Running pilot()...")
        pilot_rep = ns["pilot"]()
        print(pilot_rep)
        ns["json_write"](
            "frozen-study.json",
            dict(
                configuration=ns["protocol"](),
                frozen_at=ns["datetime"].now(ns["ZoneInfo"](ns["CFG"].timezone)).isoformat(),
                plan="plan.md",
                plan_critique="plan-critique.md",
                reference_plan_v2="sources/trick-of-the-tail-proper-plan-v2.md",
            ),
        )
    if args.run:
        frozen = json.loads(Path("frozen-study.json").read_text())
        if frozen["configuration"] != ns["protocol"]():
            raise ValueError("CFG protocol diverged from frozen-study.json")
        run_dir = ns["run_study"]()
        print("Completed live run:", run_dir)
        ns["analyze_run"](run_dir, verify_estimators=True)
    build()


if __name__ == "__main__":
    main()
