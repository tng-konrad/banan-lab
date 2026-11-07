# Trick of the tail: empirical currency tail inference and forecasting

Open [the executed notebook](trick_of_the_tail.ipynb) or [the companion article](post.md). This consolidated package adapts Shuo Li and Liuhua Peng, *Time-Varying Tails: Estimation, Inference, and Risk Measurement* ([verified local PDF](sources/2611.06005v1.pdf), arXiv:2611.06005v1, Sections 4–9 and Remarks 5.3, 6.1 and 7.2) to three daily USD exchange-rate series (`1999–2025`), building on [the v2 study specification](sources/trick-of-the-tail-proper-plan-v2.md) while archiving the earlier exploratory synthetic + EUR walkthrough in [archive/v1-time-varying-tails/](archive/v1-time-varying-tails/README.md).

Scope boundary: per explicit study brief, the synthetic Monte Carlo null-size and power simulation study was omitted so this package focuses strictly on real-data empirical inference and chronological extreme-quantile forecasting. Finite-sample critical values and empirical $p$-values use the paper's `B = 5,000` pivotal IID unit-scale Pareto calibration ($\gamma_0 = 0.3$) per distinct `(n, m, h, J, grid)` key; stability test outcomes are therefore reported as **nominal paper-calibrated pivotal inferences** rather than simulation-certified 5% error guarantees. Expected shortfall, joint risk confidence intervals and S&P 500 equity replication are outside scope.

## Install and register the isolated kernel

From `trick-of-the-tail/` with Python 3.12:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name trick-of-the-tail --display-name "Python 3 (trick-of-the-tail)"
```

Exact transitive versions are locked in [requirements.lock.txt](requirements.lock.txt), and every direct notebook import is audited in [dependency-audit.md](dependency-audit.md). JupyterLab is an optional interactive editor and is not required for offline replay validation.

## Separate FRED data caches and attribution

Raw daily exchange-rate series live under `data/` and are never embedded inside notebook code cells:

- [DEXUSEU](https://fred.stlouisfed.org/series/DEXUSEU) (`data/DEXUSEU.csv`, USD per euro, copied unchanged from [archive/v1-time-varying-tails/data/DEXUSEU.csv](archive/v1-time-varying-tails/data/DEXUSEU.csv), SHA-256 `882c3431d37aee37fd7fd8e17e1715be46b3529a51446f559978fed60cdc4f40`)
- [DEXUSUK](https://fred.stlouisfed.org/series/DEXUSUK) (`data/DEXUSUK.csv`, USD per pound, SHA-256 `e1cc3c8ab506146e1596094cf7ff1a66ca4b778ff78a20cd4b17add16bfd55bd`)
- [DEXJPUS](https://fred.stlouisfed.org/series/DEXJPUS) (`data/DEXJPUS.csv`, yen per USD, inverted to USD per yen before log differencing, SHA-256 `81852a3263d9297d5452fd87f3e67fb8453f8195a2eb3d1ddfad618d20003764` — exact hash verified in [data/source-manifest.json](data/source-manifest.json))

Source attribution: Board of Governors of the Federal Reserve System (US), H.10 Foreign Exchange Rates release, distributed via FRED, Federal Reserve Bank of St. Louis (Public Domain: Citation Requested).

To fetch fresh candidate CSVs for inspection without overwriting the pinned study caches:

```bash
mkdir -p data
curl --fail --location 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXUSEU&cosd=1999-01-04&coed=2025-12-31' -o data/DEXUSEU.candidate.csv
curl --fail --location 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXUSUK&cosd=1999-01-04&coed=2025-12-31' -o data/DEXUSUK.candidate.csv
curl --fail --location 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXJPUS&cosd=1999-01-04&coed=2025-12-31' -o data/DEXJPUS.candidate.csv
shasum -a 256 data/*.candidate.csv
```

## Run All: offline replay or live study

Default `CFG.mode = "replay"` (`TAILS_MODE=replay`) verifies all source, calibration and run SHA-256 checksums, reconciles GARCH/GJR conditional variances in pure NumPy without calling an optimizer, recomputes interior tail paths, Bartlett HAC factors, 9-slot Holm step-down $p$-values, past-only quantile forecasts, Kupiec/Christoffersen likelihood-ratio diagnostics and `5,000`-draw paired moving-block bootstraps, and exports all 13 single-panel PNG assets and CSV sidecars.

Copy these paths together for standalone replay:

- `trick_of_the_tail.ipynb`
- `data/` (`DEXUSEU.csv`, `DEXUSUK.csv`, `DEXJPUS.csv`, `source-manifest.json`)
- `sources/` (`2611.06005v1.pdf`, `trick-of-the-tail-proper-plan-v2.md`)
- `calibration/` (seven hashed `B = 5,000` Pareto calibration caches)
- `frozen-study.json`, `pilot-admission.json`, `latest-run.json`, and the `runs/<timestamp>/` directory named in `latest-run.json`

To execute a new live run under the frozen protocol, set `TAILS_MODE=live` before launching the kernel (or set `CFG.mode = "live"`). Live runs create a new timestamped directory under `runs/` without overwriting prior runs. Authoring helpers and verification scripts live locally under [src/](src/utils.py) and [scripts/](scripts/build_notebook.py), while the delivered notebook embeds all project code inline inside `# Utils` with zero imports from local modules.

## Package artifacts

- [Study plan](plan.md) and [independent plan critique](plan-critique.md)
- [Frozen protocol](frozen-study.json), [pilot admission](pilot-admission.json), [latest run pointer](latest-run.json), and [primary comparison record](primary-comparison.json)
- [Asset manifest](asset-manifest.json): 13 single-panel `graphs/graph_<NN>_<ddmm>.png` and `graphs/tabler_<NN>_<ddmm>.png` files with full-precision `.csv` sidecars
- [Post outline](post-plan.md), [outline review](post-plan-review.md), [companion article](post.md), [SEO description](seo.txt), and [LinkedIn draft](linkedin.md)
- [Validation summary](validation.md), [fresh-kernel replay record](fresh-kernel-validation.json), [visual QA record](visual-qa.json), and [delivery audit](delivery-audit.json)
- [Archived v1 exploratory study](archive/v1-time-varying-tails/README.md) (`time_varying_tails.ipynb`, synthetic + EUR walkthrough)
