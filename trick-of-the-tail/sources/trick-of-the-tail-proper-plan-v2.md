# Trick of the tail: implementation plan for a stronger empirical study

Status: planning only, revised v2 after independent critique, 7 November 2026. Implementation, installation, downloads, fitting and experiments are not authorized by this planning request. Independent review findings and accepted dispositions are recorded in Section 12. The existing `time-varying-tails/` study, source caches, article and bundle remain historical evidence.

## 1. Objective and decisions this study should support

Extend the local demonstration within Li and Peng's *Time-Varying Tails: Estimation, Inference, and Risk Measurement*, arXiv:2611.06005v1. The verified local source is [the supplied paper](papers/2611.06005v1.pdf); use its Sections 4–9 and Remarks 5.3, 6.1 and 7.2. The authoring contract is [paper-to-notebook-workflow.md](paper-to-notebook-workflow.md).

Answer two distinct questions:

1. **Inference:** is there evidence against constant tail shape after accounting for extremal dependence and volatility filtering?
2. **Prediction:** does updating tail shape improve extreme-quantile forecasts beyond updating the threshold alone?

A stability rejection does not establish forecast value; forecast value does not require rejection. More significant results are not the objective. Constant shape, inconclusive tests, inferior kernel forecasts and failed calibration are legitimate outcomes.

The previous experiment found that the kernel beat pooling in the independent drifting Pareto world, while rectangular rolling Hill had slightly lower gamma error. Pooling won in the stationary and AR designs. Its IID-reference band covered only 55% of dependent paths. These motivate the extension; they are not expected outcomes to reproduce on real data. EUR/USD 1999–2025 has already been inspected. Its further analysis is exploratory, and reconstructed past-only forecasts are temporal pseudo-out-of-sample evidence rather than a pristine new holdout.

## 2. Scope and correspondence with the paper

| Component | Planned implementation | Claim boundary |
|---|---|---|
| Local quantile and Hill ratio | Preserve Eqs 4.2, 4.10–4.11, Epanechnikov weights, strict exceedances and weighted-CDF left inverse | Same estimator mechanism; optimized paths require reference checks |
| Dependence adjustment | Implement residualized exceedances and local Bartlett HAC factor, Eqs 5.2–5.3 | Variance correction; not a remedy for finite-threshold/smoothing bias |
| Uniform bands | Eq 5.7, finite-sample Pareto calibration with identical full studentization | Simultaneous coverage on the declared finite grid; no empirical true gamma available |
| Stability test | Eqs 6.2–6.4, precision-weighted projection centering and matched pivotal calibration | Approximate finite-sample validity assessed in simulations, not guaranteed by a smooth empirical path |
| Volatility filtering | GARCH(1,1) and GJR-GARCH(1,1), Student-t likelihood, absolute standardized innovations | Filtered absolute tails and raw positive-loss tails are different estimands |
| Risk forecasting | Eq 7.1, past-only, one-sided endpoint implementation | Deliberate boundary prediction design; interior band theory does not apply |
| Evaluation | Three FRED exchange-rate series, three threshold/shape constructions plus rolling Hill | Restricted financial-domain adaptation, not S&P 500 replication or general superiority |

Keep expected shortfall, Eq 7 joint risk-confidence intervals, an S&P reproduction, alternative tail models and automatic tuning outside the required implementation. They are not needed to answer the two questions. Do not substitute risk confidence intervals that treat an estimated threshold as known.

## 3. Data, exposure direction and evidence status

Use these three fixed series for 4 January 1999 through 31 December 2025:

| Series | Official source | Price used before losses |
|---|---|---|
| EUR | [DEXUSEU](https://fred.stlouisfed.org/series/DEXUSEU), USD per euro | Published quote |
| GBP | [DEXUSUK](https://fred.stlouisfed.org/series/DEXUSUK), USD per pound | Published quote |
| JPY | [DEXJPUS](https://fred.stlouisfed.org/series/DEXJPUS), yen per USD | Reciprocal quote, USD per yen |

All represent losses on a foreign-currency holding valued in USD: `loss_t = -100 * diff(log(USD_per_foreign_currency))`. Inverting JPY before differencing is essential. The official JPY/GBP metadata were checked during planning; their histories have not been downloaded or inspected here. All three are related USD exchange rates, not independent market replications or a diverse asset panel.

After implementation approval, copy the existing EUR cache unchanged into the new package, download GBP/JPY separately, and save original bytes, exact URLs, timestamps, metadata, rights/attribution and SHA-256. Pin the observed revision; never silently replace the existing EUR cache. Record previous cache hash `882c3431d37aee37fd7fd8e17e1715be46b3529a51446f559978fed60cdc4f40`. Its old sample has 6,769 signed losses; recompute and report new-package counts rather than assume them for other series.

Restrict dates explicitly, require unique increasing dates and positive quote levels, remove missing quotes without interpolation, and retain signed/zero losses. Returns span successive available quotes and may span holidays; these are next-observed-quote forecasts, not fixed 24-hour returns. Each series keeps its own observation index and dates. Raw-series tests use the available sample independently. Across-series forecast summaries use the intersection of scored dates, retaining series-specific results on their full dates as well. No filled returns or score-based exclusions.

Freeze a source/split manifest before empirical inference or backtest scoring. New GBP/JPY analyses are prespecified extensions, not independent confirmation of a hypothesis developed without seeing financial data. Strong confirmatory evidence would need a subsequently untouched period or external sample under the frozen protocol.

## 4. Numerical implementation and reference checks

Retain the current scalar/direct estimator as a readable reference; add an optimized implementation for thousands of repetitions and daily endpoints. Sorting, cumulative weighted-CDF inversion and positive-threshold masking must preserve finite-sample semantics. Never log negative/zero observations then mask invalid values afterward.

For each interior evaluation point u, calculate `Z_t(u) = 1{X_t>b_hat(u)} * (log(X_t/b_hat(u)) - gamma_hat(u))`. With `a_t(u)=K_h(u-t/n)*Z_t(u)`, implement the HAC factor as `1 + 2*sum_{j=1..J} (1-j/J)*sum_t a_t*a_(t+j)/sum_t a_t^2`, using only valid lag pairs, without wrapping indices. Keep the paper's 0.05 positive floor, and record unfloored values and every floor activation. The j=J taper is zero. Zero denominators/nonfinite results are explicit invalid fits, not silently repaired.

On the fixed interior grid, use trapezoidal quadrature consistently for projection centering, T and Lambda: `gamma_tilde = integral(gamma_hat/S_hat)/integral(1/S_hat)`, `T=integral((gamma_hat-gamma_tilde)^2/S_hat)`, `Lambda=mh*T/(gamma_tilde^2*(3/5))`. An ordinary mean or pooled Hill center is not interchangeable. Bands use `gamma_hat ± c*gamma_hat*sqrt(S_hat*(3/5)/(mh))`; do not clip a negative lower endpoint without reporting the change.

Required checks include hand-computed HAC/lag fixtures, direct weighted-quantile/tie fixtures, gamma/threshold agreement between optimized and reference implementations (`rtol=1e-9, atol=1e-11` float64), constant global-scale invariance, endpoint support and exclusion of the forecast target. Test shuffled/independent residuals as diagnostics, not as a proof that any realized HAC factor must equal one. Verify that changing values strictly after a forecast origin leaves every earlier filter, threshold, gamma and forecast unchanged. Check GJR leverage sign using signed returns: negative returns correspond to positive losses. Reconcile innovations, conditional variance and standardization on a short independent recursion fixture.

## 5. Interior empirical inference and multiplicity

Baseline: m=200, h=.10, J=20, 41 equally spaced points on the common interval [.15,.85]. Use each series' own n, so q=m/n differs. Report n, q, mh and actual weighted exceedance information. This follows the paper's empirical-scale settings but is not a claim that its asymptotic rate requirements are numerically satisfied. In particular, large J relative to mh requires finite-sample validation.

Three targets per series: raw positive-loss upper tail, absolute standardized GARCH innovations, and absolute standardized GJR innovations. Fit full-sample filters only for retrospective inference; these fits never feed the forecasting experiment. Use signed percentage returns, a constant conditional mean, Student-t innovations with estimated degrees of freedom, and constrained covariance-stationary variance parameters. Define residuals using the fitted conditional mean/variance. Fix the package API, variance backcast, initialization, parameter bounds and convergence tolerances after source/API inspection and development-only checks, before the study freeze. Log them; do not rely on undocumented defaults.

Estimate the paths, thresholds, HAC factors, projection levels, omnibus statistics and bands. Report residual/extreme-score autocorrelations and convergence diagnostics. Do not assert that filtering removed dependence merely because average S is near one. A smaller residual gamma estimates a different target, not an automatic correction of raw gamma.

For each distinct `(n,m,h,J,kernel,grid,quadrature,floor)` configuration, simulate 5,000 IID unit-scale Pareto samples, recompute the full estimator/HAC/projection, and retain band maximum statistics and Lambda. Use gamma=.3, strict quantiles, independent calibration streams and empirical .95 cutoff with `method='higher'`. For band calibration, the maximum statistic is `max_u abs(gamma_hat(u)-gamma_0)/(gamma_hat(u)*sqrt(S_hat(u)*(3/5)/(m*h)))`, with generating gamma_0=.3. Do not replace gamma_0 by the estimated projection level: projection centering belongs to Lambda, not this band statistic. Test p-values use `(1 + number(calibration_stat >= observed_stat))/(B+1)`. Hash all calibration-key fields; no reuse across incompatible n, grids, lags or supports. Gamma-scale pivotality is a theoretical motivation, not a finite-sample guarantee under every generator.

The primary inference family contains nine baseline stability tests (three series by three targets); report raw and Holm-adjusted p-values at nominal familywise alpha=.05. This is nominal, paper-calibrated inference subject to finite-sample validation limits, not a demonstrated guarantee of actual 5% familywise control. Dependence between the exchange rates does not justify counting them as independent confirmations. Bands are 95% within each target's 41-point path, not familywise across all nine paths. Label that distinction explicitly.

EUR robustness uses six one-factor deviations: m=150/250, h=.075/.125, J=10/30, retaining other baseline values and the same [.15,.85] grid. Recalibrate each configuration. GBP/JPY use baseline only in the required profile. Show all robustness results; no best-setting selection and no significance claim based on a sensitivity result after a null baseline. Sensitivity p-values are descriptive, outside the nine-test primary family.

## 6. Validate calibration and the entire filter-and-test pipeline first

Use separate calibration, development/pilot, validation and power RNG namespaces. Proposed master seed 20261107, PCG64/SeedSequence `[master,phase,series,design,replicate,stage]`; phase codes 10 pilot, 20 calibration, 30 null validation, 40 power, 50 uncertainty. Save exact mappings and draw IDs. No outcomes determine seed or replicate selection.

At every distinct empirical n and baseline inference configuration used by the three series, run 500 independent validation histories under each fixed null. Cache reuse requires an exactly identical full configuration, not merely a similar n:

* IID Pareto gamma=.3; deterministic scale drift `1+.8*sin(2*pi*u)^2` multiplying that Pareto; IID Student-t with df=10/3.
* AR-Pareto with coefficient .3, innovation coefficient .7 and constant gamma=.3.
* Stationary stochastic volatility: log-scale AR(.97), stationary SD .2, independent standardized t(10/3) innovations.
* Stationary GARCH-t: omega=1e-5, alpha=.105, beta=.879, standardized t(10/3) innovations. Simulate 5,000 burn-in steps from unconditional variance; multiply simulated returns by 100 consistently for percent-return numerical units.

Apply the raw-series test to every null. For GARCH-t, stochastic volatility and IID Student-t, additionally fit both volatility filters afresh on every simulated history and test the absolute standardized innovations. A fit to the observed series reused inside simulations does not validate filtering. For GARCH raw tails, assess constant-shape rejection size without inventing an exact known gamma for band coverage. Check whole-grid band coverage only where a genuine marginal/innovation gamma reference is known and the target matches it.

For each series, add two fitted-null diagnostics: 500 histories generated from its development-prefix fitted GARCH parameters, and 500 from its development-prefix fitted GJR parameters. Generate full empirical-length histories, after 5,000 burn-in steps, and refit both filters on every history. Only generating parameters come from the 2008 prefix; prefix-length histories would validate a different n. Reject invalid/nonstationary generating fits explicitly. These are model-conditional size checks, not replacements for pivotal p-values or assumption-free validation. Include all generation/refitting costs in pilot projections. Exact prefix is defined in Section 7; generating parameters are frozen before full-sample tests.

Reproducibility details: AR null starts at X=1 and burns 5,000 steps with constant gamma=.3; the SV log-scale starts at zero, burns 5,000 steps, and uses innovation SD `.2*sqrt(1-.97**2)`. Standardize Student-t draws as `T_nu/sqrt(nu/(nu-2))` whenever unit-variance innovations are specified, requiring nu>2. GARCH/GJR initial variance is its finite unconditional variance (GJR denominator `1-alpha-beta-leverage/2` for symmetric innovations); discard generator burn-in only. For symmetric-return generators, calculate signed percentage returns first and define losses as their negatives; SV and IID-t have one-percentage-point innovation scale, while GARCH uses the explicit factor 100 above. Pareto and AR-positive observations are already losses. All fitted residual observations are retained, including initialization observations under the frozen backcast convention; mark the first 250 for diagnostics, without trimming them in response to results. Full-pipeline simulations must use that identical fitting/initialization treatment.

Admission mapping: each raw target is checked against all six fixed raw nulls at its own configuration; each filtered target is checked against that filter applied to the three designated fixed nulls (GARCH-t, SV and IID-t), plus both series-specific fitted stationary nulls. Run the fitted-null raw tests as additional diagnostics. A relevant failure blocks that target's nominal rejection claim. EUR sensitivity configurations are recalibrated but remain descriptive because a full new null-validation family is not part of the required profile.

Report rejection counts/denominators, Wilson intervals and calibration-cutoff Monte Carlo uncertainty separately. Bootstrap the stored 5,000 calibration statistics 1,000 times and recompute validation rejection rates conditional on saved validation statistics, to expose cutoff uncertainty; this is not another independent null sample.

All 500 planned histories remain in every simulation denominator, whether attempted or not. Record successful rejections, successful nonrejections, unresolved fit failures and unattempted work separately. Do not report successful-fits-only rejection rates. An unresolved simulated-filter failure makes that pipeline's size assessment incomplete and blocks admission; optionally show the worst-case size range R/500 through (R+F+U)/500 (R rejections, F unresolved failures, U unattempted histories), clearly labelled as bounds. Band failures similarly remain visible rather than being dropped.

Interpretability gate: a target/pipeline whose baseline null-size Wilson upper endpoint exceeds 10% on a relevant simulated null is not admitted for a nominal-5% empirical rejection claim. Report its empirical p-values descriptively and mark calibration inadequate. An upper endpoint below 10% excludes severe inflation only under that simulated null; it does not establish exact size. Report strong conservatism and power as well. No post hoc cutoff inflation or lag adjustment to obtain a pass. Any methodological revision requires a new version, new validation streams and independent re-review before empirical interpretation.

Power at the EUR empirical n and baseline settings: 500 independent histories each for IID Pareto slope `.2+.3u`, Pareto break `.3+.2*1{u>=.5}`, and absolute Student-t slope `.2+.15u` with df=1/gamma. These are raw-target power studies; do not force a variance-filter interpretation onto infinite-variance Pareto alternatives. Rejection is expected to vary by departure size and bias. No power minimum is an outcome gate.

## 7. Strictly past-only quantile forecasting

Initial prefix: all observed returns through 31 December 2008. Score the next available observations dated 1 January 2009 through 31 December 2025. Verify at least 2,000 initial observations per series. Prefix data may be used for fit initialization, debugging and simulation-parameter diagnostics; no tuning on later loss scores. Dates are an a priori implementation choice for this extension, not an untouched historical split.

Use **raw signed losses** for the required forecast comparison, keeping it separate from the filtered-tail inference exercise. Primary window is expanding; required sensitivity is rolling 2,500 observations. At every origin t, use observations through t-1 only, with one-sided Epanechnikov support at u=1, h=.10, fixed m=200 and q_t=m/n_t. Require p<q_t and valid positive thresholds. No endpoint confidence bands from the interior theory.

For each window rule and series, compare all four fixed constructions:

| Arm | Intermediate threshold | Gamma | Purpose |
|---|---|---|---|
| A: fully static | Initial-window threshold | Initial-window kernel endpoint gamma | Cost of leaving both unchanged |
| B: dynamic threshold, constant shape | Current kernel endpoint threshold | Same initial-window gamma as A | Adapt scale/location without updating shape |
| C: dynamic threshold and shape | Same current threshold as B | Current kernel endpoint gamma | Incremental value of updating gamma |
| D: rectangular rolling Hill | Rectangular endpoint threshold on the same support | Current rectangular Hill gamma | Practical simple-method comparator |

A and B use the initial kernel endpoint gamma from the common 2008 prefix for both window designs, explicitly recording this design choice. A's quantile remains fixed: its extrapolation uses initial q_0. B/C use q_t; D uses its own matching empirical quantile at q_t. Thus B and C differ only in gamma updating. A-versus-C changes multiple components and cannot isolate tail shape. No full-sample gamma or residual filter enters these forecasts.

For tail probabilities p=.01 (primary) and .025 (secondary), calculate `Q_hat = b_hat*(q/p)^gamma_hat`. The quantile level is tau=1-p. Primary loss is pinball/check loss `rho_tau(y-Q)= (tau-1{y<Q})*(y-Q)`. Equal losses at y=Q contribute zero; a violation is strictly y>Q. Retain every date, input cutoff, n_t, q_t, threshold, gamma, forecast, realized loss, score and fit status.

Primary prediction contrast: mean daily loss(C)-loss(B), EUR, p=.01, expanding window. Negative is favorable. Report A/B/C/D losses and all differences without selecting the best reported method. A claim of incremental predictive improvement under the declared approximate uncertainty procedure requires the primary paired 95% block-bootstrap interval below zero and no unresolved failure accounting. GBP/JPY, rolling windows, other pairs and p=.025 are prespecified supporting analyses; do not promote them to primary after seeing outcomes.

Do not expand to filtered quantile forecasting in the required version. That would need its own signed-tail mapping, conditional mean/variance forecasts, residual threshold and constant-shape ablations; absolute residual inference is not automatically a one-sided loss forecast.

## 8. Forecast uncertainty, calibration and failures

Use a paired circular moving-block bootstrap of the chronological per-date score differences: block length 20 observed forecast dates, 5,000 resamples, concatenate sampled blocks and truncate to original length, percentile .025/.975 with `method='linear'`. Report sensitivity at lengths 10 and 60 even when their signs or interval conclusions disagree. These intervals are not guaranteed valid under unrestricted nonstationarity; they support a conditional historical comparison under a block-resampling approximation, not assumption-free future superiority. This is uncertainty conditional on the realized forecasting protocol, subject to dependence/nonstationarity limitations; it does not rerun model selection or establish independent market replications. Across-series summaries resample the same common-date blocks jointly; report equal series weights and retain the primary EUR result separately.

Report quantile loss, violations/attempted forecasts, expected nominal violation count, rolling 250-observation violation counts, unconditional-coverage and first-order independence tests as diagnostics. Define these tests from their likelihoods, handle zero cell counts by continuous `0*log(0)=0`, and return N/A when a transition test is unidentified. Rare violations make asymptotic p-values fragile; do not equate nonrejection with calibration. At 99%, only tens of violations may be expected in the scored period. Interval reporting is essential; no “closer to 1% wins” ranking without predictive-loss evidence.

Fail explicitly on missing source files, invalid thresholds, zero score variance when a method needs it, optimizer nonconvergence or nonfinite outputs. For filters, use a deterministic two-attempt development-frozen initialization schedule and unchanged constraints; retain both attempts. A remaining invalid filter blocks that target's complete inference, not the other targets. Preserve all nine prespecified Holm slots: use p=1 only internally for failed or unadmitted slots, label their reported adjusted decision N/A, and retain their observed paper-calibrated p-values as descriptive if available. Never silently shrink the family; passing the simulation screen does not guarantee the remaining empirical p-values are exactly valid. No post hoc removal of a hard series. For forecasts, any missing arm/date prevents a complete primary paired comparison: retain partial results and denominators, report incomplete, repair correctness issues under a versioned rerun before claiming a winner. Do not assign zero loss, silently carry a forecast forward or report a favorable common-success subset as the original comparison.

## 9. Feasibility, environment and freeze gates

Planning inspection found the existing numerical/notebook environment, but `arch`, `scipy` and `statsmodels` are absent from `.venv`. Do not install them for this request. After implementation approval, create an isolated `.venv-tail-proper`, use CPU float64 and pin a compatible released `arch`/SciPy stack after API/version checks. `statsmodels` is optional; do not add it merely to compute simple diagnostics. Record package versions, BLAS/thread settings, platform and kernel imports. Existing M4 Max/128 GB hardware is historical feasibility context, not a current runtime guarantee. No LLM or accelerator is required.

Proposed numerical allowance: **180 minutes**, including pilots, calibration, null/power simulation, forecasts, replay and exports, with 16 GiB process RSS ceiling and at most four CPU workers; downloads/install and prose authoring are separately timed. Reserve 30 minutes for replay/export and failure diagnostics. Guard total job wall time and aggregate worker RSS; record active stage timings. No one-hour Jupyter cell without checkpoints.

Pilot before study outcomes: 20 calibration draws per distinct profile; ten null histories per design/configuration including both fitted-null generators, with complete GARCH/GJR refits; 100 daily forecast origins from the 2008 development prefix, using earlier pseudo-origins rather than 2009+ outcomes. Benchmark the complete endpoint and HAC paths, parameter-boundary cases, convergence and cache reuse. Project all tasks with a 3x safety factor plus reserve. Profile switching may use these timings only, never empirical p-values or forecast errors.

Required full profile is the counts above. If it cannot fit 180 minutes, do not silently reduce validation precision, omit controls or replace daily forecasts with weekly ones. Prepare a revised workload/count budget with independent critique before evaluation. An explicit possible revision is fewer precision-labelled simulation draws while retaining every null, comparator and empirical series; its confidence intervals and interpretability gates must be reconsidered first. Do not increase the budget automatically.

Gate sequence: paper-formula/reference agreement → data and environment verification → development pilot → freeze hashes/settings/IDs/streams/budget → calibration/null/power validation → independent assessment of calibration limitations → empirical inference and chronological forecasts → analysis/export → fresh-kernel acceptance. Study membership and primary comparisons cannot change after freeze. Inadequate null size changes the allowed interpretation, not which unfavorable results are retained.

## 10. Implementation stages and retained evidence

Build a new `trick-of-the-tail-proper/` package without editing the old experiment. Proposed stages:

1. **Protocol and sources:** transfer this reviewed plan, pin data/source PDF/environment, freeze date/exposure/split manifests.
2. **Estimator and HAC:** implement reference and optimized interior/endpoint paths; complete numerical and leakage fixtures.
3. **Filters:** add deterministic GARCH/GJR adapters and recursion checks, preserving raw parameters, optimization traces and residuals.
4. **Pilot and freeze:** project measured full-pipeline costs, admit or revise; save immutable configuration and code hashes before evaluation.
5. **Calibration and validation:** save complete calibration statistics, simulated inputs/seeds, refitted filter results, null/power outcomes and admission dispositions.
6. **Real study:** write retrospective inference evidence separately from chronological forecast records; checkpoint per series/window and fail visibly.
7. **Analysis:** compute multiplicity corrections, all loss/coverage summaries and paired uncertainty from raw evidence. Write a machine-readable primary-result record even for null/negative/incomplete outcomes.
8. **Notebook and publication:** produce the inline notebook, individually numbered assets, independently reviewed outline, article/SEO/LinkedIn and validation record; no publishing or sending.

Suggested paths: `data/`, `sources/`, `calibration/<key>/`, `runs/<timestamp>/`, `graphs/`, `trick_of_the_tail_proper.ipynb`, `requirements.txt`, `dependency-audit.md`, `README.md`, `post-plan.md`, `post-plan-review.md`, `post.md`, `seo.txt`, `linkedin.md`, `asset-manifest.json`, `validation.md`, `delivery-audit.json`. Authoring helpers may live outside the package, but the notebook must not import repository implementation modules.

Retain full-precision per-origin and per-replicate evidence, all attempts/failures, convergence/floor flags, source/code/config hashes, calibration cache keys, timestamps, RSS/time traces, bootstrap indices and Holm family membership. Live modes create new runs. Replay reconstructs all scores/tests/assets from genuine evidence and never fetches data, refits filters or creates replacement study inputs. Deterministic estimator recomputation can be a separate verification mode. Preserve immutable inference and forecasting timings during formatting reruns.

## 11. Notebook, graphics and final acceptance

One imports cell; adjustable settings under CFG; all reusable functions/classes inline under themed Utils; short narrative calls followed immediately by explanations. Shared notebook/article headings: (1) A tail curve is not a finding, (2) One threshold and dependent extremes, (3) Can the test control false alarms?, (4) Raw losses and standardized innovations, (5) Does tail shape stay constant?, (6) Predict tomorrow using yesterday, (7) What does updating shape add?, (8) Robustness and remaining uncertainty, (9) Closing time. Setup/Utils are notebook-only.

Proposed main/supporting assets in one shared display sequence: 01 paper-to-study scope table; 02 null-size graph; 03 power graph; 04 raw EUR path/band; 05 GARCH absolute-residual path/band; 06 GJR path/band; 07 baseline nine-test table with Holm p-values; 08 primary paired forecast-loss graph; 09 A/B/C/D score table; 10 violations graph; 11 rolling-window sensitivity graph; 12 GBP/JPY supporting forecast table; 13 threshold/bandwidth/HAC robustness table; 14 filter/HAC diagnostics table; 15 workload/failure table. Split any later added graph by target/series into separate files, updating notebook/post placement and manifest. No multipanel figures.

Capture Amsterdam export date once and use `graphs/graph_<NN>_<ddmm>.png` or `graphs/tabler_<NN>_<ddmm>.png`, two digits shared across graphs/tables. Supply full-precision sidecars, exact captions, alt text, interpretation and source run. Show the same PNG bytes inline. Archive old outputs and inspect every final image; additional assets do not authorize changing comparisons.

Acceptance requires all declared work complete or explicitly marked incomplete; numerical/reference and future-perturbation checks passed; calibration limits and multiplicity visible; full primary denominators; no forecast look-ahead; current import audit; independent outline critique/dispositions; fresh-kernel copied-directory replay with network/fitting/generation blocked; exact source/ID/config/hash agreement and numerical tolerances set above. Recompute test statistics, forecast losses, adjustments and uncertainty from external evidence. Cross-check all metrics, signs, grid coverage labels, dates, section order and asset references. Validate ZIP content byte hashes.

The final empirical claim must say separately whether baseline shape stability was rejected under adequately validated settings, whether C improved over B on the primary prediction contrast, and whether supporting series/window designs agreed. A mixed or null result still completes a stronger empirical study.

## 12. Independent critique and disposition

The independent `tails_critic` agent reviewed this document and the relevant paper formulas. It found no fundamental scientific blocker and confirmed the HAC formula, projection centering, Lambda normalization, exposure directions and primary C-versus-B contrast. The primary agent assessed and accepted all findings below. These revisions complete the requested planning stage; implementation remains unstarted.

| Finding | Assessment and change |
|---|---|
| P2: Band and test calibration could be conflated | Accepted. Band maxima center on generating gamma_0=.3; only Lambda uses projection centering. |
| P2: Fitted-null repetitions and sample length missing | Accepted. Specify 500 histories per prefix-fitted GARCH/GJR generator for each series, at full empirical n, with both filters refitted and costs piloted. |
| P2: Admission did not cover other sample configurations explicitly | Accepted. Validate every distinct baseline empirical configuration and map raw/filtered targets to relevant fixed and fitted nulls; sensitivity tests stay descriptive. |
| P2: Failed replicates and failed Holm slots underspecified | Accepted. Keep all attempted replicates; unresolved failures block admission, with optional worst-case bounds. Keep nine Holm slots, using conservative internal placeholders and N/A decisions for unavailable tests. |
| P2: Error-control wording too strong | Accepted. A gross-inflation screen is not proof of 5% size; nominal Holm decisions remain conditional on validation limits. |
| P3: Predictive bootstrap interpretation | Accepted. Treat moving-block intervals as approximate under an evolving/nonstationary historical protocol and retain all block-length sensitivity outcomes. |
| P3: Simulation initialization and sign conventions | Accepted. Specify AR/SV states and burn-in, standardized t innovations, GARCH/GJR unconditional-variance initialization, return/loss orientation and retention of fitted initialization observations. |

The review caused no change to the primary comparison, data period, arm membership or outcome-independent workload gates. All newly explicit simulation costs are included in the pre-execution pilot; no feasibility measurement is claimed for this extended design.

A focused independent re-review of v2 found no blocking inconsistencies. Its final accounting clarification was accepted: worst-case rejection bounds include unattempted histories as well as failed fits. Local source links and the twelve-section document structure were checked.
