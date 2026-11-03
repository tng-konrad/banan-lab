# Source.ag interview preparation — initial plan for independent critique

Date: 1 November 2026. Status: planning only; implementation requires the user's approval.

## 1. Intended outcome and boundaries

Prepare the candidate for the discussion Tobias described: forecasting greenhouse harvest with limited observations, unfamiliar conditions, simulated future climate, and physical constraints. The aim is an explanation the candidate can defend at a whiteboard, supported later by a small, reproducible experiment.

The central question is: **When observations do not cover the conditions we must forecast, which structure can support extrapolation, which assumptions remain unverified, and when should the system reduce its confidence or defer to a grower?**

This phase produces the plan, an independent critic review, and a revised plan only. No dataset downloads, simulator, model, notebook, completed product report, or production implementation until approval. Existing repository work is unrelated and will remain untouched. Assume a technical conversation of roughly 45–60 minutes, with an optional 15–20 minute notebook walkthrough; these are preparation choices, not a confirmed interview format.

## 2. Preliminary findings that shape the plan

- The [job description](https://careers.source.ag/o/staff-data-scientist-forecasting) emphasizes unusual weeks, sparse manual samples, delayed feedback, changing grower decisions, hybrid modeling, and responsibility for the whole forecasting system. Prepare diagnosis and production judgment alongside modeling.
- [Harvest Forecast](https://www.source.ag/products/harvest-forecast) advertises forecasts up to eight weeks ahead and daily automated updates. Use weekly yield as the decision target and daily updates as the proposed serving cadence; distinguish target frequency, update frequency, sensor frequency, and manual sampling frequency.
- The [tomato measurement protocol](https://docs.source.ag/en/articles/630345-what-are-the-plant-measurement-requirements-for-tomato-harvest-forecast) specifies 40 stems twice weekly for cultivations of at least one hectare and 20 stems for cultivations below 8,000 m². The gap between these thresholds is not explained there. The [general FAQ](https://www.source.ag/faq) gives a less specific recommendation of around 20 plants per variety. Record this source discrepancy, not a universal sampling rule.
- Source's [bias troubleshooting guide](https://docs.source.ag/en/articles/630344-why-is-my-harvest-forecast-consistently-too-high-or-too-low) identifies fruit weight, sampling representativeness, waste, strategy, and unobserved crop stress as possible error sources. Its [color-scale explanation](https://docs.source.ag/en/articles/632831-how-does-color-scale-affect-the-timing-of-my-predicted-harvest) shows that picking strategy changes weekly timing.
- The [March 2026 model announcement](https://www.source.ag/news/source-advances-harvest-forecasting-for-tomatoes-with-next-generation-ai-model) describes combined horticultural science and learning across cultivations, with less manual input. It does not disclose enough to identify the internal model architecture. Published improvements are company claims, not independently reproduced benchmarks.
- [WUR's second Autonomous Greenhouse Challenge dataset](https://research.wur.nl/en/datasets/autonomous-greenhouse-challenge-second-edition-2019/) contains climate, control, crop, and harvest data from six cherry-tomato compartments during one six-month experiment. This is a useful realism reference, not six independent commercial sites or multiple seasons.

Unknowns to retain explicitly: Source's raw sensor cadence, label revision delays, historical weather/climate forecast vintages, exact simulator interface, internal uncertainty method, training volume, and deployment architecture.

## 3. Deliverables after approval

| Deliverable | Scope and use |
| --- | --- |
| `product-and-data-report.md` | Evidence-backed product and data briefing, with dated sources, conflicts, unknowns, and interview implications. Approximately 6–8 readable pages equivalent. |
| `data/README.md` and prepared data | Download instructions and provenance for the public dataset; a compact synthetic tomato panel, schema, generation assumptions, and a clear distinction between observed and hidden quantities. |
| `constrained-harvest-forecasting.ipynb` | Small CPU notebook that explains, runs, and compares the forecasting approaches and OOD failures. Target 30–40 cells, about six figures, and under ten minutes to execute. |
| `production-and-interview-guide.md` | Production architecture, monitoring/action table, a two-minute opening answer, whiteboard discussion, skeptical follow-ups, questions for Tobias, and a short first-month diagnostic plan. |

Keep reusable simulation, fitting, and evaluation behavior in a small module; the notebook explains the experiment. Use a run script and meaningful checks for constraints and time availability. Do not create a service or cloud deployment. The approved implementation will live in an isolated `source-ag-interview/` directory.

## 4. Product and data investigation

Prioritize current official product pages, Help Center instructions, job requirements, technical publications, APIs, and customer cases. Use older announcements to explain naming/history, not as current specifications. Search broadly for harvest-related evidence, with a bounded stop once the data and workflow questions are covered.

The report will map:

1. Who uses the forecast: growers, sales, cooperatives, logistics, and operations; decisions and costs of shortages versus surplus.
2. Crop coverage and terminology: current Harvest Forecast versus older Source Cultivate descriptions; tomato will be the notebook crop. Do not transfer pepper-specific development timings to tomato.
3. Forecast semantics: fresh harvest mass, gross versus saleable/net mass, fruit count/weight decomposition, area normalization, target period, horizon, forecast issue time, and revisions.
4. Input chain: weather forecast → simulated indoor climate conditional on controls → crop development → harvest readiness → picking strategy and waste → registered harvest.
5. Data dictionary: variable, units, entity, measurement method, cadence, observation time, arrival time, revision time, known-at-issue status, and evidence status.
6. Claim register: verified public statement, company-reported result, inference, proposed demo assumption, or unknown. No unsupported assertion that the demo reproduces Source's model.

Public starting points also include the [sensor API announcement](https://www.source.ag/news/source-ag-unveils-source-apis-for-direct-data-management-with-source-ag-solutions), [harvest API announcement](https://www.source.ag/news/source-ag-launches-the-source-api-for-harvest), and [Harvest House case](https://www.source.ag/news/harvest-house-partnership). Platform-available data is not necessarily consumed by every harvest model.

## 5. Dataset strategy: real anchor, controlled synthetic experiment

Recommendation: use public data to ground the variables and plausible regimes, and a transparent synthetic panel to control which observations are scarce and which future conditions are unseen. Do not pool real and synthetic test scores.

### Real-data candidate and feasibility gate

Use the [WUR/4TU dataset, DOI 10.4121/uuid:88d22c60-21b3-4ea8-90db-20249a5be2a7](https://doi.org/10.4121/uuid:88d22c60-21b3-4ea8-90db-20249a5be2a7). Indexed repository metadata describes a CC0 archive of approximately 8.4 MB. The landing-page description has been read; the archive has not been downloaded or inspected. Direct DOI retrieval was intermittent during planning.

After approval, inspect license, archive contents, units, timestamps, harvest frequency, cumulative versus incremental quantities, missingness, and crop-state variables. Determine whether input availability can be reconstructed. Allow one bounded inspection pass before choosing the implementation route. If vintaged weather forecasts are absent, do not use realized future weather as though it were available at issuance. Real-data work can remain an illustrative EDA and plausibility check; an honest synthetic study is preferable to an invalid historical forecast.

### Proposed synthetic schema

Start with approximately 12 fictional greenhouse units across two seasons, with a few tomato varieties and different control strategies. These counts are demonstration settings, not Source data estimates. Generate hourly climate but daily crop dynamics; record manual samples twice weekly from a small, spatially biased subset of plants.

| Table | Fields and temporal meaning |
| --- | --- |
| Greenhouse/crop metadata | Greenhouse, compartment, season, variety, area, planting date, stem density; all with validity periods. |
| Sensor observations | Air temperature, relative humidity, PAR, CO₂; sensor ID, observation and arrival time, quality flags, drift and missingness. |
| Plant samples | Sample/stem IDs, development-stage fruit counts, fruit weight, sampling area; sparse observations and recording delays. |
| Grower plan versions | Picking days/rounds, ripeness target, planned climate controls; issue time and future effective dates. |
| Climate forecast vintages | Issue time, target day, ensemble member, predicted indoor temperature/light; forecast uncertainty grows with horizon. |
| Harvest registrations | Daily fresh kg/m², waste, recording/revision time; aggregate into weekly forecast targets. |
| Hidden evaluation truth | True crop cohorts, sensor bias, shock labels, realized future plans; excluded from model features. |

The generator will represent fruit cohorts progressing toward maturity, temperature-dependent development, light-sensitive growth, stress/abortion, sampled fruit weight, and delayed harvesting. Distinguish true conditions, sensor readings, sparse crop observations, planned actions, and realized actions. Preserve units; do not confuse photon flux with daily light integral or fresh mass with dry matter.

Prevent a trivial simulator contest: the fitted process model will be simpler than the generator; test omitted stress effects, incorrect parameter priors, and changed harvest behavior. Run several fixed seeds. Document that these are sensitivity experiments, not evidence of commercial performance.

## 6. Forecast and modeling contract

At each issue date, predict fresh saleable harvest kg/m² for future weeks 1–8, with the third week as the headline business horizon. Produce daily paths internally where needed, then aggregate. All models receive only measurements that have arrived and forecast/plan versions issued by that date.

Separate two questions:

- Conditional forecast: what harvest follows if the currently stated climate and picking plan is followed?
- Operational forecast: what harvest is likely when weather, crop observations, and future grower actions are uncertain?

The demo will state which one is scored, and expose plan changes as a separate error source. A scenario forecast is not a causal estimate of what an intervention would achieve.

Compare four deliberately small approaches:

| Model | Purpose |
| --- | --- |
| Recent-harvest baseline | Last-week/recent-average harvest, to reveal whether complexity earns its place. |
| Pooled gradient boosting | A capable data-only predictor using lags, sampled crop state, forecast climate, and plans. No global monotonicity on temperature. |
| Simple process model | Development/fruit-inventory dynamics with bounded parameters and sparse-state updates. Tests how far structure alone goes. |
| Constrained hybrid | The same dynamics with a small regularized learned correction to development/survival/weight parameters and partial pooling across cultivations. |

Learn corrections within valid state transitions rather than adding an unconstrained yield residual. Explain alternatives—monotonic models, Gaussian processes, Bayesian state-space models, PINNs, and post-hoc projection—briefly, without implementing a model zoo. Physics-informed does not automatically mean a neural network or a PDE residual loss.

## 7. What the physical constraints actually mean

Use fruit-count bookkeeping as the minimal defensible structure:

`inventory_next = inventory_now + newly_set - harvested - aborted_or_discarded`

Track readiness separately so fruit cannot be harvested before it reaches the ready pool. Couple daily decisions through the shared inventory; independent weekly clipping is insufficient. Convert harvested fruit counts to mass with positive fruit weight and consistent units. Waste cannot exceed the harvested amount to which it applies.

Hard constraints: nonnegative states and harvest, valid survival/selection fractions, count accounting, eligible inventory limits, and daily-to-weekly aggregation. These are guarantees inside the specified model, conditional on its initial state and inputs.

Soft scientific assumptions: response curves, development thresholds, plausible parameter ranges, stress effects, and variety differences. Their numerical values require sources or explicit demo labels. Higher temperature is not globally beneficial; light saturation and stress may change response direction. Do not call empirical yield caps physical laws, or claim a fresh-mass/carbon balance without modeling the required water and carbon flows.

At least one figure will show a forecast that satisfies every hard constraint but remains wrong because the crop-state estimate or process assumptions are wrong. The claim to test is improved behavior under specified shifts, not guaranteed accuracy outside the training support.

## 8. Evaluation and uncertainty

Use forward-in-time splits with greenhouse/season holdouts, a separate calibration period, and a final untouched test. Prevent overlapping forecast labels from leaking across fitting/calibration boundaries; preprocessing and imputation are fitted on training data only. Backtests use the observations and forecast vintages available at each historical issue time.

Evaluate separate cases:

1. Ordinary future weeks within the familiar regime.
2. An unfamiliar climate regime, with a plausible simulated indoor-climate forecast rather than perfect future observations.
3. A new greenhouse/variety and limited initial measurements.
4. Sensor drift and nonrepresentative plant sampling.
5. An unannounced picking-plan change.
6. Structural mismatch such as unobserved heat damage or abortion.

Show one-variable extrapolation only within physically coherent trajectories; do not vary temperature while holding all incompatible greenhouse quantities fixed and call it a realistic counterfactual.

Report MAE and signed bias by horizon and regime, a proper interval score, empirical coverage and width, and counts/magnitudes of physical violations. Use kg/m² so low-harvest weeks remain meaningful; avoid MAPE near zero. Include an illustrative asymmetric commitment cost with clearly declared cost ratios, not invented Source economics. Count independent seasons/episodes as well as forecast rows.

For uncertainty, distinguish observation/state uncertainty, weather and climate uncertainty, parameter/process uncertainty, and action uncertainty. Generate feasible joint paths, aggregate each path first, and only then compute weekly quantiles. Marginal quantiles should not be presented as a jointly feasible scenario.

Use held-out residual calibration as a diagnostic baseline, with horizon-specific coverage. Do not promise nominal OOD coverage from conformal prediction: exchangeability or shift-specific assumptions matter; see [Barber et al., Conformal prediction beyond exchangeability](https://arxiv.org/abs/2202.13415). OOD distance is a warning signal, not a probability of failure. Discuss abstention, measurement requests, scenario ranges, and a trusted fallback when evidence is inadequate.

## 9. Notebook narrative

1. State the decision, target, horizon, and limits of the exercise.
2. Show available observations versus hidden crop state and future outcomes.
3. Plot what is rare or absent in training; distinguish covariate, process, and decision shifts.
4. Fit the baseline and data-only model.
5. Introduce the process model and constrained hybrid, with a compact accounting diagram.
6. Compare ordinary and OOD forecasts, constraint violations, and failure cases.
7. Assess intervals, business cost sensitivity, and the effect of bad sampling or changed plans.
8. Translate results into production actions and interview talking points.

Target figures: data/availability overview; training support; ordinary versus shift forecasts; coherent response/constraint check; interval coverage versus width; business/monitoring failure case. Tables should distinguish actual numerical results from assumptions and untested proposals. CPU-only, fixed seeds, modest dependencies, and clean-kernel execution.

## 10. Production design to explain, not deploy

Propose a daily batch forecast with event-triggered refresh for material new crop observations or revised plans. Describe immutable raw events, validation, as-of feature assembly, state estimation, forecast scenarios, constraint validation, a versioned forecast store/API, and delayed evaluation. The serving schedule is a proposed architecture, not a claim about Source's implementation.

Every forecast stores its issue time, target period, model and simulator versions, data snapshot, input forecast/plan versions, uncertainty, support/quality flags, and later corrections. Preserve originally issued forecasts when a grower changes the plan or actual harvest is corrected.

| Monitor | What to examine | Candidate action |
| --- | --- | --- |
| Availability and schema | Late events, missing streams, units, duplicate readings, timestamp alignment | Quarantine invalid inputs; mark stale forecasts. |
| Sensors | Stuck values, bias against reference sensors, impossible jumps, cross-sensor consistency | Request inspection; downweight/replace with explicit uncertainty. |
| Crop sampling | Coverage across greenhouse sections, sample turnover, weights/count disagreement | Improve sampling before adding model complexity. |
| Climate simulation | Predicted versus realized indoor climate, by issue horizon and simulator version | Diagnose upstream bias; refresh scenarios. |
| Model behavior | Physical violations, unfamiliar regimes, parameter boundaries, unstable revisions | Block invalid outputs; use a validated fallback or human review. |
| Delayed forecast quality | Bias/MAE, interval coverage and width, proper scores by horizon/site/regime | Recalibrate or retrain with mature labels; retain old versions. |
| Management changes | Original versus revised plan and realized harvest | Explain revisions; do not silently relabel past forecasts. |
| Decision use | Grower overrides, forecast adoption, shortfall/oversupply cost | Review with product and growers; distinguish predictive from causal evidence. |

Add label maturity and revision policies; retraining versus recalibration versus state-update distinctions; shadow deployment; prospective review through unusual weeks; rollback and ownership; versioned upstream-simulator changes; and the inability of a short deployment window to prove seasonal generalization. Alarm thresholds must come from calibration data and decision costs, not arbitrary universal drift cutoffs.

## 11. Interview preparation and completion criteria

Prepare a two-minute answer built around: clarify the failure → audit data/target → choose defensible structure → test shifts → quantify limitations → operate responsibly. Prepare concise replies to: why not just clip; why not a PINN; what happens when the simulator is wrong; how to calibrate with little data; which measurement would you buy next; how to detect leakage; and what would make you refuse a confident prediction.

Questions for Tobias will cover which OOD regimes dominate, simulator outputs and version history, observed versus planned actions, first-harvest labels, sample selection, uncertainty users need, and the costs of over/under-delivery. The user's account of the upcoming conversation takes precedence over the website's generic interview-stage description. Include a project-story scaffold without inventing the candidate's career achievements.

Implementation sequence after approval: evidence report → dataset feasibility/schema → fixed split/scenario contract → simulator and baselines → constrained model and uncertainty → notebook/figures → production/rehearsal guide → clean execution and evidence review.

Acceptance criteria: source-backed product claims; clear real/synthetic separation; explicit as-of data; no hidden-truth model inputs; meaningful conservation and unit checks; untouched OOD tests; uncertainty assessed honestly; at least one counterexample to the hybrid model; results reproduced from a clean kernel; and a short explanation the candidate can deliver without reading code.

## 12. Independent critique requested

Ask a separate critic to challenge scientific correctness, identifiability, dataset feasibility, false confidence from simulation, leakage, uncertainty, scope, and interview usefulness. Preserve findings, assess each recommendation, and update this plan before requesting approval. Critique may strengthen or narrow the plan but may not authorize implementation.
