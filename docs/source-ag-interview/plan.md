# Source.ag interview preparation — revised plan

Date: 1 November 2026. Status: **approved by the user; implementation delivered in [source-ag-interview](../../source-ag-interview/README.md)**.

This replaces [the initial plan](plan-v1.md). An independent critic reviewed that draft; [the critique and decisions](plan-critique.md) explain the changes. The user subsequently approved implementation. The sections below preserve the approved scope; execution evidence is linked from the delivered package.

## 1. The interview argument

Prepare a defensible answer to: **How can we forecast greenhouse harvest beyond the conditions represented in limited training data, while respecting physical constraints and being honest about uncertainty?**

The working position is that physical constraints restrict what a model can predict. Accurate extrapolation additionally requires mechanisms that remain valid, a credible estimate of the current crop, and usable assumptions about future climate and grower actions. A physically feasible prediction can still be confidently wrong.

Start by clarifying what “out of distribution” means in the actual failure: unfamiliar input combinations, an incorrect crop-state estimate, changed biology, a changed picking decision, or an upstream climate-simulator error. A predicted yield outside the historical target range is not, by itself, a physical violation. Each failure suggests a different intervention; model complexity is only one option.

Prepare for roughly a 45–60 minute technical discussion, with a 15–20 minute optional walkthrough. These are preparation assumptions, not a confirmed interview format. The user's account of Tobias's intended discussion takes precedence over the generic hiring stages on the website.

## 2. What preliminary research establishes

| Finding | Consequence for preparation | Evidence |
| --- | --- | --- |
| The role emphasizes sparse samples, unusual weeks, delayed feedback, changing grower behavior, hybrid models, and end-to-end responsibility. | Prepare diagnosis, evaluation, and production decisions alongside the model. | [Current job description](https://careers.source.ag/o/staff-data-scientist-forecasting) |
| Harvest Forecast advertises up to eight weeks ahead and daily automated updates. | Separate forecast horizon, target aggregation, refresh cadence, and measurement cadence. | [Harvest Forecast](https://www.source.ag/products/harvest-forecast) |
| The tomato-specific protocol calls for 40 stems twice weekly at ≥1 hectare and 20 below 8,000 m²; the general FAQ says around 20 plants per variety. | Prefer specific dated instructions; preserve the unexplained 8,000–10,000 m² gap and plant/stem distinction. | [Tomato protocol](https://docs.source.ag/en/articles/630345-what-are-the-plant-measurement-requirements-for-tomato-harvest-forecast), [general FAQ](https://www.source.ag/faq) |
| Fruit weight, representative sampling, waste, crop stress, and harvest strategy can affect forecast error. | Audit the observation and decision processes before assuming a modeling failure. | [Troubleshooting guide](https://docs.source.ag/en/articles/630344-why-is-my-harvest-forecast-consistently-too-high-or-too-low) |
| Picking ripeness changes the timing of harvest. | Separate biological readiness from registered harvest. | [Color-scale explanation](https://docs.source.ag/en/articles/632831-how-does-color-scale-affect-the-timing-of-my-predicted-harvest) |
| The March 2026 announcement describes horticultural science combined with learning across cultivations. | A hybrid is product-relevant, but the internal architecture remains unknown. | [Model announcement](https://www.source.ag/news/source-advances-harvest-forecasting-for-tomatoes-with-next-generation-ai-model) |

These are planning anchors, not the completed product report. Company performance claims will be identified as such, including missing metric definitions or evaluation details. No claim will be made that the demo reconstructs Source's proprietary model.

## 3. Deliverables after approval

| Deliverable | Purpose and completion standard |
| --- | --- |
| `source-ag-interview/product-and-data-report.md` | A one-page briefing followed by a sourced product/data analysis: business decisions, crop coverage, forecast semantics, input chain, cadence, public evidence, conflicts, and unknowns. |
| `source-ag-interview/data/README.md` and compact dataset | A reproducible synthetic dataset plus public-data download/provenance instructions and a small realism appendix. All synthetic parameters and limitations explicit. |
| `source-ag-interview/constrained-harvest-forecasting.ipynb` | Approximately 30–40 cells and six figures, CPU-only, target runtime below ten minutes; a three-week forecast experiment with an honest OOD test. |
| `source-ag-interview/production-and-interview-guide.md` | Architecture and monitoring decisions, a two-minute answer, a whiteboard explanation, skeptical follow-ups, questions for Tobias, and a practical first-month investigation. |

Reusable simulation, fitting, and evaluation will live in one small module, with a run script and focused checks. The notebook explains and investigates. No service, cloud deployment, full greenhouse simulator, or production implementation is included. Existing unrelated repository work will be preserved.

## 4. Product and data report plan

Research current official product pages, Help Center documentation, APIs, customer cases, the job description, and relevant primary scientific literature. Use older Source Cultivate/Track names only with dates and context. Broader company coverage will stay relevant to harvest prediction rather than become a funding-history report.

Map the workflow: **observations and forecasts → indoor-climate scenarios conditional on controls → crop development and fruit readiness → picking and waste → recorded harvest → commercial decisions**. Confirm each arrow where public documentation permits; label the rest as inference or proposed architecture.

Produce a data dictionary with variable, units, entity, collection method, measurement cadence, arrival/revision timing, and whether it is known at issuance. Cover climate computers/sensors; sparse plant measurements; cultivar, area and planting metadata; harvest registrations; weather and climate simulations; and grower plans. Distinguish platform availability from confirmed use by the harvest model.

Starting sources include the [sensor API announcement](https://www.source.ag/news/source-ag-unveils-source-apis-for-direct-data-management-with-source-ag-solutions), [harvest API announcement](https://www.source.ag/news/source-ag-launches-the-source-api-for-harvest), and [Harvest House case](https://www.source.ag/news/harvest-house-partnership). Use [Vanthoor et al.'s tomato yield model](https://research.wur.nl/en/publications/a-methodology-for-model-based-greenhouse-design-part-2-descriptio/) and [GreenLight](https://research.wur.nl/en/publications/greenlight-an-open-source-model-for-greenhouses-with-supplemental/) as scientific reference candidates, not dependencies to implement wholesale.

Record a source date, retrieval date, and status for every material claim: public fact, company-reported result, inference, demo assumption, or unknown. Unknowns include exact sensor cadence, sample/label revision delays, forecast-vintage availability, simulator outputs, internal uncertainty methods, and deployment details. Do not import pepper-specific biological timing into the tomato demo; the [pepper documentation](https://docs.source.ag/en/articles/629922-how-fruit-set-patterns-affect-your-pepper-harvest-forecast) will be used only to explain why crops and horizons differ.

## 5. Dataset: a real reference and a deliberately limited simulation

### Public reference

The recommended candidate is [WUR's second Autonomous Greenhouse Challenge dataset](https://research.wur.nl/en/datasets/autonomous-greenhouse-challenge-second-edition-2019/), [DOI/download landing page](https://doi.org/10.4121/uuid:88d22c60-21b3-4ea8-90db-20249a5be2a7). It describes indoor/outdoor climate, controls, crop measurements, and harvest from six compartments in one six-month cherry-tomato experiment. Indexed 4TU metadata lists a CC0 archive of approximately 8.4 MB; archive access, license contents, and schema have not yet been verified. Direct DOI retrieval was intermittent during planning.

After approval, limit archive inspection and cleaning to approximately one hour: check license, files, units, time coverage, missingness, harvest recording, and whether quantities are cumulative. Produce a small schema/EDA appendix and plausible input ranges. If access or usability fails, preserve that status and use cited documentation for the synthetic assumptions. A real-data forecasting benchmark is outside the core scope.

Six compartments in one experiment do not demonstrate transfer across commercial sites. Controls are decisions rather than randomized treatments. Without historical weather/climate forecast vintages, realized future conditions cannot serve as operational forecast inputs.

### Synthetic core

Use **one tomato variety, six fictional greenhouse units, and two 336-day seasons**. Generate daily climate summaries, crop dynamics, and harvest; twice-weekly observations of a small stem sample; and weekly forecast origins. This is an intentionally small panel. Dense sensor streams are covered in the report and production discussion, not an hourly simulation engine.

Tables: greenhouse/crop metadata; daily sensor observations; sparse plant samples with sample IDs; issue-dated indoor-climate scenarios; versioned fixed picking plans; gross/net harvest registrations; and a separate hidden-truth table used only by the evaluator. Fields include units, entity keys, `observed_at`, `available_at`, `issued_at`, target/effective dates, and provenance where applicable.

The generator represents age/maturity cohorts, positive fruit weight, temperature-dependent development, bounded survival, light effects, and harvest removal. Manual observation noise, sensor noise, and label delays are present but are not each a separate benchmark. Use a declared 48-hour label-finalization delay for the demo; this is not a Source fact.

Forecast climate must be generated from information available at issuance and independent future innovations. Do not create apparently realistic forecasts by taking realized future weather and adding noise. The core OOD event is a sustained changed climate regime visible in recent observations, not a requirement to predict an unknowable heatwave three weeks before any signal exists.

## 6. Exact forecast and split contract

**Scored target:** registered saleable fresh kg/m² in the third future calendar week. At Monday 00:00 issue time `t`, target days run from `t + 14 days` through `t + 20 days`, inclusive. Inputs must have arrived strictly before issue time. Predict gross harvested mass and subtract post-harvest waste under the declared observation model; do not mix pre-harvest abortion with post-harvest waste.

**Core interpretation:** conditional on the recorded picking plan, which remains unchanged in scored cases, while future climate and crop state are uncertain. Score against realized net harvest. Changed-plan forecasting is discussed separately; the demo does not identify a causal decomposition of forecast error or optimize cultivation policy.

Daily updates and one-to-eight-week coverage are explained in the production/product discussion. The notebook scores one horizon so the methodological argument remains inspectable. The job description's three-week commercial example motivates this choice.

Dates below are fictional experimental dates, not a reconstruction of WUR or Source records. Season 1 runs 2024-01-01–2024-12-01; season 2 runs 2025-01-06–2025-12-07. The same six units appear in both; unseen-site generalization is not claimed.

| Partition | Weekly Monday issue dates | Origins per unit | Last target week | Latest label availability | Use |
| --- | --- | ---: | --- | --- | --- |
| Warm-up | First eight weeks of each season | — | — | — | Initialize state from observations; no scored forecasts. |
| Training | 2024-02-26–2024-06-24 | 18 | 2024-07-08–2024-07-14 | 2024-07-17 | Fit using mature labels only. |
| Validation | 2024-07-22–2024-08-19 | 5 | 2024-09-02–2024-09-08 | 2024-09-11 | Select a small fixed set of regularization choices. |
| Interval diagnostics | 2024-09-16–2024-11-04 | 8 | 2024-11-18–2024-11-24 | 2024-11-27 | Assess interval behavior; no model-selection reuse. |
| Final test | 2025-03-03–2025-11-10 | 37 | 2025-11-24–2025-11-30 | 2025-12-03 | Untouched ordinary, climate-shift, and mismatch branches. |

After validation, optionally refit the selected specification on training/validation labels available before 2024-09-16, then freeze parameters. Assimilate new crop observations causally at each subsequent origin; this is a state update, not parameter refitting. Early test-season measurements may initialize the new crop, but future test outcomes never initialize a past forecast.

Freeze seeds `11, 23, 37, 51, 73`, split membership, scenario settings, and feature definitions before scoring. Each seed represents a separate synthetic world; sites sharing climate are dependent. Report five worlds, one test season per world, weather-event counts, weekly origins, and greenhouse rows separately. The eight diagnostic weeks do not support a strong empirical calibration claim.

## 7. Four models and an identifiability budget

1. **Recent-harvest baseline:** the latest fully available mature calendar week's net kg/m². With Monday issuance and the declared 48-hour finalization delay, the immediately preceding week is not yet mature; use the week before it. All model harvest-lag features follow the same availability rule.
2. **Pooled gradient boosting:** lagged harvest, sparse observed stage counts/weight, crop age, accumulated development summaries, and forecast climate/plan features.
3. **Process-only model:** a small cohort-readiness and harvest-inventory model, with fixed biological response shapes and a global development-rate parameter fitted from admissible training observations.
4. **Constrained hybrid:** the same process model plus one regularized development-rate correction per greenhouse, shrunk toward the pooled rate. No simultaneous learning of abortion, fruit weight, sampling bias, and harvest behavior.

The process-only model tests whether the learned local correction adds value. The boosted model tests this particular data-only baseline; results cannot establish superiority over all statistical or machine-learning methods. Give it the same admissible information and relevant derived development summaries. Post-hoc nonnegative clipping can be illustrated without training another model, but is not a substitute for dynamic accounting.

| Quantity | What informs it | Core treatment and limitation |
| --- | --- | --- |
| Initial cohort state | Sparse stage-count samples, crop age, sample/stem density | Causal constrained state estimate with eight-week warm-up; no hidden-state initialization. |
| Development rate | Training sample transitions and mature harvest observations | One pooled rate; hybrid adds a strongly regularized unit offset. State/parameter confounding remains possible. |
| Fruit weight | Latest available sampled weights | Carry forward with declared uncertainty; do not separately learn a hidden weight process. |
| Survival/abortion | Limited crop observations and stated assumptions | Fixed response in the core; sensitivity/mismatch case exposes incorrect assumptions. |
| Picking and waste | Recorded plan and registrations | Fixed plan; declared waste handling, not inferred grower policy. |

Fit against the observation likelihood/loss for recorded samples and mature harvest, not the generator's latent transition rates. Fix nuisance parameters to cited or explicitly illustrative assumptions. If development rate is weakly identified, shrink more strongly or retain the pooled process model; do not conceal optimizer instability. Observation assimilation is a correction to an estimated state, not biological creation of fruit.

## 8. Constraint and fairness contract

Use count accounting, with pre-harvest losses distinguished from harvested fruit:

`N_next = N_now + newly_set - gross_fruit_removed_at_harvest - preharvest_losses`

Track the ready pool so harvested fruit cannot exceed eligible inventory. Convert counts to fresh mass using positive weights; net mass is gross mass minus bounded post-harvest waste. Couple the full trajectory before aggregating weeks. All states and rates have valid support.

Hard model invariants include nonnegativity, valid fractions, count accounting, eligibility, and aggregation. Development curves, plausible parameter bounds, and light/temperature responses are scientific assumptions whose validity has a domain. Do not impose globally increasing yield with temperature, call an empirical yield cap a physical law, or claim carbon/energy conservation from fruit counting alone.

Separate three diagnostics:

- **Internal invariants:** meaningful for process models that expose a state; record numeric tolerance. A mass-only baseline receives “not applicable,” not an invented conservation score.
- **Shared observable output checks:** nonnegative predicted net harvest, valid units, ordered intervals, and aggregation consistency where outputs exist.
- **Synthetic oracle diagnostics:** hidden true inventory can explain forecast error after scoring, under a declared weight/timing convention. It is never a serving-time constraint, feature, or tuning signal, and an error against noisy truth is not automatically a broken physical law.

The generator and process model deliberately share cohort bookkeeping and some response assumptions. List these shared assumptions prominently. Withhold latent state, parameters, realized future climate, and test-event labels from every model. Hyperparameters may not be selected on OOD performance.

Test three paired branches, with parameters frozen in advance:

| Branch | Question | Interpretation |
| --- | --- | --- |
| Ordinary future season | Does the model forecast useful quantities in familiar conditions? | Establish a baseline without selecting the winner on test scores. |
| Coherent climate shift | Does the structural model remain useful outside training climate support? | Use a sustained shifted regime, jointly plausible climate features, and imperfect issue-time climate scenarios. |
| Process mismatch | What if a survival/stress assumption fails? | Add an unmodeled crop-loss mechanism; show how a physically valid forecast can be wrong. Do not tune the shock until a preferred ranking appears. |

Freeze event onset/duration and scientific or illustrative parameter ranges after the public-data plausibility check and before model fitting. Report score changes across seeds, not a guaranteed hybrid win. The supported conclusion is conditional on the specified synthetic worlds. New varieties, cold starts, sensor drift, biased sampling, and changed picking plans remain interview/production scenarios, not five additional implemented benchmarks.

## 9. Uncertainty, metrics, and notebook story

Use the term **prediction interval** for future harvest; uncertainty in an estimated mean is a different object. For the core, propagate a modest ensemble of issue-time climate paths and one declared crop-state uncertainty distribution through the process models. Aggregate each feasible path into the target week before computing quantiles. Label the output **model-based scenario intervals**: assumed scenario distributions do not establish calibration.

Report MAE, signed bias, interval coverage and width, and a proper interval score, separated by branch. Point forecasts are compared across all four models; the interval analysis is a scoped diagnostic of the process/hybrid scenario distributions, not a probabilistic leaderboard against models without comparable distributions. Show sample/episode counts and treat the eight held-out diagnostic weeks as descriptive. No nominal OOD coverage guarantee or certified safety claim.

Discuss residual recalibration and time-aware conformal approaches in the guide; do not present a small or dependent calibration set as evidence of guaranteed coverage. [Barber et al.](https://arxiv.org/abs/2202.13415) is a primary reference for why exchangeability and drift assumptions matter. Discuss observation, parameter, process, and future-action uncertainty even where the small demo does not estimate them.

Add one small asymmetric under/over-commitment cost sensitivity table with explicitly illustrative ratios. Avoid MAPE near zero and invented Source economics. Distinguish OOD detection from calibrated failure probabilities. A conservative response may be wider scenarios, another measurement, a validated fallback, or review—not an arbitrary confident extrapolation.

The six planned notebook figures are: (1) observations versus latent crop state; (2) training support and shifted regime; (3) ordinary/climate-shift forecast comparison; (4) a feasible but wrong mismatch forecast; (5) interval coverage and width; (6) a compact forecast-to-monitoring decision example. A short comparison table shows accuracy and applicable constraint checks. Keep most technical machinery in the small module and the argument visible in the notebook.

## 10. Production and operational discussion

Describe a proposed daily batch forecast, with refreshes after material crop measurements or plan revisions:

`immutable input events → validation → as-of features/state update → climate/crop scenarios → output checks → versioned forecast store/API → delayed evaluation`

Every issued forecast retains its target period, data/feature snapshot, crop and climate-model versions, weather/plan vintages, quality/support flags, and uncertainty assumptions. Store revisions as new forecasts. Preserve the original commitment-time forecast when actuals or plans change.

| Monitor | Relevant failure | Action/owner to discuss |
| --- | --- | --- |
| Freshness, schema, units, timestamps and sample counts | Missing/late data, kg versus grams, stale sample, DST/aggregation mistakes | Data engineering validation; quarantine or explicitly stale forecast. |
| Sensor drift and consistency | Stuck sensor, local bias, disagreement with reference sensors | Operations inspection; downweight only with an uncertainty/fallback policy. |
| Sampling representativeness | A few vigorous plants stand in for the greenhouse; correlated stems give false sample size | Crop-science review and revised sampling, not just model retraining. |
| Upstream climate forecasts | Simulator/version bias and errors by horizon | Compare issued versus realized climate; engineering and model-owner investigation. |
| Physical and numerical checks | Invalid states, negative outputs, unstable parameter boundaries | Block invalid output; validated fallback or human review. |
| Support and state uncertainty | Novel climate/crop regime or uncertain inventory | Extra measurements/scenarios; reduce reliance on unvalidated extrapolation. |
| Mature forecast errors and intervals | Bias, poor coverage, excess width, peak/end-season failures | Data-science review by horizon/site/season; recalibrate or retrain only with mature labels. |
| Plan changes, overrides and forecast revisions | Grower acts differently or forecast affects behavior | Preserve versions; product/grower review, avoid causal claims from residual changes. |
| Decision outcomes and serving reliability | Costly shortfalls, stale delivery, degraded batch job | Product and engineering alerting with named escalation and rollback responsibility. |

For each monitor, the guide will specify granularity, immediate versus delayed availability, candidate response, and threshold-setting evidence. Small-sample coverage estimates need counts and uncertainty; pooled coverage can conceal poor behavior for a specific grower or regime.

Cover leakage from finalized labels, full-series interpolation, retrospectively edited plans, future climate, and preprocessing on later periods. Define label maturity/revision handling, distinguish state update/recalibration/retraining, and explain simulator version migrations. Recommend shadow forecasts and prospective review through meaningful crop events. A short successful deployment cannot establish seasonal robustness. Retraining is not an automatic response to every drift alert, and the process model is not automatically a safe fallback when its assumptions fail.

## 11. Interview rehearsal and staff-level judgment

Prepare a two-minute opening answer and a longer whiteboard version with this sequence: clarify the failure and decision → inspect state/data/plan quality → encode defensible structure → test familiar and shifted conditions → quantify what remains uncertain → define operating actions.

Prepare answers to: why not clip; why not a PINN; how to use a flawed simulator; what partial pooling can and cannot identify; why future covariates leak; how to validate rare events; what an interval means; and what would trigger human review. Include one measurement-value example comparing another routine sensor reading with a representative crop sample or a fruit-weight check, without pretending a numerical value-of-information result has been computed.

Questions for Tobias: which OOD failures matter most; how plant samples are chosen; what the climate simulator emits; whether historical input forecasts/plans are archived; when harvest labels settle; which grower actions change after issuance; what commercial error costs and uncertainty users actually need; and which fix the team currently suspects lies outside the model.

Prepare a project-story scaffold about diagnosing a forecasting failure, changing direction, shipping a fix, and maintaining it. Use only the candidate's verified experience; do not invent achievements. A first-month plan will prioritize reconstructing issued forecasts, identifying high-cost failure weeks, and choosing the first data/model/product fix with crop scientists and engineers.

## 12. Execution order and acceptance gates

After approval: (1) report and bounded data inspection; (2) freeze schema, scenario parameters, and the dated split manifest; (3) simulator, causal observations, and four models; (4) independent checks and evaluation; (5) executed notebook and visually inspected figures; (6) production/rehearsal guide and final consistency review.

Acceptance requires source-backed product statements; transparent real/synthetic separation; no latent-truth or future-vintage leakage; meaningful unit/accounting checks; the declared split counts and label cutoffs; deterministic runs across fixed seeds; honest interval labels; visible process-mismatch limitations; and clean-kernel reproducibility. No promised model ranking. Check the combined package can support a clear short explanation without reading source code.

Optional extensions—full eight-week evaluation, unseen-site adaptation, learned survival, more baselines, full conformal calibration, and a real-data forecast benchmark—are explicitly excluded from the approved core unless the user later expands scope.

**Approved and implemented.** See the [delivered package](../../source-ag-interview/README.md) and its validation records.
