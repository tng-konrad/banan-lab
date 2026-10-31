# Implementation amendments before held-out inference

No held-out query has been scored. The original full-study corpus and cohorts remain available; v5.3 additionally freezes a smaller, exploratory profile as described below.

## v5.1 development evidence

Live implementation exposed a mismatch in literal-delimiter escaping during the Unicode token-accounting probe. The gate failed and its record is retained. After correction, both adapters matched complete rendered token counts, including near the common input limit. Native oversized requests were rejected and all complete corpus singletons fitted the common policy on design histories.

The first development demo/pilot also exposed a client bug: HTTP 500 context-limit responses were classified as input-limit failures but still retried by a status-only condition. The contract forbids this retry. The records are preserved under `results/development/` and excluded from reported comparisons. They are not silently reclassified as valid results. The source server responses remain available for diagnosis.

## v5.2 corrected execution

Only transient infrastructure errors can receive the one identical-payload retry. Context-size, schema, truncation, and other protocol failures cannot. The notebook validates this behavior and reruns the training demo and pilot with the corrected adapter.

The resource gate now has a fail-fast implementation. It cannot turn a failure into a pass or change the held-out sample. For each observed pilot timing `t` from method `m` and domain `d`, its contribution to the minimum possible frozen forecast is `N_eval[d] * t / n_pilot[d]`. Add those contributions and the predeclared 25% reserve, while assigning zero time to every unobserved timing and zero scheduling overhead. If even this lower bound exceeds two hours, the resource gate cannot pass. The partial run is retained as incomplete, and the bound is explicitly labeled a lower bound rather than a point forecast.

Likewise, after a completed pilot C/D pair containing an invalid query, the all-valid pilot requirement is already impossible, so the pilot can stop. All ten pilot queries are still required for a passing gate. Neither stopping decision consults retrieval quality. Extra adapter/variability experiments are skipped after an incomplete pilot instead of continuing to consume inference time.

A failed or incomplete pilot leaves the 259-query four-option study unfulfilled. Missing comparative plots are declared unavailable; diagnostic charts show observed and unattempted counts. The worked training example remains separate and cannot supply missing full-corpus measurements.

## v5.3 local two-hour profile

The standalone notebook defaults to offline replay and selects `STUDY_PROFILE = 'mini_2h'` for future live runs. It embeds an additional fixed corpus of 32 pages (eight per domain), all 329 structural passages on those pages, eight training pilot queries (two per domain), and 24 validation queries from 24 distinct dialogues (six per domain). Pages are selected by SHA256 before queries. Hash-selected dialogues/queries must have annotated evidence in that fixed corpus. Original full-study evaluation dialogues and all fifteen explored validation dialogues are excluded. All four methods receive the same complete search corpus; annotations determine eligibility only and never enter requests.

This changes task difficulty. Mini accuracy and intervals are exploratory and cannot confirm full-corpus results, the paper's benchmark, or the original two-percentage-point noninferiority claim. The original 413-page/259-query profile is retained. There is no automatic dataset reduction based on held-out outcomes.

All four methods and the 3-page navigation/16-candidate/12-prefix limits remain. Native context churn diagnostics alternated three 16-candidate states and a return to the first. Sixteen questions produced intermittent server context rejections; asking the first four questions passed those four diagnostic requests. v5.3 asks four questions per request for both adapters, retaining the full shared candidate state and stable aliases, merging all question groups before ranking. This does not establish Jev-style shared-forward execution or solve the server failure conclusively. The new live gate must score every question group over alternating states. No passages are truncated and no global cache setting is changed.

A single live workflow deadline now covers gates, warmup, pilot, diagnostics and evaluation; downloads and offline plotting are outside it. The evaluation forecast uses domain-weighted pilot timings, 25% reserve, and uncovered scheduling cost; it must fit the time remaining after checks and diagnostics. Pilot failure leaves evaluation unrun. The 60–90-minute target is provisional; no v5.3 live benchmark has been launched here. Local replay can load a result JSON explicitly, without rebuilding the notebook. Historical embedded requests are reconstructed under their recorded v5.2 configuration so that new settings cannot relabel old measurements.

## v5.4 external data and requested local execution

The user explicitly replaced the embedded-corpus requirement with a separate download and asked to execute the restricted notebook here. The notebook now contains only frozen selection metadata/checksums and inline implementation. The corpus archive and tokenizer are streamed to temporary files, checked against pinned SHA256 hashes, atomically cached, and verified before reuse. Annotation cohorts and normalized passages are reconstructed from the archive and checked against frozen object hashes; study membership is unchanged. Full request/response traces and captured results are read from external result JSON files rather than a hidden notebook bundle.

The existing verified downloads are reused for inference. An independent cold-download check fetches fresh copies and checks cache reuse and rejection of corrupted data. Historical v5.2 full-corpus evidence is preserved under `results/archive/v5.2-full-corpus/`. The actual notebook executes in a fresh Jupyter kernel with `MODE = 'evaluate'`, the mini profile and the existing two-hour workflow budget. The delivered source is reset to replay after execution; metadata records the mode responsible for displayed outputs. Failed gates or incomplete runs remain visible and cannot be relabeled as a completed comparison.

### Measured v5.4 execution

The actual notebook completed its code execution in 1,707.76 seconds (28.46 minutes), with zero notebook exceptions. The native compatibility/churn gate passed. Pilot `pilot-20261030T144327Z` completed eleven hierarchy method-queries (five C and six D), all with valid rankings, and then stopped under the existing resource rule. Its lower bound for the 24-query evaluation was 5,700.67 seconds (95.01 minutes), compared with 5,509.87 seconds (91.83 minutes) remaining. The lower bound includes 25% reserve and assigns zero duration to all unobserved timings. Evaluation did not start; A/B were not reached in the pilot. The run supports execution stability and a negative feasibility finding, not a completed retrieval comparison. Earlier optimistic runtime estimates were not confirmed. Historical full-corpus examples have been moved out of active result filenames to preserve matching live/replay cohorts.

## v5.5 clock-integrity correction after the genuine run

UTC timestamps span approximately 103.6 minutes from gate start to notebook finish, while the active timer reports 28.46 minutes. Six large request-to-request gaps account for approximately 75 minutes not counted by the active timer. Suspension and wall-clock changes cannot be distinguished from these records. This invalidates any wall-clock efficiency interpretation; it does not invalidate parsed ranking outputs. No raw recorded timings, rankings, inputs or model responses were rewritten.

Future workflow and evaluation deadlines now enforce the stricter of monotonic and epoch-wall remaining time, preventing pauses or backward clock changes from extending the allowed budget. Analysis compares run timestamp span with the recorded timer and requires timing integrity for an efficiency claim. Historical latency plots identify active-timer measurements when that audit fails. The clock fix is tested offline, including an expired wall deadline with an unexpired monotonic deadline; no new live model benchmark was run after it. The genuine model evidence retains its v5.4 configuration and incomplete resource-gate status.

## v5.6 user-launched three-hour / twelve-query profile

The user requested an immediately runnable local notebook with a three-hour budget, the same 32-page/329-passage corpus, and twelve evaluation queries. The default is now `MODE = 'evaluate'`, `STUDY_PROFILE = 'mini_3h'`, `RUN_BUDGET_HOURS = 3.0`. No model inference was launched in Codex for this revision. The delivered notebook has clean outputs; a snapshot of the earlier executed notebook is preserved separately.

The twelve queries are the first three per domain in the original frozen validation-query order. No model outcomes were used to select or drop a query. The same eight training pilot queries remain. All four methods run on the same twelve evaluation queries (48 method-query observations). Domain-weighted runtime projections now use three evaluation queries and two pilot queries per domain. New profile/input/label objects have frozen hashes, while the original 24-query objects remain unchanged for historical replay. Labels, analysis cohorts, figure counts and captions use each run's recorded profile. The three-hour budget includes checks, pilot, warmup, diagnostics and evaluation and retains strict wall/monotonic enforcement and the timing-integrity gate. Offline checks cover profile counts, corpus membership, balanced independent dialogues, projection arithmetic, and historical trace replay. Feasibility remains conditional on the fresh local pilot.
