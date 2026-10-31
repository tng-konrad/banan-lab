# Companion post plan: independent critique and revision

Reviewed artifacts: `post-plan.md`, `post-guide.md`, the executed notebook, `graphs/manifest.json`, the captured evaluation/pilot traces and the derived analysis. Critic: independent `post_plan_critic` agent, read-only. The reviewer confirmed that the plan's reported recall, latency, uncertainty and failure counts match the captured run, and that all twelve placeholders correspond to notebook visuals.

## Priority findings and disposition

1. **C/D did not causally isolate decision-model architecture. Accepted.** The controller rules and starting page corpus match, but the model selects branches. C made 275 successful scored requests and D 235, without retries; the resulting passage pools differ. The revised plan and notebook call this an end-to-end replacement of a model-plus-adapter implementation. Different work is a legitimate effect of navigation, not evidence that the comparison is unfair. It also means the time ratio is not a per-decision serving-speed estimate. B/C likewise changes discovery pools and compute; A/B adds a combined judging/rescoring pipeline.

2. **Qwen's tied scores were missing from the narrative. Accepted, with denominator correction.** Independently recomputed from raw requests: among groups with at least two scores, D had 177/234 all-equal groups (75.6%) and 231/234 groups with some tie. C had zero all-equal or tied groups out of 273. Single-question groups are excluded because equality is automatic. Requests are not independent evaluation examples. The frozen ranking sorts equal scores by candidate ID. This limits interpretation of the particular numeric-score adapter, but does not establish that its scores were wrong: many groups can contain genuinely irrelevant candidates. It neither erases D's observed result nor explains why it was faster. A new JSON audit exposes these counts without adding another figure or retuning the held-out run.

3. **Two accessibility descriptions were stale. Accepted.** The mini C/D plot compares quality difference with zero, not the original full-design minus-two-point margin. The latency plot has observations for all four methods, so its alt text must not say the other methods were unattempted. Both descriptions now depend on the actual scope and completion data.

4. **Define the decision model before discussing its plumbing. Accepted.** Added a plain-language account of native bounded choice scores versus generated structured scores. Distinguished decision training from the separate shared-forward serving mechanism used in the paper. “No fine-tuning in this demo” does not mean Nimble had no prior decision training.

5. **Give runtime an execution context. Accepted within the available evidence.** Captured metadata verifies macOS arm64, Ollama 0.40.13, explicit roughly 9B Q8 models and residency checks. The run receipt does not record CPU model or RAM capacity. The plan identifies that gap and requires verification before a hardware-specific publication claim; it does not silently promote older planning inventory into fresh benchmark evidence. “Laptop” is not a feasibility guarantee for every machine.

6. **Say what the bootstrap interval covers. Accepted.** The paired interval resamples dialogues/queries within one captured run. It does not estimate repeated-execution, hardware or serving-configuration variability. It remains useful for the observed query mixture, with substantial quality uncertainty.

7. **Twelve visuals can overwhelm a short post. Accepted as editorial prioritization, not deletion.** All filenames, section headings and placeholders remain. Graphs 2/5/8/10 carry the central narrative; the other eight are supporting insets. Added section word budgets and shorter readings, moved filename bookkeeping out of article prose, and reduced the planned timer/259-query-design history to a compact reproducibility note.

8. **Show a tiny implementation sketch. Accepted.** Added the conversation → page scores → three pages → passage scores → ten results sequence without adding a figure, method or model run.

## Additional analysis by the primary agent

The strict-ID result is not the only available evidence. Exact normalized-text matching gives A 4/12, B 7/12, C 6/12 and D 6/12 at rank one, versus strict-ID counts of 4/12, 7/12, 5/12 and 5/12. Both hierarchies have one top-ranked text match with a different structural ID. The plan now reports that sensitivity briefly and qualifies the failure figure as structural-ID misses, rather than declaring that every mismatch semantically lost the answer. Strict-ID recall remains primary, and the C/D tie and negative efficiency conclusion are unchanged.

The tied-score finding warrants disclosure, not a rescue narrative after an unfavorable outcome. We did not change the prompt, score precision, tie rule, controller, models, cohort or captured inference. Alternative ranking interfaces belong in training-only development before a new frozen evaluation. A separate same-candidate, repeated-run serving study would be needed to isolate per-decision execution cost. No fifth method or new inference is introduced here.

The guide's conversational ladder, short paragraphs, source links, `Closing time` and exact subscribe footer remain appropriate. Its time-series episode numbers, forecasting notation and ADF diagnostics are not imposed on this retrieval experiment. Public notebook URL, publication date and full hardware attribution remain explicit editorial checks before publication.

## Verification

The revised notebook is rebuilt and replayed from the saved traces. Acceptance independently recomputes recall, verifies the score-resolution audit excluding singletons, checks the corrected alt text, compares notebook/post headings, resolves all twelve placeholders and compares inline PNG bytes with the exported files. The measured numerical receipt must remain unchanged. No new Ollama inference is required.


**Completed checks:** all 44 code cells replayed without errors and without new model calls. The measured numerical receipt and raw inference file hash are unchanged. All twelve visual placeholders, eleven main section headings, inline/exported PNG bytes, score-resolution counts and corrected accessibility descriptions passed acceptance. The critic's read-only follow-up found no new material inaccuracies in the revision.
