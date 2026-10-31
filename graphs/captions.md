# Notebook visuals

One sequence in notebook display order, including graphs and tables; literal suffix 0111.

## Visual 1: graph_1_0111.png

**Notebook/post section:** A small corpus, chosen before the queries

All four methods search the same 32-page, 329-passage corpus. The frozen validation cohort has 12 queries in 12 dialogues. The corpus was selected first, eight pages per domain; queries require in-corpus gold evidence. Original holdout and explored dialogues are excluded. Results describe this restricted task.

We search 32 pages and 329 complete passages. The pilot uses 8 training queries and evaluation has 12 queries. Domain balance makes the design easy to inspect; it does not make twelve queries representative of all RAG workloads.

**Alt text:** Corpus hierarchy beside grouped bars for the training-design and held-out query counts in four domains.

## Visual 2: graph_2_0111.png

**Notebook/post section:** A decision model versus an ordinary LLM

Conceptual A/B/C/D schematic. C and D use the same navigation controller with different scoring adapters; no performance values are implied.

A→B adds judging/rescoring; B→C changes discovery, candidate pools and workload; C→D replaces the model and scoring adapter under common navigation rules. The initial page corpus matches, but selected pages and downstream work can differ. These are end-to-end system comparisons, not causal isolation of architecture or training.

**Alt text:** Four horizontal paths compare lexical retrieval, lexical retrieval with Nimble, and hierarchical retrieval using either Nimble or Qwen.

## Visual 3: post_3_0111.png

**Notebook/post section:** Keep the test honest

Frozen settings and the joint quality, latency, and validity requirements. These are criteria, not measured success.

This table is our contract with the experiment: corpus, cohort, models and resource limits are fixed. The mini study is explicitly exploratory. A successful run does not by itself establish a retrieval-quality or efficiency win.

**Alt text:** Two-column table lists corpus, cohorts, controller limits, models, and the requirements for an efficiency claim.

## Visual 4: graph_4_0111.png

**Notebook/post section:** Keep the test honest

Run pilot-20261030T213827Z: 8 training pilot queries, domain-weighted to 12 queries over all four options. Forecast 75.35 minutes including 25% reserve and uncovered scheduling, against 133.64 available minutes; gate passed. Only two or three queries per domain makes the forecast uncertain.

The training pilot forecast 75.3 evaluation minutes, including reserve, against 133.6 minutes remaining. Resource gate: PASS. This was a forecast made before evaluation, not a new measurement during replay.

**Alt text:** Stacked projected runtime by method, reserve, and scheduling, compared with the fixed laptop budget.

## Visual 5: graph_5_0111.png

**Notebook/post section:** What happened on the laptop?

Run evaluation-20261030T222218Z; mini_3h: 12 validation queries / 12 dialogues / 32 pages; attempted/scheduled: A 12/12, B 12/12, C 12/12, D 12/12; status complete. Strict passage-ID R@1/5/10; failed rankings count as misses. Unattempted outcomes are not invented. Intervals are 95% percentile bounds from 10,000 paired whole-dialogue resamples within domain. Restricted-corpus mini comparison is exploratory and cannot confirm the original full-study claim.

At rank one, A: 33.33%; B: 58.33%; C: 41.67%; D: 41.67%. Read Recall@1 and Recall@10 separately: a method can return useful evidence in its bundle without choosing it first. The error bars are exploratory dialogue-bootstrap intervals, not proof of a universal ordering. Exact normalized-text R@1 is A: 33.33%; B: 58.33%; C: 50.00%; D: 50.00%; this sensitivity leaves strict-ID scoring primary and does not establish equivalence or an efficiency win.

**Alt text:** Grouped recall bars for A, B, C, and D at ranks one, five, and ten, with available dialogue-cluster intervals.

## Visual 6: post_6_0111.png

**Notebook/post section:** What happened on the laptop?

Run evaluation-20261030T222218Z; mini_3h: 12 validation queries / 12 dialogues / 32 pages; attempted/scheduled: A 12/12, B 12/12, C 12/12, D 12/12; status complete. Strict-ID recall percentages and all-attempt p50/p95 seconds, with observed/scheduled denominators and failures. Unrounded values are in sidecars. Restricted-corpus mini comparison is exploratory and cannot confirm the original full-study claim.

Each method attempted A: 12/12, B: 12/12, C: 12/12, D: 12/12. The table puts completed counts, failures, strict recall and full-query latency together. For scale, the BM25 median is 0.003 seconds; model-assisted retrieval takes much longer here.

**Alt text:** Four method rows compare query completion, failures, recall, and latency.

## Visual 7: graph_7_0111.png

**Notebook/post section:** What happened on the laptop?

Run evaluation-20261030T222218Z; mini_3h: 12 validation queries / 12 dialogues / 32 pages; attempted/scheduled: A 12/12, B 12/12, C 12/12, D 12/12; status complete. Operational R@1 versus all-attempt median full-query seconds on a logarithmic time axis. C/D run first; A/B timings come from the later sweep and are secondary. Restricted-corpus mini comparison is exploratory and cannot confirm the original full-study claim.

Left is faster, higher is better. The logarithmic horizontal axis matters: milliseconds and minutes occupy the same plot. A/B were measured in the later sweep, so their speed comparisons with C/D are descriptive. The picture does not include answer-generation quality or a modern dense retrieval baseline.

**Alt text:** A quality-versus-latency scatter plot with one colored, shaped marker per method.

## Visual 8: graph_8_0111.png

**Notebook/post section:** Does the decision model earn its place?

Run evaluation-20261030T222218Z; mini_3h: 12 validation queries / 12 dialogues / 32 pages; attempted/scheduled: A 12/12, B 12/12, C 12/12, D 12/12; status complete. C minus D R@1 in percentage points versus zero (exploratory comparison), and ratio of median full-query times versus one. Intervals are 95% percentile bounds from 10,000 paired whole-dialogue resamples within domain. Verdict: exploratory mini comparison. Validity gates must also pass. Restricted-corpus mini comparison is exploratory and cannot confirm the original full-study claim.

C−D Recall@1 is 0.0 percentage points, with a 95% interval from -33.3 to 33.3. The median-time ratio C/D is 1.87 (95% interval 1.80–2.07); a ratio above one means Nimble took longer. Equal observed recall is not evidence of equivalence. The interval resamples queries from one captured run, not repeated runs or hardware. Different selected branches and workloads are included in this end-to-end time comparison.

**Alt text:** Two interval plots compare the quality difference with zero and the time ratio with equality.

## Visual 9: post_9_0111.png

**Notebook/post section:** Does the decision model earn its place?

Run evaluation-20261030T222218Z; mini_3h: 12 validation queries / 12 dialogues / 32 pages; attempted/scheduled: A 12/12, B 12/12, C 12/12, D 12/12; status complete. Restricted 32-page corpus and 12 queries; descriptive evidence only. No original full-study noninferiority claim. Intervals are 95% percentile bounds from 10,000 paired whole-dialogue resamples within domain. Restricted-corpus mini comparison is exploratory and cannot confirm the original full-study claim.

The recorded verdict is **exploratory mini comparison**. Completion of all four mini methods is distinct from running our original local 259-query study. Input fit, warm residency and timing integrity describe execution validity; they do not turn an uncertain quality comparison into a win.

**Alt text:** A table lists each primary quality, latency, and validity requirement and the resulting verdict.

## Visual 10: graph_10_0111.png

**Notebook/post section:** Where did the hierarchy lose the answer?

Run evaluation-20261030T222218Z; mini_3h: 12 validation queries / 12 dialogues / 32 pages; attempted/scheduled: A 12/12, B 12/12, C 12/12, D 12/12; status complete. Strict structural-ID counts partition every scheduled query into a rank-one ID hit, retained-page ID miss, pruned gold page, execution failure, or unattempted. The categories sum to the scheduled denominator. An ID miss may still match the target text; see the exact-text sensitivity. Restricted-corpus mini comparison is exploratory and cannot confirm the original full-study claim.

Nimble pruned the gold page on 0 queries; Qwen did so on 2. Retaining a page is only the first hurdle: Nimble then had 7 rank-one structural-ID misses on retained pages, and Qwen 5. An ID miss may still match the target text.

**Score resolution:** C made 275 scored request groups and D 235. Excluding singletons, D had all-equal scores in 177/234 groups and some tie in 231/234; C had 0/273 all-equal groups and 0/273 groups with any tie. Equal scores fall back to candidate-ID order. These are dependent request groups, not extra evaluation queries; ties can be appropriate for irrelevant candidates. They describe this adapter, not a general ordinary-LLM limitation or a cause of its speed.

**Alt text:** Two stacked bars show Nimble and Qwen strict-ID hits, retained-page ID misses, page pruning, and execution status.

## Visual 11: post_11_0111.png

**Notebook/post section:** Where did the hierarchy lose the answer?

Run evaluation-20261030T222218Z; mini_3h: 12 validation queries / 12 dialogues / 32 pages; attempted/scheduled: A 12/12, B 12/12, C 12/12, D 12/12; status complete. Strict R@1 by domain with the attempted A/B/C/D query counts. Full R@10 and failure data are in the numeric sidecars. The micro average describes this selected mixture. Restricted-corpus mini comparison is exploratory and cannot confirm the original full-study claim.

The domain breakdown is an inspection aid, not four small leaderboards. With three validation queries per domain, a single changed answer moves recall by 33.3 percentage points. Do not choose a domain-specific winner from these counts.

**Alt text:** Domain rows compare the four methods’ recall at rank one with explicit query counts.

## Visual 12: graph_12_0111.png

**Notebook/post section:** Where did the hierarchy lose the answer?

Run evaluation-20261030T222218Z; mini_3h: 12 validation queries / 12 dialogues / 32 pages; attempted/scheduled: A 12/12, B 12/12, C 12/12, D 12/12; status complete. ECDFs include all attempted query times, including retries and failure time. Crosses mark failed queries. Unattempted queries have no latency; log time axis. Restricted-corpus mini comparison is exploratory and cannot confirm the original full-study claim.

Every attempted query contributes to its curve, including any failed attempt. The right tail exposes slower cases that a median hides. The curves reuse original inference durations; the time taken to redraw them is not a speed benchmark.

**Alt text:** Latency ECDFs for observed methods A, B, C, D, with median and ninety-fifth-percentile timings and failure markers. All four methods have attempted queries.
