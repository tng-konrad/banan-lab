Local hierarchical decision retrieval with Ollama — revised plan v2
=================================================================

This revision incorporates an independent critic's review. The first deliverable is **a local adaptation of DecisionRAG testing hierarchical retrieval and candidate-context effects**. It cannot establish the paper's Jev accuracy, specialized single-forward-pass execution, or hosted latency. The original is preserved in `docs/plan-v1.md`; review findings and decisions are in `docs/plan-critique.md`.

Data preparation has run. Retrieval software, inference, and measured results remain planned work.

1. Freeze a small core scope.

   The supplied paper, *Retriever-Free Retrieval: Navigating Documents with Decision Models* (Wan et al., arXiv:2610.35868v1), scores alternatives together while navigating a document hierarchy. Its MultiDoc2Dial configuration includes help-center routing, ten retained pages, two pages of retrieval memory, and twenty-way choice reranking. Its specialized model evaluates questions with a shared forward pass.

   The core local method is stateless **page → passage** navigation, three retained pages, binary judgments, and bounded shared rescoring. Compare BM25, BM25's top fifty passages ranked by the same judge, and navigation ranked by that judge. Defer choice, routing, memory, dense retrieval, answer generation, fine-tuning, and a chat UI.

   Paper reference results are R@1 47.4% and R@10 83.6% for Jev over 4,201 queries, versus 33.6% and 81.1% for dense retrieval plus Qwen3-Reranker-8B. Appendix D.1's shared-context result, 44.4% versus 36.7%, uses a different 297-turn design experiment. The 0.68-second median is a hosted API measurement on 48 timed queries. None is a direct target for the changed local method and subset.

2. Preserve the data and strengthen the held-out policy.

   Use the [public MultiDoc2Dial archive](https://doc2dial.github.io/multidoc2dial/file/multidoc2dial.zip), linked by the [IBM repository](https://github.com/IBM/multidoc2dial). The existing audit independently recovers 488 raw documents, 413 merged pages, 4,110 structural passages, 661 validation dialogues, and 4,201 eligible queries. Each eligible user turn precedes an agent response other than `respond_no_solution`; its evidence resolves to one structural passage. Local IDs preserve raw document identity within merged pages. Matching counts alone does not prove compatibility with every published passage-ID convention.

   Search all 413 pages and 4,110 passages. Never pass the query's annotated domain, target response, gold page, or gold passage to retrieval. Corpus domain metadata may appear as candidate information. Preserve the exact frozen manifests:

   | Purpose | Source | Whole dialogues | Queries |
   |---|---|---:|---:|
   | Design and pilot | Train | 20 | 125 |
   | Evaluation | Validation | 40 | 259 |

   Selection uses seed `42`, SHA256 ordering, and equal dialogue counts per domain. The micro average describes this selected mixture, not the original validation weighting. Also report each domain.

   Tune only on training design queries. Freeze backend, prompts, serialization, ordering, budgets, primary comparison, retries, and resource limits before viewing evaluation scores. Execute the 259-query evaluation once. A wide interval means inconclusive; it does not trigger expansion or retuning. Any later experiment needs a new protocol. Full-validation scores including already used or explored material are descriptive, not independent confirmation.

   Federal REAL ID remains a display fixture: eight passages and 56 labeled examples from fifteen validation dialogues outside the frozen evaluation. All fifteen are conservatively registered as explored in `docs/explored_validation_dialogues.json`. Its archived text contains obsolete dates. Its raw document ID has no eligible training examples; do not relabel these as training. Debugging and prompt changes use actual training-design cases. A ten-document smoke corpus can use one training query's gold document plus nine hash-selected distractors, fixed before outputs. A single-document test checks section selection, not corpus retrieval.

   Before inference, export separate `*_inputs.jsonl` and `*_labels.jsonl`, keyed by query ID. Inputs contain IDs, turn order, and the six preceding utterances including the current user turn. Labels contain gold IDs, reference answers, and annotation metadata. The current combined manifests are preparation records, never API payloads. Build requests by explicitly selecting history and adding corpus candidates. Inject unique sentinels into label-only fields in a test and verify they never enter the complete rendered request. Answer text legitimately present in retrieved evidence is allowed.

3. Pass backend and resource gates before evaluation.

   Observed hardware is an Apple M4 Max, 16 CPU cores, and 128 GiB RAM. The installed Ollama client reports 0.33.2; the listed local model is `qwen3-coder-next:q4_K_M` (51 GB). These observations establish capacity, not inference speed. The running server and decision model remain untested.

   The [decision guide](https://docs.ollama.com/capabilities/decision) documents local `/v1/systemone` from Ollama 0.35.0 without an API key. Use direct HTTP, initially `nimble:9b-q8_0`, and pin its digest. The [model card](https://ollama.com/library/nimble) describes binary/choice probabilities and rereading the prompt per question. One HTTP request does not establish Jev's shared-forward-pass mechanism. Catalog and prose disagree about context size: inspect the installed configuration and validate it. The [API](https://docs.ollama.com/api/systemone) limits local text plus schema to 64 KiB and rejects oversized input.

   Planned setup, not executed during this review:

   ```bash
   curl --fail http://localhost:11434/api/version
   ollama pull nimble:9b-q8_0
   ```

   Gate A verifies server version/digest, binary and multi-question responses, and choice probabilities. Reject missing/extra IDs, nonfinite probabilities, and values outside [0,1]; declare a choice-normalization tolerance. Confirm how the adapter supplies shared candidates to each question, record context configuration, and verify a safely rejected over-limit request. Version strings alone do not pass the gate. Unsupported native inference stops this study. Generated JSON with the coding model is a separately named fallback demo with different score and speed claims.

   Gate B runs all three core methods on ten hash-selected training queries, with domain quotas 3/3/2/2 for DMV/SSA/VA/student aid. Add input-size stress checks using the largest metadata, a 63-passage page, and the longest complete passage. These are feasibility checks, not quality results. Begin with one request in flight; concurrency changes require another design pilot and a new frozen setting.

   At sixteen candidates/request, page screening alone needs at least 26 requests/query, or **6,734 requests over 259 queries**, before passage scoring, rescoring, splits, and retries. A three-page pool can contain 135 passages; the largest merged page has 23,769 words. Memory capacity does not remove these costs.

   Proposed default budget: **two hours wall time for the complete three-method evaluation**, excluding downloads and initial corpus/model loading. This is both the pilot's planning threshold and a hard deadline for the timed evaluation, configurable only before evaluation. At the deadline, cancel the active client request, stop scheduling, retain completed traces, and mark the run incomplete. Project runtime as 259 times the sum of observed mean pilot times for the three methods, plus 25% reserve. Record observed p50/p95, token usage, requests/decisions, context splits, and peak memory; a ten-query tail estimate is provisional. Proceed only if the projection fits, every pilot query succeeds, and complete singleton passages fit the validated context. Otherwise optimize on training data and repeat the gate, or deliver the small demonstration plus a feasibility report. Do not silently shrink the corpus or query set.

4. Implement one exact ranking contract.

   Persist the full corpus with stable IDs and source-order indices. Page descriptions contain normalized titles and deduplicated headings; passage descriptions contain complete text plus title/heading path. Both judged systems use identical passage descriptions. Present candidates in corpus order even when a discoverer ranked them. Use short opaque aliases and record the mapping. Model queries contain only the last six role-tagged utterances. Earlier dataset replies are identical across methods; generated replies are not fed back.

   Initial design settings are `page_budget=3`, `binary_batch_max=16`, `rescore_prefix_max=12`, `output_k=10`, and `bm25_pool=50`. Guard request JSON at 48 KiB, below the server limit; this byte guard does not prove token fit. The context ceiling must not exceed the installed/validated limit and initially is at most 8,192 tokens. Count the backend-rendered prompt with a matching tokenizer/template where available. Otherwise report exact token fit as unverified and rely on the explicit rejection contract below.

   `binary_batches` greedily packs candidates in source order, respecting sixteen questions and input limits. Each candidate has a fixed `noul` relevance question and sees every candidate of its batch. A local byte/token admission rejection or recognized server input-size rejection splits the batch into contiguous halves; a singleton rejection fails the query. Nothing is truncated, summarized, dropped, or assigned a fabricated probability. Other errors follow the retry policy. Log actual batches: changing context limits changes this method.

   `rank_pool` pools binary scores and sorts by score descending, then stable corpus ID. This is approximate cross-context screening, not calibrated global comparison. Take the first twelve candidates and rescore the largest leading prefix that fits one common context, presented in source order. On a local byte/token admission rejection or recognized server input-size rejection, remove its lowest-ranked candidate and retry until at least two fit. Reorder this prefix by its new scores and append the untouched coarse-ranked remainder. If fewer than two fit, keep the coarse ranking and explicitly record skipped rescoring. This declared fallback applies identically to all methods and removes nothing from the discovered pool. An ordinary failed rescore call fails the query; it does not trigger the size fallback. Log shortlist gold coverage, prefix length, and skip rate.

   ```python
   def rank_pool(history, items, n, judge):
       coarse = judge.binary_batches(history[-6:], source_order(items), max_items=16)
       ordered = stable_score_order(coarse)
       prefix, refined = judge.rescore_fitting_prefix(
           history[-6:], ordered[:12], presentation_order="source", min_items=2)
       if refined is not None:
           ordered = stable_score_order(refined) + ordered[len(prefix):]
       return ordered[:n]

   def navigate(history, corpus, judge):
       pages = rank_pool(history, corpus.pages, n=3, judge=judge)
       return rank_pool(history, corpus.passages(pages), n=10, judge=judge)

   def bm25_then_judge(history, corpus, bm25, judge):
       ids = bm25.top_k(history[-6:], k=50)
       return rank_pool(history, corpus.passages_by_id(ids), n=10, judge=judge)
   ```

   The helper returns the actual leading prefix and its scores, or an empty prefix/`None` when fewer than two fit. There is no hidden choice, routing, memory, or generation stage. Rescoring cannot recover candidates discarded by screening. Never claim all selected passages shared a context when only batches or a shortlist did.

   Requests use `state={conversation, candidates}` and `questions={alias: {type: "noul", instructions: ...}}`; scores come from `answers[alias].noul`. Fix the page question and use the identical passage question in both judged systems. Treat corpus content as evidence, not instructions. Save complete requests and responses. Never serialize an entire labeled query record.

   A small package needs `data.py`, `judge.py`, `retrieve.py`, `evaluate.py`, one CLI, and one explanatory notebook. Suggested dependencies: `httpx`, `numpy`, a pinned BM25 implementation, and `pytest`. Freeze BM25 normalization, stemming/stopwords, parameters, and six-turn query serialization. All passage methods see the same title/heading/text content. Tests cover label mapping, ordering/ties, prefix replacement without duplicate IDs, splits without lost candidates, rendered-input isolation, and failure accounting. Persisting the full runnable corpus and separated input/label files remains implementation work.

5. Freeze one primary comparison and define failures.

   Required methods: BM25 top ten, BM25 top fifty plus `rank_pool`, and navigation plus `rank_pool`. They share corpus, histories, representations, output limit, judge, prompt, batch policy, and rescore rules wherever passages are judged. Their discovered pools and compute differ. This is a **system-level quality/cost comparison**, not isolated causal proof about hierarchy. Report pool size, gold coverage, candidates judged, and runtime.

   Primary contrast: **navigation minus BM25-plus-judge on operational R@1**, across all 259 scheduled queries. In a completed run, exhausted transport/schema/context failures have empty rankings and contribute no hit to R@1/5/10. Retain failure categories; never fabricate zero relevance probabilities. Valid empty results and invalid calls have different statuses. All methods attempt the same fixed set. Additionally report accuracy on the intersection of queries successfully executed by every method, showing its denominator. An aborted run is incomplete and has no confirmatory verdict.

   Initial retry policy: one identical-payload retry for timeout or transient transport/server errors; a 120-second request timeout capped by remaining run time. No retry for malformed schemas or unsupported APIs. Size splitting and prefix reduction are separate admission actions. Count all attempts and time. Run queries in manifest order; within each query, order the three methods using a seeded permutation fixed before evaluation. Abort on an unsupported backend or three consecutive queries within any one method that fail from transport/server infrastructure after retries; a successful query for that method resets its counter. The counter counts exhausted queries, not individual requests or admission rejections. Retain traces and mark an aborted evaluation incomplete. Do not switch model or omit observations mid-run.

   Report strict ID R@1/5/10, gold-page retention, initial rescore-shortlist gold coverage, and conditional R@1 given page survival. With one gold passage, Hit@k equals Recall@k. Add a secondary exact-normalized-text duplicate sensitivity, with normalization frozen. Use paired 95% intervals from 10,000 whole-dialogue bootstrap resamples within domain. Only the primary contrast is confirmatory for this subset; other metrics and unadjusted intervals are descriptive/exploratory. Above-zero primary intervals support local improvement, zero-crossing intervals are inconclusive, and below-zero intervals favor the baseline. No outcome establishes the paper's benchmark replication.

6. Keep mechanism diagnostics separate.

   After the core, use fifty hash-selected training-design queries for an exploratory **competitor-context diagnostic**. Each fixed pool contains gold plus nine BM25 hard negatives, excluding exact normalized gold-text copies. Freeze pool, shuffled order, aliases, descriptions, and question wording. Compare the same binary question over its target alone versus the complete ten-candidate state. Send one question/request in both arms so the other question schemas do not change the comparison. Rank the same pool. Gold inclusion makes this an oracle diagnostic, never end-to-end retrieval. Establish input fit before scoring; if pools cannot fit under one frozen policy, report unsupported instead of dropping long cases. Findings on design data remain exploratory.

   A separate **choice diagnostic** compares one choice question over those pools with shared binary ranking. It changes decision type; its whole effect is not attributable to competitor context. A pipeline choice ablation reuses saved upstream page/passage traces and permutes only the top five returned candidates. Its target is R@1: such a permutation cannot change R@5 or R@10. A five-item context overflow fails that ablation query under the same scoring contract, rather than changing the pool after an outcome. Check position sensitivity with a small design-only permutation exercise before freezing order.

   Paper Appendix D.1's separate-per-page versus shared-across-pages experiment is distinct. A later paper-aligned configuration can add routing, memory, ten pages, and twenty-way choice after another resource gate. A small dense embedding baseline with exact cosine search is another extension. Neither is required for the core. Changes need versioned protocols; tuning on evaluation results forfeits held-out status.

7. Deliver evidence and state completion accurately.

   Time live requests without response-cache hits. Separate initialization from full-query p50/p95 and stage times, including retries and admission attempts. Save model digest, server/context/software versions, prompts, complete inputs, aliases, ordering, batches, rescore changes, rankings, failures, and corpus/config hashes. Cache identity includes all content, ordering, backend, and settings. Repeat several design queries before freezing to assess variability. Exact replay checks the evidence trail, not correctness or statistical gain.

   | Increment | Required evidence |
   |---|---|
   | Demonstration | Training-based ten-document example, visible selection path, valid IDs, local timing |
   | Core evaluation | Passed gates, frozen three-method run, primary interval, all-query/common-success metrics, coverage, failures, and compute |
   | Optional extensions | Fixed-pool diagnostics and trace-reused ablations, each with a limited claim |

   Completion means a working small demo and a reproducible, qualified result—even if the method loses. A failed full-corpus gate permits a useful smaller demo and feasibility report, but leaves the 259-query objective unfulfilled. No inference, upgrade, or model download occurred while revising this plan.

Current evidence: `data/prepared/dataset_audit.json`, the two frozen combined manifests, the explanatory fixture, `docs/explored_validation_dialogues.json`, and `scripts/audit_multidoc2dial.py`. Source: [arXiv version 1](https://arxiv.org/abs/2610.35868v1), supplied at `/Users/kbanachewicz/Downloads/2610.35868v1.pdf`. Paper facts were checked against that PDF; corpus counts were recomputed. Backend assumptions remain subject to live gates.
