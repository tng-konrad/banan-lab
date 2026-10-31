Build a local DecisionRAG reproduction with Ollama
=================================================

The recommended first demo is **hierarchical retrieval over MultiDoc2Dial, with a local decision model, a BM25 comparison, and a frozen evaluation subset**. This exercises the paper's central idea on the same corpus while keeping the software small. An answer-writing UI can follow after the retrieval measurements work.

This is a method reproduction with a replacement model. Matching Jev's reported accuracy, specialized forward-pass implementation, or hosted latency is a separate objective. The data preparation below has been executed; the retrieval experiments have not.

1. Understand the experiment being reproduced.

   The supplied paper, *Retriever-Free Retrieval: Navigating Documents with Decision Models* (Wan et al., arXiv:2610.35868v1), calls its method DecisionRAG. Its hierarchy contains broad groups, documents, and passages. At each level a decision model receives the query and candidate descriptions together, assigns each candidate a binary relevance probability, and expands only the retained branches. An optional multiple-choice judgment reranks the best passages. Candidate probabilities from different binary questions need not sum to one; choice probabilities do.

   Three mechanisms deserve separate experiments: navigating the hierarchy without a first-stage search; judging candidates in a context that includes their alternatives; and replacing generated navigation text with direct decisions. The paper's specialized decision model evaluates multiple questions against one shared context in a forward pass. A normal LLM producing a JSON ranking would reproduce only part of this design.

   These are reference results from the paper, not locally measured targets:

   | Experiment | Paper result | Qualification |
   |---|---|---|
   | MultiDoc2Dial, Jev | R@1 47.4%, R@10 83.6% | Full 4,201-query validation set; Table 2 |
   | Same corpus, dense + Qwen3-Reranker-8B | R@1 33.6%, R@10 81.1% | Different retrieval stack; Table 2 |
   | Shared passage context versus separate page contexts | R@1 44.4% versus 36.7% | 297 design turns, own notepad, different configuration from Table 2; Appendix D.1 |
   | MultiDoc2Dial Jev retrieval latency | Median 0.68 seconds | Hosted API, 48 timed queries, concurrent requests within a query; Appendix F |

   The gains are empirical, not guarantees for every decision model. Appendix G even shows general 8B judges losing R@1 when the page budget grows from three to ten. More exploration is not automatically better. The large-corpus experiment also builds its hierarchy using embeddings offline: the claim concerns retrieval at query time, not a universal ban on preprocessing.

2. Use the verified corpus and a concrete starting document.

   The [IBM MultiDoc2Dial repository](https://github.com/IBM/multidoc2dial) links the [public archive](https://doc2dial.github.io/multidoc2dial/file/multidoc2dial.zip). It contains archived government help documents, human-written conversations, and span-level evidence labels. This is a better first benchmark than manually inventing questions for an unrelated PDF: the structure and labels already exist.

   I downloaded the 6,868,509-byte archive and independently reconstructed the paper's data counts:

   | Data item | Verified count |
   |---|---:|
   | Raw documents | 488 |
   | Pages after merging title-part suffixes | 413 |
   | Structural passages | 4,110 |
   | Validation dialogues | 661 |
   | Eligible validation queries | 4,201 |

   The audit script groups adjacent spans using the official preprocessing rule: continue a passage if the section ID or stripped section title matches the preceding span. It merges page titles after removing their trailing `#number`, while keeping raw document identity inside passage IDs. It constructs queries from user turns followed immediately by an agent response, excluding `respond_no_solution`. Counting every agent turn would incorrectly produce 4,699 evaluation cases. Every eligible validation query resolves to exactly one structural gold passage.

   The concrete first document is **Federal REAL ID**, raw ID `Federal REAL ID#3_0`, in the DMV domain. Its complete archived text, eight structural passages, and labeled exploratory examples are saved in `data/prepared/example_document.json`. Those examples come from validation dialogues outside the frozen 40-dialogue evaluation subset. Use a question about which identification documents already qualify, or whether the archived guide allows obtaining a REAL ID online. Show the retrieved paragraph and its ID. Treat the text as an archived retrieval fixture: it contains obsolete dates. Keep prompt tuning on the designated training design split; the fixture is for an explanatory display.

   For a visual smoke test, use that document plus nine other DMV documents chosen by a fixed hash of document ID before judging model outputs. The single-document version checks section selection; the ten-document version also checks document selection. Neither supplies a benchmark performance claim.

   For quantitative evaluation, keep **all 413 pages and all 4,110 passages** in the searchable corpus. Reduce the query count, not the search space. Two manifests are already frozen with seed `42` and deterministic SHA256 ordering within each of the four domains:

   | Purpose | Source split | Whole dialogues | Eligible queries |
   |---|---|---:|---:|
   | Design and pilot | Train | 20 | 125 |
   | Evaluation | Validation | 40 | 259 |

   The evaluation subset has 66 DMV, 60 SSA, 69 VA, and 64 student-aid queries. Its 259 queries are sufficient for an initial measured demo; its 40 dialogue clusters still give uncertain estimates of small gains. Expand to the full validation set if the difference is small. Use the design split to choose prompts, budgets, and request sizes, then freeze them before evaluation.

   Alternatives for later work are [ConditionalQA](https://github.com/haitian-sun/ConditionalQA), for standalone scenario questions and multiple evidence paragraphs, or a structured technical manual with independently annotated question-to-section labels. The paper's ConditionalQA setup has 652 pages and 58,483 paragraph units, so it is a larger first extraction task. Natural Questions' 2.68 million paragraphs and MTRAG's extra relevance judging are unnecessary for the first demo. The attached 21-page paper itself is a convenient additional PDF smoke test, but manually labeling it would not reproduce its benchmark results.

3. Establish the local inference path before implementing retrieval.

   This laptop reports an Apple M4 Max, 16 CPU cores, and 128 GiB RAM. Its local model inventory currently contains `qwen3-coder-next:q4_K_M` (51 GB). The installed Ollama client reports version `0.33.2`; the running server's version still needs an explicit API check.

   The [Ollama decision guide](https://docs.ollama.com/capabilities/decision) documents local `/v1/systemone` support from version `0.35.0`. Check the server version and upgrade Ollama if needed. Use direct HTTP so the experiment does not depend on SDK release alignment. Local decision calls require no API key.

   Use **`nimble:9b-q8_0`** as the initial decision model. The [published tag list](https://ollama.com/library/nimble/tags) lists a 9.5 GB download, comfortably within this laptop's memory budget; measure actual runtime memory during the pilot. Pin the resolved model digest. The [model card](https://ollama.com/library/nimble) documents binary and choice probabilities, up to 64 questions per request, and 2–26 choice options. It also says the prompt is read per question: one API request therefore does not establish Jev's shared-forward-pass execution. Describe this as a local substitute and measure its speed independently.

   The model catalog and prose contain different context-length claims. Begin conservatively with an 8,192-token ceiling, leave room for instructions and question overhead, and verify the installed model metadata and actual endpoint behavior. Also respect the [System One API](https://docs.ollama.com/api/systemone) limit of 64 KiB for local text plus schema. Start with 16 candidates per binary request and a final choice among five passages; increase only after the design pilot. Never silently truncate candidates. Recursively split oversized candidate batches; stop with an explicit error if one complete passage cannot fit. The longest audited passage has 3,027 words, so this check matters.

   Planned setup commands, **not executed in this analysis**, are:

   ```bash
   curl --fail http://localhost:11434/api/version
   ollama pull nimble:9b-q8_0
   ```

   An example decision request is:

   ```json
   {
     "model": "nimble:9b-q8_0",
     "state": {
       "conversation": [{"role": "user", "text": "Which IDs already qualify?"}],
       "candidates": [
         {"id": "p1", "text": "The enhanced license and passport already comply."},
         {"id": "p2", "text": "A replacement card can be requested after loss."}
       ]
     },
     "questions": {
       "p1": {"type": "noul", "instructions": "Should the next reply be based on candidate p1?"},
       "p2": {"type": "noul", "instructions": "Should the next reply be based on candidate p2?"}
     },
     "keep_alive": "10m"
   }
   ```

   Read relevance from `answers[id].noul`. Both questions see both candidates. The text in this request is illustrative, not a quotation from the dataset. A final `choice` question maps each shortlisted passage ID to its description and ranks using `probabilities`.

   If updating Ollama is inconvenient, the already installed coding model can generate a schema-constrained list of passage IDs using [structured outputs](https://docs.ollama.com/capabilities/structured-outputs). That is a fallback demonstration of navigation. Its generated scores are not decision-model probabilities, and its latency tests a different implementation. Keep its result rows separate.

4. Implement the smallest useful retrieval loop.

   Start with two levels, **page → passage**. Each page description contains its title and deduplicated headings. Each passage retains its text, title, heading path, and immutable ID. Candidate order follows source document order, not relevance labels. Store the hierarchy as ordinary JSON; no vector database, agent framework, fine-tuning, or automatic summary generation is needed.

   The initial configuration keeps three pages, scores their passages, and applies one choice judgment to the five best passages. Return ten passage IDs with their provenance. This is an explicitly simplified configuration. A subsequent paper-aligned configuration adds help-center routing, ten retained pages, two pages of retrieval memory, and choice reranking of up to twenty passages.

   ```python
   def retrieve(history, corpus, judge, page_budget=3):
       query = history[-6:]
       page_scores = judge.binary_batches(query, corpus.page_descriptions())
       pages = stable_top_k(page_scores, k=page_budget)
       candidates = corpus.passages_in_source_order(pages)
       passage_scores = judge.binary_batches(query, candidates)
       ranked = stable_top_k(passage_scores, k=min(20, len(candidates)))
       if len(ranked) >= 2:
           shortlist = ranked[:5]
           ranked = judge.choice_rank(query, shortlist) + ranked[5:]
       return ranked[:10]
   ```

   The sketch omits transport, batching, and logging, not additional retrieval stages. Binary scores from different batches are only approximate comparisons; rescore shortlisted pages together before pruning when a level needs several requests, and reserve a shared final request for passage comparison. Record when the complete passage pool could not share a context. Never report a split-context run as an all-candidates-in-one-context experiment.

   The later help-center configuration should follow Appendix B.2: retain the best center, its runner-up when the top score is below 0.8, and the center of the most recently retrieved page. The 0.8 rule is Jev-specific; either copy it explicitly as an uncalibrated reproduction setting or tune a replacement on design data. Memory lists pages from the system's own earlier top-ranked passages, never gold pages. Reset it between dialogues and between methods. Keep histories identical across methods using the dataset's earlier utterances; do not feed generated answers back during retrieval evaluation.

   A compact implementation can use Python, `httpx`, `numpy`, a BM25 library, and `pytest`. Keep reusable code in `decisionrag/data.py`, `judge.py`, `retrieve.py`, and `evaluate.py`; expose one CLI for prepare, query, and evaluate. A notebook should inspect examples and reports by calling those modules. Add tests for label mapping, irreversible pruning, stable ties, response-ID validation, no target-answer leakage, and per-dialogue memory reset. No model-mocking test can establish retrieval quality; that requires the frozen experiments.

5. Validate mechanisms before claiming a reproduction.

   Use these comparisons in order. The first three are the required small demo; the others strengthen the scientific result.

   | Comparison | Controlled question |
   |---|---|
   | BM25 over all passages | Does the new method improve on a reproducible lexical baseline? |
   | BM25 top 50 + the same decision judge | Does navigating find useful evidence that reranking the BM25 candidate pool misses? |
   | Hierarchical decision retrieval | What quality and local cost does the core method achieve? |
   | Dense retrieval, then dense top 50 + the same judge | Does the finding survive an embedding baseline? |
   | Same hierarchy without final choice | Does the last choice step help rank one? |
   | Same hierarchy with page budgets 3 and 10 | What accuracy/coverage tradeoff does exploration create? |

   The inexpensive dense extension uses [qwen3-embedding:0.6b](https://ollama.com/library/qwen3-embedding:0.6b), listed at 639 MB, through `/api/embed`. Store normalized vectors and perform exact NumPy cosine search over 4,110 passages. Use the model's documented retrieval query instruction, the same six-turn text, and `truncate:false`. Freeze embedding preparation independently of labels. This remains a smaller baseline than the paper's 8B embedding plus 8B reranker, so it cannot support the original headline comparison.

   Separately isolate the **shared-context hypothesis** on fixed candidate pools. For 50 deterministically selected design queries, combine the gold passage with 19 BM25 hard negatives and shuffle with a fixed seed. Compare: (a) one candidate in each context; (b) all 20 candidates in each shared context; (c) one multiple-choice question over those same 20 candidates. Each run sees identical candidate content, and the gold is included only because this is an explicitly labeled diagnostic, not end-to-end retrieval. Keep the questions and model fixed; vary only the presence of competing passages. If 20 complete passages do not fit, declare a smaller fixed pool size for every query before the measured run, or report the diagnostic as unsupported. Do not drop difficult queries after observing results.

   Appendix D.1's shared-across-pages versus separate-per-page design is a different ablation from independent single-candidate contexts. It can be added as its own row. Neither diagnostic should be compared numerically to the paper's 47.4% full-pipeline result.

   Report strict passage-ID R@1, R@5, and R@10 on the 259 evaluation queries. With one gold passage per query, Hit@k equals Recall@k. Also report gold-page retention after page selection, candidate-pool gold coverage, and R@1 conditional on the gold page surviving. These identify routing losses separately from ranking losses. Show per-domain results alongside the primary micro average. Use paired 95% bootstrap intervals with 10,000 resamples of whole dialogues; for this stratified subset, resample dialogues within domain and state that choice.

   Strict IDs are the primary result. As a sensitivity check, allow exactly identical normalized passage texts under different IDs, report the normalization, and retain both metrics. Do not count a merely related paragraph as correct or let the evaluated model grade its own retrieval.

   Classify a comparison as supportive when its paired interval is above zero, inconclusive when it includes zero, and contrary when it is below zero. A valid execution with no gain is still a useful reproduction finding. Matching 47.4% on a different subset or a smaller corpus would not establish benchmark replication.

6. Measure runtime honestly and keep a replayable evidence trail.

   First run a transport check, then ten design queries. Measure real wall time before projecting the 259-query run; do not assume the paper's sub-two-second latency transfers. Start with one request in flight. Increase to two only if the pilot improves end-to-end latency without memory pressure. Batch APIs can still perform substantial work per question internally. Report model loading separately, then warm live retrieval p50/p95, request count, candidate count, input/output usage, retries, failure rate, and observed memory. Record each stage's duration. Time BM25 and dense retrieval from the same query boundary, including dense query encoding; account for offline corpus preparation separately.

   Save ordered candidate IDs, exact rendered context, returned probabilities, retained branches, final ranking, model digest, Ollama server version, prompt version, configuration, source checksum, and timing to JSONL. A cache key must include all content and ordering plus model/configuration identity. Use stored responses for exact analysis replay, and bypass response caching for latency runs. Repeat several live queries to assess inference variability; replay proves artifact reproducibility, not model correctness.

   Acceptance requires valid IDs, complete candidate accounting, successful label mapping, no silent context truncation, and explicit failures. An exhausted request retry becomes a failed query, not a quietly omitted observation or a zero score. Report performance with its failure denominator and a separate successful-query sensitivity if necessary.

   Optional answer generation comes last. Give one fixed local generator the top five passages from every method, plus no-context and gold-context controls. Request passage citations, review a small blinded set manually, and keep answer faithfulness separate from retrieval scores. The current coding model can serve this UI role; choosing a smaller answer model is a later optimization. This step is unnecessary for the first claim about retrieval.

7. Build in three reviewable increments.

   | Increment | Deliverable | Completion evidence |
   |---|---|---|
   | Core demo | Local decision adapter, ten-document query example, page-to-passage loop | Displayed path, ranked evidence, valid IDs, measured local latency |
   | Measured reproduction | Full searchable corpus, BM25 and same-judge reranking, frozen 259-query run | Metrics, paired intervals, failure analysis, complete traces |
   | Stronger validation | Shared-context diagnostic, page/choice ablations, optional dense baseline | Controlled comparisons and a clear verdict for each hypothesis |

   A reasonable engineering scope is a few small modules plus one notebook; estimate execution time from the pilot rather than a speculative laptop benchmark. Keep fine-tuning, million-passage corpora, a vector database, and a polished chat application outside these increments.

   Current workspace artifacts are:

   - `scripts/audit_multidoc2dial.py`: executable standard-library data audit and manifest builder.
   - `data/prepared/dataset_audit.json`: passed corpus contract, archive hash, and subset counts.
   - `data/prepared/design_manifest.json`: 125 labeled training queries in complete dialogues.
   - `data/prepared/evaluation_manifest.json`: 259 labeled validation queries in complete dialogues.
   - `data/prepared/example_document.json`: concrete document, passages, and exploratory examples outside the evaluation subset.
   - `data/raw/multidoc2dial.zip`: downloaded archive, excluded from Git.

   Reproduce the data preparation from the repository root with `python3 scripts/audit_multidoc2dial.py`. Retrieval modules, a notebook, and model results are planned work, not present deliverables. No Ollama upgrade or model download was performed during this analysis.

Paper source: [arXiv version 1](https://arxiv.org/abs/2610.35868v1), supplied locally at `/Users/kbanachewicz/Downloads/2610.35868v1.pdf`. Method details and reference metrics above were checked against the supplied PDF, particularly Sections 3–6 and Appendices A, B, D, F, and G. Dataset counts were recomputed from the downloaded archive; they are not inferred from the paper.
