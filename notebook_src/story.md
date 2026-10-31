<!-- section: setup -->
## Run the notebook

This edition reruns the saved experiment in **replay** mode: it recomputes metrics, confidence intervals, figures and tables, with **no new model calls**. The captured live study searched 32 pages and 329 complete passages, used eight training pilot queries and twelve validation queries, and finished in 107.1 minutes. Replay time is not inference time.

Captured execution context: macOS arm64, Ollama 0.40.13, and both roughly 9B Q8_0 models resident. The run receipt does not record CPU model or RAM capacity, so these timings are not a promise for every laptop.

All implementation code is inline. The public corpus and tokenizer are downloaded separately into `data/`, checked against pinned SHA256 hashes, and reused on later runs. Saved model traces live in `results/*.json`. No project helper scripts are runtime dependencies. To share these measured results, include those JSON files; the data cache can be downloaded again.

Install the dependencies into the notebook's kernel, then restart and Run All:

```bash
python -m pip install numpy==2.0.2 pandas==2.3.3 matplotlib==3.9.4 httpx==0.28.1 rank-bm25==0.2.2 tokenizers==0.22.2 ipykernel==6.31.0 jupyterlab
jupyter lab decisionrag_ollama_demo.ipynb
```

For a fresh experiment, set `CFG.mode = 'evaluate'`, start a compatible local Ollama server at `http://127.0.0.1:11434`, and separately install `nimble:9b-q8_0` and `qwen3.5:9b-q8_0`. The live cap remains three hours for checks, warmup, pilot, diagnostics and evaluation. Downloads and offline plotting are outside it. `pilot` stops before evaluation; `demo` is a training example. Models are never downloaded automatically.

Every numbered visual is saved in **`./graphs`**. We use one sequence in notebook display order, including tables: plots are `graph_<number>_0111.png`; tables are `post_<number>_0111.png`. The suffix `0111` is literal. Sidecars and a manifest record source runs, captions and measured values. `post-plan.md` uses the same headings and visual numbers. All reusable functions and helper classes live in the themed Utils section; expand those cells to inspect the implementation. Settings live in CFG under Setup.

<!-- section: data -->
## A small corpus, chosen before the queries

We use **MultiDoc2Dial**, conversations grounded in archived US government help pages. A query includes the recent conversation; its target is an annotated structural passage supporting the next reply. The four domains are DMV, Social Security, Veterans Affairs and student aid. Archived pages can contain obsolete administrative information.

The reconstructed archive contains 413 merged pages and 4,110 passages. For the laptop test we first select eight pages per domain by hash, then keep all 329 complete passages on those 32 pages. Every method sees the same corpus. No query-specific domain or answer-containing shortlist is supplied.

We choose two training dialogues and three validation dialogues per domain, one eligible query per dialogue. Eligibility requires evidence inside the fixed corpus. Validation queries follow a frozen hash order without consulting model outcomes. Previously explored dialogues and the larger original evaluation cohort are excluded. This is a restricted task; its scores cannot be compared directly with full-corpus benchmark scores.

<!-- section: bm25 -->
## BM25: give keywords the first turn

We start with **BM25**, which rewards matching query terms while controlling for repetition and passage length. It is cheap, deterministic and a useful baseline before asking a language model to do anything expensive.


In plain English, rare matching words help, repeating one word eventually stops helping much, and a long passage does not win simply by containing more words. We fix $k_1=1.5$ and $b=0.75$: higher $k_1$ lets repeated terms matter for longer; higher $b$ applies stronger length normalization.

Option **A** returns ten passages. It searches titles, parent headings, local headings and complete passage text, using Unicode word tokens and case folding. The query is the retained conversation history. There is no stemming or fitted embedding model. If user and document use different words, a lexical match can miss the point. That motivates our next rung; it does not guarantee that rung will win.

<!-- section: judge -->
## Add a judge

Option **B** keeps BM25 for discovery, then asks **Nimble** whether each of the first fifty passages would help answer the request in context. Its native binary decision score orders passages; we do not claim it is a calibrated probability of retrieval success.

Here a decision model specializes in bounded judgments and exposes native choice scores. Nimble supplies yes/no relevance scores, while the ordinary LLM control generates structured numeric scores. The paper's shared-forward serving is a separate efficiency mechanism that this implementation does not establish. We do no dataset-specific fine-tuning; Nimble arrives with prior decision training.


The judge sees conversation and candidates together. We sort its scores, then jointly rescore a bounded leading prefix. A larger lexical pool can rescue more misses at greater inference cost; evidence outside the first fifty remains unreachable. A/B compares the full judging/rescoring addition to BM25. B/C changes discovery, candidate pools and compute, rather than isolating hierarchy at equal work.

The utilities define complete candidate states, strict parsing and common context admission. Candidate text is evidence, never an instruction. We keep up to sixteen complete candidates in a state but ask at most four relevance questions per request for **both** models. This addresses the context-cache failure seen during development without silently trimming passage text.

<!-- section: navigation -->
## Let the judge navigate

Option **C** uses the judge for discovery as well. We rank page metadata—domain, title and complete headings—keep three pages, then rank every passage on those pages and return ten passages. The core mechanism is:

1. Score pages using the conversation.
2. Keep the three highest-ranked pages.
3. Score the complete passages on those pages.
4. Return the ten highest-ranked passages.

Keeping fewer pages saves passage work but can prune the answer before passage ranking begins. Keeping more pages costs more. The page budget, sixteen-candidate limit, four-question groups and twelve-item maximum rescoring prefix were fixed before evaluation. Cross-batch scores are approximate; prefix rescoring does not calibrate the entire corpus.

The controller in Utils also implements A and B so preprocessing, timing and failures share one path. It is stateless across queries. The paper's help-center routing, retrieval memory, twenty-way choice, ten retained pages and Jev's specialized shared-forward serving are outside this adaptation.

<!-- section: control -->
## A decision model versus an ordinary LLM

Could an ordinary LLM do the same job? Option **D** uses the **identical controller**, with **Qwen3.5 9B Q8_0** generating structured relevance scores instead of Nimble's native decisions. Nimble is also a 9B Q8_0 model from the Qwen3.5 family. Their weights, training and scoring interfaces still differ.

C versus D is a practical end-to-end replacement: same starting corpus, representations, controller rules, page budget, question groups, rescoring and output count. The models can choose different pages, so downstream candidate states and total work differ. That is part of the end-to-end result, not an unfair comparison or a per-decision speed test. Qwen's generated scores are ranking values, not calibrated probabilities. Neither model generates an answer or a chain of reasoning for retrieval.

**Strength:** we compare two concrete ways to make the same decisions locally. **Limitation:** this concerns these models, adapters and serving paths. Several questions in one Ollama request do not prove a shared transformer forward pass, and caching may differ. A local loss does not refute the paper's hosted execution claim; a local win would not establish general decision-model superiority.

<!-- section: protocol -->
## Keep the test honest

We freeze corpus, queries, parameters and schedule before evaluation. The target reply, reference answer and gold passage IDs stay out of model inputs. Both Q8 models must remain resident. The common input contract allows 8,192 tokens minus 512 reserved for output, with a 48 KiB guard. An oversized complete singleton fails rather than being silently shortened.

Eight training queries supply a 32-combination pilot. Its domain-weighted runtime forecast includes 25% reserve and must fit the remaining three-hour workflow allowance. We use the stricter of wall and monotonic deadlines. An earlier attempt had large clock gaps; it motivated a timing-integrity gate, not a claim of clean wall-clock speed from an active timer.

C/D run before A/B, with seeded order within each query. This prevents B priming Nimble's cache before C, while making speed comparisons across the two sweeps descriptive. Identical-candidate adapter checks and repeat probes use training data only. All four methods then receive the same twelve validation queries, one per new dialogue, three per domain.

<!-- section: metrics -->
### What counts as a useful retrieval?

Our quality measure is **strict passage-ID Recall@k**:


In plain English, did the annotated structural passage appear in the first $k$ results? Recall@1 counts correct first choices; Recall@10 measures whether a ten-passage evidence bundle contains it. These are different questions. Exact-text sensitivity is also retained because duplicate text can have different structural IDs.

Execution failures count as misses. Unattempted queries stay separate with no invented latency. Timing includes admission, retries and loading inside a query; checks, pilot and diagnostics also consume the workflow cap. Python `tracemalloc` is not total laptop or GPU memory.

We resample whole dialogues within domain 10,000 times, keeping C/D paired. Twelve dialogues give broad exploratory intervals. Our original local 259-query design's two-point noninferiority rule remains visible for provenance; completing this mini run cannot establish that larger claim. The 259-query design is our larger local experiment, not the paper's full benchmark. Failing to establish noninferiority does not prove one model is inferior.

These intervals resample queries from one captured run. They do not measure variability across repeated executions, hardware or serving configurations. Multiple request groups within one query are not extra independent evaluation examples.

<!-- section: results -->
## What happened on the laptop?

Now we let the saved rankings answer the question. Metrics, confidence intervals and plots are recomputed from raw traces, rather than copied from earlier prose. Successful execution means the protocol ran; recall measures how often it found the evidence.

<!-- section: value -->
## Does the decision model earn its place?

We compare C and D jointly on quality and full-query latency. Equal observed Recall@1 does not establish equivalent quality. The paired interval shows the uncertainty; the time ratio shows which implementation took longer on this machine.

<!-- section: failures -->
## Where did the hierarchy lose the answer?

A hierarchy can fail before the passage judge gets a vote. We separate pruning the gold page from retaining it but missing the annotated structural ID at rank one. An ID mismatch can still match the target text, so the exact-text sensitivity matters. Domain results help inspect examples, but three queries per domain cannot establish a reliable domain ranking.

We also inspect score resolution. Equal relevance scores are broken by candidate ID in the frozen controller. The diagnostic below counts only scored groups with at least two questions; singleton groups are trivially all-equal. Many groups may genuinely contain only irrelevant candidates, so ties do not prove incorrect scores or explain the speed difference. They describe this numeric-score adapter's ordering behavior, not every ordinary LLM. We preserve the result rather than retuning the prompt on held-out outcomes.

The latency distribution includes every attempt, making slow queries and failures visible. Exact-text sensitivity, common-success comparisons, candidate traces and request evidence remain in the sidecars.

<!-- section: closing -->
## Closing time

We climbed from lexical retrieval to semantic judging, let the judge navigate a hierarchy, then gave an ordinary LLM the same job. These compare complete implementations: adding judging/rescoring, changing discovery, and replacing the model-plus-adapter under common navigation rules.

The result reading below comes from the captured run. This is a restricted-corpus mechanism test, not full-paper replication. There is no dense retrieval/reranking baseline, answer generation or test of the paper's hosted execution. Better Recall@10 can help RAG while a worse first choice still matters. More machinery has to pay its way; it does not get a participation trophy ;-)

A larger follow-up should freeze new queries, broaden the corpus and add a modern dense baseline. Test serving optimizations separately from quality, using the same end-to-end timing boundary. Do not tune on these twelve outcomes and continue calling them a fresh test.

The companion article is `post.md`; `post-plan.md` retains the outline and visual placeholders. Narrative section titles match across the notebook and article; Setup and Utils collect notebook-specific mechanics.
