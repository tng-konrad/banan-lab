Implementations of papers I find interesting - mostly serves as a storage for the notebooks that come iwth the Substack posts.

Work very much in progress in terms of organization.

## STAIR

STAIR investigates whether fine-tuning a language model with document structure, such as a table of contents or section hierarchy, improves targeted section retrieval over flat text search. This adaptation evaluates FLAN-T5-small, Qwen3.5-2B, and Qwen3.5-4B on an accounting textbook to test whether larger backbones make hierarchical catalogues more effective. The accompanying notebook recomputes all metrics, tables, and figures offline from stored trial predictions without requiring local GPU execution.

- **Notebook:** [stair_model_comparison.ipynb](papers/stair/stair_model_comparison.ipynb)
- **Original Paper:** [STAIR: A novel dataset and LLM based retriever for document structure augmentation (arXiv:2609.03874)](https://arxiv.org/abs/2609.03874)

## Memory Tax

Memory Tax investigates whether building dedicated agent memory improves on retrieving evidence directly from raw history, separating the answers memory gains from those it loses. This adaptation compares raw retrieval, compressed session notes, and memory plus source using Qwen3.5-4B through local Ollama, with no-context and gold-evidence controls. The evaluation covers 60 source-reviewed questions from one LoCoMo conversation, with AI-assisted grading. The accompanying notebook recomputes all metrics, tables, and figures offline from stored evidence without requiring Ollama inference.

- **Notebook:** [memory_tax_ollama_demo.ipynb](papers/memory%20tax/memory_tax_ollama_demo.ipynb)
- **Original Paper:** [Memory Tax: Dissecting the Net Value of Agent Memory (arXiv:2610.34073)](https://arxiv.org/abs/2610.34073)

## DEFT

DEFT investigates whether a frozen time-series foundation model's forecasts can be improved under a tight budget of trajectory-level scores by decomposing candidate paths into trend and residual components and refining them separately. This adaptation compares DEFT against Best-of-N quantile selection, raw-horizon Cross-Entropy Method variants (including prior-seeded CEM), pool-only recombination, and random search using a frozen `amazon/chronos-2` checkpoint on 64 M4 monthly and 64 NN5 daily series under coarse five-level and continuous hindsight MSE feedback, alongside follow-up checks on implementation conventions and proposal-variance smoothing. The accompanying notebook recomputes all metrics, tables, and figures offline from cached forecasts and retained search traces without requiring model inference.

- **Notebook:** [deft_forecast_editing.ipynb](papers/deft-expert-guided-forecast-editing/deft_forecast_editing.ipynb)
- **Original Paper:** [Expert-Guided Forecast Editing for Time-Series Foundation Models (arXiv:2607.19659v1)](https://arxiv.org/abs/2607.19659v1)

## e-means

e-means investigates whether replacing Lloyd's full-batch k-means with exponential moving average mini-batch updates, empirical scaling laws, cosine annealing, and dead-cluster revival improves Inverted File (IVF) indexing for Retrieval-Augmented Generation without per-dataset tuning. This local Apple Silicon CPU adaptation compares 2-epoch and 3-epoch `NumpyEmeans` (`faiss-cpu==1.16.0`) against 25-epoch, wall-clock-matched 9-epoch, and 3-epoch Lloyd's k-means, controlled and out-of-the-box `scikit-learn` Mini-Batch k-means, stale Wave-0 Lloyd, and an Exact Flat oracle using `BAAI/bge-base-en-v1.5` (768-d) embeddings across six complete CQADupStack StackExchange domains (215,185 posts, 5,030 queries, k = 2,048, five seeds) under both static indexing and moderate-to-severe corpus drift. The accompanying executed notebook embeds all rendered outputs, tables, and figures, with standalone scripts included to re-download, embed, and reproduce the full run locally.

- **Notebook:** [emeans_rag_demo.ipynb](papers/emeans-rag-ivf/emeans_rag_demo.ipynb)
- **Original Paper:** [E-Means: Scaling Laws for Mini-Batch K-Means (arXiv:2611.09501)](https://arxiv.org/abs/2611.09501)
