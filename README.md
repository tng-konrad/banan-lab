Implementations of papers I find interesting - mostly serves as a storage for the notebooks that come iwth the Substack posts.

Work very much in progress in terms of organization.

## STAIR

STAIR investigates whether fine-tuning a language model with document structure, such as a table of contents or section hierarchy, improves targeted section retrieval over flat text search. This adaptation evaluates FLAN-T5-small, Qwen3.5-2B, and Qwen3.5-4B on an accounting textbook to test whether larger backbones make hierarchical catalogues more effective. The accompanying notebook recomputes all metrics, tables, and figures offline from stored trial predictions without requiring local GPU execution.

- **Notebook:** [stair_model_comparison.ipynb](stair/stair_model_comparison.ipynb)
- **Original Paper:** [STAIR: A novel dataset and LLM based retriever for document structure augmentation (arXiv:2609.03874)](https://arxiv.org/abs/2609.03874)
