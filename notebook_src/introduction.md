# Decision models, meet the document pile

I've been hearing about **decision models** for a while. A model built to make bounded judgments sounds useful—but what does it buy us inside a retrieval system? A new paper, *Retriever-Free Retrieval: Navigating Documents with Decision Models* ([Wan et al., arXiv:2610.35868v1](https://arxiv.org/abs/2610.35868v1)), gave us a concrete way to find out. Let's take the idea for a test drive.

Retrieval matters because a RAG generator works with the evidence we hand it. A fluent answer cannot quote the passage the retrieval stage never found. Instead of retrieving a shortlist first, the paper uses relevance decisions to walk a document hierarchy. We can test that mechanism on a laptop, with a few practical concessions.

**Compare adding a judge, changing discovery and replacing the model that makes the judgments.** We start with BM25, add a Nimble judge, let Nimble navigate pages and passages, then give ordinary Qwen the same job. These are practical pipeline comparisons, not causal isolation of training or architecture. The goal is practical understanding and an honest comparison. A more elaborate pipeline is allowed to lose.

📓 All code lives in the companion notebook: `decisionrag_ollama_demo.ipynb`.

The writing outline is `post-plan.md`. This edition replays the completed 32-page, twelve-query study and exports its visuals to `./graphs`. It preserves the genuine inference measurements; rerunning analysis does not manufacture a new speed benchmark.
