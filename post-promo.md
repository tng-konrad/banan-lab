# Publication copy

## SEO description — 130 characters

Can decision models improve RAG retrieval? A local Ollama test compares BM25, Nimble and Qwen, with results and a Python notebook.

## LinkedIn post

I've been hearing about decision models for a while. So I gave Nimble a retrieval job.

Inspired by a new paper, I built a local Ollama experiment comparing four approaches:

• BM25
• BM25 + Nimble judging
• Hierarchical retrieval with Nimble
• The same navigation rules with ordinary Qwen

32 pages. Twelve evaluation queries. No guaranteed winner.

BM25 + Nimble judging had the highest observed first-choice recall. Nimble's hierarchy led on top-ten recall, but both hierarchies found the annotated passage first on five of twelve queries. Nimble took 1.87× as long in median query time.

A small, exploratory comparison of these models and adapters. Enough to make the test drive interesting ;-)

The post covers the implementation, the results and a wrinkle involving Qwen's tied scores. A self-contained Python notebook comes with it, so you can inspect the evidence and run your own experiment.

Read the post: [add published Substack URL]

#RAG #LLMs #Ollama
