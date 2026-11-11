def retrieval_values(hypotheses, gold_id):
    positions = [i + 1 for i, row in enumerate(hypotheses[:3]) if row.get("section_id") == gold_id]
    rank = min(positions) if positions else None
    return {"r1": int(rank == 1), "r3": int(rank is not None),
            "ndcg3": float(1 / np.log2(rank + 1)) if rank is not None else 0.0,
            "invalid1": int(not hypotheses or hypotheses[0].get("validity") != "valid")}


def lexical_rankings(corpus, queries):
    records = []
    for method in ("A", "B"):
        documents = [row["title"] + ("\n" + row["content"] if method == "A" else "") for row in corpus]
        tokens = [re.findall(r"\w+", document.casefold(), re.UNICODE) for document in documents]
        index = BM25Okapi(tokens, k1=1.5, b=0.75)
        for query in queries:
            started = time.monotonic()
            scores = index.get_scores(re.findall(r"\w+", query["query"].casefold(), re.UNICODE))
            order = sorted(range(len(corpus)), key=lambda i: (-float(scores[i]), f"finance:{i:03d}"))[:3]
            hypotheses = [{"rank": rank, "section_id": f"finance:{i:03d}",
                           "canonical_title": canonical(corpus[i]["title"]), "raw_text": corpus[i]["title"],
                           "validity": "valid", "score": float(scores[i])} for rank, i in enumerate(order, 1)]
            records.append({"method": method, "mode": "lexical", "query_id": query["query_id"],
                            "hypotheses": hypotheses, "elapsed_seconds": time.monotonic() - started})
    return records


def score_records(records, queries, corpus):
    gold = {row["query_id"]: row["gold_title"] for row in queries}
    ids = {canonical(row["title"]): f"finance:{i:03d}" for i, row in enumerate(corpus)}
    scored = []
    seen = set()
    for row in records:
        key = (row["method"], row["mode"], row["query_id"])
        if key in seen or row["query_id"] not in gold:
            raise ValueError("Duplicate or unadmitted inference record")
        seen.add(key)
        values = retrieval_values(row["hypotheses"], ids[gold[row["query_id"]]])
        scored.append({"method": row["method"], "mode": row["mode"], "query_id": row["query_id"],
                       "gold_title": gold[row["query_id"]], **values})
    return pd.DataFrame(scored)


def paired_result(frame, memberships):
    pair = frame.loc[(frame["method"].isin(["D", "F"])) & (frame["mode"] == "constrained")]
    wide = pair.pivot(index="query_id", columns="method", values="r1")
    if wide.isna().any().any() or len(wide) != len(memberships):
        raise ValueError("Primary paired contrast lacks matched outcomes")
    gold = pair.drop_duplicates("query_id").set_index("query_id")["gold_title"]
    delta = wide["F"] - wide["D"]
    clusters = pd.DataFrame({"gold": gold.loc[wide.index], "delta": delta}).groupby("gold")["delta"].agg(["sum", "count"])
    rng = np.random.default_rng(CFG.bootstrap_seed)
    sampled = rng.integers(0, len(clusters), size=(CFG.bootstrap_replicates, len(clusters)))
    estimates = clusters["sum"].to_numpy()[sampled].sum(axis=1) / clusters["count"].to_numpy()[sampled].sum(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {"difference_pp": float(delta.mean() * 100), "ci95_pp": [float(low * 100), float(high * 100)],
            "gains": int(((wide.F == 1) & (wide.D == 0)).sum()),
            "losses": int(((wide.F == 0) & (wide.D == 1)).sum()),
            "both_right": int(((wide.F == 1) & (wide.D == 1)).sum()),
            "both_wrong": int(((wide.F == 0) & (wide.D == 0)).sum()),
            "queries": len(wide), "document_clusters": len(clusters),
            "unstable_small_support": len(clusters) < 10,
            "interval": "95% percentile paired released-document-cluster bootstrap; query weighted; conditional on one book and seed"}


def compute_analysis(records, memberships, corpus, semantic_audit):
    queries = memberships["test"]
    frame = score_records(records, queries, corpus)
    expected = [("A", "lexical"), ("B", "lexical")] + [(method, mode) for method in "CDEF" for mode in ("constrained", "unconstrained")]
    for method, mode in expected:
        actual = frame.loc[(frame.method == method) & (frame["mode"] == mode)]
        if len(actual) != len(queries) or set(actual.query_id) != {row["query_id"] for row in queries}:
            raise ValueError(f"Incomplete measured comparison: {method}/{mode}")
    table = frame.groupby(["method", "mode"]).agg(queries=("r1", "size"), r1=("r1", "mean"),
                                                    r3=("r3", "mean"), ndcg3=("ndcg3", "mean"),
                                                    invalid1=("invalid1", "mean")).reset_index()
    paired = paired_result(frame, queries)
    flags = {row["query_id"] for row in semantic_audit["rows"] if row["split"] == "test" and row["status"] != "supported"}
    sensitivity = frame.loc[~frame.query_id.isin(flags)]
    sensitivity_table = sensitivity.groupby(["method", "mode"]).agg(queries=("r1", "size"), r1=("r1", "mean")).reset_index()
    support_counts = Counter(row["gold_title"] for row in memberships["train"])
    support = []
    for name, low, high in [("0", 0, 0), ("1–2", 1, 2), ("3–5", 3, 5), ("6+", 6, float("inf"))]:
        titles = [canonical(row["title"]) for row in corpus if low <= support_counts[canonical(row["title"])] <= high]
        for method in "DF":
            part = frame.loc[(frame.method == method) & (frame["mode"] == "constrained") & frame.gold_title.isin(titles)]
            support.append({"bin": name, "method": method, "candidate_documents": len(titles),
                            "evaluated_documents": part.gold_title.nunique(), "queries": len(part),
                            "query_weighted_r1": float(part.r1.mean()) if len(part) else None,
                            "macro_document_r1": float(part.groupby("gold_title").r1.mean().mean()) if len(part) else None})
    return {"summary": table.to_dict(orient="records"), "per_query": frame.to_dict(orient="records"),
            "primary": paired, "support_bins": support, "sensitivity": sensitivity_table.to_dict(orient="records"),
            "sensitivity_flagged_test_ids": sorted(flags & {row["query_id"] for row in queries})}
