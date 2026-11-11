def method_colors():
    return {"A": "#425563", "B": "#8C6BB1", "C": "#A2ABB2", "D": "#287C8E", "E": "#CF8B2B", "F": "#228B68"}


def publication_table(frame, title, widths=None, height=None):
    frame = frame.astype(str).copy()
    for i, column in enumerate(frame.columns):
        limit = 18 if i == 0 else 35
        frame[column] = frame[column].map(lambda value: "\n".join(textwrap.fill(part, limit) for part in value.splitlines()))
    fig, ax = plt.subplots(figsize=(CFG.img_dim1, height or CFG.img_dim2))
    ax.axis("off")
    ax.set_title(title, fontsize=15, weight="bold", pad=20)
    table = ax.table(cellText=frame.values, colLabels=list(frame.columns), loc="center", cellLoc="left", colLoc="left", colWidths=widths)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, col), cell in table.get_celld().items():
        values = frame.iloc[row - 1].tolist() if row else list(frame.columns)
        lines = max(len(str(value).splitlines()) for value in values)
        cell.set_height(.034 * lines + .026)
        cell.set_text_props(color="black", weight="bold" if row == 0 else "normal", va="center")
        cell.set_facecolor("#DDE7EC" if row == 0 else "#F0F5F7" if row % 2 else "white")
        cell.set_edgecolor("#C4CFD4")
    fig.tight_layout()
    return fig


def chosen_examples(result):
    frame = pd.DataFrame(result["analysis"]["per_query"])
    pair = frame.loc[(frame.method.isin(["D", "F"])) & (frame["mode"] == "constrained")].pivot(index="query_id", columns="method", values="r1")
    lookup = {(row["query_id"], row["method"], row["mode"]): row for row in result["records"]}
    queries = {row["query_id"]: row for row in result["memberships"]["test"]}
    cases = []
    for name, condition in [("Gain", (pair.F == 1) & (pair.D == 0)),
                            ("Loss", (pair.F == 0) & (pair.D == 1)),
                            ("Both wrong", (pair.F == 0) & (pair.D == 0))]:
        ids = sorted(pair.index[condition])
        if not ids:
            cases.append({"case": name, "query_id": None, "note": "No matching outcome observed"})
            continue
        query_id = ids[0]
        cases.append({"case": name, "query_id": query_id, "query": queries[query_id]["query"],
                      "gold_title": queries[query_id]["gold_title"],
                      "F": lookup[(query_id, "F", "constrained")]["hypotheses"][0]["raw_text"],
                      "D": lookup[(query_id, "D", "constrained")]["hypotheses"][0]["raw_text"]})
    invalid = sorted((row for row in result["records"] if row["mode"] == "unconstrained" and
                      row["hypotheses"][0]["validity"] != "valid"), key=lambda row: (row["query_id"], row["method"]))
    if invalid:
        row = invalid[0]
        cases.append({"case": "Invalid", "query_id": row["query_id"], "query": queries[row["query_id"]]["query"],
                      "gold_title": queries[row["query_id"]]["gold_title"], "method": row["method"],
                      "raw_output": row["hypotheses"][0]["raw_text"]})
    else:
        cases.append({"case": "Invalid", "query_id": None, "note": "No invalid unconstrained outcome observed"})
    return cases


def export_publication(result):
    """Derived exports only. Immutable inference and checkpoint files are not changed."""
    output = CFG.graph_folder
    output.mkdir(parents=True, exist_ok=True)
    colors = method_colors()
    audit = result["audit"]
    analysis = result["analysis"]
    summary = pd.DataFrame(analysis["summary"])
    assets = []
    def save(fig, number, prefix, caption, reading, data):
        name = f"{prefix}_{number}_{CFG.asset_suffix}.png"
        path = output / name
        fig.savefig(path, dpi=CFG.export_dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        json_write(path.with_suffix(".json"), data)
        if isinstance(data, list) and (not data or isinstance(data[0], dict)):
            pd.DataFrame(data).to_csv(path.with_suffix(".csv"), index=False)
        assets.append({"number": number, "filename": name, "source_run": result["run"].name,
                       "sha256": digest_bytes(path.read_bytes()), "caption": caption,
                       "reading": reading, "alt_text": caption})
    # 1: the historical failure remains the starting point of the revised study.
    fig, axes = plt.subplots(1, 2, figsize=(CFG.img_dim1, CFG.img_dim2), gridspec_kw={"width_ratios": [.8, 1.5]})
    axes[0].bar(["Released\nanchors", "Source\nleaves"], [audit["corpus_documents"], audit["source_leaf_count"]], color=[colors["D"], colors["C"]])
    axes[0].set_ylim(0, 120)
    axes[0].set_ylabel("Count")
    for i, value in enumerate([audit["corpus_documents"], audit["source_leaf_count"]]):
        axes[0].text(i, value + 2, str(value), ha="center", weight="bold")
    axes[1].axis("off")
    axes[1].text(.02, .88, "Released title: 1.3.1 The Journal", weight="bold", transform=axes[1].transAxes)
    axes[1].text(.02, .66, "Its body contains both:\n\n1.2 Net Income—A Critical Amount\n1.3.1 The Journal", transform=axes[1].transAxes,
                 bbox={"boxstyle": "round,pad=.7", "fc": "#F8E9D7", "ec": "#BB7A32"}, va="top")
    axes[1].text(.02, .20, "9 confirmed multi-leaf documents\n5 source/release numbering differences\nAll 86 document labels retained", transform=axes[1].transAxes)
    fig.suptitle("A document title anchors content; it does not bound it", fontsize=15)
    fig.tight_layout()
    save(fig, 1, "graph", "The original source-leaf assumption failed; v4 retains 86 released documents as verified title anchors.",
         "The tree represents source-derived anchor paths, not a partition of the bodies. The nine positive boundary cases remain evidence.",
         {"released_anchors": 86, "source_leaves": 99, "confirmed_multi_leaf": 9, "numbering_discrepancies": 5})
    # 2: access-to-information comparison.
    methods = pd.DataFrame([("A", "BM25 content", "Title + complete body", "None"), ("B", "BM25 titles", "All titles", "None"),
                            ("C", "Untrained ToC", "Hierarchy + query", "None"), ("D", "Trained no-ToC", "Book + query", "Query → title"),
                            ("E", "Trained flat", "All titles + query", "Query → title"), ("F", "Trained ToC", "Hierarchy + query", "Query → title")],
                           columns=["ID", "Method", "Inference input", "Local supervision"])
    save(publication_table(methods, "Six methods, one frozen candidate universe", [.06, .28, .37, .29]), 2, "post",
         "All methods retrieve the same unchanged 86 documents; C/D/E/F share a tokenizer and valid-label trie.",
         "F–D is the primary contrast. F–E adds ancestor semantics and organization; it does not isolate topology.", methods.to_dict(orient="records"))
    # 3: training adequacy and selection, not test-driven optimization.
    fig, axes = plt.subplots(1, 2, figsize=(CFG.img_dim1, CFG.img_dim2))
    curves = []
    for selection in result["selections"]:
        method = selection["method"]
        history = json_read(result["run"] / "training" / f"{method}.json")["history"]
        curves.extend(history)
        axes[0].plot([row["epoch"] for row in history], [row["train_loss"] for row in history], marker="o", color=colors[method], label=method)
        axes[1].plot([row["epoch"] for row in history], [100 * row["dev_r1"] for row in history], marker="o", color=colors[method], label=method)
        chosen = history[selection["selected_epoch"] - 1]
        axes[1].scatter(chosen["epoch"], 100 * chosen["dev_r1"], s=160, facecolors="none", edgecolors=colors[method], linewidths=2)
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Mean target-token loss per example")
    axes[1].set(title="Development Recall@1", xlabel="Epoch", ylabel="Correct labels (%)", ylim=(0, 100))
    for ax in axes:
        ax.legend(title="Method")
        ax.set_xticks(sorted({row["epoch"] for row in curves}))
    fig.suptitle("Matched maximum budgets; separate dev-selected checkpoints", fontsize=14)
    fig.tight_layout()
    save(fig, 3, "graph", "Training/dev curves for D/E/F. Rings indicate the earliest best development checkpoint within each arm.",
         "Lower loss is an optimization diagnostic. It does not establish full convergence, seed stability or held-out benefit.", curves)
    # 4: exact same held-out denominator in each method/mode.
    order = [("A", "lexical"), ("B", "lexical")] + [(method, mode) for method in "CDEF" for mode in ("constrained", "unconstrained")]
    ordered = pd.DataFrame([summary.loc[(summary.method == method) & (summary["mode"] == mode)].iloc[0] for method, mode in order])
    fig, axes = plt.subplots(1, 3, figsize=(CFG.img_dim1, CFG.img_dim2), sharey=True)
    y = np.arange(len(ordered))
    labels = [method + " · " + ("lexical" if mode == "lexical" else "constr." if mode == "constrained" else "unconstr.") for method, mode in order]
    for ax, metric, title in zip(axes, ["r1", "r3", "ndcg3"], ["Recall@1", "Recall@3", "nDCG@3"]):
        ax.barh(y, ordered[metric].to_numpy(), color=[colors[method] for method, _ in order], alpha=.9)
        ax.set(xlim=(0, 1), title=title)
        ax.set_yticks(y, labels)
        ax.set_xlabel("Score (0–1)")
    axes[0].invert_yaxis()
    fig.suptitle(f"Released-document label retrieval · {len(result['memberships']['test'])} held-out queries", fontsize=14)
    fig.tight_layout()
    save(fig, 4, "graph", "Held-out label metrics for all methods and both neural decode modes, with identical query membership.",
         "Invalid outputs keep their rank and receive zero relevance; constrained legality alone is not accuracy.", analysis["summary"])
    # 5: the single primary paired contrast and its descriptive interval.
    primary = analysis["primary"]
    fig, axes = plt.subplots(1, 2, figsize=(CFG.img_dim1, CFG.img_dim2), gridspec_kw={"width_ratios": [1.25, 1]})
    categories = ["Both right", "Both wrong", "F gains", "F losses"]
    values = [primary["both_right"], primary["both_wrong"], primary["gains"], primary["losses"]]
    axes[0].bar(categories, values, color=["#7F9D8B", colors["C"], colors["F"], "#B65E50"])
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].set(ylabel="Paired query count", title="F versus D, constrained")
    for i, value in enumerate(values):
        axes[0].text(i, value + .5, str(value), ha="center")
    low, high = primary["ci95_pp"]
    delta = primary["difference_pp"]
    axes[1].axvline(0, color="#888888", linewidth=1)
    axes[1].plot([low, high], [0, 0], color=colors["F"], linewidth=3)
    axes[1].scatter([delta], [0], s=100, color=colors["F"])
    span = max(abs(low), abs(high), 5) * 1.35
    axes[1].set(xlim=(-span, span), ylim=(-1, 1), yticks=[], xlabel="F − D Recall@1 (percentage points)", title="95% descriptive cluster interval")
    axes[1].text(.5, .30, f"{delta:+.2f} pp\n[{low:+.2f}, {high:+.2f}]", ha="center", transform=axes[1].transAxes)
    fig.suptitle("Count the gains and the losses", fontsize=15)
    fig.tight_layout()
    save(fig, 5, "graph", "Primary paired constrained F–D Recall@1 difference with a 95% percentile document-cluster bootstrap interval.",
         "The interval is conditional on this book, sample and seed. It is not a cross-book or training-seed uncertainty estimate.", primary)
    # 6: invalid labels versus enforced label legality.
    fig, axes = plt.subplots(1, 2, figsize=(CFG.img_dim1, CFG.img_dim2))
    invalid, correct = [], []
    for method in "CDEF":
        invalid.append(float(summary.loc[(summary.method == method) & (summary["mode"] == "unconstrained"), "invalid1"].iloc[0]))
        correct.append(float(summary.loc[(summary.method == method) & (summary["mode"] == "constrained"), "r1"].iloc[0]))
    axes[0].bar(list("CDEF"), np.array(invalid) * 100, color=[colors[m] for m in "CDEF"])
    axes[0].set(title="Unconstrained invalid top-1", ylabel="Queries (%)", ylim=(0, 100))
    axes[1].bar(list("CDEF"), np.array(correct) * 100, color=colors["F"], label="Correct valid label")
    axes[1].bar(list("CDEF"), (1 - np.array(correct)) * 100, bottom=np.array(correct) * 100, color=colors["C"], label="Wrong valid label")
    axes[1].set(title="Constrained: all labels legal", ylabel="Queries (%)", ylim=(0, 100))
    axes[1].legend(loc="upper center", bbox_to_anchor=(.5, -.10), fontsize=9, ncol=2)
    fig.suptitle("A valid title can still be wrong", fontsize=15)
    fig.tight_layout()
    save(fig, 6, "graph", "Unconstrained invalid-label rates and constrained correct/wrong-valid outcomes on the same queries.",
         "The trie enforces constrained validity; any reduction in unconstrained invalidity is a separate learned-adherence observation.", {"unconstrained_invalid": invalid, "constrained_correct": correct})
    # 7: both query-weighted and macro document summaries, with denominators.
    support = pd.DataFrame(analysis["support_bins"])
    bins = ["0", "1–2", "3–5", "6+"]
    fig, axes = plt.subplots(1, 2, figsize=(CFG.img_dim1, CFG.img_dim2))
    for ax, metric, title in zip(axes, ["query_weighted_r1", "macro_document_r1"], ["Query-weighted Recall@1", "Macro by evaluated document"]):
        for index, method in enumerate("DF"):
            part = support.loc[support.method == method].set_index("bin").loc[bins]
            values = [np.nan if pd.isna(v) else 100 * v for v in part[metric]]
            ax.bar(np.arange(4) + (index - .5) * .35, values, width=.35, color=colors[method], label=method)
        ax.set(xticks=np.arange(4), xticklabels=bins, ylim=(0, 100), ylabel="Correct labels (%)", title=title, xlabel="Training examples per document")
        ax.legend()
    counts = support.loc[support.method == "F"]
    fig.suptitle("Rarely trained document labels · descriptive support bins", fontsize=14)
    footer = "; ".join(f"{row['bin']}: q={row['queries']}, evaluated docs={row['evaluated_documents']}, candidates={row['candidate_documents']}" for _, row in counts.iterrows())
    fig.text(.5, -.02, textwrap.fill(footer, 110), ha="center", fontsize=8, va="top")
    fig.tight_layout()
    save(fig, 7, "graph", "Training-support bins for constrained D/F, including query and document denominators; empty bins are not zero accuracy.",
         "Support is observational. Zero-training labels still occur in the complete catalogue and are not unseen-book targets.", analysis["support_bins"])
    # 8: deterministic examples with complete strings in sidecars.
    examples = chosen_examples(result)
    def short(text, limit):
        return text if len(text) <= limit else text[:limit - 1] + "…"
    display_rows = []
    for row in examples:
        values = ["Gold: " + short(row.get("gold_title", "—"), 55)]
        if row["case"] == "Invalid" and row.get("method"):
            values.append(row["method"] + " unconstrained: " + short(row["raw_output"], 55))
        else:
            values.extend(["F: " + short(row.get("F", "—"), 55), "D: " + short(row.get("D", "—"), 55)])
        display_rows.append({"Case": row["case"], "Query": short(row.get("query", row.get("note", "")), 95),
                             "Gold / outputs": "\n".join(values)})
    frame = pd.DataFrame(display_rows)
    save(publication_table(frame, "Fixed-rule examples: gains, losses and invalid labels", [.16, .39, .45], height=CFG.example_figure_height), 8, "post",
         "Lowest-qualified-ID gain, loss and both-wrong cases; lowest-ID unconstrained invalid output, with method-ID tie breaking.",
         "Display excerpts are shortened for layout only. Complete questions, target strings, outputs and IDs are in the JSON/CSV sidecars.", examples)
    # 9: recorded timing coverage and sampled shared-memory counters.
    times = json_read(result["run"] / "execution_time.json")
    memory = json_read(result["run"] / "memory_samples.json")["samples"]
    pilot = times["pilot_seconds"]
    training = times["training_and_development_seconds"]
    remainder = max(0, times["automated_seconds_before_exports"] - pilot - training)
    fig, axes = plt.subplots(1, 2, figsize=(CFG.img_dim1, CFG.img_dim2))
    axes[0].bar(["Pilot", "Train + dev\n+ checks", "Inference +\nanalysis + load"], np.array([pilot, training, remainder]) / 60, color=[colors["C"], colors["D"], colors["F"]])
    axes[0].set(ylabel="Wall minutes", title="Recorded automated stages")
    rss = max(row["rss_bytes"] for row in memory) / 2**30
    available = min(row["available_bytes"] for row in memory) / 2**30
    axes[1].bar(["Peak sampled\nprocess RSS", "Minimum sampled\navailable RAM"], [rss, available], color=[colors["E"], colors["C"]])
    axes[1].set(ylabel="GiB", title="Shared-memory counters, separately")
    for i, value in enumerate([rss, available]):
        axes[1].text(i, value + 1, f"{value:.1f}", ha="center")
    fig.suptitle(f"Executed backend: {result['config']['device']} float32 · input lengths and full timings in evidence", fontsize=12)
    fig.tight_layout()
    save(fig, 9, "graph", "Recorded automated-stage times and sampled host-memory counters; model fitting and held-out inference are distinct costs.",
         "Preparation/probes/authoring are additional and not fully metered here. Replay is not fresh inference speed. Shared counters are not additive VRAM pools.", {"timing": times, "peak_sampled_rss_gib": rss, "min_sampled_available_gib": available})
    # 10: paper/local correspondence and explicit remaining limitations.
    limits = pd.DataFrame([("Retrieval unit", "86 unchanged released document labels; source-heading anchors may have multi-section bodies."),
                           ("Original failure", "Nine merged-boundary cases retained; original v3 stop and source excerpts preserved."),
                           ("Backbone", "Full FLAN-T5-small tuning; not Mistral/LoRA replication."),
                           ("Scope", "One book, one seed; generated questions; paragraph independence unverified."),
                           ("Uncertainty", "Conditional document-cluster interval; no cross-book or seed claim."),
                           ("Input length", "Complete ToC prompts reach 1,020 tokens; tokenizer metadata says 512. Runtime admission is not a long-context quality guarantee."),
                           ("Hardware", f"{result['config']['device']} float32; strict MPS training admission failed on deterministic embedding backward.")],
                          columns=["Item", "Local scope / limit"])
    save(publication_table(limits, "What the local result can establish", [.22, .78]), 10, "post",
         "Paper/local differences, the original source-boundary failure and unresolved generalization/runtime limits.",
         "Any positive result remains a local released-label finding. The decision changed the interpretation; it did not clean or repartition the bodies.", limits.to_dict(orient="records"))
    json_write(output / "asset_manifest.json", {"source_run": result["run"].name, "assets": assets})
    return assets


def show_visual(assets, number):
    row = next(item for item in assets if item["number"] == number)
    display(Image(filename=str(CFG.graph_folder / row["filename"])))
    display(Markdown("*Caption:* " + row["caption"] + "\n\n" + row["reading"]))


def result_reading(result):
    table = pd.DataFrame(result["analysis"]["summary"])
    def score(method, mode):
        return float(table.loc[(table.method == method) & (table["mode"] == mode), "r1"].iloc[0]) * 100
    f, d, e, a = score("F", "constrained"), score("D", "constrained"), score("E", "constrained"), score("A", "lexical")
    return (f"**Observed outcome:** constrained hierarchy F={f:.2f}%, no-ToC D={d:.2f}%, flat catalogue E={e:.2f}% and BM25 content A={a:.2f}% Recall@1. "
            f"F–D={f-d:+.2f} pp; descriptive F–E={f-e:+.2f} pp. A gain over no-ToC alone does not establish an added benefit from hierarchy over a catalogue. "
            "These are released-document label scores on the frozen local sample, not exact source-section or downstream-answer quality.")


def write_current_report(result, assets):
    analysis = result["analysis"]
    primary = analysis["primary"]
    summary = pd.DataFrame(analysis["summary"])
    def r1(method, mode):
        return float(summary.loc[(summary.method == method) & (summary["mode"] == mode), "r1"].iloc[0]) * 100
    low, high = primary["ci95_pp"]
    timing = json_read(result["run"] / "execution_time.json")
    rows = "\n".join(f"| {row['method']} | {row['mode']} | {row['queries']} | {100*row['r1']:.2f} | {100*row['r3']:.2f} | {row['ndcg3']:.4f} | {100*row['invalid1']:.2f} |" for row in analysis["summary"])
    sensitivity_rows = "\n".join(f"| {row['method']} | {row['mode']} | {row['queries']} | {100*row['r1']:.2f} |" for row in analysis["sensitivity"])
    flat_difference = r1("F", "constrained") - r1("E", "constrained")
    content_difference = r1("F", "constrained") - r1("A", "lexical")
    report = f"""# STAIR v4 build and experiment report

Source run: `{result['run'].name}`. Numerical study: complete. Notebook report replay and publication checks have separate receipts; they are not new inference measurements.

## The revision and its preserved failure

The original source-leaf contract stopped correctly: at least nine of the 86 released documents contain multiple source-leaf headings. The user approved treating unchanged released titles as document anchors. All 86 candidates, raw bodies and supplied labels remain intact; their source paths describe anchors, not content boundaries. The original 99-leaf source ToC, duplicate printed 6.4, five numbering differences, exact excerpts and v3 failure run remain in the evidence. See [the durable decision record](docs/stair-section-boundary-decision.md). This is a changed interpretation, not repaired segmentation.

## What ran

One pinned FLAN-T5-small checkpoint, three full-fine-tuned arms D/E/F, untrained C, and two BM25 arms A/B. Profile: `{result['config']['profile']}`; train/dev/test = {len(result['memberships']['train'])}/{len(result['memberships']['dev'])}/{len(result['memberships']['test'])}. Effective training batch = 8; microbatch = {result['config']['microbatch']}; inference batch = {result['config']['inference_batch']}. Each trained arm completed the same maximum update budget, then selected its earliest best development checkpoint. Selected updates can differ.

The executed backend was **{result['config']['device']} float32**, with strict deterministic algorithms. MPS inference was available, but deterministic MPS embedding backward failed in PyTorch 2.15.0; the planned whole-run CPU fallback passed its own pilot. No silent unsupported-operation CPU fallback or mid-comparison backend switch was used.

The frozen sample followed exact/near-duplicate component filtering and hash order. The semantic audit inspected 20 selected query/body pairs per split before fitting; flags remain in the primary denominator. The sensitivity analysis excludes only flagged test members from that same selected set. Answers were not model inputs or targets. No generated question replacement, dataset substitution or test-driven tuning occurred.

## Captured held-out results

| Method | Mode | Queries | R@1 (%) | R@3 (%) | nDCG@3 | Invalid top-1 (%) |
|---|---|---:|---:|---:|---:|---:|
{rows}

Primary constrained F–D Recall@1 difference: **{primary['difference_pp']:+.2f} percentage points**; descriptive 95% document-cluster percentile interval **[{low:+.2f}, {high:+.2f}] pp**. Paired outcomes: {primary['gains']} gains, {primary['losses']} losses, {primary['both_right']} both right and {primary['both_wrong']} both wrong. Document clusters: {primary['document_clusters']}.

F constrained R@1 = {r1('F','constrained'):.2f}%; D = {r1('D','constrained'):.2f}%; flat catalogue E = {r1('E','constrained'):.2f}%; BM25 content A = {r1('A','lexical'):.2f}%. The comparison with E concerns ancestor semantics plus organization, not topology alone. BM25 sees source bodies while the neural prompts do not. None of these local scores reproduces the paper's Mistral result.

The descriptive F–E difference is **{flat_difference:+.2f} pp**, and F–A is **{content_difference:+.2f} pp**. A positive F–D contrast alone cannot establish added value from hierarchy over a catalogue. These unfavorable or null comparisons must remain visible in the notebook and companion outline.

### Predeclared annotation sensitivity

Exclude only preflagged audited test members, without replacement: {', '.join(analysis['sensitivity_flagged_test_ids'])}. The remaining unaudited questions are not certified correct by this sensitivity check. Primary results above retain all selected queries.

| Method | Mode | Sensitivity queries | R@1 (%) |
|---|---|---:|---:|
{sensitivity_rows}

## Cost and limits

Recorded automated time before exports: {timing['automated_seconds_before_exports']/60:.2f} minutes. CPU pilot: {timing['pilot_seconds']/60:.2f} minutes; training/development/checkpoint diagnostics: {timing['training_and_development_seconds']/60:.2f} minutes. Model/data preparation, environment installation, earlier development probes, manual audit and notebook authoring are additional project costs and are not fully time-metered here. Batch throughput does not equal singleton latency. Replay recomputes evidence; it does not establish new model speed.

The corpus is one book, the training uses one seed, queries are externally generated, and source-paragraph independence is unverified. The bootstrap interval is conditional on this run's released-document clusters; it does not cover new books, seeds or independent paragraphs. Tokenizer metadata has a 512-token limit while selected complete hierarchy prompts reach 1,020 tokens; runtime compatibility is not a guarantee of long-context learning quality. Mixed-content documents and preflagged ambiguous questions remain visible. Correct label retrieval does not establish exact source-leaf boundaries or downstream answer correctness.

## Reproduction and artifacts

The notebook contains all study code inline. Cached corpus/source files and immutable run evidence are separate. `report` validates hashes, recomputes metrics and exports all ten figures without fitting or model inference. Checkpoints are retained locally under the run but are optional for report replay. The full training environment is pinned in `requirements-stair.lock.txt`. See [local instructions](docs/stair-local-demo.md).

All ten assets have captions, readings, source-run identity and data sidecars in `graphs/stair/`. A companion outline is in `stair-post-plan.md`; it is a local draft, not a published article or promotional message. The original failed source-leaf notebook and bundle remain under `runs/stair/history/v3-source-leaf-contract/`.

Attribution: SearchTome at revision `{CFG.dataset_revision}`, CC BY-SA 4.0. Christine Jonick, *Principles of Financial Accounting*, University of North Georgia Press (2017), attributing the University System of Georgia; retain copyright-page exceptions. FLAN-T5-small revision `{result['config']['model_revision']}`, Apache-2.0 model card.
"""
    (CFG.root / "stair-build-report.md").write_text(report, encoding="utf-8")
    by_number = {row["number"]: row for row in assets}
    placements = {"One book and a map of its sections": [1], "Start by searching the text": [2],
                  "Learn to return a section title": [3], "Give the model the table of contents": [4],
                  "A valid title can still be wrong": [], "Count the gains and the losses": [5, 6],
                  "What happens to rarely trained sections": [7], "Cost, failures and what remains uncertain": [8, 9, 10]}
    outline = ["# Does giving a retriever the contents page help it find the right document?",
               f"Status: local evidence-based post outline; source run `{result['run'].name}`. Editor review is recorded separately; no independent-agent approval or publication is implied.",
               "Open with the practical problem, then the first build's failed one-leaf-per-document assumption and the approved anchor interpretation. Introduce STAIR and distinguish the local small-model adaptation from its benchmark.",
               "Companion notebook publication URL: pending. Keep the local notebook link until a real public destination is verified.",
               f"Evidence to report: F–D {primary['difference_pp']:+.2f} pp, conditional interval [{low:+.2f}, {high:+.2f}] pp; F={r1('F','constrained'):.2f}%, D={r1('D','constrained'):.2f}%, E={r1('E','constrained'):.2f}%, A={r1('A','lexical'):.2f}% Recall@1. Descriptive F–E={flat_difference:+.2f} pp and F–A={content_difference:+.2f} pp. Preserve these comparisons, exact denominators and access differences. A positive F–D alone is not evidence that hierarchy adds value over a catalogue."]
    for heading, numbers in placements.items():
        outline.append("## " + heading)
        if not numbers:
            outline.append("Explain legal label strings, wrong-but-valid retrieval and broader document bodies; avoid factual-hallucination language. The numerical validity breakdown follows with the paired outcome counts.")
        for number in numbers:
            row = by_number[number]
            outline.extend([f"> [Insert Visual {number} here — {row['caption']}]",
                            f"![{row['alt_text']}](graphs/stair/{row['filename']})",
                            "*Caption:* " + row["caption"], "*Reading:* " + row["reading"]])
    outline.extend(["## Closing time", "State the observed gain/loss pattern and the flat/BM25 comparisons without claiming that hierarchy must win. Retain the original section-boundary failure as the reason for the anchor contract, and the CPU fallback as a separate reproducibility detail. Make no cross-book, seed-stability, serving-speed or downstream-answer claim. Article, SEO/LinkedIn drafts and publication are later work."])
    (CFG.root / "stair-post-plan.md").write_text("\n\n".join(outline) + "\n", encoding="utf-8")
