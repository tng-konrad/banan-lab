"""Weight-free report helpers, copied inline into the current STAIR notebook.

The scoring functions retain the frozen pilot's definitions. No training or
generation framework is imported by this module.
"""
from collections import Counter
import io
import hashlib
import json
import math
from pathlib import Path
import textwrap
import unicodedata

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import HTML, Image, Markdown, display


def canonical(value):
    return " ".join(unicodedata.normalize("NFC", value).split())


def score(records, queries, corpus):
    ids = {canonical(row["title"]): f"finance:{index:03d}" for index, row in enumerate(corpus)}
    gold = {row["query_id"]: row["gold_title"] for row in queries}
    if len(records) != len(gold) or {row["query_id"] for row in records} != set(gold):
        raise ValueError("Missing/duplicate/mismatched raw predictions")
    result = {}
    for record in records:
        target = ids[gold[record["query_id"]]]
        positions = [rank for rank, hypothesis in enumerate(record["hypotheses"][:3], 1)
                     if hypothesis.get("section_id") == target]
        rank = min(positions) if positions else None
        first = record["hypotheses"][0] if record["hypotheses"] else {}
        result[record["query_id"]] = {
            "gold_title": gold[record["query_id"]], "r1": int(rank == 1),
            "r3": int(rank is not None), "ndcg3": 1 / math.log2(rank + 1) if rank is not None else 0.,
            "invalid1": int(first.get("validity") != "valid")}
    return result


def summary(scored):
    return {"queries": len(scored), **{key: float(np.mean([row[key] for row in scored.values()]))
            for key in ["r1", "r3", "ndcg3", "invalid1"]}}


def query_seconds(records):
    if "timing" not in records[0]:
        return sum(row["elapsed_seconds"] for row in records) / len(records)
    elapsed, index = 0., 0
    while index < len(records):
        timing = records[index]["timing"]
        elapsed += timing["end_to_end_seconds"]
        index += timing["batch_size"]
    return elapsed / len(records)


def clustered(delta, gold):
    labels = sorted(set(gold))
    gold = np.array(gold)
    delta = np.array(delta, dtype=float)
    sums = np.array([delta[gold == label].sum() for label in labels])
    counts = np.array([(gold == label).sum() for label in labels])
    sample = np.random.default_rng(42001).integers(0, len(labels), (10000, len(labels)))
    estimate = sums[sample].sum(1) / counts[sample].sum(1)
    lo, hi = np.quantile(estimate, [.025, .975]) * 100
    return {"difference_pp": float(delta.mean() * 100), "ci95_pp": [float(lo), float(hi)],
            "queries": len(delta), "document_clusters": len(labels), "unstable_small_support": len(labels) < 10,
            "interval_scope": "exploratory paired released-document-cluster interval, one book/seed"}


def compare(left, right):
    if set(left) != set(right):
        raise ValueError("Pairing changed")
    ids = sorted(left)
    a = np.array([left[key]["r1"] for key in ids])
    b = np.array([right[key]["r1"] for key in ids])
    return clustered(b - a, [left[key]["gold_title"] for key in ids]) | {
        "gains": int(((a == 0) & (b == 1)).sum()), "losses": int(((a == 1) & (b == 0)).sum()),
        "both_right": int(((a == 1) & (b == 1)).sum()), "both_wrong": int(((a == 0) & (b == 0)).sum())}


def load_evidence(root, runs, seal_hashes):
    """Verify consumed bytes and recompute all captured metrics and paired effects."""
    root = Path(root).resolve()
    seals, consumed, objects = {}, {}, {}
    for name, relative in runs.items():
        path = root / relative / "pilot_manifest.json"
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != seal_hashes[name]:
            raise ValueError(f"Pinned {name} seal changed: {path}")
        seals[name] = json.loads(raw)
        if seals[name]["validation"]["status"] != "passed":
            raise ValueError("Source experiment is not validated")

    def checked(name, relative, as_bytes=False):
        entry = seals[name]["files"][relative]
        path = root / runs[name] / relative
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError(f"Changed evidence: {path}")
        consumed[str(Path(runs[name]) / relative)] = entry["sha256"]
        if as_bytes:
            return raw
        if relative.endswith(".jsonl"):
            return [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
        return json.loads(raw)

    data = checked("4b", "inputs.json")
    if data != checked("2b", "inputs.json"):
        raise ValueError("2B and 4B cohort membership differs")
    if [len(data[k]) for k in ["train935", "tune", "screen", "corpus"]] != [935, 48, 127, 86]:
        raise ValueError("Unexpected study dimensions")
    train_ids, tune_ids, screen_ids = [set(r["query_id"] for r in data[k]) for k in ["train935", "tune", "screen"]]
    if train_ids & tune_ids or train_ids & screen_ids or tune_ids & screen_ids:
        raise ValueError("Train/tune/screen membership overlaps")
    reserve = checked("4b", "confirmation_reserve.json")
    reserve_ids = set(reserve["ids"])
    if len(reserve_ids) != 240 or reserve["model_predictions"] != 0:
        raise ValueError("Future confirmation boundary changed")
    if (train_ids | tune_ids | screen_ids) & reserve_ids:
        raise ValueError("Reserve membership overlaps the reported study")
    audit = checked("4b", "source_audit.json")
    if audit["confirmed_multi_leaf_documents"] != 9:
        raise ValueError("Source-boundary interpretation changed")
    flags = {r["query_id"] for r in checked("4b", "screen_semantic_audit.json")["rows"] if r["status"] != "supported"}
    if len(flags) != 3:
        raise ValueError("Annotation sensitivity changed")
    captured = checked("4b", "quality/analysis.json")
    definitions = {"qwen4b": checked("4b", "quality/cells.json"),
                   "qwen2b": checked("2b", "quality/cells.json"),
                   "flan": checked("4b", "flan_baseline/result.json")["cells"]}
    frames, recomputed, prediction_rows = {}, {}, 0
    for family, cells in definitions.items():
        name = "2b" if family == "qwen2b" else "4b"
        folder = "flan_baseline" if family == "flan" else "quality"
        for cell in cells:
            for mode in ["constrained", "unconstrained"]:
                rows = checked(name, f"{folder}/predictions/{cell}/{mode}.jsonl")
                if any(r["query_id"] in reserve_ids for r in rows):
                    raise ValueError("Reserve prediction detected")
                for row in rows:
                    for hypothesis in row["hypotheses"]:
                        if len(hypothesis["token_ids"]) != len(hypothesis["token_logprobs"]):
                            raise ValueError("Token-score lengths differ")
                        if not np.isclose(np.mean(hypothesis["token_logprobs"]), hypothesis["score"], atol=1e-3, rtol=1e-3):
                            raise ValueError("Mean beam score reconstruction failed")
                key = (family, cell, mode)
                frames[key] = score(rows, data["screen"], data["corpus"])
                recomputed[key] = summary(frames[key]) | {"query_seconds": query_seconds(rows)}
                prediction_rows += len(rows)
    lexical = checked("4b", "flan_baseline/lexical/lexical.jsonl")
    for method in "AB":
        rows = [r for r in lexical if r["method"] == method]
        key = ("lexical", method, "lexical")
        frames[key] = score(rows, data["screen"], data["corpus"])
        recomputed[key] = summary(frames[key]) | {"query_seconds": query_seconds(rows)}
        prediction_rows += len(rows)
    if len(recomputed) != len(captured["metrics"]):
        raise ValueError("Missing metric rows")
    for row in captured["metrics"]:
        key = (row["family"], row["cell"], row["mode"])
        for field, value in recomputed[key].items():
            if value != row[field]:
                raise ValueError(f"Metric differs from sealed report: {key}, {field}")
        sensitivity = summary({qid: r for qid, r in frames[key].items() if qid not in flags})
        if key[0] != "lexical" and sensitivity["r1"] != row["sensitivity_r1"]:
            raise ValueError("Sensitivity metric differs")
    for effect in captured["effects"]:
        left, right = [frames[tuple(effect[side])] for side in ["left", "right"]]
        paired = compare(left, right)
        for field, value in paired.items():
            if value != effect[field]:
                raise ValueError(f"Paired effect differs: {effect['contrast']}, {field}")
        ids = sorted(left)
        invalid = clustered([right[i]["invalid1"] - left[i]["invalid1"] for i in ids], [left[i]["gold_title"] for i in ids])
        sensitivity = compare({i: r for i, r in left.items() if i not in flags}, {i: r for i, r in right.items() if i not in flags})
        if (invalid["difference_pp"] != effect["invalid_change_pp"] or invalid["ci95_pp"] != effect["invalid_ci95_pp"]
                or sensitivity["difference_pp"] != effect["sensitivity_difference_pp"]):
            raise ValueError("Effect validity/sensitivity differs")
    costs, selections, contracts, models = [], [], {}, {}
    for name, family in [("2b", "qwen2b"), ("4b", "qwen4b")]:
        contracts[family] = checked(name, "quality/execution_contract.json")
        models[family] = checked(name, "quality/model_loading.json")
        for trajectory in checked(name, "quality/trajectories.json"):
            fmt, history = trajectory["method"], trajectory["history"]
            if [r["update"] for r in history] != list(range(32, 385, 32)):
                raise ValueError("Incomplete tuning trajectory")
            for budget in [128, 384]:
                selected = max((r for r in history if r["update"] <= budget), key=lambda r: (r["tune_r1"], -r["update"]))
                definition = definitions[family][f"n935_u{budget}_{fmt}"]
                seconds = sum(sum(r[k] for k in ["training_seconds", "tuning_seconds", "checkpoint_seconds"]) for r in history if r["update"] <= budget)
                if selected != trajectory["selected"][str(budget)] or definition["selected_update"] != selected["update"]:
                    raise ValueError("Checkpoint selection changed")
                if definition["training_seconds"] != seconds:
                    raise ValueError("Recorded search cost changed")
                selections.append({"Model": family, "Method": fmt, "Search budget": budget,
                                   "Selected update": selected["update"], "Tuning R@1 (%)": 100 * selected["tune_r1"],
                                   "Fit/tune/save (h)": seconds / 3600})
            costs.append({"Model": family, "Method": fmt, "Fit/tune/save (h)": seconds / 3600,
                          "Constrained s/query": recomputed[(family, f"n935_u384_{fmt}", "constrained")]["query_seconds"]})
    for name in ["comparison.png", "tuning.png"]:
        objects[name] = checked("4b", "quality/figures/" + name, as_bytes=True)
    evidence = {"analysis": captured, "frames": frames, "metrics": recomputed, "data": data,
                "flags": flags, "audit": audit, "costs": costs, "selections": selections,
                "contracts": contracts, "models": models, "figures": objects,
                "validation": checked("4b", "quality/validation.json"),
                "stages": checked("4b", "stage_reports.json"),
                "consumed_files": consumed, "source_seals": seal_hashes,
                "proof": {"metric_rows_recomputed": len(recomputed), "paired_effects_recomputed": len(captured["effects"]),
                          "raw_prediction_rows_read": prediction_rows, "reserve_questions": 240,
                          "new_training_updates": 0, "new_model_inference_calls": 0}}
    return evidence


def start_publication(folder, plan, width=14, dpi=180):
    global _ASSETS
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    _ASSETS = {"folder": folder, "plan": plan, "width": width, "dpi": dpi, "assets": []}


def publish_png(data, kind, rows=None, source_sha256=None):
    state = _ASSETS
    number = len(state["assets"]) + 1
    expected_kind, title = state["plan"][number - 1]
    if kind != expected_kind:
        raise ValueError("Notebook visual display order changed")
    filename = f"stair_{kind}_{number:02d}.png"
    path = state["folder"] / filename
    path.write_bytes(data)
    entry = {"number": number, "kind": kind, "title": title, "file": filename,
             "sha256": hashlib.sha256(data).hexdigest()}
    if rows is not None:
        sidecar = path.with_suffix(".json")
        sidecar.write_text(json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        pd.DataFrame(rows).to_csv(path.with_suffix(".csv"), index=False)
        entry["data_file"] = sidecar.name
    if source_sha256 is not None:
        entry["saved_source_sha256"] = source_sha256
    state["assets"].append(entry)
    display(Image(data=data))
    display(Markdown(f"**{kind.title()} {number:02d} — {title}.** Export: `graphs/{filename}`."))


def table(rows, decimals=2):
    frame = pd.DataFrame(rows)
    number = len(_ASSETS["assets"]) + 1
    title = _ASSETS["plan"][number - 1][1]
    columns = list(frame.columns)
    def formatted(value):
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{value:.{decimals}f}"
        if isinstance(value, (int, np.integer)):
            return f"{value:,}" if abs(value) >= 10000 else str(value)
        return str(value)
    raw = [[formatted(v) for v in row] for row in frame.itertuples(index=False, name=None)]
    cap = 46 if len(columns) <= 3 else 26
    weights = [min(cap, max(len(str(c)), max((len(r[i]) for r in raw), default=8))) for i, c in enumerate(columns)]
    weights = np.array([max(10, w) for w in weights], dtype=float)
    widths = weights / weights.sum()
    wraps = [max(10, int(w * _ASSETS["width"] * 11)) for w in widths]
    headers = [textwrap.fill(str(c), width=w) for c, w in zip(columns, wraps)]
    content = [[textwrap.fill(v, width=w, break_long_words=False) for v, w in zip(row, wraps)] for row in raw]
    header_lines = max(s.count("\n") + 1 for s in headers)
    row_lines = [max(s.count("\n") + 1 for s in row) for row in content]
    heights = [.25 * header_lines + .14] + [.24 * n + .13 for n in row_lines]
    height = sum(heights) + .2
    fig, ax = plt.subplots(figsize=(_ASSETS["width"], height))
    ax.axis("off")
    grid = ax.table(cellText=content, colLabels=headers, colWidths=widths,
                    cellLoc="left", colLoc="left", loc="center")
    grid.auto_set_font_size(False)
    grid.set_fontsize(10)
    for (r, c), cell in grid.get_celld().items():
        cell.set_height(heights[r] / sum(heights))
        cell.set_text_props(color="black", weight="bold" if r == 0 else "normal")
        cell.set_edgecolor("#D4DADF")
        cell.set_facecolor("#E6EDF1" if r == 0 else ("#FFFFFF" if r % 2 else "#F5F7F9"))
        cell.PAD = .08
    fig.tight_layout(pad=0.4)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=_ASSETS["dpi"], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    publish_png(buffer.getvalue(), "table", frame.to_dict(orient="records"))


def finish_publication(evidence):
    if len(_ASSETS["assets"]) != len(_ASSETS["plan"]):
        raise ValueError("Not all notebook visuals were exported")
    manifest = {"notebook": "stair_model_comparison.ipynb", "source_seals": evidence["source_seals"],
                "numbering": "one shared display-order sequence for tables and graphs",
                "new_model_calls": 0, "assets": _ASSETS["assets"]}
    path = _ASSETS["folder"] / "stair_assets.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Exported {len(manifest['assets'])} numbered PNGs to {path.parent}")


def resource_table(evidence):
    labels = {"admission": "Admission", "gradientcheck": "Gradient checks", "quality": "Quality run", "flan_baseline": "Matched FLAN control"}
    table([{"Stage": labels[name], "Peak RSS (GiB)": row["peak_rss_gib"],
            "Peak MPS driver (GiB)": row["peak_mps_driver_gib"],
            "Minimum available (GiB)": row["minimum_available_gib"],
            "New swap (GiB)": row["swap_growth_gib"]}
           for name, row in evidence["analysis"]["resources"].items()])


def recovery_table(evidence):
    r = evidence["analysis"]["recovery"]
    table([{"Measurement": label, "Captured value": value} for label, value in [
        ("Quality worker time, both attempts (h)", f"{r['quality_worker_attempt_seconds']/3600:.2f}"),
        ("Quality wall elapsed (h)", f"{r['quality_elapsed_seconds']/3600:.2f}"),
        ("Time outside quality workers (h)", f"{r['time_outside_quality_workers_seconds']/3600:.2f}"),
        ("Discarded updates replayed exactly", r["discarded_updates_replayed"]),
        ("Minimum host-headroom guard (GiB)", r["available_memory_floor_gib"]),
        ("Original quality clock retained", "Yes" if r["original_quality_clock_retained"] else "No")]])


def accuracy_table(evidence):
    lookup = {(r["family"], r["cell"], r["mode"]): r for r in evidence["analysis"]["metrics"]}
    rows = []
    for budget in [0, 128, 384]:
        for method in ("C" if budget == 0 else "DEF"):
            cell = "C" if budget == 0 else f"n935_u{budget}_{method}"
            values = [100 * lookup[(f, cell, "constrained")]["r1"] for f in ["flan", "qwen2b", "qwen4b"]]
            rows.append(dict(zip(["Search budget", "Method", "FLAN R@1 (%)", "2B R@1 (%)", "4B R@1 (%)", "4B minus 2B (pp)"],
                                 [budget, method, *values, values[2] - values[1]])))
    table(rows)


def effect_table(evidence):
    wanted = {"4B_minus_2B_n935_u384_E", "E_minus_D_384", "F_minus_D_384", "F_minus_E_384"}
    labels = {"4B_minus_2B_n935_u384_E": "4B flat minus 2B flat", "E_minus_D_384": "E minus D (4B)",
              "F_minus_D_384": "F minus D (4B)", "F_minus_E_384": "F minus E (4B)"}
    rows = [{"Contrast": labels[r["contrast"]], "Change (pp)": r["difference_pp"],
             "95% cluster interval (pp)": f"[{r['ci95_pp'][0]:+.2f}, {r['ci95_pp'][1]:+.2f}]",
             "Gains": r["gains"], "Losses": r["losses"],
             "124-query sensitivity (pp)": r["sensitivity_difference_pp"]}
            for r in evidence["analysis"]["effects"] if r["contrast"] in wanted and r["right"][2] == "constrained"]
    table(rows)


def rank_table(evidence):
    table([{"Model": r["family"], "Method": r["method"], "R@1 (%)": 100*r["r1"],
            "R@3 (%)": 100*r["r3"], "nDCG@3": r["ndcg3"]}
           for r in evidence["analysis"]["metrics"] if r["mode"] == "constrained" and r["budget"] == 384], 4)


def validity_table(evidence):
    table([{"Model": r["family"], "Method": r["method"], "Search budget": r["budget"],
            "Unconstrained R@1 (%)": 100*r["r1"], "Invalid top-1 (%)": 100*r["invalid1"]}
           for r in evidence["analysis"]["metrics"] if r["mode"] == "unconstrained"])


def tuning_figure(evidence):
    colors = {"flan": "tab:blue", "qwen2b": "tab:orange", "qwen4b": "tab:green"}
    a = evidence["analysis"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained", sharey=True)
    for ax, fmt in zip(axes, "DEF"):
        for family in colors:
            t = next(t for t in a["training_traces"][family] if t["method"] == fmt)
            h = t["history"]
            best = max(h, key=lambda r: (r["tune_r1"], -r["update"]))
            ax.plot([r["update"] for r in h], [100 * r["tune_r1"] for r in h], color=colors[family], label=family.upper(), marker="o", markersize=3)
            ax.scatter([best["update"]], [100 * best["tune_r1"]], color=colors[family], marker="*", s=130, zorder=4)
        ax.axvline(128, color="gray", linestyle="--", alpha=.5)
        ax.set_ylim(0, 100)
        ax.set_xlabel(f"Optimizer updates ({fmt})")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Tuning Recall@1 (%)")
    axes[0].legend(fontsize=8)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=_ASSETS["dpi"], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    publish_png(buffer.getvalue(), "graph")


def comparison_figure(evidence):
    colors = {"flan": "tab:blue", "qwen2b": "tab:orange", "qwen4b": "tab:green"}
    a = evidence["analysis"]
    lookup = {(r["family"], r["cell"], r["mode"]): r for r in a["metrics"]}
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), layout="constrained")
    for col, budget in enumerate([128, 384]):
        for i, family in enumerate(colors):
            keys = ["C"] + [f"n935_u{budget}_{fmt}" for fmt in "DEF"]
            x = np.arange(4) + (i - 1) * .25
            axes[0, col].bar(x, [100 * lookup[(family, k, "constrained")]["r1"] for k in keys], .24, label=family.upper(), color=colors[family])
            axes[1, col].bar(x, [100 * lookup[(family, k, "unconstrained")]["invalid1"] for k in keys], .24, label=family.upper(), color=colors[family])
        axes[0, col].axhline(100 * lookup[("lexical", "A", "lexical")]["r1"], color="gray", linestyle="--", label="Content BM25")
        for row in [0, 1]:
            axes[row, col].set_xticks(np.arange(4), list("CDEF"))
            axes[row, col].set_ylim(0, 100)
            axes[row, col].grid(axis="y", alpha=.2)
            axes[row, col].set_axisbelow(True)
        axes[1, col].set_xlabel(f"{budget}-update search budget")
    axes[0, 0].set_ylabel("Constrained screen Recall@1 (%)")
    axes[1, 0].set_ylabel("Unconstrained invalid top-1 (%)")
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=_ASSETS["dpi"], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    publish_png(buffer.getvalue(), "graph")


def conclusion(evidence):
    a = evidence["analysis"]
    p = a["primary"]
    hierarchy = next(r for r in a["effects"] if r["contrast"] == "F_minus_E_384" and r["right"][2] == "constrained")
    totals = {family: sum(r["Fit/tune/save (h)"] for r in evidence["costs"] if r["Model"] == family)
              for family in ["qwen2b", "qwen4b"]}
    return (f"The flat-catalogue upgrade changes Recall@1 by **{p['difference_pp']:+.2f} percentage points**, "
            f"with an exploratory 95% interval **[{p['ci95_pp'][0]:+.2f}, {p['ci95_pp'][1]:+.2f}]**. "
            f"The hierarchy's point advantage over 4B flat is {hierarchy['difference_pp']:+.2f} points and "
            f"{hierarchy['sensitivity_difference_pp']:+.2f} after the fixed ambiguity sensitivity check. "
            f"The D/E/F search cost rises from {totals['qwen2b']:.2f} to {totals['qwen4b']:.2f} hours "
            f"({totals['qwen4b']/totals['qwen2b']:.2f}×). **Keep 2B flat as the practical reference for now.** "
            "This is a development-screen decision from one book and one seed. The 240-question reserve is unscored.")
