"""Audit the downloaded corpus and freeze small design/evaluation manifests.

Standard library only. No model calls, remote code execution, or extraction of
arbitrary ZIP members. Structural boundaries follow the adjacency rule in IBM's
scripts/data_preprocessor.py; eligible queries follow doc2dial_pub.py.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics
import zipfile


SOURCE = "https://doc2dial.github.io/multidoc2dial/file/multidoc2dial.zip"
DOMAINS = ("dmv", "ssa", "va", "studentaid")


def page_id(domain, title):
    # Includes #3, not just the #1/#2 examples in the paper.
    return domain + ":" + re.sub(r"#\d+$", "", title)


def build_passages(documents):
    passages, span_map = [], {}
    for domain in DOMAINS:
        for doc_id, doc in documents[domain].items():
            groups = []
            previous_section = previous_title = None
            for span in doc["spans"].values():
                title = span["title"].strip()
                same_group = (previous_section == span["id_sec"]
                              or previous_title == title)
                if not groups or not same_group:
                    groups.append([])
                groups[-1].append(span)
                previous_section, previous_title = span["id_sec"], title
            for index, spans in enumerate(groups):
                pid = f"{domain}:{doc_id}:p{index:03d}"
                passage = {
                    "passage_id": pid,
                    "page_id": page_id(domain, doc["title"]),
                    "raw_document_id": doc_id,
                    "domain": domain,
                    "title": doc["title"],
                    "heading": spans[0]["title"],
                    "parent_headings": [x["text"] for x in spans[0]["parent_titles"]],
                    "span_ids": [s["id_sp"] for s in spans],
                    "text": " ".join(s["text_sp"] for s in spans).strip(),
                }
                passages.append(passage)
                for span in spans:
                    span_map[domain, doc_id, span["id_sp"]] = pid
    return passages, span_map


def queries(dialogue, domain, span_map):
    """Predict the next agent response, excluding respond_no_solution.

    References and target replies are evaluation-only fields. History contains
    only earlier utterances; the target answer never enters the model input.
    """
    turns = dialogue["turns"]
    for i, turn in enumerate(turns[:-1]):
        target = turns[i + 1]
        if (turn["role"] != "user" or target["role"] != "agent"
                or target["da"] == "respond_no_solution"):
            continue
        gold = sorted({span_map[domain, ref["doc_id"], ref["id_sp"]]
                       for ref in target["references"]})
        if len(gold) != 1:
            raise ValueError(f"Expected one gold passage: {dialogue['dial_id']}, {turn['turn_id']}")
        yield {
            "query_id": f"{dialogue['dial_id']}_{turn['turn_id']}",
            "dialogue_id": dialogue["dial_id"],
            "domain": domain,
            "user_turn_id": turn["turn_id"],
            "target_agent_turn_id": target["turn_id"],
            "history": [{"role": t["role"], "text": t["utterance"]}
                        for t in turns[max(0, i - 5):i + 1]],
            "gold_passage_ids": gold,
            "reference_answer": target["utterance"],
        }


def select(dialogues, count, seed):
    return sorted(dialogues, key=lambda d: hashlib.sha256(
        f"{seed}:{d['dial_id']}".encode()).hexdigest())[:count]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/raw/multidoc2dial.zip"))
    parser.add_argument("--output", type=Path, default=Path("data/prepared"))
    parser.add_argument("--seed", default="42")
    args = parser.parse_args()
    with zipfile.ZipFile(args.archive) as archive:
        documents = json.loads(archive.read("multidoc2dial/multidoc2dial_doc.json"))["doc_data"]
        splits = {split: json.loads(archive.read(
            f"multidoc2dial/multidoc2dial_dial_{split}.json"))["dial_data"]
            for split in ("train", "validation")}
    passages, span_map = build_passages(documents)
    eligible = {domain: [q for d in splits["validation"][domain]
                         for q in queries(d, domain, span_map)] for domain in DOMAINS}
    pages = {p["page_id"] for p in passages}
    counts = {"raw_documents": sum(map(len, documents.values())),
              "merged_pages": len(pages), "passages": len(passages),
              "validation_dialogues": sum(map(len, splits["validation"].values())),
              "validation_queries": sum(map(len, eligible.values()))}
    expected = {"raw_documents": 488, "merged_pages": 413, "passages": 4110,
                "validation_dialogues": 661, "validation_queries": 4201}
    if counts != expected:
        raise ValueError(f"Corpus contract failed: {counts!r} != {expected!r}")
    args.output.mkdir(parents=True, exist_ok=True)
    manifests = {}
    evaluation_dialogue_ids = set()
    for name, split, n in (("design", "train", 5), ("evaluation", "validation", 10)):
        selected = {domain: select(splits[split][domain], n, args.seed) for domain in DOMAINS}
        entries = [q for domain in DOMAINS for d in selected[domain]
                   for q in queries(d, domain, span_map)]
        manifest = {"source_split": split, "selection_seed": args.seed,
                    "selection_rule": "First n dialogues per domain by SHA256(seed + ':' + dialogue_id)",
                    "dialogue_ids": {k: [d["dial_id"] for d in v] for k, v in selected.items()},
                    "query_count": len(entries), "queries": entries}
        write_json(args.output / f"{name}_manifest.json", manifest)
        if name == "evaluation":
            evaluation_dialogue_ids = {d["dial_id"] for ds in selected.values() for d in ds}
        manifests[name] = {"dialogues": n * len(DOMAINS), "queries": len(entries),
                           "queries_by_domain": dict(Counter(q["domain"] for q in entries))}
    words = sorted(len(p["text"].split()) for p in passages)
    audit = {"status": "data_contract_passed_no_model_results",
             "source": SOURCE,
             "archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
             "counts": counts,
             "eligible_queries_by_domain": {k: len(v) for k, v in eligible.items()},
             "passages_by_domain": dict(Counter(p["domain"] for p in passages)),
             "word_lengths": {"median": statistics.median(words),
                              "p95": words[int(.95 * len(words))], "max": max(words)},
             "subsets": manifests,
             "notes": ["All eligible validation references resolve to one structural passage.",
                       "Local passage IDs preserve raw document identity within merged pages.",
                       "No retrieval, generation, model downloads, or Ollama upgrades performed."]}
    write_json(args.output / "dataset_audit.json", audit)
    target_doc_id = "Federal REAL ID#3_0"
    doc = documents["dmv"][target_doc_id]
    target_passages = [p for p in passages if p["raw_document_id"] == target_doc_id]
    target_ids = {p["passage_id"] for p in target_passages}
    target_queries = [q for d in splits["validation"]["dmv"]
                      if d["dial_id"] not in evaluation_dialogue_ids
                      for q in queries(d, "dmv", span_map)
                      if target_ids.intersection(q["gold_passage_ids"])]
    if not target_queries:
        raise ValueError("Example document has no labeled queries outside the evaluation subset")
    write_json(args.output / "example_document.json", {
        "source": SOURCE, "raw_document_id": target_doc_id,
        "title": doc["title"], "text": doc["doc_text"], "passages": target_passages,
        "query_split": "validation", "excluded_from_frozen_evaluation": True,
        "queries": target_queries,
        "usage": "Archived benchmark text for retrieval research; contains outdated dates. "
                 "These examples come from dialogues outside the frozen evaluation subset. "
                 "They are exploratory, not a separate held-out evaluation."})
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
