def source_toc(book_text):
    """Parse numbered source entries, including wrapped titles and duplicate codes.

    Identity includes source order. The printed book repeats number 6.4; treating
    that code alone as an ID would merge two different sections.
    """
    pages = book_text.split("\f")
    result = []
    for pdf_page in CFG.source_toc_pdf_pages:
        current = []
        for line in pages[pdf_page - 1].splitlines():
            line = line.strip()
            if re.match(r"^\d+(?:\.\d+)+\s", line):
                if current:
                    raise ValueError(f"Unparsed source ToC entry on PDF page {pdf_page}: {current}")
                current = [line]
            elif current:
                current.append(line)
            joined = " ".join(current)
            match = re.match(r"^(\d+(?:\.\d+)+)\s+(.*?)\s+\.(?:\s*\.)*\s*(\d+)$", joined)
            if match:
                code, title, printed_page = match.groups()
                result.append({"source_id": f"toc-{len(result):03d}", "code": code,
                               "title": canonical(title), "printed_page": int(printed_page),
                               "toc_pdf_page": pdf_page, "source_order": len(result)})
                current = []
        if current:
            raise ValueError(f"Incomplete source ToC entry on PDF page {pdf_page}: {current}")
    if not result:
        raise ValueError("Source ToC parser returned no entries")
    for row in result:
        row["leaf"] = not any(other["code"].startswith(row["code"] + ".") for other in result)
    return result


def heading_matches(body, leaves):
    """Positive evidence of known source-leaf headings inside one released document.

    Both the exact printed code and the complete source heading must match.
    Wrapped headings are allowed. Decimal amounts cannot qualify by code alone.
    A single detected heading does not certify clean section boundaries.
    """
    found = []
    for leaf in leaves:
        words = canonical(leaf["title"]).split()
        pattern = r"^\s*" + re.escape(leaf["code"]) + r"\s+" + r"\s+".join(map(re.escape, words))
        pattern += r"(?=\s|$)"
        match = re.search(pattern, body, re.I | re.M)
        if match:
            start = match.start()
            found.append({"source_id": leaf["source_id"], "source_code": leaf["code"],
                          "source_title": leaf["title"], "printed_page": leaf["printed_page"],
                          "toc_pdf_page": leaf["toc_pdf_page"], "content_offset": start,
                          "content_line": body[:start].count("\n") + 1,
                          "evidence_excerpt": body[start:start + 500].strip()})
    return sorted(found, key=lambda row: row["content_offset"])


def audit_release(contract=None):
    contract = contract or CFG.target_contract
    if contract not in {"source_leaf_v3", "released_document_anchor_v4"}:
        raise ValueError("Unrecognized retrieval-unit contract")
    validate_inputs()
    raw = CFG.data_folder / "raw/finance"
    corpus = read_jsonl(raw / "corpus.jsonl")
    book = (CFG.data_folder / "source_book/book.txt").read_text(encoding="utf-8")
    toc = source_toc(book)
    leaves = [row for row in toc if row["leaf"]]
    titles = [canonical(row["title"]) for row in corpus]
    title_set = set(titles)
    title_collisions = [title for title, count in Counter(titles).items() if count > 1]
    empty_content = [i for i, row in enumerate(corpus) if not row["content"].strip()]
    content_groups = defaultdict(list)
    target_crosswalk, documents, problems = [], [], []
    for index, row in enumerate(corpus):
        title = canonical(row["title"])
        content_groups[digest_bytes(row["content"].encode())].append(index)
        match = re.fullmatch(r"(\d+(?:\.\d+)+)\s+(.+)", title)
        candidates = []
        if match:
            code, semantic_title = match.groups()
            candidates = [leaf for leaf in leaves if source_match_key(leaf["title"]) == source_match_key(semantic_title)
                          and leaf["code"] == code]
            if not candidates:
                # Source-numbering discrepancy is recorded, never used to relabel targets.
                candidates = [leaf for leaf in leaves if source_match_key(leaf["title"]) == source_match_key(semantic_title)
                              and leaf["code"].split(".")[0] == code.split(".")[0]]
        if len(candidates) != 1:
            problems.append({"section_id": f"finance:{index:03d}", "title": title,
                             "reason": "source_leaf_mapping_not_unique", "matches": len(candidates)})
        mapped = candidates[0] if len(candidates) == 1 else None
        if mapped:
            target_crosswalk.append({"section_id": f"finance:{index:03d}", "release_title": row["title"],
                                     "canonical_title": title, "source_id": mapped["source_id"],
                                     "source_title": mapped["code"] + " " + mapped["title"],
                                     "source_printed_page": mapped["printed_page"],
                                     "source_toc_pdf_page": mapped["toc_pdf_page"],
                                     "numbering_differs": match.group(1) != mapped["code"]})
        headings = heading_matches(row["content"], leaves)
        documents.append({"section_id": f"finance:{index:03d}", "release_title": row["title"],
                          "body_sha256": digest_bytes(row["content"].encode()),
                          "mapped_source_id": mapped["source_id"] if mapped else None,
                          "source_leaf_headings": headings,
                          "confirmed_multi_leaf": len(headings) > 1,
                          "target_heading_detected": mapped is not None and any(h["source_id"] == mapped["source_id"] for h in headings)})
    duplicates = [group for group in content_groups.values() if len(group) > 1]
    query_counts, missing_gold, query_id_collisions = {}, [], []
    for split in ("train", "dev", "test"):
        rows = read_jsonl(raw / f"{split}.jsonl")
        query_counts[split] = len(rows)
        ids = [f"finance:{split}:{row['query_id']}" for row in rows]
        query_id_collisions.extend(key for key, n in Counter(ids).items() if n > 1)
        for row in rows:
            if canonical(row["gold_doc_title"]) not in title_set:
                missing_gold.append({"split": split, "query_id": row["query_id"], "title": row["gold_doc_title"]})
    multi_leaf = [row for row in documents if row["confirmed_multi_leaf"]]
    unseen_heading = [row["release_title"] for row in documents if not row["target_heading_detected"]]
    target_sources = {row["source_id"] for row in target_crosswalk}
    observed_sources = {heading["source_id"] for row in documents for heading in row["source_leaf_headings"]}
    source_missing_as_targets = [row for row in leaves if row["source_id"] not in target_sources]
    duplicate_source_codes = [code for code, n in Counter(row["code"] for row in toc).items() if n > 1]
    fatal = []
    if title_collisions or empty_content or duplicates or missing_gold or query_id_collisions:
        fatal.append("release_structural_integrity_failed")
    if problems or unseen_heading:
        fatal.append("source_leaf_mapping_or_heading_verification_failed")
    if multi_leaf and contract == "source_leaf_v3":
        fatal.append("released_documents_contain_multiple_source_leaves")
    if len(target_sources) != len(target_crosswalk):
        fatal.append("release_to_source_mapping_not_injective")
    return {"status": "stopped_at_data_gate" if fatal else "data_gate_passed_with_boundary_warnings",
            "target_contract": contract,
            "historical_source_leaf_gate": {"status": "failed" if multi_leaf else "passed",
                                            "confirmed_multi_leaf_documents": len(multi_leaf)},
            "quality_warnings": ["released_documents_contain_multiple_source_leaves"] if multi_leaf else [],
            "fatal_reasons": fatal, "dataset_revision": CFG.dataset_revision,
            "source_pdf_sha256": CFG.source_pdf_sha256,
            "corpus_documents": len(corpus), "query_counts": query_counts,
            "source_toc_entries": len(toc), "source_leaf_count": len(leaves),
            "source_duplicate_number_codes": duplicate_source_codes,
            "confirmed_multi_leaf_documents": len(multi_leaf),
            "numbering_discrepancies": sum(row["numbering_differs"] for row in target_crosswalk),
            "source_leaves_not_direct_targets": source_missing_as_targets,
            "non_target_source_headings_observed_in_content": sorted(observed_sources - target_sources),
            "title_collisions": title_collisions, "empty_content": empty_content,
            "duplicate_content_groups": duplicates, "missing_gold_titles": missing_gold,
            "query_id_collisions": query_id_collisions, "mapping_problems": problems,
            "undetected_target_headings": unseen_heading, "crosswalk": target_crosswalk,
            "source_toc": toc, "document_audit": documents,
            "model_downloaded": False, "training_started": False, "test_predictions": 0,
            "execution_flags_scope": "Operations inside this source-audit function only; complete run status/model provenance are separate",
            "detection_limit": "Multiple exact source-leaf headings establish merged content. A single heading does not certify clean boundaries; unmarked spillover may remain."}


def build_anchor_hierarchy(audit):
    if audit["fatal_reasons"]:
        raise ValueError(f"Anchor admission failed: {audit['fatal_reasons']}")
    toc = audit["source_toc"]
    source_by_id = {row["source_id"]: row for row in toc}
    pages = (CFG.data_folder / "source_book/book.txt").read_text().split("\f")
    chapters = {}
    for page_num in CFG.source_toc_pdf_pages:
        current = []
        for line in pages[page_num - 1].splitlines():
            line = line.strip()
            if re.match(r"^Chapter \d+:", line):
                current = [line]
            elif current:
                current.append(line)
            match = re.match(r"^Chapter (\d+):\s+(.*?)\s+(\d+)$", " ".join(current))
            if match:
                number, title, printed_page = match.groups()
                chapters[number] = {"node_id": "chapter-" + number,
                                    "label": "Chapter " + number + ": " + canonical(title),
                                    "toc_pdf_page": page_num, "source_printed_page": int(printed_page)}
                current = []
    nodes, edges, lines = {}, [], []
    target_ids = []
    for anchor in audit["crosswalk"]:
        source = source_by_id[anchor["source_id"]]
        parts = source["code"].split(".")
        chapter = chapters.get(parts[0])
        if not chapter:
            raise ValueError("Missing verified source chapter: " + parts[0])
        parent_id = chapter["node_id"]
        if parent_id not in nodes:
            nodes[parent_id] = {**chapter, "parent_id": None, "target": False}
            lines.append(chapter["label"])
        for depth in range(2, len(parts)):
            prefix = ".".join(parts[:depth])
            ancestors = [row for row in toc if row["code"] == prefix]
            if len(ancestors) != 1 or ancestors[0]["leaf"]:
                raise ValueError("Ambiguous/missing source ancestor: " + prefix)
            ancestor = ancestors[0]
            ancestor_id = ancestor["source_id"]
            if ancestor_id not in nodes:
                label = prefix + " " + ancestor["title"]
                nodes[ancestor_id] = {"node_id": ancestor_id, "parent_id": parent_id,
                                      "label": label, "target": False,
                                      "toc_pdf_page": ancestor["toc_pdf_page"]}
                edges.append([parent_id, ancestor_id])
                lines.append("  " * (depth - 1) + label)
            parent_id = ancestor_id
        target_id = anchor["section_id"]
        if target_id in nodes:
            raise ValueError("Duplicate target ID")
        nodes[target_id] = {"node_id": target_id, "parent_id": parent_id, "target": True,
                            "label": anchor["canonical_title"], "source_id": source["source_id"]}
        edges.append([parent_id, target_id])
        target_ids.append(target_id)
        lines.append("  " * (len(parts) - 1) + anchor["canonical_title"])
    if len(target_ids) != audit["corpus_documents"] or len(set(target_ids)) != len(target_ids):
        raise ValueError("Incomplete anchor coverage")
    for node_id, node in nodes.items():
        visited = set()
        current = node_id
        while current is not None:
            if current in visited or current not in nodes:
                raise ValueError("Cycle or disconnected source hierarchy")
            visited.add(current)
            current = nodes[current]["parent_id"]
        if node["target"] and any(parent == node_id for parent, _ in edges):
            raise ValueError("A retrieval anchor is also an ancestor")
    return {"nodes": list(nodes.values()), "edges": edges, "target_ids": target_ids,
            "hierarchical_text": "\n".join(lines),
            "flat_text": "\n".join(row["canonical_title"] for row in audit["crosswalk"]),
            "meaning": "Verified ancestor paths of released-document title anchors; not a content partition"}


def freeze_memberships(audit):
    """Apply the prospectively declared duplicate components before hash sampling."""
    raw = CFG.data_folder / "raw/finance"
    rows = []
    for priority, split in enumerate(("train", "dev", "test")):
        for item in read_jsonl(raw / f"{split}.jsonl"):
            query_id = f"finance:{split}:{item['query_id']}"
            normalized = canonical(item["query"]).casefold()
            rows.append({"query_id": query_id, "query": item["query"], "split": split,
                         "priority": priority, "gold_title": canonical(item["gold_doc_title"]),
                         "answer": item["answer"], "normalized": normalized,
                         "tokens": set(re.findall(r"\w+", normalized, re.UNICODE))})
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, row in enumerate(rows):
        left = row["tokens"]
        for j in range(i):
            right = rows[j]["tokens"]
            if min(len(left), len(right)) < 0.9 * max(len(left), len(right)):
                continue
            union_size = len(left | right)
            similarity = len(left & right) / union_size if union_size else 1.0
            if row["normalized"] == rows[j]["normalized"] or similarity >= 0.9:
                parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)
    eligible = defaultdict(list)
    exclusions = []
    for component in groups.values():
        first_priority = min(row["priority"] for row in component)
        retained = min((row for row in component if row["priority"] == first_priority), key=lambda r: r["query_id"])
        component_id = digest_bytes("|".join(sorted(r["query_id"] for r in component)).encode())[:16]
        for row in component:
            row["component_id"] = component_id
            row.pop("tokens")
            if row is retained:
                row["sample_hash"] = digest_bytes(f"stair-v1|{row['split']}|{row['query_id']}|{row['normalized']}".encode())
                eligible[row["split"]].append(row)
            else:
                exclusions.append({"query_id": row["query_id"], "retained_id": retained["query_id"],
                                   "component_id": component_id, "reason": "exact_or_jaccard_0.9_component; split priority then lowest qualified ID"})
    profiles = {}
    for name, sizes in (("preferred", CFG.preferred_profile), ("fallback", CFG.fallback_profile)):
        profiles[name] = {}
        for split in ("train", "dev", "test"):
            ordered = sorted(eligible[split], key=lambda row: row["sample_hash"])
            if len(ordered) < sizes[split]:
                raise ValueError(f"Insufficient duplicate-filtered {split} queries for {name}: {len(ordered)}")
            profiles[name][split] = ordered[:sizes[split]]
    return {"profiles": profiles, "exclusions": exclusions,
            "eligible_counts": {split: len(items) for split, items in eligible.items()},
            "paragraph_independence": "unverified: the release has no source paragraph IDs",
            "rule": "connected components of normalized exact matches or word-token-set Jaccard >=0.9; train/dev/test priority; lowest qualified ID"}
