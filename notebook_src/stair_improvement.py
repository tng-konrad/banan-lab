"""Isolated FLAN improvement pilots; original notebook/run artifacts stay immutable."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ["STAIR_ROOT"] = str(ROOT)
os.environ["STAIR_MODE"] = "report"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
HELPERS = ["00_setup.py", "01_evidence.py", "02_data.py", "04_model.py", "05_metrics.py", "06_runtime.py"]
for _name in HELPERS:
    exec(compile((ROOT / "notebook_src/stair" / _name).read_text(), _name, "exec"), globals())

from transformers.modeling_outputs import BaseModelOutput
from safetensors import safe_open

OLD_RUN = ROOT / "runs/stair/anchor-20261106T160836172708Z"
NEW_ROOT = ROOT / "runs/stair/improvement-pilots"


def protected_artifacts():
    paths = [ROOT / "stair_local_demo.ipynb", ROOT / "stair-build-report.md", ROOT / "stair-post-plan.md",
             ROOT / "stair-local-demo-bundle.zip", ROOT / "runs/stair/latest.json", OLD_RUN / "run_manifest.json",
             ROOT / "runs/stair/qwen-pilots/latest.json"]
    return {str(path.relative_to(ROOT)): digest_bytes(path.read_bytes()) for path in paths}


def stripped_row(row):
    return {key: row[key] for key in ["query_id", "query", "gold_title", "split", "normalized", "sample_hash", "component_id"]}


def prepare_run():
    validate_inputs()
    validate_model_cache()
    audit = audit_release()
    hierarchy = build_anchor_hierarchy(audit)
    if audit["fatal_reasons"] or audit["confirmed_multi_leaf_documents"] != 9:
        raise ValueError("Source/anchor contract differs from the approved evidence")
    frozen = json_read(CFG.data_folder / "frozen_memberships.json")
    if freeze_memberships(audit) != frozen:
        raise ValueError("Historical duplicate/hash memberships changed")
    small = frozen["profiles"]["preferred"]
    exclusions = {row["query_id"] for row in frozen["exclusions"]}
    components = {row["query_id"]: row["component_id"]
                  for profile in frozen["profiles"].values() for rows in profile.values() for row in rows}
    components.update({row["retained_id"]: row["component_id"] for row in frozen["exclusions"]})
    eligible = {}
    for split in ["train", "dev", "test"]:
        rows = []
        for raw in read_jsonl(CFG.data_folder / "raw/finance" / (split + ".jsonl")):
            query_id = f"finance:{split}:{raw['query_id']}"
            if query_id in exclusions:
                continue
            normalized = canonical(raw["query"]).casefold()
            rows.append({"query_id": query_id, "query": raw["query"], "gold_title": canonical(raw["gold_doc_title"]),
                         "split": split, "normalized": normalized,
                         "sample_hash": digest_bytes(f"stair-v1|{split}|{query_id}|{normalized}".encode()),
                         "component_id": components.get(query_id, digest_bytes(query_id.encode())[:16])})
        eligible[split] = sorted(rows, key=lambda row: row["sample_hash"])
    for split in eligible:
        if [row["query_id"] for row in eligible[split][:len(small[split])]] != [row["query_id"] for row in small[split]]:
            raise ValueError("Small cohort is no longer the historical hash prefix")
    tune_ids = {row["query_id"] for row in small["dev"]}
    old_test = {row["query_id"] for row in small["test"]}
    screen = [row for row in eligible["dev"] if row["query_id"] not in tune_ids]
    reserve = [row for row in eligible["test"] if row["query_id"] not in old_test][:240]
    corpus = read_jsonl(CFG.data_folder / "raw/finance/corpus.jsonl")
    hierarchy["corpus"] = corpus
    data = {"train256": [stripped_row(row) for row in eligible["train"][:256]],
            "train935": [stripped_row(row) for row in eligible["train"]],
            "tune": [stripped_row(row) for row in eligible["dev"][:48]],
            "screen": [stripped_row(row) for row in screen], "diagnostic_ids": [row["query_id"] for row in screen[:16]],
            "corpus": corpus, "hierarchy": hierarchy,
            "historical_test_ids": sorted(old_test), "paragraph_independence": "unverified"}
    if [len(data[key]) for key in ["train256", "train935", "tune", "screen"]] != [256, 935, 48, 127]:
        raise ValueError("Planned cohort counts changed")
    if {r["query_id"] for r in data["tune"]} & {r["query_id"] for r in data["screen"]}:
        raise ValueError("Tune/screen leakage")
    run = NEW_ROOT / datetime.now(timezone.utc).strftime("cheap-%Y%m%dT%H%M%S%fZ")
    run.mkdir(parents=True)
    json_write(run / "inputs.json", data)
    json_write(run / "confirmation_reserve.json", {"ids": [row["query_id"] for row in reserve],
               "queries": len(reserve), "model_predictions": 0, "policy": "original hash order, excluding historical 120; no query/gold contents in this file"})
    json_write(run / "source_audit.json", audit)
    json_write(run / "protected_artifacts.json", protected_artifacts())
    for path in [ROOT / "plans/stair-improvement-pilot-plan.md", ROOT / "plans/stair-improvement-pilot-review.md",
                 ROOT / "requirements-stair.lock.txt", ROOT / "docs/stair-section-boundary-decision.md"]:
        shutil.copy2(path, run / path.name)
    for name in HELPERS:
        target = run / "source_snapshot/base" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "notebook_src/stair" / name, target)
    shutil.copy2(__file__, run / "source_snapshot/stair_improvement.py")
    semantic = []
    by_title = {canonical(row["title"]): row["content"] for row in corpus}
    for row in screen[:20]:
        semantic.append({"query_id": row["query_id"], "query": row["query"], "gold_title": row["gold_title"],
                         "body": by_title[row["gold_title"]], "status": "pending", "reason": ""})
    json_write(run / "semantic_review_inputs.json", semantic)
    json_write(NEW_ROOT / "prepared.json", {"run_id": run.name, "status": "awaiting output-blind semantic audit"})
    print(json.dumps({"run": str(run), "counts": {key: len(data[key]) for key in ["train256", "train935", "tune", "screen"]},
                      "reserve": len(reserve), "semantic_audit_rows": len(semantic)}, indent=2))
    return run


def training_groups(rows, order_seed=42002):
    """Deterministic shuffled stream, carrying remainders into a full group of eight."""
    pending = []
    epoch = 1
    while True:
        for index in np.random.default_rng(order_seed + epoch).permutation(len(rows)):
            pending.append(rows[int(index)])
            if len(pending) == 8:
                yield pending
                pending = []
        epoch += 1


def shadow_selection(history):
    best = None
    stale = 0
    stop = None
    selected = None
    for row in history:
        if best is None or row["tune_r1"] > best["tune_r1"]:
            best = row
            stale = 0
        else:
            stale += 1
        if row["update"] >= 128 and stale >= 4:
            stop, selected = row["update"], best
            break
    if stop is None:
        stop, selected = history[-1]["update"], best
    return {"would_stop_at_update": stop, "selected_update": selected["update"],
            "selected_checkpoint": selected["checkpoint"], "selected_tune_r1": selected["tune_r1"],
            "cost_kind": "hypothetical policy replay; controlled trajectory actually ran to 384"}


def exact_scores(model, input_ids, attention_mask, paths, batch_size=8):
    """Full-vocabulary EOS-inclusive mean log-probability; one encoder call per query."""
    model.eval()
    started = time.monotonic()
    rows = []
    with torch.inference_mode():
        encoded = model.get_encoder()(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        for start in range(0, len(paths), batch_size):
            subset = paths[start:start + batch_size]
            width = max(map(len, subset))
            labels = torch.full((len(subset), width), -100, dtype=torch.long, device=input_ids.device)
            for index, path in enumerate(subset):
                labels[index, :len(path)] = torch.tensor(path, dtype=torch.long, device=input_ids.device)
            valid = labels != -100
            decoder = model._shift_right(labels)
            output = model(encoder_outputs=BaseModelOutput(last_hidden_state=encoded.last_hidden_state.expand(len(subset), -1, -1)),
                           attention_mask=attention_mask.expand(len(subset), -1),
                           decoder_input_ids=decoder, decoder_attention_mask=valid.long(), use_cache=False, return_dict=True)
            logp = torch_F.log_softmax(output.logits.float(), dim=-1)
            chosen = logp.gather(2, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
            means = ((chosen * valid).sum(dim=1) / valid.sum(dim=1)).cpu().tolist()
            chosen = chosen.cpu().tolist()
            for index, path in enumerate(subset):
                rows.append({"candidate_index": start + index, "score": means[index],
                             "token_ids": path, "token_logprobs": chosen[index][:len(path)]})
            del output, logp, chosen
    return rows, time.monotonic() - started


def exact_record(model, tokenizer, fmt, query, hierarchy, paths, candidate_batch=8):
    started = time.monotonic()
    prompt = prompt_text(fmt, query["query"], hierarchy)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=False)
    if encoded.input_ids.shape[1] > CFG.encoder_cap:
        raise ValueError("Incomplete input forbidden")
    scores, model_seconds = exact_scores(model, encoded.input_ids, encoded.attention_mask, paths, candidate_batch)
    order = sorted(scores, key=lambda row: (-row["score"], row["candidate_index"]))
    hypotheses = []
    for rank, row in enumerate(order[:3], 1):
        title = canonical(hierarchy["corpus"][row["candidate_index"]]["title"])
        hypotheses.append({"rank": rank, "section_id": f"finance:{row['candidate_index']:03d}",
                           "raw_text": title, "canonical_title": title, "validity": "valid", "complete_eos": True,
                           "score": row["score"], "token_ids": row["token_ids"], "token_logprobs": row["token_logprobs"]})
    return {"method": fmt, "mode": "exact", "query_id": query["query_id"], "hypotheses": hypotheses,
            "prompt_sha256": digest_bytes(prompt.encode()), "input_tokens": encoded.input_ids.shape[1],
            "candidate_batch": candidate_batch, "candidates": scores,
            "timing": {"batch_size": 1, "model_seconds": model_seconds, "end_to_end_seconds": time.monotonic() - started}}


class CheapGuard:
    def __init__(self, run, started_wall, prior_charge_seconds=0, active_clock=False):
        self.run, self.started_wall = run, started_wall
        self.active_clock = active_clock
        self.prior_charge_seconds = prior_charge_seconds
        self.started_monotonic = time.monotonic()
        self.phase = "A"
        self.phase_started = time.time()
        self.phase_started_monotonic = self.started_monotonic
        self.swap_start = psutil.swap_memory().used
        self.samples = []
        self.violation = None
        self.stop = threading.Event()
        self.worker = threading.Thread(target=self.watch, daemon=True)
        self.worker.start()

    def elapsed(self):
        if self.active_clock:
            return self.prior_charge_seconds + time.monotonic() - self.started_monotonic
        return time.time() - self.started_wall

    def phase_elapsed(self):
        return time.monotonic()-self.phase_started_monotonic if self.active_clock else time.time()-self.phase_started

    def sample(self):
        virtual = psutil.virtual_memory()
        row = {"elapsed_seconds": self.elapsed(), "wall_elapsed_seconds":time.time()-self.started_wall, "phase": self.phase,
               "rss_bytes": psutil.Process().memory_info().rss, "available_bytes": virtual.available,
               "swap_used_bytes": psutil.swap_memory().used}
        self.samples.append(row)
        if row["rss_bytes"] > 48 * 2**30 or row["available_bytes"] < 32 * 2**30:
            self.violation = "Memory limit"
        if row["swap_used_bytes"] - self.swap_start > 2**30:
            self.violation = "Swap-growth limit"
        phase_cap = 2700 if self.phase == "A" else 8100
        if row["elapsed_seconds"] >= 10800 or self.phase_elapsed() >= phase_cap:
            self.violation = "Compute deadline"
        return row

    def watch(self):
        while not self.stop.wait(.5):
            self.sample()
            if self.violation:
                json_write(self.run / "resource_violation.json", {"reason": self.violation, "sample": self.samples[-1]})
                json_write(self.run / "memory_samples.json", self.samples)
                os._exit(70)

    def check(self):
        self.sample()
        if self.violation:
            raise RuntimeError(self.violation)

    def admit(self, projected_seconds, label):
        elapsed = self.elapsed()
        phase_elapsed = self.phase_elapsed()
        phase_remaining = (2700 if self.phase == "A" else 8100) - phase_elapsed
        needed = 1.25 * projected_seconds + 900
        decision = {"label": label, "elapsed_seconds": elapsed, "projected_seconds": projected_seconds,
                    "margin_and_reserve_seconds": needed, "remaining_phase_seconds": phase_remaining,
                    "remaining_total_seconds": 10800 - elapsed,
                    "admitted": needed <= min(phase_remaining, 10800 - elapsed)}
        with (self.run / "admissions.jsonl").open("a") as handle:
            handle.write(json.dumps(decision) + "\n")
        if not decision["admitted"]:
            raise RuntimeError("Stage does not fit the approved budget: " + label)
        return decision

    def close(self):
        self.stop.set()
        self.worker.join(timeout=2)
        json_write(self.run / "memory_samples.json", self.samples)


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def emit(run, kind, **values):
    row = {"kind": kind, "timestamp": datetime.now(timezone.utc).isoformat(), **values}
    append_jsonl(run / "events.jsonl", row)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def model_for(fmt, path=None):
    seed_model(42, "cpu")
    if path is None:
        return load_base("cpu")
    model = AutoModelForSeq2SeqLM.from_pretrained(path, local_files_only=True, dtype=torch.float32,
                                                 attn_implementation="eager").to("cpu")
    if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("CPU float32 admission failed")
    return model


def beam_queries(model, tokenizer, trie, fmt, mode, queries, hierarchy, guard):
    return run_neural_queries(model, tokenizer, trie, fmt, mode, queries, hierarchy, "cpu", 4, guard)


def save_records(run, key, mode, records):
    path = run / "predictions" / key / (mode + ".jsonl")
    if path.exists():
        raise ValueError("Would overwrite immutable prediction evidence")
    for row in records:
        append_jsonl(path, row)


def quality(records, queries, corpus):
    frame = score_records(records, queries, corpus)
    return {"queries": len(frame), **{key: float(frame[key].mean()) for key in ["r1", "r3", "ndcg3", "invalid1"]}}


def prediction_seconds(records):
    # Batched records share one timing; charge the completed call only once.
    total = 0.0
    offset = 0
    while offset < len(records):
        timing = records[offset]["timing"]
        total += timing["end_to_end_seconds"]
        offset += timing["batch_size"]
    return total


def exact_admission(model, tokenizer, trie, fmt, data, paths, guard):
    queries = data["tune"][:8]
    candidates, timings = {}, {}
    # Full singleton candidate scoring on one query, then eight-query batched parity/timing.
    baseline = [exact_record(model, tokenizer, fmt, query, data["hierarchy"], paths, 1) for query in queries]
    for batch in [8, 16]:
        results = []
        for query in queries:
            guard.check()
            results.append(exact_record(model, tokenizer, fmt, query, data["hierarchy"], paths, batch))
        candidates[batch] = results
        timings[batch] = [row["timing"]["end_to_end_seconds"] for row in results]
        for scalar, result in zip(baseline, results):
            if not np.allclose([row["score"] for row in scalar["candidates"]],
                               [row["score"] for row in result["candidates"]], atol=1e-4, rtol=1e-4):
                raise ValueError("Exact candidate singleton score parity failed")
            if [r["section_id"] for r in scalar["hypotheses"]] != [r["section_id"] for r in result["hypotheses"]]:
                raise ValueError("Exact candidate singleton rank parity failed")
    for left, right in zip(candidates[8], candidates[16]):
        if not np.allclose([r["score"] for r in left["candidates"]], [r["score"] for r in right["candidates"]], atol=1e-4, rtol=1e-4):
            raise ValueError("Exact candidate batch-size score parity failed")
        if [r["section_id"] for r in left["hypotheses"]] != [r["section_id"] for r in right["hypotheses"]]:
            raise ValueError("Exact candidate batch-size rank parity failed")
    beam, beam_times = beam_queries(model, tokenizer, trie, fmt, "constrained", queries, data["hierarchy"], guard)
    singleton, _ = run_neural_queries(model, tokenizer, trie, fmt, "constrained", queries, data["hierarchy"], "cpu", 1, guard)
    if not equal_decoding(beam, singleton):
        raise ValueError("Beam singleton parity failed")
    for exact, approx in zip(candidates[8], beam):
        by_index = {r["candidate_index"]: r["score"] for r in exact["candidates"]}
        for hypothesis in approx["hypotheses"]:
            index = int(hypothesis["section_id"].split(":")[1])
            if not np.isclose(by_index[index], hypothesis["score"], atol=1e-4, rtol=1e-4):
                raise ValueError("Teacher-forced and independently reconstructed beam scores differ")
        if exact["hypotheses"][0]["score"] + 1e-4 < approx["hypotheses"][0]["score"]:
            raise ValueError("Exhaustive score optimum is below the beam result")
    return {"timings": timings, "beam_calls": beam_times, "score_and_rank_parity": "passed",
            "beam_score_reconstruction": "passed", "scalar_candidate_parity": "passed"}


def paired_summary(left, right, queries, corpus):
    """Paired right-minus-left R@1 with gold-document cluster resampling."""
    l = score_records(left, queries, corpus).set_index("query_id")
    r = score_records(right, queries, corpus).set_index("query_id")
    ids = [row["query_id"] for row in queries]
    if set(l.index) != set(ids) or set(r.index) != set(ids):
        raise ValueError("Incomplete paired comparison")
    delta = r.loc[ids, "r1"].to_numpy() - l.loc[ids, "r1"].to_numpy()
    sums, counts = [], []
    gold = l.loc[ids, "gold_title"].to_numpy()
    for title in sorted(set(gold)):
        mask = gold == title
        sums.append(float(delta[mask].sum()))
        counts.append(int(mask.sum()))
    rng = np.random.default_rng(42001)
    sampled = rng.integers(0, len(sums), (10000, len(sums)))
    bootstrap = np.array(sums)[sampled].sum(1) / np.array(counts)[sampled].sum(1)
    lo, hi = np.quantile(bootstrap, [.025, .975]) * 100
    return {"difference_pp": float(delta.mean() * 100), "ci95_pp": [float(lo), float(hi)],
            "gains": int(((l.loc[ids, "r1"].to_numpy() == 0) & (r.loc[ids, "r1"].to_numpy() == 1)).sum()),
            "losses": int(((l.loc[ids, "r1"].to_numpy() == 1) & (r.loc[ids, "r1"].to_numpy() == 0)).sum()),
            "queries": len(ids), "document_clusters": len(sums), "unstable_small_support": len(sums) < 10,
            "uncertainty_scope": "descriptive, one book and seed; exploratory development screen"}


def worker(run, started_wall):
    if (run / "recovery.json").exists():
        return recovery_worker(run, started_wall)
    import socket
    def blocked(*args, **kwargs):
        raise AssertionError("Network is forbidden during the cached-input CPU pilot")
    socket.socket.connect = blocked
    socket.create_connection = blocked
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    data = json_read(run / "inputs.json")
    semantic = json_read(run / "screen_semantic_audit.json")
    if semantic["systematic_label_failure_found"] or any(row["status"] == "pending" for row in semantic["rows"]):
        raise ValueError("Output-blind semantic audit not admitted")
    if [row["query_id"] for row in semantic["rows"]] != [row["query_id"] for row in data["screen"][:20]]:
        raise ValueError("Semantic audit membership changed")
    if protected_artifacts() != json_read(run / "protected_artifacts.json"):
        raise ValueError("Protected scientific artifacts changed before the new pilot")
    shutil.copy2(__file__, run / "source_snapshot/stair_improvement.py")
    guard = CheapGuard(run, started_wall)
    cells, sources = {}, {}
    cache = {}
    try:
        environment = package_inventory() | {"device": "cpu", "dtype": "float32", "threads": 8, "interop_threads": 1,
             "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(), "network": "blocked"}
        json_write(run / "environment.json", environment)
        emit(run, "started", environment=environment, phase="A exact scoring")
        memberships = {"profiles": {"preferred": {"train": data["train935"], "dev": data["tune"] + data["screen"]}}}
        tokenizer, paths, tokens = tokenizer_admission(data["corpus"], data["hierarchy"], memberships)
        json_write(run / "tokenizer_admission.json", tokens)
        trie = TargetTrie(paths)
        diag_set = set(data["diagnostic_ids"])
        diagnostic = [row for row in data["screen"] if row["query_id"] in diag_set]
        selected = {row["method"]: row for row in json_read(OLD_RUN / "checkpoint_selection.json")}
        source_paths = {fmt: OLD_RUN / selected[fmt]["selected_checkpoint"] for fmt in "DEF"}
        source_paths["C"] = None
        admission = {}
        for fmt in "CDEF":
            model = model_for(fmt, source_paths[fmt])
            admission[fmt] = exact_admission(model, tokenizer, trie, fmt, data, paths, guard)
            del model
            release_model()
            emit(run, "exact_admission", format=fmt, checks="passed", timing=admission[fmt]["timings"])
        batch = min([8, 16], key=lambda b: sum(sum(value["timings"][b]) for value in admission.values()))
        # Persist string keys after selection: JSON roundtrip naturally converts integer dictionary keys.
        json_write(run / "exact_admission.json", {"candidate_batch": batch, "formats": admission})
        exact_remaining = sum(float(np.quantile(value["timings"][batch], .9)) * 127 for value in admission.values())
        beam_remaining = sum(float(np.quantile([r["end_to_end_seconds"] / r["batch_size"] for r in value["beam_calls"]], .9)) * (127 + 16 * 4) for value in admission.values())
        guard.admit(exact_remaining + beam_remaining, "A complete C/D/E/F screen and diagnostics")
        lex = lexical_rankings(data["corpus"], data["screen"])
        save_records(run, "lexical", "lexical", lex)
        for fmt in "CDEF":
            model = model_for(fmt, source_paths[fmt])
            checkpoint = (source_paths[fmt] or CFG.model_folder) / "model.safetensors"
            sha = digest_bytes(checkpoint.read_bytes())
            key = "original_" + fmt
            approx, _ = beam_queries(model, tokenizer, trie, fmt, "constrained", data["screen"], data["hierarchy"], guard)
            save_records(run, key, "constrained", approx)
            exhaustive = []
            for index, query in enumerate(data["screen"]):
                guard.check()
                exhaustive.append(exact_record(model, tokenizer, fmt, query, data["hierarchy"], paths, batch))
                if (index + 1) % 32 == 0:
                    emit(run, "exact_screen_progress", format=fmt, queries=index + 1)
            save_records(run, key, "exact", exhaustive)
            unc, _ = beam_queries(model, tokenizer, trie, fmt, "unconstrained", diagnostic, data["hierarchy"], guard)
            save_records(run, key, "unconstrained", unc)
            for mode, recs in [("constrained", approx), ("exact", exhaustive), ("unconstrained", unc)]:
                cache[(sha, fmt, mode)] = (key, recs)
            cells[key] = {"method": fmt, "checkpoint": str(checkpoint.parent), "checkpoint_sha256": sha,
                          "pool": 256 if fmt != "C" else 0, "budget": "historical selected", "kind": "A"}
            emit(run, "A_screen_complete", format=fmt, beam=quality(approx, data["screen"], data["corpus"]),
                 exact=quality(exhaustive, data["screen"], data["corpus"]))
            del model
            release_model()
        json_write(run / "cells.json", cells)
        emit(run, "phase_A_complete", elapsed_seconds=time.time() - started_wall)
        guard.phase = "BC"
        guard.phase_started = time.time()
        train_admission = {}
        for fmt in "DEF":
            model = model_for(fmt)
            optimizer = make_optimizer(model)
            ordered = sorted(data["train935"], key=lambda r: len(tokenizer.encode(prompt_text(fmt, r["query"], data["hierarchy"]))))
            timings = []
            for group in [ordered[:4] + ordered[-4:], data["train935"][:8]]:
                guard.check()
                tick = time.monotonic()
                loss = train_update(model, tokenizer, optimizer, fmt, group, data["hierarchy"], "cpu", 4)
                timings.append(time.monotonic() - tick)
            train_admission[fmt] = {"update_seconds": timings, "last_loss": loss, "pilot_weights_discarded": True}
            del optimizer, model
            release_model()
        train_projection = sum(float(np.quantile(v["update_seconds"], .9)) * 384 * 2 for v in train_admission.values())
        dev_projection = sum(float(np.quantile([r["end_to_end_seconds"] / r["batch_size"] for r in admission[fmt]["beam_calls"]], .9))
                             * (48 * 12 * 2 + 127 * 6 + 16 * 4 * 6) for fmt in "DEF")
        guard.admit(train_projection + dev_projection + 600, "BC six 384-update trajectories plus screening")
        json_write(run / "training_admission.json", {"formats": train_admission, "projected_training_seconds": train_projection,
                    "projected_evaluation_seconds": dev_projection})
        complete_training_and_screen(run, data, tokenizer, paths, trie, diagnostic, guard, cells, cache, batch)
        if protected_artifacts() != json_read(run / "protected_artifacts.json"):
            raise ValueError("Historical protected artifacts changed")
        json_write(run / "status.json", {"status": "complete", "elapsed_seconds": time.time() - started_wall,
            "backend": "cpu", "dtype": "float32", "new_test_predictions": 0, "qwen_quality_stage": "deferred by user",
            "original_artifacts_unchanged": True})
        emit(run, "complete", elapsed_seconds=time.time() - started_wall)
    except Exception as error:
        import traceback
        json_write(run / "failure.json", {"error": str(error), "traceback": traceback.format_exc(),
                    "phase": guard.phase, "elapsed_seconds": time.time() - started_wall})
        json_write(run / "status.json", {"status": "incomplete", "error": str(error), "new_test_predictions": 0})
        raise
    finally:
        guard.close()


def recovery_worker(run, started_wall):
    """Finish the unchanged scientific matrix, preserving the interrupted attempt."""
    import socket
    def blocked(*args, **kwargs):
        raise AssertionError("Network is forbidden during recovery")
    socket.socket.connect = blocked
    socket.create_connection = blocked
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    recovery = json_read(run / "recovery.json")
    prior = Path(recovery["source_run"])
    for name in ["inputs.json", "screen_semantic_audit.json", "confirmation_reserve.json", "protected_artifacts.json"]:
        if digest_bytes((run/name).read_bytes()) != digest_bytes((prior/name).read_bytes()):
            raise ValueError("Recovery changed frozen input evidence: "+name)
    if protected_artifacts() != json_read(run / "protected_artifacts.json"):
        raise ValueError("Protected scientific artifacts changed before recovery")
    shutil.copy2(__file__, run / "source_snapshot/stair_improvement.py")
    data = json_read(run / "inputs.json")
    guard = CheapGuard(run, started_wall, recovery["prior_charge_seconds"], active_clock=True)
    guard.phase = "BC"
    prior_A = next(row["elapsed_seconds"] for row in read_jsonl(prior/"events.jsonl") if row["kind"] == "phase_A_complete")
    guard.phase_started_monotonic -= recovery["prior_charge_seconds"]-prior_A
    try:
        environment = package_inventory() | {"device":"cpu","dtype":"float32","threads":8,"interop_threads":1,
            "deterministic_algorithms":True,"network":"blocked"}
        if environment != json_read(prior/"environment.json"):
            raise ValueError("Recovery environment differs from original")
        json_write(run/"environment.json",environment)
        emit(run,"recovery_started",prior_charge_seconds=recovery["prior_charge_seconds"],
             source_run=prior.name,clock="monotonic active elapsed with prior pause-adjusted charge")
        for name in HELPERS:
            if digest_bytes((run/"source_snapshot/base"/name).read_bytes()) != digest_bytes((ROOT/"notebook_src/stair"/name).read_bytes()):
                raise ValueError("Scientific helper changed before recovery: "+name)
        memberships={"profiles":{"preferred":{"train":data["train935"],"dev":data["tune"]+data["screen"]}}}
        tokenizer,paths,tokens=tokenizer_admission(data["corpus"],data["hierarchy"],memberships)
        if tokens != json_read(run/"tokenizer_admission.json"):
            raise ValueError("Recovery token admission differs")
        trie=TargetTrie(paths)
        diagnostic=[row for row in data["screen"] if row["query_id"] in set(data["diagnostic_ids"])]
        cells=json_read(run/"cells.json")
        cache={}
        for key,cell in cells.items():
            if cell["kind"] != "A":
                raise ValueError("Unexpected screened training cell before recovery")
            for mode in ["constrained","exact","unconstrained"]:
                cache[(cell["checkpoint_sha256"],cell["method"],mode)]=(key,read_jsonl(run/"predictions"/key/(mode+".jsonl")))
        trajectories=json_read(run/"trajectories.json")
        if {(t["pool"],t["method"]) for t in trajectories} != {(256,"D"),(256,"E"),(256,"F"),(935,"D")}:
            raise ValueError("Recovery completed-trajectory matrix changed")
        for t in trajectories:
            for row in t["history"]:
                if digest_bytes((run/row["checkpoint"]/"model.safetensors").read_bytes()) != row["checkpoint_sha256"]:
                    raise ValueError("Completed checkpoint changed before recovery")
        timings={}
        for fmt in "EF":
            model=model_for(fmt)
            optimizer=make_optimizer(model)
            order=sorted(data["train935"],key=lambda row:len(tokenizer.encode(prompt_text(fmt,row["query"],data["hierarchy"]))))
            tick=time.monotonic()
            train_update(model,tokenizer,optimizer,fmt,order[:4]+order[-4:],data["hierarchy"],"cpu",4)
            timings[fmt]=time.monotonic()-tick
            del model,optimizer
            release_model()
        projected=sum(timings.values())*384+600
        guard.admit(projected,"Recovery: fresh n935 E/F fits, full screening and finalization reserve")
        json_write(run/"recovery_admission.json",{"update_seconds":timings,"projected_remaining_seconds":projected,
            "incomplete_E_restarts_from_base":True,"pilot_weights_discarded":True})
        batch=json_read(run/"exact_admission.json")["candidate_batch"]
        complete_training_and_screen(run,data,tokenizer,paths,trie,diagnostic,guard,cells,cache,batch)
        if protected_artifacts() != json_read(run/"protected_artifacts.json"):
            raise ValueError("Protected historical artifacts changed during recovery")
        json_write(run/"status.json",{"status":"complete","elapsed_seconds":guard.elapsed(),
            "recovery_active_seconds":time.monotonic()-guard.started_monotonic,
            "recovery_wall_seconds":time.time()-started_wall,"clock_accounting":"pause-adjusted prior plus monotonic recovery",
            "original_attempt_wall_seconds":json_read(prior/"process_status.json")["elapsed_seconds"],
            "backend":"cpu","dtype":"float32","new_test_predictions":0,"qwen_quality_stage":"deferred by user",
            "original_artifacts_unchanged":True})
        emit(run,"complete",elapsed_seconds=guard.elapsed())
    except Exception as error:
        import traceback
        json_write(run/"failure.json",{"error":str(error),"traceback":traceback.format_exc(),"phase":guard.phase,
            "elapsed_seconds":guard.elapsed()})
        json_write(run/"status.json",{"status":"incomplete","error":str(error),"new_test_predictions":0})
        raise
    finally:
        guard.close()


def complete_training_and_screen(run, data, tokenizer, paths, trie, diagnostic, guard, cells, cache, batch):
    trajectories = json_read(run / "trajectories.json") if (run / "trajectories.json").exists() else []
    for pool in [256, 935]:
        rows = data["train" + str(pool)]
        for fmt in "DEF":
            if any(t["pool"] == pool and t["method"] == fmt for t in trajectories):
                continue
            model = model_for(fmt)
            trajectory_started = time.monotonic()
            optimizer = make_optimizer(model)
            stream = training_groups(rows)
            history, block_losses = [], []
            tick_block = time.monotonic()
            for update in range(1, 385):
                guard.check()
                loss = train_update(model, tokenizer, optimizer, fmt, next(stream), data["hierarchy"], "cpu", 4)
                block_losses.append(loss)
                if update % 32:
                    continue
                train_seconds = time.monotonic() - tick_block
                eval_started = time.monotonic()
                tune, _ = beam_queries(model, tokenizer, trie, fmt, "constrained", data["tune"], data["hierarchy"], guard)
                score = quality(tune, data["tune"], data["corpus"])
                tuning_seconds = time.monotonic() - eval_started
                save_started = time.monotonic()
                path = run / "checkpoints" / f"n{pool}_{fmt}" / f"step-{update}"
                path.mkdir(parents=True)
                model.save_pretrained(path, safe_serialization=True)
                checkpoint_sha = digest_bytes((path / "model.safetensors").read_bytes())
                row = {"pool": pool, "method": fmt, "update": update, "example_passes": update * 8,
                       "train_loss": float(np.mean(block_losses)), "training_seconds": train_seconds,
                       "tuning_seconds": tuning_seconds, "checkpoint_seconds": time.monotonic() - save_started,
                       "tune_r1": score["r1"], "checkpoint": str(path.relative_to(run)), "checkpoint_sha256": checkpoint_sha}
                history.append(row)
                save_records(run, f"tune_n{pool}_{fmt}_{update}", "constrained", tune)
                if pool == 256 and update <= 128:
                    old = json_read(OLD_RUN / "training" / (fmt + ".json"))["history"][update // 32 - 1]
                    if not np.isclose(old["train_loss"], row["train_loss"], atol=1e-6, rtol=1e-6) or old["dev_r1"] != row["tune_r1"]:
                        raise ValueError(f"Historical prefix loss/development parity failed: {fmt}/{update}")
                    if checkpoint_sha != old["checkpoint_sha256"]:
                        # Do not conflate serialization differences with numerical weight differences.
                        old_path = OLD_RUN / old["checkpoint"] / "model.safetensors"
                        with safe_open(old_path, framework="pt", device="cpu") as old_weights:
                            current = model.state_dict()
                            if any(not torch.equal(old_weights.get_tensor(name), current[name]) for name in old_weights.keys()):
                                raise ValueError(f"Historical prefix tensor parity failed: {fmt}/{update}")
                    row["historical_prefix_parity"] = "passed"
                if (run / "recovery.json").exists() and pool == 935 and fmt == "E" and update <= 256:
                    prior_run = Path(json_read(run / "recovery.json")["source_run"])
                    prior = read_jsonl(prior_run / "histories/n935_E.jsonl")[update // 32 - 1]
                    if not np.isclose(prior["train_loss"], row["train_loss"], atol=1e-6, rtol=1e-6) or prior["tune_r1"] != row["tune_r1"]:
                        raise ValueError("Interrupted E trajectory numerical replay failed")
                    if checkpoint_sha != prior["checkpoint_sha256"]:
                        with safe_open(prior_run / prior["checkpoint"] / "model.safetensors", framework="pt", device="cpu") as old_weights:
                            current = model.state_dict()
                            if any(not torch.equal(old_weights.get_tensor(name), current[name]) for name in old_weights.keys()):
                                raise ValueError("Interrupted E trajectory weight replay failed")
                    row["interrupted_attempt_prefix_parity"] = "passed"
                append_jsonl(run / "histories" / f"n{pool}_{fmt}.jsonl", row)
                emit(run, "training_progress", **row)
                block_losses = []
                tick_block = time.monotonic()
            selected_budgets = {}
            for budget in [128, 384]:
                best = max((row for row in history if row["update"] <= budget), key=lambda r: (r["tune_r1"], -r["update"]))
                selected_budgets[str(budget)] = best
            trajectory = {"pool": pool, "method": fmt, "history": history, "selected": selected_budgets,
                          "shadow_stopping": shadow_selection(history),
                          "actual_trajectory_seconds": time.monotonic() - trajectory_started}
            trajectories.append(trajectory)
            json_write(run / "trajectories.json", trajectories)
            del optimizer, model
            release_model()
            emit(run, "trajectory_complete", pool=pool, method=fmt, selected=selected_budgets,
                 shadow_stopping=trajectory["shadow_stopping"])
    for trajectory in trajectories:
        fmt, pool = trajectory["method"], trajectory["pool"]
        selections = dict(trajectory["selected"])
        shadow = trajectory["shadow_stopping"]
        selections["shadow"] = next(row for row in trajectory["history"] if row["update"] == shadow["selected_update"])
        for budget_text, best in selections.items():
            budget = int(budget_text) if budget_text != "shadow" else "shadow"
            sha = best["checkpoint_sha256"]
            key = f"n{pool}_u{budget}_{fmt}"
            limit = budget if isinstance(budget, int) else shadow["would_stop_at_update"]
            charged = sum(row["training_seconds"] + row["tuning_seconds"] + row["checkpoint_seconds"]
                          for row in trajectory["history"] if row["update"] <= limit)
            cells[key] = {"kind": "BC", "method": fmt, "pool": pool, "budget": budget,
                          "selected_update": best["update"], "checkpoint": best["checkpoint"], "checkpoint_sha256": sha,
                          "charged_training_seconds": charged,
                          "training_cost_kind": "measured budget-prefix" if isinstance(budget, int) else "hypothetical stopping-policy replay"}
            need = [mode for mode in ["constrained", "unconstrained"] if (sha, fmt, mode) not in cache]
            if need:
                model = model_for(fmt, run / best["checkpoint"])
                # Selected-checkpoint reload and batched/singleton equality before screen inference.
                reference = read_jsonl(run / "predictions" / f"tune_n{pool}_{fmt}_{best['update']}" / "constrained.jsonl")[:8]
                reloaded, _ = beam_queries(model, tokenizer, trie, fmt, "constrained", data["tune"][:8], data["hierarchy"], guard)
                singleton, _ = run_neural_queries(model, tokenizer, trie, fmt, "constrained", data["tune"][:8], data["hierarchy"], "cpu", 1, guard)
                if not equal_decoding(reference, reloaded) or not equal_decoding(reloaded, singleton):
                    raise ValueError("Selected checkpoint reload/singleton parity failed")
                unc_batch, _ = beam_queries(model, tokenizer, trie, fmt, "unconstrained", data["tune"][:8], data["hierarchy"], guard)
                unc_single, _ = run_neural_queries(model, tokenizer, trie, fmt, "unconstrained", data["tune"][:8], data["hierarchy"], "cpu", 1, guard)
                if not equal_decoding(unc_batch, unc_single):
                    raise ValueError("Unconstrained selected-checkpoint singleton parity failed")
                for mode in need:
                    queries = data["screen"] if mode == "constrained" else diagnostic
                    records, _ = beam_queries(model, tokenizer, trie, fmt, mode, queries, data["hierarchy"], guard)
                    cache[(sha, fmt, mode)] = (key, records)
                del model
                release_model()
            for mode in ["constrained", "unconstrained"]:
                origin, records = cache[(sha, fmt, mode)]
                if origin != key:
                    cells[key].setdefault("reused_from", {})[mode] = origin
                save_records(run, key, mode, records)
            emit(run, "BC_screen_complete", cell=key,
                 metrics=quality(cache[(sha, fmt, "constrained")][1], data["screen"], data["corpus"]))
            json_write(run / "cells.json", cells)
    # Exact-scoring interactions: only if Pilot A has a prespecified lead.
    leads = []
    for fmt in "DEF":
        sha = cells["original_" + fmt]["checkpoint_sha256"]
        approx = cache[(sha, fmt, "constrained")][1]
        exact = cache[(sha, fmt, "exact")][1]
        qa, qe = quality(approx, data["screen"], data["corpus"]), quality(exact, data["screen"], data["corpus"])
        paired = paired_summary(approx, exact, data["screen"], data["corpus"])
        inconclusive = paired["ci95_pp"][0] <= -5 and paired["ci95_pp"][1] >= 5
        time_a, time_e = prediction_seconds(approx), prediction_seconds(exact)
        if not inconclusive and (qe["r1"] - qa["r1"] >= .05 or (time_e <= .75 * time_a and qe["r1"] >= qa["r1"] - .01)):
            leads.append(fmt)
    if leads:
        choices = []
        for key, cell in cells.items():
            if cell["kind"] != "BC" or (cell["pool"] == 256 and cell["budget"] == 128):
                continue
            recs = read_jsonl(run / "predictions" / key / "constrained.jsonl")
            choices.append((quality(recs, data["screen"], data["corpus"])["r1"], prediction_seconds(recs), cell["charged_training_seconds"], key))
        chosen, seen = [], set()
        remaining = list(choices)
        while remaining and len(chosen) < 2:
            top_score = max(row[0] for row in remaining)
            near = [row for row in remaining if row[0] >= top_score - .01]
            score, seconds, train_cost, key = min(near, key=lambda x: (x[1], x[2], x[3]))
            identity = (cells[key]["method"], cells[key]["checkpoint_sha256"])
            if identity not in seen:
                chosen.append(key)
                seen.add(identity)
            remaining = [row for row in remaining if (cells[row[3]]["method"], cells[row[3]]["checkpoint_sha256"]) != identity]
        json_write(run / "interaction_selection.json", {"exact_leading_formats": leads,
            "criterion": "highest absolute screening R@1 configurations; within 1pp prefer lower query/training cost; distinct format/checkpoint; exploratory",
            "candidates": chosen})
        for key in chosen:
            fmt = cells[key]["method"]
            cell = cells[key]
            cached = cache.get((cell["checkpoint_sha256"], fmt, "exact"))
            if cached:
                origin, recs = cached
                cell.setdefault("reused_from", {})["exact"] = origin
            else:
                model = model_for(fmt, run / cell["checkpoint"])
                recs = []
                for query in data["screen"]:
                    guard.check()
                    recs.append(exact_record(model, tokenizer, fmt, query, data["hierarchy"], paths, batch))
                cache[(cell["checkpoint_sha256"], fmt, "exact")] = (key, recs)
                del model
                release_model()
            save_records(run, key, "exact", recs)
            emit(run, "interaction_complete", cell=key, metrics=quality(recs, data["screen"], data["corpus"]))
    else:
        json_write(run / "interaction_selection.json", {"candidates": [], "reason": "Pilot A had no quality/efficiency lead under the predeclared thresholds"})
    json_write(run / "cells.json", cells)
