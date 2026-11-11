def seal_run(run):
    files = sorted(path for path in run.rglob("*") if path.is_file() and path.name != "run_manifest.json")
    json_write(run / "run_manifest.json", {"files": [{"path": str(path.relative_to(run)),
                                                     "sha256": digest_bytes(path.read_bytes())} for path in files]})


def save_prediction(run, record):
    filename = record["query_id"].replace(":", "_") + ".json"
    target = run / "predictions" / record["method"] / record["mode"] / filename
    if target.exists():
        raise ValueError("An inference record would be overwritten")
    json_write(target, record)


def train_arm(fmt, tokenizer, trie, hierarchy, selected, config, run, guard):
    device, microbatch, batch_size = config["device"], config["microbatch"], config["inference_batch"]
    seed_model(CFG.seed, device)
    started = time.monotonic()
    model = load_base(device)
    optimizer = make_optimizer(model)
    initial_hash = digest_bytes((CFG.model_folder / "model.safetensors").read_bytes())
    history, dev_by_epoch = [], {}
    epochs = (CFG.preferred_profile if config["profile"] == "preferred" else CFG.fallback_profile)["epochs"]
    updates = 0
    for epoch in range(1, epochs + 1):
        order = np.random.default_rng(CFG.order_seed + epoch).permutation(len(selected["train"]))
        ordered = [selected["train"][int(i)] for i in order]
        losses = []
        epoch_started = time.monotonic()
        for offset in range(0, len(ordered), CFG.effective_batch):
            guard.check()
            group = ordered[offset:offset + CFG.effective_batch]
            losses.append(train_update(model, tokenizer, optimizer, fmt, group, hierarchy, device, microbatch))
            updates += 1
        synchronize(device)
        train_seconds = time.monotonic() - epoch_started
        dev_started = time.monotonic()
        outputs, timings = run_neural_queries(model, tokenizer, trie, fmt, "constrained", selected["dev"],
                                              hierarchy, device, batch_size, guard)
        dev_score = float(score_records(outputs, selected["dev"], hierarchy["corpus"]).r1.mean())
        checkpoint = run / "checkpoints" / fmt / f"epoch-{epoch}"
        model.save_pretrained(checkpoint, safe_serialization=True)
        checkpoint_hash = digest_bytes((checkpoint / "model.safetensors").read_bytes())
        dev_by_epoch[epoch] = outputs
        history.append({"method": fmt, "epoch": epoch, "optimizer_updates": updates,
                        "train_loss": float(np.mean(losses)), "dev_r1": dev_score,
                        "training_seconds": train_seconds, "development_and_save_seconds": time.monotonic() - dev_started,
                        "checkpoint": str(checkpoint.relative_to(run)), "checkpoint_sha256": checkpoint_hash})
        json_write(run / "training" / f"{fmt}.json", {"base_sha256": initial_hash, "history": history})
        print(f"{fmt} epoch {epoch}/{epochs}: loss={np.mean(losses):.4f}; dev R@1={dev_score:.3f}; updates={updates}", flush=True)
    best = max(history, key=lambda row: (row["dev_r1"], -row["epoch"]))
    del optimizer, model
    release_model()
    checkpoint = run / best["checkpoint"]
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, local_files_only=True, dtype=torch.float32,
                                                 attn_implementation="eager").to(device)
    diagnostics = selected["dev"][:8]
    reloaded, _ = run_neural_queries(model, tokenizer, trie, fmt, "constrained", diagnostics,
                                    hierarchy, device, batch_size, guard)
    if not equal_decoding(dev_by_epoch[best["epoch"]][:8], reloaded):
        raise ValueError("Selected checkpoint save/reload prediction mismatch")
    for mode in ("constrained", "unconstrained"):
        batched, _ = run_neural_queries(model, tokenizer, trie, fmt, mode, diagnostics, hierarchy, device, batch_size, guard)
        singleton, _ = run_neural_queries(model, tokenizer, trie, fmt, mode, diagnostics, hierarchy, device, 1, guard)
        if not equal_decoding(batched, singleton):
            raise ValueError("Trained-checkpoint singleton parity failed before test inference")
    fit, _ = run_neural_queries(model, tokenizer, trie, fmt, "constrained", selected["train"][:16], hierarchy, device, batch_size, guard)
    fit_r1 = float(score_records(fit, selected["train"][:16], hierarchy["corpus"]).r1.mean())
    del model
    release_model()
    summary = {"method": fmt, "selected_epoch": best["epoch"], "selected_optimizer_updates": best["optimizer_updates"],
               "maximum_optimizer_updates": updates, "selected_checkpoint": best["checkpoint"],
               "checkpoint_sha256": best["checkpoint_sha256"], "base_sha256": initial_hash,
               "train_diagnostic_r1": fit_r1, "elapsed_seconds": time.monotonic() - started,
               "reload_and_singleton_checks": "passed"}
    json_write(run / "training" / f"{fmt}_selection.json", summary)
    return summary


def run_experiment():
    validate_model_cache()
    audit = audit_release()
    hierarchy = build_anchor_hierarchy(audit)
    corpus = read_jsonl(CFG.data_folder / "raw/finance/corpus.jsonl")
    hierarchy["corpus"] = corpus
    frozen = json_read(CFG.data_folder / "frozen_memberships.json")
    semantic = json_read(CFG.data_folder / "semantic_audit.json")
    # Recompute the prospective membership to reject stale caches before fitting.
    recalculated = freeze_memberships(audit)
    if frozen != recalculated:
        raise ValueError("Frozen membership differs from the declared duplicate/hash policy")
    for split in ("train", "dev", "test"):
        expected = {row["query_id"] for row in frozen["profiles"]["preferred"][split][:20]}
        actual = {row["query_id"] for row in semantic["rows"] if row["split"] == split}
        if actual != expected:
            raise ValueError("Semantic audit does not match frozen prospective membership")
    if semantic["systematic_label_failure_found"]:
        raise ValueError("Systematic semantic-label failure requires plan revision")
    run_id = "anchor-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run = CFG.run_folder / run_id
    run.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(CFG.root).free < 20 * 2**30:
        raise RuntimeError("Less than the reserved 20 GiB workspace disk is available")
    json_write(run / "frozen_cfg.json", {key: str(value) if isinstance(value, Path) else value
               for key, value in vars(CFG).items() if not key.startswith("_")})
    json_write(run / "optimizer_and_loss_contract.json", {"optimizer": "AdamW", "learning_rate": CFG.learning_rate,
               "betas": [0.9, 0.999], "epsilon": 1e-8, "weight_decay": .01, "schedule": "constant",
               "gradient_clip_norm": 1.0, "fused": False, "foreach": False,
               "loss": "mean nonpadding target-token loss including EOS per example; equal example mean per update",
               "partial_group": "divide by actual example count; no carry across epochs", "dropout": "native pretrained"})
    for name, value in [("data_audit.json", audit), ("anchor_hierarchy.json", {k: v for k, v in hierarchy.items() if k != "corpus"}),
                        ("frozen_memberships.json", frozen), ("semantic_audit.json", semantic),
                        ("source_manifest.json", json_read(CFG.data_folder / "source_manifest.json")),
                        ("source_book_manifest.json", json_read(CFG.data_folder / "source_book/manifest.json")),
                        ("model_manifest.json", json_read(CFG.data_folder / "model_manifest.json"))]:
        json_write(run / name, value)
    pd.DataFrame(audit["crosswalk"]).to_csv(run / "target_crosswalk.csv", index=False)
    decision = CFG.root / "docs/stair-section-boundary-decision.md"
    if decision.exists():
        shutil.copy2(decision, run / "section-boundary-decision.md")
    torch.use_deterministic_algorithms(True)
    device = "mps" if torch.backends.mps.is_built() and torch.backends.mps.is_available() else "cpu"
    torch.set_num_threads(4 if device == "mps" else 8)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # A fresh Run All sets this once; repeated report never invokes it.
    guard = RunGuard(device, run)
    stage = "input_admission"
    json_write(run / "status.json", {"status": "running", "stage": stage, "device": device})
    json_write(CFG.run_folder / "latest.json", {"run_id": run_id, "status": "running"})
    try:
        tokenizer, sequences, token_report = tokenizer_admission(corpus, hierarchy, frozen)
        trie = TargetTrie(sequences)
        json_write(run / "tokenizer_admission.json", token_report)
        stage = "pilot"
        print("Starting bounded training/dev-only pilot on", device, flush=True)
        try:
            config = pilot_configuration(device, tokenizer, trie, hierarchy, frozen, run, guard)
        except RuntimeError as pilot_error:
            if device != "mps" or guard.violation:
                raise
            json_write(run / "mps_admission_failure.json", {"stage": "training/dev pilot before any test inference",
                       "error": str(pilot_error), "policy": "strict determinism retained; no unsupported-op CPU fallback",
                       "next": "separately pilot a whole-run CPU backend within the same remaining clock"})
            print("MPS admission failed; piloting whole-run CPU:", str(pilot_error), flush=True)
            device = "cpu"
            guard.device = device
        if device == "cpu" and (run / "mps_admission_failure.json").exists():
            release_model()
            torch.set_num_threads(8)
            config = pilot_configuration(device, tokenizer, trie, hierarchy, frozen, run, guard)
        selected = frozen["profiles"][config["profile"]]
        manifest = {"target_contract": CFG.target_contract, "profile": config["profile"], "device": device,
                    "dtype": "float32", "microbatch": config["microbatch"], "accumulation": config["accumulation"],
                    "effective_batch": 8, "inference_batch": config["inference_batch"],
                    "model_revision": json_read(CFG.data_folder / "model_manifest.json")["revision"],
                    "seed": CFG.seed, "selected_membership": {split: [row["query_id"] for row in rows] for split, rows in selected.items()},
                    "test_predictions_before_freeze": 0, "decoder": {"beams": 4, "returns": 3, "early_stopping": "never",
                    "length_penalty": 1.0, "renormalize_logits": False, "max_new_tokens": CFG.decoder_cap},
                    "environment": {**package_inventory(), "torch": torch.__version__,
                                    "mps_built": torch.backends.mps.is_built(), "mps_available": torch.backends.mps.is_available(),
                                    "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads()},
                    "model_mode_order": "A/B lexical, then C/D/E/F; within each neural model alternate constrained/unconstrained blocks of up to eight frozen queries"}
        json_write(run / "frozen_run_manifest.json", manifest)
        json_write(run / "selected_membership.json", selected)
        json_write(run / "rng_seeds.json", {"model": CFG.seed, "epoch_order": CFG.order_seed,
                                           "bootstrap": CFG.bootstrap_seed, "sampling": "SHA256 stair-v1; no PRNG"})
        print("Frozen:", {key: manifest[key] for key in ["profile", "device", "microbatch", "inference_batch"]}, flush=True)
        stage = "training"
        selections = [train_arm(fmt, tokenizer, trie, hierarchy, selected, config, run, guard) for fmt in "DEF"]
        json_write(run / "checkpoint_selection.json", selections)
        # All training/checkpoint/singleton gates have passed before any test prediction.
        stage = "test_inference"
        json_write(run / "status.json", {"status": "running", "stage": stage, "device": device})
        records = lexical_rankings(corpus, selected["test"])
        for row in records:
            save_prediction(run, row)
        for fmt in "CDEF":
            seed_model(CFG.seed, device)
            if fmt == "C":
                model = load_base(device)
            else:
                path = run / next(row["selected_checkpoint"] for row in selections if row["method"] == fmt)
                model = AutoModelForSeq2SeqLM.from_pretrained(path, local_files_only=True, dtype=torch.float32,
                                                             attn_implementation="eager").to(device)
            tick = time.monotonic()
            for offset in range(0, len(selected["test"]), 8):
                block = selected["test"][offset:offset + 8]
                for mode in ("constrained", "unconstrained"):
                    outputs, _ = run_neural_queries(model, tokenizer, trie, fmt, mode, block, hierarchy, device,
                                                    config["inference_batch"], guard)
                    for record in outputs:
                        save_prediction(run, record)
                    records.extend(outputs)
            print(f"{fmt} test modes finished in {time.monotonic() - tick:.1f}s", flush=True)
            del model
            release_model()
        stage = "analysis"
        analysis = compute_analysis(records, selected, corpus, semantic)
        json_write(run / "analysis.json", analysis)
        pd.DataFrame(analysis["summary"]).to_csv(run / "metrics.csv", index=False)
        pd.DataFrame(analysis["per_query"]).to_csv(run / "per_query_metrics.csv", index=False)
        (run / "predictions.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in records))
        json_write(run / "execution_time.json", {"automated_seconds_before_exports": time.monotonic() - guard.started,
                                               "pilot_seconds": config["elapsed_seconds"],
                                               "training_and_development_seconds": sum(row["elapsed_seconds"] for row in selections),
                                               "preparation_and_authoring_excluded_and_reported_separately": True})
        status = {"status": "complete", "target_contract": CFG.target_contract, "device": device,
                  "profile": config["profile"], "test_queries": len(selected["test"]),
                  "test_records": len(records), "neural_test_predictions": 8 * len(selected["test"]),
                  "historical_v3_source_leaf_gate": "failed and preserved", "v4_boundary_warnings": 9,
                  "stage": "numerical_experiment_complete; exports/replay verification follow"}
        json_write(run / "status.json", status)
        json_write(CFG.run_folder / "latest.json", {"run_id": run_id, "status": "complete"})
        return {"run": run, "audit": audit, "hierarchy": hierarchy, "memberships": selected,
                "analysis": analysis, "records": records, "config": manifest, "selections": selections}
    except Exception as error:
        json_write(run / "status.json", {"status": "incomplete", "stage": stage, "error": repr(error),
                                        "device": device, "no_unmatched_winner_claim": True})
        json_write(CFG.run_folder / "latest.json", {"run_id": run_id, "status": "incomplete"})
        raise
    finally:
        guard.close()
        seal_run(run)
