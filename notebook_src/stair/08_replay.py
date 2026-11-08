def load_experiment():
    validate_inputs()
    latest = json_read(CFG.run_folder / "latest.json")
    run = CFG.run_folder / (CFG.resume_run_id or latest["run_id"])
    status = json_read(run / "status.json")
    if status.get("status") != "complete" or status.get("target_contract") != CFG.target_contract:
        raise ValueError("Complete v4 evidence is required for report mode; no training or download is started")
    manifest = json_read(run / "run_manifest.json")
    for row in manifest["files"]:
        if row["path"].startswith("checkpoints/"):
            continue
        path = run / row["path"]
        if not path.exists() or digest_bytes(path.read_bytes()) != row["sha256"]:
            raise ValueError("Immutable run evidence hash mismatch: " + row["path"])
    records = read_jsonl(run / "predictions.jsonl")
    selected = json_read(run / "selected_membership.json")
    corpus = read_jsonl(CFG.data_folder / "raw/finance/corpus.jsonl")
    semantic = json_read(run / "semantic_audit.json")
    analysis = compute_analysis(records, selected, corpus, semantic)
    if analysis != json_read(run / "analysis.json"):
        raise ValueError("Recomputed metrics differ from captured evidence")
    return {"run": run, "audit": json_read(run / "data_audit.json"),
            "hierarchy": json_read(run / "anchor_hierarchy.json"), "memberships": selected,
            "analysis": analysis, "records": records,
            "config": json_read(run / "frozen_run_manifest.json"),
            "selections": json_read(run / "checkpoint_selection.json")}
