class RunGuard:
    def __init__(self, device, run):
        self.device = device
        self.run = run
        self.started = time.monotonic()
        self.deadline = self.started + CFG.measured_budget_minutes * 60
        self.swap_baseline = psutil.swap_memory().used
        self.records = []
        self.violation = None
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._watch, daemon=True)
        self.worker.start()

    def _watch(self):
        while not self.stop_event.is_set():
            self.sample()
            self.stop_event.wait(0.5)

    def sample(self):
        virtual = psutil.virtual_memory()
        row = {"elapsed_seconds": time.monotonic() - self.started,
               "rss_bytes": psutil.Process().memory_info().rss, "available_bytes": virtual.available,
               "swap_used_bytes": psutil.swap_memory().used}
        if self.device == "mps":
            row["mps_tensor_bytes"] = torch.mps.current_allocated_memory()
            row["mps_driver_bytes"] = torch.mps.driver_allocated_memory()
            row["mps_recommended_bytes"] = torch.mps.recommended_max_memory()
            if row["mps_driver_bytes"] > min(48 * 2**30, 0.8 * row["mps_recommended_bytes"]):
                self.violation = "MPS driver allocation ceiling"
        if row["rss_bytes"] > CFG.rss_limit_gib * 2**30:
            self.violation = "Process RSS ceiling"
        if row["available_bytes"] < CFG.available_floor_gib * 2**30:
            self.violation = "System available-memory floor"
        if row["swap_used_bytes"] - self.swap_baseline > CFG.swap_growth_limit_gib * 2**30:
            self.violation = "System swap-growth ceiling"
        if time.monotonic() > self.deadline:
            self.violation = "Automated execution deadline"
        self.records.append(row)
        return row

    def check(self):
        self.sample()
        if self.violation:
            raise RuntimeError(self.violation)

    def close(self):
        self.stop_event.set()
        self.worker.join(timeout=2)
        json_write(self.run / "memory_samples.json", {"samples": self.records, "violation": self.violation,
                   "interpretation": "sampled counters, not exact peaks; shared CPU/GPU figures are not additive"})


def make_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=CFG.learning_rate, betas=(0.9, 0.999),
                             eps=1e-8, weight_decay=0.01, foreach=False, fused=False)


def run_neural_queries(model, tokenizer, trie, fmt, mode, queries, hierarchy, device, batch_size, guard=None):
    records, timings = [], []
    for start in range(0, len(queries), batch_size):
        if guard:
            guard.check()
        output, timing = decode_batch(model, tokenizer, trie, fmt, mode, queries[start:start + batch_size], hierarchy, device)
        records.extend(output)
        timings.append(timing)
    return records, timings


def equal_decoding(left, right):
    if len(left) != len(right):
        return False
    for a, b in zip(left, right):
        aa, bb = a["hypotheses"], b["hypotheses"]
        if len(aa) != len(bb):
            return False
        for x, y in zip(aa, bb):
            if x["token_ids"] != y["token_ids"] or x["validity"] != y["validity"] or x["section_id"] != y["section_id"]:
                return False
            if not np.isclose(x["score"], y["score"], atol=1e-4, rtol=1e-4):
                return False
    return True


def pilot_configuration(device, tokenizer, trie, hierarchy, memberships, run, guard):
    started = time.monotonic()
    train = memberships["profiles"]["preferred"]["train"]
    dev = memberships["profiles"]["preferred"]["dev"][:8]
    train_costs = defaultdict(dict)
    decode_costs = defaultdict(dict)
    batch_parity = {1: True, 2: True, 4: True}
    history = []
    for fmt in "DEF":
        seed_model(CFG.seed, device)
        model = load_base(device)
        optimizer = make_optimizer(model)
        ordered = sorted(train, key=lambda row: len(prompt_text(fmt, row["query"], hierarchy)))
        screen_rows = [ordered[0], ordered[len(ordered) // 2], ordered[-1]] + train[:5]
        for microbatch, accumulation in CFG.batch_accumulation_candidates:
            guard.check()
            synchronize(device)
            tick = time.monotonic()
            loss = train_update(model, tokenizer, optimizer, fmt, screen_rows, hierarchy, device, microbatch)
            synchronize(device)
            train_costs[fmt][microbatch] = time.monotonic() - tick
            history.append({"format": fmt, "kind": "batch_screen", "microbatch": microbatch, "loss": loss,
                            "update_seconds": train_costs[fmt][microbatch]})
        del optimizer, model
        release_model()
    microbatch = min((1, 2, 4), key=lambda size: (sum(train_costs[fmt][size] for fmt in "DEF"), size))
    for fmt in "CDEF":
        seed_model(CFG.seed, device)
        model = load_base(device)
        if fmt != "C":
            optimizer = make_optimizer(model)
            before = teacher_loss(model, tokenizer, fmt, train[:8], hierarchy, device)
            # At least sixteen forward/backward microbatches at the selected setting.
            update_times = []
            for index in range(max(2, 16 * microbatch // 8)):
                guard.check()
                tick = time.monotonic()
                train_update(model, tokenizer, optimizer, fmt, train[:8], hierarchy, device, microbatch)
                synchronize(device)
                update_times.append(time.monotonic() - tick)
            train_costs[fmt]["p90_update"] = float(np.quantile(update_times, .9))
            after = teacher_loss(model, tokenizer, fmt, train[:8], hierarchy, device)
            if not after < before:
                raise ValueError("Pilot failed to reduce teacher-forced loss")
            history.append({"format": fmt, "kind": "loss_reduction", "before": before, "after": after,
                            "optimization_sanity_only": True})
            del optimizer
        for mode in ("constrained", "unconstrained"):
            baseline, base_timings = run_neural_queries(model, tokenizer, trie, fmt, mode, dev, hierarchy, device, 1, guard)
            decode_costs[fmt + "/" + mode][1] = [row["end_to_end_seconds"] for row in base_timings]
            for batch_size in (2, 4):
                try:
                    candidate, timings = run_neural_queries(model, tokenizer, trie, fmt, mode, dev, hierarchy, device, batch_size, guard)
                    parity = equal_decoding(baseline, candidate)
                    batch_parity[batch_size] &= parity
                    decode_costs[fmt + "/" + mode][batch_size] = [row["end_to_end_seconds"] for row in timings]
                    if parity:
                        while len(decode_costs[fmt + "/" + mode][batch_size]) < 8:
                            _, repeated = run_neural_queries(model, tokenizer, trie, fmt, mode, dev,
                                                            hierarchy, device, batch_size, guard)
                            decode_costs[fmt + "/" + mode][batch_size].extend(row["end_to_end_seconds"] for row in repeated)
                    history.append({"format": fmt, "mode": mode, "kind": "batch_parity", "batch_size": batch_size, "passed": parity})
                except Exception as error:
                    batch_parity[batch_size] = False
                    history.append({"format": fmt, "mode": mode, "kind": "batch_parity", "batch_size": batch_size, "passed": False, "error": repr(error)})
            # Same four-beam search, differing only in how many final hypotheses are returned.
            singles, _ = decode_batch(model, tokenizer, trie, fmt, mode, dev[:1], hierarchy, device, return_count=1)
            if singles[0]["hypotheses"][0]["token_ids"] != baseline[0]["hypotheses"][0]["token_ids"]:
                raise ValueError("Top output changes with number of returned sequences")
        del model
        release_model()
        if time.monotonic() - guard.started > CFG.pilot_budget_minutes * 60:
            raise RuntimeError("Pilot allowance exhausted")
    allowed = [size for size in (1, 2, 4) if batch_parity[size]]
    inference_batch = min(allowed, key=lambda size: sum(np.quantile(values[size], .9) / size for values in decode_costs.values()))
    # Eight completed call timings per format/mode at the chosen setting. Singleton
    # baseline already has eight; batched settings receive repeated fixed dev sets.
    p90_decode = {key: float(np.quantile(values[inference_batch], .9)) for key, values in decode_costs.items()}
    elapsed = time.monotonic() - guard.started
    projections = {}
    chosen_profile = None
    for name, sizes in (("preferred", CFG.preferred_profile), ("fallback", CFG.fallback_profile)):
        training = sum(train_costs[fmt]["p90_update"] * sizes["epochs"] * (sizes["train"] / 8) for fmt in "DEF")
        dev_cost = sum(p90_decode[fmt + "/constrained"] * sizes["epochs"] * np.ceil(sizes["dev"] / inference_batch) for fmt in "DEF")
        test_cost = sum(value * np.ceil(sizes["test"] / inference_batch) for value in p90_decode.values())
        # Additional measured load/save, fixed diagnostic and reload checks.
        overhead = 8 * 60
        projected = training + dev_cost + test_cost + overhead
        projections[name] = {"training_seconds": training, "development_seconds": dev_cost,
                             "test_seconds": test_cost, "load_save_diagnostic_reserve_seconds": overhead,
                             "admission_bound_seconds": elapsed + 1.25 * projected + 20 * 60}
        if chosen_profile is None and projections[name]["admission_bound_seconds"] <= CFG.measured_budget_minutes * 60:
            chosen_profile = name
    result = {"device": device, "microbatch": microbatch, "accumulation": 8 // microbatch,
              "inference_batch": inference_batch, "batch_parity": batch_parity, "profile": chosen_profile,
              "train_costs": dict(train_costs), "p90_batch_decode_seconds": p90_decode,
              "projection": projections, "elapsed_seconds": time.monotonic() - started,
              "history": history, "pilot_weights_discarded": True}
    json_write(run / ("pilot_" + device + ".json"), result)
    if time.monotonic() - guard.started > CFG.pilot_budget_minutes * 60 or chosen_profile is None:
        raise RuntimeError("No sample profile fits the admitted runtime envelope")
    return result
