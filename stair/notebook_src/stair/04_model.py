def synchronize(device):
    if str(device) == "mps":
        torch.mps.synchronize()


def seed_model(seed, device):
    random.seed(seed)
    torch.manual_seed(seed)
    if str(device) == "mps":
        torch.mps.manual_seed(seed)


def release_model(model=None):
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def load_base(device):
    manifest = json_read(CFG.data_folder / "model_manifest.json")
    if manifest["model"] != CFG.model:
        raise ValueError("Unexpected base model identity")
    model = AutoModelForSeq2SeqLM.from_pretrained(CFG.model_folder, local_files_only=True,
                                                 dtype=torch.float32, attn_implementation="eager")
    model.to(device)
    if any(parameter.device.type != str(device) or parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("Model placement/dtype admission failed")
    return model


def prompt_text(fmt, query, hierarchy):
    context = "" if fmt == "D" else hierarchy["flat_text"] if fmt == "E" else hierarchy["hierarchical_text"]
    return ("Return exactly one document title from Principles of Financial Accounting.\n"
            + context + "\nQuestion: " + query + "\nTitle:")


def tokenizer_admission(corpus, hierarchy, memberships):
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_folder, local_files_only=True, use_fast=True)
    original_limit = tokenizer.model_max_length
    tokenizer.model_max_length = CFG.encoder_cap  # Explicit admission cap, not a pretraining claim.
    sequences, title_by_tokens = [], {}
    for row in corpus:
        title = canonical(row["title"])
        tokens = tokenizer.encode(title, add_special_tokens=True)
        decoded = canonical(tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False))
        if tokenizer.unk_token_id in tokens or decoded != title or tuple(tokens) in title_by_tokens:
            raise ValueError(f"Unrepresentable or non-injective target: {title}")
        if tokens[-1] != tokenizer.eos_token_id or len(tokens) > CFG.decoder_cap:
            raise ValueError("Target EOS/length admission failed")
        sequences.append(tokens)
        title_by_tokens[tuple(tokens)] = title
    lengths = {}
    for fmt in ("D", "E", "F"):
        values = [len(tokenizer.encode(prompt_text(fmt, row["query"], hierarchy), truncation=False))
                  for split in memberships["profiles"]["preferred"].values() for row in split]
        if max(values) > CFG.encoder_cap:
            raise ValueError("Complete input exceeds cap; no truncation permitted")
        lengths[fmt] = {"min": min(values), "median": float(np.median(values)), "max": max(values)}
    return tokenizer, sequences, {"tokenizer_class": type(tokenizer).__name__, "original_tokenizer_limit": original_limit,
                                   "encoder_cap": CFG.encoder_cap, "input_lengths": lengths,
                                   "max_target_length": max(map(len, sequences)), "target_roundtrip": "passed",
                                   "target_injectivity": "passed"}


class TargetTrie:
    def __init__(self, sequences):
        self.root = {}
        self.eos_id = int(sequences[0][-1])
        for tokens in sequences:
            node = self.root
            for token in tokens:
                node = node.setdefault(int(token), {})

    def allowed(self, prefix):
        node = self.root
        for position, token in enumerate(prefix):
            if int(token) not in node:
                raise ValueError("Decoder prefix is outside the admitted target trie")
            node = node[int(token)]
            if int(token) == self.eos_id:
                # Transformers 4.57 retains finished beam slots while other
                # beams/queries continue. Only a verified complete title can
                # reach this no-op state; it cannot legalize an invalid title.
                if any(int(extra) not in (0, self.eos_id) for extra in prefix[position + 1:]):
                    raise ValueError("Unexpected tokens after a completed target")
                return [self.eos_id]
        return sorted(node)


class TrieMask(LogitsProcessor):
    def __init__(self, trie):
        self.trie = trie
        self.checked_normalization = False

    def __call__(self, input_ids, scores):
        if not self.checked_normalization:
            normalizers = torch.logsumexp(scores, dim=-1)
            if not torch.allclose(normalizers, torch.zeros_like(normalizers), atol=1e-4, rtol=1e-4):
                raise ValueError("Trie mask must receive full-vocabulary log probabilities")
            self.checked_normalization = True
        masked = torch.full_like(scores, -float("inf"))
        for i, sequence in enumerate(input_ids.tolist()):
            allowed = self.trie.allowed(sequence[1:])
            if not allowed:
                raise ValueError("Nonterminal hypothesis has no allowed continuation")
            masked[i, allowed] = scores[i, allowed]
        return masked


def decode_batch(model, tokenizer, trie, fmt, mode, queries, hierarchy, device, return_count=3):
    started = time.monotonic()
    prompts = [prompt_text(fmt, row["query"], hierarchy) for row in queries]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
    if encoded.input_ids.shape[1] > CFG.encoder_cap:
        raise ValueError("Complete batch exceeds encoder cap")
    lengths = encoded.attention_mask.sum(dim=1).tolist()
    encoded = encoded.to(device)
    config = GenerationConfig.from_model_config(model.config)
    config.update(num_beams=4, num_return_sequences=return_count, do_sample=False,
                  max_new_tokens=CFG.decoder_cap, length_penalty=1.0, early_stopping="never",
                  renormalize_logits=False, forced_eos_token_id=None, forced_bos_token_id=None,
                  repetition_penalty=1.0, no_repeat_ngram_size=0, remove_invalid_values=False,
                  return_dict_in_generate=True, output_scores=True, use_cache=True)
    processor = LogitsProcessorList([TrieMask(trie)]) if mode == "constrained" else LogitsProcessorList()
    model.eval()
    synchronize(device)
    model_started = time.monotonic()
    with torch.inference_mode():
        output = model.generate(**encoded, generation_config=config, logits_processor=processor)
        transitions = model.compute_transition_scores(output.sequences, output.scores,
                                                      output.beam_indices, normalize_logits=False)
    synchronize(device)
    model_seconds = time.monotonic() - model_started
    ids = output.sequences.cpu().tolist()
    transitions = transitions.cpu().tolist()
    scores = output.sequences_scores.cpu().tolist()
    mapping = {canonical(row["title"]): f"finance:{i:03d}" for i, row in enumerate(hierarchy["corpus"])}
    records = []
    for i, query in enumerate(queries):
        hypotheses = []
        for rank in range(return_count):
            index = i * return_count + rank
            generated = ids[index][1:]
            complete = tokenizer.eos_token_id in generated
            length = generated.index(tokenizer.eos_token_id) + 1 if complete else len(generated)
            generated = generated[:length]
            token_scores = transitions[index][:length]
            reconstructed = float(sum(token_scores) / max(length, 1))
            if not np.isfinite(reconstructed) or not np.isclose(reconstructed, scores[index], atol=1e-4, rtol=1e-4):
                raise ValueError(f"Beam score reconstruction failed: {reconstructed} vs {scores[index]}")
            text = tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            title = canonical(text)
            section_id = mapping.get(title) if complete else None
            reason = "valid" if section_id else "incomplete" if not complete else "empty" if not title else "nonexistent_title"
            if mode == "constrained" and reason != "valid":
                raise ValueError("Constrained decoder violated the admitted label invariant")
            hypotheses.append({"rank": rank + 1, "raw_text": text, "token_ids": generated,
                               "token_logprobs": token_scores, "score": reconstructed,
                               "validity": reason, "section_id": section_id,
                               "canonical_title": title, "complete_eos": complete})
        hypotheses.sort(key=lambda row: (-row["score"], (0, row["section_id"]) if row["section_id"] else (1, tuple(row["token_ids"]))))
        for rank, hypothesis in enumerate(hypotheses, 1):
            hypothesis["rank"] = rank
        records.append({"query_id": query["query_id"], "method": fmt, "mode": mode,
                        "prompt_sha256": digest_bytes(prompts[i].encode()), "input_tokens": lengths[i],
                        "batch_size": len(queries), "hypotheses": hypotheses})
    elapsed = time.monotonic() - started
    timing = {"batch_size": len(queries), "model_seconds": model_seconds, "end_to_end_seconds": elapsed,
              "amortized_query_seconds": elapsed / len(queries), "format": fmt, "mode": mode}
    for row in records:
        row["timing"] = timing
    del output, encoded
    return records, timing


def collate_training(tokenizer, fmt, rows, hierarchy, device):
    inputs = tokenizer([prompt_text(fmt, row["query"], hierarchy) for row in rows],
                       return_tensors="pt", padding=True, truncation=False).to(device)
    labels = tokenizer([row["gold_title"] for row in rows], return_tensors="pt", padding=True).input_ids
    labels[labels == tokenizer.pad_token_id] = -100
    return inputs, labels.to(device)


def per_example_loss(logits, labels):
    losses = torch_F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=-100, reduction="none")
    lengths = (labels != -100).sum(dim=1)
    return losses.sum(dim=1) / lengths


def train_update(model, tokenizer, optimizer, fmt, rows, hierarchy, device, microbatch):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    values = []
    for start in range(0, len(rows), microbatch):
        batch = rows[start:start + microbatch]
        inputs, labels = collate_training(tokenizer, fmt, batch, hierarchy, device)
        output = model(**inputs, labels=labels, use_cache=False)
        losses = per_example_loss(output.logits, labels)
        if not torch.isfinite(losses).all():
            raise ValueError("Nonfinite training loss")
        (losses.sum() / len(rows)).backward()
        values.extend(losses.detach().cpu().tolist())
        del output, inputs, labels
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or any(g.device.type != str(device) for g in gradients):
        raise ValueError("Gradient placement admission failed")
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True, foreach=False)
    if not torch.isfinite(norm) or float(norm) == 0:
        raise ValueError("Zero/nonfinite gradient norm")
    optimizer.step()
    return float(np.mean(values))


def teacher_loss(model, tokenizer, fmt, rows, hierarchy, device):
    model.eval()
    inputs, labels = collate_training(tokenizer, fmt, rows, hierarchy, device)
    with torch.inference_mode():
        logits = model(**inputs, labels=labels, use_cache=False).logits
        return float(per_example_loss(logits, labels).mean().cpu())
