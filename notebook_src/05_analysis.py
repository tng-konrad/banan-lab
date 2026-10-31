def labels_for(run):
    if is_mini_profile(run.get('configuration', {}).get('study_profile')) and run.get('kind') != 'demo':
        scope = study_scope(run)
        return scope['evaluation_labels'] if run.get('kind') == 'evaluation' else scope['design_labels']
    return EVALUATION_LABELS if run.get('kind') == 'evaluation' else DESIGN_LABELS

def annotate_rows(run):
    labels = labels_for(run)
    output = []
    for row in run.get('rows', []):
        label = labels[row['query_id']]
        gold = label['gold_passage_ids'][0]
        ranking = row['ranking'] if row['status'] == 'success' else []
        text = normalize_text(BY_ID[gold]['text'])
        gold_page = BY_ID[gold]['page_id']
        page_retained = gold_page in row['selected_pages']
        enriched = {**row, 'domain': label['domain'], 'gold_passage_id': gold,
                    'gold_page_id': gold_page, 'gold_page_retained': page_retained}
        for k in (1, 5, 10):
            enriched[f'r{k}'] = int(gold in ranking[:k])
            enriched[f'text_r{k}'] = int(any(normalize_text(BY_ID[x]['text']) == text for x in ranking[:k]))
        for stage in ('page', 'passage'):
            event = next((x for x in row['events'] if x.get('stage') == stage and 'coarse_ranking' in x), None)
            target = gold_page if stage == 'page' else gold
            enriched[stage + '_prefix_contains_gold'] = bool(event and target in event['rescore_prefix'])
            enriched[stage + '_coarse_top10_gold'] = bool(event and target in event['coarse_ranking'][:10])
            enriched[stage + '_final_top10_gold'] = bool(event and target in event['ranking'][:10])
        enriched['failure_class'] = ('execution failure' if row['status'] != 'success' else
            'correct rank one' if enriched['r1'] else 'gold page pruned' if
            row['option'] in ('C', 'D') and not page_retained else 'retained page, wrong rank one')
        output.append(enriched)
    return output

def usage_summary(run, option):
    records = [r for r in run.get('requests', []) if r['option'] == option]
    attempts = [a for r in records for a in r['attempts']]
    inputs, outputs, cached = [], [], []
    for attempt in attempts:
        r = attempt.get('response', {})
        if 'usage' in r:
            inputs.append(r['usage'].get('input_tokens'))
            outputs.append(r['usage'].get('output_tokens'))
        else:
            inputs.append(r.get('prompt_eval_count'))
            outputs.append(r.get('eval_count'))
        cached.append(r.get('prompt_eval_cached_count'))
    return {'requests': len(records), 'attempts': len(attempts),
        'input_tokens_reported': sum(x for x in inputs if x is not None),
        'output_tokens_reported': sum(x for x in outputs if x is not None),
        'cached_tokens_reported': sum(x for x in cached if x is not None),
        'cache_counter_missing_attempts': sum(x is None for x in cached),
        'score_batches': sum('scores' in r for r in records),
        'all_equal_batches': sum(r.get('all_equal') is True for r in records),
        'tied_batches': sum('scores' in r and len(set(r['scores'].values())) < len(r['scores']) for r in records)}

# %%
def summarize_run(run):
    rows = annotate_rows(run)
    n = run['scheduled_queries']
    summary = []
    for option in CFG.methods:
        subset = [r for r in rows if r['option'] == option]
        attempted = len(subset)
        times = [r['seconds'] for r in subset]
        data = {'option': option, 'method': CFG.methods[option], 'scheduled': n, 'attempted': attempted,
            'valid': sum(r['status'] == 'success' for r in subset),
            'failed': sum(r['status'] != 'success' for r in subset), 'unattempted': n - attempted,
            'quality_denominator': attempted, 'latency_denominator': attempted,
            'p50_seconds': float(np.median(times)) if times else None,
            'p95_seconds': float(np.percentile(times, 95)) if times else None,
            **usage_summary(run, option)}
        for k in (1, 5, 10):
            data[f'r{k}'] = sum(r[f'r{k}'] for r in subset) / attempted if attempted else None
            data[f'text_r{k}'] = sum(r[f'text_r{k}'] for r in subset) / attempted if attempted else None
        successful = [r for r in subset if r['status'] == 'success']
        retained = [r for r in successful if r['gold_page_retained']]
        data['gold_page_retention'] = len(retained) / attempted if attempted and option in ('C', 'D') else None
        data['r1_given_page_survival'] = sum(r['r1'] for r in retained) / len(retained) if retained else None
        for stage in ('page', 'passage'):
            for metric in ('prefix_contains_gold', 'coarse_top10_gold', 'final_top10_gold'):
                data[stage + '_' + metric] = (sum(r[stage + '_' + metric] for r in subset) / attempted
                    if attempted else None)
        summary.append(data)
    domains = []
    for domain in CFG.domains:
        for option in CFG.methods:
            subset = [r for r in rows if r['domain'] == domain and r['option'] == option]
            if subset:
                domains.append({'domain': domain, 'option': option, 'attempted': len(subset),
                    'r1': sum(r['r1'] for r in subset) / len(subset),
                    'r10': sum(r['r10'] for r in subset) / len(subset),
                    'failed': sum(r['status'] != 'success' for r in subset)})
    return {'rows': rows, 'methods': summary, 'domains': domains,
            'run_id': run['run_id'], 'kind': run['kind'], 'cohort': run['cohort'],
            'status': run['status'], 'scheduled_queries': n, 'study_profile': study_scope(run)['profile']}

# %%
def dialogue_bootstrap(run, analysis):
    # Complete all-four observations required for a common full-query matrix.
    qids = run['query_ids']
    rows = {(r['query_id'], r['option']): r for r in analysis['rows']}
    if any((qid, m) not in rows for qid in qids for m in CFG.methods):
        return {'status': 'unavailable', 'reason': 'Incomplete paired schedule; no invented outcomes for unattempted queries'}
    labels = labels_for(run)
    metric = np.array([[[rows[qid, m][f'r{k}'] for k in (1, 5, 10)] for m in CFG.methods] for qid in qids], dtype=float)
    seconds = np.array([[rows[qid, m]['seconds'] for m in CFG.methods] for qid in qids])
    by_domain = {}
    for domain in CFG.domains:
        dialogues = sorted({rows[qid, 'A']['dialogue_id'] for qid in qids if labels[qid]['domain'] == domain})
        if dialogues:
            by_domain[domain] = [np.array([i for i, qid in enumerate(qids)
                if rows[qid, 'A']['dialogue_id'] == dialogue], dtype=int) for dialogue in dialogues]
    rng = np.random.default_rng(CFG.experiment['bootstrap_seed'])
    quality = np.empty((CFG.experiment['bootstrap_samples'], 4, 3))
    deltas, ratios = [], []
    for b in range(CFG.experiment['bootstrap_samples']):
        indices = np.concatenate([np.concatenate([groups[i] for i in rng.integers(0, len(groups), size=len(groups))])
                                  for groups in by_domain.values()])
        means = metric[indices].mean(axis=0)
        quality[b] = means
        deltas.append(float(means[2, 0] - means[3, 0]))
        ratios.append(float(np.median(seconds[indices, 2]) / np.median(seconds[indices, 3])))
    intervals = {m: {f'r{k}': np.percentile(quality[:, i, j], [2.5, 97.5]).tolist()
        for j, k in enumerate((1, 5, 10))} for i, m in enumerate(CFG.methods)}
    paired = {'quality_difference': float(metric[:, 2, 0].mean() - metric[:, 3, 0].mean()),
        'quality_difference_ci': np.percentile(deltas, [2.5, 97.5]).tolist(),
        'median_time_ratio': float(np.median(seconds[:, 2]) / np.median(seconds[:, 3])),
        'median_time_ratio_ci': np.percentile(ratios, [2.5, 97.5]).tolist()}
    return {'status': 'available', 'method': 'paired percentile whole-dialogue bootstrap stratified by domain',
        'resamples': CFG.experiment['bootstrap_samples'], 'seed': CFG.experiment['bootstrap_seed'],
        'queries': len(qids), 'dialogues': sum(map(len, by_domain.values())),
        'quality_intervals': intervals, 'paired': paired}

def primary_verdict(run, bootstrap):
    gates = validity_gates(run)
    if bootstrap.get('status') != 'available':
        gates.update({'quality_noninferiority': None, 'median_time_superiority': None})
        return {'verdict': 'inconclusive', 'gates': gates, 'reason': bootstrap.get('reason')}
    paired = bootstrap['paired']
    quality_pass = paired['quality_difference_ci'][0] > -CFG.experiment['quality_margin']
    latency_pass = paired['median_time_ratio_ci'][1] < 1
    gates.update({'quality_noninferiority': quality_pass, 'median_time_superiority': latency_pass})
    verdict = 'local efficiency win for C' if all(gates.values()) else 'inconclusive'
    if is_mini_profile(study_scope(run)['profile']):
        return {'verdict': 'exploratory mini comparison', 'gates': gates,
            'quality_margin': CFG.experiment['quality_margin'], **paired,
            'reason': f"Restricted 32-page corpus and {study_scope(run)['evaluation_queries']} queries; descriptive evidence only. No original full-study noninferiority claim.",
            'scope': MINI['claim_scope']}
    return {'verdict': verdict, 'gates': gates, 'quality_margin': CFG.experiment['quality_margin'],
        **paired, 'reason': 'All predeclared conditions pass.' if all(gates.values()) else
        'No confirmatory efficiency claim: ' + ', '.join(k for k, v in gates.items() if not v),
        'scope': 'These two trained models, adapters, and local serving paths only.'}

# %%
def common_success_sensitivities(run, analysis):
    rows = {(r['query_id'], r['option']): r for r in analysis['rows']}
    output = []
    for methods in (('A', 'B'), ('B', 'C'), ('C', 'D'), tuple(CFG.methods)):
        qids = [qid for qid in run['query_ids'] if all((qid, m) in rows and
            rows[qid, m]['status'] == 'success' for m in methods)]
        output.append({'options': list(methods), 'queries': len(qids),
            'results': {m: {'r1': float(np.mean([rows[qid, m]['r1'] for qid in qids])) if qids else None,
                'p50_seconds': float(np.median([rows[qid, m]['seconds'] for qid in qids])) if qids else None}
                for m in methods}, 'status': 'descriptive sensitivity; never repairs primary validity gates'})
    return output

def analyze(run):
    with recorded_configuration(run):
        return analyze_under_config(run)

def analyze_under_config(run):
    data = summarize_run(run)
    data['timing_audit'] = run_clock_audit(run)
    data['bootstrap'] = dialogue_bootstrap(run, data)
    data['primary'] = primary_verdict(run, data['bootstrap'])
    data['common_success'] = common_success_sensitivities(run, data)
    # Small reproducibility fingerprint, recomputed from raw rows (not copied metrics).
    data['numerical_receipt'] = sha256({k: data[k] for k in ('methods', 'domains', 'bootstrap', 'primary', 'common_success')})
    return data

# %%
# ANALYSIS_RUN: all reported findings come from these freshly computed objects.
EVIDENCE_CHECKS = {name: verify_request_evidence(run) for name, run in RUNS.items()
    if name in ('demo', 'pilot', 'evaluation') and run.get('rows')}
ACTIVE_RUN = RUNS.get('evaluation') or RUNS.get('pilot') or RUNS.get('demo')
ANALYSIS = analyze(ACTIVE_RUN) if ACTIVE_RUN and ACTIVE_RUN.get('rows') else None
if ANALYSIS:
    display(Markdown(f"**Analysis cohort:** {ANALYSIS['cohort']}. "
        f"Run `{ANALYSIS['run_id']}`; status **{ANALYSIS['status']}**. "
        f"Primary verdict: **{ANALYSIS['primary']['verdict']}**. "
        + ANALYSIS['primary']['reason']))
    results_display = pd.DataFrame(ANALYSIS['methods'])[[
        'option', 'method', 'scheduled', 'attempted', 'valid', 'failed', 'r1', 'r5', 'r10', 'p50_seconds', 'p95_seconds']]
    # The same values appear in the numbered, exported results table below.
    if ACTIVE_RUN.get('failure'):
        display(Markdown('**Why the run stopped:** `' + ACTIVE_RUN['failure']['category'] +
            '` — ' + ACTIVE_RUN['failure']['message']))
    invalid_rows = [r for r in ANALYSIS['rows'] if r['status'] != 'success']
    if invalid_rows:
        display(Markdown('\n'.join(f"- {r['option']} / `{r['query_id']}`: "
            f"{r['failure']['category']} — {r['failure']['message']}" for r in invalid_rows)))
    display(Markdown('R@k is strict structural passage-ID recall. Failure = no hit; '
        'unattempted queries are displayed separately and have no invented latency. '
        'Timing includes admission, retries, and any loading inside the measured query. '
        'The exact-text sensitivity, page-retention diagnostics, and common-success cohorts are in the JSON sidecars.'))
else:
    display(Markdown('No measured retrieval results are available. The corpus and implementation can still be inspected.'))
