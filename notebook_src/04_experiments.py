def offline_checks():
    checks = []
    def check(name, fn):
        assert fn(), name
        checks.append(name)
    check('strict structural reference mapping', lambda: validate_dataset()['passages'] == 4110)
    probe = copy.deepcopy(DEMO_INPUT)
    probe.update({'gold_passage_ids': ['LABEL_SENTINEL_X7'], 'reference_answer': 'ANSWER_SENTINEL_Q9',
                  'domain': 'DOMAIN_SENTINEL_R3'})
    batch = shared_batch(probe, DEMO_RETRIEVER.passages[:2], 'passage')
    rendered = stable_json([payload_for(a, batch) for a in ('nimble', 'qwen')])
    check('label and target isolation', lambda: not any(x in rendered for x in (
        'LABEL_SENTINEL_X7', 'ANSWER_SENTINEL_Q9', 'DOMAIN_SENTINEL_R3')))
    check('stable score ties', lambda: sorted_scores({'b': .5, 'a': .5, 'c': .1}) == ['a', 'b', 'c'])
    check('prefix replacement preserves pool', lambda: replace_prefix(['a', 'b', 'c'],
        ['a', 'b'], {'a': .1, 'b': .9}) == ['b', 'a', 'c'])
    aliases = list(batch['aliases'])
    good_n = {'answers': {a: {'type': 'noul', 'noul': .2} for a in aliases}}
    good_q = {'done': True, 'done_reason': 'stop', 'response': stable_json({
        'scores': {a: .2 for a in aliases}})}
    check('both valid response parsers', lambda: parse_scores('nimble', good_n, batch) ==
          parse_scores('qwen', good_q, batch))
    bad_cases = [('nimble', {'answers': {}}),
        ('nimble', {'answers': {a: {'type': 'noul', 'noul': float('nan')} for a in aliases}}),
        ('qwen', {**good_q, 'thinking': 'hidden explanation'}),
        ('qwen', {**good_q, 'done_reason': 'length'}),
        ('qwen', {**good_q, 'response': '{"scores":{"c001":0.2,"c001":0.3,"c002":0.2}}'}),
        ('qwen', {**good_q, 'response': '{"scores":{"extra":0}}'}),
        ('qwen', {**good_q, 'response': stable_json({'scores': {a: 1.2 for a in aliases}})})]
    for adapter, value in bad_cases:
        try:
            parse_scores(adapter, value, batch)
            raise AssertionError('Invalid response accepted')
        except StudyFailure:
            pass
    checks.append('malformed, truncated, extra, nonfinite, and out-of-range responses rejected')
    events = []
    candidates = FULL_RETRIEVER.passages[:40]
    batches = deterministic_batches(DEMO_INPUT, candidates, 'passage', events)
    flattened = [cid for b in batches for cid in b['candidate_ids']]
    check('deterministic splitting loses no candidates', lambda:
          flattened == [c['id'] for c in sorted(candidates, key=lambda c: SOURCE_ORDER[c['id']])])
    check('same input partitions across adapters', lambda: batches ==
          deterministic_batches(DEMO_INPUT, candidates, 'passage', []))
    whole = shared_batch(DEMO_INPUT, FULL_RETRIEVER.passages[:16], 'passage')
    class CapturingBackend:
        def __init__(self):
            self.calls = []
        def call(self, adapter, part):
            self.calls.append((adapter, part))
            return {cid: .25 for cid in part['aliases'].values()}
    for adapter in ('nimble', 'qwen'):
        capture = CapturingBackend()
        scores = score_batch(capture, adapter, whole)
        assert len(capture.calls) == math.ceil(16 / CFG.experiment['questions_per_request'])
        assert len(scores) == 16
        assert all(part['state'] == whole['state'] and part['candidate_ids'] == whole['candidate_ids']
                   for _, part in capture.calls)
        assert [alias for _, part in capture.calls for alias in part['aliases']] == list(whole['aliases'])
        assert all(len(part['questions']) <= CFG.experiment['questions_per_request'] for _, part in capture.calls)
    checks.append('four-question requests preserve full candidate context, stable aliases and all scores in both adapters')
    huge = copy.deepcopy(DEMO_RETRIEVER.passages[0]); huge['text'] = ' z' * 20000
    for adapter in ('nimble', 'qwen'):
        try:
            common_admission(shared_batch(DEMO_INPUT, [huge], 'passage'))
            raise AssertionError('Over-limit input accepted by ' + adapter)
        except LocalSizeRejection:
            pass
    checks.append('common over-limit rejection before either backend')
    class RejectingBackend:
        def __init__(self):
            self.records = []
        def call(self, *args):
            raise StudyFailure('server_input_limit', 'Deliberate fixture rejection')
    for option in ('B', 'C', 'D'):
        row = DEMO_RETRIEVER.run(option, DEMO_INPUT, RejectingBackend())
        assert row['status'] == 'failed' and row['ranking'] == []
        assert row['failure']['category'] == 'server_input_limit'
    checks.append('B/C/D share server-limit failures without fallback')
    for adapter in ('nimble', 'qwen'):
        calls = []
        def reject_request(request):
            calls.append(request)
            return httpx.Response(500, json={'error': 'Context size has been exceeded.'})
        fixture = OllamaBackend()
        fixture.client.close()
        fixture.client = httpx.Client(base_url='http://offline-fixture', transport=httpx.MockTransport(reject_request))
        try:
            fixture.call(adapter, batch)
            raise AssertionError('Context rejection accepted')
        except StudyFailure as exc:
            assert exc.category == 'server_input_limit'
        finally:
            fixture.close()
        assert len(calls) == len(fixture.records[0]['attempts']) == 1
    checks.append('HTTP 500 context-limit errors never receive the transport retry')
    class ConstantBackend:
        def __init__(self):
            self.records = []
        def call(self, adapter, batch):
            return {cid: .1 for cid in batch['aliases'].values()}
    for option in ('C', 'D'):
        row = DEMO_RETRIEVER.run(option, DEMO_INPUT, ConstantBackend())
        assert row['status'] == 'success'
        assert all(BY_ID[x]['page_id'] in row['selected_pages'] for x in row['ranking'])
    checks.append('no passage evidence outside selected branches')
    assert not validity_gates({'kind': 'evaluation', 'status': 'complete',
        'scheduled_queries': 259, 'rows': [{'option': 'C', 'status': 'failed'}],
        'input_fit': True, 'warm_residency': True})['all_C_D_valid']
    checks.append('one failure prevents success-selected efficiency verdict')
    bound = pilot_runtime_lower_bound([{'query_id': FULL_PILOT_INPUTS[0]['query_id'], 'option': 'C', 'seconds': 325}],
        {'configuration': {'study_profile': 'full'}})
    assert math.isclose(bound['seconds'], 66 * 325 / 3 * 1.25)
    checks.append('early budget rejection uses domain weights and zero unobserved timings')
    assert len(MINI_RETRIEVER.pages) == 32 and len(MINI_RETRIEVER.passages) == 329
    for labels, quota in ((MINI_DESIGN_LABELS, 2), (MINI_EVALUATION_LABELS, len(MINI_EVALUATION_INPUTS) // 4)):
        assert Counter(l['domain'] for l in labels.values()) == dict.fromkeys(CFG.domains, quota)
        assert all(pid in MINI_RETRIEVER.passage_by_id for l in labels.values() for pid in l['gold_passage_ids'])
    assert not (set(EMBEDDED_DATA['explored_validation_dialogues']) &
        {q['dialogue_id'] for q in MINI_EVALUATION_INPUTS})
    checks.append('mini corpus has full gold evidence, balanced queries and independent evaluation dialogues')
    return {'status': 'passed', 'checks': checks}

def run_clock_audit(run):
    if not run.get('finished_at') or run.get('elapsed_seconds') is None:
        return {'status': 'unavailable', 'reason': 'No complete timestamp/timer pair'}
    span = (datetime.fromisoformat(run['finished_at']) - datetime.fromisoformat(run['started_at'])).total_seconds()
    active = run['elapsed_seconds']
    discrepancy = abs(span - active)
    return {'status': 'passed' if discrepancy <= max(5, .01 * active) else 'failed',
        'timestamp_span_seconds': span, 'recorded_timer_seconds': active,
        'discrepancy_seconds': discrepancy,
        'interpretation': 'A large discrepancy can reflect suspension or clock changes. Active-timer measurements cannot establish wall-clock speed in that run.'}

def validity_gates(run):
    rows = run.get('rows', [])
    n = run.get('scheduled_queries', 259)
    return {'full_four_option_run': run.get('kind') == 'evaluation' and
            run.get('status') == 'complete' and len(rows) == n * 4 and n == 259,
        'input_fit': run.get('input_fit') is True and not any(
            (r.get('failure') or {}).get('category') in ('local_input_limit', 'server_input_limit', 'token_accounting')
            for r in rows),
        'warm_residency': run.get('warm_residency') is True,
        'timing_integrity': run_clock_audit(run)['status'] == 'passed',
        'all_C_D_valid': all(sum(r['option'] == m and r['status'] == 'success'
            for r in rows) == n for m in ('C', 'D'))}

# %%
def both_resident(snapshot):
    names = {m['name']: m for m in snapshot.get('models', [])}
    return all(name in names and names[name].get('context_length', 0) >= CFG.experiment['context_tokens']
               for name in CFG.experiment['models'].values())

def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))
    temp.replace(path)

def export_frozen_inputs(folder=None):
    folder = CFG.output_folder / 'frozen_inputs' if folder is None else Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, inputs, labels in (('design', DESIGN_INPUTS, DESIGN_LABELS),
                                 ('evaluation', EVALUATION_INPUTS, EVALUATION_LABELS),
                                 ('mini_design', MINI_DESIGN_INPUTS, MINI_DESIGN_LABELS),
                                 ('mini_evaluation', MINI_EVALUATION_INPUTS, MINI_EVALUATION_LABELS),
                                 ('mini_2h_evaluation', DATASET['mini_profile']['evaluation_inputs'], DATASET['mini_profile']['evaluation_labels']),
                                 ('mini_3h_evaluation', DATASET['mini_3h_profile']['evaluation_inputs'], DATASET['mini_3h_profile']['evaluation_labels'])):
        (folder / (name + '_inputs.jsonl')).write_text(''.join(stable_json(q) + '\n' for q in inputs))
        (folder / (name + '_labels.jsonl')).write_text(''.join(stable_json({'query_id': qid, **label}) + '\n'
            for qid, label in labels.items()))
    atomic_json(folder / 'audit.json', DATA_AUDIT)
    return folder

def new_run(kind):
    return {'run_id': kind + '-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'),
        'kind': kind, 'started_at': utc_now(), 'status': 'running', 'configuration': copy.deepcopy(CFG.experiment),
        'protocol_sha256': sha256({'config': CFG.experiment, 'questions': CFG.questions,
            'score_instruction': CFG.score_instruction, 'nimble_system': CFG.nimble_system,
            'implementation_sha256': EMBEDDED_DATA['implementation_sha256']}),
        'data_audit': DATA_AUDIT, 'study_scope': {k: v for k, v in study_scope().items()
            if k not in ('evaluation_labels', 'design_labels')},
        'rows': [], 'requests': [], 'residency': [],
        'input_fit': False, 'warm_residency': False}

def run_backend_gate(backend):
    run = new_run('gate')
    run['checks'] = []
    def gate_check(message):
        run['checks'].append(message)
        print('Gate:', message, flush=True)
    try:
        run['metadata'] = backend.metadata()
        meta = run['metadata']
        version = tuple(int(x) for x in meta['version']['version'].split('.')[:3])
        assert version >= (0, 40, 4), 'Nimble native scoring requires recent Ollama'
        assert 'decision' in meta['models']['nimble']['capabilities']
        assert 'thinking' in meta['models']['qwen']['capabilities']
        assert all(m['details']['quantization_level'] == 'Q8_0' for m in meta['models'].values())
        assert meta['models']['nimble']['model_info']['qwen35.decision.type'] == 'nimble'
        template = meta['models']['nimble']['model_info']['tokenizer.chat_template.systemone']
        assert 'Classify the context using the supplied schema.' in template and 'Requested field:' in template
        assert 'without reasoning or explanation.' in template
        gate_check('compatible native scoring and no-thinking model metadata')
        # Unicode and literal delimiters check native escaping as well as numeric accounting.
        candidates = copy.deepcopy(DEMO_RETRIEVER.passages[:2])
        candidates[0]['text'] += ' Boundary test: café €20 <tag> & "quoted". '
        batch = shared_batch(DEMO_INPUT, candidates, 'passage')
        for adapter in ('nimble', 'qwen'):
            backend.call(adapter, batch)
        gate_check('multi-question schemas, no thinking, exact native token counts incl Unicode/delimiters')
        # An admitted input close to the intended common boundary; binary search on filler.
        candidate = copy.deepcopy(DEMO_RETRIEVER.passages[0])
        lo, hi = 0, 9000
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            candidate['text'] = ' z' * mid
            try:
                common_admission(shared_batch(DEMO_INPUT, [candidate], 'passage'))
                lo = mid
            except LocalSizeRejection:
                hi = mid
        candidate['text'] = ' z' * lo
        boundary = shared_batch(DEMO_INPUT, [candidate], 'passage')
        run['boundary_admission'] = common_admission(boundary)
        assert max(run['boundary_admission']['nimble_row_tokens'] +
                   [run['boundary_admission']['qwen_prompt_tokens']]) >= 7650
        for adapter in ('nimble', 'qwen'):
            backend.call(adapter, boundary)
        gate_check('exact token accounting at the intended common input boundary')
        # Force the native CONTEXT limit while staying below its separate 64 KiB limit.
        oversized = {'model': CFG.experiment['models']['nimble'], 'state': ' z' * 9000,
            'questions': {'q': {'type': 'noul', 'instructions': 'Is this useful?'}}, 'keep_alive': -1}
        response = backend.client.post('/v1/systemone', json=oversized, timeout=backend.timeout())
        run['native_overlimit_response'] = {'status': response.status_code, 'body': response.json()}
        assert response.status_code in (400, 413) and re.search(
            r'context|too long|batch', response.text, re.I), 'Native overflow did not reject safely'
        gate_check('native decision context overflow safely rejected without truncation')
        # Every complete singleton across full corpus, using the largest design history.
        longest_history = max(DESIGN_INPUTS, key=lambda q: token_count(stable_json(model_history(q))))
        max_tokens, failures = 0, []
        for stage, pool in (('page', FULL_RETRIEVER.pages), ('passage', FULL_RETRIEVER.passages)):
            for candidate in pool:
                try:
                    admitted = common_admission(shared_batch(longest_history, [candidate], stage))
                    max_tokens = max(max_tokens, *admitted['nimble_row_tokens'], admitted['qwen_prompt_tokens'])
                except LocalSizeRejection as exc:
                    failures.append({'stage': stage, 'id': candidate['id'], 'reason': str(exc)})
        run['singleton_stress'] = {'failures': failures, 'maximum_tokens': max_tokens,
            'history_query_id': longest_history['query_id'], 'checked_candidates': 4523}
        assert not failures, 'Complete singleton cannot fit the common input contract'
        gate_check('all complete page/passages fit, using longest design history')
        # Pack the largest page and all metadata, proving no lost candidates on splitting.
        largest_page = max(FULL_RETRIEVER.pages, key=lambda p:
            sum(c['page_id'] == p['id'] for c in CORPUS))
        stress_passages = [p for p in FULL_RETRIEVER.passages if BY_ID[p['id']]['page_id'] == largest_page['id']]
        run['stress_batches'] = {}
        for stage, pool in (('page', FULL_RETRIEVER.pages), ('passage', stress_passages)):
            events = []
            batches = deterministic_batches(longest_history, pool, stage, events)
            assert sum(len(b['aliases']) for b in batches) == len(pool)
            run['stress_batches'][stage] = {'batches': len(batches), 'candidates': len(pool),
                'splits': len(events), 'max_tokens': max(max(common_admission(b)['nimble_row_tokens'] +
                    [common_admission(b)['qwen_prompt_tokens']]) for b in batches)}
        # The one-prompt boundary test does not exercise cache reuse across states.
        page_batches = deterministic_batches(DEMO_INPUT, FULL_RETRIEVER.pages, 'page', [])
        sequence = [page_batches[i] for i in (0, 1, 2, 0)]
        run['question_group_stress'] = []
        for sequence_index, batch in enumerate(sequence):
            for adapter in ('nimble', 'qwen'):
                scores = score_batch(backend, adapter, batch)
                assert len(scores) == len(batch['aliases'])
                run['question_group_stress'].append({'sequence_index': sequence_index,
                    'adapter': adapter, 'candidate_ids': batch['candidate_ids'],
                    'questions_per_request': CFG.experiment['questions_per_request'], 'scored': len(scores)})
        gate_check('both adapters score every candidate through four-question groups across alternating full candidate states')
        snapshot = backend.residency(); run['residency'].append(snapshot)
        assert both_resident(snapshot), 'Both models must remain loaded'
        run['input_fit'] = run['warm_residency'] = True
        gate_check('both Q8 models resident simultaneously')
        run['status'] = 'passed'
    except (AssertionError, StudyFailure, httpx.HTTPError, KeyError) as exc:
        run['status'] = 'failed'
        run['failure'] = {'category': getattr(exc, 'category', type(exc).__name__), 'message': str(exc)}
    run['requests'] = copy.deepcopy(backend.records)
    run['finished_at'] = utc_now()
    atomic_json((CFG.result_folder / 'gate.json'), run)
    atomic_json((CFG.result_folder / 'gates') / (run['run_id'] + '.json'), run)
    return run

# %%
def freeze_schedule(inputs):
    schedule = []
    for pair in (('C', 'D'), ('A', 'B')):
        for query in inputs:
            order = list(pair)
            rng = random.Random('42:' + ''.join(pair) + ':' + query['query_id'])
            rng.shuffle(order)
            schedule.extend({'query_id': query['query_id'], 'option': option} for option in order)
    return schedule

def execute_queries(kind, inputs, retriever, backend, gate):
    run = new_run(kind)
    run['metadata'] = gate.get('metadata', {})
    run['scheduled_queries'] = len(inputs)
    scope = study_scope()
    run['cohort'] = (f"{scope['profile']}: {scope['evaluation_queries']} validation queries / "
        f"{scope['evaluation_dialogues']} dialogues / {scope['pages']} pages" if kind == 'evaluation'
        else f"training-only {scope['profile']} pilot / {scope['pages']} pages" if kind == 'pilot'
        else 'training-only ten-page example')
    run['input_fit'] = gate['status'] == 'passed' and gate['input_fit']
    run['warm_residency'] = gate.get('warm_residency', False)
    run['schedule'] = freeze_schedule(inputs)
    run['schedule_sha256'] = sha256(run['schedule'])
    run['query_ids'] = [q['query_id'] for q in inputs]
    if kind == 'demo':
        run['demo_page_ids'] = DEMO_PAGE_IDS
    by_q = {q['query_id']: q for q in inputs}
    path = CFG.result_folder / ('evaluation.json' if kind == 'evaluation' else kind + '.json')
    request_start = len(backend.records)
    run['request_index_offset'] = request_start
    consecutive = Counter()
    started = time.monotonic()
    previous_deadline = backend.deadline
    previous_wall_deadline = backend.wall_deadline
    if kind == 'evaluation':
        backend.deadline = min(previous_deadline or float('inf'), started + CFG.experiment['evaluation_seconds'])
        backend.wall_deadline = min(previous_wall_deadline or float('inf'), time.time() + CFG.experiment['evaluation_seconds'])
    tracemalloc.start()
    try:
        # Warm-up is excluded from per-query timings, included in workflow cost.
        warm_start = len(backend.records)
        warm_batch = shared_batch(DEMO_INPUT, DEMO_RETRIEVER.passages[:2], 'passage')
        try:
            for adapter in ('nimble', 'qwen'):
                backend.call(adapter, warm_batch)
        finally:
            run['warmup_requests'] = copy.deepcopy(backend.records[warm_start:])
            request_start = len(backend.records)
            run['request_index_offset'] = request_start
        snapshot = backend.residency(); run['residency'].append(snapshot)
        run['warm_residency'] &= both_resident(snapshot)
        for i, item in enumerate(run['schedule']):
            if backend.remaining() <= 0:
                raise StudyFailure('deadline', 'Hard evaluation wall-time budget reached')
            row = retriever.run(item['option'], by_q[item['query_id']], backend)
            # Indices in exported runs are local and reconstruct the exact request sequence.
            row['request_indices'] = [j - request_start for j in row['request_indices']]
            run['rows'].append(row)
            run['requests'] = backend.records[request_start:]
            snapshot = backend.residency(); run['residency'].append(snapshot)
            run['warm_residency'] &= both_resident(snapshot)
            # Reload time is exposed on generation, absent on the native decision response.
            for record in run['requests']:
                for attempt in record['attempts']:
                    response = attempt.get('response', {})
                    if response.get('load_duration', 0) > 500_000_000:
                        run['warm_residency'] = False
            category = (row.get('failure') or {}).get('category')
            consecutive[item['option']] = consecutive[item['option']] + 1 if category == 'infrastructure' else 0
            run['elapsed_seconds'] = time.monotonic() - started
            run['python_peak_bytes'] = tracemalloc.get_traced_memory()[1]
            atomic_json(path, run)
            print(f"{kind}: {i + 1}/{len(run['schedule'])} method-queries; "
                  f"{item['option']} {row['status']}; elapsed {run['elapsed_seconds']:.1f}s", flush=True)
            if category in ('unsupported_backend', 'token_accounting', 'deadline') or consecutive[item['option']] >= 3:
                raise StudyFailure(category or 'infrastructure', 'Run stopped by frozen abort rule')
            if kind == 'pilot':
                bound = pilot_runtime_lower_bound(run['rows'], run)
                run['runtime_lower_bound'] = bound
                # A mathematical impossibility check, independent of quality outcomes.
                available = max(0, backend.remaining())
                if bound['seconds'] > available:
                    raise StudyFailure('resource_gate', 'Even zero time on every remaining pilot observation cannot meet the evaluation budget')
                if (i + 1) % 2 == 0 and any(r['status'] != 'success' for r in run['rows']):
                    raise StudyFailure('pilot_gate', 'The first completed C/D pair has an invalid query; the all-valid pilot requirement cannot pass')
        run['status'] = 'complete'
    except (StudyFailure, httpx.HTTPError, KeyboardInterrupt) as exc:
        run['status'] = 'incomplete'
        run['failure'] = {'category': getattr(exc, 'category', type(exc).__name__), 'message': str(exc)}
    finally:
        run['python_peak_bytes'] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        run['elapsed_seconds'] = time.monotonic() - started
        run['finished_at'] = utc_now()
        run['requests'] = copy.deepcopy(backend.records[request_start:])
        atomic_json(path, run)
        backend.deadline = previous_deadline
        backend.wall_deadline = previous_wall_deadline
    return run

# %%
def pilot_runtime_lower_bound(rows, run=None):
    scope = study_scope(run)
    counts = Counter(label['domain'] for label in scope['evaluation_labels'].values())
    quotas = scope['pilot_quotas']
    known = defaultdict(float)
    for row in rows:
        domain = scope['design_labels'][row['query_id']]['domain']
        known[row['option']] += counts[domain] * row['seconds'] / quotas[domain]
    return {'seconds': 1.25 * sum(known.values()), 'method_seconds_without_reserve': dict(known),
        'observed_method_queries': len(rows), 'scheduled_method_queries': scope['pilot_queries'] * 4,
        'interpretation': 'Lower bound under the frozen projection formula, assigning zero seconds to every unobserved pilot timing and zero scheduling overhead. Not a point forecast.'}

def pilot_projection(run, available_seconds=None):
    scope = study_scope(run)
    counts = Counter(label['domain'] for label in scope['evaluation_labels'].values())
    budget = CFG.experiment['evaluation_seconds'] if available_seconds is None else max(0, available_seconds)
    by_group = defaultdict(list)
    for row in run['rows']:
        by_group[row['option'], scope['design_labels'][row['query_id']]['domain']].append(row['seconds'])
    projections = {}
    for option in CFG.methods:
        if not all(by_group[option, domain] for domain in CFG.domains):
            return {'passed': False, 'reason': 'Pilot incomplete; the pass requirements are not met',
                'seconds': None, 'lower_bound': pilot_runtime_lower_bound(run['rows'], run),
                'budget_seconds': budget}
        projections[option] = sum(counts[d] * statistics.mean(by_group[option, d]) for d in CFG.domains)
    scheduling = max(0, run['elapsed_seconds'] - sum(r['seconds'] for r in run['rows']))
    scheduling_scaled = scheduling * scope['evaluation_queries'] / scope['pilot_queries']
    seconds = 1.25 * sum(projections.values()) + scheduling_scaled
    all_success = len(run['rows']) == scope['pilot_queries'] * 4 and all(r['status'] == 'success' for r in run['rows'])
    clock_audit = run_clock_audit(run)
    return {'passed': all_success and run['input_fit'] and run['warm_residency'] and
            clock_audit['status'] == 'passed' and
            seconds <= budget, 'seconds': seconds,
        'budget_seconds': budget, 'method_seconds_without_reserve': projections,
        'scheduling_seconds_scaled': scheduling_scaled, 'all_pilot_queries_valid': all_success,
        'timing_audit': clock_audit,
        'warning': 'Only two or three design queries/domain; the forecast is uncertain.'}

def adapter_check(backend):
    records_start = len(backend.records)
    output = []
    for query in PILOT_INPUTS:
        pool = [ACTIVE_RETRIEVER.passage_by_id[x] for x in ACTIVE_RETRIEVER.lexical(query, 10)]
        batches = deterministic_batches(query, pool, 'passage', [])
        row = {'query_id': query['query_id'], 'candidate_ids': [c['id'] for c in pool], 'adapters': {}}
        for adapter in ('nimble', 'qwen'):
            start, scores = time.perf_counter(), {}
            for batch in batches:
                scores.update(score_batch(backend, adapter, batch))
            row['adapters'][adapter] = {'ranking': sorted_scores(scores), 'scores': scores,
                                      'seconds': time.perf_counter() - start}
        output.append(row)
    return {'cohort': 'training only; identical BM25 top-ten pools, not a fifth retrieval option',
            'rows': output, 'requests': copy.deepcopy(backend.records[records_start:])}

# %%
def execute_mode(mode):
    export_frozen_inputs()
    backend = OllamaBackend()
    workflow_started = time.monotonic()
    backend.deadline = workflow_started + CFG.experiment['workflow_budget_seconds']
    backend.wall_deadline = time.time() + CFG.experiment['workflow_budget_seconds']
    try:
        gate = run_backend_gate(backend)
        if mode == 'gate':
            return gate
        if gate['status'] != 'passed':
            gate['requested_mode'] = mode
            gate['evaluation_not_started'] = 'Backend compatibility gate failed; inspect the recorded failure.'
            return gate
        if mode == 'demo':
            return execute_queries('demo', [DEMO_INPUT], DEMO_RETRIEVER, backend, gate)
        pilot = execute_queries('pilot', PILOT_INPUTS, ACTIVE_RETRIEVER, backend, gate)
        pilot['projection'] = pilot_projection(pilot, backend.remaining())
        if pilot['status'] != 'complete':
            pilot['adapter_check'] = {'status': 'not_run', 'reason': 'Pilot already failed; no extra model calls'}
            if mode == 'evaluate':
                pilot['evaluation_not_started'] = 'The pilot was incomplete or failed its gate.'
            atomic_json('results/pilot.json', pilot)
            return pilot
        try:
            pilot['adapter_check'] = adapter_check(backend)
        except StudyFailure as exc:
            pilot['projection']['passed'] = False
            pilot['adapter_check'] = {'status': 'failed', 'category': exc.category, 'message': str(exc)}
        # Repeat a fixed design probe after the pilot to expose output variability.
        repeat = shared_batch(PILOT_INPUTS[0], ACTIVE_RETRIEVER.passages[:2], 'passage')
        try:
            pilot['repeat_probe'] = {a: [backend.call(a, repeat), backend.call(a, repeat)]
                                     for a in ('nimble', 'qwen')}
        except StudyFailure as exc:
            pilot['repeat_probe'] = {'status': 'failed', 'category': exc.category, 'message': str(exc)}
            pilot['input_fit'] = False
        pilot['projection'] = pilot_projection(pilot, backend.remaining())
        if pilot.get('adapter_check', {}).get('status') == 'failed':
            pilot['projection']['passed'] = False
        pilot['workflow_elapsed_seconds'] = time.monotonic() - workflow_started
        atomic_json('results/pilot.json', pilot)
        if mode == 'pilot':
            return pilot
        if not pilot['projection']['passed']:
            pilot['evaluation_not_started'] = 'The pilot could not authorize evaluation under the chosen budget and validity controls.'
            atomic_json('results/pilot.json', pilot)
            return pilot
        return execute_queries('evaluation', ACTIVE_EVALUATION_INPUTS, ACTIVE_RETRIEVER, backend, gate)
    finally:
        backend.close()

def verify_request_evidence(run):
    with recorded_configuration(run):
        return verify_request_evidence_under_config(run)

def verify_request_evidence_under_config(run):
    """Rebuild every captured core request from immutable inputs and corpus IDs."""
    profile = study_scope(run)['profile']
    inputs = (mini_profile_data(profile)['evaluation_inputs'] if run.get('kind') == 'evaluation'
        else mini_profile_data(profile)['design_inputs']) if (
        is_mini_profile(profile) and run.get('kind') != 'demo') else (
        EVALUATION_INPUTS if run.get('kind') == 'evaluation' else DESIGN_INPUTS)
    by_q = {q['query_id']: q for q in inputs}
    pools = {'page': {c['id']: c for c in FULL_RETRIEVER.pages},
             'passage': {c['id']: c for c in FULL_RETRIEVER.passages}}
    rows = run.get('rows', [])
    assert len({(r['option'], r['query_id']) for r in rows}) == len(rows)
    checked = 0
    for row in rows:
        assert row['query_id'] in by_q
        if row['option'] in ('C', 'D') and row['status'] == 'success':
            assert all(BY_ID[cid]['page_id'] in row['selected_pages'] for cid in row['ranking'])
        for i in row['request_indices']:
            record = run['requests'][i]
            assert (record['query_id'], record['option']) == (row['query_id'], row['option'])
            batch = shared_batch(by_q[row['query_id']], [pools[record['stage']][cid]
                for cid in record['candidate_ids']], record['stage'])
            batch = select_questions(batch, record['aliases'])
            request = payload_for(record['adapter'], batch)
            assert sha256(request) == record['request_sha256'] == sha256(record['request'])
            assert batch['aliases'] == record['aliases']
            assert common_admission(batch) == record['admission']
            for attempt in record['attempts']:
                if attempt.get('status') == 'success':
                    assert parse_scores(record['adapter'], attempt['response'], batch) == record['scores']
            checked += 1
    return {'status': 'passed', 'query_method_rows': len(rows), 'reconstructed_requests': checked}

# %%
CHECKS = offline_checks()
display(Markdown(f"**Offline checks: {CHECKS['status']}** ({len(CHECKS['checks'])} checks)."))

# %%
# ORCHESTRATE: builder tags this cell so development tooling can load definitions.
if CFG.mode == 'replay':
    if CFG.replay_run_file:
        local = json.loads(Path(CFG.replay_run_file).read_text())
        if local.get('data_audit', {}).get('corpus_sha256') != DATA_AUDIT['corpus_sha256']:
            raise ValueError('Selected result file belongs to a different corpus')
        RUNS = {local['kind']: local}
    else:
        RUNS = copy.deepcopy(CAPTURED)
    if not RUNS:
        display(Markdown('**Not run:** no genuine model results have been embedded. Only corpus/schematic assets are available.'))
else:
    LIVE_RESULT = execute_mode(CFG.mode)
    # Newly measured output must never inherit an embedded historical evaluation.
    RUNS = {LIVE_RESULT['kind']: LIVE_RESULT}
    if (CFG.result_folder / 'gate.json').exists():
        RUNS['gate'] = json.loads((CFG.result_folder / 'gate.json').read_text())
    if CFG.mode == 'evaluate' and LIVE_RESULT['kind'] == 'evaluation' and (CFG.result_folder / 'pilot.json').exists():
        RUNS['pilot'] = json.loads((CFG.result_folder / 'pilot.json').read_text())

display(Markdown(f"**Execution mode:** `{CFG.mode}`. **Configured protocol/profile:** `{CFG.experiment['protocol']}` / `{CFG.study_profile}`. "
    'Corpus and tokenizer are verified external cached downloads. '
    'Result files retain their measured profile and protocol; read the run status below.'))

display(Markdown('\n'.join(f"- **{key}: {run.get('status')}**; `{run.get('run_id')}`; "
    f"{len(run.get('rows', []))} recorded method-query rows." for key, run in RUNS.items())))
FROZEN_INPUT_EXPORT = export_frozen_inputs()
