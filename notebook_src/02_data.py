# Corpus, annotations, tokenizer and inference traces are external cached files.
# This notebook contains implementation and frozen selection metadata only.
def fetch_cached(url, destination, expected_sha256):
    destination = Path(destination)
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError('Cached file checksum mismatch: ' + str(destination))
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + '.part')
    digest = hashlib.sha256()
    try:
        print('Downloading', url, 'to', destination, flush=True)
        with httpx.stream('GET', url, follow_redirects=True, timeout=CFG.download_timeout) as response:
            response.raise_for_status()
            with partial.open('wb') as output:
                for chunk in response.iter_bytes():
                    digest.update(chunk); output.write(chunk)
        if digest.hexdigest() != expected_sha256:
            raise ValueError('Downloaded file checksum mismatch: ' + str(destination))
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    return destination

def make_mini_profile(corpus, train_queries, validation_queries, excluded_dialogues):
    """Corpus first; then one eligible query per dialogue. No model scores used."""
    by_id = {p['passage_id']: p for p in corpus}
    def order(prefix, value):
        return hashlib.sha256((prefix + ':' + value).encode()).hexdigest()
    selected_pages = [page for domain in CFG.domains for page in sorted(
        {p['page_id'] for p in corpus if p['domain'] == domain},
        key=lambda page: order('mini-pages-42', page))[:8]]
    selected = set(selected_pages)
    def choose(queries, count, prefix, exclude):
        output = []
        for domain in CFG.domains:
            grouped = defaultdict(list)
            for query in queries:
                if query['domain'] == domain and query['dialogue_id'] not in exclude and all(
                        by_id[pid]['page_id'] in selected for pid in query['gold_passage_ids']):
                    grouped[query['dialogue_id']].append(query)
            if len(grouped) < count:
                raise ValueError(f'Mini corpus has only {len(grouped)} eligible {domain} dialogues, needs {count}')
            for dial in sorted(grouped, key=lambda d: order(prefix, d))[:count]:
                output.append(min(grouped[dial], key=lambda q: order(prefix + '-query', q['query_id'])))
        return output
    design = choose(train_queries, 2, 'mini-design-42', set())
    evaluation = choose(validation_queries, 6, 'mini-evaluation-42', set(excluded_dialogues))
    return {'profile': 'mini_2h', 'page_ids': selected_pages,
        'design_queries': design, 'evaluation_queries': evaluation,
        'selection': 'Eight pages per domain by SHA256(mini-pages-42:page_id); one hash-selected eligible query per hash-selected dialogue, two training and six validation dialogues per domain. Original frozen evaluation and explored validation dialogues excluded.',
        'claim_scope': 'Restricted-corpus mechanism demonstration; exploratory comparison, not full-paper reproduction or the original confirmatory benchmark.'}


def make_three_hour_profile(legacy):
    """Freeze the first three hash-ordered queries/domain before any inference."""
    labels = legacy['evaluation_labels']
    inputs = [q for domain in CFG.domains for q in
        [q for q in legacy['evaluation_inputs'] if labels[q['query_id']]['domain'] == domain][:3]]
    return {**legacy, 'profile': 'mini_3h', 'evaluation_inputs': inputs,
        'evaluation_labels': {q['query_id']: labels[q['query_id']] for q in inputs},
        'selection': legacy['selection'] + ' The three-hour profile takes the first three frozen hash-ordered validation queries per domain, with no model-outcome selection.'}

def is_mini_profile(profile):
    return profile in ('mini_2h', 'mini_3h')

def mini_profile_data(profile):
    return DATASET['mini_3h_profile'] if profile == 'mini_3h' else DATASET['mini_profile']

def prepare_public_archive(archive_bytes):
    """Normalize the pinned archive and reconstruct both frozen study profiles."""
    assert hashlib.sha256(archive_bytes).hexdigest() == CFG.source_sha256
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        docs = json.loads(archive.read('multidoc2dial/multidoc2dial_doc.json'))['doc_data']
        splits = {s: json.loads(archive.read('multidoc2dial/multidoc2dial_dial_' + s + '.json'))['dial_data']
                  for s in ('train', 'validation')}
    passages, span_map = [], {}
    for domain in CFG.domains:
        for raw_id, doc in docs[domain].items():
            groups, previous_section, previous_title = [], None, None
            for span in doc['spans'].values():
                title = span['title'].strip()
                if not groups or not (previous_section == span['id_sec'] or previous_title == title):
                    groups.append([])
                groups[-1].append(span)
                previous_section, previous_title = span['id_sec'], title
            for index, spans in enumerate(groups):
                pid = f'{domain}:{raw_id}:p{index:03d}'
                passages.append({'passage_id': pid,
                    'page_id': domain + ':' + re.sub(r'#\d+$', '', doc['title']),
                    'raw_document_id': raw_id, 'domain': domain, 'title': doc['title'],
                    'heading': spans[0]['title'],
                    'parent_headings': [x['text'] for x in spans[0]['parent_titles']],
                    'span_ids': [s['id_sp'] for s in spans],
                    'text': ' '.join(s['text_sp'] for s in spans).strip()})
                for span in spans:
                    span_map[domain, raw_id, span['id_sp']] = pid
    rebuilt = {'corpus': passages}
    for name, split in (('design', 'train'), ('evaluation', 'validation')):
        inputs, labels = [], {}
        for domain in CFG.domains:
            by_dial = {d['dial_id']: d for d in splits[split][domain]}
            for dial_id in EMBEDDED_DATA['frozen_dialogue_ids'][name][domain]:
                turns = by_dial[dial_id]['turns']
                for i, turn in enumerate(turns[:-1]):
                    target = turns[i + 1]
                    if turn['role'] != 'user' or target['role'] != 'agent' or target['da'] == 'respond_no_solution':
                        continue
                    qid = f"{dial_id}_{turn['turn_id']}"
                    gold = sorted({span_map[domain, ref['doc_id'], ref['id_sp']]
                                   for ref in target['references']})
                    assert len(gold) == 1
                    inputs.append({'query_id': qid, 'dialogue_id': dial_id,
                        'user_turn_id': turn['turn_id'],
                        'history': [{'role': t['role'], 'text': t['utterance']}
                                    for t in turns[max(0, i - 5):i + 1]]})
                    labels[qid] = {'domain': domain, 'gold_passage_ids': gold,
                        'target_agent_turn_id': target['turn_id'],
                        'reference_answer': target['utterance']}
        rebuilt[name + '_inputs'], rebuilt[name + '_labels'] = inputs, labels
    # Rebuild mini selection from all eligible source queries, not the original
    # 125/259-query subsets. Membership may use annotations; inference may not.
    eligible = {}
    for split in ('train', 'validation'):
        eligible[split] = []
        for domain in CFG.domains:
            for dialogue in splits[split][domain]:
                turns = dialogue['turns']
                for i, turn in enumerate(turns[:-1]):
                    target = turns[i + 1]
                    if turn['role'] != 'user' or target['role'] != 'agent' or target['da'] == 'respond_no_solution':
                        continue
                    gold = sorted({span_map[domain, ref['doc_id'], ref['id_sp']] for ref in target['references']})
                    assert len(gold) == 1
                    eligible[split].append({'query_id': f"{dialogue['dial_id']}_{turn['turn_id']}",
                        'dialogue_id': dialogue['dial_id'], 'domain': domain,
                        'user_turn_id': turn['turn_id'], 'target_agent_turn_id': target['turn_id'],
                        'history': [{'role': t['role'], 'text': t['utterance']} for t in turns[max(0, i - 5):i + 1]],
                        'gold_passage_ids': gold, 'reference_answer': target['utterance']})
    excluded = set(EMBEDDED_DATA['explored_validation_dialogues']) | {q['dialogue_id'] for q in rebuilt['evaluation_inputs']}
    mini = make_mini_profile(passages, eligible['train'], eligible['validation'], excluded)
    for name in ('design', 'evaluation'):
        combined = mini.pop(name + '_queries')
        mini[name + '_inputs'] = [{k: q[k] for k in ('query_id', 'dialogue_id', 'user_turn_id', 'history')} for q in combined]
        mini[name + '_labels'] = {q['query_id']: {k: q[k] for k in (
            'domain', 'gold_passage_ids', 'target_agent_turn_id', 'reference_answer')} for q in combined}
    rebuilt['mini_profile'] = mini
    rebuilt['mini_3h_profile'] = make_three_hour_profile(mini)
    for key, value in rebuilt.items():
        assert sha256(value) == EMBEDDED_DATA['expected_hashes'][key], 'Rebuild mismatch: ' + key
    return rebuilt

ARCHIVE_PATH = fetch_cached(CFG.source_url, CFG.data_folder / 'raw/multidoc2dial.zip', CFG.source_sha256)
DATASET = prepare_public_archive(ARCHIVE_PATH.read_bytes())
CORPUS = DATASET['corpus']
DESIGN_INPUTS, DESIGN_LABELS = DATASET['design_inputs'], DATASET['design_labels']
EVALUATION_INPUTS, EVALUATION_LABELS = DATASET['evaluation_inputs'], DATASET['evaluation_labels']
TOKENIZER_PATH = fetch_cached(EMBEDDED_DATA['tokenizer_url'], CFG.data_folder / 'runtime/tokenizer.json',
    EMBEDDED_DATA['attribution']['tokenizer_sha256'])
TOKENIZER = Tokenizer.from_file(str(TOKENIZER_PATH))
CAPTURED = {name: json.loads((CFG.result_folder / (name + '.json')).read_text())
    for name in ('gate', 'demo', 'pilot', 'evaluation') if (CFG.result_folder / (name + '.json')).exists()}
BY_ID = {p['passage_id']: p for p in CORPUS}

def rebuild_from_public_archive(archive_bytes=None):
    rebuilt = prepare_public_archive(ARCHIVE_PATH.read_bytes() if archive_bytes is None else archive_bytes)
    return {key: sha256(value) for key, value in rebuilt.items()}


MINI = mini_profile_data(CFG.study_profile)
MINI_CORPUS = [p for p in CORPUS if p['page_id'] in set(MINI['page_ids'])]
MINI_DESIGN_INPUTS, MINI_DESIGN_LABELS = MINI['design_inputs'], MINI['design_labels']
MINI_EVALUATION_INPUTS, MINI_EVALUATION_LABELS = MINI['evaluation_inputs'], MINI['evaluation_labels']
assert len(MINI['page_ids']) == 32 and len(MINI_DESIGN_INPUTS) == 8
assert len(MINI_EVALUATION_INPUTS) == (12 if CFG.study_profile == 'mini_3h' else 24)
assert len({q['dialogue_id'] for q in MINI_EVALUATION_INPUTS}) == len(MINI_EVALUATION_INPUTS)
assert not ({q['dialogue_id'] for q in EVALUATION_INPUTS} & {q['dialogue_id'] for q in MINI_EVALUATION_INPUTS})
assert all(BY_ID[pid]['page_id'] in set(MINI['page_ids']) for labels in
    (MINI_DESIGN_LABELS, MINI_EVALUATION_LABELS) for label in labels.values() for pid in label['gold_passage_ids'])
ACTIVE_CORPUS = MINI_CORPUS if is_mini_profile(CFG.study_profile) else CORPUS
ACTIVE_EVALUATION_INPUTS = MINI_EVALUATION_INPUTS if is_mini_profile(CFG.study_profile) else EVALUATION_INPUTS
ACTIVE_EVALUATION_LABELS = MINI_EVALUATION_LABELS if is_mini_profile(CFG.study_profile) else EVALUATION_LABELS

def study_scope(run=None):
    profile = (run or {}).get('configuration', {}).get('study_profile',
        'full' if run is not None else CFG.study_profile)
    mini = is_mini_profile(profile)
    corpus = MINI_CORPUS if mini else CORPUS
    data = mini_profile_data(profile) if mini else None
    labels = data['evaluation_labels'] if mini else EVALUATION_LABELS
    design = data['design_labels'] if mini else DESIGN_LABELS
    return {'profile': profile, 'pages': len({p['page_id'] for p in corpus}),
        'passages': len(corpus), 'evaluation_queries': len(labels),
        'evaluation_dialogues': len({q['dialogue_id'] for q in data['evaluation_inputs']}) if mini else 40,
        'design_queries': len(design), 'pilot_queries': 8 if mini else 10,
        'corpus_sha256': sha256(corpus), 'evaluation_labels': labels,
        'design_labels': design, 'pilot_quotas': dict.fromkeys(CFG.domains, 2) if mini else
            {'dmv': 3, 'ssa': 3, 'va': 2, 'studentaid': 2}}

def validate_dataset():
    assert len(CORPUS) == len(BY_ID) == 4110
    assert len({p['page_id'] for p in CORPUS}) == 413
    assert len(DESIGN_INPUTS) == 125 and len(EVALUATION_INPUTS) == 259
    assert len({q['dialogue_id'] for q in EVALUATION_INPUTS}) == 40
    assert Counter(q['domain'] for q in EVALUATION_LABELS.values()) == {
        'dmv': 66, 'ssa': 60, 'va': 69, 'studentaid': 64}
    assert not ({q['dialogue_id'] for q in DESIGN_INPUTS} &
                {q['dialogue_id'] for q in EVALUATION_INPUTS})
    assert not ({q['dialogue_id'] for q in EVALUATION_INPUTS} &
                set(EMBEDDED_DATA['explored_validation_dialogues']))
    for inputs, labels in ((DESIGN_INPUTS, DESIGN_LABELS),
                           (EVALUATION_INPUTS, EVALUATION_LABELS)):
        assert set(q['query_id'] for q in inputs) == set(labels)
        for q in inputs:
            assert set(q) == {'query_id', 'dialogue_id', 'user_turn_id', 'history'}
            assert 1 <= len(q['history']) <= 6
            assert q['history'][-1]['role'] == 'user'
            gold = labels[q['query_id']]['gold_passage_ids']
            assert len(gold) == 1 and gold[0] in BY_ID
    return {'passages': len(CORPUS), 'pages': 413, 'design_queries': 125,
            'evaluation_queries': 259, 'evaluation_dialogues': 40,
            'corpus_sha256': sha256(CORPUS),
            'evaluation_inputs_sha256': sha256(EVALUATION_INPUTS),
            'evaluation_labels_sha256': sha256(EVALUATION_LABELS)}

DATA_AUDIT = validate_dataset()


def model_history(query):
    # Allowlist, rather than filtering gold-related keys after serialization.
    return [{'role': t['role'], 'text': t['text']} for t in query['history'][-6:]]

def select_pilot_queries():
    quotas = {'dmv': 3, 'ssa': 3, 'va': 2, 'studentaid': 2}
    ordered = sorted(DESIGN_INPUTS, key=lambda q: sha256(('42:' + q['query_id']).encode()))
    return [q for domain in CFG.domains for q in [q for q in ordered
        if DESIGN_LABELS[q['query_id']]['domain'] == domain][:quotas[domain]]]

def training_demo_corpus(query):
    demo_gold = DESIGN_LABELS[query['query_id']]['gold_passage_ids'][0]
    demo_page = BY_ID[demo_gold]['page_id']
    all_page_ids = sorted({p['page_id'] for p in CORPUS})
    page_ids = [demo_page] + sorted([p for p in all_page_ids if p != demo_page],
        key=lambda p: sha256(('demo42:' + p).encode()))[:9]
    return page_ids, [p for p in CORPUS if p['page_id'] in page_ids]

FULL_PILOT_INPUTS = select_pilot_queries()
PILOT_INPUTS = MINI_DESIGN_INPUTS if is_mini_profile(CFG.study_profile) else FULL_PILOT_INPUTS
DEMO_INPUT = FULL_PILOT_INPUTS[0]
DEMO_PAGE_IDS, DEMO_CORPUS = training_demo_corpus(DEMO_INPUT)
assert len(DEMO_PAGE_IDS) == 10

display(Markdown(f"**Archive rebuilt and verified:** {DATA_AUDIT['pages']} pages / "
    f"{DATA_AUDIT['passages']:,} passages. **Selected profile:** `{CFG.study_profile}`; "
    f"{study_scope()['pages']} pages / {study_scope()['passages']} passages / "
    f"{study_scope()['evaluation_queries']} validation queries."))
if CAPTURED and not CAPTURED.get('evaluation'):
    pilot_status = CAPTURED.get('pilot', {}).get('status', 'not run')
    display(Markdown('**Held-out evaluation: NOT RUN.** Captured pilot status: **' + pilot_status +
        '**. The notebook demonstrates the four implementations and preserves actual compatibility/runtime findings; '
        'it does not establish the paper’s accuracy or a decision-model efficiency win.'))
