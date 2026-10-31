class StudyFailure(Exception):
    def __init__(self, category, message):
        super().__init__(message)
        self.category = category

class LocalSizeRejection(StudyFailure):
    def __init__(self, message):
        super().__init__('local_input_limit', message)

def corpus_candidates(corpus):
    pages = {}
    passages = {}
    for p in corpus:
        page = pages.setdefault(p['page_id'], {'id': p['page_id'], 'domain': p['domain'],
            'title': re.sub(r'#\d+$', '', p['title']), 'headings': []})
        heading = ' > '.join(p['parent_headings'] + [p['heading']])
        if heading not in page['headings']:
            page['headings'].append(heading)
        passages[p['passage_id']] = {'id': p['passage_id'], 'domain': p['domain'],
            'title': p['title'], 'heading': p['heading'],
            'parent_headings': p['parent_headings'], 'text': p['text']}
    return list(pages.values()), list(passages.values())

def shared_batch(query, candidates, stage):
    candidates = sorted(candidates, key=lambda c: SOURCE_ORDER[c['id']])
    aliases = {f'c{i + 1:03d}': c['id'] for i, c in enumerate(candidates)}
    state = {'conversation': model_history(query), 'candidates': [
        {'alias': alias, **candidate} for alias, candidate in zip(aliases, candidates)]}
    questions = {a: {'type': 'noul', 'instructions': CFG.questions[stage].format(alias=a)}
                 for a in aliases}
    return {'state': stable_json(state), 'questions': questions, 'aliases': aliases,
            'stage': stage, 'candidate_ids': list(aliases.values())}

def select_questions(batch, aliases):
    """Keep every candidate in state; ask only the selected relevance questions."""
    aliases = list(aliases)
    assert aliases and len(aliases) == len(set(aliases))
    assert set(aliases) <= set(batch['aliases'])
    return {**batch, 'questions': {a: batch['questions'][a] for a in aliases},
            'aliases': {a: batch['aliases'][a] for a in aliases}}

def question_batches(batch):
    aliases = list(batch['aliases'])
    size = CFG.experiment.get('questions_per_request', 16)
    return [select_questions(batch, aliases[i:i + size]) for i in range(0, len(aliases), size)]

def score_batch(backend, adapter, batch):
    """Both adapters see identical candidate states and four-question groups."""
    scores = {}
    for part in question_batches(batch):
        received = backend.call(adapter, part)
        assert set(received) == set(part['aliases'].values())
        assert not (set(scores) & set(received))
        scores.update(received)
    assert set(scores) == set(batch['aliases'].values())
    return scores

# %%

def nimble_rendered_prompts(batch):
    # Installed tokenizer.chat_template.systemone, pinned and boundary-checked.
    schema = [{'name': name, 'description': q['instructions'], 'choices': [
        {'code': 'A', 'value': False}, {'code': 'B', 'value': True}]}
        for name, q in batch['questions'].items()]
    data = json.dumps({'context': batch['state'], 'schema': schema}, ensure_ascii=False)
    # The native Jinja template escapes chat delimiters after JSON serialization.
    data = data.replace('<', r'\u003c').replace('>', r'\u003e')
    return ['<|im_start|>system\n' + CFG.nimble_system + '<|im_end|>\n<|im_start|>user\n'
        + data + '\n\nRequested field: ' + json.dumps(name)
        + '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
        for name in batch['questions']]

def qwen_rendered_prompt(batch):
    # raw=True makes this exact prompt the server input. No implicit template.
    content = stable_json({'state': json.loads(batch['state']),
        'questions': batch['questions']})
    return ('<|im_start|>system\n' + CFG.score_instruction + '<|im_end|>\n'
        '<|im_start|>user\n' + content + '<|im_end|>\n'
        '<|im_start|>assistant\n<think>\n\n</think>\n\n')

def score_schema(batch):
    aliases = list(batch['aliases'])
    return {'type': 'object', 'properties': {'scores': {'type': 'object',
        'properties': {a: {'type': 'number', 'minimum': 0, 'maximum': 1} for a in aliases},
        'required': aliases, 'additionalProperties': False}},
        'required': ['scores'], 'additionalProperties': False}

def payload_for(adapter, batch):
    if adapter == 'nimble':
        return {'model': CFG.experiment['models'][adapter], 'state': batch['state'],
                'questions': batch['questions'], 'keep_alive': CFG.experiment['keep_alive']}
    return {'model': CFG.experiment['models']['qwen'], 'prompt': qwen_rendered_prompt(batch),
        'raw': True, 'stream': False, 'think': False, 'format': score_schema(batch),
        'keep_alive': CFG.experiment['keep_alive'], 'options': {'temperature': CFG.qwen_options['temperature'],
            'seed': CFG.experiment['seed'], 'num_predict': CFG.experiment['output_reserve'],
            'num_ctx': CFG.experiment['context_tokens'], 'presence_penalty': CFG.qwen_options['presence_penalty'],
            'repeat_penalty': CFG.qwen_options['repeat_penalty']}}

def token_count(text):
    return len(TOKENIZER.encode(text, add_special_tokens=False).ids)

def common_admission(batch):
    counts = [token_count(p) for p in nimble_rendered_prompts(batch)]
    qwen_count = token_count(qwen_rendered_prompt(batch))
    sizes = {a: len(stable_json(payload_for(a, batch)).encode()) for a in ('nimble', 'qwen')}
    receipt = {'nimble_row_tokens': counts, 'qwen_prompt_tokens': qwen_count,
               'request_bytes': sizes, 'input_ceiling': CFG.experiment['context_tokens'] - CFG.experiment['output_reserve']}
    if max(counts + [qwen_count]) > receipt['input_ceiling']:
        raise LocalSizeRejection('Complete rendered input exceeds common token ceiling')
    if max(sizes.values()) > CFG.experiment['json_byte_limit']:
        raise LocalSizeRejection('Request exceeds common 48 KiB guard')
    return receipt

def parse_scores(adapter, response, batch):
    expected = set(batch['aliases'])
    if adapter == 'nimble':
        answers = response.get('answers')
        if not isinstance(answers, dict) or set(answers) != expected:
            raise StudyFailure('invalid_schema', 'Nimble answer aliases differ')
        if any(not isinstance(v, dict) or set(v) != {'type', 'noul'} or v['type'] != 'noul'
               for v in answers.values()):
            raise StudyFailure('invalid_schema', 'Invalid noul response')
        scores = {a: v['noul'] for a, v in answers.items()}
    else:
        if response.get('thinking'):
            raise StudyFailure('unexpected_thinking', 'Unexpected reasoning output')
        if response.get('done') is not True or response.get('done_reason') != 'stop':
            raise StudyFailure('truncated_generation', 'Generation did not terminate normally')
        try:
            def unique_object(pairs):
                result = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError('Duplicate JSON key: ' + key)
                    result[key] = value
                return result
            parsed = json.loads(response['response'], object_pairs_hook=unique_object)
        except (KeyError, TypeError, ValueError) as exc:
            raise StudyFailure('invalid_schema', str(exc))
        if not isinstance(parsed, dict) or set(parsed) != {'scores'}:
            raise StudyFailure('invalid_schema', 'Expected only scores object')
        scores = parsed['scores']
        if not isinstance(scores, dict) or set(scores) != expected:
            raise StudyFailure('invalid_schema', 'Qwen score aliases differ')
    if any(isinstance(s, bool) or not isinstance(s, (int, float)) or not math.isfinite(s)
           or not 0 <= s <= 1 for s in scores.values()):
        raise StudyFailure('invalid_score', 'Scores must be finite numbers within [0,1]')
    return {batch['aliases'][a]: float(scores[a]) for a in batch['aliases']}

# %%
class OllamaBackend:
    def __init__(self, url=CFG.ollama_url, deadline=None):
        self.client = httpx.Client(base_url=url, timeout=CFG.experiment['request_timeout'], trust_env=False)
        self.deadline = deadline
        self.wall_deadline = None
        self.records = []
        self.current_query = None
        self.current_option = None

    def close(self):
        self.client.close()

    def remaining(self):
        limits = [CFG.experiment['request_timeout']] if self.deadline is None and self.wall_deadline is None else []
        if self.deadline is not None:
            limits.append(self.deadline - time.monotonic())
        if self.wall_deadline is not None:
            limits.append(self.wall_deadline - time.time())
        return min(limits)

    def timeout(self):
        remaining = self.remaining()
        if remaining <= 0:
            raise StudyFailure('deadline', 'Whole-workflow time allowance exhausted')
        return min(CFG.experiment['request_timeout'], remaining)

    def metadata(self):
        version = self.client.get('/api/version', timeout=self.timeout()); version.raise_for_status()
        tags = self.client.get('/api/tags', timeout=self.timeout()); tags.raise_for_status()
        models = {}
        for adapter, model in CFG.experiment['models'].items():
            response = self.client.post('/api/show', json={'model': model}, timeout=self.timeout())
            response.raise_for_status()
            models[adapter] = response.json()
        return {'version': version.json(), 'tags': tags.json(), 'models': models,
                'captured_at': utc_now(), 'software': software_versions(),
                'hardware': {'platform': platform.platform(), 'machine': platform.machine()},
                'configuration': copy.deepcopy(CFG.experiment), 'data_audit': DATA_AUDIT}

    def residency(self):
        response = self.client.get('/api/ps', timeout=self.timeout()); response.raise_for_status()
        return {'captured_at': utc_now(), **response.json()}

    def call(self, adapter, batch):
        admitted = common_admission(batch)  # Same admission, even for B.
        payload = payload_for(adapter, batch)
        path = '/v1/systemone' if adapter == 'nimble' else '/api/generate'
        trace = {'query_id': self.current_query, 'option': self.current_option,
            'adapter': adapter, 'stage': batch['stage'], 'aliases': batch['aliases'],
            'candidate_ids': batch['candidate_ids'], 'request': payload,
            'request_sha256': sha256(payload), 'admission': admitted, 'attempts': [],
            'started_at': utc_now()}
        self.records.append(trace)
        for attempt in range(CFG.experiment['transient_retries'] + 1):
            remaining = self.remaining()
            if remaining <= 0:
                raise StudyFailure('deadline', 'Study deadline exhausted')
            start = time.perf_counter()
            item = {'attempt': attempt + 1, 'started_at': utc_now()}
            trace['attempts'].append(item)
            retry = False
            try:
                response = self.client.post(path, json=payload,
                    timeout=min(CFG.experiment['request_timeout'], remaining))
                item['http_status'] = response.status_code
                try:
                    item['response'] = response.json()
                except ValueError:
                    item['response_text'] = response.text
                    raise StudyFailure('invalid_schema', 'Server returned non-JSON')
                if response.status_code >= 400:
                    category = ('server_input_limit' if response.status_code == 413 or
                        re.search(r'context|too long|64 KiB|batch size', response.text, re.I) else
                        'unsupported_backend' if response.status_code in (400, 404, 501) else 'infrastructure')
                    retry = category == 'infrastructure' and response.status_code in (408, 429, 500, 502, 503, 504)
                    raise StudyFailure(category, response.text[:1000])
                scores = parse_scores(adapter, item['response'], batch)
                # Actual prompt accounting is a continuing no-truncation check.
                actual = item['response'].get('prompt_eval_count') if adapter == 'qwen' else None
                if actual is not None and actual != admitted['qwen_prompt_tokens']:
                    raise StudyFailure('token_accounting', f'Qwen prompt tokens {actual} != expected {admitted["qwen_prompt_tokens"]}')
                if adapter == 'nimble':
                    actual = item['response'].get('usage', {}).get('input_tokens')
                    expected = sum(admitted['nimble_row_tokens'])
                    if actual != expected:
                        raise StudyFailure('token_accounting', f'Nimble prompt tokens {actual} != expected {expected}')
                item['status'] = 'success'
                trace['scores'] = scores
                trace['distinct_scores'] = len(set(scores.values()))
                trace['all_equal'] = len(set(scores.values())) == 1
                return scores
            except httpx.TransportError as exc:
                retry = True
                failure = StudyFailure('infrastructure', type(exc).__name__ + ': ' + str(exc))
                item['error'] = {'category': failure.category, 'message': str(failure)}
            except StudyFailure as exc:
                failure = exc
                item['error'] = {'category': exc.category, 'message': str(exc)}
            finally:
                item['seconds'] = time.perf_counter() - start
            if not retry or attempt == CFG.experiment['transient_retries']:
                raise failure

# %%
def deterministic_batches(query, candidates, stage, events):
    candidates = sorted(candidates, key=lambda c: SOURCE_ORDER[c['id']])
    def fit_chunk(chunk):
        batch = shared_batch(query, chunk, stage)
        try:
            common_admission(batch)
            return [batch]
        except LocalSizeRejection as exc:
            events.append({'stage': stage, 'event': 'local_split',
                'candidate_ids': [c['id'] for c in chunk], 'reason': str(exc)})
            if len(chunk) == 1:
                raise
            mid = len(chunk) // 2
            return fit_chunk(chunk[:mid]) + fit_chunk(chunk[mid:])
    output = []
    for i in range(0, len(candidates), CFG.experiment['binary_batch_max']):
        output.extend(fit_chunk(candidates[i:i + CFG.experiment['binary_batch_max']]))
    return output

def sorted_scores(scores):
    return sorted(scores, key=lambda cid: (-scores[cid], cid))

def replace_prefix(coarse, prefix, rescored):
    assert set(prefix) == set(rescored)
    result = sorted_scores(rescored) + coarse[len(prefix):]
    assert len(result) == len(set(result)) == len(coarse)
    assert set(result) == set(coarse)
    return result

def rank_pool(backend, adapter, query, candidates, stage, events):
    start = time.perf_counter()
    coarse_scores = {}
    for batch in deterministic_batches(query, candidates, stage, events):
        coarse_scores.update(score_batch(backend, adapter, batch))
    coarse = sorted_scores(coarse_scores)
    by_id = {c['id']: c for c in candidates}
    prefix = coarse[:CFG.experiment['rescore_prefix_max']]
    while len(prefix) >= 2:
        batch = shared_batch(query, [by_id[x] for x in prefix], stage)
        try:
            common_admission(batch)
            rescored = score_batch(backend, adapter, batch)
            ranked = replace_prefix(coarse, prefix, rescored)
            break
        except LocalSizeRejection:
            prefix = prefix[:-1]
    else:
        prefix, rescored, ranked = [], {}, coarse
    result = {'stage': stage, 'candidate_ids': [c['id'] for c in candidates],
        'coarse_scores': coarse_scores, 'coarse_ranking': coarse,
        'rescore_prefix': prefix, 'rescore_scores': rescored,
        'rescore_skipped': not bool(prefix), 'ranking': ranked,
        'seconds': time.perf_counter() - start}
    events.append(result)
    return ranked

# %%
class Retriever:
    def __init__(self, corpus):
        self.corpus = corpus
        self.pages, self.passages = corpus_candidates(corpus)
        self.passage_by_id = {c['id']: c for c in self.passages}
        self.bm25 = BM25Okapi([tokenize(' '.join([p['title'], *p['parent_headings'],
            p['heading'], p['text']])) for p in corpus],
            k1=CFG.experiment['bm25_k1'], b=CFG.experiment['bm25_b'])
        self.passage_ids = [p['passage_id'] for p in corpus]

    def lexical(self, query, k):
        words = tokenize(' '.join(t['text'] for t in model_history(query)))
        scores = self.bm25.get_scores(words)
        indices = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), self.passage_ids[i]))[:k]
        return [self.passage_ids[i] for i in indices]

    def run(self, option, query, backend):
        start, request_start = time.perf_counter(), len(backend.records)
        backend.current_query, backend.current_option = query['query_id'], option
        row = {'query_id': query['query_id'], 'dialogue_id': query['dialogue_id'],
            'option': option, 'started_at': utc_now(), 'events': [], 'selected_pages': [],
            'ranking': [], 'status': 'failed', 'failure': None}
        try:
            if option == 'A':
                ranking = self.lexical(query, CFG.experiment['output_k'])
            elif option == 'B':
                candidates = [self.passage_by_id[x] for x in self.lexical(query, CFG.experiment['bm25_pool'])]
                ranking = rank_pool(backend, 'nimble', query, candidates, 'passage', row['events'])
            else:
                adapter = 'nimble' if option == 'C' else 'qwen'
                pages = rank_pool(backend, adapter, query, self.pages, 'page', row['events'])
                row['selected_pages'] = pages[:CFG.experiment['page_budget']]
                allowed = {p['passage_id'] for p in self.corpus
                           if p['page_id'] in row['selected_pages']}
                candidates = [p for p in self.passages if p['id'] in allowed]
                ranking = rank_pool(backend, adapter, query, candidates, 'passage', row['events'])
                assert set(ranking) <= allowed, 'Passage branch leaked'
            row['ranking'], row['status'] = ranking[:CFG.experiment['output_k']], 'success'
        except StudyFailure as exc:
            row['failure'] = {'category': exc.category, 'message': str(exc)}
        finally:
            row['seconds'] = time.perf_counter() - start
            row['request_indices'] = list(range(request_start, len(backend.records)))
            row['request_count'] = len(row['request_indices'])
            row['attempt_count'] = sum(len(r['attempts']) for r in backend.records[request_start:])
        return row

FULL_RETRIEVER = Retriever(CORPUS)
MINI_RETRIEVER = Retriever(MINI_CORPUS)
ACTIVE_RETRIEVER = MINI_RETRIEVER if is_mini_profile(CFG.study_profile) else FULL_RETRIEVER
DEMO_RETRIEVER = Retriever(DEMO_CORPUS)
SOURCE_ORDER = {c['id']: i for i, c in enumerate(FULL_RETRIEVER.pages + FULL_RETRIEVER.passages)}
assert len(SOURCE_ORDER) == 4523
