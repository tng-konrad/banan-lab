"""Bounded diagnostic only. Does not alter notebook or benchmark results."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import httpx
from run_live import load_runtime

s = load_runtime()
pilot = json.loads(Path('results/pilot.json').read_text())
record = next(r for r in pilot['requests'] if r['adapter'] == 'nimble')
original = record['request']
questions = list(original['questions'].items())
rows = []
started = time.monotonic()
cases = [(str(count), original, count) for count in (1, 4, 8, 16, 16)]
if '--churn' in sys.argv:
    development = json.loads(Path('results/development/pilot-v5.1-retry-policy-violation.json').read_text())
    states = []
    for r in development['requests']:
        if r['adapter'] == 'nimble' and r['stage'] == 'page' and len(r['request']['questions']) == 16:
            if not any(p['state'] == r['request']['state'] for p in states):
                states.append(r['request'])
        if len(states) == 3:
            break
    cases = [(str(count) + '-' + str(i), states[j], count)
             for count in (16, 4) for i, j in enumerate((0, 1, 2, 0))]
with httpx.Client(base_url='http://127.0.0.1:11434', timeout=30, trust_env=False) as client:
    for name, source, count in cases:
        if time.monotonic() - started > 75:
            break
        payload = copy.deepcopy(source)
        payload['questions'] = dict(list(source['questions'].items())[:count])
        batch = {'state': payload['state'], 'questions': payload['questions']}
        counts = [s['token_count'](p) for p in s['nimble_rendered_prompts'](batch)]
        before = time.perf_counter()
        try:
            response = client.post('/v1/systemone', json=payload, timeout=min(30, 90-(time.monotonic()-started)))
            result = {'case': name, 'questions': count, 'candidate_count': len(json.loads(payload['state'])['candidates']),
                'per_question_prompt_tokens': counts, 'http_status': response.status_code,
                'response': response.json(), 'seconds': time.perf_counter()-before}
        except httpx.HTTPError as exc:
            result = {'questions': count, 'error': str(exc), 'seconds': time.perf_counter()-before}
        rows.append(result)
        print(json.dumps({k:v for k,v in result.items() if k != 'response'}), flush=True)
    ps = client.get('/api/ps').json()
result = {'captured_at': datetime.now(timezone.utc).isoformat(),
    'purpose': 'diagnostic only, identical full candidate state, varying question count; no benchmark result replaced',
    'original_request_sha256': record['request_sha256'], 'rows': rows, 'residency': ps}
Path('results/context_churn_diagnostic.json' if '--churn' in sys.argv else 'results/context_diagnostic.json').write_text(json.dumps(result, indent=2))
