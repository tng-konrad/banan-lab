"""Offline integration checks for mini scope, projection and recorded settings.

Fixtures verify plumbing only; they are never saved as model measurements.
"""
from collections import Counter
import json
from pathlib import Path
import time
import matplotlib
matplotlib.use('Agg')
from run_live import load_runtime

s = load_runtime()
assert s['STUDY_PROFILE'] == 'mini_3h'
assert len(s['ACTIVE_RETRIEVER'].pages) == 32
assert len(s['ACTIVE_RETRIEVER'].passages) == 329
assert s['MODE'] == 'replay' and s['CONFIG']['workflow_budget_seconds'] == 10800
assert len(s['PILOT_INPUTS']) == 8 and len(s['ACTIVE_EVALUATION_INPUTS']) == 12
assert len(s['freeze_schedule'](s['ACTIVE_EVALUATION_INPUTS'])) == 48

class FixtureBackend:
    def __init__(self):
        self.records = []
    def call(self, adapter, batch):
        assert set(batch['candidate_ids']) <= set(s['SOURCE_ORDER'])
        assert len(batch['aliases']) <= 4
        return {cid: .4 for cid in batch['aliases'].values()}

# Run each complete controller on a real mini query from every domain, with
# constant scores. This checks partitions/retained branches, not model quality.
checked = []
for domain in s['DOMAINS']:
    q = next(q for q in s['MINI_EVALUATION_INPUTS'] if s['MINI_EVALUATION_LABELS'][q['query_id']]['domain'] == domain)
    for method in s['METHODS']:
        row = s['ACTIVE_RETRIEVER'].run(method, q, FixtureBackend())
        assert row['status'] == 'success', row.get('failure')
        assert len(row['ranking']) == 10
        assert all(cid in s['MINI_RETRIEVER'].passage_by_id for cid in row['ranking'])
        if method in 'CD':
            assert len(row['selected_pages']) == 3
            assert all(s['BY_ID'][cid]['page_id'] in row['selected_pages'] for cid in row['ranking'])
        checked.append((domain, method))

# Closed-form runtime fixture: 1 second for each of 48 evaluation method-queries,
# then 25% reserve = 60 seconds. The remaining budget, not a reset 3h, governs.
run = s['new_run']('pilot')
run.update(status='complete', input_fit=True, warm_residency=True, elapsed_seconds=32,
    started_at='2026-10-30T10:00:00+00:00', finished_at='2026-10-30T10:00:32+00:00',
    rows=[{'query_id': q['query_id'], 'option': m, 'seconds': 1., 'status': 'success'}
          for q in s['PILOT_INPUTS'] for m in s['METHODS']])
assert s['pilot_runtime_lower_bound'](run['rows'], run)['seconds'] == 60
p = s['pilot_projection'](run, 61)
assert p['seconds'] == 60 and p['passed']
assert not s['pilot_projection'](run, 59)['passed']
assert s['labels_for']({'kind': 'evaluation', 'configuration': run['configuration']}) == s['MINI_EVALUATION_LABELS']
old = json.loads(Path('results/archive/v5.2-full-corpus/pilot.json').read_text())
assert s['labels_for'](old) == s['DESIGN_LABELS']
assert s['verify_request_evidence'](old)['reconstructed_requests'] == 32
assert s['CONFIG']['questions_per_request'] == 4

assert s['study_scope']({'configuration': {'study_profile': 'mini_2h'}})['evaluation_queries'] == 24
assert s['study_scope']({'configuration': {'study_profile': 'mini_3h'}})['evaluation_queries'] == 12
assert Counter(l['domain'] for l in s['MINI_EVALUATION_LABELS'].values()) == dict.fromkeys(s['DOMAINS'], 3)

# Expired workflow blocks a request before HTTP dispatch.
backend = s['OllamaBackend'](deadline=time.monotonic() - 1)
try:
    backend.timeout()
    raise AssertionError('Expired workflow accepted')
except s['StudyFailure'] as exc:
    assert exc.category == 'deadline'
finally:
    backend.close()

# A suspended/shifted wall clock cannot extend the remaining allowance.
backend = s['OllamaBackend'](deadline=time.monotonic() + 100)
backend.wall_deadline = time.time() - 1
try:
    backend.timeout()
    raise AssertionError('Expired wall deadline accepted despite unexpired active deadline')
except s['StudyFailure'] as exc:
    assert exc.category == 'deadline'
finally:
    backend.close()
assert s['run_clock_audit']({'started_at': '2026-10-30T10:00:00+00:00',
    'finished_at': '2026-10-30T10:05:00+00:00', 'elapsed_seconds': 10})['status'] == 'failed'
assert s['run_clock_audit']({'started_at': '2026-10-30T10:00:00+00:00',
    'finished_at': '2026-10-30T10:00:10+00:00', 'elapsed_seconds': 10})['status'] == 'passed'

# New gates fail as data, without any model/network calls in this fixture.
class FailingMetadata:
    records = []
    def metadata(self):
        raise s['StudyFailure']('deadline', 'Fixture deadline')
original_atomic = s['atomic_json']
s['atomic_json'] = lambda *args: None
try:
    gate = s['run_backend_gate'](FailingMetadata())
    assert gate['status'] == 'failed' and gate['failure']['category'] == 'deadline'
finally:
    s['atomic_json'] = original_atomic

preview = Path('artifacts/decisionrag/mini-profile-preview'); preview.mkdir(parents=True, exist_ok=True)
fig, _ = s['plot_scope']()
fig.savefig(preview / 'mini_scope.png', dpi=150); s['plt'].close(fig)
for stem, (fig, _) in s['publication_tables'](None, {}).items():
    if stem == 'table_01_experiment_contract':
        fig.savefig(preview / 'mini_contract.png', dpi=150)
    s['plt'].close(fig)

result = {'status': 'passed', 'authoring_source_sha256': s['EMBEDDED_DATA']['authoring_source_sha256'],
    'profile': 'mini_3h', 'pages': 32, 'passages': 329,
    'pilot_queries': 8, 'evaluation_queries': 12, 'evaluation_method_queries': 48,
    'controller_fixture_checks': len(checked), 'live_model_calls': 0,
    'checks': ['all four controllers across all domains retain corpus membership',
        'domain-weighted remaining-budget projection matches closed form',
        'old sixteen-question evidence reconstructs under old configuration',
        'expired active or wall deadline blocks HTTP dispatch', 'clock discrepancy blocks efficiency claims', 'failed backend gate returns recorded status'],
    'live_performance_validation': 'not run'}
Path('results/mini_profile_validation.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
