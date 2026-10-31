"""Verify the notebook's optional archive rebuild and actual-demo bootstrap."""
import json
from pathlib import Path
from run_live import load_runtime

s = load_runtime()
hashes = s['rebuild_from_public_archive'](Path('data/raw/multidoc2dial.zip').read_bytes())
demo = json.loads(Path('results/archive/v5.2-full-corpus/demo.json').read_text())
analysis = s['analyze'](demo)
assert analysis['bootstrap']['status'] == 'available'
assert analysis['bootstrap']['resamples'] == 10000
assert analysis['primary']['verdict'] == 'inconclusive'
assert analysis['bootstrap']['dialogues'] == 1
for row in analysis['methods']:
    for k in (1, 5, 10):
        assert analysis['bootstrap']['quality_intervals'][row['option']]['r' + str(k)] == [row['r' + str(k)]] * 2
result = {'status': 'passed', 'archive_rebuild_hashes': hashes,
    'bootstrap_check': '10,000 whole-dialogue draws on the actual one-dialogue training example produce the expected degenerate intervals; no inferential claim',
    'demo_primary_verdict': analysis['primary']['verdict']}
Path('results/rebuild_and_bootstrap_validation.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
