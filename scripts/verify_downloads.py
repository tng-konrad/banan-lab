"""Test the notebook's real cold downloads, cache reuse and checksum rejection."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import httpx

ROOT = Path(__file__).resolve().parents[1]
nb = json.loads((ROOT / 'decisionrag_ollama_demo.ipynb').read_text())
meta_cell = next(c for c in nb['cells'] if 'embedded-data' in c.get('metadata', {}).get('tags', []))
meta = ast.literal_eval(ast.parse(''.join(meta_cell['source'])).body[0].value)
# Development only: use the exact function copied into the notebook, no new downloader.
source = next(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code' and
              'def fetch_cached(' in ''.join(c['source']))
function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'fetch_cached')
scope = {'Path': Path, 'hashlib': hashlib, 'httpx': httpx}
exec(compile(ast.Module(body=[function], type_ignores=[]), '<notebook download>', 'exec'), scope)
fetch = scope['fetch_cached']
work = Path(tempfile.mkdtemp(prefix='decisionrag-download-'))
files = [('corpus', 'https://doc2dial.github.io/multidoc2dial/file/multidoc2dial.zip',
          'f0c034c249663d7b3cb08b19cf2cc2c3d101372485be982621d4711931a1ce00', 'multidoc2dial.zip'),
         ('tokenizer', meta['tokenizer_url'], meta['attribution']['tokenizer_sha256'], 'tokenizer.json')]
records = []
for kind, url, digest, filename in files:
    path = fetch(url, work / filename, digest)
    before = path.stat().st_mtime_ns
    fetch(url, path, digest)
    assert path.stat().st_mtime_ns == before
    records.append({'kind': kind, 'url': url, 'sha256': digest, 'bytes': path.stat().st_size,
        'cold_download': 'passed', 'cache_reuse': 'passed'})
bad = work / 'bad.zip'; bad.write_bytes(b'invalid corpus')
try:
    fetch(files[0][1], bad, files[0][2])
    raise AssertionError('Bad cache accepted')
except ValueError:
    pass
report = {'status': 'passed', 'corpus_embedded': False, 'files': records,
    'checksum_rejection': 'passed', 'isolated_directory': str(work)}
(ROOT / 'results/download_validation.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
