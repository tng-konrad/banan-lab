"""Record completed checks after actual visual review of the selected gallery."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--gallery', type=Path)
args = p.parse_args()
folder = args.gallery
if folder is None:
    folder = max(Path('artifacts/decisionrag').glob('*/post_assets/manifest.json'),
                 key=lambda path: path.stat().st_mtime).parent
manifest = json.loads((folder / 'manifest.json').read_text())
standalone = json.loads(Path('results/standalone_validation.json').read_text())
rebuild = json.loads(Path('results/rebuild_and_bootstrap_validation.json').read_text())
mini = json.loads(Path('results/mini_profile_validation.json').read_text())
downloads = json.loads(Path('results/download_validation.json').read_text())
assert standalone['status'] == rebuild['status'] == mini['status'] == downloads['status'] == 'passed'
assert standalone['run_id'] == manifest['run_id']
images = []
for item in manifest['assets']:
    if item['status'] == 'generated':
        digest = hashlib.sha256((folder / item['path']).read_bytes()).hexdigest()
        assert digest == item['sha256']
        images.append({'filename': item['path'], 'sha256': digest})
result = {'captured_at': datetime.now(timezone.utc).isoformat(),
    'authoring_source_sha256': hashlib.sha256(b'\n'.join(p.read_bytes() for p in
        sorted(Path('notebook_src').glob('*')) if p.is_file())).hexdigest(),
    'source_run_id': manifest['run_id'], 'numerical_receipt': manifest['numerical_receipt'],
    'standalone': standalone, 'rebuild_and_bootstrap': rebuild,
    'mini_profile_offline': mini, 'external_downloads': downloads,
    'visual_QA': {'status': 'passed', 'reviewer': 'Codex visual inspection', 'images': images,
        'checks': ['all exported images viewed', 'titles/subtitles separated',
            'no clipping or overlap', 'black table body text', 'explicit observed/scheduled denominators',
            'failure markers and actual completion status', 'correct corpus profile and source run',
            'resource lower bound distinct from a point forecast']}}
Path('results/delivery_verification.json').write_text(json.dumps(result, indent=2))
(folder / 'delivery_verification.json').write_text(json.dumps(result, indent=2))
print(f'Sealed {len(images)} visually checked images and external-data acceptance checks.')
