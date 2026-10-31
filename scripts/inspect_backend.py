import json
from pathlib import Path
import httpx

out = Path('data/runtime')
with httpx.Client(base_url='http://127.0.0.1:11434', timeout=180) as client:
    result = {'version': client.get('/api/version').json(), 'tags': client.get('/api/tags').json(),
              'ps': client.get('/api/ps').json(), 'models': {}}
    for model in ('nimble:9b-q8_0', 'qwen3.5:9b-q8_0'):
        response = client.post('/api/show', json={'model': model})
        response.raise_for_status()
        result['models'][model] = response.json()
    (out / 'backend_metadata.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
