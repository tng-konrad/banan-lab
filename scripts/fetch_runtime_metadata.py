"""Development-only: vendor tokenizer and inspect the installed server's source."""
import hashlib
import json
from pathlib import Path
import httpx

out = Path('data/runtime')
out.mkdir(parents=True, exist_ok=True)
with httpx.Client(timeout=120, follow_redirects=True) as client:
    for filename in ('tokenizer.json', 'tokenizer_config.json'):
        url = 'https://huggingface.co/Qwen/Qwen3.5-9B/resolve/main/' + filename
        response = client.get(url)
        response.raise_for_status()
        (out / filename).write_bytes(response.content)
        print(filename, len(response.content), hashlib.sha256(response.content).hexdigest(), flush=True)
    url = 'https://api.github.com/repos/ollama/ollama/git/trees/v0.40.13?recursive=1'
    response = client.get(url)
    response.raise_for_status()
    tree = response.json()
    paths = [x['path'] for x in tree['tree'] if x['type'] == 'blob' and (
        x['path'].startswith('decision/') or 'systemone' in x['path'].lower())]
    (out / 'ollama_source_paths.json').write_text(json.dumps(paths, indent=2))
    print('Decision source files:', paths, flush=True)
    for path in paths:
        if path.endswith('.go'):
            response = client.get('https://raw.githubusercontent.com/ollama/ollama/v0.40.13/' + path)
            response.raise_for_status()
            target = out / 'source' / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(response.content)
