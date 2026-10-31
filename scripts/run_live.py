"""Development runner. All experiment logic is read from notebook code cells.

No implementation here is required by the notebook. This command exposes its
live functions while keeping its shipped configuration safely in offline replay.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

def load_runtime():
    nb = json.loads((ROOT / 'decisionrag_ollama_demo.ipynb').read_text())
    scope = {'__name__': '__notebook__'}
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        tags = cell.get('metadata', {}).get('tags', [])
        if 'orchestrate' in tags or 'analysis-run' in tags or 'figures-run' in tags:
            continue
        source = ''.join(cell['source'])
        exec(compile(source, 'decisionrag_ollama_demo.ipynb', 'exec'), scope)
    return scope

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('probe', 'gate', 'demo', 'pilot', 'evaluate'))
    args = parser.parse_args()
    s = load_runtime()
    if args.mode == 'probe':
        b = s['OllamaBackend']()
        batch = s['shared_batch'](s['DEMO_INPUT'], s['DEMO_RETRIEVER'].passages[:2], 'passage')
        print('admission', s['common_admission'](batch), flush=True)
        for adapter in ('nimble', 'qwen'):
            try:
                print(adapter, b.call(adapter, batch), flush=True)
            except Exception as exc:
                print(type(exc).__name__, str(exc), flush=True)
        print(json.dumps(b.records, indent=2), flush=True)
        Path('results').mkdir(exist_ok=True)
        Path('results/backend_probe.json').write_text(json.dumps(b.records, indent=2))
        print('residency', b.residency(), flush=True)
        b.close()
    else:
        result = s['execute_mode'](args.mode)
        print(json.dumps({k: result.get(k) for k in ('run_id', 'kind', 'status', 'elapsed_seconds',
            'failure', 'projection')}, indent=2), flush=True)

if __name__ == '__main__':
    main()
