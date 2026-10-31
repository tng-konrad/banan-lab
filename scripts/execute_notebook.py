"""Execute the actual notebook in a fresh local kernel; stream progress to stdout.

No experiment implementation lives here. The original launch configuration is
restored after execution; metadata records the mode used for the outputs.
"""
import argparse
import ast
from datetime import datetime, timezone
import json
import re
from pathlib import Path
import tempfile
import time
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parents[1]

class StreamingClient(NotebookClient):
    def process_message(self, msg, cell, cell_index):
        if msg['msg_type'] == 'stream':
            print(msg['content']['text'], end='', flush=True)
        return super().process_message(msg, cell, cell_index)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('evaluate', 'pilot', 'replay'), required=True)
    args = p.parse_args()
    target = ROOT / 'decisionrag_ollama_demo.ipynb'
    nb = nbformat.read(target, as_version=4)
    setup = next(c for c in nb.cells if c.cell_type == 'code' and 'configuration' in c.metadata.get('tags', []))
    original_setup = setup.source
    setup.source = re.sub(r"^    mode = '(?:evaluate|replay)'", '    mode = ' + repr(args.mode), setup.source, count=1, flags=re.M)
    work = Path(tempfile.mkdtemp(prefix='decisionrag-live-kernel-'))
    spec = work / 'kernels/decisionrag-live'; spec.mkdir(parents=True)
    (spec / 'kernel.json').write_text(json.dumps({'argv': [str(ROOT / '.venv/bin/python'), '-m',
        'ipykernel_launcher', '-f', '{connection_file}'], 'display_name': 'DecisionRAG live', 'language': 'python',
        'env': {'MPLCONFIGDIR': str(work / 'mpl'), 'IPYTHONDIR': str(work / 'ipython'),
                'TOKENIZERS_PARALLELISM': 'false'}}))
    km = KernelManager(kernel_name='decisionrag-live',
        kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(work / 'kernels')]))
    config_class = next(node for node in ast.parse(setup.source).body if isinstance(node, ast.ClassDef) and node.name == 'CFG')
    budget_node = next(node for node in config_class.body
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
            t.id == 'run_budget_hours' for t in node.targets))
    budget_seconds = float(ast.literal_eval(budget_node.value)) * 3600
    client = StreamingClient(nb, km=km, timeout=int(budget_seconds + 600),
        resources={'metadata': {'path': str(ROOT)}})
    started = time.monotonic()
    wall_started = datetime.now(timezone.utc)
    status = 'failed'
    try:
        client.execute()
        status = 'passed'
    finally:
        km.shutdown_kernel(now=True)
        setup.source = original_setup
        nb.metadata['last_execution'] = {'mode': args.mode, 'study_profile': 'mini_3h',
            'status': status, 'elapsed_seconds': time.monotonic() - started,
            'started_at': wall_started.isoformat(),
            'finished_at': datetime.now(timezone.utc).isoformat(),
            'wall_elapsed_seconds': (datetime.now(timezone.utc) - wall_started).total_seconds(),
            'workflow_budget_seconds': budget_seconds,
            'note': 'Original launch configuration restored; outputs reflect the recorded execution mode.'}
        nbformat.write(nb, target)
        (ROOT / 'results').mkdir(exist_ok=True)
        (ROOT / 'results/notebook_execution.json').write_text(json.dumps(dict(nb.metadata['last_execution']), indent=2))
    print('\nNotebook execution:', json.dumps(dict(nb.metadata['last_execution'])), flush=True)

if __name__ == '__main__':
    main()
