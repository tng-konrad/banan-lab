"""Execute the notebook with declared external cache/results in an isolated directory.

The helper is a development acceptance check, never a notebook dependency.
It also independently verifies exported recall from captured rankings/labels.
"""
import argparse
import ast
import base64
import gzip
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import tempfile
import time

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / '.venv/bin/python'

BLOCK_NETWORK = '''\
import builtins, io, os, socket, sys
_original_open = builtins.open
_original_io_open = io.open
_repo_root = REPO_ROOT
def _no_network(*args, **kwargs):
    raise RuntimeError("Network access blocked during standalone replay acceptance check")
socket.create_connection = _no_network
socket.socket.connect = _no_network
socket.socket.connect_ex = _no_network
def _guard_path(file):
    if isinstance(file, (str, bytes, os.PathLike)):
        name = os.path.realpath(os.fsdecode(file))
        if name.startswith(_repo_root + os.sep) and not name.startswith(_repo_root + '/.venv/'):
            raise RuntimeError("Repository file access blocked during standalone replay: " + name)
def _guarded_open(file, *args, **kwargs):
    _guard_path(file)
    return _original_open(file, *args, **kwargs)
def _guarded_io_open(file, *args, **kwargs):
    _guard_path(file)
    return _original_io_open(file, *args, **kwargs)
builtins.open = _guarded_open
io.open = _guarded_io_open
sys.path = [p for p in sys.path if os.path.realpath(p) != _repo_root]
'''

def embedded(nb):
    cell = next(c for c in nb.cells if 'embedded-data' in c.metadata.get('tags', []))
    expression = ast.parse(cell.source).body[0].value
    return ast.literal_eval(expression)

def structural_checks(nb):
    nbformat.validate(nb)
    code = '\n'.join(c.source for c in nb.cells if c.cell_type == 'code' and
                     'embedded-data' not in c.metadata.get('tags', []))
    assert re.search(r"^MODE = '(?:evaluate|replay)'", code, re.M)
    metadata = embedded(nb)
    assert not ({'corpus', 'mini_profile', 'captured', 'tokenizer_json'} & set(metadata))
    assert 'expected_hashes' in metadata
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not node.level and not (node.module or '').startswith(('scripts', 'notebook_src'))
        if isinstance(node, ast.Import):
            assert not any(a.name.startswith(('scripts', 'notebook_src')) for a in node.names)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in ('exec', 'eval', 'compile', '__import__')
    return {'status': 'passed', 'cells': len(nb.cells), 'project_imports': 0,
            'encoded_executable_source': False}

def independent_metrics(data, folder, work):
    captured = {name: json.loads((work / 'results' / (name + '.json')).read_text())
        for name in ('gate', 'demo', 'pilot', 'evaluation') if (work / 'results' / (name + '.json')).exists()}
    run = captured.get('evaluation') or captured.get('pilot') or captured.get('demo')
    if not run or not run.get('rows'):
        return {'status': 'not_applicable'}
    profile = run.get('configuration', {}).get('study_profile')
    mini = profile in ('mini_2h', 'mini_3h') and run['kind'] != 'demo'
    label_name = ((profile + '_') if mini and run['kind'] == 'evaluation' else 'mini_' if mini else '') + ('evaluation' if run['kind'] == 'evaluation' else 'design')
    label_rows = [json.loads(line) for line in (work / 'artifacts/decisionrag/frozen_inputs' /
        (label_name + '_labels.jsonl')).read_text().splitlines()]
    labels = {row['query_id']: row for row in label_rows}
    sidecar = json.loads((folder / 'table_02_four_method_results.json').read_text())
    checked = []
    for method in 'ABCD':
        rows = [r for r in run['rows'] if r['option'] == method]
        published = next(r for r in sidecar['data']['methods'] if r['option'] == method)
        assert published['attempted'] == len(rows)
        for k in (1, 5, 10):
            hits = sum(r['status'] == 'success' and
                labels[r['query_id']]['gold_passage_ids'][0] in r['ranking'][:k] for r in rows)
            expected = hits / len(rows) if rows else None
            assert published['r' + str(k)] == expected
        assert published['failed'] == sum(r['status'] != 'success' for r in rows)
        checked.append(method)
    primary = json.loads((folder / 'table_03_primary_verdict.json').read_text())['data']
    if run['kind'] != 'evaluation' or any(r['status'] != 'success' for r in run['rows'] if r['option'] in 'CD'):
        assert primary['verdict'] != 'local efficiency win for C'
    return {'status': 'passed', 'methods': checked, 'tolerance': 1e-12,
            'checks': ['strict ID recall from raw rankings', 'denominators and failures', 'no unsupported primary win']}

def publication_contract(nb, work):
    """Check the user-visible contract, including every displayed table/image."""
    book = json.loads((work / 'graphs/manifest.json').read_text())
    generated = [a for a in book['assets'] if a['status'] == 'generated']
    plan = (ROOT / 'post-plan.md').read_text()
    notebook_text = '\n'.join(c.source for c in nb.cells if c.cell_type == 'markdown')
    notebook_headings = re.findall(r'^## .+$', notebook_text, re.M)
    post_headings = re.findall(r'^## .+$', plan, re.M)
    assert notebook_headings == post_headings, (notebook_headings, post_headings)
    assert re.findall(r'^### .+$', notebook_text, re.M) == re.findall(r'^### .+$', plan, re.M)
    visual_cells = [c for c in nb.cells if 'book-visual' in c.metadata.get('tags', [])]
    assert len(visual_cells) == len(book['assets'])
    displayed = []
    for cell in nb.cells:
        for output in cell.get('outputs', []):
            data = output.get('data', {})
            assert '<table' not in data.get('text/html', '').lower(), 'An unnumbered table was displayed'
            if 'image/png' in data:
                displayed.append(hashlib.sha256(base64.b64decode(data['image/png'])).hexdigest())
    expected = [a['sha256'] for a in generated]
    assert displayed == expected, {'inline_count': len(displayed), 'exported_count': len(expected),
        'mismatches': [(i + 1, actual, wanted) for i, (actual, wanted) in
            enumerate(zip(displayed, expected)) if actual != wanted]}
    for index, (cell, asset) in enumerate(zip(visual_cells, book['assets']), 1):
        assert asset['number'] == index
        assert asset['stem'] in cell.metadata['tags']
        prefix = 'post' if asset['kind'] == 'table' else 'graph'
        assert asset['filename'] == f'{prefix}_{index}_0111.png'
        assert f'(graphs/{asset["filename"]})' in plan
        if asset['status'] == 'generated':
            path = work / 'graphs' / asset['path']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == asset['sha256']
            assert asset['caption'] and asset['reading'] and asset['alt_text']
    audit_path = work / 'graphs' / book['score_resolution_audit']['path']
    assert hashlib.sha256(audit_path.read_bytes()).hexdigest() == book['score_resolution_audit']['sha256']
    audit = json.loads(audit_path.read_text())
    available = [json.loads(p.read_text()) for p in (work / 'results').glob('*.json')]
    run = next(r for r in available if r.get('run_id') == audit['source_run_id'])
    assert audit['source_run_id'] == run['run_id']
    for method in ('C', 'D'):
        groups = [r for r in run['requests'] if r['option'] == method and 'scores' in r]
        vectors = [list(r['scores'].values()) for r in groups if len(r['scores']) > 1]
        recorded = audit['methods'][method]
        assert recorded['scored_request_groups'] == len(groups)
        assert recorded['http_attempts_for_scored_groups'] == sum(len(g['attempts']) for g in groups)
        assert recorded['singleton_groups_excluded'] == len(groups) - len(vectors)
        assert recorded['multi_question_groups'] == len(vectors)
        assert recorded['all_equal_multi_question_groups'] == sum(min(v) == max(v) for v in vectors)
        assert recorded['tied_multi_question_groups'] == sum(len(set(v)) < len(v) for v in vectors)
    paired_alt = next((a['alt_text'] for a in generated if a['stem'] == 'graph_05_primary_decision_model_comparison'), None)
    if paired_alt and run['configuration']['study_profile'].startswith('mini'):
        assert 'zero' in paired_alt and 'minus-two' not in paired_alt
    latency_alt = next((a['alt_text'] for a in generated if a['stem'] == 'graph_07_query_latency_distribution'), None)
    if latency_alt and all(any(r['option'] == method for r in run['rows']) for method in 'ABCD'):
        assert 'Other methods are unattempted' not in latency_alt
    return {'status': 'passed', 'visuals': len(generated), 'tables': sum(a['kind'] == 'table' for a in generated),
        'matching_section_headings': len(notebook_headings), 'numbering': book['numbering'],
        'checks': ['notebook and post headings match in order', 'each post placeholder resolves',
            'one numbering sequence in display order', 'inline PNG bytes match exported files',
            'all displayed tables are numbered exports', 'each visual has a caption and reading',
            'score-resolution diagnostic excludes singletons and matches raw requests',
            'alt text matches mini reference and actual method completion']}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write-executed', action='store_true')
    args = parser.parse_args()
    target = ROOT / 'decisionrag_ollama_demo.ipynb'
    nb = nbformat.read(target, as_version=4)
    structure = structural_checks(nb)
    data = embedded(nb)
    setup = next(c for c in nb.cells if c.cell_type == 'code' and re.search(r"^MODE = '(?:evaluate|replay)'", c.source, re.M))
    original_setup = setup.source
    shipped_mode = re.search(r"^MODE = '(evaluate|replay)'", setup.source, re.M).group(1)
    setup.source = re.sub(r"^MODE = '(?:evaluate|replay)'", "MODE = 'replay'", setup.source, count=1, flags=re.M)
    work = Path(tempfile.mkdtemp(prefix='decisionrag-standalone-'))
    shutil.copy2(target, work / target.name)
    copied = [target.name]
    # External downloads and result traces are declared data dependencies.
    for relative in ('data/raw/multidoc2dial.zip', 'data/runtime/tokenizer.json'):
        dest = work / relative; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, dest); copied.append(relative)
    for name in ('gate', 'demo', 'pilot', 'evaluation'):
        src = ROOT / 'results' / (name + '.json')
        if src.exists():
            dest = work / 'results' / src.name; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest); copied.append(str(dest.relative_to(work)))
    kernels = work / 'kernels'; spec = kernels / 'decisionrag-qa'; spec.mkdir(parents=True)
    (spec / 'kernel.json').write_text(json.dumps({'argv': [str(PYTHON), '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'DecisionRAG isolated acceptance kernel', 'language': 'python',
        'env': {'MPLCONFIGDIR': str(work / 'mpl-cache'), 'IPYTHONDIR': str(work / 'ipython'),
                'PYTHONPYCACHEPREFIX': str(work / 'pycache'), 'TOKENIZERS_PARALLELISM': 'false'}}))
    nb.cells.insert(0, nbformat.v4.new_code_cell(BLOCK_NETWORK.replace('REPO_ROOT', repr(str(ROOT)))))
    manager = KernelManager(kernel_name='decisionrag-qa',
        kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
    client = NotebookClient(nb, km=manager, timeout=240, resources={'metadata': {'path': str(work)}})
    start = time.perf_counter()
    try:
        client.execute()
    finally:
        manager.shutdown_kernel(now=True)
    elapsed = time.perf_counter() - start
    assert not any(output.output_type == 'error' for cell in nb.cells if cell.cell_type == 'code'
                   for output in cell.get('outputs', []))
    manifests = list(work.glob('artifacts/decisionrag/*/post_assets/manifest.json'))
    assert len(manifests) == 1
    folder = manifests[0].parent
    manifest = json.loads(manifests[0].read_text())
    numerical = independent_metrics(data, folder, work)
    nbformat.write(nb, work / 'completed-before-publication-check.ipynb')
    publication = publication_contract(nb, work)
    for item in manifest['assets']:
        if item['status'] != 'generated':
            continue
        path = folder / item['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
        with Image.open(path) as image:
            assert image.size == (item['width_pixels'], item['height_pixels'])
            assert image.getbbox() is not None
    verification_path = ROOT / 'results/delivery_verification.json'
    verification = json.loads(verification_path.read_text()) if verification_path.exists() else {}
    relevant_visual = (verification.get('authoring_source_sha256') == data['authoring_source_sha256']
        and verification.get('source_run_id') == manifest['run_id'])
    sealed_visual = verification.get('visual_QA', {}).get('images', []) if relevant_visual else []
    if sealed_visual:
        assert {i['filename']: i['sha256'] for i in sealed_visual} == {
            i['path']: i['sha256'] for i in manifest['assets'] if i['status'] == 'generated'}
    report = {'status': 'passed', 'network': 'blocked in kernel', 'repository_file_reads': 'blocked except installed .venv packages',
        'copied_input_files': copied, 'corpus_embedded': False, 'shipped_mode': shipped_mode, 'validation_mode': 'replay', 'live_model_calls': 0, 'fresh_kernel': str(PYTHON), 'execution_seconds': elapsed,
        'structural_checks': structure, 'independent_numerical_check': numerical,
        'generated_images': sum(a['status'] == 'generated' for a in manifest['assets']),
        'run_id': manifest['run_id'], 'experiment_status': manifest['experiment_status'], 'publication_status': manifest['publication_status'],
        'isolated_directory': str(work), 'executed_notebook_errors': 0,
        'numerical_receipt': manifest.get('numerical_receipt'), 'publication_contract': publication}
    report['sealed_visual_hash_check'] = 'passed' if sealed_visual else 'not yet sealed'
    nb.cells.pop(0)
    nbformat.write(nb, work / target.name)
    if args.write_executed:
        setup.source = original_setup
        nbformat.write(nb, target)
        relative = folder.relative_to(work)
        destination = ROOT / relative
        shutil.copytree(folder, destination, dirs_exist_ok=True)
        shutil.copytree(work / 'graphs', ROOT / 'graphs', dirs_exist_ok=True)
        (destination / 'standalone_validation.json').write_text(json.dumps(report, indent=2))
    (ROOT / 'results/standalone_validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
