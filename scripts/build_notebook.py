"""Authoring only: copy inline code and frozen metadata into one notebook.

The emitted notebook never imports this script or notebook_src. External data are downloaded at runtime; no corpus is encoded into the notebook. Preserve the frozen input manifests.
"""
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import nbformat
import pprint
import re

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))  # Authoring only; never emitted.
from audit_multidoc2dial import build_passages, queries, DOMAINS

def load_json(path):
    return json.loads((ROOT / path).read_text())

def split_manifest(manifest):
    inputs, labels = [], {}
    for q in manifest['queries']:
        inputs.append({k: q[k] for k in ('query_id', 'dialogue_id', 'user_turn_id', 'history')})
        labels[q['query_id']] = {k: q[k] for k in (
            'domain', 'gold_passage_ids', 'target_agent_turn_id', 'reference_answer')}
    return inputs, labels

def embedded_data():
    with zipfile.ZipFile(ROOT / 'data/raw/multidoc2dial.zip') as archive:
        docs = json.loads(archive.read('multidoc2dial/multidoc2dial_doc.json'))['doc_data']
        raw_splits = {split: json.loads(archive.read('multidoc2dial/multidoc2dial_dial_' + split + '.json'))['dial_data']
                      for split in ('train', 'validation')}
    corpus, span_map = build_passages(docs)
    implementation = b'\n'.join((ROOT / 'notebook_src' / name).read_bytes() for name in (
        '01_setup.py', '02_data.py', '03_retrieval.py', '04_experiments.py'))
    data = {'corpus': corpus, 'frozen_dialogue_ids': {}, 'captured': {},
            'implementation_sha256': hashlib.sha256(implementation).hexdigest()}
    sources = b'\n'.join(p.read_bytes() for p in sorted((ROOT / 'notebook_src').glob('*')) if p.is_file())
    data['authoring_source_sha256'] = hashlib.sha256(sources).hexdigest()
    for name in ('design', 'evaluation'):
        manifest = load_json('data/prepared/' + name + '_manifest.json')
        data[name + '_inputs'], data[name + '_labels'] = split_manifest(manifest)
        data['frozen_dialogue_ids'][name] = manifest['dialogue_ids']
    explored = load_json('docs/explored_validation_dialogues.json')
    data['explored_validation_dialogues'] = explored['dialogue_ids']
    selection_source = ast.parse((ROOT / 'notebook_src/02_data.py').read_text())
    selection_functions = [node for node in selection_source.body if isinstance(node, ast.FunctionDef)
        and node.name in ('make_mini_profile', 'make_three_hour_profile')]
    namespace = {'hashlib': hashlib, 'defaultdict': defaultdict, 'DOMAINS': DOMAINS}
    exec(compile(ast.Module(body=selection_functions, type_ignores=[]), '<inline mini selection>', 'exec'), namespace)
    eligible = {split: [q for domain in DOMAINS for dialogue in raw_splits[split][domain]
                       for q in queries(dialogue, domain, span_map)] for split in raw_splits}
    excluded = set(explored['dialogue_ids']) | {q['dialogue_id'] for q in data['evaluation_inputs']}
    mini = namespace['make_mini_profile'](corpus, eligible['train'], eligible['validation'], excluded)
    for name in ('design', 'evaluation'):
        mini[name + '_inputs'], mini[name + '_labels'] = split_manifest({'queries': mini.pop(name + '_queries')})
    data['mini_profile'] = mini
    data['mini_3h_profile'] = namespace['make_three_hour_profile'](mini)
    active = data['mini_3h_profile']
    print('Mini profile:', len(mini['page_ids']), 'pages;', sum(p['page_id'] in set(mini['page_ids']) for p in corpus),
          'passages;', len(mini['design_inputs']), 'pilot;', len(active['evaluation_inputs']), 'evaluation queries (mini_3h)')
    data['expected_hashes'] = {key: hashlib.sha256(json.dumps(data[key], ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest() for key in (
        'corpus', 'design_inputs', 'design_labels', 'evaluation_inputs', 'evaluation_labels', 'mini_profile', 'mini_3h_profile')}
    for key in ('corpus', 'design_inputs', 'design_labels', 'evaluation_inputs', 'evaluation_labels', 'mini_profile', 'mini_3h_profile', 'captured'):
        data.pop(key, None)
    data['tokenizer_url'] = 'https://huggingface.co/Qwen/Qwen3.5-9B/resolve/main/tokenizer.json'
    data['attribution'] = {
        'corpus': 'MultiDoc2Dial: https://doc2dial.github.io/multidoc2dial/',
        'tokenizer': 'Qwen3.5-9B, Apache 2.0: https://huggingface.co/Qwen/Qwen3.5-9B',
        'tokenizer_sha256': hashlib.sha256((ROOT / 'data/runtime/tokenizer.json').read_bytes()).hexdigest(),
    }

    return data

def story_sections():
    text = (ROOT / 'notebook_src/story.md').read_text()
    chunks = re.split(r'<!-- section: ([a-z0-9]+) -->', text)
    return {chunks[i]: chunks[i + 1].strip() for i in range(1, len(chunks), 2)}

def build():
    nb = nbformat.v4.new_notebook()
    nb.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                   'language_info': {'name': 'python', 'version': '3.9.6'}}
    execution = ROOT / 'results/notebook_execution.json'
    if execution.exists():
        nb.metadata['prior_live_execution'] = json.loads(execution.read_text())
    cells = [nbformat.v4.new_markdown_cell((ROOT / 'notebook_src/introduction.md').read_text())]
    sections = story_sections()
    def md(key):
        cells.append(nbformat.v4.new_markdown_cell(sections[key], metadata={'story_section': key}))
    def code(source, tags=(), hidden=False):
        metadata = {'tags': list(tags)}
        if hidden:
            metadata['jupyter'] = {'source_hidden': True}
        cells.append(nbformat.v4.new_code_cell(source, metadata=metadata))
    def source_parts(name):
        return (ROOT / 'notebook_src' / name).read_text().split('\n# %%\n')
    def add_part(name, part, hidden=False):
        tags = ['implementation', name[:2]]
        for marker, tag in (('# ORCHESTRATE:', 'orchestrate'), ('# ANALYSIS_RUN:', 'analysis-run')):
            if marker in part:
                tags.append(tag)
        code(part, tags, hidden)
    def visual(stem, prepared=None):
        args = (', prepared=' + prepared) if prepared else ', analysis=ANALYSIS, runs=RUNS'
        code('show_book_visual(' + repr(stem) + args + ')', ['figures-run', 'book-visual', stem])
    md('setup')
    add_part('01_setup.py', source_parts('01_setup.py')[0])
    # Plot functions are defined before the narrative uses them; nothing is run here.
    for part in source_parts('06_figures.py'):
        if '# FIGURES_RUN:' not in part:
            add_part('06_figures.py', part, hidden=True)
    add_part('07_book.py', (ROOT / 'notebook_src/07_book.py').read_text(), hidden=True)
    md('data')
    code('# Frozen selection IDs and checksums only; no corpus body.\nEMBEDDED_DATA = ' +
         pprint.pformat(embedded_data(), width=110, sort_dicts=False), ['embedded-data'], True)
    for part in source_parts('02_data.py'):
        add_part('02_data.py', part)
    visual('graph_02_corpus_and_evaluation_scope', 'plot_scope(book_source_run())')
    md('bm25')
    md('judge')
    retrieval = source_parts('03_retrieval.py')
    for index in (0, 1):
        add_part('03_retrieval.py', retrieval[index])
    md('navigation')
    for index in (3, 4):
        add_part('03_retrieval.py', retrieval[index])
    md('control')
    add_part('03_retrieval.py', retrieval[2])
    visual('graph_01_four_retrieval_paths', 'plot_paths()')
    md('protocol')
    visual('table_01_experiment_contract', 'prepare_book_contract()')
    for part in source_parts('04_experiments.py'):
        add_part('04_experiments.py', part)
    md('metrics')
    for part in source_parts('05_analysis.py'):
        add_part('05_analysis.py', part)
    code("# FIGURES_RUN: save both the detailed evidence gallery and the requested graphs folder.\n"
         "ASSET_DIR, ASSET_MANIFEST, RECEIPT = export_publication(RUNS, ANALYSIS, show=False)\n"
         "BOOK_MANIFEST = export_book_assets(ASSET_DIR, ASSET_MANIFEST, ANALYSIS, RUNS)\n"
         "display(Markdown(f'**Exports verified:** {RECEIPT[\"export_validation\"][\"generated_images\"]} figures/tables; numbered copies in `./graphs`.'))",
         ['figures-run', 'book-export'])
    visual('graph_09_laptop_feasibility')
    md('results')
    for stem in ('graph_03_retrieval_accuracy', 'table_02_four_method_results', 'graph_04_quality_latency_tradeoff'):
        visual(stem)
    md('value')
    for stem in ('graph_05_primary_decision_model_comparison', 'table_03_primary_verdict'):
        visual(stem)
    md('failures')
    for stem in ('graph_06_where_retrieval_fails', 'table_04_domain_results', 'graph_07_query_latency_distribution'):
        visual(stem)
    md('closing')
    code('show_closing_reading(ANALYSIS)', ['figures-run', 'closing-reading'])
    nb.cells = cells
    target = ROOT / 'decisionrag_ollama_demo.ipynb'
    nbformat.write(nb, target)
    print('Built', target, 'cells=', len(cells), 'bytes=', target.stat().st_size)

if __name__ == '__main__':
    build()
