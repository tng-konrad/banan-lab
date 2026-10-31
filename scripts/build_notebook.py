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
    namespace = {'hashlib': hashlib, 'defaultdict': defaultdict, 'CFG': type('CFG', (), {'domains': DOMAINS})}
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
                   'language_info': {'name': 'python', 'version': sys.version.split()[0]}}
    execution = ROOT / 'results/notebook_execution.json'
    if execution.exists():
        nb.metadata['prior_live_execution'] = json.loads(execution.read_text())
    cells = [nbformat.v4.new_markdown_cell((ROOT / 'notebook_src/introduction.md').read_text())]
    sections = story_sections()

    def markdown(text, **metadata):
        cells.append(nbformat.v4.new_markdown_cell(text, metadata=metadata))

    def code(source, explanation, tags=(), hidden=False):
        metadata = {'tags': list(tags)}
        if hidden:
            metadata['jupyter'] = {'source_hidden': True}
        cells.append(nbformat.v4.new_code_cell(source.strip(), metadata=metadata))
        markdown(explanation, explains_code=True)

    def section(key):
        title, body = sections[key].split('\n', 1)
        # Setup/Utils are notebook-only; all narrative titles match the article.
        level = '##' if key == 'metrics' else '#'
        markdown(level + ' ' + title.lstrip('# ').strip(), story_section=key)
        return body.strip()

    sources = {p.name: p.read_text() for p in sorted((ROOT / 'notebook_src').glob('0*.py'))}
    trees = {name: ast.parse(source) for name, source in sources.items()}
    definitions = {}
    runtime = {}
    for name, tree in trees.items():
        lines = sources[name].splitlines()
        statements = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if node.name == 'CFG':
                    continue
                first = min([node.lineno] + [d.lineno for d in node.decorator_list])
                definitions[node.name] = '\n'.join(lines[first-1:node.end_lineno])
            elif name != '01_setup.py':
                statements.append(ast.get_source_segment(sources[name], node))
        runtime[name] = '\n'.join(statements)

    markdown('# Setup')
    setup = sources['01_setup.py'].split('# %%')
    code(setup[0],
        'The first group imports Python’s **standard library**: collections, paths, JSON, hashes, clocks and archive tools. '
        'These ship with Python. The second group imports packages installed into the notebook kernel. '
        '**NumPy** (`np`) supplies arrays and resampling; **pandas** (`pd`) organizes results; **Matplotlib** (`plt`) draws the figures. '
        '`BM25Okapi` builds the lexical baseline, `Tokenizer` checks complete request sizes, and `httpx` talks to Ollama or downloads pinned public files. '
        'IPython displays the same PNG bytes that we export for the post. Importing these tools does not call a model.', ['setup', 'imports'])
    code(setup[1],
        '**CFG** is the single place to edit settings. Paths select the external data cache, captured JSON results and export folders. '
        '`mode="replay"` rebuilds the saved experiment; change the class field to `"evaluate"` and restart/Run All for fresh local inference. '
        '`study_profile="mini_3h"` fixes 32 pages and 12 evaluation queries; `run_budget_hours=3.0` sets the whole live-workflow allowance. '
        '`CFG.experiment` groups model tags, BM25 parameters, navigation limits, context guards and bootstrap settings. '
        'Replay temporarily reads each capture’s original experiment settings so a formatting change cannot relabel old evidence.\n\n'
        '`plt.style.use("seaborn-v0_8")` selects Matplotlib’s bundled style; the separate seaborn package is unnecessary. '
        '`figure.figsize` reads `CFG.img_dim1` and `CFG.img_dim2` in inches (9.5 × 6.5 here), and every plotting helper inherits those defaults. '
        '`export_dpi=200` controls PNG resolution. Seeds of 42 stabilize new random operations; recorded bootstrap draws retain their separate saved seed. '
        'Seeds do not guarantee identical model responses across machines or executions. The assertions reject invalid modes and budgets early.', ['setup', 'configuration'])

    markdown('# Utils\n\nAll reusable functions and helper classes are collected here. '
        'Definitions are grouped by responsibility and do not start inference. The narrative resumes after these utilities; '
        'its section titles match `post.md`.')

    groups = [
      ('Boilerplate and evidence files', [
        ('stable_json sha256 utc_now software_versions recorded_configuration atomic_json',
         'These helpers serialize JSON consistently, hash data, record UTC timestamps and package versions, and write files atomically. '
         '`recorded_configuration` temporarily restores a capture’s experiment settings and restores CFG afterward. This preserves request hashes and provenance when replaying older runs.')]),
      ('Corpus preparation and query boundaries', [
        ('fetch_cached', 'The downloader streams to a temporary file, verifies SHA256 and only then promotes it into the cache. '
         'A valid existing file is reused; a corrupt cached or downloaded file raises an error. No corpus body is stored inside the notebook.'),
        ('normalize_text tokenize make_mini_profile make_three_hour_profile is_mini_profile mini_profile_data',
         'Text normalization supports duplicate-text checks; Unicode word tokens are the frozen BM25 preprocessing. '
         'The selection helpers choose pages before eligible dialogue queries, using fixed hash order. The three-hour profile keeps three validation dialogues per domain without consulting model outcomes.'),
        ('prepare_public_archive rebuild_from_public_archive study_scope validate_dataset model_history select_pilot_queries training_demo_corpus',
         'Archive preparation reconstructs complete structural passages and verifies frozen checksums. Dataset checks keep training and evaluation dialogues separate. '
         '`model_history` allowlists recent role/text fields, preventing reference answers or gold IDs from entering requests. `study_scope` describes a capture’s actual cohort.')]),
      ('Judgment requests and local serving', [
        ('StudyFailure LocalSizeRejection corpus_candidates shared_batch select_questions question_batches score_batch',
         'Candidates become either complete page metadata or complete passages. Shared states use stable aliases; question batches ask at most four judgments while preserving the full candidate state. '
         'Typed failures distinguish malformed responses, deadlines and context rejection. These definitions make no HTTP requests.'),
        ('nimble_rendered_prompts qwen_rendered_prompt score_schema payload_for token_count common_admission parse_scores',
         'Adapters render the same judgment problem in Nimble’s native interface or Qwen’s constrained numeric JSON. '
         'The common admission guard counts both rendered prompts and checks request bytes. Parsing rejects missing aliases, extra fields and invalid scores rather than silently repairing them.'),
        ('OllamaBackend', 'The backend records requests, attempts, model residency, timing and response evidence. '
         'Both active-clock and wall-clock deadlines constrain live requests; retry limits stay explicit. The class is defined here but instantiated only by a live run or an offline fixture.')]),
      ('Ranking and navigation', [
        ('deterministic_batches sorted_scores replace_prefix rank_pool',
         'Batching keeps whole candidates and splits oversized states deterministically. Scores are sorted with a fixed candidate-ID tie break. '
         'A bounded leading prefix is rescored together; the remaining tail keeps its coarse order. This approximate cross-batch ranking is part of the experiment’s limits.'),
        ('Retriever', 'One controller implements A–D: lexical retrieval, lexical discovery plus Nimble judging, and hierarchical navigation with either model. '
         'For C/D it ranks page metadata, retains three pages and ranks their passages. A pruned gold page cannot be recovered later. Sharing the controller keeps these rules identical across C and D.')]),
      ('Experiment orchestration and validation', [
        ('offline_checks run_clock_audit validity_gates both_resident export_frozen_inputs new_run',
         'Offline fixtures check mechanics without model calls. Clock and residency gates determine whether a run supports timing comparisons. '
         'New-run receipts preserve configuration and input hashes; frozen inputs and labels are exported separately so the model never receives the scoring labels.'),
        ('run_backend_gate freeze_schedule execute_queries',
         'The live gate checks serving behavior before the seeded evaluation schedule. Each query records its terminal status, selected pages, rankings and elapsed time. '
         'A failure counts as a retrieval miss; unattempted work is kept distinct. Scheduling does not reset the workflow deadline.'),
        ('pilot_runtime_lower_bound pilot_projection adapter_check execute_mode',
         'Training-only pilot timings project evaluation work with a 25% reserve against the remaining three-hour budget. '
         'Adapter probes inspect matching candidate states. `execute_mode` starts gates, pilot and evaluation only when explicitly called in a live mode.'),
        ('verify_request_evidence verify_request_evidence_under_config',
         'Replay reconstructs each captured request from frozen conversations and candidate IDs, then checks its exact hash, context-admission record and parsed scores. '
         'A mismatch fails loudly instead of blessing a changed experiment as the original result.')]),
      ('Metrics and uncertainty', [
        ('labels_for annotate_rows usage_summary summarize_run',
         'These functions attach scoring labels after retrieval, count strict passage-ID Recall@1/5/10, retain exact-text sensitivity and summarize latency and resource evidence. '
         'Denominators distinguish scheduled, attempted, valid and failed work. Stored timings remain the original inference timings.'),
        ('dialogue_bootstrap primary_verdict common_success_sensitivities analyze analyze_under_config',
         'The bootstrap resamples whole dialogues within domain and keeps C/D paired for 10,000 draws. '
         'The primary verdict jointly considers quality and latency, with explicit validity gates and an exploratory mini-study qualification. '
         '`analyze` returns recomputed results; its wrapper uses the capture’s configuration so uncertainty calculations remain reproducible.')]),
      ('Plotting and publication', [
        ('wrap percent seconds_label figure_title cohort_subtitle table_figure',
         'Formatting helpers produce readable labels and shared title spacing. Tables use black text and explicit light cell backgrounds for contrast. '
         'No helper overrides figure size: all graphs and tables inherit CFG’s Matplotlib default.'),
        ('plot_paths plot_scope', 'These draw the four retrieval paths and corpus/query scope from the fixed configuration and actual capture. '
         'Their sidecars retain the plotted data; diagrams are conceptual, while scope counts describe the selected cohort.'),
        ('plot_accuracy plot_tradeoff plot_primary', 'Accuracy plots show strict-ID recall and dialogue-bootstrap intervals. '
         'The quality/latency plot uses a log time axis so millisecond BM25 and model-based retrieval remain visible together. The paired panel reads C−D quality and C/D time directly from analysis.'),
        ('plot_failures plot_latency plot_example plot_feasibility',
         'Diagnostics distinguish page pruning, retained-page ID misses and query latency distributions. The optional worked example requires genuine training traces. '
         'Feasibility plots separate the pilot forecast from a resource lower bound; neither is a new inference measurement.'),
        ('publication_tables flatten_sidecar asset_caption',
         'Publication tables and captions are generated from analysis rather than hand-copied numbers. Sidecars retain machine-readable values behind rounded display text, denominators and cohort qualifications.'),
        ('validate_exports export_publication export_publication_under_config',
         'The exporter saves each figure with CFG’s DPI, checks image dimensions and content hashes, and records captions plus JSON/CSV sidecars. '
         'A numerical receipt checks that visual changes leave the scientific results unchanged.'),
        ('book_identity book_source_run score_resolution_audit book_reading',
         'Book helpers map the shared article sections to one numbered display sequence. Score-resolution diagnostics exclude singleton groups and retain the frozen tie rule. '
         'Readings are computed from the captured run, so the notebook’s interpretation stays attached to its actual evidence.'),
        ('show_book_visual prepare_book_contract export_book_assets show_closing_reading',
         'Display uses the same PNG bytes exported for the article. Numbered copies go into CFG.graph_folder, with `graph_` for plots and `post_` for tables. '
         'Unused table objects are closed to avoid phantom inline figures; the closing reading reports the measured comparison without adding model calls.')]),
    ]
    emitted = set()
    for title, batches in groups:
        markdown('## ' + title, utility_group=title)
        for names, explanation in batches:
            names = names.split()
            emitted.update(names)
            code('\n\n'.join(definitions[name] for name in names), explanation, ['utils'], hidden=True)
    assert emitted == set(definitions), {'missing': set(definitions)-emitted, 'extra': emitted-set(definitions)}

    body = section('setup')
    markdown(body)
    body = section('data')
    code('# Frozen selection IDs and checksums only; no corpus body.\nEMBEDDED_DATA = ' +
         pprint.pformat(embedded_data(), width=110, sort_dicts=False),
         'This metadata freezes dialogue IDs, exclusions and expected checksums. It contains neither the corpus text nor captured inference responses. '
         'The next step reconstructs data from the separate cached public archive and verifies the selected inputs against these checksums.', ['embedded-data'], True)
    # Keep each data-initialization cell focused on one action.
    data_runtime = runtime['02_data.py']
    boundaries = [data_runtime.index('MINI ='), data_runtime.index('DATA_AUDIT =')]
    code(data_runtime[:boundaries[0]],
         'The archive and tokenizer are downloaded only if absent, then verified. Archive reconstruction produces the complete corpus and separate input/label objects. '
         'CAPTURED reads external result JSON when present; it does not invent model outputs. The tokenizer is used locally for context admission.', ['data-init'])
    code(data_runtime[boundaries[0]:boundaries[1]],
         'The active profile selects 32 fixed pages, all 329 complete passages and 12 validation queries. Assertions check dialogue uniqueness, corpus membership and exclusions. '
         'All four retrieval methods will use this same active corpus.', ['data-init'])
    code(data_runtime[boundaries[1]:], body + '\n\n`validate_dataset` verifies the full reconstruction before choosing pilot inputs. '
         'The separately labelled ten-page training demo is retained for compatibility; it is not the twelve-query evaluation.', ['data-init'])

    def visual(stem, explanation, prepared=None):
        args = (', prepared=' + prepared) if prepared else ', analysis=ANALYSIS, runs=RUNS'
        code('show_book_visual(' + repr(stem) + args + ')', explanation,
             ['figures-run', 'book-visual', stem])

    visual('graph_02_corpus_and_evaluation_scope',
        'Graph 1 checks the fixed corpus and training/evaluation allocation before we compare methods. The counts are read from the selected capture, not inferred from model success.',
        'plot_scope(book_source_run())')
    body = section('bm25')
    code(runtime['03_retrieval.py'] + '\n\nBM25_EXAMPLE = ACTIVE_RETRIEVER.lexical(PILOT_INPUTS[0], CFG.experiment["output_k"])\nprint("Training example: BM25 returned", len(BM25_EXAMPLE), "passage IDs.")',
         body + '\n\nThis cell builds the lexical indices and retrieves one training example without a model call. The returned IDs show that the baseline is ready; they are not held-out accuracy estimates.', ['method-example'])
    body = section('judge')
    code('JUDGE_CANDIDATES = [ACTIVE_RETRIEVER.passage_by_id[pid] for pid in BM25_EXAMPLE[:4]]\nJUDGE_BATCH = shared_batch(PILOT_INPUTS[0], JUDGE_CANDIDATES, "passage")\nprint("Training example:", len(JUDGE_BATCH["questions"]), "relevance questions; no request sent.")', body,
         ['method-example'])
    body = section('navigation')
    code('PAGE_BATCHES = deterministic_batches(PILOT_INPUTS[0], ACTIVE_RETRIEVER.pages, "page", [])\nprint("Training example:", len(PAGE_BATCHES), "page states; retain", CFG.experiment["page_budget"], "pages after scoring.")',
         body + '\n\nHere we prepare complete page states and check their sizes locally. Scoring and branch selection happen inside `Retriever.run` during live execution, or are read from captured traces during replay.', ['method-example'])
    body = section('control')
    code('NIMBLE_PAYLOAD = payload_for("nimble", JUDGE_BATCH)\nQWEN_PAYLOAD = payload_for("qwen", JUDGE_BATCH)\nprint("Same training state, two adapters:", NIMBLE_PAYLOAD["model"], "and", QWEN_PAYLOAD["model"])',
         body + '\n\n`payload_for` constructs both adapters locally. Nothing is sent to Ollama in this illustrative cell. The common input and parsing rules are in Utils; model-dependent branches are retained in the actual end-to-end results.', ['method-example'])
    visual('graph_01_four_retrieval_paths', 'Graph 2 gathers the four paths into one diagram: A/B changes judging, B/C changes discovery, and C/D replaces the model plus adapter under common navigation rules.', 'plot_paths()')
    body = section('protocol')
    visual('table_01_experiment_contract', body + '\n\nTable 3 records the frozen comparison contract. Its values come from the selected run’s scope and configuration.', 'prepare_book_contract()')
    experiments = sources['04_experiments.py'].split('\n# %%\n')
    code(experiments[-2], 'These offline checks exercise parsing, ranking, batching and controller boundaries without inference. A pass verifies plumbing, not the relevance quality of either model.', ['offline-checks'])
    code(experiments[-1], 'In replay mode this loads the external captured runs, checks corpus identity and exports frozen inputs/labels. '
         'In evaluate mode the same cell runs the live workflow under the configured deadline. Read the reported run status before interpreting any metrics.', ['orchestrate'])
    body = section('metrics')
    analysis_run = sources['05_analysis.py'].split('\n# %%\n')[-1]
    code(analysis_run, body, ['analysis-run'])
    code('# FIGURES_RUN: regenerate the detailed gallery and the numbered article assets.\n'
         'ASSET_DIR, ASSET_MANIFEST, RECEIPT = export_publication(RUNS, ANALYSIS, show=False)\n'
         'BOOK_MANIFEST = export_book_assets(ASSET_DIR, ASSET_MANIFEST, ANALYSIS, RUNS)\n'
         'display(Markdown(f"**Exports verified:** {RECEIPT[\"export_validation\"][\"generated_images\"]} figures/tables; numbered copies in `./graphs`."))',
         'This regenerates all available plots and tables from recomputed analysis, then writes numbered article copies and machine-readable sidecars. '
         'The numerical receipt must match the captured experiment; replay timings are not substituted for inference timings.', ['figures-run', 'book-export'])
    visual('graph_09_laptop_feasibility', 'Graph 4 compares the training-only runtime forecast with the workflow allowance. Observed inference time remains separate from this replay’s plotting time.')
    body = section('results')
    for stem, explanation in [
        ('graph_03_retrieval_accuracy', body + '\n\nGraph 5 reports strict-ID recall with dialogue-bootstrap intervals. A wider interval means the twelve-query sample leaves substantial uncertainty.'),
        ('table_02_four_method_results', 'Table 6 places the four methods on the same attempted-query denominator, with quality, latency and failures side by side. These are the original measured inference times.'),
        ('graph_04_quality_latency_tradeoff', 'Graph 7 compares quality against full-query latency. Its logarithmic x-axis accommodates BM25’s millisecond runtime and the much slower model-based methods.')]:
        visual(stem, explanation)
    body = section('value')
    visual('graph_05_primary_decision_model_comparison', body + '\n\nGraph 8 keeps C/D paired. Zero quality difference is not proof of equivalence; a latency ratio above one means C took longer.')
    visual('table_03_primary_verdict', 'Table 9 combines the quality uncertainty and timing gates into the exploratory verdict. A completed run is not automatically evidence of decision-model superiority.')
    body = section('failures')
    visual('graph_06_where_retrieval_fails', body + '\n\nGraph 10 separates where the two hierarchies lose the annotated ID. The exact-text sensitivity prevents an ID mismatch from being mistaken for wholly different evidence.')
    visual('table_04_domain_results', 'Table 11 reports each domain separately for inspection. With only three evaluation queries per domain, a one-query swing is large and does not support a stable domain ranking.')
    visual('graph_07_query_latency_distribution', 'Graph 12 shows the full distribution of attempted-query latencies. Each curve’s jumps are individual queries, not independent request groups or repeated hardware trials.')
    body = section('closing')
    code('show_closing_reading(ANALYSIS)', body, ['figures-run', 'closing-reading'])
    nb.cells = cells
    nbformat.validate(nb)
    target = ROOT / 'decisionrag_ollama_demo.ipynb'
    nbformat.write(nb, target)
    print('Built', target, 'cells=', len(cells), 'bytes=', target.stat().st_size)

if __name__ == '__main__':
    build()
