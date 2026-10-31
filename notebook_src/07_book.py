# Presentation only: original inference rows, rankings and timings are never changed.

def book_identity(stem):
    number, spec = next((i, spec) for i, spec in enumerate(CFG.book_assets, 1) if spec[0] == stem)
    return {'number': number, 'section': spec[1], 'kind': 'table' if spec[2] == 'post' else 'graph',
            'filename': f'{spec[2]}_{number}_0111.png'}

def book_source_run():
    if CFG.mode != 'replay':
        return None
    if CFG.replay_run_file:
        return json.loads(Path(CFG.replay_run_file).read_text())
    return CAPTURED.get('evaluation') or CAPTURED.get('pilot') or CAPTURED.get('demo') or CAPTURED.get('gate')

def score_resolution_audit(run):
    output = {}
    for method in ('C', 'D'):
        groups = [r for r in (run or {}).get('requests', []) if r['option'] == method and 'scores' in r]
        multi = [r for r in groups if len(r['scores']) >= 2]
        output[method] = {'scored_request_groups': len(groups),
            'http_attempts_for_scored_groups': sum(len(r['attempts']) for r in groups),
            'singleton_groups_excluded': len(groups) - len(multi), 'multi_question_groups': len(multi),
            'all_equal_multi_question_groups': sum(len(set(r['scores'].values())) == 1 for r in multi),
            'tied_multi_question_groups': sum(len(set(r['scores'].values())) < len(r['scores']) for r in multi)}
    return {'source_run_id': (run or {}).get('run_id'), 'methods': output,
        'tie_rule': 'Equal scores are ordered by candidate ID; the frozen ranking policy is unchanged.',
        'scope': 'Successful scored request groups, excluding singletons for tie diagnostics. These are not independent queries or evidence that tied scores are incorrect.'}

def book_reading(stem, analysis=None, runs=None):
    if stem == 'graph_02_corpus_and_evaluation_scope':
        s = study_scope(book_source_run())
        return (f"We search {s['pages']} pages and {s['passages']} complete passages. "
            f"The pilot uses {s['pilot_queries']} training queries and evaluation has {s['evaluation_queries']} queries. "
            'Domain balance makes the design easy to inspect; it does not make twelve queries representative of all RAG workloads.')
    if stem == 'graph_01_four_retrieval_paths':
        return ('A→B adds judging/rescoring; B→C changes discovery, candidate pools and workload; '
            'C→D replaces the model and scoring adapter under common navigation rules. The initial page corpus matches, '
            'but selected pages and downstream work can differ. These are end-to-end system comparisons, not causal isolation of architecture or training.')
    if stem == 'table_01_experiment_contract':
        return ('This table is our contract with the experiment: corpus, cohort, models and resource limits are fixed. '
            'The mini study is explicitly exploratory. A successful run does not by itself establish a retrieval-quality or efficiency win.')
    if stem == 'graph_09_laptop_feasibility':
        p = (runs or {}).get('pilot', {}).get('projection', {})
        if p.get('seconds') is not None:
            return (f"The training pilot forecast {p['seconds']/60:.1f} evaluation minutes, including reserve, against "
                f"{p['budget_seconds']/60:.1f} minutes remaining. Resource gate: {'PASS' if p['passed'] else 'NOT MET'}. "
                'This was a forecast made before evaluation, not a new measurement during replay.')
        return 'The available pilot does not supply a complete point forecast; an incomplete gate stays visible.'
    if not analysis:
        return 'Measured results are unavailable; no outcome is inferred from the implementation alone.'
    by_m = {r['option']: r for r in analysis['methods']}
    if stem == 'graph_03_retrieval_accuracy':
        return ('At rank one, ' + '; '.join(f"{m}: {percent(by_m[m]['r1'])}" for m in CFG.methods) + '. '
            'Read Recall@1 and Recall@10 separately: a method can return useful evidence in its bundle without choosing it first. '
            'The error bars are exploratory dialogue-bootstrap intervals, not proof of a universal ordering. '
            'Exact normalized-text R@1 is ' + '; '.join(f"{m}: {percent(by_m[m]['text_r1'])}" for m in CFG.methods) +
            '; this sensitivity leaves strict-ID scoring primary and does not establish equivalence or an efficiency win.')
    if stem == 'table_02_four_method_results':
        return ('Each method attempted ' + ', '.join(f"{m}: {by_m[m]['attempted']}/{by_m[m]['scheduled']}" for m in CFG.methods) + '. '
            'The table puts completed counts, failures, strict recall and full-query latency together. '
            'For scale, the BM25 median is ' + seconds_label(by_m['A']['p50_seconds']) + ' seconds; model-assisted retrieval takes much longer here.')
    if stem == 'graph_04_quality_latency_tradeoff':
        return ('Left is faster, higher is better. The logarithmic horizontal axis matters: milliseconds and minutes occupy the same plot. '
            'A/B were measured in the later sweep, so their speed comparisons with C/D are descriptive. '
            'The picture does not include answer-generation quality or a modern dense retrieval baseline.')
    if stem == 'graph_05_primary_decision_model_comparison':
        p = analysis['bootstrap']
        if p.get('status') != 'available':
            return 'A complete paired cohort is unavailable, so no paired advantage is reported.'
        v = p['paired']; lo, hi = v['quality_difference_ci']; tlo, thi = v['median_time_ratio_ci']
        return (f"C−D Recall@1 is {100*v['quality_difference']:.1f} percentage points, with a 95% interval "
            f"from {100*lo:.1f} to {100*hi:.1f}. The median-time ratio C/D is {v['median_time_ratio']:.2f} "
            f"(95% interval {tlo:.2f}–{thi:.2f}); a ratio above one means Nimble took longer. "
            'Equal observed recall is not evidence of equivalence. The interval resamples queries from one captured run, '
            'not repeated runs or hardware. Different selected branches and workloads are included in this end-to-end time comparison.')
    if stem == 'table_03_primary_verdict':
        return (f"The recorded verdict is **{analysis['primary']['verdict']}**. "
            'Completion of all four mini methods is distinct from running our original local 259-query study. '
            'Input fit, warm residency and timing integrity describe execution validity; they do not turn an uncertain quality comparison into a win.')
    if stem == 'graph_06_where_retrieval_fails':
        counts = {m: Counter(r['failure_class'] for r in analysis['rows'] if r['option'] == m) for m in ('C', 'D')}
        source = (runs or {}).get('evaluation') or (runs or {}).get('pilot') or (runs or {}).get('demo')
        audit = score_resolution_audit(source)['methods']; c, d = audit['C'], audit['D']
        return (f"Nimble pruned the gold page on {counts['C']['gold page pruned']} queries; Qwen did so on "
            f"{counts['D']['gold page pruned']}. Retaining a page is only the first hurdle: "
            f"Nimble then had {counts['C']['retained page, wrong rank one']} rank-one structural-ID misses on retained pages, "
            f"and Qwen {counts['D']['retained page, wrong rank one']}. An ID miss may still match the target text.\n\n"
            f"**Score resolution:** C made {c['scored_request_groups']} scored request groups and D {d['scored_request_groups']}. "
            f"Excluding singletons, D had all-equal scores in {d['all_equal_multi_question_groups']}/{d['multi_question_groups']} groups "
            f"and some tie in {d['tied_multi_question_groups']}/{d['multi_question_groups']}; C had "
            f"{c['all_equal_multi_question_groups']}/{c['multi_question_groups']} all-equal groups and "
            f"{c['tied_multi_question_groups']}/{c['multi_question_groups']} groups with any tie. "
            'Equal scores fall back to candidate-ID order. These are dependent request groups, not extra evaluation queries; '
            'ties can be appropriate for irrelevant candidates. They describe this adapter, not a general ordinary-LLM limitation or a cause of its speed.')
    if stem == 'table_04_domain_results':
        return ('The domain breakdown is an inspection aid, not four small leaderboards. '
            'With three validation queries per domain, a single changed answer moves recall by 33.3 percentage points. '
            'Do not choose a domain-specific winner from these counts.')
    if stem == 'graph_07_query_latency_distribution':
        return ('Every attempted query contributes to its curve, including any failed attempt. '
            'The right tail exposes slower cases that a median hides. The curves reuse original inference durations; '
            'the time taken to redraw them is not a speed benchmark.')
    raise KeyError(stem)

def show_book_visual(stem, prepared=None, analysis=None, runs=None):
    identity = book_identity(stem)
    path = CFG.graph_folder / identity['filename']
    if prepared is not None:
        fig, _ = prepared
        CFG.graph_folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=CFG.export_dpi)
        plt.close(fig)
    else:
        entry = next(a for a in BOOK_MANIFEST['assets'] if a['stem'] == stem)
        if entry['status'] != 'generated':
            display(Markdown(f"**Visual {identity['number']} unavailable:** {entry['reason']}"))
            return
    display(Markdown(f"**{'Table' if identity['kind'] == 'table' else 'Graph'} {identity['number']}** · `{path}`"))
    display(NotebookImage(filename=str(path)))
    display(Markdown(book_reading(stem, analysis, runs)))

def prepare_book_contract():
    source = book_source_run()
    with recorded_configuration(source or {}):
        tables = publication_tables(None, {source['kind']: source} if source else {})
    for stem, (fig, _) in tables.items():
        if stem != 'table_01_experiment_contract':
            plt.close(fig)
    return tables['table_01_experiment_contract']

def export_book_assets(folder, manifest, analysis, runs):
    CFG.graph_folder.mkdir(parents=True, exist_ok=True)
    by_stem = {a['stem']: a for a in manifest['assets']}
    assets = []
    for stem, _, _ in CFG.book_assets:
        identity = book_identity(stem); src = by_stem[stem]
        entry = {**identity, 'stem': stem, 'status': src['status'], 'source_run_id': src['run_id'],
                 'reason': src.get('reason'), 'path': None}
        if src['status'] == 'generated':
            name = Path(identity['filename'])
            for key, suffix in (('path', '.png'), ('json_path', '.json'), ('csv_path', '.csv')):
                dest = CFG.graph_folder / name.with_suffix(suffix)
                dest.write_bytes((folder / src[key]).read_bytes())
                entry[key] = dest.name
                entry[key.replace('_path', '') + '_sha256' if key != 'path' else 'sha256'] = hashlib.sha256(dest.read_bytes()).hexdigest()
            entry.update(caption=src['caption'], alt_text=src['alt_text'], reading=book_reading(stem, analysis, runs))
        assets.append(entry)
    book = {'schema_version': 1, 'source_run_id': manifest['run_id'], 'mode': CFG.mode,
        'numbering': 'One sequence in notebook display order, including graphs and tables; literal suffix 0111.',
        'numerical_receipt': manifest['numerical_receipt'], 'assets': assets}
    source = runs.get('evaluation') or runs.get('pilot') or runs.get('demo') or runs.get('gate')
    audit = score_resolution_audit(source)
    atomic_json(CFG.graph_folder / 'score_resolution_audit.json', audit)
    book['score_resolution_audit'] = {'path': 'score_resolution_audit.json',
        'sha256': hashlib.sha256((CFG.graph_folder / 'score_resolution_audit.json').read_bytes()).hexdigest()}
    atomic_json(CFG.graph_folder / 'manifest.json', book)
    captions = ['# Notebook visuals', '', book['numbering'], '']
    gallery = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Decision models: notebook visuals</title>',
        '<style>body{font:18px system-ui;max-width:1050px;margin:40px auto;padding:0 20px;color:#111}img{max-width:100%}figure{margin:45px 0}</style>',
        '<h1>Decision models: notebook visuals</h1><p>' + html.escape(book['numbering']) + '</p>']
    for a in assets:
        captions += [f"## Visual {a['number']}: {a['filename']}", '', '**Notebook/post section:** ' + a['section'], '']
        if a['status'] == 'generated':
            captions += [a['caption'], '', a['reading'], '', '**Alt text:** ' + a['alt_text'], '']
            gallery += [f"<figure><h2>Visual {a['number']}: {html.escape(a['section'])}</h2><img src=\"{a['path']}\" alt=\"{html.escape(a['alt_text'], quote=True)}\"><figcaption>" + html.escape(a['reading']) + '</figcaption></figure>']
        else:
            captions += ['Not generated: ' + a['reason'], '']
    (CFG.graph_folder / 'captions.md').write_text('\n'.join(captions))
    (CFG.graph_folder / 'index.html').write_text('\n'.join(gallery) + '</html>')
    return book

def show_closing_reading(analysis):
    if not analysis or analysis['status'] != 'complete':
        display(Markdown('The comparison is incomplete; no ordering or efficiency claim is supported.'))
        return
    by_m = {r['option']: r for r in analysis['methods']}
    best = max(by_m, key=lambda m: by_m[m]['r1'])
    c, d = by_m['C'], by_m['D']
    display(Markdown(f"**Reading of this captured run:** {best} had the highest observed Recall@1 "
        f"({percent(by_m[best]['r1'])}). C reached {percent(c['r10'])} Recall@10, versus {percent(d['r10'])} for D. "
        f"C/D Recall@1 was {percent(c['r1'])}/{percent(d['r1'])}; their median latencies were "
        f"{c['p50_seconds']:.1f}/{d['p50_seconds']:.1f} seconds. These twelve-query results do not establish "
        'the expected A < B < C ordering or a general decision-model advantage. The paired intervals and execution gates remain part of the finding.'))
