
def wrap(text, width=55):
    return '\n'.join(textwrap.wrap(str(text), width=width, break_long_words=True, replace_whitespace=False))

def percent(value):
    return '—' if value is None else f'{value * 100:.2f}%'

def seconds_label(value):
    return '—' if value is None else f'{value:.3f}'

def figure_title(fig, title, subtitle=''):
    fig.suptitle(title, fontsize=18, fontweight='bold', x=.04, ha='left', y=.98)
    if subtitle:
        subtitle_y = .98 - 28 / (fig.get_size_inches()[1] * 72)
        fig.text(.04, subtitle_y, subtitle, fontsize=11, ha='left', va='top')

def cohort_subtitle(analysis):
    qualifier = ('MINI VALIDATION · exploratory' if is_mini_profile(analysis.get('study_profile'))
        else 'HELD-OUT') if analysis['kind'] == 'evaluation' else 'TRAINING ONLY · exploratory'
    return f"{qualifier} · {analysis['scheduled_queries']} scheduled queries · {analysis['status']}"

def table_figure(title, columns, rows, widths=None, subtitle=''):
    fig, ax = plt.subplots()
    figure_title(fig, title, subtitle)
    ax.axis('off')
    table = ax.table(cellText=rows, colLabels=columns, colWidths=widths,
        cellLoc='left', colLoc='left', bbox=[0, 0, 1, .84])
    table.auto_set_font_size(False); table.set_fontsize(11.5)
    for (row, col), cell in table.get_celld().items():
        cell.set_text_props(color='black')
        cell.set_edgecolor('#D3D8DE'); cell.set_facecolor('white')
        if row == 0:
            cell.set_facecolor('#EAF0F6'); cell.set_text_props(weight='bold', color='black')
    fig.subplots_adjust(left=.04, right=.98, top=.87, bottom=.06)
    return fig

# %%
def plot_paths():
    fig, ax = plt.subplots(); ax.axis('off')
    figure_title(fig, 'Four ways to find the next piece of evidence', 'Conceptual diagram · same corpus and six-turn conversation')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    paths = {'A': ['All passages', 'BM25', 'Top 10 passages'],
        'B': ['BM25 top 50', 'Nimble judgments\n+ prefix rescore', 'Top 10 passages'],
        'C': ['All page metadata', 'Nimble → 3 pages\n→ their passages', 'Nimble ranking\n→ top 10'],
        'D': ['All page metadata', 'Qwen scores → 3 pages\n→ their passages', 'Qwen ranking\n→ top 10']}
    for i, (method, boxes) in enumerate(paths.items()):
        y = .84 - i * .22
        ax.text(.005, y, method, fontsize=19, fontweight='bold', color=CFG.colors[method], va='center')
        for j, text in enumerate(boxes):
            x = .10 + .31 * j
            ax.add_patch(FancyBboxPatch((x, y - .075), .26, .15,
                boxstyle='round,pad=0.012', linewidth=1.6, edgecolor=CFG.colors[method], facecolor='white'))
            ax.text(x + .13, y, text, ha='center', va='center', fontsize=11)
            if j < 2:
                ax.add_patch(FancyArrowPatch((x + .273, y), (x + .30, y),
                    arrowstyle='-|>', mutation_scale=15, color=CFG.colors[method]))
    fig.subplots_adjust(left=.04, right=.98, top=.84, bottom=.05)
    return fig, {'paths': paths, 'configuration': CFG.experiment}

def plot_scope(run=None):
    scope = study_scope(run)
    mini = is_mini_profile(scope['profile'])
    fig, axes = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1.3]})
    figure_title(fig, 'A restricted corpus for a local demo' if mini else
        'A small query set searches the complete corpus',
        'Fixed corpus for all four methods · no target answer in the request')
    axes[0].axis('off')
    for y, title, detail in ((.82, '8 pages per domain' if mini else '488 raw documents',
                             'Selected by hash before query selection' if mini else 'Merged by canonical title'),
                             (.52, f"{scope['pages']:,} pages", 'Domain, title, complete headings'),
                             (.22, f"{scope['passages']:,} passages", 'Complete structural sections')):
        axes[0].text(.05, y, title, fontsize=16, weight='bold')
        axes[0].text(.05, y - .10, detail, fontsize=10)
    count_data = [{'domain': d, 'design': sum(l['domain'] == d for l in scope['design_labels'].values()),
        'evaluation': sum(l['domain'] == d for l in scope['evaluation_labels'].values())} for d in CFG.domains]
    x = np.arange(4)
    axes[1].bar(x - .19, [r['design'] for r in count_data], .36, color='#929DA8', label=f"{scope['design_queries']} training-design queries")
    bars = axes[1].bar(x + .19, [r['evaluation'] for r in count_data], .36, color=CFG.colors['C'], label=f"{scope['evaluation_queries']} validation queries")
    axes[1].bar_label(bars, padding=3, fontsize=11)
    axes[1].set_xticks(x, ['DMV', 'SSA', 'VA', 'Student\naid'])
    axes[1].set_ylabel('Selected user turns'); axes[1].set_ylim(0, 8 if mini else 86)
    axes[1].legend(loc='upper center', bbox_to_anchor=(.47, 1.22), frameon=False, fontsize=10)
    fig.text(.05, .045, f"{scope['pilot_queries']} pilot dialogues · {scope['evaluation_dialogues']} evaluation dialogues · one query per dialogue" if mini else
        '20 training dialogues for design · 40 validation dialogues for evaluation · dialogue-cluster uncertainty', fontsize=10)
    fig.subplots_adjust(left=.05, right=.98, top=.72, bottom=.17, wspace=.3)
    return fig, {'counts': count_data, 'audit': {k: v for k, v in scope.items()
        if k not in ('design_labels', 'evaluation_labels')}}

# %%
def plot_accuracy(analysis):
    fig, ax = plt.subplots()
    figure_title(fig, 'Does judging and navigation improve retrieval?', cohort_subtitle(analysis))
    x = np.arange(3); width = .18
    bootstrap = analysis['bootstrap']
    for i, row in enumerate(analysis['methods']):
        method = row['option']; values = [row[f'r{k}'] for k in (1, 5, 10)]
        positions = x + (i - 1.5) * width
        ax.bar(positions, values, width, color=CFG.colors[method], label=method + ' · ' + CFG.methods[method])
        if bootstrap['status'] == 'available':
            limits = np.array([bootstrap['quality_intervals'][method][f'r{k}'] for k in (1, 5, 10)])
            ax.errorbar(positions, values, yerr=[np.maximum(0, np.array(values) - limits[:, 0]),
                np.maximum(0, limits[:, 1] - np.array(values))], fmt='none', ecolor='black', capsize=3, lw=1)
    ax.set_xticks(x, ['Recall@1', 'Recall@5', 'Recall@10']); ax.set_ylim(0, 1.06)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y * 100:.0f}%'))
    ax.set_ylabel('Strict passage-ID recall'); ax.grid(axis='y', alpha=.18); ax.set_axisbelow(True)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.17), ncol=2, frameon=False, fontsize=10)
    fig.subplots_adjust(left=.11, right=.98, top=.77, bottom=.24)
    return fig, {'methods': analysis['methods'], 'bootstrap': bootstrap}

def plot_tradeoff(analysis):
    fig, ax = plt.subplots()
    figure_title(fig, 'Retrieval quality has a time cost', cohort_subtitle(analysis))
    for i, row in enumerate(analysis['methods']):
        if row['attempted']:
            m = row['option']
            ax.scatter(row['p50_seconds'], row['r1'], color=CFG.colors[m], marker=CFG.markers[m],
                s=110, label=m + ' · ' + CFG.methods[m])
            offsets = {'A': (9, 10), 'B': (-13, 13), 'C': (12, 0), 'D': (-14, -18)}
            ax.annotate(m, (row['p50_seconds'], row['r1']),
                xytext=offsets[m], textcoords='offset points', fontsize=10)
    ax.set_xscale('log'); ax.set_ylim(-.03, 1.03)
    ax.set_xlabel('Median full-query seconds · logarithmic scale'); ax.set_ylabel('Recall@1')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y * 100:.0f}%'))
    ax.grid(alpha=.18)
    ax.legend(loc='upper left', fontsize=9, frameon=False)
    ax.margins(x=.4)
    fig.text(.10, .04, 'C/D: primary sweep first. A/B: later sweep; speed comparisons with C/D are descriptive.', fontsize=10)
    fig.subplots_adjust(left=.10, right=.96, top=.77, bottom=.20)
    return fig, {'methods': analysis['methods']}

# %%
def plot_primary(analysis):
    paired = analysis['bootstrap']['paired']
    primary = analysis['primary']
    mini = is_mini_profile(analysis.get('study_profile'))
    fig, axes = plt.subplots(1, 2)
    figure_title(fig, 'Does Nimble offer a useful efficiency tradeoff?', cohort_subtitle(analysis))
    for ax, value, interval, boundary, title, xlabel, scale in (
        (axes[0], paired['quality_difference'], paired['quality_difference_ci'], 0 if mini else -.02,
            'Quality difference: C − D', 'Percentage points in Recall@1', 100),
        (axes[1], paired['median_time_ratio'], paired['median_time_ratio_ci'], 1,
            'Median query time: C / D', 'Ratio · less than 1 favors C', 1)):
        value *= scale; interval = [v * scale for v in interval]; boundary *= scale
        ax.axvline(boundary, color='#A23D32', ls='--', lw=1.5,
            label=('Equal recall' if mini else '−2 pp margin') if scale == 100 else 'Equal median time')
        if scale == 100:
            ax.axvline(0, color='#BBBBBB', lw=1)
        ax.errorbar(value, 0, xerr=[[max(0, value - interval[0])],
            [max(0, interval[1] - value)]], fmt='o', color=CFG.colors['C'], capsize=7, ms=9, lw=2)
        ax.set_yticks([]); ax.set_ylim(-1, 1); ax.set_xlabel(xlabel, fontsize=10)
        ax.set_title(title, fontsize=12)
        lo, hi = min(interval[0], boundary, value), max(interval[1], boundary, value)
        pad = max((hi - lo) * .22, .5 if scale == 100 else .05)
        ax.set_xlim(lo - pad, hi + pad); ax.legend(loc='upper center', frameon=False, fontsize=9)
    fig.text(.05, .08, 'VERDICT: ' + primary['verdict'].upper(), fontsize=13, weight='bold')
    fig.text(.05, .025, wrap(primary['reason'], 120), fontsize=9, va='bottom')
    fig.subplots_adjust(left=.05, right=.98, top=.70, bottom=.30, wspace=.28)
    return fig, {'primary': primary, 'bootstrap': analysis['bootstrap']}

def plot_failures(analysis):
    categories = ['correct rank one', 'retained page, wrong rank one', 'gold page pruned', 'execution failure', 'unattempted']
    colors = ['#2B7A4B', '#83A6C8', '#D99746', '#AF3E43', '#D0D0D0']
    fig, ax = plt.subplots()
    figure_title(fig, 'Where the hierarchy misses the target passage', cohort_subtitle(analysis))
    category_labels = {'correct rank one': 'rank-one ID hit',
        'retained page, wrong rank one': 'page kept, rank-one ID miss'}
    records = []
    for y, method in enumerate(('C', 'D')):
        subset = [r for r in analysis['rows'] if r['option'] == method]
        counts = Counter(r['failure_class'] for r in subset)
        counts['unattempted'] = analysis['scheduled_queries'] - len(subset)
        assert sum(counts.values()) == analysis['scheduled_queries']
        left = 0
        for category, color in zip(categories, colors):
            count = counts[category]
            ax.barh(y, count, left=left, color=color, height=.5,
                label=category_labels.get(category, category) if y == 0 else None)
            if count >= max(1, analysis['scheduled_queries'] * .06):
                ax.text(left + count / 2, y, str(count), ha='center', va='center', color='white' if category in ('correct rank one', 'execution failure') else 'black', weight='bold')
            left += count
        records.append({'option': method, **{c: counts[c] for c in categories}})
    ax.set_yticks([0, 1], ['C · Nimble', 'D · Qwen']); ax.invert_yaxis()
    ax.set_xlabel('Queries · categories sum to the scheduled denominator')
    ax.set_xlim(0, max(1, analysis['scheduled_queries']))
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.28), ncol=2, frameon=False, fontsize=10)
    fig.subplots_adjust(left=.17, right=.97, top=.74, bottom=.33)
    return fig, {'counts': records}

# %%
def plot_latency(analysis):
    fig, ax = plt.subplots()
    figure_title(fig, 'How long did each query take?', cohort_subtitle(analysis))
    all_times = [r['seconds'] for r in analysis['rows']]
    plot_min, plot_max = min(all_times) / 1.25, max(all_times) * 1.2
    for method in CFG.methods:
        rows = [r for r in analysis['rows'] if r['option'] == method]
        if not rows:
            continue
        values = np.sort([r['seconds'] for r in rows])
        fail_count = sum(r['status'] != 'success' for r in rows)
        ax.step(np.r_[plot_min, values, plot_max],
            np.r_[0, np.arange(1, len(values) + 1) / len(values), 1], where='post',
            color=CFG.colors[method], label=f"{method} · n={len(rows)}, failed={fail_count} · p50 {np.median(values):.3f}s · p95 {np.percentile(values, 95):.3f}s")
        failed = [r['seconds'] for r in rows if r['status'] != 'success']
        if failed:
            ax.scatter(failed, [.04] * len(failed), marker='x', s=70, color=CFG.colors[method])
    ax.set_xscale('log'); ax.set_ylim(0, 1.04)
    clock_issue = analysis.get('timing_audit', {}).get('status') == 'failed'
    ax.set_xlabel(('Recorded active-timer seconds' if clock_issue else 'Full-query seconds') + ' · logarithmic scale'); ax.set_ylabel('Fraction of attempted queries')
    ax.grid(alpha=.2)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.24), ncol=2, fontsize=8, frameon=False)
    fig.text(.10, .045, 'UTC/timer discrepancy detected: these active timings cannot establish wall-clock speed.' if clock_issue else
        'All measured attempts included; × = failed query. Unattempted queries have no latency.', fontsize=10)
    fig.subplots_adjust(left=.10, right=.98, top=.77, bottom=.32)
    return fig, {'queries': [{k: r[k] for k in ('option', 'query_id', 'seconds', 'status', 'failure')}
                              for r in analysis['rows']]}

# %%
def plot_example(demo):
    analysis = summarize_run(demo)
    rows = {r['option']: r for r in analysis['rows']}
    fig = plt.figure()
    figure_title(fig, 'One real training query, four observed paths', 'Ten fixed pages · illustrative example, not held-out evidence')
    last = model_history(DEMO_INPUT)
    context = '\n'.join(t['role'] + ': ' + t['text'] for t in last[-3:])
    fig.text(.045, .87, wrap(context, 110), fontsize=10, va='top')
    for x, method in ((.045, 'C'), (.53, 'D')):
        row = rows.get(method)
        if row is None:
            continue
        fig.text(x, .69, method + ' · ' + CFG.methods[method], color=CFG.colors[method], weight='bold', fontsize=13)
        parts = [f"Status: {row['status']} · {row['seconds']:.2f}s",
            f"R@1={row['r1']} · R@10={row['r10']}", 'Retained pages:']
        for i, pid in enumerate(row['selected_pages'], 1):
            parts.append(wrap(f'{i}. {pid}', 47))
        parts.append('\nReturned passage IDs (first 3):')
        for i, pid in enumerate(row['ranking'][:3], 1):
            parts.append(wrap(f'{i}. {pid}', 47))
        if row['failure']:
            parts.append(wrap(row['failure']['category'], 47))
        fig.text(x, .65, '\n'.join(parts), va='top', fontsize=9.5, linespacing=1.4)
    bottom = []
    for method in ('A', 'B'):
        row = rows.get(method)
        if row:
            top = row['ranking'][0] if row['ranking'] else '(no valid result)'
            bottom.append(f"{method}: {row['seconds']:.3f}s · R@1={row['r1']} · first ID: {top}")
    gold = DESIGN_LABELS[DEMO_INPUT['query_id']]['gold_passage_ids'][0]
    bottom.append('Annotated gold: ' + gold)
    fig.text(.045, .14, '\n'.join(wrap(t, 112) for t in bottom), fontsize=9.5, va='top', linespacing=1.4)
    return fig, {'query': DEMO_INPUT, 'page_ids': DEMO_PAGE_IDS,
                 'gold_passage_id': gold, 'rows': analysis['rows'], 'run_id': demo['run_id']}

# %%
def plot_feasibility(pilot):
    projection = pilot['projection']
    is_bound = projection.get('seconds') is None
    evidence = projection.get('lower_bound', {}) if is_bound else projection
    estimates = evidence.get('method_seconds_without_reserve', {})
    fig, ax = plt.subplots()
    scope = study_scope(pilot)
    subtitle = ('Training-only LOWER BOUND · unfinished pilot · all unobserved timings assigned zero' if is_bound else
        f"Training-only forecast · {scope['pilot_queries']} pilot queries · projected to {scope['evaluation_queries']} queries")
    figure_title(fig, 'Can the four-method study fit the laptop budget?', subtitle)
    bottom = 0
    for method, seconds in estimates.items():
        minutes = seconds / 60
        ax.barh(0, minutes, left=bottom, height=.5, color=CFG.colors[method], label=method + ' · ' + CFG.methods[method])
        if minutes > 4:
            ax.text(bottom + minutes / 2, 0, f'{minutes:.1f}m', ha='center', va='center', color='white', weight='bold')
        bottom += minutes
    total = (evidence.get('seconds') or 0) / 60
    reserve = max(0, total - bottom)
    ax.barh(0, reserve, left=bottom, height=.5, color='#D5DADE', hatch='///',
        label='25% reserve only' if is_bound else '25% reserve + scheduling')
    budget_minutes = projection['budget_seconds'] / 60
    ax.axvline(budget_minutes, color='#A23D32', ls='--', label=f'Available evaluation budget: {budget_minutes:.1f}m')
    ax.set_yticks([]); ax.set_xlabel(('Minimum possible projected' if is_bound else 'Projected') + ' recorded-timer minutes'); ax.set_ylim(-.7, .7)
    ax.set_xlim(0, max(total * 1.1, budget_minutes * 1.2, 1)); ax.legend(loc='upper center', bbox_to_anchor=(.5, -.23), ncol=2, frameon=False, fontsize=10)
    fig.text(.05, .80, f"Resource gate: {'PASS' if projection['passed'] else 'FAIL'} · " +
        ('lower bound ' if is_bound else 'forecast ') + f'{total:.1f} minutes', fontsize=13, weight='bold')
    fig.subplots_adjust(left=.05, right=.98, top=.73, bottom=.34)
    return fig, projection

# %%
def publication_tables(analysis, runs):
    source = runs.get('evaluation') or runs.get('pilot') or runs.get('demo') or runs.get('gate')
    scope = study_scope(source)
    mini = is_mini_profile(scope['profile'])
    contract = [
        ['Primary contrast', 'C (Nimble) vs D (generated Qwen scores), same controller'],
        ['Corpus', f"{scope['pages']} pages / {scope['passages']:,} complete structural passages"],
        ['Validation cohort', f"{scope['evaluation_queries']} queries / {scope['evaluation_dialogues']} dialogues" +
            ('; exploratory restricted corpus' if mini else '; full-study holdout')],
        ['Design cohort', f"{scope['design_queries']} training queries; {scope['pilot_queries']}-query pilot"],
        ['Controller', f"3 pages; candidates ≤16; questions/request ≤{CFG.experiment.get('questions_per_request', 16)}; rescore ≤12"],
        ['Common input', '8,192-token ceiling minus 512 reserved; 48 KiB guard'],
        ['Models', 'Nimble 9B Q8_0 and Qwen3.5 9B Q8_0; local Ollama'],
        ['Quality comparison' if mini else 'Quality requirement',
            'Report paired C − D R@1 interval; exploratory' if mini else
            'Paired 95% lower bound of C − D R@1 > −2 pp'],
        ['Time comparison' if mini else 'Time requirement',
            'Report paired median-time ratio C/D interval; exploratory' if mini else
            'Paired 95% upper bound of median-time ratio C/D < 1'],
        ['Validity requirement', 'Complete mini schedule; input fit; warm residence; all C/D valid' if mini else
            'Full run + input fit + warm residence + all C/D valid'],
    ]
    output = {'table_01_experiment_contract': (table_figure('The experiment contract',
        ['Setting', 'Frozen specification'], [[a, wrap(b, 69)] for a, b in contract], [.23, .77]), {'rows': contract})}
    if analysis:
        rows = [[r['option'], str(r['attempted']) + '/' + str(r['scheduled']), str(r['failed']),
            percent(r['r1']), percent(r['r5']), percent(r['r10']), seconds_label(r['p50_seconds']),
            seconds_label(r['p95_seconds'])] for r in analysis['methods']]
        output['table_02_four_method_results'] = (table_figure('Quality, time, and failures for all four methods',
            ['Method', 'Done/N', 'Fail', 'R@1', 'R@5', 'R@10', 'p50 (s)', 'p95 (s)'], rows,
            [.10, .11, .07, .14, .14, .14, .15, .15], cohort_subtitle(analysis)), {'methods': analysis['methods']})
        primary = analysis['primary']
        gates = [[key.replace('_', ' '), 'PASS' if passed is True else 'NOT MET' if passed is False else 'UNAVAILABLE']
                 for key, passed in primary['gates'].items()]
        if is_mini_profile(study_scope()['profile']):
            complete_mini = analysis['status'] == 'complete' and all(
                r['attempted'] == r['scheduled'] and not r['failed'] for r in analysis['methods'])
            gates[0] = ['Original local 259-query study', 'OUTSIDE MINI SCOPE']
            gates.insert(0, ['Complete four-method mini run', 'PASS' if complete_mini else 'NOT MET'])
        gates += [['Verdict', primary['verdict']], ['Reason', wrap(primary['reason'], 63)]]
        output['table_03_primary_verdict'] = (table_figure('What can the primary comparison claim?',
            ['Condition', 'Evidence status'], gates, [.35, .65], cohort_subtitle(analysis)), primary)
        domain_rows = []
        for domain in CFG.domains:
            selected = [r for r in analysis['domains'] if r['domain'] == domain]
            if selected:
                by_m = {r['option']: r for r in selected}
                counts = '/'.join(str(by_m[m]['attempted']) if m in by_m else '0' for m in CFG.methods)
                domain_rows.append([domain.upper(), counts] + [percent(by_m[m]['r1']) if m in by_m else '—' for m in CFG.methods])
        output['table_04_domain_results'] = (table_figure('Observed Recall@1 by help center',
            ['Domain', 'N: A/B/C/D', 'A', 'B', 'C', 'D'], domain_rows,
            [.18, .20, .155, .155, .155, .155], cohort_subtitle(analysis)), {'domains': analysis['domains']})
    else:
        gate = runs.get('gate', {})
        output['table_03_primary_verdict'] = (table_figure('The primary comparison has not run',
            ['Condition', 'Status'], [['Held-out comparison', 'NOT RUN'],
                ['Backend gate', gate.get('status', 'not run')], ['Conclusion', 'No efficiency claim available']], [.4, .6]),
            {'verdict': 'not run', 'backend_gate': gate.get('status')})
    return output

def flatten_sidecar(value):
    """A deterministic tabular view; nested values are JSON, precision retained."""
    if isinstance(value, list) and all(isinstance(x, dict) for x in value):
        records = value
    elif isinstance(value, dict):
        records = None
        for key in ('methods', 'domains', 'queries', 'counts', 'rows'):
            candidate = value.get(key)
            if isinstance(candidate, list) and all(isinstance(x, dict) for x in candidate):
                records = candidate
                break
        if records is None:
            records = [value]
    else:
        records = [{'value': value}]
    return pd.DataFrame([{key: stable_json(item) if isinstance(item, (dict, list)) else item
                          for key, item in record.items()} for record in records])

def asset_caption(stem, run, analysis, source_run=None):
    if stem == 'graph_01_four_retrieval_paths':
        return ('Conceptual A/B/C/D schematic. C and D use the same navigation controller with different scoring adapters; no performance values are implied.',
                'Four horizontal paths compare lexical retrieval, lexical retrieval with Nimble, and hierarchical retrieval using either Nimble or Qwen.')
    if stem == 'graph_02_corpus_and_evaluation_scope':
        scope = study_scope(run)
        return (f"All four methods search the same {scope['pages']}-page, {scope['passages']:,}-passage corpus. "
                f"The frozen validation cohort has {scope['evaluation_queries']} queries in {scope['evaluation_dialogues']} dialogues. " +
                ('The corpus was selected first, eight pages per domain; queries require in-corpus gold evidence. Original holdout and explored dialogues are excluded. Results describe this restricted task.' if is_mini_profile(scope['profile']) else
                 'The complete corpus is searched; training-design data is separate.'),
                'Corpus hierarchy beside grouped bars for the training-design and held-out query counts in four domains.')
    if stem == 'table_01_experiment_contract':
        return ('Frozen settings and the joint quality, latency, and validity requirements. These are criteria, not measured success.',
                'Two-column table lists corpus, cohorts, controller limits, models, and the requirements for an efficiency claim.')
    if stem == 'graph_08_worked_navigation_example':
        return (f"Run {source_run['run_id']}: one fixed training query against ten fixed pages. The figure shows observed retained pages and passage IDs, plus A/B comparisons. Scores are not model explanations; this example is illustrative and gold-page inclusion is intentional.",
                'One training conversation followed by the observed Nimble and Qwen navigation paths and lexical baselines.')
    if stem == 'graph_09_laptop_feasibility':
        p = source_run['projection']
        scope = study_scope(source_run)
        budget_minutes = p['budget_seconds'] / 60
        clock_note = (' The source run has large UTC/active-timer gaps: these values reproduce its active-timer budget calculation, not a wall-clock forecast.'
            if run_clock_audit(source_run)['status'] == 'failed' else '')
        if p.get('seconds') is None:
            bound = p['lower_bound']
            return (f"Run {source_run['run_id']}: {bound['observed_method_queries']} of {bound['scheduled_method_queries']} scheduled training method-queries were observed. "
                f"Even assigning zero seconds to all unobserved pilot timings gives a {(bound['seconds']/60):.2f}-minute lower bound under the frozen domain-weighted projection formula, including 25% reserve and zero scheduling overhead. "
                f'The recorded available evaluation budget is {budget_minutes:.2f} minutes. This is a mathematical lower bound, not a point runtime forecast. The failed pilot cannot authorize evaluation.' + clock_note,
                'Observed contributions to the minimum possible projected runtime, with a 25-percent reserve, compared with the recorded budget.')
        return (f"Run {source_run['run_id']}: {scope['pilot_queries']} training pilot queries, domain-weighted to {scope['evaluation_queries']} queries over all four options. Forecast {(p.get('seconds') or 0)/60:.2f} minutes including 25% reserve and uncovered scheduling, against {budget_minutes:.2f} available minutes; gate {'passed' if p['passed'] else 'failed'}. Only two or three queries per domain makes the forecast uncertain." + clock_note,
                'Stacked projected runtime by method, reserve, and scheduling, compared with the fixed laptop budget.')
    if not analysis:
        return ('No held-out performance result is available. The displayed verdict reports experiment status only.',
                'A table identifies the missing experiment and its current gate status.')
    counts = ', '.join(r['option'] + ' ' + str(r['attempted']) + '/' + str(r['scheduled']) for r in analysis['methods'])
    prefix = f"Run {analysis['run_id']}; {analysis['cohort']}; attempted/scheduled: {counts}; status {analysis['status']}. "
    exploratory = (' Restricted-corpus mini comparison is exploratory and cannot confirm the original full-study claim.'
        if is_mini_profile(analysis.get('study_profile')) else
        ' Training-only estimates cannot confirm held-out performance.' if analysis['kind'] != 'evaluation' else '')
    if analysis.get('timing_audit', {}).get('status') == 'failed':
        exploratory += ' UTC timestamps and the active timer disagree; no wall-clock efficiency claim is supported.'
    intervals = ('Intervals are 95% percentile bounds from 10,000 paired whole-dialogue resamples within domain.'
                 if analysis['bootstrap']['status'] == 'available' else 'Paired intervals are unavailable for the incomplete schedule.')
    descriptions = {
        'graph_03_retrieval_accuracy': ('Strict passage-ID R@1/5/10; failed rankings count as misses. Unattempted outcomes are not invented. ' + intervals,
            'Grouped recall bars for A, B, C, and D at ranks one, five, and ten, with available dialogue-cluster intervals.'),
        'graph_04_quality_latency_tradeoff': ('Operational R@1 versus all-attempt median full-query seconds on a logarithmic time axis. C/D run first; A/B timings come from the later sweep and are secondary.',
            'A quality-versus-latency scatter plot with one colored, shaped marker per method.'),
        'graph_05_primary_decision_model_comparison': ('C minus D R@1 in percentage points versus ' +
            ('zero (exploratory comparison)' if is_mini_profile(analysis.get('study_profile')) else 'the −2 pp margin') +
            ', and ratio of median full-query times versus one. ' + intervals + ' Verdict: ' + analysis['primary']['verdict'] + '. Validity gates must also pass.',
            'Two interval plots compare the quality difference with ' +
                ('zero' if is_mini_profile(analysis.get('study_profile')) else 'the minus-two-percentage-point margin') +
                ' and the time ratio with equality.'),
        'graph_06_where_retrieval_fails': ('Strict structural-ID counts partition every scheduled query into a rank-one ID hit, retained-page ID miss, pruned gold page, execution failure, or unattempted. The categories sum to the scheduled denominator. An ID miss may still match the target text; see the exact-text sensitivity.',
            'Two stacked bars show Nimble and Qwen strict-ID hits, retained-page ID misses, page pruning, and execution status.'),
        'graph_07_query_latency_distribution': ('ECDFs include all attempted query times, including retries and failure time. Crosses mark failed queries. Unattempted queries have no latency; log time axis.',
            'Latency ECDFs for observed methods ' + ', '.join(r['option'] for r in analysis['methods'] if r['attempted']) +
                ', with median and ninety-fifth-percentile timings and failure markers.' +
                (' Methods with no attempted queries are omitted.' if any(not r['attempted'] for r in analysis['methods']) else
                 ' All four methods have attempted queries.')),
        'table_02_four_method_results': ('Strict-ID recall percentages and all-attempt p50/p95 seconds, with observed/scheduled denominators and failures. Unrounded values are in sidecars.',
            'Four method rows compare query completion, failures, recall, and latency.'),
        'table_03_primary_verdict': (analysis['primary']['reason'] + ' ' + intervals,
            'A table lists each primary quality, latency, and validity requirement and the resulting verdict.'),
        'table_04_domain_results': ('Strict R@1 by domain with the attempted A/B/C/D query counts. Full R@10 and failure data are in the numeric sidecars. The micro average describes this selected mixture.',
            'Domain rows compare the four methods’ recall at rank one with explicit query counts.'),
    }
    caption, alt = descriptions[stem]
    return prefix + caption + exploratory, alt

# %%
def validate_exports(asset_dir, manifest):
    checked = []
    for item in manifest['assets']:
        if item['status'] == 'not_generated':
            assert item['path'] is None and item['reason']
            continue
        for key, hash_key in (('path', 'sha256'), ('json_path', 'json_sha256'), ('csv_path', 'csv_sha256')):
            path = asset_dir / item[key]
            assert path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == item[hash_key]
        sidecar = json.loads((asset_dir / item['json_path']).read_text())
        assert sidecar['run_id'] == item['run_id']
        assert sidecar['data_sha256'] == sha256(sidecar['data'])
        assert item['caption'] and item['alt_text']
        assert item['width_pixels'] == round(CFG.img_dim1 * CFG.export_dpi)
        checked.append(item['path'])
    assert [i['stem'] for i in manifest['assets']] == [s[0] for s in CFG.asset_specs]
    captions = (asset_dir / 'captions.md').read_text()
    assert all(i['stem'] in captions for i in manifest['assets'])
    return {'status': 'passed', 'generated_images': len(checked), 'checked_files': checked,
            'checks': ['declared order', 'image and sidecar hashes', 'sidecar data hashes',
                       'captions and alt text', 'image widths', 'unavailable entries have no path']}

def export_publication(runs, analysis, show=True):
    source = runs.get('evaluation') or runs.get('pilot') or runs.get('demo') or runs.get('gate') or {}
    with recorded_configuration(source):
        return export_publication_under_config(runs, analysis, show)

def export_publication_under_config(runs, analysis, show=True):
    source = (runs.get('evaluation') or runs.get('pilot') or runs.get('demo') or runs.get('gate') or {})
    run_id = source.get('run_id', 'not-run-corpus-only')
    folder = CFG.output_folder / run_id / 'post_assets'; folder.mkdir(parents=True, exist_ok=True)
    manifest = {'schema_version': 1, 'run_id': run_id, 'created_at': utc_now(),
        'mode': CFG.mode, 'cohort': source.get('cohort', 'corpus and compatibility only'),
        'experiment_status': source.get('status', 'not_run'),
        'publication_status': 'completed-comparison' if analysis and analysis['kind'] == 'evaluation'
            and analysis['status'] == 'complete' else 'demo-or-feasibility-only',
        'study_profile': study_scope(source or None)['profile'],
        'claim_scope': MINI['claim_scope'] if is_mini_profile(study_scope(source or None)['profile']) else
            'Original full-corpus local adaptation; backend and validity gates apply.',
        'numerical_receipt': analysis['numerical_receipt'] if analysis else None,
        'assets': []}
    figures = {'graph_01_four_retrieval_paths': plot_paths(), 'graph_02_corpus_and_evaluation_scope': plot_scope(source or None)}
    if analysis:
        figures.update({'graph_06_where_retrieval_fails': plot_failures(analysis),
            'graph_07_query_latency_distribution': plot_latency(analysis)})
        if all(r['attempted'] == analysis['scheduled_queries'] for r in analysis['methods']):
            figures.update({'graph_03_retrieval_accuracy': plot_accuracy(analysis),
                'graph_04_quality_latency_tradeoff': plot_tradeoff(analysis)})
        if analysis['bootstrap']['status'] == 'available':
            figures['graph_05_primary_decision_model_comparison'] = plot_primary(analysis)
    if runs.get('demo') and len(runs['demo'].get('rows', [])) == 4:
        figures['graph_08_worked_navigation_example'] = plot_example(runs['demo'])
    if runs.get('pilot', {}).get('projection', {}).get('seconds') is not None or runs.get('pilot', {}).get('projection', {}).get('lower_bound'):
        figures['graph_09_laptop_feasibility'] = plot_feasibility(runs['pilot'])
    figures.update(publication_tables(analysis, runs))
    captions = ['# Figure captions and alt text', '', 'All timings are the original captured measurements. Offline replay does not benchmark inference.', '']
    gallery = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Decision retrieval evidence gallery</title>',
        '<style>body{font:18px system-ui;max-width:1050px;margin:40px auto;padding:0 20px;color:#111;background:white}img{width:100%;height:auto}figure{margin:45px 0}figcaption{font-size:16px;line-height:1.5}code{word-break:break-all}.status{padding:18px;background:#edf2f7}</style>',
        '<h1>Decision retrieval: evidence gallery</h1>',
        '<p class="status">' + html.escape(manifest['publication_status'] + ' · ' + run_id + ' · ' + manifest['cohort']) + '</p>']
    for stem, title, category in CFG.asset_specs:
        item = {'stem': stem, 'title': title, 'category': category, 'run_id': run_id,
            'cohort': manifest['cohort'], 'status': 'not_generated', 'path': None,
            'reason': 'Required genuine run or complete paired observations are unavailable.'}
        if analysis and analysis['status'] != 'complete' and stem in (
                'graph_03_retrieval_accuracy', 'graph_04_quality_latency_tradeoff', 'graph_05_primary_decision_model_comparison'):
            item['reason'] = 'Incomplete four-option pilot; comparative accuracy/latency inference is unavailable. Unattempted outcomes are not fabricated.'
        if stem not in figures:
            captions += ['## ' + stem, '', 'Not generated: ' + item['reason'], '']
            gallery += ['<h2>' + html.escape(title) + '</h2><p>Not generated: ' + item['reason'] + '</p>']
            manifest['assets'].append(item)
            continue
        fig, values = figures[stem]
        special_source = runs.get('demo') if category == 'demo' else runs.get('pilot') if category == 'pilot' else None
        caption, alt = asset_caption(stem, source, analysis, special_source)
        provenance = special_source or source
        item.update({'status': 'generated', 'reason': None, 'path': stem + '.png',
            'json_path': stem + '.json', 'csv_path': stem + '.csv',
            'run_id': provenance.get('run_id', run_id),
            'cohort': provenance.get('cohort', 'conceptual or corpus-wide'),
            'caption': caption, 'alt_text': alt,
            'intervals': analysis['bootstrap'].get('method') if analysis and category in ('performance', 'paired') else None,
            'source_data_columns': list(values) if isinstance(values, dict) else None})
        fig.savefig(folder / item['path'], dpi=CFG.export_dpi)
        png_header = (folder / item['path']).read_bytes()[:24]
        item['width_pixels'] = int.from_bytes(png_header[16:20], 'big')
        item['height_pixels'] = int.from_bytes(png_header[20:24], 'big')
        sidecar = {'run_id': item['run_id'], 'cohort': item['cohort'], 'data': values, 'data_sha256': sha256(values)}
        atomic_json(folder / item['json_path'], sidecar)
        flatten_sidecar(values).to_csv(folder / item['csv_path'], index=False)
        for key, hash_key in (('path', 'sha256'), ('json_path', 'json_sha256'), ('csv_path', 'csv_sha256')):
            item[hash_key] = hashlib.sha256((folder / item[key]).read_bytes()).hexdigest()
        captions += ['## ' + stem, '', '**Caption:** ' + caption, '', '**Alt text:** ' + alt, '']
        gallery += ['<figure id="' + stem + '"><img loading="lazy" src="' + item['path'] + '" alt="' + html.escape(alt, quote=True) + '"><figcaption><strong>' + html.escape(stem) + '</strong> — ' + html.escape(caption) + '</figcaption></figure>']
        manifest['assets'].append(item)
        if show:
            display(Markdown('### ' + title)); display(fig); display(Markdown(caption))
        plt.close(fig)
    (folder / 'captions.md').write_text('\n'.join(captions))
    (folder / 'index.html').write_text('\n'.join(gallery) + '</html>')
    notes = ['# Evidence for a later post', '', 'This file maps measured findings to assets; it is not an article.', '',
        '- Scope: ' + manifest['publication_status'] + '; ' + manifest['cohort'] + '.',
        '- `graph_01_four_retrieval_paths.png`: the four methods and the specific C/D control.',
        '- `graph_02_corpus_and_evaluation_scope.png`: actual search corpus and frozen sample.',
        '- Claim scope: ' + manifest['claim_scope']]
    if analysis:
        notes += ['- `table_02_four_method_results.png`: ' + '; '.join(r['option'] + ' R@1 ' + percent(r['r1']) +
                f" (attempted {r['attempted']}/{r['scheduled']})" for r in analysis['methods']) + '.',
            '- `table_03_primary_verdict.png`: ' + analysis['primary']['verdict'] + '. ' + analysis['primary']['reason'],
            '- `graph_07_query_latency_distribution.png`: all attempted times; do not use replay runtime as inference speed.']
        if analysis['kind'] != 'evaluation':
            notes += ['- No held-out replication claim is supported. These measurements use training data only; the validation comparison remains unfulfilled.']
    if runs.get('pilot', {}).get('projection'):
        notes += ['- `graph_09_laptop_feasibility.png`: ' + stable_json(runs['pilot']['projection'])]
    notes += ['- Limitations: three-page/stateless adaptation; no Jev, routing, memory, dense baseline, or answer generation; backend caching may differ.',
              '- Scores are adapter-specific ranking values. Do not compare their absolute scales or call generated Qwen scores probabilities.']
    (folder / 'post_notes.md').write_text('\n'.join(notes) + '\n')
    atomic_json(folder / 'manifest.json', manifest)
    validation = validate_exports(folder, manifest)
    original_verification = EMBEDDED_DATA.get('delivery_verification')
    visual_hashes = {i['filename']: i['sha256'] for i in (original_verification or {}).get('visual_QA', {}).get('images', [])}
    current_hashes = {i['path']: i['sha256'] for i in manifest['assets'] if i['status'] == 'generated'}
    visual_matches = bool(visual_hashes) and visual_hashes == current_hashes
    receipt = {'status': validation['status'], 'mode': CFG.mode, 'generated_at': utc_now(),
        'run_id': run_id, 'experiment_status': source.get('status'), 'publication_status': manifest['publication_status'],
        'software': software_versions(), 'offline_checks': CHECKS, 'export_validation': validation,
        'request_evidence_validation': EVIDENCE_CHECKS,
        'data_audit': DATA_AUDIT, 'numerical_receipt': manifest['numerical_receipt'],
        'timing_audit': analysis.get('timing_audit') if analysis else None,
        'original_inference_captured_at': source.get('started_at'),
        'replay_note': 'Metrics and figures recomputed. Historical inference timings reproduced, not remeasured.',
        'visual_QA': 'passed: generated image hashes match the visually inspected delivery' if visual_matches else
            'Separate visual inspection required; structural validation is not visual validation.',
        'original_delivery_verification': original_verification,
        'memory_note': 'Python tracemalloc is not total process/GPU memory; residency snapshots separately report allocated model VRAM.'}
    atomic_json(folder / 'reproducibility_receipt.json', receipt)
    if analysis:
        atomic_json(folder / 'analysis.json', {k: v for k, v in analysis.items() if k != 'rows'})
        pd.DataFrame([{k: r[k] for k in ('option', 'query_id', 'dialogue_id', 'domain',
            'status', 'seconds', 'r1', 'r5', 'r10', 'text_r1', 'text_r5', 'text_r10',
            'gold_page_retained', 'failure_class')} for r in analysis['rows']]).to_csv(folder / 'query_results.csv', index=False)
    return folder, manifest, receipt

# %%
# FIGURES_RUN: same Matplotlib objects are displayed inline and saved for the post.
ASSET_DIR, ASSET_MANIFEST, RECEIPT = export_publication(RUNS, ANALYSIS)
display(Markdown(f"**Exports verified:** {RECEIPT['export_validation']['generated_images']} figures/tables. "
    f"Gallery: `{ASSET_DIR / 'index.html'}`. Status: **{RECEIPT['publication_status']}**."))
display(pd.DataFrame([{'asset': a['stem'], 'status': a['status'], 'source_run': a['run_id']}
                      for a in ASSET_MANIFEST['assets']]))
display(Markdown('**Interpretation boundary.** Exact replay validates the evidence trail. It cannot establish model correctness, '
    'general decision-model superiority, or reproduction of the paper’s hosted execution. A complete local result applies '
    'to these particular trained models, this frozen subset, and this serving setup. A failed gate or partial run remains visible.'))
