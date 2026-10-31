import json
from run_live import load_runtime

s = load_runtime()
r = json.load(open('results/gate.json'))['requests'][0]
b = {'state': r['request']['state'], 'questions': r['request']['questions']}
prompts = s['nimble_rendered_prompts'](b)
for variant in ('original', 'single', 'literal'):
    values = [p if variant == 'original' else p.replace(r'\\u003c', r'\u003c').replace(r'\\u003e', r'\u003e')
              if variant == 'single' else p.replace(r'\\u003c', '<').replace(r'\\u003e', '>') for p in prompts]
    print(variant, [s['token_count'](p) for p in values])
print('observed', r['attempts'][0]['response']['usage'])
