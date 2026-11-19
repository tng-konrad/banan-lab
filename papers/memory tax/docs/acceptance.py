"""Fresh-kernel offline report acceptance; run with this project's .venv Python."""
from pathlib import Path
import argparse
import base64
import hashlib
import json
import os
import time

import nbformat
from nbclient import NotebookClient

parser=argparse.ArgumentParser()
parser.add_argument('--project',default=str(Path(__file__).resolve().parents[1]))
parser.add_argument('--cwd',default=None,help='Optional repository-root launch directory')
args=parser.parse_args()
p=Path(args.project).absolute();path=p/'memory_tax_ollama_demo.ipynb'
runtime=p/'.runtime'
for name in ['ipython','jupyter','matplotlib']:(runtime/name).mkdir(parents=True,exist_ok=True)
os.environ.update(IPYTHONDIR=str(runtime/'ipython'),JUPYTER_RUNTIME_DIR=str(runtime/'jupyter'),MPLCONFIGDIR=str(runtime/'matplotlib'))
os.environ['JUPYTER_PATH']=str(p/'.venv/share/jupyter')
nb=nbformat.read(path,as_version=4)
guard=nbformat.v4.new_code_cell("def _network_blocked(*a, **k):\n    raise RuntimeError('Network blocked during acceptance replay')\nimport socket, urllib.request\nsocket.socket.connect = _network_blocked\nsocket.create_connection = _network_blocked\nurllib.request.urlopen = _network_blocked")
nb.cells.insert(4,guard);started=time.monotonic()
NotebookClient(nb,timeout=300,kernel_name='memory-tax',resources={'metadata':{'path':args.cwd or str(p)}}).execute()
nb.cells.remove(guard)
nbformat.write(nb,path)
manifest=json.loads((p/'graphs/manifest.json').read_text());metrics=json.loads((p/'graphs/metrics.json').read_text())
assert [r['Correct'] for r in metrics['performance']]==[45,30,43,0,49]
assert [(r['benefit_count'],r['tax_count']) for r in metrics['paired']]==[(2,17),(4,6)]
assert [a['number'] for a in manifest['artifacts']]==[f'{i:02d}' for i in range(1,36)]
inline=[base64.b64decode(o['data']['image/png']) for c in nb.cells if c.cell_type=='code' for o in c.outputs if 'image/png' in o.get('data',{})]
assert len(inline)==35
for a,b in zip(manifest['artifacts'],inline):
 assert (p/'graphs'/a['file']).read_bytes()==b
 for f in a['sidecars']:assert (p/'graphs'/f).is_file()
assert not any(o.output_type=='error' for c in nb.cells if c.cell_type=='code' for o in c.outputs)
result={'passed':True,'project':str(p),'launch_cwd':args.cwd or str(p),'seconds':time.monotonic()-started,'network':'Python socket connect/create_connection and urllib blocked in kernel','assets':len(inline),'inline_equals_saved':True,'metrics_preserved':True,'record_equality':'asserted in notebook against checksum-validated immutable original','historical_run_id':manifest['inference_run_id'],'kernel':'memory-tax','export_timestamp':manifest['execution_timestamp']}
(p/'docs').mkdir(exist_ok=True)
(p/'docs/acceptance-result.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
