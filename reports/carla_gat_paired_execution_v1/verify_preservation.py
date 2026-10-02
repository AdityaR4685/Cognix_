"""Read-only byte/hash audit of protected local files and upstream artifacts."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
snapshot=json.loads((HERE/'preservation_snapshot.json').read_text())
def digest(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
changed=[]; count=0
for name,expected in snapshot['workspace_files'].items():
    if digest(ROOT/name)!=expected: changed.append(name)
    count+=1
extra={}
for group in ('upstream','artifact'):
    extra[group]={}
    for name,expected in snapshot[group].items():
        path=ROOT/'reports/carla_gat_preregistration_v1/protocol.json' if name=='protocol.json' else Path(name)
        if not path.is_absolute(): path=ROOT/path
        actual=digest(path); extra[group][name]=actual
        if actual!=expected: changed.append(name)
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
if head!=snapshot['git_head']: changed.append('git_HEAD')
value={'passed':not changed,'protected_workspace_files':count,'upstream_and_graph_hashes':extra,
       'changed':changed,'git_HEAD':head,'scope':'byte hashes only; no CAL/TEST arrays loaded or scientific calculations'}
with (HERE/'active_execution_preservation.json').open('x') as f: json.dump(value,f,sort_keys=True,indent=2)
print(json.dumps({'passed':value['passed'],'protected_workspace_files':count,'upstream_files':len(extra['upstream']),
                  'graph_files':len(extra['artifact']),'changed':changed}))
if changed: raise SystemExit(1)
