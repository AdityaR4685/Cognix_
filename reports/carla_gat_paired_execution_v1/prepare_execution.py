"""Seal engineering sources and snapshot existing inputs without changing them."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
def digest(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

sources=['sitecustomize.py','execution_observer.py','execute_full.py','aggregate_results.py']
files={name:digest(HERE/name) for name in sources}
(HERE/'engineering_source_hashes.json').write_text(json.dumps(files,sort_keys=True,indent=2)+'\n')
snapshot={str(p.relative_to(ROOT)).replace('\\','/'):digest(p)
          for directory in ['cognix','tests','reports'] for p in (ROOT/directory).rglob('*')
          if p.is_file() and '__pycache__' not in p.parts and not p.suffix=='.pyc'
          and not p.is_relative_to(HERE) and 'carla_gat_paired_runs_v1' not in p.parts
          and 'carla_gat_paired_raw_runs_v1' not in p.parts}
old=json.loads((ROOT/'reports/carla_kaggle_validation_bundle_v1/preservation_snapshot.json').read_text())
snapshot_record={'workspace_files':snapshot,'artifact':old['artifact'],'upstream':old.get('upstream',{}),
                 'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
(HERE/'preservation_snapshot.json').write_text(json.dumps(snapshot_record,sort_keys=True,indent=2)+'\n')
archive=HERE/'engineering_execution_v1.zip'
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
    for name in sources+['engineering_source_hashes.json']: z.write(HERE/name,name)
sha=digest(archive)
(HERE/'engineering_execution_v1.zip.sha256').write_text(sha+'  '+archive.name+'\n')
print(json.dumps({'archive_sha256':sha,'bytes':archive.stat().st_size,'sources':files,
                  'protected_workspace_files':len(snapshot),'base64':base64.b64encode(archive.read_bytes()).decode()}))
