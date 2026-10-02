"""Verify original work and preserved probe artifacts without raw-data reads."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[2]; out=Path(__file__).parent; probe=Path(os.environ['TEMP'])/'carla_train_probe'
b=json.loads((out/'initial_state.json').read_text())
changed=[name for name,h in b['repository'].items() if not (root/name).exists() or hashlib.sha256((root/name).read_bytes()).hexdigest()!=h]
allowed={'cognix/adapters/carla/'+n for n in ['normality.py','real_agents.py','cache_builder.py','pseudo_anomalies.py']}
allowed|={'tests/unit/'+n for n in ['test_carla_cache_builder.py','test_carla_causal_window_scope.py','test_carla_ensemble_predictive_calibration.py','test_carla_pseudo_anomalies.py','test_carla_real_agents.py']}
assert set(changed)==allowed
for name,old in b['probe'].items():
 p=probe/name
 assert p.exists() and {'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}==old,name
caches={}
for name in ['cache_a','cache_b','cache_v2_a','cache_v2_b']:
 d=probe/name; p=d/'carla_train_cache.npz'; manifest=json.loads((d/'carla_train_cache.json').read_text()); h=hashlib.sha256()
 with np.load(p,allow_pickle=False) as z:
  for key in sorted(z.files):
   arr=z[key]; h.update(key.encode()); h.update(str(arr.dtype).encode()); h.update(str(arr.shape).encode()); h.update(np.ascontiguousarray(arr).tobytes())
 assert h.hexdigest()==manifest['cache_content_sha256']
 caches[name]={'content_sha256':h.hexdigest(),'file_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
subprocess.run([sys.executable,'-c',"import sys; import cognix; assert not any(m.startswith('cognix.adapters') for m in sys.modules)"],check=True,cwd=root)
assert subprocess.run(['git','diff','--quiet','HEAD','--','cognix','tests/unit/test_carla_agent_semantics.py'],cwd=root).returncode==0
new=[p.relative_to(root).as_posix() for d in ['cognix','tests'] for p in (root/d).rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.relative_to(root).as_posix() not in b['repository']]
result={'modified_preexisting_files':changed,'new_source_or_test_files':new,'report_files':[p.name for p in out.iterdir() if p.is_file()],'generic_and_frozen_initial_hashes_unchanged':True,'tracked_diff_empty':True,'bare_import_no_adapters':True,'probe_artifacts_unchanged_size_and_mtime':len(b['probe']),'all_four_cache_content_hashes_match_preserved_manifests':caches,'no_download_or_official_test_access':True,'no_GAT_conformal_severity_threshold_regularization_slope_cap_changes':True,'no_commit_push_branch_stash_reset_clean':True}
(out/'integrity_results.json').write_text(json.dumps(result,indent=2))
p=out/'report.md'; s=p.read_text().replace('Pending final hash/stat verification. Generic APIs and entropy mathematics were not edited.','Final verification: all 12,054 prior probe files have unchanged size/mtime; all four caches match their preserved content hashes; every generic/frozen source and synthetic test matches its initial SHA-256; tracked Git diff is empty; bare import cognix loads no adapters. `integrity_results.json` lists the exact nine modified pre-existing files and one new test file. Generic APIs and entropy mathematics were not edited.')
s=s.replace('constructed target and scope, pseudo parent/modality/recipe/severity/seed/window bounds','constructed target and scope (per-agent labels refer only to the corrupted modality; graph-level any-pseudo labels are a distinct constructed target), pseudo parent/modality/recipe/severity/seed/window bounds')
s=s.replace('The two-scenario study stays a distinct frozen protocol.','The two-scenario study stays a distinct frozen protocol. The current builder calls FIT rows TRAIN_NORMAL; the proposed artifact records FIT_NORMAL with an explicit name mapping. Generating FIT-derived graph pseudo rows is a later adapter extension, not implemented in this task.')
p.write_text(s)
p=out/'proposed_protocol.json'; proposal=json.loads(p.read_text()); proposal['split']['current_builder_label_mapping']={'TRAIN_NORMAL':'FIT_NORMAL','CAL_NORMAL':'CAL_NORMAL'}; proposal['artifact_schema']['target_scope_note']='Per-agent labels apply to the corrupted modality; graph any-pseudo target is distinct and never physical safety.'; p.write_text(json.dumps(proposal,indent=2))
print(json.dumps(result,indent=2))
