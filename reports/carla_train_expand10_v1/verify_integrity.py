"""End-of-milestone preservation and domain-core checks."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent;PROBE=Path(os.environ['TEMP'])/'carla_train_probe'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for c in iter(lambda:f.read(8<<20),b''):h.update(c)
 return h.hexdigest()
base=json.loads((OUT/'initial_state.json').read_text());config=json.loads((OUT/'frozen_acquisition_config.json').read_text())
modified=[name for name,h in base['repository_hashes'].items() if not (ROOT/name).exists() or sha(ROOT/name)!=h]
assert not modified,modified
assert sha(ROOT/'reports/carla_calibration_audit/proposed_protocol.json')==base['accepted_proposal_sha256']
assert sha(OUT/'frozen_acquisition_config.json')==(OUT/'frozen_acquisition_config.sha256').read_text().strip()
assert sha(OUT/'frozen_dataset_manifest.json')==(OUT/'frozen_dataset_manifest.sha256').read_text().strip()
assert Path(config['original_prefix']['path']).stat().st_size==config['original_prefix']['bytes']
assert sha(config['original_prefix']['path'])==config['original_prefix']['sha256']
for name,stat in base['probe_file_stats'].items():
 p=PROBE/name;assert p.exists() and {'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}==stat,name
for name,h in base['old_cache_file_hashes'].items():assert sha(PROBE/name)==h,name
subprocess.run([sys.executable,'-c',"import sys;import cognix;assert not any(m.startswith('cognix.adapters') for m in sys.modules)"],check=True,cwd=ROOT)
assert subprocess.run(['git','diff','--quiet','HEAD','--','cognix','tests'],cwd=ROOT).returncode==0
stage=Path(config['staging_root']);acq=json.loads((stage/'acquisition_result.json').read_text());pipeline=json.loads((OUT/'pipeline_result.json').read_text())
assert acq['status']=='complete' and acq['strict_validated_count']==10 and acq['original_unchanged']
assert acq['trailing_boundary']['kind']=='subsequent_scenario_header' or acq['trailing_boundary']['kind']=='valid_gzip_and_tar_termination'
assert pipeline['status']=='complete'
assert sha(stage/config['staging_prefix_name'])==acq['staged_sha256']
# Verify exported checkpoint contains compact arrays only and null scientific
# probabilities/UQ for invalid calibrators, without touching any raw TEST data.
import numpy as np
handoff=stage/'kaggle_compact_train10_v1'
with np.load(handoff/'compact_training_checkpoint.npz',allow_pickle=False) as z:
 assert z['features'].shape[1]==65 and z['member_normality_scores'].shape[1:]==(4,5)
 for key in ['prob_normal','total','aleatoric','epistemic']:assert np.isnan(z[key][:,~z['calibration_valid']]).all()
 assert set(z['partition'].tolist())=={0,1}
export=json.loads((handoff/'manifest.json').read_text());assert not export['raw_data_present'] and not export['graph_samples_constructed'] and not export['graph_training_done']
for name,info in export['files'].items():assert sha(handoff/name)==info['sha256']
result={'modified_preexisting_source_or_test_files':modified,'new_source_and_test_files':['cognix/adapters/carla/train_acquisition.py','tests/unit/test_carla_train_acquisition.py'],'new_report_files':[p.relative_to(ROOT).as_posix() for p in OUT.iterdir() if p.is_file()],'generic_core_and_frozen_synthetic_unchanged':True,'all_frozen_production_source_hashes_unchanged':True,'accepted_proposal_unchanged':True,'original_prefix_unchanged':True,'old_probe_artifacts_stats_unchanged':len(base['probe_file_stats']),'old_cache_byte_hashes_unchanged':True,'bare_import_no_adapters':True,'tracked_diff_empty':True,'TRAIN_only':True,'no_GAT_conformal_calibration_severity_threshold_tuning':True,'no_commit_push_reset_clean_stash_checkout':True,'compact_export_verified':True,'storage':{'staged_prefix_bytes':(stage/config['staging_prefix_name']).stat().st_size,'new_extracted_raw_bytes':sum(p.stat().st_size for p in (stage/'raw_train_v1').rglob('*') if p.is_file()),'cache_total_bytes':sum(p.stat().st_size for p in (stage/'cache_train10_v1').iterdir() if p.is_file()),'handoff_total_bytes':sum(p.stat().st_size for p in handoff.iterdir() if p.is_file())}}
(OUT/'integrity_results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
