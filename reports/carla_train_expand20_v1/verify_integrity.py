"""End-of-milestone integrity audit, preserving every older checkpoint."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
 return h.hexdigest()
base=json.loads((OUT/'initial_state.json').read_text());config=json.loads((OUT/'frozen_acquisition_config.json').read_text())
for group in ['repository_hashes','prior_report_hashes']:
 for rel,h in base[group].items():assert sha(ROOT/rel)==h,rel
for path,stat in base['prior_file_stats'].items():
 p=Path(path);assert {'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}==stat,path
for path,h in base['prior_cache_export_hashes'].items():assert sha(path)==h,path
for key in ['original_3GB_prefix','N10_prefix']:
 r=base[key];assert Path(r['path']).stat().st_size==r['bytes'] and sha(r['path'])==r['sha256']
for name in ['frozen_acquisition_config','frozen_compute_reuse_policy','frozen_dataset_manifest']:
 assert sha(OUT/(name+'.json'))==(OUT/(name+'.sha256')).read_text().strip()
assert subprocess.run(['git','diff','--quiet','HEAD','--','cognix','tests'],cwd=ROOT).returncode==0
subprocess.run([sys.executable,'-c',"import sys;import cognix;assert not any(m.startswith('cognix.adapters') for m in sys.modules)"],cwd=ROOT,check=True)
stage=Path(config['staging_root']);acq=json.loads((OUT/'acquisition_result.json').read_text());pipeline=json.loads((OUT/'pipeline_result.json').read_text())
assert acq['status']=='complete' and acq['strict_validated_count']==20 and acq['original_unchanged'] and pipeline['status']=='complete'
assert sha(stage/config['staging_prefix_name'])==acq['staged_sha256']
manifest=json.loads((OUT/'frozen_dataset_manifest.json').read_text());assert len(manifest['FIT_NORMAL'])==15 and len(manifest['CAL_NORMAL'])==5
reuse=json.loads((OUT/'compute_reuse_results.json').read_text());assert reuse['safe_reuse'] and reuse['assembled_clean_features_byte_identical']
assert set(reuse['clean_reused_scenarios'])==set(config['preserved_N10']['archive_order_ids']) and reuse['clean_reused_rows']==29990
assert reuse['new_clean_camera_extractor_calls']==reuse['new_clean_seg_extractor_calls']==reuse['new_clean_feature_rows'] and reuse['old_clean_scenario_raw_feature_extraction_calls']==0
handoff=stage/'kaggle_compact_train20_v1';export=json.loads((handoff/'manifest.json').read_text())
for name,info in export['files'].items():assert sha(handoff/name)==info['sha256']
assert sha(handoff/'manifest.json')==pipeline['kaggle']['export_manifest_sha256']
with np.load(handoff/'compact_training_checkpoint.npz',allow_pickle=False) as z:
 assert z['features'].shape[1]==65 and z['member_normality_scores'].shape[1:]==(4,5)
 for key in ['prob_normal','total','aleatoric','epistemic']:assert np.isnan(z[key][:,~z['calibration_valid']]).all()
 assert set(z['partition'])=={0,1} and np.all(z['partition'][z['is_pseudo']]==1)
assert not export['raw_data_present'] and not export['graph_samples_constructed'] and not export['graph_training_done']
result={'N2_and_N10_reports_byte_hashes_preserved':len(base['prior_report_hashes']),
 'older_raw_and_artifact_file_stats_preserved':len(base['prior_file_stats']),
 'N2_and_N10_cache_export_byte_hashes_preserved':True,'original_3GB_prefix_preserved':True,'N10_prefix_preserved':True,
 'generic_and_frozen_sources_tests_preserved':True,'all_scientific_source_hashes_unchanged':True,
 'bare_import_no_CARLA_adapters':True,'tracked_diff_empty':True,'no_TEST_GAT_conformal_calibration_feature_recipe_tuning_commit_push':True,
 'compact_export_verified':True,'compute_reuse_verified':True,'modified_preexisting_files':[],
 'new_source_and_test_files':[],'new_milestone_files':sorted(p.relative_to(ROOT).as_posix() for p in OUT.rglob('*') if p.is_file()),
 'new_staging_locations':{name:str(stage/name) for name in [config['staging_prefix_name'],'raw_train_v1','scenario_manifests','new_clean_blocks_v1','logical_train20_v1','cache_train20_v1','kaggle_compact_train20_v1']},
 'storage':{'N20_prefix_bytes':(stage/config['staging_prefix_name']).stat().st_size,
 'new_raw_bytes':sum(p.stat().st_size for p in (stage/'raw_train_v1').rglob('*') if p.is_file()),
 'new_clean_blocks_bytes':sum(p.stat().st_size for p in (stage/'new_clean_blocks_v1').iterdir() if p.is_file()),
 'cache_bytes':sum(p.stat().st_size for p in (stage/'cache_train20_v1').iterdir() if p.is_file()),
 'handoff_bytes':sum(p.stat().st_size for p in handoff.iterdir() if p.is_file())}}
(OUT/'integrity_results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
