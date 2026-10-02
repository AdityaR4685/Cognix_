"""Freeze N=20 before network access; reuse and preserve accepted N=10."""
import hashlib, json, os, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).parent
sys.path.insert(0,str(ROOT))
N10=ROOT/'reports/carla_train_expand10_v1'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for c in iter(lambda:f.read(8<<20),b''):h.update(c)
 return h.hexdigest()
def exclusive(p,raw):
 if p.exists():assert p.read_bytes()==raw,str(p)
 else:
  with p.open('xb') as f:f.write(raw)
old=json.loads((N10/'frozen_acquisition_config.json').read_text())
dataset=json.loads((N10/'frozen_dataset_manifest.json').read_text())
acq=json.loads((N10/'acquisition_result.json').read_text())
assert sha(N10/'frozen_acquisition_config.json')==(N10/'frozen_acquisition_config.sha256').read_text().strip()
assert sha(N10/'frozen_dataset_manifest.json')==(N10/'frozen_dataset_manifest.sha256').read_text().strip()
assert acq['status']=='complete' and acq['strict_validated_count']==10
assert json.loads((N10/'pipeline_result.json').read_text())['status']=='complete'
prefix=Path(acq['staged_prefix']); assert prefix.stat().st_size==acq['staged_bytes'] and sha(prefix)==acq['staged_sha256']
for rel,h in old['source_hashes'].items():assert sha(ROOT/rel)==h,rel
config=dict(old)
config.update(protocol_version='carla_train_expand20_v1',target_complete_train_scenarios=20,
 selection_rule='first 20 complete official TRAIN scenarios in parsed archive order; whole scenarios; selection independent of model results',
 original_prefix={'path':str(prefix),'bytes':prefix.stat().st_size,'sha256':acq['staged_sha256']},
 staging_root=str(Path(os.environ['TEMP'])/'carla_train_expand20'),staging_prefix_name='train_prefix_expand20.tar.gz',
 study_role='N=20 is the final planned pre-GAT TRAIN-development expansion; invalid calibration leads to a methodology decision, not automatic acquisition beyond N=20',
 final_planned_train_development_target=20,no_automatic_further_expansion=True,
 preserved_N10={'report_sha256':sha(N10/'report.md'),'dataset_manifest_sha256':sha(N10/'frozen_dataset_manifest.json'),
 'acquisition_config_sha256':sha(N10/'frozen_acquisition_config.json'),'archive_order_ids':dataset['archive_order_ids'],
 'prefix_boundary':acq['trailing_boundary'],'remote_validator':acq['remote_validator'],'remote_total_bytes':acq['remote_total_bytes']},
 existing_scenario_paths={s['scenario_id']:s['actual_path'] for s in dataset['scenarios']},
 parser_recovery='one sequential replay of the preserved N10 staging copy through one live gzip/TAR decoder; the prior process has ended and its zlib state cannot safely be serialized',
 resource_limits={'max_staged_compressed_bytes':70_000_000_000,'minimum_free_bytes_to_start':100_000_000_000,
 'compressed_read_chunk_bytes':262144,'max_decompressed_chunk_bytes':1048576},
 baseline_tests=505)
config.pop('preferred_later_target',None)
config['split']={**old['split'],'FIT_NORMAL':15,'CAL_NORMAL':5,'recompute_once_from_complete_N20_list':True,'do_not_manually_extend_N10_membership':True}
raw=(json.dumps(config,sort_keys=True,indent=2)+'\n').encode()
exclusive(OUT/'frozen_acquisition_config.json',raw)
exclusive(OUT/'frozen_acquisition_config.sha256',(hashlib.sha256(raw).hexdigest()+'\n').encode())
# All prior source/tests and report artifacts get byte hashes. Large existing
# raw trees also get immutable size/mtime inventories; cache/export bytes are hashed.
probe=Path(old['original_prefix']['path']).parent; oldstage=Path(old['staging_root'])
repo_files=[p for d in ['cognix','tests'] for p in (ROOT/d).rglob('*') if p.is_file() and '__pycache__' not in str(p)]
reports=[p for p in (ROOT/'reports').rglob('*') if p.is_file() and not p.is_relative_to(OUT) and '__pycache__' not in str(p)]
prior_files=[p for root in [probe,oldstage] for p in root.rglob('*') if p.is_file() and '__pycache__' not in str(p) and 'logical_train10_v1' not in p.parts]
small_artifact_roots=[probe/n for n in ['cache_a','cache_b','cache_v2_a','cache_v2_b']]+[oldstage/'cache_train10_v1',oldstage/'kaggle_compact_train10_v1']
original=Path(old['original_prefix']['path']);assert sha(original)==old['original_prefix']['sha256']
state={'git_status':subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),
 'repository_hashes':{p.relative_to(ROOT).as_posix():sha(p) for p in repo_files},
 'prior_report_hashes':{p.relative_to(ROOT).as_posix():sha(p) for p in reports},
 'prior_file_stats':{str(p):{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in prior_files},
 'prior_cache_export_hashes':{str(p):sha(p) for d in small_artifact_roots for p in d.rglob('*') if p.is_file()},
 'original_3GB_prefix':old['original_prefix'],'N10_prefix':config['original_prefix'],
 'tracked_state':subprocess.check_output(['git','diff','HEAD','--','cognix','tests'],cwd=ROOT,text=True)}
assert not state['tracked_state']
exclusive(OUT/'initial_state.json',(json.dumps(state,indent=2)+'\n').encode())

def generate(name,text):exclusive(OUT/name,text.encode('utf-8'))
text=(N10/'acquire.py').read_text().replace('expand10','expand20').replace('target=10','target=20')
text=text.replace("oldroot=ORIGINAL.parent/'dataset'\nreuse={sid:oldroot/'train'/sid for sid in ['Town01/scenario-1','Town01/scenario-10']}","reuse={sid:Path(path) for sid,path in CONFIG['existing_scenario_paths'].items()}")
text=text.replace("network_started=None; response_total=None; remote_validator=None;", "network_started=None; response_total=CONFIG['preserved_N10']['remote_total_bytes']; remote_validator=CONFIG['preserved_N10']['remote_validator'];")
text=text.replace("assert not os.path.samefile(ORIGINAL,PREFIX)\n#", "assert not PREFIX.is_symlink() and not os.path.samefile(ORIGINAL,PREFIX)\n#")
generate('acquire.py',text)
text=(N10/'freeze_dataset.py').read_text().replace('8/2','15/5').replace("==10","==20").replace("range(1,11)","range(1,21)").replace("==8","==15").replace("==2\n","==5\n")
text=text.replace("'preferred_later_target':config['preferred_later_target'],","'final_planned_train_development_target':20,'no_automatic_further_expansion':True,")
text=text.replace("mapping=part.mapping()", "assert ids[:10]==config['preserved_N10']['archive_order_ids']\nfrom collections import Counter\ntown_counts=dict(Counter(r['town'] for r in records))\nmapping=part.mapping()")
text=text.replace("'no_model_results_used':True", "'town_counts':town_counts,'unique_towns':len(town_counts),'N10_role_changes':{sid:{'old':('FIT_NORMAL' if sid in json.loads((ROOT/'reports/carla_train_expand10_v1/frozen_dataset_manifest.json').read_text())['FIT_NORMAL'] else 'CAL_NORMAL'),'new':('FIT_NORMAL' if mapping[sid]=='TRAIN_NORMAL' else 'CAL_NORMAL')} for sid in ids[:10]},'no_model_results_used':True")
generate('freeze_dataset.py',text)
generate('create_logical_view.ps1',(N10/'create_logical_view.ps1').read_text())
text=(N10/'build_and_analyze.py').read_text().replace('train10','train20').replace("==8 and len(manifest['CAL_NORMAL'])==2","==15 and len(manifest['CAL_NORMAL'])==5")
text=text.replace("'FIT_scenarios':8", "'FIT_scenarios':15").replace("'CAL_scenarios':2", "'CAL_scenarios':5")
text=text.replace("'raw_score_disagreement':{'normal':dist(M[y==1].std(axis=1)),'pseudo':dist(M[y==0].std(axis=1))},", "'raw_score_disagreement':{'normal':dist(M[y==1].std(axis=1)),'pseudo':dist(M[y==0].std(axis=1))},'calibrated_probability_disagreement':None,")
text=text.replace("  if validity[j]:", "  if name=='GNSS':\n   from gnss_diagnostics import analyze_gnss\n   r['pre_exponential_diagnostics']=analyze_gnss(agent,features[np.r_[normal_indices,pseudo_indices],lo:hi],M,y)\n   write_json(OUT/'gnss_distance_diagnostics.json',r['pre_exponential_diagnostics'])\n  if validity[j]:")
text=text.replace("   q=P.mean(axis=1);", "   r['calibrated_probability_disagreement']={label:dist(P[ids].std(axis=1)) for label,ids in [('normal',normal_indices),('pseudo',pseudo_indices)]}\n   q=P.mean(axis=1);")
text=text.replace("'schema_version':1,'protocol_version':config['protocol_version']", "'schema_version':2,'protocol_version':config['protocol_version']")
generate('build_and_analyze.py',text)
for name in ['acquire.py','freeze_dataset.py','build_and_analyze.py']:compile((OUT/name).read_text(),str(OUT/name),'exec')
assert 'target=20' in (OUT/'acquire.py').read_text()
assert "len(part.cal_normal)==5" in (OUT/'freeze_dataset.py').read_text()
print('FROZEN N20 CONFIG SHA256',hashlib.sha256(raw).hexdigest(),flush=True)
print('PRESERVED N10 PREFIX',json.dumps(config['original_prefix']),flush=True)
print('NO NETWORK REQUEST HAS BEEN MADE',flush=True)
