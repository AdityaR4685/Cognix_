"""Single production cache, unchanged four-agent audit, compact scientific export."""
import hashlib,json,os,subprocess,sys,threading,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));OUT=Path(__file__).parent
from cognix.adapters.carla import cache_builder as cb
from cognix.adapters.carla.normality import ensemble_to_uncertainty,_bernoulli_entropy
from cognix.adapters.carla.real_agents import RealCameraAgent,RealSegAgent,RealGNSSAgent,RealIMUAgent
config=json.loads((OUT/'frozen_acquisition_config.json').read_text());stage=Path(config['staging_root'])
manifest_raw=(OUT/'frozen_dataset_manifest.json').read_bytes();manifest=json.loads(manifest_raw);manifest_hash=hashlib.sha256(manifest_raw).hexdigest()
assert manifest_hash==(OUT/'frozen_dataset_manifest.sha256').read_text().strip()
assert len(manifest['FIT_NORMAL'])==8 and len(manifest['CAL_NORMAL'])==2
assert all(s['strict_train_loader_validated'] and s['complete'] for s in manifest['scenarios'])
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for c in iter(lambda:f.read(8<<20),b''):h.update(c)
 return h.hexdigest()
for rel,h in config['source_hashes'].items():assert sha(ROOT/rel)==h,rel
view=stage/'logical_train10_v1';cache=stage/'cache_train10_v1';handoff=stage/'kaggle_compact_train10_v1'
# The junction view is created by the orchestration PowerShell host. Verify
# its frozen targets here rather than launching another PowerShell runtime.
for entry in manifest['scenarios']:
 logical=view/'train'/entry['scenario_id']
 assert logical.is_dir() and logical.resolve()==Path(entry['actual_path']).resolve(),str(logical)
state={'phase':'starting','normal_camera_rows':0,'normal_seg_rows':0,'scenario_id':None,'pseudo_scenario':None,'started_monotonic':time.monotonic()};lock=threading.Lock();stop=threading.Event()
def log(kind,**kwargs):
 print(json.dumps({'kind':kind,**kwargs}),flush=True)
def progress():
 while not stop.wait(10):
  p={**state,'elapsed_s':round(time.monotonic()-state['started_monotonic'],1)};p.pop('started_monotonic')
  tmp=stage/'pipeline_progress.json.tmp';tmp.write_text(json.dumps(p,indent=2));os.replace(tmp,stage/'pipeline_progress.json')
threading.Thread(target=progress,daemon=True).start()
def dist(x):
 x=np.asarray(x,float)
 return {'min':float(x.min()),'p10':float(np.percentile(x,10)),'median':float(np.median(x)),'p90':float(np.percentile(x,90)),'max':float(x.max())}
def write_json(path,data):path.write_text(json.dumps(data,indent=2),encoding='utf-8')
try:
 state['phase']='cache_preprocessing';build_start=time.monotonic()
 part=cb.TrainCalPartition(tuple(manifest['FIT_NORMAL']),tuple(manifest['CAL_NORMAL']),2026,'explicit')
 if (cache/'carla_train_cache.npz').exists():
  a,cm=cb.load_cache(cache);assert cb._content_sha256(cache/'carla_train_cache.npz')==cm['cache_content_sha256']
  assert cm['partition']['TRAIN_NORMAL']==list(part.train_normal) and cm['partition']['CAL_NORMAL']==list(part.cal_normal)
  build_s=json.loads((OUT/'cache_result.json').read_text())['preprocessing_wall_s'] if (OUT/'cache_result.json').exists() else None;log('existing_own_cache_reused',content_sha256=cm['cache_content_sha256'])
 else:
  # Observability wrappers only: every invocation delegates unchanged to the
  # original production extractor. No features/inputs/recipes are altered.
  original_camera,original_seg=cb.extract_camera_features,cb.extract_seg_features
  original_scenario,original_pseudo=cb._scenario_rows,cb._pseudo_for_scenario
  def camera(path):
   value=original_camera(path);state['normal_camera_rows']+=1;return value
  def seg(path):
   value=original_seg(path);state['normal_seg_rows']+=1;return value
  def scenario(loader,m,*args):
   state['scenario_id']=m.scenario_id;log('scenario_preprocessing_start',scenario_id=m.scenario_id);return original_scenario(loader,m,*args)
  def pseudo(raw,features,sid,*args):
   state['pseudo_scenario']=sid;log('cal_pseudo_preprocessing_start',scenario_id=sid);return original_pseudo(raw,features,sid,*args)
  cb.extract_camera_features,cb.extract_seg_features= camera,seg
  cb._scenario_rows,cb._pseudo_for_scenario=scenario,pseudo
  try:
   cm=cb.build_cache(view,cache,scenarios=manifest['archive_order_ids'],partition=part,window_length=12,pseudo_recipes=tuple(config['pseudo']['recipes']),max_pseudo_per_tick=1,pseudo_severities=None,pseudo_seed=0)
  finally:
   cb.extract_camera_features,cb.extract_seg_features=original_camera,original_seg
   cb._scenario_rows,cb._pseudo_for_scenario=original_scenario,original_pseudo
  build_s=time.monotonic()-build_start;a,cm=cb.load_cache(cache)
 assert set(a['source_split'])=={'train'} and set(a['scenario_id'])==set(manifest['archive_order_ids'])
 assert cm['cache_schema_version']==config['features']['cache_schema_version']
 assert cb._content_sha256(cache/'carla_train_cache.npz')==cm['cache_content_sha256']
 expected_ticks={s['scenario_id']:s['tick_coverage']['rgb_count'] for s in manifest['scenarios']}
 for sid in manifest['archive_order_ids']:
  sel=a['scenario_id']==sid;assert np.array_equal(a['tick'][sel],np.arange(1,expected_ticks[sid]))
  assert set(a['partition'][sel])=={manifest['builder_partition'][sid]}
 for key in ['camera','seg','gnss','imu','cal_features']:assert np.isfinite(a[key]).all()
 assert set(a['cal_parent_scenario'])<=set(manifest['CAL_NORMAL'])
 assert np.all(a['cal_target_normal']==0)
 assert np.array_equal(a['cal_window_end_tick'],a['cal_parent_tick'])
 assert np.array_equal(a['cal_window_start_tick'],np.maximum(0,a['cal_parent_tick']-11))
 parent_index={(str(s),int(t)):i for i,(s,t) in enumerate(zip(a['scenario_id'],a['tick']))}
 parents=np.array([parent_index[(str(s),int(t))] for s,t in zip(a['cal_parent_scenario'],a['cal_parent_tick'])])
 offsets={'Camera':(0,18),'Seg':(18,47),'GNSS':(47,55),'IMU':(55,65)}
 for name,(lo,hi) in offsets.items():
  intact=a['cal_modality']!=name.lower();assert np.array_equal(a['cal_features'][intact,lo:hi],a[name.lower()][parents[intact]])
 cache_result={'content_sha256':cm['cache_content_sha256'],'file_sha256':sha(cache/'carla_train_cache.npz'),'cache_schema_version':cm['cache_schema_version'],'npz_bytes':(cache/'carla_train_cache.npz').stat().st_size,'manifest_bytes':(cache/'carla_train_cache.json').stat().st_size,'preprocessing_wall_s':round(build_s,3) if build_s is not None else None,'production_builds':1,'clean_rows':len(a['tick']),'FIT_rows':int(np.sum(a['partition']=='TRAIN_NORMAL')),'CAL_rows':int(np.sum(a['partition']=='CAL_NORMAL')),'CAL_pseudo_rows':len(parents),'frozen_dataset_manifest_sha256':manifest_hash,'acquisition_config_sha256':manifest['acquisition_config_sha256'],'consistency_checks_passed':True,'pseudo_audit':cm['pseudo_anomalies'],'cache_path':str(cache)}
 write_json(OUT/'cache_result.json',cache_result);write_json(cache/'dataset_binding.json',cache_result)
 log('production_cache_verified',**cache_result)
 state['phase']='four_agent_calibration'
 fit=a['partition']=='TRAIN_NORMAL';cal=a['partition']=='CAL_NORMAL'
 full=np.column_stack([a[k] for k in ['camera','seg','gnss','imu']]);features=np.vstack([full,a['cal_features']]);n_clean=len(full);n_rows=len(features)
 scores=np.empty((n_rows,4,5),dtype=np.float64)
 probability=np.full((n_rows,4),np.nan);total=probability.copy();aleatoric=probability.copy();epistemic=probability.copy();validity=np.zeros(4,dtype=bool);target_agent=np.ones((n_rows,4),dtype=np.uint8)
 fitted_arrays={};results={};all_start=time.monotonic()
 for j,(name,cls) in enumerate([('Camera',RealCameraAgent),('Seg',RealSegAgent),('GNSS',RealGNSSAgent),('IMU',RealIMUAgent)]):
  state['agent']=name;state['phase']='agent_fit_and_member_scores';lo,hi=offsets[name];agent=cls(seed=42,n_members=5).fit(a[name.lower()][fit])
  S=np.array([agent.predict_normality(row) for row in features[:,lo:hi]]);scores[:,j,:]=S
  selected=np.where(a['cal_modality']==name.lower())[0];target_agent[n_clean+selected,j]=0
  normal_indices=np.flatnonzero(cal);pseudo_indices=n_clean+selected
  M=np.vstack([S[normal_indices],S[pseudo_indices]]);y=np.r_[np.ones(len(normal_indices)),np.zeros(len(pseudo_indices))]
  state['phase']='agent_calibration_audit';agent.fit_calibrator(M,y);audit=agent._calibrator.optimization_info;validity[j]=audit['finite_optimum']
  np.savez(OUT/f'{name.lower()}_cal_member_scores.npz',member_scores=M,target_normal=y)
  r={'FIT_scenarios':8,'FIT_rows':int(fit.sum()),'CAL_scenarios':2,'CAL_normal_rows':len(normal_indices),'CAL_pseudo_rows':len(pseudo_indices),'member_score_distributions':{'normal':dist(M[y==1].ravel()),'pseudo':dist(M[y==0].ravel())},'ensemble_mean_distributions':{'normal':dist(M[y==1].mean(axis=1)),'pseudo':dist(M[y==0].mean(axis=1))},'raw_score_disagreement':{'normal':dist(M[y==1].std(axis=1)),'pseudo':dist(M[y==0].std(axis=1))},'audit':audit,'jensen_descriptive':agent._cal_jensen_diag,'calibration_valid':bool(validity[j]),'mapping_parameters_raw':{'slope':agent._calibrator.a,'intercept':agent._calibrator.b,'scientific_valid':bool(validity[j])},'scientific_probability_and_uq':None,'prediction_semantics':config['prediction_semantics']}
  for k,member in enumerate(agent._ensemble._members):
   fitted_arrays[f'{name.lower()}_{k}_mean']=member._mean
   fitted_arrays[f'{name.lower()}_{k}_precision']=member._prec
   fitted_arrays[f'{name.lower()}_{k}_distance_scale']=np.array(member._d2_scale)
  if validity[j]:
   P=agent._calibrator._sigmoid(agent._calibrator.a*S+agent._calibrator.b)
   q=P.mean(axis=1);T=_bernoulli_entropy(q);A=_bernoulli_entropy(P).mean(axis=1);E=np.maximum(0,T-A)
   probability[:,j]=q;total[:,j]=T;aleatoric[:,j]=A;epistemic[:,j]=E
   # Direct comparison to the canonical production helper, not a new UQ metric.
   for i in np.random.default_rng(0).choice(n_rows,size=min(100,n_rows),replace=False):
    u=ensemble_to_uncertainty(P[i]);assert np.allclose([q[i],T[i],A[i],E[i]],[u.prediction,u.total,u.aleatoric,u.epistemic],rtol=1e-12,atol=1e-14)
   r['scientific_probability_and_uq']={label:{field:dist(values[ids]) for field,values in [('prob_normal',q),('total',T),('aleatoric',A),('epistemic',E)]} for label,ids in [('normal',normal_indices),('pseudo',pseudo_indices)]}
  results[name]=r
  write_json(OUT/'calibration_results.json',{'frozen_dataset_manifest_sha256':manifest_hash,'cache_content_sha256':cm['cache_content_sha256'],'study_role':config['study_role'],'agents':results,'elapsed_s':round(time.monotonic()-all_start,2)})
  log('agent_calibration_complete',agent=name,valid=bool(validity[j]),status=audit['status'],gradient_norm=audit['gradient_norm'],structural=audit['separation']['pooled_members']['classification'])
 state['phase']='compact_kaggle_export'
 # No FIT pseudo generation, graph construction, graph hyperparameters or GPU
 # training. This artifact is the current clean FIT/CAL + CAL-pseudo checkpoint.
 handoff.mkdir(exist_ok=True);scenario_ids=manifest['sorted_canonical_ids'];sid_to_int={s:i for i,s in enumerate(scenario_ids)}
 sample_sid=np.r_[a['scenario_id'],a['cal_parent_scenario']];sample_tick=np.r_[a['tick'],a['cal_parent_tick']].astype(np.int32)
 partition=np.r_[np.where(fit,0,1),np.ones(len(parents))].astype(np.uint8)
 recipes=sorted(set(a['cal_recipe_id']));recipe_to_int={s:i for i,s in enumerate(recipes)}
 modality_to_int={n.lower():i for i,n in enumerate(['Camera','Seg','GNSS','IMU'])}
 payload={'scenario_index':np.array([sid_to_int[str(s)] for s in sample_sid],dtype=np.uint16),'tick':sample_tick,'partition':partition,'features':features,'member_normality_scores':scores,'calibration_valid':validity,'prob_normal':probability,'total':total,'aleatoric':aleatoric,'epistemic':epistemic,'target_normal':np.r_[np.ones(n_clean),np.zeros(len(parents))].astype(np.uint8),'target_normal_per_agent':target_agent,'is_pseudo':np.r_[np.zeros(n_clean),np.ones(len(parents))].astype(bool),'pseudo_parent_scenario_index':np.r_[np.full(n_clean,-1),[sid_to_int[str(s)] for s in a['cal_parent_scenario']]].astype(np.int32),'pseudo_parent_tick':np.r_[np.full(n_clean,-1),a['cal_parent_tick']].astype(np.int32),'recipe_index':np.r_[np.full(n_clean,-1),[recipe_to_int[str(s)] for s in a['cal_recipe_id']]].astype(np.int16),'corrupted_modality_index':np.r_[np.full(n_clean,-1),[modality_to_int[str(s)] for s in a['cal_modality']]].astype(np.int8),'severity':np.r_[np.full(n_clean,np.nan),a['cal_severity']],'pseudo_seed':np.r_[np.full(n_clean,-1),a['cal_seed']].astype(np.int64),'window_start_tick':np.r_[np.maximum(0,a['tick']-11),a['cal_window_start_tick']].astype(np.int32),'window_end_tick':sample_tick}
 assert np.isnan(probability[:,~validity]).all() and np.isnan(epistemic[:,~validity]).all()
 assert all(np.isfinite(x[:,validity]).all() for x in [probability,total,aleatoric,epistemic])
 np.savez_compressed(handoff/'compact_training_checkpoint.npz',**payload)
 np.savez_compressed(handoff/'fitted_oneclass_parameters.npz',**fitted_arrays)
 export={'schema_version':1,'protocol_version':config['protocol_version'],'study_role':config['study_role'],'frozen_dataset_manifest_sha256':manifest_hash,'acquisition_config_sha256':manifest['acquisition_config_sha256'],'cache_content_sha256':cm['cache_content_sha256'],'source_code_hashes':config['source_hashes'],'git_head':config['git_head'],'n_rows':n_rows,'clean_rows':n_clean,'CAL_pseudo_rows':len(parents),'FIT_pseudo_rows':0,'scenario_dictionary':[{'scenario_id':sid,'town':sid.split('/')[0],'partition':'FIT_NORMAL' if sid in manifest['FIT_NORMAL'] else 'CAL_NORMAL','source_hash_bundle_sha256':next(s['source_hash_bundle_sha256'] for s in manifest['scenarios'] if s['scenario_id']==sid)} for sid in scenario_ids],'partition_dictionary':{'0':'FIT_NORMAL','1':'CAL_NORMAL'},'agent_order':['Camera','Seg','GNSS','IMU'],'recipe_dictionary':recipes,'feature_dims':config['features']['dims'],'calibration_audit':{name:r['audit'] for name,r in results.items()},'calibration_parameters_raw':{name:r['mapping_parameters_raw'] for name,r in results.items()},'calibration_status':{name:r['audit']['status'] for name,r in results.items()},'nullable_encoding':'NaN plus per-agent calibration_valid mask; invalid probability/total/aleatoric/epistemic unavailable for scientific use','target_normal_semantics':'constructed observation is clean (1) versus has controlled pseudo corruption (0), not physical safety','target_normal_per_agent_semantics':'only the corrupted modality has target 0; untouched modality blocks have constructed target 1','prediction_semantics':config['prediction_semantics'],'calibration_sampling':'CAL normal observations + modality-matched pseudo; counts recorded per agent','window_scope':'GNSS/IMU causal [max(0,t-11),t]; Camera/Seg instantaneous tick t; stored window bounds describe parent observation provenance','seeds':{'split':2026,'bootstrap':42,'pseudo':0},'raw_data_present':False,'graph_samples_constructed':False,'graph_training_done':False,'future_graph_training_constraint':'FIT-derived pseudo must be generated in a separately authorized stage; present CAL pseudo rows are not graph-training examples','arrays':{k:{'dtype':str(v.dtype),'shape':list(v.shape)} for k,v in payload.items()},'files':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in handoff.iterdir() if p.is_file()}}
 export['feature_blocks']=[{'agent':name,'start_inclusive':lo,'stop_exclusive':hi} for name,(lo,hi) in offsets.items()]
 export['uncertainty_definition']={'total':'H(mean(member_prob_normal))','aleatoric':'mean(H(member_prob_normal))','epistemic':'max(0,total-aleatoric)','entropy_log_base':'natural','validated_agents_only':True}
 export['corrupted_modality_dictionary']={'-1':'clean','0':'Camera','1':'Seg','2':'GNSS','3':'IMU'}
 write_json(handoff/'manifest.json',export)
 export_result={'artifact_path':str(handoff),'total_bytes':sum(p.stat().st_size for p in handoff.iterdir() if p.is_file()),'n_rows':n_rows,'n_graph_training_pseudo_rows':0,'calibration_valid':{n:r['calibration_valid'] for n,r in results.items()},'scientific_invalid_values_null':True,'raw_data_present':False,'export_manifest_sha256':sha(handoff/'manifest.json')}
 write_json(OUT/'kaggle_artifact_result.json',export_result);log('compact_export_complete',**export_result)
 state['phase']='complete'
 write_json(OUT/'pipeline_result.json',{'status':'complete','cache':cache_result,'kaggle':export_result,'total_pipeline_wall_s':round(time.monotonic()-state['started_monotonic'],2),'agent_analysis_wall_s':round(time.monotonic()-all_start,2),'no_calibration_recipe_changes':True,'no_GAT_conformal':True})
finally:
 stop.set();p={**state,'elapsed_s':round(time.monotonic()-state['started_monotonic'],1)};p.pop('started_monotonic');write_json(stage/'pipeline_progress.json',p)
