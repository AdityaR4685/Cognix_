"""Exact cache reuse in N20 orchestration; production science stays unchanged."""
import ast, hashlib, inspect, json
from collections import Counter
from pathlib import Path
import numpy as np
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class VerifiedReuse:
 def __init__(self,root,config,manifest=None):
  from cognix.adapters.carla import cache_builder as cb
  self.cb=cb;self.root=Path(root);self.stage=Path(config['staging_root']);self.target_manifest=manifest;old=self.root/'reports/carla_train_expand10_v1'
  self.config=json.loads((old/'frozen_acquisition_config.json').read_text())
  self.manifest=json.loads((old/'frozen_dataset_manifest.json').read_text())
  self.cache=Path(self.config['staging_root'])/'cache_train10_v1'
  self.arrays,self.cm=cb.load_cache(self.cache)
  result=json.loads((old/'cache_result.json').read_text())
  assert sha(self.cache/'carla_train_cache.npz')==result['file_sha256']
  assert cb._content_sha256(self.cache/'carla_train_cache.npz')==self.cm['cache_content_sha256']==result['content_sha256']
  assert self.cm['window_policy']['window_length_ticks']==config['window_length_ticks']==self.config['window_length_ticks']==12
  assert self.cm['cache_schema_version']==config['features']['cache_schema_version']==self.config['features']['cache_schema_version']==2
  assert self.cm['feature_dims']==cb.FEATURE_DIMS
  assert config['features']==self.config['features'] and config['pseudo']==self.config['pseudo']
  assert config['source_hashes']==self.config['source_hashes']
  for rel,h in config['source_hashes'].items():assert sha(self.root/rel)==h,rel
  assert self.cm['extractor_version']==cb._extractor_version() and self.cm['code_version']==cb._code_version()
  # In unchanged production code, partition_label is read once, only to write
  # metadata. It is never an input to any feature operation or causal window.
  tree=ast.parse(inspect.getsource(cb._scenario_rows));uses=[n for n in ast.walk(tree) if isinstance(n,ast.Name) and n.id=='partition_label']
  assert len(uses)==1
  partition_values=[value for d in ast.walk(tree) if isinstance(d,ast.Dict) for key,value in zip(d.keys,d.values) if isinstance(key,ast.Constant) and key.value=='partition']
  assert len(partition_values)==1 and any(n is uses[0] for n in ast.walk(partition_values[0]))
  self.ids=set(self.manifest['archive_order_ids']);self.oldcal=set(self.manifest['CAL_NORMAL'])
  for sid in self.ids:
   rows=self.arrays['scenario_id']==sid;record=next(r for r in self.manifest['scenarios'] if r['scenario_id']==sid)
   assert np.array_equal(self.arrays['tick'][rows],np.arange(1,record['tick_coverage']['rgb_count']))
   assert np.all(self.arrays['source_split'][rows]=='train')
   for key in ['camera','seg','gnss','imu']:assert np.isfinite(self.arrays[key][rows]).all()
  if manifest is not None:
   assert manifest['archive_order_ids'][:10]==self.manifest['archive_order_ids']
   for record in self.manifest['scenarios']:
    new=next(r for r in manifest['scenarios'] if r['scenario_id']==record['scenario_id'])
    assert new['source_files']==record['source_files'] and new['source_hash_bundle_sha256']==record['source_hash_bundle_sha256']
    assert Path(new['actual_path']).resolve()==Path(record['actual_path']).resolve()
  self.report={'safe_reuse':True,'old_cache_file_sha256':result['file_sha256'],'old_cache_content_sha256':result['content_sha256'],
   'feature_code_unchanged':True,'window_unchanged':True,'schema_dimensions_unchanged':True,
   'partition_independence_verified_from_unchanged_production_AST':True,
   'source_hashes_reverified_by_streamed_archive_payload_comparison':manifest is not None,
   'clean_reused_scenarios':[],'clean_reused_rows':0,'pseudo_reused_scenarios':[],'pseudo_reused_rows':0,
   'old_CAL_to_FIT_pseudo_excluded':sorted(self.oldcal-set(manifest['CAL_NORMAL'])) if manifest else None}
 def clean_rows(self,m,partition_label,window_length):
  assert m.scenario_id in self.ids and window_length==12
  sel=self.arrays['scenario_id']==m.scenario_id
  rows={key:self.arrays[key][sel].copy() for key in ['scenario_id','town','tick','source_split','camera','seg','gnss','imu']}
  rows['partition']=np.array([partition_label]*len(rows['tick']),dtype=str)
  raw={'scenario_path':m.path}
  if partition_label=='CAL_NORMAL' and m.scenario_id not in self.oldcal:
   raw.update(gnss=self.cb._gnss_table(m.path),imu=self.cb._imu_accel_table(m.path))
  self.report['clean_reused_scenarios'].append(m.scenario_id);self.report['clean_reused_rows']+=len(rows['tick'])
  return rows,raw
 def pseudo_rows(self,sid,recipes,seed,max_per_tick,severities):
  assert sid in self.oldcal and tuple(recipes)==tuple(self.config['pseudo']['recipes'])
  assert seed==self.config['pseudo']['seed']==0 and max_per_tick==1 and severities is None
  indices=np.flatnonzero(self.arrays['cal_parent_scenario']==sid)
  ticks=self.arrays['cal_parent_tick'][indices]
  assert np.array_equal(ticks,np.arange(1,np.sum(self.arrays['scenario_id']==sid)+1))
  assert np.array_equal(self.arrays['cal_recipe_id'][indices],np.array([recipes[int(t)%len(recipes)] for t in ticks]))
  assert np.all(self.arrays['cal_target_normal'][indices]==0)
  assert np.array_equal(self.arrays['cal_window_start_tick'][indices],np.maximum(0,ticks-11))
  assert np.array_equal(self.arrays['cal_window_end_tick'][indices],ticks)
  meta=[{key:self.arrays['cal_'+key][i].item() for key in ['parent_scenario','parent_tick','recipe_id','modality','severity','seed','window_start_tick','window_end_tick']} for i in indices]
  for row in meta:row.update(source_split='train',synthetic_corruption=True)
  feats=[row.copy() for row in self.arrays['cal_features'][indices]]
  self.report['pseudo_reused_scenarios'].append(sid);self.report['pseudo_reused_rows']+=len(indices)
  return meta,feats,dict(Counter(row['recipe_id'] for row in meta)),{}
 def new_clean_rows(self,m,partition_label,window_length):
  assert m.scenario_id not in self.ids and window_length==12
  r=next(r for r in self.target_manifest['scenarios'] if r['scenario_id']==m.scenario_id)
  stem=f"{r['archive_ordinal']:02d}_{m.scenario_id.replace('/','_')}";folder=self.stage/'new_clean_blocks_v1'
  metadata=json.loads((folder/(stem+'.json')).read_text());path=folder/(stem+'.npz')
  assert sha(path)==metadata['npz_sha256']
  assert metadata['source_hash_bundle_sha256']==r['source_hash_bundle_sha256']
  assert metadata['source_hashes']==self.config['source_hashes'] and metadata['features']==self.config['features']
  assert metadata['window_length']==12 and not metadata['partition_assigned'] and metadata['no_model_fit_or_pseudo']
  with np.load(path,allow_pickle=False) as z:rows={key:z[key] for key in z.files}
  assert 'partition' not in rows and np.array_equal(rows['tick'],np.arange(1,m.n_ticks))
  assert np.all(rows['scenario_id']==m.scenario_id) and np.all(rows['town']==m.town) and np.all(rows['source_split']=='train')
  for key,dim in self.config['features']['dims'].items():assert rows[key].shape==(m.n_ticks-1,dim) and np.isfinite(rows[key]).all()
  rows['partition']=np.array([partition_label]*len(rows['tick']),dtype=str)
  raw={'scenario_path':m.path}
  if partition_label=='CAL_NORMAL':raw.update(gnss=self.cb._gnss_table(m.path),imu=self.cb._imu_accel_table(m.path))
  self.report.setdefault('new_precomputed_scenarios',[]).append(m.scenario_id)
  for field,value in [('new_precomputed_clean_rows',len(rows['tick'])),('new_precomputed_camera_calls',metadata['clean_camera_calls']),('new_precomputed_seg_calls',metadata['clean_seg_calls']),('new_clean_precompute_active_wall_s',metadata['extraction_wall_s'])]:self.report[field]=self.report.get(field,0)+value
  self.report.setdefault('new_clean_block_file_hashes',{})[m.scenario_id]=metadata['npz_sha256']
  return rows,raw
 def verify_assembled(self,a):
  for sid in self.ids:
   old=self.arrays['scenario_id']==sid;new=a['scenario_id']==sid
   for key in ['scenario_id','town','tick','source_split','camera','seg','gnss','imu']:
    assert np.array_equal(a[key][new],self.arrays[key][old]),(sid,key)
    if key in ['tick','camera','seg','gnss','imu']:assert a[key][new].tobytes()==self.arrays[key][old].tobytes(),(sid,key,'byte equality')
  for sid in self.report['pseudo_reused_scenarios']:
   old=self.arrays['cal_parent_scenario']==sid;new=a['cal_parent_scenario']==sid
   for key in [k for k in self.arrays if k.startswith('cal_')]:assert np.array_equal(a[key][new],self.arrays[key][old]),(sid,key)
  self.report['assembled_clean_features_byte_identical']=True
