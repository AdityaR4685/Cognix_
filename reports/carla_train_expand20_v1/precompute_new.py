"""Compute new clean feature blocks once while acquisition continues.

No FIT/CAL membership, models, pseudo rows or calibration is selected here.
Only completed new scenarios under the frozen extraction protocol qualify.
"""
import hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent;sys.path.insert(0,str(ROOT))
from cognix.adapters.carla import cache_builder as cb
from cognix.adapters.carla.carlanomaly_loader import CarlAnomalyLoader,CarlAnomalySplit
config=json.loads((OUT/'frozen_acquisition_config.json').read_text());stage=Path(config['staging_root']);blocks=stage/'new_clean_blocks_v1';blocks.mkdir(exist_ok=True)
oldids=set(config['preserved_N10']['archive_order_ids']);started=time.monotonic();state={'phase':'waiting_for_new_complete_scenarios','scenario_id':None,'clean_camera_calls':0,'clean_seg_calls':0,'completed_new_blocks':0}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for c in iter(lambda:f.read(8<<20),b''):h.update(c)
 return h.hexdigest()
def publish():
 state['elapsed_s']=round(time.monotonic()-started,2)
 p=stage/'precompute_progress.json';tmp=p.with_suffix('.json.tmp');tmp.write_text(json.dumps(state,indent=2));os.replace(tmp,p)
original_camera,original_seg=cb.extract_camera_features,cb.extract_seg_features
last=0
def observe():
 global last
 if time.monotonic()-last>10:last=time.monotonic();publish()
def camera(path):
 assert Path(path).parents[2].name+'/'+Path(path).parents[1].name not in oldids
 value=original_camera(path);state['clean_camera_calls']+=1;observe();return value
def seg(path):
 assert Path(path).parents[2].name+'/'+Path(path).parents[1].name not in oldids
 value=original_seg(path);state['clean_seg_calls']+=1;observe();return value
cb.extract_camera_features,cb.extract_seg_features=camera,seg
for rel,h in config['source_hashes'].items():assert sha(ROOT/rel)==h
assert json.loads((OUT/'reuse_preflight_results.json').read_text())['checks_passed']
publish()
try:
 while True:
  found=[]
  for p in sorted((stage/'scenario_manifests').glob('*.json')):
   try:r=json.loads(p.read_text())
   except (json.JSONDecodeError,OSError):continue
   if r['scenario_id'] not in oldids and r.get('complete') and r.get('strict_train_loader_validated'):found.append(r)
  for r in found:
   sid=r['scenario_id'];stem=f"{r['archive_ordinal']:02d}_{sid.replace('/','_')}";dest=blocks/(stem+'.npz');meta_path=blocks/(stem+'.json')
   if meta_path.exists():
    meta=json.loads(meta_path.read_text());assert meta['npz_sha256']==sha(dest);continue
   assert not dest.exists(),'Partial new feature block requires investigation; never recompute automatically'
   state.update(phase='new_clean_feature_extraction',scenario_id=sid);publish();t=time.monotonic()
   loader=CarlAnomalyLoader(Path(r['actual_path']).parents[2],strict_layout=True);m=loader.build_manifest(CarlAnomalySplit.TRAIN,sid)
   # Fixed metadata placeholder only; the verified feature computation never
   # reads this label. Drop it entirely before publishing the feature block.
   rows,raw=cb._scenario_rows(loader,m,'UNASSIGNED',12);rows.pop('partition')
   assert len(rows['tick'])==r['tick_coverage']['rgb_count']-1
   tmp=dest.with_suffix('.npz.tmp')
   with tmp.open('xb') as f:np.savez(f,**rows)
   os.replace(tmp,dest)
   meta={'scenario_id':sid,'archive_ordinal':r['archive_ordinal'],'clean_rows':len(rows['tick']),
    'source_hash_bundle_sha256':hashlib.sha256(json.dumps(r['source_files'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),
    'source_hashes':config['source_hashes'],'window_length':12,'features':config['features'],'npz_sha256':sha(dest),
    'extraction_wall_s':round(time.monotonic()-t,3),'clean_camera_calls':len(rows['tick']),'clean_seg_calls':len(rows['tick']),
    'partition_assigned':False,'no_model_fit_or_pseudo':True}
   meta_path.write_text(json.dumps(meta,indent=2));print(json.dumps({'kind':'new_clean_block_complete',**meta}),flush=True)
   del rows,raw
  state['completed_new_blocks']=len(list(blocks.glob('*.json')))
  if state['completed_new_blocks']==10:state['phase']='complete';publish();break
  result=stage/'acquisition_result.json'
  if result.exists():assert json.loads(result.read_text())['status']=='complete','Acquisition failed; preserve blocks and stop'
  state['phase']='waiting_for_new_complete_scenarios';publish();time.sleep(10)
finally:cb.extract_camera_features,cb.extract_seg_features=original_camera,original_seg;publish()
