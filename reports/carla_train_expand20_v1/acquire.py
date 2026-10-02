"""Authorized TRAIN expansion: one staged replay, then one live decoder."""
import hashlib,json,os,shutil,sys,time,urllib.request,urllib.error
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT)); REPORT=Path(__file__).parent
from cognix.adapters.carla.train_acquisition import AcquisitionError,TrainTarStream,GzipTrainStream,validate_range_response
from cognix.adapters.carla.carlanomaly_loader import CarlAnomalyLoader,CarlAnomalySplit,Modality
import numpy as np
import pandas as pd
CONFIG_PATH=REPORT/'frozen_acquisition_config.json'; CONFIG_RAW=CONFIG_PATH.read_bytes(); CONFIG=json.loads(CONFIG_RAW)
CONFIG_HASH=hashlib.sha256(CONFIG_RAW).hexdigest()
assert CONFIG_HASH==(REPORT/'frozen_acquisition_config.sha256').read_text().strip()
STAGE=Path(CONFIG['staging_root']); ORIGINAL=Path(CONFIG['original_prefix']['path']); PREFIX=STAGE/CONFIG['staging_prefix_name']; RAW=STAGE/'raw_train_v1'
START=time.monotonic(); CHUNK=CONFIG['resource_limits']['compressed_read_chunk_bytes']
STAGE.mkdir(parents=True,exist_ok=True)
LOG=STAGE/'acquisition_log.jsonl'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for c in iter(lambda:f.read(8<<20),b''):h.update(c)
 return h.hexdigest()
def log(kind,**values):
 entry={'utc':datetime.now(timezone.utc).isoformat(),'kind':kind,**values}
 with LOG.open('a',encoding='utf-8') as f:f.write(json.dumps(entry)+'\n')
 print(json.dumps(entry),flush=True)
def write_json(path,data):
 tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(data,indent=2)); os.replace(tmp,path)
def summary(s):
 return {'scenario_id':s['scenario_id'],'archive_ordinal':s['archive_ordinal'],'rgb_frames':len(s['rgb_ticks']),'seg_frames':len(s['seg_ticks']),'complete':s.get('complete',False)}
for rel,expected in CONFIG['source_hashes'].items():assert sha(ROOT/rel)==expected,rel
assert ORIGINAL.stat().st_size==CONFIG['original_prefix']['bytes'] and sha(ORIGINAL)==CONFIG['original_prefix']['sha256']
assert shutil.disk_usage(STAGE).free>=CONFIG['resource_limits']['minimum_free_bytes_to_start']
if not PREFIX.exists():
 with ORIGINAL.open('rb') as src,PREFIX.open('xb') as dst:shutil.copyfileobj(src,dst,8<<20)
 assert not os.path.samefile(ORIGINAL,PREFIX)
 assert sha(PREFIX)==CONFIG['original_prefix']['sha256']
 log('independent_copy_verified',bytes=PREFIX.stat().st_size,sha256=CONFIG['original_prefix']['sha256'])
assert not PREFIX.is_symlink() and not os.path.samefile(ORIGINAL,PREFIX)
# The preserved two scenarios remain at their original paths, without copies.
reuse={sid:Path(path) for sid,path in CONFIG['existing_scenario_paths'].items()}
def validate_boundary(record,evidence):
 sid=record['scenario_id']; path=Path(record['actual_path'])
 loader=CarlAnomalyLoader(path.parents[2],strict_layout=True)
 loader.list_towns(CarlAnomalySplit.TRAIN)
 m=loader.build_manifest(CarlAnomalySplit.TRAIN,sid)
 m.require(Modality.RGB_FRONT,Modality.SEGMENTATION_FRONT,Modality.GNSS,Modality.IMU)
 ticks=record['rgb_ticks']; expected=set(range(len(ticks)))
 if not ticks or ticks!=expected or record['seg_ticks']!=ticks:
  raise AcquisitionError(f'Noncontiguous/misaligned frame coverage: {sid}')
 if m.n_rgb_ticks!=len(ticks) or m.n_seg_ticks!=len(ticks):raise AcquisitionError(f'Loader counts disagree: {sid}')
 tables={}
 for p in sorted(path.glob('*.feather')):
  df=pd.read_feather(p); tables[p.name]={'rows':len(df),'columns':list(df.columns),'sha256':record['source_files'][p.name]['sha256']}
  if p.name in ['gnss.feather','imu.feather']:
   if len(df)!=len(ticks):raise AcquisitionError(f'Sensor row count mismatch: {sid}/{p.name}')
   index_col,index=loader._sensor_index(df,p)
   if index is not None and not np.array_equal(index,np.arange(len(ticks))):raise AcquisitionError(f'Sensor tick mismatch: {sid}/{p.name}')
   required=['altitude','latitude','longitude'] if p.name=='gnss.feather' else ['acceleration_x','acceleration_y','acceleration_z']
   if not set(required)<=set(df.columns) or not np.isfinite(df[required].to_numpy(dtype=float)).all():raise AcquisitionError(f'Invalid sensor schema/values: {sid}/{p.name}')
 record['tables']=tables
 record['tick_coverage']={'rgb_count':len(ticks),'rgb_range':[min(ticks),max(ticks)],'seg_count':len(ticks),'seg_range':[min(ticks),max(ticks)],'ticks_contiguous':True}
 record['strict_train_loader_validated']=True
 data={**record,'rgb_ticks':sorted(ticks),'seg_ticks':sorted(record['seg_ticks']),'complete':True,'completion_evidence':evidence,'acquisition_config_sha256':CONFIG_HASH}
 (STAGE/'scenario_manifests').mkdir(exist_ok=True)
 target=STAGE/'scenario_manifests'/f"{record['archive_ordinal']:02d}_{sid.replace('/','_')}.json"
 if target.exists():assert json.loads(target.read_text())==data
 else:target.write_text(json.dumps(data,indent=2))
 log('scenario_complete',**summary(data),ticks=len(ticks),reused=record['reused'],boundary=evidence)
parser=TrainTarStream(RAW,reuse,validate_boundary,target=20)
decoder=GzipTrainStream(parser); prefix_digest=hashlib.sha256(); last_progress=0
network_bytes=0; network_started=None; response_total=CONFIG['preserved_N10']['remote_total_bytes']; remote_validator=CONFIG['preserved_N10']['remote_validator']; status='running'; failure=None
progress_path=STAGE/'progress.json'
def progress(phase,force=False):
 global last_progress
 now=time.monotonic()
 if not force and now-last_progress<10:return
 last_progress=now
 p={'phase':phase,'protocol_sha256':CONFIG_HASH,'compressed_bytes':decoder.compressed_bytes,'network_bytes':network_bytes,'elapsed_s':round(now-START,1),'network_elapsed_s':round(now-network_started,1) if network_started else None,'completed_scenarios':sum(s.get('complete',False) for s in parser.scenarios),'scenarios':[summary(s) for s in parser.scenarios],'tar_offset':parser.offset,'parser_state':parser.state,'current_member':parser.member.name if parser.member else None,'prefix_bytes':PREFIX.stat().st_size,'free_disk_bytes':shutil.disk_usage(STAGE).free}
 write_json(progress_path,p)
 if force:log('progress',**p)
class TrainOnlyRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,req,fp,code,msg,headers,newurl):
  if newurl!=CONFIG['archive_url']:raise AcquisitionError('Unregistered archive redirect refused: '+newurl)
  return super().redirect_request(req,fp,code,msg,headers,newurl)
opener=urllib.request.build_opener(TrainOnlyRedirect())
try:
 log('replay_start',staged_bytes=PREFIX.stat().st_size,protocol_sha256=CONFIG_HASH)
 with PREFIX.open('rb') as f:
  while not parser.done:
   c=f.read(CHUNK)
   if not c:break
   prefix_digest.update(c);decoder.feed(c);progress('staged_replay')
 progress('staged_replay_complete',True)
 assert decoder.compressed_bytes==PREFIX.stat().st_size or parser.done
 failures=0
 while not parser.done:
  start=PREFIX.stat().st_size
  if start>=CONFIG['resource_limits']['max_staged_compressed_bytes']:raise AcquisitionError('Preregistered compressed storage ceiling reached')
  headers={'Range':f'bytes={start}-','Accept-Encoding':'identity','User-Agent':'COGNIX-TRAIN-expand20/1'}
  if remote_validator:headers['If-Range']=remote_validator
  log('range_request',expected_start=start,url=CONFIG['archive_url'])
  if network_started is None:network_started=time.monotonic()
  try:
   with opener.open(urllib.request.Request(CONFIG['archive_url'],headers=headers),timeout=90) as response:
    end,total=validate_range_response(response.status,response.headers,start,response_total)
    response_total=total
    validator=response.headers.get('ETag') or response.headers.get('Last-Modified')
    if remote_validator and validator and validator!=remote_validator:raise AcquisitionError('Archive validator changed')
    remote_validator=remote_validator or validator
    log('range_response_verified',status=response.status,content_range=response.headers.get('Content-Range'),validator=remote_validator)
    got=0
    with PREFIX.open('ab') as f:
     while not parser.done:
      c=response.read(CHUNK)
      if not c:break
      if got+len(c)>end-start+1:raise AcquisitionError('Response exceeded declared range')
      if PREFIX.stat().st_size+len(c)>CONFIG['resource_limits']['max_staged_compressed_bytes']:raise AcquisitionError('Compressed storage ceiling reached')
      if shutil.disk_usage(STAGE).free<10_000_000_000:raise AcquisitionError('Insufficient disk reserve')
      f.write(c);f.flush();got+=len(c);network_bytes+=len(c);prefix_digest.update(c)
      decoder.feed(c);progress('range_stream')
    log('range_stream_closed',start=start,end_exclusive=PREFIX.stat().st_size,received_bytes=got,target_proven=parser.done)
    if not got and not parser.done:raise AcquisitionError('Empty HTTP range stream')
    failures=0
  except (urllib.error.URLError,TimeoutError,ConnectionError,OSError) as exc:
   failures+=1;log('transport_retry',attempt=failures,error=repr(exc),resume_start=PREFIX.stat().st_size)
   if failures>=10:raise
   time.sleep(min(failures*2,10))
 status='complete' if parser.done else 'incomplete'
except BaseException as exc:
 status='failed';failure=repr(exc);log('failure',error=failure)
finally:
 parser.close()
 original_after={'bytes':ORIGINAL.stat().st_size,'sha256':sha(ORIGINAL)}
 assert original_after['bytes']==CONFIG['original_prefix']['bytes'] and original_after['sha256']==CONFIG['original_prefix']['sha256'],'Original prefix changed'
 report={'status':status,'failure':failure,'config_sha256':CONFIG_HASH,'original_before':CONFIG['original_prefix'],'original_after':original_after,'original_unchanged':True,'staged_prefix':str(PREFIX),'staged_bytes':PREFIX.stat().st_size,'staged_sha256':prefix_digest.hexdigest(),'network_bytes':network_bytes,'remote_total_bytes':response_total,'remote_validator':remote_validator,'elapsed_s':round(time.monotonic()-START,1),'replay_count_this_process':1,'scenarios':[summary(s) for s in parser.scenarios],'trailing_boundary':parser.trailing_boundary,'strict_validated_count':sum(s.get('complete',False) for s in parser.scenarios),'raw_root':str(RAW),'no_test_access':True}
 write_json(STAGE/'acquisition_result.json',report);write_json(REPORT/'acquisition_result.json',report)
 progress(status,True)
 log('acquisition_final',**report)
if status!='complete':raise SystemExit(1)
