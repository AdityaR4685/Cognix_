"""Freeze TRAIN acquisition before any network request; preserves accepted proposal."""
import hashlib,inspect,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from cognix.adapters.carla.cache_builder import CAL_PSEUDO_RECIPES,CACHE_SCHEMA_VERSION,WINDOW_LENGTH
from cognix.adapters.carla.pseudo_anomalies import RECIPES
from cognix.adapters.carla.real_features import __file__ as feature_file
OUT=Path(__file__).parent; PROBE=Path(os.environ['TEMP'])/'carla_train_probe'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(8<<20),b''): h.update(chunk)
 return h.hexdigest()
def exclusive(path,data):
 if path.exists():
  assert path.read_bytes()==data, f'Frozen file mismatch: {path}'
 else:
  with path.open('xb') as f: f.write(data)
proposal=ROOT/'reports/carla_calibration_audit/proposed_protocol.json'
prefix=PROBE/'train_prefix.tar.gz'
source_names=['cache_builder.py','carlanomaly_loader.py','normality.py','pseudo_anomalies.py','real_agents.py','real_features.py','train_acquisition.py']
source_hashes={'cognix/adapters/carla/'+n:sha(ROOT/'cognix/adapters/carla'/n) for n in source_names}
severities={rid:inspect.signature(fn).parameters['severity'].default for rid,mod,fn in RECIPES if rid in CAL_PSEUDO_RECIPES}
config={'schema_version':1,'protocol_version':'carla_train_expand10_v1','target_complete_train_scenarios':10,'selection_rule':'first 10 complete official TRAIN scenarios in parsed archive order; whole scenarios; no performance selection','source_split':'train','archive_url':'https://data.carlanomaly.de/v1/carlanomaly-base-train.tar.gz','original_prefix':{'path':str(prefix),'bytes':prefix.stat().st_size,'sha256':sha(prefix)},'staging_root':str(Path(os.environ['TEMP'])/'carla_train_expand10'),'staging_prefix_name':'train_prefix_expand10.tar.gz','independent_copy_required':True,'split':{'seed':2026,'rule':'existing NumPy PCG64 permutation of sorted canonical IDs; n_cal=max(1,round(0.25*N)), Python ties-to-even','FIT_NORMAL':8,'CAL_NORMAL':2,'builder_label_mapping':{'TRAIN_NORMAL':'FIT_NORMAL','CAL_NORMAL':'CAL_NORMAL'}},'window_length_ticks':WINDOW_LENGTH,'pseudo':{'recipes':list(CAL_PSEUDO_RECIPES),'default_severities':severities,'seed':0,'max_pseudo_per_tick':1,'source':'CAL_NORMAL only'},'features':{'dims':{'camera':18,'seg':29,'gnss':8,'imu':10},'extractor_sha256':sha(feature_file),'cache_schema_version':CACHE_SCHEMA_VERSION},'agent':{'seed':42,'n_members':5,'fit':'FIT clean only','calibration':'CAL clean + modality-matched CAL pseudo','validity_audit':'current production unchanged'},'prediction_semantics':'P(target=normal) under TRAIN-derived clean-vs-pseudo calibration','accepted_proposal_sha256':sha(proposal),'source_hashes':source_hashes,'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'resource_limits':{'max_staged_compressed_bytes':25_000_000_000,'minimum_free_bytes_to_start':80_000_000_000,'compressed_read_chunk_bytes':262144,'max_decompressed_chunk_bytes':1048576},'study_role':'intermediate multi-scenario checkpoint, not final Experiment-1 calibration dataset; no final performance/generalization claims even if every mapping validates','preferred_later_target':{'N':20,'FIT_NORMAL':15,'CAL_NORMAL':5,'requires_separate_approval':True},'prohibited':['official TEST','calibration or recipe tuning','regularization','slope caps','GAT implementation/training','conformal fitting','generic/frozen core edits','commits/pushes']}
raw=(json.dumps(config,sort_keys=True,indent=2)+'\n').encode()
exclusive(OUT/'frozen_acquisition_config.json',raw)
exclusive(OUT/'frozen_acquisition_config.sha256',(hashlib.sha256(raw).hexdigest()+'\n').encode())
files=[p for d in ['cognix','tests'] for p in (ROOT/d).rglob('*') if p.is_file() and '__pycache__' not in str(p)]
artifacts=[p for p in PROBE.rglob('*') if p.is_file() and '__pycache__' not in str(p)]
state={'git_status':subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),'repository_hashes':{p.relative_to(ROOT).as_posix():sha(p) for p in files},'accepted_proposal_sha256':sha(proposal),'probe_file_stats':{p.relative_to(PROBE).as_posix():{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in artifacts},'old_cache_file_hashes':{p.relative_to(PROBE).as_posix():sha(p) for name in ['cache_a','cache_b','cache_v2_a','cache_v2_b'] for p in (PROBE/name).iterdir() if p.is_file()},'original_prefix':config['original_prefix']}
exclusive(OUT/'initial_state.json',(json.dumps(state,indent=2)+'\n').encode())
print('FROZEN ACQUISITION CONFIG SHA256',hashlib.sha256(raw).hexdigest(),flush=True)
print('ORIGINAL PREFIX',config['original_prefix'],flush=True)
print('SOURCE HASHES',json.dumps(source_hashes),flush=True)
