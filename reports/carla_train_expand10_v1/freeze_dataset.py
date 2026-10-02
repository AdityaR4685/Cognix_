"""Freeze observed archive order and the preregistered 8/2 split, before fits."""
import hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));OUT=Path(__file__).parent
from cognix.adapters.carla.cache_builder import partition_train_scenarios
from cognix.adapters.carla.carlanomaly_loader import CarlAnomalyLoader,CarlAnomalySplit,Modality
config_raw=(OUT/'frozen_acquisition_config.json').read_bytes();config=json.loads(config_raw);config_hash=hashlib.sha256(config_raw).hexdigest();stage=Path(config['staging_root'])
result=json.loads((stage/'acquisition_result.json').read_text())
assert result['status']=='complete' and result['strict_validated_count']==10 and result['config_sha256']==config_hash
records=[]
for p in sorted((stage/'scenario_manifests').glob('*.json')):
 r=json.loads(p.read_text());assert r['complete'] and r['strict_train_loader_validated'] and r['acquisition_config_sha256']==config_hash
 loader=CarlAnomalyLoader(Path(r['actual_path']).parents[2],strict_layout=True)
 loader.list_towns(CarlAnomalySplit.TRAIN)
 m=loader.build_manifest(CarlAnomalySplit.TRAIN,r['scenario_id']);m.require(Modality.RGB_FRONT,Modality.SEGMENTATION_FRONT,Modality.GNSS,Modality.IMU)
 assert m.n_ticks==r['tick_coverage']['rgb_count']==m.n_seg_ticks
 assert m.n_feather_rows['gnss']==m.n_ticks and m.n_feather_rows['imu']==m.n_ticks
 records.append(r)
assert len(records)==10 and [r['archive_ordinal'] for r in records]==list(range(1,11))
ids=[r['scenario_id'] for r in records];part=partition_train_scenarios(ids,seed=2026,cal_fraction=.25)
assert len(part.train_normal)==8 and len(part.cal_normal)==2
mapping=part.mapping()
for r in records:
 r['builder_partition']=mapping[r['scenario_id']]
 r['partition']='FIT_NORMAL' if r['builder_partition']=='TRAIN_NORMAL' else 'CAL_NORMAL'
 r['source_hash_bundle_sha256']=hashlib.sha256(json.dumps(r['source_files'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
manifest={'schema_version':1,'protocol_version':config['protocol_version'],'study_role':config['study_role'],'preferred_later_target':config['preferred_later_target'],'acquisition_config_sha256':config_hash,'archive_order_ids':ids,'sorted_canonical_ids':sorted(ids),'split_seed':2026,'split_rule':config['split']['rule'],'builder_partition':mapping,'FIT_NORMAL':list(part.train_normal),'CAL_NORMAL':list(part.cal_normal),'feature_extractor_sha256':config['features']['extractor_sha256'],'recipe_definition_sha256':config['source_hashes']['cognix/adapters/carla/pseudo_anomalies.py'],'recipes':config['pseudo'],'window_length_ticks':12,'prediction_semantics':config['prediction_semantics'],'scenarios':records,'no_model_results_used':True,'no_test_access':True}
raw=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode();p=OUT/'frozen_dataset_manifest.json'
if p.exists():assert p.read_bytes()==raw
else:
 with p.open('xb') as f:f.write(raw)
h=hashlib.sha256(raw).hexdigest();hp=OUT/'frozen_dataset_manifest.sha256'
if hp.exists():assert hp.read_text().strip()==h
else:hp.write_text(h+'\n')
print('FROZEN DATASET SHA256',h,flush=True)
print('ARCHIVE ORDER',json.dumps(ids),flush=True)
print('FIT_NORMAL',json.dumps(list(part.train_normal)),flush=True)
print('CAL_NORMAL',json.dumps(list(part.cal_normal)),flush=True)
