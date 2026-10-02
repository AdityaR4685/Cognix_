"""Freeze observed archive order and the preregistered 15/5 split, before fits."""
import hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));OUT=Path(__file__).parent
from cognix.adapters.carla.cache_builder import partition_train_scenarios
from cognix.adapters.carla.carlanomaly_loader import CarlAnomalyLoader,CarlAnomalySplit,Modality
config_raw=(OUT/'frozen_acquisition_config.json').read_bytes();config=json.loads(config_raw);config_hash=hashlib.sha256(config_raw).hexdigest();stage=Path(config['staging_root'])
result=json.loads((stage/'acquisition_result.json').read_text())
assert result['status']=='complete' and result['strict_validated_count']==20 and result['config_sha256']==config_hash
records=[]
for p in sorted((stage/'scenario_manifests').glob('*.json')):
 r=json.loads(p.read_text());assert r['complete'] and r['strict_train_loader_validated'] and r['acquisition_config_sha256']==config_hash
 loader=CarlAnomalyLoader(Path(r['actual_path']).parents[2],strict_layout=True)
 loader.list_towns(CarlAnomalySplit.TRAIN)
 m=loader.build_manifest(CarlAnomalySplit.TRAIN,r['scenario_id']);m.require(Modality.RGB_FRONT,Modality.SEGMENTATION_FRONT,Modality.GNSS,Modality.IMU)
 assert m.n_ticks==r['tick_coverage']['rgb_count']==m.n_seg_ticks
 assert m.n_feather_rows['gnss']==m.n_ticks and m.n_feather_rows['imu']==m.n_ticks
 records.append(r)
assert len(records)==20 and [r['archive_ordinal'] for r in records]==list(range(1,21))
ids=[r['scenario_id'] for r in records];part=partition_train_scenarios(ids,seed=2026,cal_fraction=.25)
assert len(part.train_normal)==15 and len(part.cal_normal)==5
assert ids[:10]==config['preserved_N10']['archive_order_ids']
from collections import Counter
town_counts=dict(Counter(r['town'] for r in records))
mapping=part.mapping()
reuse_policy_hash=(OUT/'frozen_compute_reuse_policy.sha256').read_text().strip()
assert hashlib.sha256((OUT/'frozen_compute_reuse_policy.json').read_bytes()).hexdigest()==reuse_policy_hash
for r in records:
 r['builder_partition']=mapping[r['scenario_id']]
 r['partition']='FIT_NORMAL' if r['builder_partition']=='TRAIN_NORMAL' else 'CAL_NORMAL'
 r['source_hash_bundle_sha256']=hashlib.sha256(json.dumps(r['source_files'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
manifest={'schema_version':1,'protocol_version':config['protocol_version'],'study_role':config['study_role'],'final_planned_train_development_target':20,'no_automatic_further_expansion':True,'acquisition_config_sha256':config_hash,'archive_order_ids':ids,'sorted_canonical_ids':sorted(ids),'split_seed':2026,'split_rule':config['split']['rule'],'builder_partition':mapping,'FIT_NORMAL':list(part.train_normal),'CAL_NORMAL':list(part.cal_normal),'feature_extractor_sha256':config['features']['extractor_sha256'],'recipe_definition_sha256':config['source_hashes']['cognix/adapters/carla/pseudo_anomalies.py'],'recipes':config['pseudo'],'window_length_ticks':12,'prediction_semantics':config['prediction_semantics'],'scenarios':records,'town_counts':town_counts,'unique_towns':len(town_counts),'N10_role_changes':{sid:{'old':('FIT_NORMAL' if sid in json.loads((ROOT/'reports/carla_train_expand10_v1/frozen_dataset_manifest.json').read_text())['FIT_NORMAL'] else 'CAL_NORMAL'),'new':('FIT_NORMAL' if mapping[sid]=='TRAIN_NORMAL' else 'CAL_NORMAL')} for sid in ids[:10]},'no_model_results_used':True,'no_test_access':True}
manifest['compute_reuse_policy_sha256']=reuse_policy_hash
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
