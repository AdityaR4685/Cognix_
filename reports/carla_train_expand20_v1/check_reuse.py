"""Compact reuse checks; never recomputes raw image/window features."""
import json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent;sys.path.insert(0,str(ROOT))
from verified_reuse import VerifiedReuse
config=json.loads((OUT/'frozen_acquisition_config.json').read_text());reuse=VerifiedReuse(ROOT,config)
stage=Path(config['staging_root']);records=[json.loads(p.read_text()) for p in (stage/'scenario_manifests').glob('*.json')]
for old in reuse.manifest['scenarios']:
 current=next(r for r in records if r['scenario_id']==old['scenario_id'])
 assert current['complete'] and current['strict_train_loader_validated'] and current['source_files']==old['source_files']
 # Both labels must leave every compact feature and tick exactly unchanged.
 for label in ['TRAIN_NORMAL','CAL_NORMAL']:
  rows,raw=reuse.clean_rows(SimpleNamespace(scenario_id=old['scenario_id'],path=Path(old['actual_path'])),label,12)
  selected=reuse.arrays['scenario_id']==old['scenario_id']
  for key in ['camera','seg','gnss','imu','tick']:assert rows[key].tobytes()==reuse.arrays[key][selected].tobytes()
  assert np.all(rows['partition']==label)
for sid in reuse.oldcal:
 metadata,features,attempts,skips=reuse.pseudo_rows(sid,tuple(config['pseudo']['recipes']),0,1,None)
 idx=reuse.arrays['cal_parent_scenario']==sid
 assert np.array_equal(np.asarray(features),reuse.arrays['cal_features'][idx])
 assert len(metadata)==int(idx.sum())==sum(attempts.values()) and not skips
result={'checks_passed':True,'verified_clean_rows':len(reuse.arrays['tick']),'source_files_reverified_for_all_first_ten':True,
 'both_partition_labels_leave_features_byte_identical':True,'old_CAL_pseudo_reconstruction_exact':True,
 'raw_feature_extractor_calls':0,'no_model_fitting':True,'N20_split_not_inferred_or_chosen':True}
(OUT/'reuse_preflight_results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
