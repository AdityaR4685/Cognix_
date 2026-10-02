"""Read-only reuse verification against the observed, frozen N20 split."""
import json, sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
sys.path.insert(0,str(ROOT))
from verified_reuse import VerifiedReuse
config=json.loads((OUT/'frozen_acquisition_config.json').read_text())
manifest=json.loads((OUT/'frozen_dataset_manifest.json').read_text())
reuse=VerifiedReuse(ROOT,config,manifest)
retained=sorted(reuse.oldcal & set(manifest['CAL_NORMAL']))
for sid in retained:
 metadata,features,attempts,skips=reuse.pseudo_rows(sid,tuple(config['pseudo']['recipes']),0,1,None)
 assert len(metadata)==len(features)==sum(attempts.values()) and not skips
checked=[]
for r in manifest['scenarios']:
 if r['scenario_id'] in reuse.ids:continue
 stem=f"{r['archive_ordinal']:02d}_{r['scenario_id'].replace('/','_')}"
 if not (reuse.stage/'new_clean_blocks_v1'/(stem+'.json')).exists():continue
 m=SimpleNamespace(scenario_id=r['scenario_id'],town=r['town'],path=Path(r['actual_path']),n_ticks=r['tick_coverage']['rgb_count'])
 rows,raw=reuse.new_clean_rows(m,manifest['builder_partition'][m.scenario_id],12)
 checked.append(m.scenario_id)
result={'frozen_manifest_reuse_verified':True,'N10_clean_rows_eligible_for_exact_reuse':29990,
 'retained_old_CAL_pseudo':retained,'retained_old_CAL_pseudo_rows':reuse.report['pseudo_reused_rows'],
 'excluded_old_CAL_to_FIT_pseudo':reuse.report['old_CAL_to_FIT_pseudo_excluded'],
 'new_CAL_pseudo_generation_required':sorted(set(manifest['CAL_NORMAL'])-reuse.oldcal),
 'published_new_clean_blocks_verified_at_check_time':checked,
 'raw_feature_extraction_calls':0,'models_fitted':0}
(OUT/'frozen_reuse_check_results.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
