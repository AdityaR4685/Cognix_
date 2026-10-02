"""Read-only deterministic reproduction on frozen v2 cache. No raw-data reads."""
import hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from cognix.adapters.carla.cache_builder import load_cache
from cognix.adapters.carla.normality import EnsemblePredictiveCalibrator,ensemble_to_uncertainty
from cognix.adapters.carla.real_agents import RealCameraAgent,RealSegAgent,RealGNSSAgent,RealIMUAgent
out=Path(__file__).parent; probe=Path(os.environ['TEMP'])/'carla_train_probe'
a,manifest=load_cache(probe/'cache_v2_a')
assert set(a['source_split'])=={'train'}
result={'cache_content_sha256':manifest['cache_content_sha256'],'seed':42,'members':5,'partition':manifest['partition'],'agents':{}}
for name,cls,key,lo,hi in [('Camera',RealCameraAgent,'camera',0,18),('Seg',RealSegAgent,'seg',18,47),('GNSS',RealGNSSAgent,'gnss',47,55),('IMU',RealIMUAgent,'imu',55,65)]:
 agent=cls(seed=42,n_members=5).fit(a[key][a['partition']=='TRAIN_NORMAL'])
 normal=a[key][a['partition']=='CAL_NORMAL']; pseudo=a['cal_features'][a['cal_modality']==key,lo:hi]
 M=np.array([agent.predict_normality(r) for r in np.vstack([normal,pseudo])]); y=np.r_[np.ones(len(normal)),np.zeros(len(pseudo))]
 agent.fit_calibrator(M,y); cal=agent._calibrator
 np.savez(out/f'{key}_member_scores.npz',member_scores=M,target_normal=y)
 info=cal.optimization_info
 P=cal._sigmoid(cal.a*M+cal.b)
 uq={}
 for label,mask in [('normal',y==1),('pseudo',y==0)]:
  u=[ensemble_to_uncertainty(p) for p in P[mask]]
  uq[label]={f:float(np.median([getattr(v,f) for v in u])) for f in ['prediction','epistemic','aleatoric','total']}
 result['agents'][name]={'n_normal':len(normal),'n_pseudo':len(pseudo),'audit':info,'jensen':agent._cal_jensen_diag,'diagnostic_only_uq':uq}
 print(name,json.dumps(info),flush=True)
(out/'separation_results.json').write_text(json.dumps(result,indent=2))
