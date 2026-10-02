"""Read accepted historical results; no refitting or edits to old artifacts."""
import json
from pathlib import Path
import numpy as np
from cognix.adapters.carla.normality import EnsemblePredictiveCalibrator
def stats(x):
 x=np.asarray(x,float)
 return {'min':float(x.min()),'p10':float(np.percentile(x,10)),'median':float(np.median(x)),'p90':float(np.percentile(x,90)),'max':float(x.max())}
def histories(root):
 root=Path(root);two=root/'reports/carla_calibration_audit';ten=root/'reports/carla_train_expand10_v1'
 old=json.loads((two/'separation_results.json').read_text())['agents'];mid=json.loads((ten/'calibration_results.json').read_text())['agents'];out={}
 for name in ['Camera','Seg','GNSS','IMU']:
  entries={}
  for n,data,folder,suffix in [(2,old[name],two,'_member_scores.npz'),(10,mid[name],ten,'_cal_member_scores.npz')]:
   with np.load(folder/(name.lower()+suffix),allow_pickle=False) as z:M,y=z['member_scores'],z['target_normal']
   audit=data['audit'];r={'n_normal':int(np.sum(y==1)),'n_pseudo':int(np.sum(y==0)),'audit':audit,'valid':bool(audit['finite_optimum']),
    'raw_member_score_disagreement':{label:stats(M[y==target].std(axis=1)) for label,target in [('normal',1),('pseudo',0)]},
    'score_distributions':{label:{'ensemble_mean':stats(M[y==target].mean(axis=1)),'pooled_members':stats(M[y==target].ravel())} for label,target in [('normal',1),('pseudo',0)]},
    'calibrated_probability_disagreement':None,'uq':None,'uq_available_statistics':None}
   if r['valid'] and n==2:
    r['uq']={label:{('prob_normal' if key=='prediction' else key):{'median':value} for key,value in d.items()} for label,d in data['diagnostic_only_uq'].items()}
    r['uq_available_statistics']='accepted historical medians only; full probability-member matrices/intercept were not saved; no historical refit'
   if r['valid'] and n==10:
    params=data['mapping_parameters_raw'];P=EnsemblePredictiveCalibrator._sigmoid(params['slope']*M+params['intercept'])
    r['calibrated_probability_disagreement']={label:stats(P[y==target].std(axis=1)) for label,target in [('normal',1),('pseudo',0)]}
    r['uq']=data['scientific_probability_and_uq'];r['uq_available_statistics']='full accepted distributions; probability disagreement reconstructed from saved valid parameters and member matrices'
    for label,target in [('normal',1),('pseudo',0)]:assert np.isclose(np.median(P[y==target].mean(axis=1)),r['uq'][label]['prob_normal']['median'],rtol=1e-12,atol=1e-14)
   entries[str(n)]=r
  out[name]=entries
 return out
