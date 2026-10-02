"""Read-only pre-exponential diagnostics using unchanged fitted members."""
import numpy as np
def stats(x):
 x=np.asarray(x,float)
 return {'min':float(x.min()),'p10':float(np.percentile(x,10)),'median':float(np.median(x)),'p90':float(np.percentile(x,90)),'max':float(x.max())}
def analyze_gnss(agent,features,scores,labels):
 members=agent._ensemble._members
 d2=np.column_stack([[member.mahalanobis_distance2(row) for row in features] for member in members])
 scale=np.array([m._d2_scale for m in members]); exponent=-0.5*d2/scale
 reproduced=np.array([[float(np.exp(-0.5*float(d2[i,k])/float(scale[k]))) for k in range(len(members))] for i in range(len(features))])
 assert np.array_equal(reproduced,scores), 'Diagnostic transform must reproduce production scores exactly'
 smallest=float(np.nextafter(np.float64(0),np.float64(1)))
 out={'model':'unchanged bootstrap MahalanobisNormality','score_transform':'exp(-0.5*d2/d2_scale)',
 'distance_scale_per_member':scale.tolist(),'float64_smallest_positive_subnormal':smallest,
 'float64_smallest_positive_normal':float(np.finfo(np.float64).tiny),'log_smallest_positive_subnormal':float(np.log(smallest)),
 'scientific_score_transform_changed':False,'exact_production_score_reproduction':True,'classes':{}}
 for label,target in [('normal',1),('pseudo',0)]:
  selected=labels==target;s=scores[selected];e=exponent[selected];positive=s[s>0]
  out['classes'][label]={'n_observations':int(selected.sum()),'member_entries':s.size,
   'squared_mahalanobis':stats(d2[selected]),'normalized_squared_distance':stats(d2[selected]/scale),'exponential_argument':stats(e),
   'score':stats(s),'exact_zero_entries':int(np.sum(s==0)),'exact_zero_fraction':float(np.mean(s==0)),
   'all_members_zero_observations':int(np.sum(np.all(s==0,axis=1))),
   'smallest_positive_observed_score':float(positive.min()) if len(positive) else None,
   'arguments_below_log_smallest_subnormal':int(np.sum(e<np.log(smallest))),
   'all_zero_scores_explained_by_exp_underflow':bool(np.all(e[s==0]<np.log(smallest))) if np.any(s==0) else None}
 out['diagnosis']='Large Mahalanobis distances create large negative exponential arguments; float64 exp underflows to exactly zero where recorded. No explicit score clamp/threshold is used.'
 return out
