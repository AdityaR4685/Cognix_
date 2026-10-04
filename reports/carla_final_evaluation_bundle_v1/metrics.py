"""Only formulas in the sealed final_metrics_spec and seed_aggregation_spec.

No fitting, threshold search, ensemble, randomness, or TEST access.
"""
import itertools
import math
import statistics
import numpy as np

METRICS = ('AUROC', 'AUPRC', 'F1', 'accuracy', 'balanced_accuracy', 'Brier', 'BCE', 'ECE')
SEEDS = (101, 202, 303, 404, 505)
METHODS = ('nograph', 'standard_gat', 'epistemic_gat')

def aligned(q, y):
    q, y = np.asarray(q, dtype=np.float64), np.asarray(y)
    if q.ndim != 1 or q.shape != y.shape or not len(q) or y.dtype.kind not in 'biu' or not np.isin(y,(0,1)).all() or not np.isfinite(q).all() or not ((0<=q)&(q<=1)).all():
        raise ValueError('MISSING_MISALIGNED_OR_NONFINITE_REQUIRED_OUTPUT')
    return q, y.astype(np.int64)

def classification(q, y, hard_threshold):
    q, y = aligned(q,y)
    if not math.isfinite(hard_threshold) or not 0<=hard_threshold<=1:
        raise ValueError('INVALID_FROZEN_DEVELOPMENT_THRESHOLD')
    r=1-q; pred=r>=hard_threshold
    n=len(y); positives=int(y.sum()); negatives=n-positives
    tp=int(np.sum(pred & (y==1))); fp=int(np.sum(pred & (y==0)))
    fn=positives-tp; tn=negatives-fp
    # Ascending tied groups: all positive/negative pairs, with half weight for ties.
    order=np.argsort(r,kind='stable'); s=r[order]; labels=y[order]
    starts=np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1]; ends=np.r_[starts[1:],n]
    group_n=ends-starts; group_p=np.add.reduceat(labels,starts); group_neg=group_n-group_p
    previous_neg=np.cumsum(group_neg)-group_neg
    auc=None if not positives or not negatives else float(np.sum(group_p*(previous_neg+.5*group_neg))/(positives*negatives))
    dp=group_p[::-1]; dn=group_n[::-1]
    ap=None if not positives else float(np.sum((dp/positives)*(np.cumsum(dp)/np.cumsum(dn))))
    clipped_q=np.clip(q,1e-7,1-1e-7); clipped_r=np.clip(r,1e-7,1-1e-7)
    edges=np.arange(16,dtype=np.float64)/15
    indices=np.searchsorted(edges,q,side='right')-1;indices=np.minimum(indices,14)
    ece=0.; bin_receipts=[]
    for b in range(15):
        mask=indices==b; count=int(mask.sum())
        mq=float(q[mask].mean()) if count else None; normal=float((1-y[mask]).mean()) if count else None
        contribution=0. if not count else count/n*abs(mq-normal)
        ece+=contribution
        bin_receipts.append({'bin':b,'n':count,'mean_q':mq,'normal_fraction':normal,'contribution':contribution})
    values={'AUROC':auc,'AUPRC':ap,'F1':0. if 2*tp+fp+fn==0 else 2*tp/(2*tp+fp+fn),
      'accuracy':(tp+tn)/n,'balanced_accuracy':None if not positives or not negatives else .5*(tp/positives+tn/negatives),
      'Brier':float(np.mean((q-(1-y))**2)),
      'BCE':float(np.mean(-(1-y)*np.log(clipped_q)-y*np.log(clipped_r))),'ECE':float(ece)}
    reasons={k:('requires both official tick classes' if k in ('AUROC','balanced_accuracy') else 'no official positive ticks') for k,v in values.items() if v is None}
    return {'values':values,'undefined_reasons':reasons,'eligible_ticks':n,'positive_ticks':positives,'negative_ticks':negatives,
      'confusion':{'TP':tp,'FP':fp,'FN':fn,'TN':tn},'hard_threshold':hard_threshold,'comparison':'>=',
      'ECE_bins':bin_receipts,'BCE_clip':1e-7}

def prediction_sets(q, cutoff):
    q=np.asarray(q,dtype=np.float64)
    if q.ndim!=1 or not np.isfinite(q).all() or not ((q>=0)&(q<=1)).all() or not math.isfinite(cutoff) or not 0<=cutoff<=1:
        raise ValueError('INVALID_FROZEN_Q_OR_PROBABILITIES')
    return np.column_stack((1-q<=cutoff,q<=cutoff))

def conformal(q,y,cutoff):
    q,y=aligned(q,y); sets=prediction_sets(q,cutoff)
    sizes=sets.sum(axis=1);covered=sets[np.arange(len(y)),y]
    # The all-tick indicator is exactly max_t(1-P(Y_t)) <= Q.
    score=np.where(y==0,1-q,q)
    simultaneous=bool(covered.all())
    if simultaneous != bool(score.max()<=cutoff): raise AssertionError('BLOCK_MAX_SET_INDICATOR_MISMATCH')
    return {'eligible_ticks':len(y),'covered_ticks':int(covered.sum()),'set_size_sum':int(sizes.sum()),
      'abstaining_ticks':int(np.sum(sizes!=1)),'empty_ticks':int(np.sum(sizes==0)),'ambiguous_ticks':int(np.sum(sizes==2)),
      'simultaneous_supported':simultaneous,'tick_coverage':float(covered.mean()),'mean_set_size':float(sizes.mean()),
      'abstention_rate':float(np.mean(sizes!=1)),'empty_set_rate':float(np.mean(sizes==0)),
      'ambiguous_set_rate':float(np.mean(sizes==2)),'maximum_true_label_candidate_score':float(score.max()),'comparison':'<='}

def strict_macro(rows):
    out={}
    for name in METRICS:
        undefined={sid:row['undefined_reasons'][name] for sid,row in rows.items() if row['values'][name] is None}
        out[name]={'value':None if undefined else statistics.mean(row['values'][name] for row in rows.values()),
          'assigned_scenarios':len(rows),'defined_scenarios':len(rows)-len(undefined),'undefined_scenario_ids_and_reasons':undefined}
    return out

def conformal_summary(rows):
    if not rows: raise ValueError('EMPTY_SCENARIO_GROUP')
    total=sum(r['eligible_ticks'] for r in rows.values())
    out={'scenario_count':len(rows),'eligible_tick_count':total,
      'scenario_simultaneous_coverage':statistics.mean(int(r['simultaneous_supported']) for r in rows.values()),
      'scenario_macro_tick_coverage':statistics.mean(r['tick_coverage'] for r in rows.values()),
      'pooled_tick_coverage':sum(r['covered_ticks'] for r in rows.values())/total,
      'equal_scenario_mean_set_size':statistics.mean(r['mean_set_size'] for r in rows.values()),
      'pooled_mean_set_size':sum(r['set_size_sum'] for r in rows.values())/total,
      'abstention_rate':statistics.mean(r['abstention_rate'] for r in rows.values()),
      'empty_set_rate':statistics.mean(r['empty_set_rate'] for r in rows.values()),
      'ambiguous_set_rate':statistics.mean(r['ambiguous_set_rate'] for r in rows.values())}
    names=('scenario_simultaneous_coverage','scenario_macro_tick_coverage','pooled_tick_coverage')
    out['signed_coverage_gap']={k:out[k]-.95 for k in names}
    out['undercoverage_gap']={k:max(0.,.95-out[k]) for k in names}
    return out

def paired_summary(values_a,values_b):
    missing=[str(s) for s in SEEDS if values_a.get(s) is None or values_b.get(s) is None]
    individual={str(s):None if str(s) in missing else values_a[s]-values_b[s] for s in SEEDS}
    if missing:
        return {'complete':False,'per_seed':individual,'undefined_seed_ids':missing,'mean':None,'median':None,
          'sample_SD':None,'min':None,'max':None,'conditional_seed_CI':None,'exact_two_sided_sign_flip_p':None}
    delta=list(individual.values());mean=statistics.mean(delta);sd=statistics.stdev(delta)
    p=sum(abs(statistics.mean(sign*d for sign,d in zip(pattern,delta)))>=abs(mean)-1e-12 for pattern in itertools.product((-1,1),repeat=5))/32
    return {'complete':True,'per_seed':individual,'undefined_seed_ids':[],'mean':mean,'median':statistics.median(delta),
      'sample_SD':sd,'min':min(delta),'max':max(delta),'conditional_seed_CI':[mean-2.776445105*sd/math.sqrt(5),mean+2.776445105*sd/math.sqrt(5)],
      'exact_two_sided_sign_flip_p':p,'sign_patterns':32,
      'interpretation':'Conditional initialization summaries on fixed final data; sign symmetry assumption; no .05 superiority inference or population confidence claim.'}

def aggregate(records, arrays, registry):
    """Reconstruct every final metric solely from bound numeric intermediates."""
    ids=sorted(records); scorers=[r['scorer_id'] for r in registry['scorers']]
    per={s:{sid:records[sid]['classification'][s] for sid in ids} for s in scorers}
    strict={s:strict_macro(per[s]) for s in scorers}
    cp={s:conformal_summary({sid:records[sid]['conformal'][s] for sid in ids}) for s in scorers}
    eligible=[sid for sid in ids if per[scorers[0]][sid]['positive_ticks'] and per[scorers[0]][sid]['negative_ticks']]
    excluded={sid:per[scorers[0]][sid]['undefined_reasons']['AUROC'] for sid in ids if sid not in eligible}
    secondary={'two_class_eligible_scenario_ids':eligible,'excluded_ids_and_reasons':excluded,'eligible_count':len(eligible),
      'assigned_count':len(ids),'two_class_eligible_macro_AUROC':{s:None if not eligible else statistics.mean(per[s][sid]['values']['AUROC'] for sid in eligible) for s in scorers}}
    labels=np.concatenate([arrays[sid]['labels'] for sid in ids]); pooled={}
    for row in registry['scorers']:
        s=row['scorer_id']; q=np.concatenate([arrays[sid]['q'][s] for sid in ids])
        pooled[s]=classification(q,labels,row['frozen_anomaly_threshold'])
    secondary['pooled_AUROC']={s:pooled[s]['values']['AUROC'] for s in scorers}
    paired={}
    for a,b in [('epistemic_gat','standard_gat'),('standard_gat','nograph'),('epistemic_gat','nograph')]:
        paired[a+'_minus_'+b]=paired_summary({seed:strict[f'{a}_{seed}']['AUROC']['value'] for seed in SEEDS},
          {seed:strict[f'{b}_{seed}']['AUROC']['value'] for seed in SEEDS})
    paired['primary']='epistemic_gat_minus_standard_gat';paired['endpoint']='strict scenario-macro AUROC'
    paired['ensemble_used']=False;paired['best_seed_selection']=False
    paired['development_context']='EpistemicGAT was not consistently superior during development.'
    paired['optional_descriptive_scenario_bootstrap']='Not requested; omitted under the preregistered optional procedure. No RNG is instantiated.'
    paired['conformal_descriptive_seed_means']={}
    for m in METHODS:
        means={k:statistics.mean(cp[f'{m}_{s}'][k] for s in SEEDS) for k in cp[f'{m}_{SEEDS[0]}'] if isinstance(cp[f'{m}_{SEEDS[0]}'][k],(int,float)) and k not in ('scenario_count','eligible_tick_count')}
        for gap in ('signed_coverage_gap','undercoverage_gap'):
            means[gap]={k:statistics.mean(cp[f'{m}_{s}'][gap][k] for s in SEEDS) for k in cp[f'{m}_{SEEDS[0]}'][gap]}
        paired['conformal_descriptive_seed_means'][m]=means
    subgroup={}
    for field in ('town','anomaly_type','directory_condition'):
        subgroup[field]={}
        for value in sorted({records[sid]['metadata'][field] for sid in ids}):
            members=[sid for sid in ids if records[sid]['metadata'][field]==value]
            subgroup[field][value]={'scenario_ids':members,'scenario_count':len(members),'descriptive_only':True,
              'sparse_qualification':'Single assigned scenario; descriptive only' if len(members)==1 else None,
              'classification':{s:strict_macro({sid:per[s][sid] for sid in members}) for s in scorers},
              'conformal':{s:conformal_summary({sid:records[sid]['conformal'][s] for sid in members}) for s in scorers}}
    return {'per_scorer_per_scenario_metrics.json':per,'strict_macro_metrics.json':strict,'pooled_metrics.json':pooled,
      'secondary_metrics.json':secondary,'conformal_metrics.json':cp,'subgroup_descriptive_metrics.json':subgroup,
      'paired_seed_comparison.json':paired}
