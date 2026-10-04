"""Numeric provenance and finalization gates. Never retains TEST frames."""
import hashlib
import json
from pathlib import Path
import numpy as np
from metrics import aligned, classification, conformal, aggregate

HERE=Path(__file__).resolve().parent
EXPECTED_SIZE=91538225599
EXPECTED_SHA='267e48f2249deb0269ad950aa81bca57dc02e3bbdf2d73acc172af267b18254a'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,obj):p.write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def metadata(sid):
    from header_helpers import classify
    _,row,_,_=classify(sid,b'5')
    return row

def frozen_thresholds():
    binding=read(HERE/'protocol_bindings.json')
    for name,key in [('conformal_thresholds.json','threshold_sha256'),('threshold_calculation_receipts.json','threshold_receipts_sha256')]:
        if sha(HERE/name)!=binding[key]:raise ValueError('FROZEN_THRESHOLD_HASH_MISMATCH')
    return read(HERE/'conformal_thresholds.json')

def record_scenario(output,sid,labels,probabilities):
    # A second membership gate protects all metrics and persisted intermediates.
    if sid not in set((HERE/'final_evaluation.txt').read_text().splitlines()):raise ValueError('METRIC_CALLBACK_REJECTS_NON_FINAL_EVAL')
    registry=read(HERE/'model_registry.json')['scorers']; thresholds=frozen_thresholds()
    scorers={r['scorer_id'] for r in registry}
    if set(probabilities)!=scorers:raise ValueError('MISSING_FROZEN_SCORER_OUTPUT')
    classification_rows={};cp_rows={}
    for row in registry:
        s=row['scorer_id']; q,y=aligned(probabilities[s],labels)
        classification_rows[s]=classification(q,y,row['frozen_anomaly_threshold'])
        cp_rows[s]=conformal(q,y,thresholds[s]['Q'])
    name=hashlib.sha256(sid.encode()).hexdigest()
    numeric=output/'numeric_intermediates'/ (name+'.npz'); numeric.parent.mkdir(exist_ok=True)
    if numeric.exists():raise ValueError('DUPLICATE_SCIENTIFIC_COMPLETION')
    # Raw float32 frozen q and official integer labels reproduce all metrics.
    for values in probabilities.values():
        if np.asarray(values).dtype!=np.float32:raise ValueError('FROZEN_GRAPH_Q_MUST_BE_FLOAT32')
    np.savez_compressed(numeric,labels=np.asarray(labels,dtype=np.int64),ticks=np.arange(1,len(labels)+1,dtype=np.int64),
      **{s:np.asarray(probabilities[s]) for s in sorted(scorers)})
    record={'scenario_id':sid,'partition':'FINAL_EVAL','metadata':metadata(sid),'eligible_ticks':len(labels),'tick_0_excluded':True,
      'classification':classification_rows,'conformal':cp_rows,'numeric_artifact':numeric.relative_to(output).as_posix(),'numeric_sha256':sha(numeric)}
    destination=output/'per_scenario_evidence'/(name+'.json');destination.parent.mkdir(exist_ok=True)
    save(destination,record)
    with (output/'eval_scenario_processing_ledger.jsonl').open('a',encoding='utf-8',newline='\n') as f:
        f.write(json.dumps({'scenario_id':sid,'partition':'FINAL_EVAL','eligible_ticks':len(labels),'tick_0_excluded':True,
          'metadata':record['metadata'],'numeric_artifact':record['numeric_artifact'],'numeric_sha256':record['numeric_sha256'],
          'metric_artifact':destination.relative_to(output).as_posix(),'metric_sha256':sha(destination)},sort_keys=True,allow_nan=False)+'\n')
    return record

def require_completion(completed,evaluation,receipt,ledger):
    if len(evaluation)!=400 or len(completed)!=400 or set(completed)!=set(evaluation):raise ValueError('FINAL_EVAL_INCOMPLETE_NO_METRICS_FINALIZED')
    if receipt.get('compressed_bytes')!=EXPECTED_SIZE or receipt.get('sha256')!=EXPECTED_SHA or receipt.get('gzip_crc_validated') is not True or receipt.get('tar_complete') is not True:
        raise ValueError('ARCHIVE_INTEGRITY_REFUSAL_NO_METRICS_FINALIZED')
    expected={'FINAL_EVAL':400,'FRESH_CAL_OPAQUE_DISCARD':100,'PRIOR_EXPOSED_CAL_OPAQUE_DISCARD':125,'HISTORICAL_EXCLUDED_OPAQUE_DISCARD':2}
    if ledger.get('scenario_roots_seen')!=627 or ledger.get('scenario_roots_seen_by_role')!=expected:raise ValueError('FOUR_ROLE_RECONCILIATION_FAILED')
    for k in ('fresh_cal_scenarios_decoded','prior_exposed_cal_scenarios_decoded','historical_excluded_scenarios_decoded'):
        if ledger.get(k)!=0:raise ValueError('PROTECTED_ROLE_SCIENTIFIC_DECODE')
    if ledger.get('http_requests')!=1 or ledger.get('range_requests')!=0 or ledger.get('automatic_retry') is not False or ledger.get('raw_archive_retained') is not False:
        raise ValueError('ONE_SHOT_ACCESS_POLICY_FAILED')
    if ledger.get('compressed_bytes_received')!=EXPECTED_SIZE or ledger.get('compressed_sha256_partial_or_complete')!=EXPECTED_SHA:raise ValueError('ACCESS_RECEIPT_MISMATCH')
    accounted={'FINAL_EVAL':62847018051,'FRESH_CAL_OPAQUE_DISCARD':16288609930,'HISTORICAL_EXCLUDED_OPAQUE_DISCARD':240747299,'PRIOR_EXPOSED_CAL_OPAQUE_DISCARD':19633431385}
    if ledger.get('body_bytes_accounted_by_role')!=accounted:raise ValueError('ROLE_BODY_ACCOUNTING_FAILED')
    discarded=ledger.get('opaque_body_bytes_actually_discarded_by_role',{})
    if any(discarded.get(role)!=count for role,count in accounted.items() if role!='FINAL_EVAL'):raise ValueError('PROTECTED_ROLE_DISCARD_ACCOUNTING_FAILED')

def reconstruct(output,completed):
    registry=read(HERE/'model_registry.json');thresholds=frozen_thresholds(); arrays={}; verified={}
    for sid,record in completed.items():
        p=(output/record['numeric_artifact']).resolve()
        if not p.is_relative_to(output.resolve()) or sha(p)!=record['numeric_sha256']:raise ValueError('NUMERIC_INTERMEDIATE_INTEGRITY_FAILED')
        with np.load(p,allow_pickle=False) as z:
            scorers={r['scorer_id'] for r in registry['scorers']}
            if set(z.files)!=scorers|{'labels','ticks'} or z['labels'].dtype!=np.int64 or z['ticks'].dtype!=np.int64 or not np.array_equal(z['ticks'],np.arange(1,len(z['labels'])+1)):
                raise ValueError('NUMERIC_INTERMEDIATE_SCHEMA_FAILED')
            labels=z['labels'].copy();q={s:z[s].copy() for s in scorers}
        c={};cp={}
        for row in registry['scorers']:
            s=row['scorer_id']
            if q[s].dtype!=np.float32:raise ValueError('FROZEN_Q_PRECISION_CHANGED')
            c[s]=classification(q[s],labels,row['frozen_anomaly_threshold']);cp[s]=conformal(q[s],labels,thresholds[s]['Q'])
        if c!=record['classification'] or cp!=record['conformal'] or record['metadata']!=metadata(sid) or record['eligible_ticks']!=len(labels) or record['partition']!='FINAL_EVAL' or record['tick_0_excluded'] is not True:
            raise ValueError('SCENARIO_METRIC_RECONSTRUCTION_FAILED')
        arrays[sid]={'labels':labels,'q':q};verified[sid]={**record,'classification':c,'conformal':cp}
    return aggregate(verified,arrays,registry)

def seal_results(output):
    name='RESULT_SHA256SUMS'
    files=sorted(p for p in output.rglob('*') if p.is_file() and p.name not in {name,name+'.sha256'})
    (output/name).write_text(''.join(sha(p)+'  '+p.relative_to(output).as_posix()+'\n' for p in files),encoding='utf-8',newline='\n')
    (output/(name+'.sha256')).write_text(sha(output/name)+'  '+name+'\n',encoding='utf-8',newline='\n')
