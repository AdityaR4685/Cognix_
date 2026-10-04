"""Offline cryptographic/scientific binding preflight; never loads TEST."""
import ast
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
READY='READY_FOR_SEPARATELY_AUTHORIZED_FINAL_EVALUATION_ATTEMPT_001'
ZIP='cognix_final_evaluation_kaggle_v1.zip'
EXCLUDED={'BUNDLE_SHA256SUMS','BUNDLE_SHA256SUMS.sha256',ZIP,ZIP+'.sha256'}

def require(ok,msg):
    if not ok:raise RuntimeError('PRE_ACCESS_REFUSAL: '+msg)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(name):return json.loads((HERE/name).read_text(encoding='utf-8-sig'))
def function(s,name):
    node=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name==name)
    return ast.get_source_segment(s,node)
def manifest(d,name,original_name=None):
    original_name=original_name or name
    require((d/(name+'.sha256')).read_text().split()==[sha(d/name),original_name],'detached seal: '+name)
    rows={}
    for line in (d/name).read_text().splitlines():
        h,rel=line.split('  ',1);p=(d/rel).resolve()
        require(p.is_relative_to(d.resolve()) and rel not in rows and len(h)==64,'safe unique manifest scope')
        rows[rel]=h
    return rows

def verify(require_seal=True):
    b=read('protocol_bindings.json')
    require(b['v4_manifest_sha256']=='7cfa0339d64e4d0350e8dd5b0940913e68d95e4d0aaffb2108593a11f87e9a39','v4 binding')
    require(b['threshold_sha256']=='5e0aaf4c7af9eee0e42e323ec48ffe0b66e328d8267f8307055cdda8231e897d','frozen successful threshold hash')
    require(b['attempt003_evidence_archive_sha256']=='cf9b956ecd37ccf1762f0ef0a588cda15665ddc377d97e2adaca7f850ecb3cd9','Attempt003 external binding')
    require(b['readiness']==READY and b['FINAL_EVALUATION_ATTEMPT']=='001' and b['CALIBRATION_SOURCE']=='successful Attempt003 v4','separate namespace')
    if require_seal:
        rows=manifest(HERE,'BUNDLE_SHA256SUMS')
        require({p.relative_to(HERE).as_posix() for p in HERE.rglob('*') if p.is_file()}-EXCLUDED==set(rows),'exact sealed file scope')
        for rel,h in rows.items():require(sha(HERE/rel)==h,'sealed bytes: '+rel)
        regression=read('offline_regression_results.json')
        require(regression['groups_failed']==0 and regression['groups_passed']>=38 and regression['actual_network_calls']==0 and regression['real_TEST_bytes_used']==0,'offline regression readiness')
    prior=HERE/'provenance'
    require(sha(prior/'v4_BUNDLE_SHA256SUMS')==b['v4_manifest_sha256'],'preexisting v4 manifest')
    v4=manifest(prior,'v4_BUNDLE_SHA256SUMS','BUNDLE_SHA256SUMS')
    for rel,h in b['identical_v4_files'].items():require(sha(HERE/rel)==h==v4[rel],'unchanged scorer/checkpoint/parameters: '+rel)
    require(sha(prior/'preregistered_SHA256SUMS')==b['frozen_spec_manifest_sha256']=='e0f20f0b012eff901acd0e46884635cc904a5be88c72bbfcffe5bb0d0a1c2593','preexisting frozen spec manifest')
    prereg=manifest(prior,'preregistered_SHA256SUMS','SHA256SUMS')
    for rel,row in b['frozen_spec_bindings'].items():require(sha(HERE/rel)==row['sha256']==prereg[rel],'preregistered spec: '+rel)
    binding=read('partition_binding.json')
    roles=[]
    for name,count,key,h in [('final_evaluation.txt',400,'EVAL_membership_sha256','9410f13cc563362bf5ef7ed52333807131bea4fe5a5d51a13c1924e4c855a60a'),
      ('final_conformal_cal.txt',100,'CAL_membership_sha256','bb75bd2244c4ab4b25b246e738db281cb5371d63e57a9632c72dd320fe638736'),
      ('prior_exposed_cal.txt',125,'prior_exposed_CAL_sha256','652020b48796d8035d5b78aac9b3d1f869d9ebddcefa3963b2bea02f092bb858'),
      ('historical_exclusions.txt',2,'historical_exclusions_sha256','44f99c4238263adc090b64e8c2da8b7063b72d727aa15d3921c49df4982a20de')]:
        ids=(HERE/name).read_text().splitlines();require(len(ids)==len(set(ids))==count and sha(HERE/name)==binding[key]==h,'role membership: '+name);roles.append(set(ids))
    require(not any(roles[i]&roles[j] for i in range(4) for j in range(i+1,4)),'four-role overlap')
    inventory=set((HERE/'evidence/inventory/scenario_inventory_sorted.txt').read_text().splitlines())
    require(len(inventory)==627 and set.union(*roles)==inventory,'exact inventory union')
    source=(HERE/'scientific_runtime.py').read_text(); precursor=(prior/'v4_runner.py.txt').read_text()
    require(sha(prior/'v4_runner.py.txt')==v4['runner.py'] and sha(prior/'v4_streaming.py.txt')==v4['streaming.py'],'bound v4 scientific pipeline sources')
    for name,h in b['source_function_sha256'].items():
        require(function(source,name)==function(precursor,name) and hashlib.sha256(function(source,name).encode()).hexdigest()==h,'frozen OOV/runtime: '+name)
    labels=function((HERE/'label_adapter.py').read_text(),'labels_from_table')
    require(hashlib.sha256(labels.encode()).hexdigest()==b['label_function_sha256'],'official label function')
    require(read('label_semantics.json')['official_class_order']==['normal=0','anomaly=1'],'official class order')
    registry=read('model_registry.json')['scorers'];runs=read('frozen_method_bindings.json')['runs']
    require(len(registry)==len(runs)==15 and {(r['method'],r['seed']) for r in registry}=={(m,s) for m in ('nograph','standard_gat','epistemic_gat') for s in (101,202,303,404,505)},'all fixed method/seed identities')
    for row in registry:
        run=next(r for r in runs if (r['method'],r['seed'])==(row['method'],row['seed']))
        require(all(row[k]==run[k] for k in ['checkpoint','content_sha256','file_sha256','selected_epoch','frozen_anomaly_threshold']),'development checkpoint/hard threshold binding')
        require(sha(HERE/row['bundle_state'])==row['bundle_state_sha256'],'model state hash')
    thresholds=read('conformal_thresholds.json');receipts=read('threshold_calculation_receipts.json')
    require(sha(HERE/'conformal_thresholds.json')==b['threshold_sha256'] and sha(HERE/'threshold_calculation_receipts.json')==b['threshold_receipts_sha256'],'active threshold/receipt hash')
    require(set(thresholds)==set(receipts)=={r['scorer_id'] for r in registry},'15 independent cutoff identities')
    for s,t in thresholds.items():
        require(t['n_cal']==100 and t['alpha']==.05 and t['rank_1_based']==96 and t['comparison']=='<=' and t['quantile_is_infinite'] is False and 0<=t['Q']<=1,'frozen conformal structure')
        require(all(receipts[s][k]==v for k,v in t.items()) and receipts[s]['identical_cal_membership_sha256']==binding['CAL_membership_sha256'] and receipts[s]['method_selection_performed'] is False,'receipt/calibration binding')
    results=manifest(prior,'Attempt003_RESULT_SHA256SUMS','RESULT_SHA256SUMS')
    require(results['conformal_thresholds.json']==b['threshold_sha256'] and results['threshold_calculation_receipts.json']==b['threshold_receipts_sha256'],'original result seal threshold entries')
    require(not (HERE/'per_scorer_scenario_scores.json').exists() and not (HERE/'calibration_adapter.py').exists(),'no active CAL score/fitting dependency')
    require(read('v4_amendment_effect_audit.json')['scientific_choice_required'] is False,'amendment compatibility')
    notebook=read('cognix_final_evaluation_v1.ipynb');cells='\n'.join(''.join(c['source']) for c in notebook['cells'])
    require('EXECUTE_SEPARATELY_AUTHORIZED_FINAL_EVALUATION = False' in cells and all(not c.get('outputs') and c.get('execution_count') is None for c in notebook['cells'] if c['cell_type']=='code'),'notebook disabled/unexecuted')
    require(b['preparation_TEST_requests']==b['preparation_TEST_payload_bytes']==0 and b['final_evaluation_executed'] is False,'preparation only')
    for p in HERE.rglob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
    return {'status':'PASSED','readiness':READY,'threshold_count':15,'final_EVAL':400,'protected_fresh_CAL':100,'four_role_total':627,
      'threshold_sha256':b['threshold_sha256'],'final_evaluation_executed':False,'TEST_access_performed':False,
      'runtime_check':'Must additionally pass the exact v4 lock on the future execution host'}

if __name__=='__main__':print(json.dumps(verify(),sort_keys=True))
