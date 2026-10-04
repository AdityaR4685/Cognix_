"""Offline v4 pre-access verification: fresh partition, frozen scorer and seals."""
import ast
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
READY='READY_FOR_SEPARATELY_AUTHORIZED_FRESH_SPLIT_FINAL_CONFORMAL_CAL_ATTEMPT_003'
ORIGINAL_CAL_SHA='652020b48796d8035d5b78aac9b3d1f869d9ebddcefa3963b2bea02f092bb858'
ORIGINAL_POOL_SHA='85bb93828421c53e54064bfc7de0eb355b5e016427749ada59ee4fffdec157da'
FRESH_PARTITION_SHA='36669903304cbe3ced226299bbcca257d6dfbb64d2f180085250874aeff01ce0'
V3_MANIFEST_SHA='b0429f1b239b72310f4e4ce641194daf09b020e46614f6518e3d7e1ff11e8b05'
V3_ZIP_SHA='5c41b658d2b2542dc4c75f1398b25ab94a7cf4c23d44a4fddc21e43d3ed562d2'
def require(ok,msg):
    if not ok:raise RuntimeError('PRE_ACCESS_REFUSAL: '+msg)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(name):return json.loads((HERE/name).read_text(encoding='utf-8'))
def ids(name):return (HERE/name).read_text().splitlines()
def function(s,name):
    node=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name==name)
    return ast.get_source_segment(s,node)
def check_manifest(d,n):
    require((d/(n+'.sha256')).read_text().split()==[sha(d/n),n],'detached seal: '+n)
    seen=set()
    for line in (d/n).read_text().splitlines():
        h,rel=line.split('  ',1);p=(d/rel).resolve()
        require(p.is_relative_to(d) and rel not in seen and p.is_file() and sha(p)==h,'sealed file: '+rel)
        seen.add(rel)
    return seen
def verify(require_ready=True):
    scope=check_manifest(HERE,'BUNDLE_SHA256SUMS')
    excluded={'BUNDLE_SHA256SUMS','BUNDLE_SHA256SUMS.sha256','cognix_final_conformal_cal_kaggle_v4.zip','cognix_final_conformal_cal_kaggle_v4.zip.sha256'}
    actual={p.relative_to(HERE).as_posix() for p in HERE.rglob('*') if p.is_file()}
    require(actual-excluded==scope,'complete sealed scope')
    d=HERE/'evidence/fresh_partition'
    require(sha(d/'FRESH_PARTITION_SHA256SUMS')==FRESH_PARTITION_SHA,'fresh partition seal fixed before v4')
    fresh_scope=check_manifest(d,'FRESH_PARTITION_SHA256SUMS')
    require({p.name for p in d.iterdir() if p.is_file()}-{'FRESH_PARTITION_SHA256SUMS','FRESH_PARTITION_SHA256SUMS.sha256'}==fresh_scope,'fresh evidence scope')
    binding=read('partition_binding.json')
    for n,key in [('final_conformal_cal.txt','CAL_membership_sha256'),('final_evaluation.txt','EVAL_membership_sha256'),('prior_exposed_cal.txt','prior_exposed_CAL_sha256'),('historical_exclusions.txt','historical_exclusions_sha256')]:require(sha(HERE/n)==binding[key],'role hash: '+n)
    cal,evaluation,prior,ex=map(ids,['final_conformal_cal.txt','final_evaluation.txt','prior_exposed_cal.txt','historical_exclusions.txt'])
    counts=[100,400,125,2];roles=list(map(set,[cal,evaluation,prior,ex]))
    require(all(len(row)==len(set(row))==n for row,n in zip([cal,evaluation,prior,ex],counts)),'four exact unique role counts')
    require(not any(roles[i]&roles[j] for i in range(4) for j in range(i+1,4)),'four-role overlap')
    require(len(set.union(*roles))==627,'four-role total')
    require(sha(HERE/'prior_exposed_cal.txt')==ORIGINAL_CAL_SHA and sha(d/'fresh_source_pool.txt')==ORIGINAL_POOL_SHA,'original CAL/source pool binding')
    pool=ids('evidence/fresh_partition/fresh_source_pool.txt');sorted_pool=ids('evidence/fresh_partition/fresh_source_pool_sorted.txt')
    require(len(pool)==len(set(pool))==500 and sorted_pool==sorted(pool),'exact opaque sorted source')
    require(roles[0]|roles[1]==set(pool) and not roles[2]&set(pool) and not roles[3]&set(pool),'fresh pool exact original EVAL and retired/excluded disjoint')
    require(sha(HERE/'final_conformal_cal.txt')==sha(d/'fresh_final_conformal_cal.txt') and sha(HERE/'final_evaluation.txt')==sha(d/'fresh_final_evaluation.txt'),'active vs sealed fresh membership')
    old=HERE/'evidence/partition'
    require(sha(old/'SHA256SUMS')=='e3047f2032896cc062f4b90eab9f7488151b9680e1b3c1234a81c112dcce577c','original aggregate partition seal')
    require(sha(old/'partition_membership.json')=='e41392264aae01e47c67b27fa8ad3a1c28c7c589c55e35e778d0502d4e567ada','original membership JSON')
    require(sha(old/'final_conformal_cal.txt')==ORIGINAL_CAL_SHA and sha(old/'final_evaluation.txt')==ORIGINAL_POOL_SHA,'original membership files')
    for line in (old/'SHA256SUMS').read_text().splitlines():
        h,rel=line.split('  ',1);p=old/rel
        if not p.exists() and rel.endswith('.py'):p=old/(rel+'.txt')
        require(p.resolve().is_relative_to(old) and sha(p)==h,'historical partition evidence')
    membership=read('evidence/fresh_partition/fresh_partition_membership.json');indices=membership['permutation_indices']
    # Verify the stored bijection/assignment, never instantiate or replay RNG.
    require(len(indices)==500 and all(type(i) is int for i in indices) and set(indices)==set(range(500)),'permutation bijection')
    require(cal==[sorted_pool[i] for i in indices[:100]] and evaluation==[sorted_pool[i] for i in indices[100:]],'single recorded permutation assignment')
    require(membership['FRESH_FINAL_CONFORMAL_CAL']==cal and membership['FRESH_FINAL_EVALUATION']==evaluation and membership['PRIOR_EXPOSED_CAL']==prior and membership['HISTORICAL_EXCLUDED']==ex,'membership role record')
    seed=read('evidence/fresh_partition/fresh_partition_seed.json');receipt=read('evidence/fresh_partition/fresh_partition_execution_receipt.json');prereg=read('evidence/fresh_partition/fresh_partition_preregistration.json')
    require(seed['seed_spent'] is True and seed['seed_generated_once'] is True and not seed['redraw_allowed'] and not seed['seed_search_allowed'],'seed permanently spent')
    require(receipt['seed']==seed['seed'] and receipt['permutation_count']==receipt['RNG_instantiation_count']==1 and receipt['seed_spent'] is True,'one draw only')
    require(receipt['seed_receipt_final_sha256']==sha(d/'fresh_partition_seed.json') and seed['preregistration_sha256']==sha(d/'fresh_partition_preregistration.json'),'seed chronology bindings')
    require(prereg['seed_generated_at_preregistration'] is False and prereg['permutations_at_preregistration']==0 and prereg['scorer_frozen_before_partition'] is True,'preregistered before draw')
    require(prereg['source_membership_sha256']==ORIGINAL_POOL_SHA and prereg['v3_manifest_sha256']==V3_MANIFEST_SHA and prereg['v3_ZIP_sha256']==V3_ZIP_SHA,'precursor freeze/source binding')
    require(prereg['partition_script_sha256']==sha(d/'fresh_partition.py'),'partition execution source binding')
    audit=read('evidence/fresh_partition/fresh_pool_scientific_untouched_audit.json')
    require(all(row['SCIENTIFIC_PAYLOAD_DECODE']==0 for row in audit['attempt_evidence'].values()) and audit['Attempt003_executed'] is False,'scientifically untouched pool evidence')
    same=read('unchanged_scorer_bindings.json')
    for rel,h in same['identical_v3_file_sha256'].items():require(sha(HERE/rel)==h,'identical v3 scorer file: '+rel)
    precursor=(HERE/'evidence/v3_runner.py.txt').read_text();active=(HERE/'runner.py').read_text()
    for name in ['map_segmentation_oov_to_frozen_vocabulary','runtime_check']:
        require(function(active,name)==function(precursor,name),'frozen v3 OOV/runtime function')
    require(hashlib.sha256(function(active,'map_segmentation_oov_to_frozen_vocabulary').encode()).hexdigest()==prereg['v3_OOV_function_source_sha256'],'OOV freeze before draw')
    rows=read('model_registry.json')['scorers'];require(len(rows)==15 and len({(r['method'],r['seed']) for r in rows})==15,'15 scorer identities')
    for row in rows:require(sha(HERE/row['bundle_state'])==row['bundle_state_sha256'] and row['strict_load_passed'],'fixed scorer state')
    protocol=read('fresh_conformal_protocol.json');spec=read('conformal_specification.json')
    require(spec==protocol and spec['n_cal']==100 and spec['alpha']==.05 and spec['rank_1_based']==96 and spec['threshold_count']==15,'fresh exact conformal specification')
    original=read('evidence/protocol/conformal_protocol.json')
    for key in ['alpha','numerics','score','scenario_score','eligible_ticks','prediction_set']:require(spec[key]==original[key],'unchanged score/scenario/ties: '+key)
    from calibration_adapter import threshold
    require(threshold([.4]*100)=={'alpha':.05,'n_cal':100,'rank_1_based':96,'Q':.4,'quantile_is_infinite':False,'augmentation':'+infinity','comparison':'<='},'adapter mechanical fresh rank')
    for rel,h in read('scientific_bindings.json')['bundle_scientific_file_sha256'].items():require(sha(HERE/rel)==h,'v4 scientific file seal')
    policy=read('protected_eval_policy.json')
    require(policy['roles']=={'FRESH_CAL':100,'FINAL_EVAL_OPAQUE_DISCARD':400,'PRIOR_EXPOSED_CAL_OPAQUE_DISCARD':125,'HISTORICAL_EXCLUDED_OPAQUE_DISCARD':2},'four provenance roles')
    require(policy['expected_compressed_size']==91538225599 and policy['expected_archive_sha256']=='267e48f2249deb0269ad950aa81bca57dc02e3bbdf2d73acc172af267b18254a','archive binding')
    require(policy['original_125_CAL_retired'] is True and not policy['training'] and not policy['tuning'] and not policy['membership_override'],'no tuning/oldCAL reuse')
    require(policy['output']=='/kaggle/working/cognix_final_conformal_cal_attempt003_v2','new output generation')
    validation=read('pre_access_validation.json')
    require(validation['groups_failed']==0 and validation['fresh_split_groups_passed']>=32 and validation['v3_applicable_groups_passed']==36,'all offline regressions')
    require(validation['Attempt003_executed'] is False and validation['TEST_network_requests']==validation['TEST_payload_bytes_accessed']==0,'preparation only')
    if require_ready:require(validation['readiness']==READY,'not ready')
    return {'status':'PASSED','manifest_sha256':sha(HERE/'BUNDLE_SHA256SUMS'),'sealed_files':len(scope),'fresh_CAL':100,'final_EVAL':400,'prior_exposed_CAL':125,'historical_excluded':2,'four_role_total':627,'n_cal':100,'k':96,'scorers':15,'Attempt003_executed':False,'TEST_access_performed':False,'readiness':validation['readiness']}
if __name__=='__main__':print(json.dumps(verify(),sort_keys=True))
