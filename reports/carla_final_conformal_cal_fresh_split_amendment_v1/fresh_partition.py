"""Offline single-use fresh partition. IDs are opaque; no payload/score access."""
import ast
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
V3=ROOT/'reports/carla_final_conformal_cal_bundle_v3'
HEAD='1f8cb8cca741ad3f3b471b2d5da5129e0309c30b'
CAL_SHA='652020b48796d8035d5b78aac9b3d1f869d9ebddcefa3963b2bea02f092bb858'
POOL_SHA='85bb93828421c53e54064bfc7de0eb355b5e016427749ada59ee4fffdec157da'
V3_MANIFEST='b0429f1b239b72310f4e4ce641194daf09b020e46614f6518e3d7e1ff11e8b05'
V3_ZIP='5c41b658d2b2542dc4c75f1398b25ab94a7cf4c23d44a4fddc21e43d3ed562d2'
OOV_SEAL='c42ce9e7854144a297faed54c08fbdec07c6806d4371bf36fd2360d315a6dc93'
NETWORK_EVENTS={'socket.connect','socket.connect_ex','socket.getaddrinfo','socket.gethostbyname','socket.gethostbyaddr','socket.sendto','socket.sendmsg','urllib.Request','http.client.connect','os.system','os.posix_spawn'}
def refuse(event,args):
    if event in NETWORK_EVENTS:raise RuntimeError('BLOCKED_NETWORK_EVENT: '+event)
    if event=='subprocess.Popen':
        a=args[1];a=a if isinstance(a,list) else a.split()
        if len(a)<2 or a[0]!='git' or a[1] not in {'rev-parse','status','diff','ls-files'}:raise RuntimeError('BLOCKED_CHILD_PROCESS')
sys.addaudithook(refuse)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def text(p,s):p.write_text(s,encoding='utf-8',newline='\n')
def write(p,o):text(p,json.dumps(o,sort_keys=True,indent=2,allow_nan=False)+'\n')
def exclusive(p,o):
    with p.open('x',encoding='utf-8',newline='\n') as f:
        f.write(json.dumps(o,sort_keys=True,indent=2,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def git(*a):return subprocess.check_output(['git',*a],cwd=ROOT).decode('utf-8').strip()
def require(ok,status):
    if not ok:raise RuntimeError(status)
def inventory(p):return {f.relative_to(p).as_posix():sha(f) for f in sorted(p.rglob('*')) if f.is_file()}
def verify_seal(d,n,rename_py=False):
    require((d/(n+'.sha256')).read_text().split()==[sha(d/n),n],'BLOCKED_EVIDENCE_DETACHED_SEAL_MISMATCH')
    seen=set()
    for line in (d/n).read_text().splitlines():
        h,rel=line.split('  ',1);p=(d/rel).resolve()
        if rename_py and not p.exists() and rel.endswith('.py'):p=(d/(rel+'.txt')).resolve()
        require(p.is_relative_to(d) and p.is_file() and rel not in seen and sha(p)==h,'BLOCKED_EVIDENCE_FILE_MISMATCH: '+rel)
        seen.add(rel)
def function_bytes(p,name):
    source=p.read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)
    return ast.get_source_segment(source,node).encode()
def bind_check():
    require(git('rev-parse','HEAD')==HEAD,'BLOCKED_HEAD_MISMATCH')
    require(not git('diff','--cached','--name-only'),'BLOCKED_INDEX_NOT_EMPTY')
    require(sha(V3/'BUNDLE_SHA256SUMS')==V3_MANIFEST and sha(V3/'cognix_final_conformal_cal_kaggle_v3.zip')==V3_ZIP,'BLOCKED_V3_BINDING_MISMATCH')
    require(sha(ROOT/'reports/carla_final_conformal_cal_attempt002_oov_amendment_v1/AMENDMENT_SHA256SUMS')==OOV_SEAL,'BLOCKED_V3_AMENDMENT_BINDING_MISMATCH')
    require(sha(V3/'final_conformal_cal.txt')==CAL_SHA and sha(V3/'final_evaluation.txt')==POOL_SHA,'BLOCKED_ORIGINAL_PARTITION_BINDING_MISMATCH')
    require(sha(V3/'evidence/partition/partition_membership.json')=='e41392264aae01e47c67b27fa8ad3a1c28c7c589c55e35e778d0502d4e567ada','BLOCKED_ORIGINAL_PARTITION_BINDING_MISMATCH')
    require(sha(V3/'evidence/partition/SHA256SUMS')=='e3047f2032896cc062f4b90eab9f7488151b9680e1b3c1234a81c112dcce577c','BLOCKED_ORIGINAL_PARTITION_BINDING_MISMATCH')

def preregister():
    import platform,numpy as np
    bind_check()
    require(not (HERE/'fresh_partition_preregistration.json').exists(),'BLOCKED_PREREGISTRATION_ALREADY_EXISTS')
    protected=['carla_final_conformal_cal_bundle_v1','carla_final_conformal_cal_bundle_v2','carla_final_conformal_cal_bundle_v3',
      'carla_final_conformal_cal_attempt001_failure_v1','carla_final_conformal_cal_attempt001_traceback_addendum_v1',
      'carla_final_conformal_cal_attempt002_failure_v1','carla_final_conformal_cal_attempt002_oov_amendment_v1']
    for name in ('carla_final_conformal_cal_bundle_v1','carla_final_conformal_cal_bundle_v2','carla_final_conformal_cal_bundle_v3'):
        verify_seal(ROOT/'reports'/name,'BUNDLE_SHA256SUMS')
    require(sha(ROOT/'reports/carla_final_conformal_cal_bundle_v2/BUNDLE_SHA256SUMS')=='dd5e559ebf16d73a67d7ecc720ec7f3200465c493e6f804479155e8b32007c66','BLOCKED_V2_BINDING_MISMATCH')
    require(sha(ROOT/'reports/carla_final_conformal_cal_bundle_v2/cognix_final_conformal_cal_kaggle_v2.zip')=='2f1eeb7aff36a6608b643ef1afca5621df5b621b8da79f83ba1232d83418db79','BLOCKED_V2_BINDING_MISMATCH')
    for name,seal in [('carla_final_conformal_cal_attempt001_failure_v1','FAILURE_SHA256SUMS'),('carla_final_conformal_cal_attempt001_traceback_addendum_v1','ADDENDUM_SHA256SUMS'),('carla_final_conformal_cal_attempt002_failure_v1','FAILURE_SHA256SUMS'),('carla_final_conformal_cal_attempt002_oov_amendment_v1','AMENDMENT_SHA256SUMS')]:verify_seal(ROOT/'reports'/name,seal)
    # Nested original evidence has historical .py->.py.txt preservation renaming.
    verify_seal(V3/'evidence/partition','SHA256SUMS',rename_py=True)
    oldcal=(V3/'final_conformal_cal.txt').read_text().splitlines();pool=(V3/'final_evaluation.txt').read_text().splitlines()
    exclusions=read(V3/'partition_binding.json')['historical_exclusions']
    require(len(oldcal)==len(set(oldcal))==125 and len(pool)==len(set(pool))==500 and len(exclusions)==2,'BLOCKED_ORIGINAL_PARTITION_BINDING_MISMATCH')
    require(not set(oldcal)&set(pool) and not (set(oldcal)|set(pool))&set(exclusions),'BLOCKED_ORIGINAL_PARTITION_OVERLAP')
    ledgers={
      'Attempt001':ROOT/'reports/carla_final_conformal_cal_attempt001_failure_v1/access_ledger.json',
      'Attempt002':ROOT/'reports/carla_final_conformal_cal_attempt002_failure_v1/immutable_output/access_ledger.json'}
    evidence={}
    for name,p in ledgers.items():
        obj=read(p);require(obj['eval_scenarios_decoded']==0,'BLOCKED_FRESH_POOL_SCIENTIFIC_UNTOUCHED_STATUS_UNPROVEN')
        evidence[name]={'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'SCIENTIFIC_PAYLOAD_DECODE':0,
          'EVAL_opaque_body_accounted_bytes':obj['opaque_body_bytes_discarded_by_role']['EVAL_OPAQUE_DISCARD'],
          'transport_is_not_scientific_decoding':obj['transport_is_not_scientific_decoding'],'http_requests':obj['http_requests']}
    faildir=ROOT/'reports/carla_final_conformal_cal_attempt002_failure_v1'
    archive=faildir/'cognix_final_conformal_cal_attempt002_failed_evidence.tar.gz'
    require(sha(archive)=='a21a6f2ea64eabcdb2c636f79da1cc18709146e583910cd97a52e991331221ea','BLOCKED_ATTEMPT002_EVIDENCE_MISMATCH')
    access2=read(ledgers['Attempt002']);failure=read(faildir/'immutable_output/failure_exception_audit.json')
    require(access2['compressed_bytes_received']==62739644426 and access2['scenario_roots_seen']==455 and access2['cal_scenarios_completed']==92 and access2['range_requests']==0 and not access2['raw_archive_retained'],'BLOCKED_ATTEMPT002_EVIDENCE_MISMATCH')
    require(access2['compressed_sha256_partial_or_complete']=='5c48384eee84003a00e672de9265e27c3f8a9634deed04250a0b852780c5ba91' and failure['status']=='INCOMPLETE_NO_RETRY','BLOCKED_ATTEMPT002_EVIDENCE_MISMATCH')
    review=read(ROOT/'reports/carla_final_conformal_cal_v3_review.json')
    require(review['Attempt003_executed'] is False and review['TEST_network_requests']==review['TEST_payload_bytes_accessed']==0,'BLOCKED_FRESH_POOL_SCIENTIFIC_UNTOUCHED_STATUS_UNPROVEN')
    require(not list((ROOT/'reports').glob('*attempt003*')),'BLOCKED_UNEXPECTED_ATTEMPT003_ARTIFACT')
    # Source-path proof: EVAL bodies never enter the callback; callback gates CAL first.
    for version in (1,2,3):
        source=(ROOT/f'reports/carla_final_conformal_cal_bundle_v{version}/streaming.py').read_text()
        require('if sid in cal:' in source and 'else:\n                    discard(size)' in source and 'on_cal(sid, relative, read_exact(size))' in source,'BLOCKED_FRESH_POOL_SCIENTIFIC_UNTOUCHED_STATUS_UNPROVEN')
        runner=(ROOT/f'reports/carla_final_conformal_cal_bundle_v{version}/runner.py').read_text()
        require(runner.index('if sid not in self.cal:')<runner.index('from PIL import Image'),'BLOCKED_FRESH_POOL_SCIENTIFIC_UNTOUCHED_STATUS_UNPROVEN')
    earlier={}
    for p in [ROOT/'reports/carla_test_structure_inventory_v1/access_ledger.json',ROOT/'reports/carla_test_structure_inventory_v1/full_traversal_v1/access_ledger.json']:
        obj=read(p)
        require(obj.get('sensor_or_label_interpretation') is False or obj.get('payload_persisted_decoded_inspected_or_separately_hashed') is False,'BLOCKED_FRESH_POOL_SCIENTIFIC_UNTOUCHED_STATUS_UNPROVEN')
        earlier[p.relative_to(ROOT).as_posix()]={'sha256':sha(p),'scientific_decoding':False,'scope':'historical structural transport ledger'}
    pool_audit={'status':'SCIENTIFIC_PAYLOAD_DECODE_ZERO_ESTABLISHED_BY_PRESERVED_EVIDENCE_AND_CODE_GATES',
      'original_EVAL_count':500,'original_EVAL_sha256':POOL_SHA,'attempt_evidence':evidence,'earlier_structural_transport_evidence':earlier,
      'Attempt003_executed':False,'no_later_scientific_access_evidence_found':True,
      'later_access_basis':'Current explicit human statement that Attempt003 has not occurred, immutable v3 preparation/review counters, absence of local Attempt003 output, and known local access ledgers. No global or remote-system non-access proof is asserted.',
      'transport_limitation':'Prior EVAL opaque bytes transited/decompressed. SCIENTIFIC_PAYLOAD_DECODE=0 is evidence/code-path scoped; not NO BYTES EVER TRANSITED.',
      'score_values_used':False,'no_new_TEST_payload_access':True}
    write(HERE/'fresh_pool_scientific_untouched_audit.json',pool_audit)
    paths=git('ls-files','--others','--exclude-standard','-z').split('\0')
    meta={p:[(ROOT/p).stat().st_size,(ROOT/p).stat().st_mtime_ns] for p in paths if p and not p.startswith(HERE.relative_to(ROOT).as_posix()+'/')}
    baseline={'HEAD':HEAD,'protected_directory_file_sha256':{n:inventory(ROOT/'reports'/n) for n in protected},
      'v3_review_sha256':sha(ROOT/'reports/carla_final_conformal_cal_v3_review.json'),'unrelated_untracked_metadata':meta,
      'index':git('ls-files','--stage').splitlines(),'git_status_before':git('status','--porcelain=v1').splitlines()}
    write(HERE/'preservation_baseline.json',baseline)
    (HERE/'gitattributes_before.bin').write_bytes((ROOT/'.gitattributes').read_bytes())
    source_scientific=read(V3/'scientific_bindings.json')['bundle_scientific_file_sha256']
    for rel,h in source_scientific.items():require(sha(V3/rel)==h,'BLOCKED_V3_SCIENTIFIC_BINDING_MISMATCH')
    prereg={'status':'IMMUTABLE_PREREGISTRATION_BEFORE_SEED_OR_MEMBERSHIP','HEAD':HEAD,'V3_status':'V3_UNEXECUTED_SUPERSEDED_BY_FRESH_SPLIT_PROTOCOL',
      'v3_manifest_sha256':V3_MANIFEST,'v3_ZIP_sha256':V3_ZIP,'v3_amendment_manifest_sha256':OOV_SEAL,
      'scorer_frozen_before_partition':True,'v3_OOV_function_source_sha256':hashlib.sha256(function_bytes(V3/'runner.py','map_segmentation_oov_to_frozen_vocabulary')).hexdigest(),
      'v3_scientific_file_sha256':source_scientific,'v3_OOV_policy_record_sha256':sha(V3/'segmentation_oov_amendment.json'),
      'source':'exact original 500 FINAL_EVALUATION IDs','source_membership_sha256':POOL_SHA,'retired_original_CAL_sha256':CAL_SHA,
      'source_count':500,'fresh_CAL_count':100,'fresh_EVAL_count':400,'cal_fraction':'ceil(500/5)=100',
      'algorithm':['read exact original final_evaluation.txt, require 500 unique opaque strings','lexicographically sort IDs',
        'generate exactly one secrets.randbits(128) seed, immediately persist exclusive receipt','instantiate numpy.random.Generator(numpy.random.PCG64(recorded_seed))',
        'perform exactly one permutation of indices 0..499','first 100 permuted IDs -> FRESH_FINAL_CONFORMAL_CAL','remaining 400 -> FRESH_FINAL_EVALUATION'],
      'seed_generated_at_preregistration':False,'permutations_at_preregistration':0,'RNG':'PCG64','seed_bits':128,
      'redraw_allowed':False,'seed_search_allowed':False,'balancing':False,'stratification':False,'semantic_ID_parsing':False,'outcome_use':False,
      'Python':platform.python_version(),'NumPy':np.__version__,'partition_unit':'scenario','alpha':0.05,'n_cal':100,'rank_1_based':96,
      'scientific_scope':'Fixed v3 scorer/OOV; formal partition/count/derived rank amended. Old 125 permanently retired and opaque.',
      'fresh_pool_audit_sha256':sha(HERE/'fresh_pool_scientific_untouched_audit.json'),'partition_script_sha256':sha(Path(__file__)),
      'validity':'Conditional on fixed scorer before random assignment, scientifically undecoded original EVAL pool, uniform assignment and no post-draw adaptations: finite-catalogue scenario-level marginal randomization argument, not conditional on the realized seed/split/calibration sample; no subgroup or external-distribution guarantee.',
      'TEST_network_requests':0,'TEST_payload_bytes_accessed':0,'Attempt003_executed':False}
    exclusive(HERE/'fresh_partition_preregistration.json',prereg)
    text(HERE/'fresh_partition_preregistration.json.sha256',sha(HERE/'fresh_partition_preregistration.json')+'  fresh_partition_preregistration.json\n')
    print(json.dumps({'status':'PREREGISTERED_BEFORE_DRAW','preregistration_sha256':sha(HERE/'fresh_partition_preregistration.json'),'seed_generated':False,'TEST_requests':0}))

def check_prereg():
    bind_check();p=HERE/'fresh_partition_preregistration.json'
    require((HERE/'fresh_partition_preregistration.json.sha256').read_text().split()==[sha(p),p.name],'BLOCKED_PREREGISTRATION_HASH_MISMATCH')
    require(read(p)['partition_script_sha256']==sha(Path(__file__)),'BLOCKED_PARTITION_SCRIPT_CHANGED_AFTER_PREREGISTRATION')
    require(sha(HERE/'fresh_pool_scientific_untouched_audit.json')==read(p)['fresh_pool_audit_sha256'],'BLOCKED_FRESH_POOL_AUDIT_CHANGED')

def generate_seed():
    import platform,numpy as np
    check_prereg();p=HERE/'fresh_partition_seed.json'
    require(not p.exists() and not (HERE/'fresh_partition_execution_started.json').exists(),'BLOCKED_SEED_ALREADY_GENERATED_NO_REDRAW')
    pool=(V3/'final_evaluation.txt').read_text().splitlines();sorted_bytes=('\n'.join(sorted(pool))+'\n').encode()
    seed=secrets.randbits(128)  # Exactly one local CSPRNG seed; no search or alternatives.
    exclusive(p,{'seed':seed,'seed_bits':128,'seed_generated_once':True,'redraw_allowed':False,'seed_search_allowed':False,'seed_spent':False,
      'Python':platform.python_version(),'NumPy':np.__version__,'RNG':'PCG64','source_membership_sha256':POOL_SHA,
      'sorted_source_list_sha256':hashlib.sha256(sorted_bytes).hexdigest(),'preregistration_sha256':sha(HERE/'fresh_partition_preregistration.json'),
      'algorithm':read(HERE/'fresh_partition_preregistration.json')['algorithm'],'source_count':500,'fresh_CAL_count':100,'fresh_EVAL_count':400})
    print(json.dumps({'status':'SEED_GENERATED_ONCE_AND_DURABLY_RECORDED','seed_receipt_sha256':sha(p),'seed_spent':False}))

def draw():
    import numpy as np
    check_prereg();p=HERE/'fresh_partition_seed.json';seed=read(p)
    require(seed['seed_generated_once'] and not seed['seed_spent'] and not (HERE/'fresh_partition_execution_started.json').exists(),'BLOCKED_SEED_SPENT_OR_DRAW_ALREADY_STARTED_NO_RERUN')
    require(seed['preregistration_sha256']==sha(HERE/'fresh_partition_preregistration.json'),'BLOCKED_SEED_BINDING_MISMATCH')
    require(seed['NumPy']==np.__version__,'BLOCKED_PARTITION_NUMPY_VERSION_CHANGED')
    output_names=['fresh_source_pool.txt','fresh_source_pool_sorted.txt','fresh_final_conformal_cal.txt','fresh_final_evaluation.txt','fresh_partition_membership.json','fresh_partition_execution_receipt.json','fresh_partition_validation.json']
    require(not any((HERE/n).exists() for n in output_names),'BLOCKED_FRESH_PARTITION_OUTPUT_EXISTS')
    pool=(V3/'final_evaluation.txt').read_text().splitlines();require(len(pool)==len(set(pool))==500,'BLOCKED_SOURCE_COUNT_OR_DUPLICATES')
    sorted_pool=sorted(pool)
    require(hashlib.sha256(('\n'.join(sorted_pool)+'\n').encode()).hexdigest()==seed['sorted_source_list_sha256'],'BLOCKED_SORTED_SOURCE_HASH_MISMATCH')
    initial_seed_sha=sha(p)
    exclusive(HERE/'fresh_partition_execution_started.json',{'status':'DRAW_PERMANENTLY_RESERVED_NO_RERUN','seed_receipt_pre_spend_sha256':initial_seed_sha,'preregistration_sha256':seed['preregistration_sha256'],'permutations_allowed':1})
    rng=np.random.Generator(np.random.PCG64(seed['seed']))
    permutation=rng.permutation(500)  # EXACTLY ONE permutation; do not replay or redraw.
    cal=[sorted_pool[int(i)] for i in permutation[:100]]
    evaluation=[sorted_pool[int(i)] for i in permutation[100:]]
    oldcal=(V3/'final_conformal_cal.txt').read_text().splitlines();exclusions=read(V3/'partition_binding.json')['historical_exclusions']
    require(len(cal)==len(set(cal))==100 and len(evaluation)==len(set(evaluation))==400 and not set(cal)&set(evaluation) and set(cal)|set(evaluation)==set(pool),'BLOCKED_FRESH_PARTITION_INVALID_NO_REDRAW')
    require(not set(oldcal)&set(pool) and not set(exclusions)&set(pool),'BLOCKED_FRESH_PARTITION_OVERLAP_NO_REDRAW')
    (HERE/'fresh_source_pool.txt').write_bytes((V3/'final_evaluation.txt').read_bytes())
    text(HERE/'fresh_source_pool_sorted.txt','\n'.join(sorted_pool)+'\n')
    text(HERE/'fresh_final_conformal_cal.txt','\n'.join(cal)+'\n');text(HERE/'fresh_final_evaluation.txt','\n'.join(evaluation)+'\n')
    write(HERE/'fresh_partition_membership.json',{'source_pool':'original_FINAL_EVALUATION','source_count':500,
      'FRESH_FINAL_CONFORMAL_CAL':cal,'FRESH_FINAL_EVALUATION':evaluation,'PRIOR_EXPOSED_CAL':oldcal,'HISTORICAL_EXCLUDED':exclusions,
      'permutation_indices':permutation.tolist(),'semantic_ID_parsing':False,'composition_inspected':False})
    seed['seed_spent']=True;seed['partition_success']=True;write(p,seed)
    hashes={n:sha(HERE/n) for n in ['fresh_source_pool.txt','fresh_source_pool_sorted.txt','fresh_final_conformal_cal.txt','fresh_final_evaluation.txt','fresh_partition_membership.json','fresh_partition_seed.json']}
    write(HERE/'fresh_partition_execution_receipt.json',{'status':'COMPLETE_SINGLE_DRAW_SEED_SPENT','seed':seed['seed'],'seed_generated_once':True,'seed_spent':True,
      'permutation_count':1,'RNG_instantiation_count':1,'redraw_allowed':False,'seed_search_allowed':False,'no_RNG_replay_for_verification':True,
      'preregistration_sha256':seed['preregistration_sha256'],'seed_receipt_pre_spend_sha256':initial_seed_sha,'seed_receipt_final_sha256':sha(p),
      'Python':seed['Python'],'NumPy':seed['NumPy'],'RNG':'PCG64','algorithm':seed['algorithm'],'source_count':500,'CAL_count':100,'EVAL_count':400,
      'output_sha256':hashes,'scores_or_payloads_accessed':False,'scenario_IDs_treated_as_opaque':True,'TEST_network_requests':0,'TEST_payload_bytes_accessed':0,'Attempt003_executed':False})
    write(HERE/'fresh_partition_validation.json',{'status':'PASSED','source_count':500,'fresh_CAL_count':100,'fresh_EVAL_count':400,'prior_exposed_CAL_count':125,'historical_exclusion_count':2,
      'four_role_total':627,'CAL_EVAL_overlap':0,'fresh_union_equals_exact_original_EVAL':True,'old_CAL_fresh_pool_overlap':0,'historical_exclusion_fresh_pool_overlap':0,'duplicates':0,
      'single_draw_irrevocably_accepted':True,'descriptive_composition_counts_computed':False,'seed_spent':True,'n_cal':100,'alpha':0.05,'rank_1_based':96})
    names=output_names+['fresh_partition_seed.json','fresh_partition_preregistration.json','fresh_partition_preregistration.json.sha256','fresh_partition_execution_started.json','fresh_pool_scientific_untouched_audit.json','fresh_partition.py']
    text(HERE/'FRESH_PARTITION_SHA256SUMS',''.join(sha(HERE/n)+'  '+n+'\n' for n in sorted(names)))
    text(HERE/'FRESH_PARTITION_SHA256SUMS.sha256',sha(HERE/'FRESH_PARTITION_SHA256SUMS')+'  FRESH_PARTITION_SHA256SUMS\n')
    print(json.dumps({'status':'FRESH_PARTITION_SEALED_NO_REDRAW','CAL':100,'EVAL':400,'prior_CAL':125,'historical_exclusions':2,'k':96,
      'seed_spent':True,'fresh_partition_seal_sha256':sha(HERE/'FRESH_PARTITION_SHA256SUMS'),'composition_not_inspected':True}))

if __name__=='__main__':
    modes={'--preregister':preregister,'--generate-seed':generate_seed,'--draw-once':draw}
    require(len(sys.argv)==2 and sys.argv[1] in modes,'BLOCKED_EXPLICIT_OFFLINE_PARTITION_MODE_REQUIRED')
    modes[sys.argv[1]]()
