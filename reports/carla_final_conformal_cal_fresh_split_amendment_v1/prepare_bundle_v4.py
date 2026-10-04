"""Offline v4 construction/verification. Never executes a scientific runner."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
V3=ROOT/'reports/carla_final_conformal_cal_bundle_v3'
V4=ROOT/'reports/carla_final_conformal_cal_bundle_v4'
HEAD='1f8cb8cca741ad3f3b471b2d5da5129e0309c30b'
READY='READY_FOR_SEPARATELY_AUTHORIZED_FRESH_SPLIT_FINAL_CONFORMAL_CAL_ATTEMPT_003'
OUTPUT='/kaggle/working/cognix_final_conformal_cal_attempt003_v2'
ZIP='cognix_final_conformal_cal_kaggle_v4.zip'
def refuse(event,args):
    if event in {'socket.connect','socket.connect_ex','socket.getaddrinfo','socket.gethostbyname','socket.gethostbyaddr','socket.sendto','socket.sendmsg','urllib.Request','http.client.connect','os.system','os.posix_spawn'}:raise RuntimeError('BLOCKED_OFFLINE_NETWORK_EVENT: '+event)
    if event=='subprocess.Popen':
        a=args[1];a=a if isinstance(a,list) else a.split()
        allowed=a[0]=='git' and a[1] in {'rev-parse','status','diff','ls-files','check-attr','log'}
        allowed=allowed or (a[0].strip('"')==sys.executable and '-B' in a and any('offline_checks.py' in x for x in a))
        if not allowed:raise RuntimeError('BLOCKED_UNGUARDED_CHILD_PROCESS')
sys.addaudithook(refuse)
def require(ok,msg):
    if not ok:raise RuntimeError(msg)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def text(p,s):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s,encoding='utf-8',newline='\n')
def write(p,o):text(p,json.dumps(o,sort_keys=True,indent=2,allow_nan=False)+'\n')
def git(*a):return subprocess.check_output(['git',*a],cwd=ROOT).decode('utf-8').strip()
def inventory(p):return {f.relative_to(p).as_posix():sha(f) for f in sorted(p.rglob('*')) if f.is_file()}
def seal(d,n):
    text(d/n,''.join(sha(p)+'  '+p.relative_to(d).as_posix()+'\n' for p in sorted(d.rglob('*')) if p.is_file() and p.name not in {n,n+'.sha256',ZIP,ZIP+'.sha256'}))
    text(d/(n+'.sha256'),sha(d/n)+'  '+n+'\n')
def verify_seal(d,n):
    require((d/(n+'.sha256')).read_text().split()==[sha(d/n),n],'BLOCKED_DETACHED_SEAL_MISMATCH')
    seen=set()
    for line in (d/n).read_text().splitlines():
        h,rel=line.split('  ',1);p=(d/rel).resolve()
        require(p.is_relative_to(d) and rel not in seen and p.is_file() and sha(p)==h,'BLOCKED_SEALED_FILE_MISMATCH: '+rel)
        seen.add(rel)
def function_source(p,name):
    s=p.read_text();node=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name==name);return ast.get_source_segment(s,node)
def assert_preserved():
    baseline=read(HERE/'preservation_baseline.json')
    require(git('rev-parse','HEAD')==HEAD,'BLOCKED_HEAD_MISMATCH')
    require(git('ls-files','--stage').splitlines()==baseline['index'],'BLOCKED_GIT_INDEX_CHANGED')
    require(not git('diff','--cached','--name-only'),'BLOCKED_STAGED_CHANGES')
    require(git('diff','--name-only').splitlines()==['.gitattributes'],'BLOCKED_UNEXPECTED_TRACKED_CHANGE')
    for name,files in baseline['protected_directory_file_sha256'].items():require(inventory(ROOT/'reports'/name)==files,'BLOCKED_SEALED_PRIOR_BYTES_CHANGED: '+name)
    require(sha(ROOT/'reports/carla_final_conformal_cal_v3_review.json')==baseline['v3_review_sha256'],'BLOCKED_V3_REVIEW_CHANGED')
    for p,m in baseline['unrelated_untracked_metadata'].items():
        stat=(ROOT/p).stat();require([stat.st_size,stat.st_mtime_ns]==m,'BLOCKED_UNRELATED_FILE_CHANGED: '+p)
    return baseline
def offline(base,target,*args):
    cmd=[sys.executable,'-B',str(base/'offline_checks.py'),str(base/target),*args]
    r=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
    record={'command':cmd,'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
    if r.returncode:
        write(HERE/'offline_preparation_failure.json',record)
        raise RuntimeError(json.dumps(record))
    rows=[json.loads(line) for line in r.stdout.splitlines() if line.strip()]
    record.update(result=rows[-2],network_guard=rows[-1]);require(rows[-1]['offline_network_guard']=='PASSED','BLOCKED_NETWORK_GUARD_FAILURE')
    return record

def prepare():
    assert_preserved();require(not V4.exists(),'BLOCKED_V4_ALREADY_EXISTS')
    verify_seal(HERE,'FRESH_PARTITION_SHA256SUMS')
    partition=read(HERE/'fresh_partition_validation.json');require(partition['seed_spent'] and partition['fresh_CAL_count']==100 and partition['fresh_EVAL_count']==400,'BLOCKED_FRESH_PARTITION_INVALID')
    # All old evidence remains historical; no Attempt002 score ledger enters the bundle.
    omit={'BUNDLE_SHA256SUMS','BUNDLE_SHA256SUMS.sha256','cognix_final_conformal_cal_kaggle_v3.zip','cognix_final_conformal_cal_kaggle_v3.zip.sha256','cognix_final_conformal_cal_v3.ipynb','pre_access_validation.json','offline_test_transcripts.json','synthetic_verification_results.json','regression_verification_results.json','oov_regression_verification_results.json','unchanged_scientific_bindings.json','README.md','amendment_binding.json'}
    V4.mkdir()
    for p in sorted(V3.iterdir()):
        if p.name in omit:continue
        if p.is_dir():shutil.copytree(p,V4/p.name)
        else:shutil.copyfile(p,V4/p.name)
    for rel,name in [('runner.py','v3_runner.py.txt'),('streaming.py','v3_streaming.py.txt'),('scientific_bindings.json','v3_scientific_bindings.json')]:shutil.copyfile(V3/rel,V4/'evidence'/name)
    fresh_evidence=V4/'evidence/fresh_partition';fresh_evidence.mkdir()
    for line in (HERE/'FRESH_PARTITION_SHA256SUMS').read_text().splitlines():
        _,rel=line.split('  ',1);shutil.copyfile(HERE/rel,fresh_evidence/rel)
    for rel in ['FRESH_PARTITION_SHA256SUMS','FRESH_PARTITION_SHA256SUMS.sha256']:shutil.copyfile(HERE/rel,fresh_evidence/rel)
    # Stored partition script is evidence only and is not an allowed verification target.
    shutil.copyfile(V3/'final_conformal_cal.txt',V4/'prior_exposed_cal.txt')
    shutil.copyfile(HERE/'fresh_final_conformal_cal.txt',V4/'final_conformal_cal.txt')
    shutil.copyfile(HERE/'fresh_final_evaluation.txt',V4/'final_evaluation.txt')
    excluded=read(V3/'partition_binding.json')['historical_exclusions'];text(V4/'historical_exclusions.txt','\n'.join(excluded)+'\n')
    binding={'CAL_count':100,'EVAL_count':400,'PRIOR_EXPOSED_CAL_count':125,'historical_exclusion_count':2,'four_role_total':627,
      'CAL_membership_sha256':sha(V4/'final_conformal_cal.txt'),'EVAL_membership_sha256':sha(V4/'final_evaluation.txt'),
      'prior_exposed_CAL_sha256':sha(V4/'prior_exposed_cal.txt'),'historical_exclusions_sha256':sha(V4/'historical_exclusions.txt'),'historical_exclusions':excluded,
      'source_pool_sha256':sha(HERE/'fresh_source_pool.txt'),'sorted_source_sha256':sha(HERE/'fresh_source_pool_sorted.txt'),
      'fresh_partition_seal_sha256':sha(HERE/'FRESH_PARTITION_SHA256SUMS'),'fresh_partition_receipt_sha256':sha(HERE/'fresh_partition_execution_receipt.json'),
      'preregistration_sha256':sha(HERE/'fresh_partition_preregistration.json'),'seed_receipt_sha256':sha(HERE/'fresh_partition_seed.json'),'seed_spent':True,
      'overlap':0,'fresh_union_equals_original_EVAL':True,'old_CAL_retired':True,'RNG_replayed':False}
    write(V4/'partition_binding.json',binding)
    protocol=read(V3/'conformal_specification.json')
    protocol['n_cal']=100;protocol['rank_1_based']=96
    protocol['calibration']=protocol['calibration'].replace('FINAL_CONFORMAL_CAL','FRESH_FINAL_CONFORMAL_CAL')
    protocol['source']='fresh_split_amendment_v1; sealed v3 scorer; pre-seed preregistration'
    protocol['source_sha256']=sha(HERE/'fresh_partition_preregistration.json')
    protocol['primary_guarantee']='Conditional on scorer/OOV frozen before fresh random assignment, original 500-EVAL pool scientifically undecoded, one uniform scenario assignment and no outcome-dependent adaptations: finite-catalogue marginal randomization statement over the new assignment and a uniformly selected remaining evaluation scenario. For n_cal=100, augmented rank k=96 gives at least 96/101 >= .95 with conservative ties under the uniform-assignment premise. This is not a claim conditional on the realized seed, split or calibration sample. No iid whole-dataset or future-CARLA, subgroup, town/type, joint-all-scorers or all-EVAL-scenarios guarantee. Scenario is the randomization unit; timesteps need not be independent.'
    protocol['scope_limits'].append('Old 125 CAL are retired after post-access OOV amendment; recomputation does not restore freshness. Prior opaque EVAL transport occurred; scientific decoder access was zero according to preserved evidence/code gates. Seeded PCG64 implements the specified random assignment; the validity statement remains conditional on the intended uniform-assignment premise.')
    write(HERE/'fresh_conformal_protocol.json',protocol);write(V4/'fresh_conformal_protocol.json',protocol);write(V4/'conformal_specification.json',protocol)
    record={'status':'V3_UNEXECUTED_SUPERSEDED_BY_FRESH_SPLIT_PROTOCOL','HEAD':HEAD,'v3_manifest_sha256':sha(V3/'BUNDLE_SHA256SUMS'),'v3_ZIP_sha256':sha(V3/'cognix_final_conformal_cal_kaggle_v3.zip'),
      'v3_amendment_manifest_sha256':sha(ROOT/'reports/carla_final_conformal_cal_attempt002_oov_amendment_v1/AMENDMENT_SHA256SUMS'),
      'v3_never_executed':True,'v3_scorer_and_OOV_remain_authoritative':True,'definition':'SCORER_AND_FEATURE_RULE_IDENTICAL_TO_SEALED_V3; FORMAL_CALIBRATION_PARTITION_AND_DERIVED_CONFORMAL_RANK_AMENDED.',
      'why':'Original CAL informed a post-access preprocessing amendment; none of the 92 score values was used, but recomputing the old 125 does not restore a fresh formal calibration sample. All old 125 are permanently retired.',
      'Attempt002_external_evidence_sha256':'a21a6f2ea64eabcdb2c636f79da1cc18709146e583910cd97a52e991331221ea','Attempt002_partial_archive_sha256':'5c48384eee84003a00e672de9265e27c3f8a9634deed04250a0b852780c5ba91',
      'class32_identity':'UNKNOWN','92_partial_scores_used':False,'original_125_CAL_retired':True,'original_500_EVAL_scientific_payload_decode':0,
      'fresh_partition':binding,'unchanged':['v3 OOV and 29D segmentation','Camera/IMU','15 checkpoints/states','fitted one-class parameters','graph architecture/weights','UQ/normality/label semantics','score 1-P(Y)','scenario max t>=1','alpha .05','inclusive <=','archive URL/size/SHA','runtime pins/settings','no tuning/training','EVAL protection','single attempt/no ranking'],
      'changed':['formal CAL/EVAL source','counts 125/500 ->100/400','old125 -> prior exposed opaque role','derived rank120 ->96','fresh seed/membership/seals','four-role scanner accounting','output/provenance paths'],
      'ATTEMPT_NUMBER':'003','EXECUTION_PROTOCOL_VERSION':'v4','OUTPUT_GENERATION':'attempt003_v2','output':OUTPUT,
      'chronology':['original protocol','Attempt001 failure','v2 repair','Attempt002 partial CAL exposure/OOV failure','v3 OOV sealed, unexecuted','single fresh random split of original EVAL500','v4 fresh protocol','later separately authorized Attempt003'],
      'TEST_network_requests':0,'TEST_payload_bytes_accessed':0,'Attempt003_executed':False}
    write(HERE/'fresh_split_amendment_record.json',record);write(V4/'fresh_split_amendment_record.json',record)
    text(HERE/'report.md','# Fresh calibration protocol amendment\n\n'+
      'V3_UNEXECUTED_SUPERSEDED_BY_FRESH_SPLIT_PROTOCOL. V3 remains authoritative for the fixed scorer and OOV rule. The issue is adaptive use of the original calibration set after a post-access preprocessing amendment, not a claim that the v3 scorer is scientifically wrong.\n\n'+
      'Attempt001 failed; Attempt002 failed after 92/125 original CAL scenarios completed. Their ledgers report SCIENTIFIC_PAYLOAD_DECODE=0 for original EVAL. Prior EVAL bodies did physically transit/decompress, without image/Feather/label/feature/UQ/model entry according to preserved ledgers and source gates. Local evidence and current human attestation establish the intended untouched scientific pool; no OS-wide or remote-system access proof is claimed.\n\n'+
      'All original 125 CAL are permanently PRIOR_EXPOSED_CAL and never decoded again for formal CAL, EVAL, tuning, thresholds or model comparison. The 92 partial CAL score values were not read/analyzed or reused. Class32 meaning UNKNOWN. Recomputing those 125 does not restore a fresh calibration sample.\n\n'+
      'V3 scorer/OOV seals were verified before preregistration, seed generation and a single PCG64 permutation of opaque original 500-EVAL IDs. Exactly one secrets.randbits(128) seed was durably recorded. Seed spent; no redraw, search, balancing, stratification, semantic ID parsing, composition inspection or outcomes. Fresh CAL100 plus EVAL400 exactly equals original EVAL500; neither old CAL125 nor exclusions2 overlaps. Partition membership is sealed; no RNG replay in verification.\n\n'+
      'The scientific score remains 1-P_t(Y_t), scenario maximum for t>=1. Alpha=.05; augmented100 scores plus infinity; k=ceil(101*.95)=96; inclusive <=; one pooled threshold independently for each of15 fixed scorers. No subgroup tuning, method ranking or checkpoint/refit change.\n\n'+
      protocol['primary_guarantee']+'\n\n'+
      'Fresh CAL will be decoded only during later separately authorized Attempt003, execution protocol v4, output attempt003_v2. Four roles: FRESH_CAL100; FINAL_EVAL_OPAQUE_DISCARD400; PRIOR_EXPOSED_CAL_OPAQUE_DISCARD125; HISTORICAL_EXCLUDED_OPAQUE_DISCARD2. Only fresh CAL can enter any scientific decoder. V1/v2/v3 and historical evidence remain immutable. Preparation makes zero TEST requests/accesses zero TEST payload bytes.\n')
    # Minimal scanner patch: preserve transport/parser/caps; add disjoint fourth role.
    old=(V3/'streaming.py').read_text()
    new=old.replace('def scan(source, cal, evaluation, exclusions, on_cal, ledger, compressed_limit):','def scan(source, cal, evaluation, exclusions, on_cal, ledger, compressed_limit, *, prior_exposed_cal=frozenset()):')
    new=new.replace('    if cal & evaluation or (cal | evaluation) & exclusions:\n        raise ValueError("INVALID_MEMBERSHIP")',
      '    roles = (cal, evaluation, prior_exposed_cal, exclusions)\n    if any(roles[i] & roles[j] for i in range(4) for j in range(i + 1, 4)):\n        raise ValueError("INVALID_MEMBERSHIP")')
    new=new.replace('    body_bytes = Counter()','    body_bytes = Counter()\n    opaque_discarded = Counter()\n    seen_roles = {}')
    new=new.replace('    def discard(n):','    def discard(n, role=None):').replace('            read_exact(part)\n            n -= part','            read_exact(part)\n            if role is not None:\n                opaque_discarded[role] += part\n            n -= part')
    new=new.replace('role = "CAL"','role = "FRESH_CAL"').replace('role = "EVAL_OPAQUE_DISCARD"','role = "FINAL_EVAL_OPAQUE_DISCARD"').replace('            elif sid in exclusions:\n                role = "EXCLUDED_OPAQUE_DISCARD"','            elif sid in prior_exposed_cal:\n                role = "PRIOR_EXPOSED_CAL_OPAQUE_DISCARD"\n            elif sid in exclusions:\n                role = "HISTORICAL_EXCLUDED_OPAQUE_DISCARD"')
    new=new.replace('            if aliases.setdefault(sid, transformation)', '            seen_roles[sid] = role\n            if aliases.setdefault(sid, transformation)')
    new=new.replace('                        discard(size)','                        discard(size, role=role)').replace('                    discard(size)','                    discard(size, role=role)')
    new=new.replace('if seen != cal | evaluation | exclusions:','if seen != cal | evaluation | prior_exposed_cal | exclusions:')
    new=new.replace('"opaque_body_bytes_discarded_by_role": dict(body_bytes),','"body_bytes_accounted_by_role": dict(body_bytes),\n                       "opaque_body_bytes_actually_discarded_by_role": dict(opaque_discarded),\n                       "scenario_roots_seen_by_role": dict(Counter(seen_roles.values())),\n                       "prior_exposed_cal_scenarios_decoded": 0,\n                       "historical_excluded_scenarios_decoded": 0,')
    new=new.replace('EVAL/excluded bodies','Final EVAL/prior-exposed CAL/historical-excluded bodies')
    text(V4/'streaming.py',new)
    old=(V3/'runner.py').read_text()
    new=old.replace('/kaggle/working/cognix_final_conformal_cal_attempt003_v1',OUTPUT)
    new=new.replace('SCIENTIFIC_CALLBACK_REJECTS_NON_CAL','SCIENTIFIC_CALLBACK_REJECTS_NON_FRESH_CAL').replace('"partition": "FINAL_CONFORMAL_CAL"','"partition": "FRESH_FINAL_CONFORMAL_CAL"')
    insertion='''def require_fresh_cal_completion(completed, cal):
    if len(cal) != 100 or len(completed) != 100 or set(completed) != set(cal):
        raise ValueError("FRESH_CAL_INCOMPLETE_NO_THRESHOLDS_FINALIZED")


'''
    new=new.replace('def execute():',insertion+'def execute():',1)
    new=new.replace('    processor = CalProcessor(models, agents, cal, OUTPUT, "cuda:0")','    prior_exposed_cal = frozenset((HERE / "prior_exposed_cal.txt").read_text().splitlines())\n    processor = CalProcessor(models, agents, cal, OUTPUT, "cuda:0")')
    new=new.replace('scan(response, cal, evaluation, exclusions, processor, ledger, EXPECTED_SIZE)','scan(response, cal, evaluation, exclusions, processor, ledger, EXPECTED_SIZE, prior_exposed_cal=prior_exposed_cal)')
    new=new.replace('        if set(processor.completed) != cal:\n            raise ValueError("CAL_INCOMPLETE_NO_THRESHOLDS_FINALIZED")','        require_fresh_cal_completion(processor.completed, cal)')
    new=new.replace('"cal_scenarios": 125','"cal_scenarios": 100')
    new=new.replace('"cal_only_science": True,','"cal_only_science": True, "fresh_cal_only_science": True,\n              "prior_exposed_cal_scenarios_decoded": 0, "historical_excluded_scenarios_decoded": 0,')
    new=new.replace('"decoder_entry_policy": "CAL gate in scanner and processor"','"prior_exposed_cal_scenarios_decoded": 0, "historical_excluded_scenarios_decoded": 0,\n             "decoder_entry_policy": "FRESH_CAL gate in scanner and processor"')
    text(V4/'runner.py',new)
    # Exact inverse prevents incidental scoring/runtime/transport changes.
    inverse=new.replace(OUTPUT,'/kaggle/working/cognix_final_conformal_cal_attempt003_v1').replace('SCIENTIFIC_CALLBACK_REJECTS_NON_FRESH_CAL','SCIENTIFIC_CALLBACK_REJECTS_NON_CAL').replace('"partition": "FRESH_FINAL_CONFORMAL_CAL"','"partition": "FINAL_CONFORMAL_CAL"').replace(insertion,'')
    inverse=inverse.replace('    prior_exposed_cal = frozenset((HERE / "prior_exposed_cal.txt").read_text().splitlines())\n','').replace(', prior_exposed_cal=prior_exposed_cal)',')')
    inverse=inverse.replace('        require_fresh_cal_completion(processor.completed, cal)','        if set(processor.completed) != cal:\n            raise ValueError("CAL_INCOMPLETE_NO_THRESHOLDS_FINALIZED")')
    inverse=inverse.replace('"cal_scenarios": 100','"cal_scenarios": 125').replace('"cal_only_science": True, "fresh_cal_only_science": True,\n              "prior_exposed_cal_scenarios_decoded": 0, "historical_excluded_scenarios_decoded": 0,','"cal_only_science": True,')
    inverse=inverse.replace('"prior_exposed_cal_scenarios_decoded": 0, "historical_excluded_scenarios_decoded": 0,\n             "decoder_entry_policy": "FRESH_CAL gate in scanner and processor"','"decoder_entry_policy": "CAL gate in scanner and processor"')
    require(inverse==old,'BLOCKED_UNDECLARED_RUNNER_CHANGE')
    for name in ['map_segmentation_oov_to_frozen_vocabulary','runtime_check']:
        require(function_source(V4/'runner.py',name)==function_source(V3/'runner.py',name),'BLOCKED_V3_SCORER_OR_RUNTIME_FUNCTION_CHANGED')
    # Prior synthetic assertions are retained except path/gate terminology due to protocol role rename.
    replacements=[]
    for rel in ['synthetic_verification.py','regression_verification.py','oov_regression_verification.py']:
        s=(V3/rel).read_text();t=s.replace('SCIENTIFIC_CALLBACK_REJECTS_NON_CAL','SCIENTIFIC_CALLBACK_REJECTS_NON_FRESH_CAL').replace('/kaggle/working/cognix_final_conformal_cal_attempt003_v1',OUTPUT).replace('cognix_final_conformal_cal_v3.ipynb','cognix_final_conformal_cal_v4.ipynb')
        # Static completion assertion must point at the new exact100 production guard.
        t=t.replace("segment.index('CAL_INCOMPLETE_NO_THRESHOLDS_FINALIZED')","segment.index('require_fresh_cal_completion(processor.completed, cal)')")
        text(V4/rel,t);replacements.append({'file':rel,'v3_sha256':sha(V3/rel),'changes':'Fresh callback refusal text, notebook/output paths and completion-helper static check only; scoring assertions retained'})
    write(HERE/'prior_test_protocol_adaptations.json',replacements)
    nb=read(V3/'cognix_final_conformal_cal_v3.ipynb')
    for cell in nb['cells']:cell['source']=[s.replace('(v3)','(v4)').replace('bundle_v3','bundle_v4').replace('attempt003_v1','attempt003_v2') for s in cell['source']]
    nb['cells'][0]['source']=['# COGNIX fresh-split FINAL_CONFORMAL_CAL Attempt003, protocol v4\n','Preparation only; scorer/OOV frozen by sealed unexecuted v3. Fresh CAL100/EVAL400; old CAL125 permanently opaque. Preflight separately; execution defaults False and needs later authorization. No automatic install/download. Extract verified ZIP to /kaggle/working/cognix_cal_bundle_v4. Output: '+OUTPUT+'.\n']
    write(V4/'cognix_final_conformal_cal_v4.ipynb',nb)
    offline_source=(V3/'offline_checks.py').read_text().replace('"oov_regression_verification.py", "runner.py"','"oov_regression_verification.py", "fresh_split_verification.py", "runner.py"')
    text(V4/'offline_checks.py',offline_source)
    scientific=read(V3/'scientific_bindings.json');scientific['preparation_context']=record['definition'];scientific['fresh_partition_binding']=binding
    for rel in ['runner.py','streaming.py','conformal_specification.json']:scientific['bundle_scientific_file_sha256'][rel]=sha(V4/rel)
    write(V4/'scientific_bindings.json',scientific)
    same={rel:h for rel,h in read(V3/'scientific_bindings.json')['bundle_scientific_file_sha256'].items() if rel not in ['runner.py','streaming.py','conformal_specification.json']}
    for rel,h in same.items():require(sha(V4/rel)==sha(V3/rel)==h,'BLOCKED_FROZEN_SCORER_HASH_CHANGED')
    same.update({'runtime_requirements.txt':sha(V3/'runtime_requirements.txt'),'model_registry.json':sha(V3/'model_registry.json'),'segmentation_oov_amendment.json':sha(V3/'segmentation_oov_amendment.json')})
    for row in read(V4/'model_registry.json')['scorers']:require(sha(ROOT/row['checkpoint'])==row['file_sha256'],'BLOCKED_ORIGINAL_CHECKPOINT_HASH_CHANGED')
    write(V4/'unchanged_scorer_bindings.json',{'identity_statement':record['definition'],'identical_v3_file_sha256':same,'v3_manifest_sha256':record['v3_manifest_sha256'],
      'v3_OOV_function_source_sha256':hashlib.sha256(function_source(V4/'runner.py','map_segmentation_oov_to_frozen_vocabulary').encode()).hexdigest(),
      'runtime_function_source_identical':True,'runner_exact_inverse_passed':True,'original_checkpoint_registry':read(V4/'model_registry.json')['scorers'],
      'amended_protocol':record['changed'],'unchanged_semantics':record['unchanged']})
    policy=read(V3/'protected_eval_policy.json');policy['output']=OUTPUT;policy['roles']={'FRESH_CAL':100,'FINAL_EVAL_OPAQUE_DISCARD':400,'PRIOR_EXPOSED_CAL_OPAQUE_DISCARD':125,'HISTORICAL_EXCLUDED_OPAQUE_DISCARD':2}
    policy['gate']='Exact four-role immutable membership before any scientific callback; only FRESH_CAL admitted; CalProcessor rejects all non-FRESH_CAL'
    policy['original_125_CAL_retired']=True;policy['thresholds_require_exact_fresh_CAL_count']=100;policy['rank_1_based']=96
    policy['body_accounting_caveat']='body_bytes_accounted_by_role includes announced non-directory body size before decode/discard. opaque_body_bytes_actually_discarded_by_role records completed opaque body chunks, excluding padding. Historical v1/v2 ledgers remain unchanged.'
    write(V4/'protected_eval_policy.json',policy)
    before=(HERE/'gitattributes_before.bin').read_bytes();require((ROOT/'.gitattributes').read_bytes()==before,'BLOCKED_ATTRIBUTES_CHANGED_DURING_PREPARATION')
    extra=b'reports/carla_final_conformal_cal_fresh_split_amendment_v1/** -text\nreports/carla_final_conformal_cal_bundle_v4/** -text\n'
    (ROOT/'.gitattributes').write_bytes(before+extra)
    print(json.dumps({'status':'V4_SOURCE_PREPARED_NOT_EXECUTED','scorer_and_OOV_identical_to_v3':True,'CAL':100,'EVAL':400,'prior_CAL':125,'k':96,'Attempt003_executed':False}))

def finalize():
    import platform,numpy,PIL,pyarrow,pandas,torch
    baseline=assert_preserved();require(not (V4/'BUNDLE_SHA256SUMS').exists(),'BLOCKED_REFUSE_RESEAL_FINISHED_V4')
    # Record successful tests in full; guards prevent child networking and execute().
    transcripts={}
    for version,base,tests in [('v3',V3,['verify_bundle.py','synthetic_verification.py','regression_verification.py','oov_regression_verification.py']),
      ('v4',V4,['synthetic_verification.py','regression_verification.py','oov_regression_verification.py','fresh_split_verification.py'])]:
        for name in tests:
            print(json.dumps({'offline_check_started':version+'/'+name}),flush=True)
            record=offline(base,name);record['verification_source_sha256']=sha(base/name);record['guard_source_sha256']=sha(base/'offline_checks.py')
            transcripts[version+'/'+name]=record
    cmd=[sys.executable,'-B',str(V4/'offline_checks.py'),'--guard-selftest']
    r=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True);require(r.returncode==0,'BLOCKED_NETWORK_GUARD_SELFTEST')
    rows=[json.loads(line) for line in r.stdout.splitlines()]
    transcripts['v4/guard_selftest']={'command':cmd,'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'result':rows[-2],'network_guard':rows[-1]}
    counts={version:sum(record['result'].get('synthetic_verification_groups_passed',record['result'].get('regression_groups_passed',record['result'].get('oov_regression_groups_passed',record['result'].get('fresh_split_groups_passed',0)))) for key,record in transcripts.items() if key.startswith(version+'/')) for version in ('v3','v4')}
    require(counts=={'v3':36,'v4':70},'BLOCKED_REGRESSION_GROUP_COUNT_MISMATCH')
    fresh=transcripts['v4/fresh_split_verification.py']['result']
    previous=read(V3/'oov_regression_verification_results.json')['fitted_parameter_array_sha256']
    require(fresh['fitted_parameter_array_sha256']==previous,'BLOCKED_FITTED_PARAMETER_ARRAY_HASH_CHANGED')
    for key,name in [('synthetic_verification.py','synthetic_verification_results.json'),('regression_verification.py','regression_verification_results.json'),('oov_regression_verification.py','oov_regression_verification_results.json'),('fresh_split_verification.py','fresh_split_verification_results.json')]:write(V4/name,transcripts['v4/'+key]['result'])
    write(V4/'offline_test_transcripts.json',transcripts)
    runtime={'Python':platform.python_version(),'NumPy':numpy.__version__,'Pillow':PIL.__version__,'PyArrow':pyarrow.__version__,'pandas':pandas.__version__,'PyTorch':torch.__version__}
    record=read(HERE/'fresh_split_amendment_record.json');binding=read(V4/'partition_binding.json');seed=read(HERE/'fresh_partition_seed.json')
    oldscanner=(V3/'streaming.py').read_text();newscanner=(V4/'streaming.py').read_text()
    # Transport HashReader and caps are exact v3. Only the preregistered role/accounting patch differs.
    oldtree=ast.parse(oldscanner);newtree=ast.parse(newscanner)
    for name in ['CHUNK','MAX_MEMBER','MAX_UNCOMPRESSED','HashReader']:
        def node(tree):
            return next(n for n in tree.body if (isinstance(n,ast.ClassDef) and n.name==name) or (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)))
        require(ast.dump(node(oldtree),include_attributes=False)==ast.dump(node(newtree),include_attributes=False),'BLOCKED_UNDECLARED_TRANSPORT_CHANGE')
    for name in ['map_segmentation_oov_to_frozen_vocabulary','runtime_check']:require(function_source(V3/'runner.py',name)==function_source(V4/'runner.py',name),'BLOCKED_SCORER_RUNTIME_CHANGED')
    for folder in ['frozen_sources','states','upstream','fixtures','evidence/partition','evidence/protocol','evidence/inventory']:require(inventory(V3/folder)==inventory(V4/folder),'BLOCKED_FROZEN_V3_DIRECTORY_CHANGED')
    same=read(V4/'unchanged_scorer_bindings.json')
    for rel,h in same['identical_v3_file_sha256'].items():require(sha(V4/rel)==sha(V3/rel)==h,'BLOCKED_IDENTICAL_V3_FILE_MISMATCH')
    checkpoint_hashes={}
    for row in read(V4/'model_registry.json')['scorers']:
        h=sha(ROOT/row['checkpoint']);require(h==row['file_sha256'],'BLOCKED_ORIGINAL_CHECKPOINT_CHANGED');checkpoint_hashes[row['scorer_id']]={'original_checkpoint_sha256':h,'bundle_state_sha256':row['bundle_state_sha256'],'tensor_content_sha256':row['tensor_content_sha256'],'checkpoint_content_sha256':row['content_sha256']}
    verify_seal(HERE,'FRESH_PARTITION_SHA256SUMS')
    assert_preserved()
    validation={'status':'OFFLINE_PREPARATION_PASSED','readiness':READY,'Attempt003_executed':False,'Attempt003_authorized':False,'Kaggle_notebook_executed':False,
      'SCORER_AND_FEATURE_RULE_IDENTICAL_TO_SEALED_V3':True,'FORMAL_CALIBRATION_PARTITION_AND_DERIVED_CONFORMAL_RANK_AMENDED':True,
      'source_pool_count':500,'fresh_CAL_count':100,'fresh_EVAL_count':400,'prior_exposed_CAL_count':125,'historical_exclusions_count':2,'four_role_total':627,
      'n_cal':100,'alpha':.05,'rank_1_based':96,'scorers':15,'seed_spent':True,'one_permutation':True,'RNG_replayed_in_verification':False,
      'fresh_split_groups_passed':fresh['fresh_split_groups_passed'],'v4_total_groups_passed':counts['v4'],'v3_applicable_groups_passed':counts['v3'],'groups_failed':0,
      'all_frozen_states_strictly_loadable':True,'all_v3_fitted_parameter_hashes_identical':True,'class32_identity':'UNKNOWN','92_partial_scores_used':False,
      'original_125_CAL_retired':True,'TEST_network_requests':0,'TEST_payload_bytes_accessed':0,
      'v1_v2_v3_all_bytes_unchanged':True,'Attempt002_failure_evidence_unchanged':True,'v3_OOV_amendment_unchanged':True,'v3_review_unchanged':True,'git_index_unchanged':True,
      'local_offline_runtime':runtime,'runtime_pins_identical_to_v3':True,'runtime_limitation':'CPU synthetic verification does not certify locked Kaggle GPU numerics; exact v3 runtime_check must pass before separately authorized access.',
      'network_proof_limit':'Process-local Python audit hooks; not OS-wide packet capture or arbitrary native-library access proof.',
      'scientifically_untouched_pool_evidence_scope':'Preserved ledgers/code gates/current human attestation; prior opaque EVAL transport occurred. SCIENTIFIC_PAYLOAD_DECODE=0.',
      'unrelated_untracked_preservation':'New paths and .gitattributes only were written; all unrelated existing file sizes/mtimes and Git index preserved, no prohibited payload opened for preservation hashing.',
      'ATTEMPT_NUMBER':'003','EXECUTION_PROTOCOL_VERSION':'v4','OUTPUT_GENERATION':'attempt003_v2','commit':False,'push':False,'git_add_executed':False}
    write(V4/'pre_access_validation.json',validation)
    readme='''# COGNIX CARLA fresh-split final conformal calibration — protocol v4

READY_FOR_SEPARATELY_AUTHORIZED_FRESH_SPLIT_FINAL_CONFORMAL_CAL_ATTEMPT_003

Offline preparation only. Readiness does not authorize TEST access. Attempt003 has not executed. ATTEMPT_NUMBER=003, EXECUTION_PROTOCOL_VERSION=v4, OUTPUT_GENERATION=attempt003_v2. No Attempt004 exists here.

SCORER_AND_FEATURE_RULE_IDENTICAL_TO_SEALED_V3; FORMAL_CALIBRATION_PARTITION_AND_DERIVED_CONFORMAL_RANK_AMENDED.

V3 is the sealed, unexecuted precursor fixing the scorer and segmentation OOV rule before the fresh partition. It is superseded only as the execution/calibration protocol. Its historical OOV amendment record is preserved byte-for-byte for scorer provenance; that record's old125-recompute proposal is superseded by fresh_split_amendment_record.json. V1/v2/v3 and both failed-attempt records remain unchanged.

Original CAL125 were exposed during failed attempts and a post-access OOV amendment followed. None of the92 partial score values was used or reused. The original125 are permanently PRIOR_EXPOSED_CAL, opaque and retired from formal calibration/evaluation/tuning/model comparison. Recomputing them does not restore freshness. Class32 semantics UNKNOWN.

The original500-EVAL pool has SCIENTIFIC_PAYLOAD_DECODE=0 according to preserved ledgers/code gates and current human attestation. Opaque EVAL bytes previously transited/decompressed; this is not a no-byte-transport claim or a global access proof. A preregistration bound the frozen v3 scorer before exactly one secrets.randbits(128) seed and one PCG64 permutation of opaque sorted scenario IDs. First100 -> fresh CAL; remaining400 -> final EVAL. No balancing, stratification, semantic ID parsing, seed search/redraw, composition inspection or outcomes. Seed permanently spent. Verification checks the stored bijection without RNG replay.

Four immutable stream roles total627: FRESH_CAL100; FINAL_EVAL_OPAQUE_DISCARD400; PRIOR_EXPOSED_CAL_OPAQUE_DISCARD125; HISTORICAL_EXCLUDED_OPAQUE_DISCARD2. Only fresh CAL enters Image/Feather/labels/features/UQ/models. Both scanner and processor gate it. Prior CAL and exclusions are separate provenance classes. Announced body sizes and completed opaque discard chunks have separate counters.

Frozen segmentation: uint8 2-D/3-D; channel0 first; IDs0..28 unchanged; every ID29..255 -> existing Other22; unchanged29-bin extractor. Camera/IMU, all15 states/checkpoints, all fitted parameters, graph/UQ/normality/labels are identical to v3. Runtime pins/settings are identical: Python3.12.13, NumPy2.0.2, Pillow12.3.0, PyArrow25.0.1, pandas2.2.3, PyTorch2.10.0+cu128, CUDA12.8, cuDNN91002, exactly one Tesla T4. Local CPU tests do not certify GPU numerics.

Score=1-P_t(Y_t), scenario max over synchronized t>=1; alpha=.05; augmented100 scenario scores plus infinity; k=96; inclusive <=; one independent pooled threshold per fixed scorer. Full archive size/hash and exact completion of all100 fresh CAL are required before thresholds. No old scores/thresholds, model selection, subgroup tuning, EVAL metrics or calibration method ranking.

Validity is conditional on scorer freeze, scientifically undecoded original500-EVAL pool, one uniform scenario assignment and no outcome adaptations. It is finite-catalogue marginal over assignment and a uniformly chosen held-out scenario, not conditional on a realized seed/split/calibration sample. No subgroup/town/type, future-distribution/iid-whole-dataset, timestep independence, joint15-scorer or all400-EVAL simultaneous coverage claim. The specified seeded PCG64 implements the intended uniform-assignment premise; no independent randomness theorem for the PRNG is asserted.

Verify the detached ZIP hash before extraction into /kaggle/working/cognix_cal_bundle_v4. Notebook defaults False, runs preflight separately and never installs/downloads automatically. Offline preflight:

    python -B offline_checks.py runner.py --preflight

Only later explicit human authorization may enable --execute-authorized-final-cal. Output is fixed to /kaggle/working/cognix_final_conformal_cal_attempt003_v2 and existing output refuses. Marker before network; exactly one full sequential GET, Accept-Encoding identity, no redirects/Range/retry/resume, no raw archive retained. Expected compressed size91538225599 and full SHA267e48f2249deb0269ad950aa81bca57dc02e3bbdf2d73acc172af267b18254a. Failures emit sealed access/failure/result evidence and forbid partial thresholds.

FRESH_PARTITION_SHA256SUMS binds the irrevocable partition and preregistration. BUNDLE_SHA256SUMS covers all bundle files except itself, detached seal, ZIP and detached ZIP hash. ZIP includes both bundle seals and excludes itself; no circular dependency. Historical evidence scripts are not verification targets. Offline guards are process-level only; no OS-wide capture is claimed. Review and exact human staging commands are in the unsealed reports/carla_final_conformal_cal_v4_review.json. Nothing was staged/committed/pushed.
'''
    text(V4/'README.md',readme)
    write(HERE/'offline_verification_summary.json',{'v3_groups_passed':counts['v3'],'v4_groups_passed':counts['v4'],'fresh_four_role_groups_passed':fresh['fresh_split_groups_passed'],'groups_failed':0,
      'seed_reuse_refusal_checks':read(HERE/'seed_reuse_refusal_checks.json'),'scorer_checkpoint_hashes':checkpoint_hashes,'fitted_parameter_file_sha256':sha(V4/'upstream/fitted_oneclass_parameters.npz'),
      'fitted_parameter_array_sha256':fresh['fitted_parameter_array_sha256'],'v3_OOV_function_source_sha256':same['v3_OOV_function_source_sha256'],'transport_reader_caps_identical':True,'scorer_runtime_functions_identical':True})
    # Copy the validity report as a bound scientific narrative, without changing partition seal scope.
    shutil.copyfile(HERE/'report.md',V4/'fresh_split_validity_record.md')
    seal(HERE,'AMENDMENT_SHA256SUMS')
    write(V4/'fresh_amendment_binding.json',{'HEAD':HEAD,'fresh_amendment_manifest_sha256':sha(HERE/'AMENDMENT_SHA256SUMS'),'fresh_partition_manifest_sha256':sha(HERE/'FRESH_PARTITION_SHA256SUMS'),
      'v3_manifest_sha256':record['v3_manifest_sha256'],'v3_ZIP_sha256':record['v3_ZIP_sha256'],'v3_amendment_manifest_sha256':record['v3_amendment_manifest_sha256']})
    seal(V4,'BUNDLE_SHA256SUMS')
    transcripts['v4/final_integrity']=offline(V4,'verify_bundle.py')
    transcripts['v4/final_preflight']=offline(V4,'runner.py','--preflight')
    with zipfile.ZipFile(V4/ZIP,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(V4.rglob('*')):
            if not p.is_file() or p.name in {ZIP,ZIP+'.sha256'}:continue
            info=zipfile.ZipInfo(p.relative_to(V4).as_posix(),date_time=(2026,10,4,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16;info.create_system=3
            z.writestr(info,p.read_bytes(),compresslevel=9)
    text(V4/(ZIP+'.sha256'),sha(V4/ZIP)+'  '+ZIP+'\n')
    with zipfile.ZipFile(V4/ZIP) as z:
        require(z.testzip() is None,'BLOCKED_ZIP_CRC')
        expected={p.relative_to(V4).as_posix() for p in V4.rglob('*') if p.is_file() and p.name not in {ZIP,ZIP+'.sha256'}}
        require(len(z.namelist())==len(expected) and set(z.namelist())==expected,'BLOCKED_ZIP_SCOPE')
        for rel in z.namelist():require(hashlib.sha256(z.read(rel)).hexdigest()==sha(V4/rel),'BLOCKED_ZIP_EXTRACTED_BYTE_MISMATCH')
    assert_preserved()
    attrs=git('check-attr','text','--','reports/carla_final_conformal_cal_fresh_split_amendment_v1/fresh_partition_seed.json','reports/carla_final_conformal_cal_bundle_v4/runner.py')
    require(all(line.endswith(': text: unset') for line in attrs.splitlines()),'BLOCKED_SEALED_ATTRIBUTES')
    oldattrs=(HERE/'gitattributes_before.bin').read_bytes();require((ROOT/'.gitattributes').read_bytes().startswith(oldattrs),'BLOCKED_OLD_ATTRIBUTE_BYTES_CHANGED')
    review_path=ROOT/'reports/carla_final_conformal_cal_v4_review.json';write(review_path,{})
    review_commands=['git rev-parse HEAD','git diff -- .gitattributes',
      "& '"+sys.executable+"' -B reports/carla_final_conformal_cal_bundle_v4/offline_checks.py reports/carla_final_conformal_cal_bundle_v4/verify_bundle.py",
      "& '"+sys.executable+"' -B reports/carla_final_conformal_cal_bundle_v4/offline_checks.py reports/carla_final_conformal_cal_bundle_v4/runner.py --preflight",
      "Get-Content 'reports/carla_final_conformal_cal_fresh_split_amendment_v1/fresh_partition_execution_receipt.json'",
      "Get-Content 'reports/carla_final_conformal_cal_fresh_split_amendment_v1/FRESH_PARTITION_SHA256SUMS'",
      "Get-Content 'reports/carla_final_conformal_cal_fresh_split_amendment_v1/report.md'",
      "Get-Content 'reports/carla_final_conformal_cal_v4_review.json'",
      "Get-FileHash -Algorithm SHA256 'reports/carla_final_conformal_cal_fresh_split_amendment_v1/fresh_final_conformal_cal.txt', 'reports/carla_final_conformal_cal_fresh_split_amendment_v1/fresh_final_evaluation.txt', 'reports/carla_final_conformal_cal_fresh_split_amendment_v1/fresh_partition_seed.json'",
      "Get-FileHash -Algorithm SHA256 'reports/carla_final_conformal_cal_bundle_v4/cognix_final_conformal_cal_kaggle_v4.zip'",
      'git check-attr text -- reports/carla_final_conformal_cal_fresh_split_amendment_v1/fresh_partition_seed.json reports/carla_final_conformal_cal_bundle_v4/runner.py','git status --short']
    stages=['git add -- .gitattributes','git add -- reports/carla_final_conformal_cal_fresh_split_amendment_v1','git add -- reports/carla_final_conformal_cal_bundle_v4','git add -- reports/carla_final_conformal_cal_v4_review.json']
    paths=[HERE.relative_to(ROOT).as_posix(),V4.relative_to(ROOT).as_posix(),review_path.relative_to(ROOT).as_posix()]
    review={**validation,'HEAD':HEAD,'HEAD_subject':git('log','-1','--format=%s'),
      'v3_manifest_sha256':record['v3_manifest_sha256'],'v3_ZIP_sha256':record['v3_ZIP_sha256'],'v3_amendment_manifest_sha256':record['v3_amendment_manifest_sha256'],
      'original_CAL_membership_sha256':binding['prior_exposed_CAL_sha256'],'original_EVAL_source_pool_sha256':binding['source_pool_sha256'],
      'fresh_source_pool_sha256':binding['source_pool_sha256'],'fresh_sorted_source_sha256':binding['sorted_source_sha256'],'one_time_seed':seed['seed'],
      'seed_receipt_path':'reports/carla_final_conformal_cal_fresh_split_amendment_v1/fresh_partition_seed.json','seed_receipt_sha256':binding['seed_receipt_sha256'],'seed_receipt':seed,
      'fresh_CAL_membership_sha256':binding['CAL_membership_sha256'],'fresh_EVAL_membership_sha256':binding['EVAL_membership_sha256'],'fresh_partition_aggregate_seal_sha256':binding['fresh_partition_seal_sha256'],
      'historical_exclusions_sha256':binding['historical_exclusions_sha256'],'four_role_scanner_counts':read(V4/'protected_eval_policy.json')['roles'],
      'partition_validation':read(HERE/'fresh_partition_validation.json'),'partition_execution_receipt':read(HERE/'fresh_partition_execution_receipt.json'),
      'OOV_rule_binding':same,'class32_identity':'UNKNOWN','partial92_scores_used_for_protocol_or_partition_selection':False,'original125_CAL_retired':True,
      'scorer_checkpoint_hashes':checkpoint_hashes,'fitted_parameter_file_sha256':sha(V4/'upstream/fitted_oneclass_parameters.npz'),'fitted_parameter_array_sha256':fresh['fitted_parameter_array_sha256'],
      'regression_group_counts':counts,'all_failures':[],'expected_seed_refusals':read(HERE/'seed_reuse_refusal_checks.json'),'offline_test_transcripts':transcripts,
      'fresh_pool_scientific_untouched_audit':read(HERE/'fresh_pool_scientific_untouched_audit.json'),
      'v4_manifest_sha256':sha(V4/'BUNDLE_SHA256SUMS'),'v4_ZIP_sha256':sha(V4/ZIP),'v4_ZIP_bytes':(V4/ZIP).stat().st_size,'fresh_amendment_manifest_sha256':sha(HERE/'AMENDMENT_SHA256SUMS'),
      'ZIP_extraction_scope_and_every_byte_verified':True,'tracked_files_modified':['.gitattributes'],'created_paths':paths,
      'created_files':[p.relative_to(ROOT).as_posix() for d in (HERE,V4) for p in sorted(d.rglob('*')) if p.is_file()]+[review_path.relative_to(ROOT).as_posix()],
      'git_status':git('status','--porcelain=v1').splitlines(),'sealed_Git_attributes':attrs.splitlines(),'preserved_unrelated_untracked_files':len(baseline['unrelated_untracked_metadata']),
      'human_review_commands':review_commands,'exact_path_staging_commands_NOT_EXECUTED':stages,'NO_COMMIT':True,'NO_PUSH':True,'NO_GIT_ADD_EXECUTED':True}
    write(review_path,review)
    print(json.dumps({k:review[k] for k in ['HEAD','readiness','created_paths','tracked_files_modified','one_time_seed','seed_spent','fresh_CAL_membership_sha256','fresh_EVAL_membership_sha256','fresh_partition_aggregate_seal_sha256','four_role_scanner_counts','n_cal','alpha','rank_1_based','regression_group_counts','v4_manifest_sha256','v4_ZIP_sha256','TEST_network_requests','TEST_payload_bytes_accessed','Attempt003_executed','human_review_commands','exact_path_staging_commands_NOT_EXECUTED','NO_COMMIT','NO_PUSH','NO_GIT_ADD_EXECUTED']},sort_keys=True,indent=2))

if __name__=='__main__':
    require(sys.argv[1:] in (['--prepare'],['--seal']),'Explicit offline build mode required')
    if sys.argv[1:]==['--prepare']:prepare()
    else:finalize()
