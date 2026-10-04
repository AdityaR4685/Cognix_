"""Finalize offline regressions, immutable-byte seals, ZIP roundtrip and review.

Runs only local verification subprocesses. Never calls a scientific runner.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

sys.dont_write_bytecode=True
PREP=Path(__file__).resolve().parent
ROOT=PREP.parents[1]
BUNDLE=ROOT/'reports/carla_final_evaluation_bundle_v1'
COMPLETE=ROOT/'reports/carla_final_conformal_cal_attempt003_complete_v1'
REVIEW=ROOT/'reports/carla_final_evaluation_v1_review.json'
ZIP='cognix_final_evaluation_kaggle_v1.zip'
HEAD='cb50f935bc940c33722ffbf3956f232bc28a228e'
READY='READY_FOR_SEPARATELY_AUTHORIZED_FINAL_EVALUATION_ATTEMPT_001'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def save(p,obj):p.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT).decode()
def seal(directory,name,excluded=()):
    paths=sorted(p for p in directory.rglob('*') if p.is_file() and p.relative_to(directory).as_posix() not in set(excluded)|{name,name+'.sha256'})
    (directory/name).write_text(''.join(sha(p)+'  '+p.relative_to(directory).as_posix()+'\n' for p in paths),encoding='utf-8',newline='\n')
    (directory/(name+'.sha256')).write_text(sha(directory/name)+'  '+name+'\n',encoding='utf-8',newline='\n')
    return {p.relative_to(directory).as_posix():sha(p) for p in paths}

def check_old_seals():
    directories={f'carla_final_conformal_cal_bundle_v{i}':'BUNDLE_SHA256SUMS' for i in range(1,5)}
    directories.update({'carla_final_conformal_cal_fresh_split_amendment_v1':'AMENDMENT_SHA256SUMS',
      'carla_final_evaluation_preregistration_v1':'SHA256SUMS',
      'carla_final_conformal_cal_attempt001_failure_v1':'FAILURE_SHA256SUMS',
      'carla_final_conformal_cal_attempt001_traceback_addendum_v1':'ADDENDUM_SHA256SUMS',
      'carla_final_conformal_cal_attempt002_failure_v1':'FAILURE_SHA256SUMS',
      'carla_final_conformal_cal_attempt002_oov_amendment_v1':'AMENDMENT_SHA256SUMS',
      'carla_final_partition_v1':'SHA256SUMS'})
    # Fresh partition seal is stored in this amendment directory at the top level.
    results={}
    for name,manifest in directories.items():
        d=ROOT/'reports'/name
        if not (d/manifest).exists():
            if name=='carla_final_conformal_cal_fresh_split_amendment_v1':manifest='AMENDMENT_SHA256SUMS'
        p=d/manifest
        if not p.exists():raise RuntimeError('Missing preexisting seal: '+str(p))
        assert (d/(manifest+'.sha256')).read_text().split()==[sha(p),manifest]
        scope={}
        for line in p.read_text().splitlines():
            h,rel=line.split('  ',1);target=(d/rel).resolve()
            assert target.is_relative_to(d.resolve()) and rel not in scope and sha(target)==h,(name,rel)
            scope[rel]=h
        results[name]={'manifest':manifest,'manifest_sha256':sha(p),'files_verified':len(scope),'unchanged':True}
    return results

def checked_command(command):
    done=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,check=False)
    if done.returncode!=0:raise RuntimeError('Offline regression failure:\n'+done.stdout+'\n'+done.stderr)
    objects=[json.loads(line) for line in done.stdout.splitlines() if line.startswith('{')]
    return objects,{'command':command,'exit_code':done.returncode,'stdout':done.stdout,'stderr':done.stderr}

def zip_roundtrip():
    source={p.relative_to(BUNDLE).as_posix():sha(p) for p in BUNDLE.rglob('*') if p.is_file() and p.name not in {ZIP,ZIP+'.sha256'}}
    archive=BUNDLE/ZIP
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for rel in sorted(source):z.write(BUNDLE/rel,'cognix_final_evaluation_bundle_v1/'+rel)
    (BUNDLE/(ZIP+'.sha256')).write_text(sha(archive)+'  '+ZIP+'\n',encoding='utf-8',newline='\n')
    with tempfile.TemporaryDirectory(prefix='cognix_final_eval_zip_verify_') as tmp,zipfile.ZipFile(archive) as z:
        destination=Path(tmp).resolve();members=z.infolist()
        expected={'cognix_final_evaluation_bundle_v1/'+rel for rel in source}
        assert {m.filename for m in members}==expected and len(members)==len(expected)
        for m in members:
            path=(destination/m.filename).resolve()
            assert path.is_relative_to(destination) and not m.is_dir() and not m.filename.startswith('/') and '\\' not in m.filename and not (m.external_attr>>16)&0o170000==0o120000
            path.parent.mkdir(parents=True,exist_ok=True)
            with z.open(m) as src,path.open('xb') as out:
                import shutil
                shutil.copyfileobj(src,out)
        extracted=destination/'cognix_final_evaluation_bundle_v1'
        actual={p.relative_to(extracted).as_posix():sha(p) for p in extracted.rglob('*') if p.is_file()}
        assert actual==source,'ZIP extraction bytes/scope differ'
    return {'status':'PASSED','zip_sha256':sha(archive),'zip_bytes':archive.stat().st_size,'files_extracted_and_verified':len(source),
      'all_extracted_bytes_match_source':True,'no_extra_or_unsealed_files':True,'no_circular_hash_dependency':True,
      'sealed_bundle_manifest_sha256':sha(BUNDLE/'BUNDLE_SHA256SUMS'),'zip_scope':sorted(source)}

def main():
    assert git('rev-parse','HEAD').strip()==HEAD,'BLOCKED_HEAD_MISMATCH'
    assert not (BUNDLE/'BUNDLE_SHA256SUMS').exists(),'Sealed bundle: refuse regeneration'
    prior=check_old_seals()
    spec=importlib.util.spec_from_file_location('ingestion',COMPLETE/'ingest_attempt003.py');ingest=importlib.util.module_from_spec(spec);spec.loader.exec_module(ingest)
    ingest.seal_check(COMPLETE,'ATTEMPT003_COMPLETE_SHA256SUMS',exact=True)
    validation=ingest.validate(COMPLETE/'evidence/cognix_final_conformal_cal_attempt003_v2')
    baseline=read(COMPLETE/'preparation_initial_state.json')
    assert baseline['tracked_diff']==''
    preserved={}
    for rel,expected in baseline['existing_untracked_stat'].items():
        p=ROOT/rel;st=p.stat();actual={'size':st.st_size,'mtime_ns':st.st_mtime_ns}
        assert actual==expected,'Unrelated untracked metadata changed: '+rel
        preserved[rel]=actual
    assert git('diff','--cached','--name-only').strip()==''
    assert git('diff','--name-only').splitlines()==['.gitattributes']
    python=sys.executable
    command=[python,'-B',str(BUNDLE/'offline_checks.py'),str(BUNDLE/'offline_regression.py')]
    objects,transcript=checked_command(command);regression=objects[0]
    assert regression['groups_failed']==0 and regression['groups_passed']>=44 and objects[-1]['offline_network_guard']=='PASSED'
    save(BUNDLE/'offline_regression_results.json',regression)
    save(PREP/'offline_test_transcripts.json',{'regressions':transcript})
    save(BUNDLE/'preparation_receipt.json',{'status':'OFFLINE_PREPARATION_COMPLETE','HEAD':HEAD,'client_date':'2026-10-04','client_timezone':'Asia/Calcutta',
      'Attempt003_evidence_verified_and_sealed':True,'n_cal':100,'alpha':.05,'k':96,'threshold_count':15,
      'final_EVAL':400,'protected_fresh_CAL':100,'prior_exposed_CAL':125,'historical_excluded':2,'fixed_total':627,
      'frozen_final_spec_verified':True,'scientific_amendment_required':False,'regression_groups_passed':regression['groups_passed'],
      'regression_groups_failed':0,'runtime_actual_execution_performed':False,'runtime_future_preflight_required':True,
      'TEST_requests':0,'TEST_payload_bytes':0,'final_evaluation_executed':False,'git_add_executed':False,'commit_executed':False,'push_executed':False})
    scope=seal(BUNDLE,'BUNDLE_SHA256SUMS',{ZIP,ZIP+'.sha256'})
    objects,preflight_transcript=checked_command([python,'-B',str(BUNDLE/'offline_checks.py'),str(BUNDLE/'verify_bundle.py')])
    assert objects[0]['status']=='PASSED'
    save(PREP/'offline_test_transcripts.json',{'regressions':transcript,'sealed_bundle_verification':preflight_transcript})
    roundtrip=zip_roundtrip();save(PREP/'zip_roundtrip_verification.json',roundtrip)
    # Rehash prior protocol/evidence seals after all preparation writes.
    assert check_old_seals()==prior
    ingest.seal_check(COMPLETE,'ATTEMPT003_COMPLETE_SHA256SUMS',exact=True)
    assert sha(ingest.ARCHIVE)==ingest.ARCHIVE_SHA
    save(PREP/'preservation_verification.json',{'status':'PASSED','HEAD_unchanged':HEAD,'prior_sealed_artifacts':prior,
      'existing_untracked_files_preserved_count':len(preserved),'verification_scope':'Existing untracked path/size/mtime metadata preserved; none of their content was opened, mutated or removed. Prior scientific/protocol seals rehashed.',
      'tracked_modifications':['.gitattributes'],'staged_changes':[],'git_add_executed':False,'commit_executed':False,'push_executed':False})
    seal(PREP,'PREPARATION_SHA256SUMS')
    b=read(BUNDLE/'protocol_bindings.json');pb=read(BUNDLE/'partition_binding.json');registry=read(BUNDLE/'model_registry.json')
    review_commands=["git rev-parse HEAD","git log -1 --format=%s","git diff -- .gitattributes","git diff --cached --name-only",
      "& '"+python+"' -B '"+str(BUNDLE/'offline_checks.py')+"' '"+str(BUNDLE/'verify_bundle.py')+"'",
      "& '"+python+"' -B '"+str(BUNDLE/'offline_checks.py')+"' '"+str(BUNDLE/'offline_regression.py')+"'",
      "Get-FileHash -Algorithm SHA256 -LiteralPath '"+str(BUNDLE/ZIP)+"'","git status --short"]
    staging=["git add -- .gitattributes reports/carla_final_conformal_cal_attempt003_complete_v1/ reports/carla_final_evaluation_bundle_v1/ reports/carla_final_evaluation_preparation_v1/ reports/carla_final_evaluation_v1_review.json reports/carla_final_evaluation_v1_review.json.sha256"]
    status=git('status','--porcelain=v1','-uall','-z').rstrip('\0').split('\0')
    created=['reports/carla_final_conformal_cal_attempt003_complete_v1/','reports/carla_final_evaluation_bundle_v1/',
      'reports/carla_final_evaluation_preparation_v1/','reports/carla_final_evaluation_v1_review.json','reports/carla_final_evaluation_v1_review.json.sha256']
    # Include the two review artifacts about to be written, so the saved status
    # reflects the complete deliverable set without a self-referential digest.
    status=sorted(set(status)|{'?? reports/carla_final_evaluation_v1_review.json','?? reports/carla_final_evaluation_v1_review.json.sha256'})
    review={'readiness':READY,'HEAD':HEAD,'HEAD_subject':git('log','-1','--format=%s').strip(),
      'Attempt003_evidence_archive_SHA256':ingest.ARCHIVE_SHA,'Attempt003_result_manifest_status':'COMPLETE_CAL_ONLY',
      'Attempt003_full_archive_size_hash_receipt':validation['full_archive_receipt'],'Attempt003_access_ledger_validation':validation,
      'Attempt003_completion_seal_SHA256':sha(COMPLETE/'ATTEMPT003_COMPLETE_SHA256SUMS'),
      'Attempt003_threshold_artifact_SHA256':b['threshold_sha256'],'threshold_receipt_SHA256':b['threshold_receipts_sha256'],
      'threshold_count':15,'n_cal':100,'alpha':.05,'k':96,'inclusive_comparison':'<=','calibration_rerun_allowed':False,
      'final_EVAL':{'count':400,'sha256':pb['EVAL_membership_sha256']},'fresh_CAL_protected':{'count':100,'sha256':pb['CAL_membership_sha256']},
      'prior_exposed_CAL':{'count':125,'sha256':pb['prior_exposed_CAL_sha256']},'historical_exclusions':{'count':2,'sha256':pb['historical_exclusions_sha256']},
      'scorer_checkpoint_bindings':registry,'fitted_parameter_hashes':{k:v for k,v in b['identical_v4_files'].items() if k.startswith('upstream/')},
      'OOV_binding':{'source_function_sha256':b['source_function_sha256']['map_segmentation_oov_to_frozen_vocabulary'],
        'policy_artifact_sha256':sha(BUNDLE/'segmentation_oov_amendment.json'),'policy':'uint8 2D/3D; channel0 first; IDs0..28 unchanged; all IDs>28 to Other22; frozen29D features'},
      'final_metrics_spec':b['frozen_spec_bindings']['final_metrics_spec.json'],'seed_aggregation_spec':b['frozen_spec_bindings']['seed_aggregation_spec.json'],
      'decision_semantics':b['frozen_spec_bindings']['decision_semantics.json'],'hard_threshold_binding':b['frozen_spec_bindings']['frozen_method_bindings.json'],
      'frozen_spec_manifest_sha256':b['frozen_spec_manifest_sha256'],'frozen_spec_git_checkout_note':'Preexisting local sealed CRLF bytes preserved and copied exactly. Git blobs differ only by existing checkout line-ending normalization; no prior bytes changed.',
      'classification_metric_names':list(read(BUNDLE/'final_metrics_spec.json')['binary_metrics']),
      'conformal_metric_names':list(read(BUNDLE/'final_metrics_spec.json')['conformal_metrics']),
      'runtime_pins':{**read(BUNDLE/'runtime_lock.json'),'pandas':'2.2.3'},'runtime_check_source_sha256':b['source_function_sha256']['runtime_check'],
      'runtime_validation_limit':regression['runtime_validation_scope'],'model_validation_limit':regression['checkpoint_validation_scope'],
      'regression_groups_passed':regression['groups_passed'],'regression_groups_failed':0,'regression_groups':regression['groups'],
      'v4_amendment_effect_audit_path':'reports/carla_final_evaluation_bundle_v1/v4_amendment_effect_audit.json',
      'bundle_manifest_SHA256':sha(BUNDLE/'BUNDLE_SHA256SUMS'),'bundle_sealed_files':len(scope),'ZIP_SHA256':roundtrip['zip_sha256'],
      'ZIP_path':'reports/carla_final_evaluation_bundle_v1/'+ZIP,'zip_roundtrip_verification':roundtrip,
      'preparation_manifest_SHA256':sha(PREP/'PREPARATION_SHA256SUMS'),'preparation_TEST_requests':0,'preparation_TEST_payload_bytes':0,
      'final_evaluation_executed':False,'separate_authorization_required':True,'FINAL_EVALUATION_ATTEMPT':'001','CALIBRATION_SOURCE':'successful Attempt003 v4',
      'created_paths':created,'tracked_modifications':['.gitattributes'],'git_status':status,'staged_changes':[],
      'exact_human_review_commands':review_commands,'exact_path_staging_commands_NOT_EXECUTED':staging,
      'NO_COMMIT':True,'NO_PUSH':True,'NO_GIT_ADD_EXECUTED':True,
      'negative_development_context':'EpistemicGAT was not consistently superior during development.',
      'claims_limit':'Execution receipts and byte bindings, not OS-wide access proof; no every-scenario joint95%, joint15-scorer, subgroup-conditional, arbitrary-future, or physical-safety coverage claim.',
      'preserved_prior_seals':prior,'existing_untracked_preserved_count':len(preserved)}
    save(REVIEW,review)
    REVIEW.with_name(REVIEW.name+'.sha256').write_text(sha(REVIEW)+'  '+REVIEW.name+'\n',encoding='utf-8',newline='\n')
    assert git('rev-parse','HEAD').strip()==HEAD and git('diff','--cached','--name-only').strip()==''
    assert sorted(git('status','--porcelain=v1','-uall','-z').rstrip('\0').split('\0'))==status
    print(json.dumps({'readiness':READY,'regression_groups_passed':regression['groups_passed'],'bundle_manifest_sha256':review['bundle_manifest_SHA256'],
      'zip_sha256':review['ZIP_SHA256'],'zip_roundtrip_files':roundtrip['files_extracted_and_verified'],'review':str(REVIEW),
      'TEST_requests':0,'TEST_payload_bytes':0,'final_evaluation_executed':False,'staged_changes':0}))

if __name__=='__main__':main()
