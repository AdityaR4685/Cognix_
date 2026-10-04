"""Offline, fail-closed ingestion of the sole authorized external evidence archive."""
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V4 = ROOT / 'reports/carla_final_conformal_cal_bundle_v4'
SPEC = ROOT / 'reports/carla_final_evaluation_preregistration_v1'
ARCHIVE = Path(r'C:\Users\Aditya\Downloads\cognix_final_conformal_cal_attempt003_v2_evidence.tar.gz')
ARCHIVE_SHA = 'cf9b956ecd37ccf1762f0ef0a588cda15665ddc377d97e2adaca7f850ecb3cd9'
HEAD = 'cb50f935bc940c33722ffbf3956f232bc28a228e'
EVIDENCE_ROOT = 'cognix_final_conformal_cal_attempt003_v2'
OFFICIAL_SHA = '267e48f2249deb0269ad950aa81bca57dc02e3bbdf2d73acc172af267b18254a'
REQUIRED = {'attempt_started.json', 'archive_verification_receipt.json', 'cal_scenario_processing_ledger.jsonl',
 'conformal_thresholds.json', 'threshold_calculation_receipts.json', 'per_scorer_scenario_scores.json',
 'cal_only_descriptive_diagnostics.json', 'zero_eval_decode_receipt.json', 'failure_exception_audit.json',
 'result_manifest.json', 'access_ledger.json', 'RESULT_SHA256SUMS', 'RESULT_SHA256SUMS.sha256'}

def require(ok, message):
    if not ok:
        raise RuntimeError('BLOCKED_ATTEMPT003_EVIDENCE_VERIFICATION_FAILED: ' + message)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))

def save(path, obj):
    path.write_text(json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')

def seal_check(directory, name, expected=None, exact=False, excluded=()):
    digest = sha(directory / name)
    require(expected is None or digest == expected, 'manifest fixed binding: ' + str(directory))
    require((directory / (name + '.sha256')).read_text().split() == [digest, name], 'detached manifest seal')
    scope = {}
    for line in (directory / name).read_text().splitlines():
        h, rel = line.split('  ', 1)
        p = (directory / rel).resolve()
        require(p.is_relative_to(directory.resolve()) and rel not in scope and p.is_file() and sha(p) == h, 'sealed byte mismatch: ' + rel)
        scope[rel] = h
    if exact:
        actual = {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}
        require(actual - {name, name + '.sha256'} - set(excluded) == set(scope), 'exact sealed scope')
    return scope

def validate(d):
    scope = seal_check(d, 'RESULT_SHA256SUMS', exact=True)
    require(set(scope) == REQUIRED - {'RESULT_SHA256SUMS', 'RESULT_SHA256SUMS.sha256'}, 'result scope')
    m, fail, ledger, receipt = [read(d / n) for n in ['result_manifest.json','failure_exception_audit.json','access_ledger.json','archive_verification_receipt.json']]
    require(m == {'status':'COMPLETE_CAL_ONLY','scorers':15,'thresholds':15,'cal_scenarios':100,'eval_metrics_computed':False,'method_ranking':False}, 'manifest completion')
    require(fail == {'status':'COMPLETE','failures':[]}, 'failure audit')
    fields = {'automatic_retry':False,'cal_only_science':True,'fresh_cal_only_science':True,'cal_scenarios_completed':100,
      'compressed_bytes_received':91538225599,'compressed_sha256_partial_or_complete':OFFICIAL_SHA,'http_requests':1,
      'range_requests':0,'raw_archive_retained':False,'scenario_roots_seen':627,'eval_scenarios_decoded':0,
      'prior_exposed_cal_scenarios_decoded':0,'historical_excluded_scenarios_decoded':0,'transport_is_not_scientific_decoding':True}
    require(all(ledger.get(k) == v for k,v in fields.items()), 'access-ledger fixed fields')
    roles = {'FRESH_CAL':100,'FINAL_EVAL_OPAQUE_DISCARD':400,'PRIOR_EXPOSED_CAL_OPAQUE_DISCARD':125,'HISTORICAL_EXCLUDED_OPAQUE_DISCARD':2}
    require(ledger['scenario_roots_seen_by_role'] == roles, 'role root counts')
    accounted = {'FINAL_EVAL_OPAQUE_DISCARD':62847018051,'FRESH_CAL':16288609930,'HISTORICAL_EXCLUDED_OPAQUE_DISCARD':240747299,'PRIOR_EXPOSED_CAL_OPAQUE_DISCARD':19633431385}
    require(ledger['body_bytes_accounted_by_role'] == accounted, 'accounted role bytes')
    require(ledger['opaque_body_bytes_actually_discarded_by_role'] == {**accounted,'FRESH_CAL':65679379}, 'discarded role bytes')
    require(receipt['compressed_bytes'] == 91538225599 and receipt['sha256'] == OFFICIAL_SHA and receipt['gzip_crc_validated'] is True and receipt['tar_complete'] is True, 'full archive receipt')
    started = read(d / 'attempt_started.json')
    require(started['single_attempt'] is True and started['automatic_retry'] is False and started['manifest_sha256'] == sha(V4/'BUNDLE_SHA256SUMS'), 'v4 execution binding')
    lock = read(V4/'runtime_lock.json')
    require(all(started['runtime'][k] == lock[k] for k in started['runtime']), 'runtime receipt')
    cal = set((V4/'final_conformal_cal.txt').read_text().splitlines())
    ev = set((V4/'final_evaluation.txt').read_text().splitlines())
    rows = [json.loads(line) for line in (d/'cal_scenario_processing_ledger.jsonl').read_text().splitlines()]
    require(len(rows) == 100 and {r['scenario_id'] for r in rows} == cal and not ({r['scenario_id'] for r in rows} & ev), 'exact unique fresh CAL completion')
    scorers = {r['scorer_id'] for r in read(V4/'model_registry.json')['scorers']}
    require(all(r['partition']=='FRESH_FINAL_CONFORMAL_CAL' and r['tick_0_excluded'] is True and type(r['eligible_ticks']) is int and r['eligible_ticks']>0 and set(r['scenario_scores'])==scorers for r in rows), 'CAL ledger structure')
    require(sum(r['eligible_ticks'] for r in rows)==29900, 'eligible tick total')
    thresholds, receipts = read(d/'conformal_thresholds.json'), read(d/'threshold_calculation_receipts.json')
    scores = read(d/'per_scorer_scenario_scores.json')
    diagnostics = read(d/'cal_only_descriptive_diagnostics.json')
    require(set(thresholds)==set(receipts)==set(scores)==set(diagnostics)==scorers and len(scorers)==15, '15 fixed scorer identities')
    for scorer in sorted(scorers):
        t, r = thresholds[scorer], receipts[scorer]
        require(set(t)=={'alpha','n_cal','rank_1_based','Q','quantile_is_infinite','augmentation','comparison'}, 'threshold schema')
        require(t['alpha']==.05 and t['n_cal']==100 and t['rank_1_based']==96 and t['comparison']=='<=' and t['augmentation']=='+infinity' and t['quantile_is_infinite'] is False, 'frozen conformal parameters')
        require(type(t['Q']) in (int,float) and math.isfinite(t['Q']) and 0<=t['Q']<=1, 'finite cutoff')
        require(all(r[k]==v for k,v in t.items()) and set(r)==set(t)|{'scenario_scores_sha256','identical_cal_membership_sha256','preregistered_score','method_selection_performed'}, 'receipt exact binding')
        require(r['identical_cal_membership_sha256']==sha(V4/'final_conformal_cal.txt') and r['preregistered_score']=='max_t(1-P_t(Y_t))' and r['method_selection_performed'] is False, 'receipt score/membership')
        s = scores[scorer]
        require(len(s)==100 and {x['scenario_id'] for x in s}==cal, 'historical CAL score membership')
        require(hashlib.sha256(json.dumps(s,sort_keys=True,separators=(',',':')).encode()).hexdigest()==r['scenario_scores_sha256'], 'historical score receipt hash')
        require(all(x['score']==next(row for row in rows if row['scenario_id']==x['scenario_id'])['scenario_scores'][scorer] for x in s), 'ledger score binding')
        require(diagnostics[scorer]['cal_scenarios']==100 and diagnostics[scorer]['cal_ticks']==29900, 'CAL-only diagnostics scope')
    zero = read(d/'zero_eval_decode_receipt.json')
    require(all(zero[k]==0 for k in ['eval_scenarios_decoded','prior_exposed_cal_scenarios_decoded','historical_excluded_scenarios_decoded']), 'protected roles undecoded')
    require(not any(word in (d/'threshold_calculation_receipts.json').read_text().lower() for word in ['attempt001','attempt002']), 'no active partial threshold source')
    return {'passed':True,'checks_failed':[],'result_files_verified':len(scope),'result_manifest':m,'full_archive_receipt':receipt,
      'access_ledger_validated':True,'protected_role_scientific_decodes':{'FINAL_EVAL':0,'PRIOR_EXPOSED_CAL':0,'HISTORICAL_EXCLUDED':0},
      'unique_completed_fresh_CAL':100,'eligible_ticks':29900,'threshold_count':15,'threshold_receipts_structurally_verified':True,
      'Q_recomputed':False,'CAL_membership_sha256':sha(V4/'final_conformal_cal.txt'),'threshold_sha256':sha(d/'conformal_thresholds.json'),
      'threshold_receipts_sha256':sha(d/'threshold_calculation_receipts.json'),'RESULT_SHA256SUMS_sha256':sha(d/'RESULT_SHA256SUMS'),
      'limitation':'Executed-code receipts and cryptographic bindings; not an OS-wide access proof.'}

def main():
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if head != HEAD: raise RuntimeError('BLOCKED_HEAD_MISMATCH')
    require(sha(ARCHIVE)==ARCHIVE_SHA, 'external archive hash')
    seal_check(V4,'BUNDLE_SHA256SUMS','7cfa0339d64e4d0350e8dd5b0940913e68d95e4d0aaffb2108593a11f87e9a39',True,
      {'cognix_final_conformal_cal_kaggle_v4.zip','cognix_final_conformal_cal_kaggle_v4.zip.sha256'})
    require(sha(V4/'cognix_final_conformal_cal_kaggle_v4.zip')=='a822ed2a173f6a1b7061ec8e604eeb724d16d33e80aea4691178f87db711afc2','v4 ZIP')
    baseline = {'HEAD':head,'tracked_diff':subprocess.check_output(['git','diff','HEAD','--binary'],cwd=ROOT).decode(),
      'git_status':subprocess.check_output(['git','status','--porcelain=v1','-uall','-z'],cwd=ROOT).decode().rstrip('\0').split('\0')}
    baseline['existing_untracked_stat'] = {}
    for line in baseline['git_status']:
        if line.startswith('?? '):
            p=ROOT/line[3:]; st=p.stat()
            baseline['existing_untracked_stat'][line[3:]]={'size':st.st_size,'mtime_ns':st.st_mtime_ns}
    save(HERE/'preparation_initial_state.json',baseline)
    destination = HERE/'evidence'/EVIDENCE_ROOT
    require(not destination.exists(), 'new extraction only')
    with tarfile.open(ARCHIVE,'r:gz') as tf:
        members=tf.getmembers(); files={}; seen=set()
        require(len(members)<=100 and sum(m.size for m in members)<=50*1024*1024,'bounded evidence')
        for m in members:
            name=m.name.rstrip('/')
            parts=PurePosixPath(name).parts
            require(name and not name.startswith('/') and '\\' not in name and parts[0]==EVIDENCE_ROOT and all(x not in ('','.','..') and ':' not in x for x in name.split('/')), 'unsafe or wrong-scope tar path')
            require(m.isdir() or m.isreg(), 'unsafe tar member/link')
            require(name.casefold() not in seen, 'duplicate tar member');seen.add(name.casefold())
            if m.isdir(): require(name==EVIDENCE_ROOT and m.size==0,'unexpected directory')
            else:
                require(len(parts)==2 and parts[1] in REQUIRED,'unexpected evidence file')
                files[parts[1]]=m
        require(set(files)==REQUIRED,'required exact archive scope')
        copy=HERE/ARCHIVE.name
        require(not copy.exists(),'archive copy already exists')
        shutil.copyfile(ARCHIVE,copy)
        require(sha(copy)==ARCHIVE_SHA,'byte-for-byte copied archive')
        destination.mkdir(parents=True,exist_ok=False)
        for name,m in files.items():
            with tf.extractfile(m) as source, (destination/name).open('xb') as out: shutil.copyfileobj(source,out)
    validation=validate(destination)
    save(HERE/'attempt003_result_validation.json',validation)
    save(HERE/'attempt003_archive_binding.json',{'external_source':str(ARCHIVE),'copied_archive':copy.name,'sha256':ARCHIVE_SHA,
      'size':copy.stat().st_size,'expected_root':EVIDENCE_ROOT,'safe_exact_scope_validated':True,'original_modified':False})
    save(HERE/'attempt003_threshold_binding.json',{'source_attempt':'successful Attempt003 v4','artifact':'evidence/'+EVIDENCE_ROOT+'/conformal_thresholds.json',
      'sha256':validation['threshold_sha256'],'receipt_sha256':validation['threshold_receipts_sha256'],'threshold_count':15,'n_cal':100,'alpha':.05,'k':96,
      'comparison':'<=','score':'max_t(1-P_t(Y_t))','permanently_frozen':True,'refit_allowed':False,'rerun_allowed':False,'partial_attempt001_002_active':False})
    save(HERE/'attempt003_completion_record.json',{'status':'SEALED_SUCCESSFUL_ATTEMPT003_CAL_ONLY','HEAD':HEAD,
      'successful_v4_execution':True,'fresh_CAL_completed':100,'eligible_ticks':29900,'thresholds':15,'no_rerun':True,
      'prior_attempts':'Attempt001/002 partial failure outputs remain historical only','local_TEST_requests':0,'local_TEST_payload_bytes':0,
      'evaluation_executed':False,'validation_sha256':sha(HERE/'attempt003_result_validation.json')})
    report='''# Attempt003 calibration completion

Attempt003 was the successful v4 calibration execution. The copied evidence archive and all 11 result files verify against their SHA256 seals. Exactly 100 unique fresh CAL scenarios completed, with 29,900 eligible ticks. The executed-code receipt verifies the official full compressed archive length 91,538,225,599 bytes and SHA256 267e48f2249deb0269ad950aa81bca57dc02e3bbdf2d73acc172af267b18254a.

The verified access ledger records one GET, zero Range requests, 627 reconciled scenario roots, and no retained raw archive. No final EVAL, prior-exposed old CAL, or historical exclusion was scientifically decoded according to the bound execution receipts. These receipts are not an OS-wide access proof.

Exactly 15 frozen scorer-specific cutoffs were produced, with n_cal=100, alpha=.05, rank 96, max_t(1-P_t(Y_t)), and inclusive <=. No evaluation metric or method ranking occurred. The cutoffs are now permanently frozen. Attempt003 must never be rerun. Partial Attempt001/002 outputs remain historical only. Numeric cutoff values are deliberately omitted from this review.

No TEST access occurred during this local evidence-ingestion milestone. No calibration or final evaluation was executed locally. The original Downloads archive was left unchanged.

Chronology: development frozen -> TEST inventory/partition -> Attempt001 failure -> v2 repair -> Attempt002 failure -> v3 OOV freeze -> fresh 100/400 split -> v4 calibration protocol -> successful Attempt003 -> 15 frozen cutoffs -> offline final-evaluation preparation -> later separately authorized one-shot evaluation of only 400 FINAL_EVAL scenarios.
'''
    (HERE/'report.md').write_text(report,encoding='utf-8',newline='\n')
    for p in destination.iterdir(): p.chmod(0o444)
    copy.chmod(0o444)
    name='ATTEMPT003_COMPLETE_SHA256SUMS'
    scope=sorted(p for p in HERE.rglob('*') if p.is_file() and p.name not in {name,name+'.sha256'})
    (HERE/name).write_text(''.join(sha(p)+'  '+p.relative_to(HERE).as_posix()+'\n' for p in scope),encoding='utf-8',newline='\n')
    (HERE/(name+'.sha256')).write_text(sha(HERE/name)+'  '+name+'\n',encoding='utf-8',newline='\n')
    seal_check(HERE,name,exact=True)
    print(json.dumps({'status':'ATTEMPT003_VERIFIED_AND_SEALED','threshold_count':15,'threshold_sha256':validation['threshold_sha256'],'completion_seal':sha(HERE/name)}))

if __name__=='__main__': main()
