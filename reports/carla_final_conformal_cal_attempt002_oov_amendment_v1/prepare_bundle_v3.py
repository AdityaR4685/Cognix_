"""Offline preparation only. No scientific execution, networking, or fitting."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V1 = ROOT / 'reports/carla_final_conformal_cal_bundle_v1'
V2 = ROOT / 'reports/carla_final_conformal_cal_bundle_v2'
V3 = ROOT / 'reports/carla_final_conformal_cal_bundle_v3'
FAIL = ROOT / 'reports/carla_final_conformal_cal_attempt002_failure_v1'
HEAD = '1f8cb8cca741ad3f3b471b2d5da5129e0309c30b'
V2_SHA = 'dd5e559ebf16d73a67d7ecc720ec7f3200465c493e6f804479155e8b32007c66'
V2_ZIP_SHA = '2f1eeb7aff36a6608b643ef1afca5621df5b621b8da79f83ba1232d83418db79'
EVIDENCE_SHA = 'a21a6f2ea64eabcdb2c636f79da1cc18709146e583910cd97a52e991331221ea'
PARTIAL_SHA = '5c48384eee84003a00e672de9265e27c3f8a9634deed04250a0b852780c5ba91'
ARCHIVE = Path('C:/Users/Aditya/Downloads/cognix_final_conformal_cal_attempt002_failed_evidence.tar.gz')
READY = 'READY_FOR_SEPARATELY_AUTHORIZED_FINAL_CONFORMAL_CAL_ATTEMPT_003'
ZIP = 'cognix_final_conformal_cal_kaggle_v3.zip'
OUTPUT = '/kaggle/working/cognix_final_conformal_cal_attempt003_v1'

def guard(event, args):
    if event in {'socket.connect','socket.connect_ex','socket.getaddrinfo','socket.gethostbyname','socket.gethostbyaddr','socket.sendto','socket.sendmsg','urllib.Request', 'http.client.connect', 'os.system', 'os.posix_spawn'}:
        raise RuntimeError('OFFLINE_PREPARATION_REFUSES_NETWORK_OR_UNGUARDED_SHELL: ' + event)
sys.addaudithook(guard)

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()
def write(p, obj): text(p, json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + '\n')
def text(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(value, encoding='utf-8', newline='\n')
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def git(*args): return subprocess.check_output(['git', *args], cwd=ROOT).decode('utf-8')
def inventory(p): return {f.relative_to(p).as_posix(): sha(f) for f in sorted(p.rglob('*')) if f.is_file()}
def seal(p, name):
    text(p/name, ''.join(sha(f)+'  '+f.relative_to(p).as_posix()+'\n' for f in sorted(p.rglob('*'))
         if f.is_file() and f.name not in {name, name+'.sha256', ZIP, ZIP+'.sha256'}))
    text(p/(name+'.sha256'), sha(p/name)+'  '+name+'\n')
def offline(base, target, *args, guard_base=None):
    cmd = [sys.executable, '-B', str((guard_base or base)/'offline_checks.py'), str(base/target), *args]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode: raise RuntimeError(r.stdout+r.stderr)
    lines = [json.loads(x) for x in r.stdout.splitlines() if x.strip()]
    assert lines[-1]['offline_network_guard'] == 'PASSED'
    return dict(command=cmd, exit_code=r.returncode, stdout=r.stdout, stderr=r.stderr, result=lines[-2], network_guard=lines[-1])

def prepare():
    assert git('rev-parse', 'HEAD').strip() == HEAD, 'STOP_HEAD_MISMATCH'
    assert not V3.exists() and not FAIL.exists(), 'REFUSE_OVERWRITE'
    assert sha(V2/'BUNDLE_SHA256SUMS') == V2_SHA
    assert sha(V2/'cognix_final_conformal_cal_kaggle_v2.zip') == V2_ZIP_SHA
    before_status = git('status', '--porcelain=v1', '--untracked-files=all')
    assert not git('diff', '--name-only').strip() and not git('diff', '--cached', '--name-only').strip()
    # Metadata only for unrelated files: never open unrelated payloads to prove preservation.
    untracked = git('ls-files', '--others', '--exclude-standard', '-z').split('\0')
    metadata = {p: [int((ROOT/p).stat().st_size), int((ROOT/p).stat().st_mtime_ns)] for p in untracked if p and not p.startswith(HERE.relative_to(ROOT).as_posix()+'/')}
    baseline = {'HEAD': HEAD, 'v1': inventory(V1), 'v2': inventory(V2), 'unrelated_untracked_metadata': metadata, 'git_status_before': before_status.splitlines(), 'index_tree_before': git('ls-files', '--stage').splitlines()}
    write(HERE/'preservation_baseline.json', baseline)
    bindings = {'HEAD': HEAD, 'v2_manifest_sha256': V2_SHA, 'v2_ZIP_sha256': V2_ZIP_SHA,
                'attempt002_external_evidence_sha256': EVIDENCE_SHA, 'attempt002_partial_archive_sha256': PARTIAL_SHA}
    if ARCHIVE.exists():
        assert sha(ARCHIVE) == EVIDENCE_SHA, 'EXTERNAL_EVIDENCE_HASH_MISMATCH'
        FAIL.mkdir()
        shutil.copyfile(ARCHIVE, FAIL/ARCHIVE.name)
        allowed = {'attempt_started.json','access_ledger.json','cal_scenario_processing_ledger.jsonl','failure_exception_audit.json','result_manifest.json','RESULT_SHA256SUMS','RESULT_SHA256SUMS.sha256'}
        # Only allowlisted historical evidence; no filesystem extraction of archive paths.
        with tarfile.open(ARCHIVE, 'r:gz') as t:
            files = [m for m in t.getmembers() if m.isfile()]
            assert len(files) == 7 and {Path(m.name).name for m in files} == allowed
            assert not any(m.issym() or m.islnk() for m in t.getmembers())
            for m in files:
                assert m.size < 4*1024*1024
                (FAIL/'immutable_output').mkdir(exist_ok=True)
                (FAIL/'immutable_output'/Path(m.name).name).write_bytes(t.extractfile(m).read())
        ev = FAIL/'immutable_output'
        assert (ev/'RESULT_SHA256SUMS.sha256').read_text().split() == [sha(ev/'RESULT_SHA256SUMS'), 'RESULT_SHA256SUMS']
        for line in (ev/'RESULT_SHA256SUMS').read_text().splitlines():
            digest, rel = line.split('  ',1)
            assert rel in allowed and sha(ev/rel) == digest
        ledger = read(ev/'access_ledger.json')
        failure = read(ev/'failure_exception_audit.json')
        result = read(ev/'result_manifest.json')
        # Count records only. Do not JSON-parse, print, or analyze the contaminated score ledger.
        record_count = sum(bool(line.strip()) for line in (ev/'cal_scenario_processing_ledger.jsonl').read_bytes().splitlines())
        assert record_count == 92
        evidence_status = 'LOCALLY_VERIFIED_IMMUTABLE_HISTORICAL_EVIDENCE'
    else:
        FAIL.mkdir()
        ledger = {'automatic_retry': False,'cal_only_science': True,'cal_scenarios_completed':92,'compressed_bytes_received':62739644426,
          'compressed_sha256_partial_or_complete':PARTIAL_SHA,'eval_scenarios_decoded':0,'http_requests':1,
          'opaque_body_bytes_discarded_by_role':{'CAL':14032840421,'EVAL_OPAQUE_DISCARD':53633451131,'EXCLUDED_OPAQUE_DISCARD':240747299},
          'range_requests':0,'raw_archive_retained':False,'scenario_roots_seen':455,'transport_is_not_scientific_decoding':True}
        supplied = Path('C:/Users/Aditya/.codex/attachments/7909ff86-5071-4aea-80cb-6c33a402c47b/Pasted text.txt').read_text()
        start = supplied.index('{', supplied.index('Actual failure_exception_audit.json:'))
        failure, _ = json.JSONDecoder().raw_decode(supplied[start:])
        result = {'must_not_use_partial_scores_as_final_thresholds':True,'status':'INCOMPLETE'}
        record_count = None
        evidence_status = 'USER_SUPPLIED_EXTERNAL_EVIDENCE_NOT_LOCALLY_COPIED'
    assert ledger['http_requests'] == 1 and ledger['compressed_bytes_received'] == 62739644426
    assert ledger['compressed_sha256_partial_or_complete'] == PARTIAL_SHA and ledger['scenario_roots_seen'] == 455
    assert ledger['cal_scenarios_completed'] == failure['cal_completed'] == 92 and ledger['eval_scenarios_decoded'] == 0
    assert failure['message'] == 'segmentation class ids out of range [0, 29): min=1, max=32'
    assert failure['exception'] == 'ValueError' and failure['status'] == 'INCOMPLETE_NO_RETRY'
    assert result == {'must_not_use_partial_scores_as_final_thresholds':True,'status':'INCOMPLETE'}
    caveat = 'Historical opaque_body_bytes_discarded_by_role increments before CAL decode/discard decisions: CAL=14032840421 is body accounting, NOT entirely opaque/discarded. EVAL=53633451131 and EXCLUDED=240747299 are opaque transport/discard. Original ledger unchanged.'
    failure_record = {**bindings, 'evidence_status':evidence_status,'archive_search_scope':['workspace filenames','Downloads exact filename','Documents exact filename','Codex attachments exact filename'],
      'actual_access_ledger':ledger,'actual_failure_exception_audit':failure,'actual_result_manifest':result,
      'processing_ledger_record_count_only':record_count,'request_method':'GET','cal_total':125,'no_final_thresholds':True,
      'partial_scores_forbidden_as_final_thresholds':True,'no_retry_or_resume':True,'Attempt002_will_never_be_retried':True,
      'raw_archive_not_retained':True,'body_accounting_caveat':caveat,'observed_failure_min':1,'observed_failure_max':32,
      'class32_semantic_identity':'UNKNOWN','failing_scenario':'UNKNOWN_NOT_RECORDED','failing_tick':'UNKNOWN_NOT_RECORDED',
      'unrecorded_TEST_properties':'UNKNOWN; no frequency, pixel count, image mode or shape inferred',
      'partial_CAL_score_values_used_for_repair_or_tuning':False,'score_ledger_handling':'byte-for-byte preservation, hashing and record counting only',
      'v1_v2_immutable':True,'Attempt003_executed':False}
    write(FAIL/'failure_record.json',failure_record)
    text(FAIL/'report.md', '# Attempt 002 immutable failure record\n\n'+
      f'HEAD {HEAD}; v2 manifest {V2_SHA}; v2 ZIP {V2_ZIP_SHA}.\n\n'+
      'One GET; 62,739,644,426 compressed bytes; 455 roots; 92/125 CAL completed; 0 EVAL decoded. INCOMPLETE_NO_RETRY. No final thresholds. No retry/resume; raw archive not retained.\n\n'+
      f'Partial compressed SHA256: {PARTIAL_SHA}. External evidence SHA256: {EVIDENCE_SHA}; {evidence_status}.\n\n'+
      'Exact exception: segmentation class ids out of range [0, 29): min=1, max=32. Class 32 semantics, failing scenario/tick and unrecorded properties UNKNOWN. The 92 partial CAL score values are contaminated for repair selection; they were not parsed or used for policy/tuning.\n\n'+caveat+'\n\n'+
      'v1/v2 remain immutable. Attempt 002 will never be retried. Exact actual traceback follows (historical, not synthetic).\n\n```text\n'+failure['traceback']+'```\n')
    seal(FAIL,'FAILURE_SHA256SUMS')
    amendment = {**bindings,'policy_decision':'ACCEPT_FIXED_OOV_TO_EXISTING_OTHER_22','decision_basis':'Human-supplied public taxonomy and frozen model dimensional compatibility; no CAL outcome/frequency use',
      'supplied_taxonomy_provenance':{'source_status':'HUMAN_SUPPLIED_EXTERNAL_PROVENANCE_NOT_NETWORK_FETCHED','classic_CARLA':{'Other':22,'GuardRail':28},'carla_gen_README_ObjectLabel_h_patch':{'MyAnimals':29,'MyCars':30}},
      'policy':{'input':'uint8 2-D or 3-D; channel 0 selected first','preserve':'all IDs 0..28','map':'ALL IDs >28 to existing Other bin 22','histogram_dimensions':29,'extractor':'unchanged frozen segmentation_histogram_features'},
      'class32_semantic_identity':'UNKNOWN','partial_access_disclosure':'Post-Attempt-002 scientific compatibility amendment after 92 partial CAL scenarios; not method/science identity with v2',
      'partial_scores_used':False,'frequencies_used':False,'EVAL_scientific_decoding':0,'Attempt002_thresholds_exist':False,
      'Attempt003_recompute':'ALL 125 CAL scenarios from byte zero; no partial score reuse; separately authorized only',
      'no_fitting_training_tuning':True,'v1_v2_immutable':True,
      'unchanged':['CAL/EVAL membership','historical exclusions','all 15 checkpoints and hashes','one-class fitted parameters','Camera feature path','IMU feature path','graph architecture and weights','uncertainty definitions','label semantics','conformal score','alpha=.05','k=120','inclusive <=','scenario max','runtime version pins','archive URL/size/SHA','single-stream transport','EVAL opaque discard','no tuning/training','one attempt/no auto retry'],
      'amended':['TEST-time segmentation OOV preprocessing: all uint8 IDs >28 -> existing bin 22, after channel-0 selection'],
      'administrative_changes':['new attempt003 output path','new v3 seals and provenance','explicit enforcement of existing pandas==2.2.3 runtime_requirements pin; existing runtime_lock.json kept byte-identical'],
      'rule_does_not_depend_on':['29/30/31/32 identity','anomaly directory','town','scenario type','labels','CAL membership','frequency','outcomes'],
      'scientific_limitation':'Deterministic generic fallback preserves dimension but changes OOV inputs and can change scores; does not prove semantic equivalence or eliminate post-access amendment concerns.'}
    write(HERE/'amendment_record.json',amendment)
    text(HERE/'root_cause_audit.md', '# Offline root cause and fixed policy decision\n\n'+
      'v2 separates strict uint8 HxWx3 camera validation from uint8 2-D/3-D segmentation validation. Failure traceback enters the unchanged extractor at real_features.py:95, whose class range check is [0,29). Thus representation handling is no longer this failure. No unrecorded TEST image shape or mode is inferred.\n\n'+
      'The frozen extractor selects channel 0 for 3-D, bins 29 dimensions, and divides by total pixels (the old L2 docstring is inaccurate). Frozen graph_fit_export.OFFSETS Seg=(18,47) and restored one-class vectors bind 29-dimensional input. Extending n_classes to 33 invalidates the fitted model; no refit/checkpoint changes are allowed.\n\n'+
      'Human-supplied classic CARLA taxonomy defines Other=22 and GuardRail=28. The supplied carla-gen README patch names custom labels 29 and 30. These facts justify possibility of custom/OOV labels beyond 28; they establish no semantic identity for observed ID 32. Class 32 remains UNKNOWN. These sources were not fetched.\n\n'+
      'Decision: uniformly collapse every uint8 semantic ID >28 into existing generic Other=22 after selecting channel 0. Preserve 0..28 and the unchanged 29-bin extractor. This is a dimensional compatibility policy, not an assertion that an OOV class has the CARLA Other ground-truth meaning. No partial CAL score, method comparison, outcome, frequency or class count enters the decision.\n\n'+
      'Transparent post-failure scientific compatibility amendment: 92 partial CAL scenarios had already been accessed. The policy can change scores for OOV inputs. No v2 science-identity claim is made. Attempt 003, if separately authorized, starts at byte zero and recomputes all 125 CAL scenarios. No Attempt-002 thresholds exist or may be reused; no EVAL science was decoded. v1/v2 remain immutable.\n\n'+
      'Runtime audit: v2 runtime_requirements.txt already pins pandas==2.2.3, but runtime_check omits its comparison. v3 adds an explicit check of that existing pin while preserving runtime_lock.json and runtime_requirements.txt byte-for-byte. This changes enforcement, not version selection or scientific method.\n')
    # Copy only v2 material; stale derived reports/build script/notebook/ZIP omitted.
    omit = {'BUNDLE_SHA256SUMS','BUNDLE_SHA256SUMS.sha256','cognix_final_conformal_cal_kaggle_v2.zip','cognix_final_conformal_cal_kaggle_v2.zip.sha256','cognix_final_conformal_cal_v2.ipynb','prepare_bundle_v2.py','post_failure_repair_record.json','post_failure_repair_record.md','regression_verification_results.json','synthetic_verification_results.json','offline_test_transcripts.json','pre_access_validation.json','unchanged_scientific_bindings.json','README.md'}
    V3.mkdir()
    for p in sorted(V2.iterdir()):
        if p.name in omit: continue
        if p.is_dir(): shutil.copytree(p,V3/p.name)
        else: shutil.copyfile(p,V3/p.name)
    # Historical provenance remains inert and explicitly scoped as historical.
    shutil.copyfile(V2/'runner.py',V3/'evidence/v2_runner.py.txt')
    shutil.copyfile(V2/'regression_verification.py',V3/'evidence/v2_regression_verification.py.txt')
    shutil.copyfile(V2/'scientific_bindings.json',V3/'evidence/v2_scientific_bindings.json')
    shutil.copyfile(HERE/'amendment_record.json',V3/'segmentation_oov_amendment.json')
    shutil.copyfile(HERE/'root_cause_audit.md',V3/'segmentation_oov_amendment.md')
    shutil.copyfile(FAIL/'failure_record.json',V3/'attempt002_failure_binding.json')
    old = (V2/'runner.py').read_text()
    insertion = '''def map_segmentation_oov_to_frozen_vocabulary(array):
    """Fixed post-failure amendment: channel 0, all OOV -> existing Other=22."""
    import numpy as np
    if not isinstance(array, np.ndarray) or array.dtype != np.uint8 or array.ndim not in (2, 3) or (array.ndim == 3 and array.shape[2] < 1):
        raise ValueError("FROZEN_SEGMENTATION_SCHEMA_REQUIRES_UINT8_2D_OR_3D_CHANNEL_MAP")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("FROZEN_SEGMENTATION_SCHEMA_REQUIRES_NONEMPTY_SPATIAL_MAP")
    class_map = array[..., 0] if array.ndim == 3 else array
    mapped = class_map.copy()
    mapped[mapped > 28] = 22
    return mapped


'''
    gate = '''                if array.dtype != np.uint8 or array.ndim not in (2, 3) or (array.ndim == 3 and array.shape[2] < 1):
                    raise ValueError("FROZEN_SEGMENTATION_SCHEMA_REQUIRES_UINT8_2D_OR_3D_CHANNEL_MAP")
                feature = segmentation_histogram_features(array)'''
    new_gate = '                feature = segmentation_histogram_features(map_segmentation_oov_to_frozen_vocabulary(array))'
    assert old.count(gate) == 1
    new = old.replace('class CalProcessor:',insertion+'class CalProcessor:',1).replace(gate,new_gate).replace('/kaggle/working/cognix_final_conformal_cal_attempt002_v1',OUTPUT)
    # Existing version pin was never enforced: enforce explicitly without changing lock bytes.
    new = new.replace('    import pyarrow\n    expected', '    import pyarrow\n    import pandas\n    expected',1)
    new = new.replace('    torch.set_num_threads(1)', '    if pandas.__version__ != "2.2.3":\n        raise RuntimeError("SEALED_PANDAS_VERSION_MISMATCH: pandas==2.2.3")\n    torch.set_num_threads(1)',1)
    text(V3/'runner.py',new)
    inverse = new.replace(insertion,'',1).replace(new_gate,gate).replace(OUTPUT,'/kaggle/working/cognix_final_conformal_cal_attempt002_v1')
    inverse = inverse.replace('    import pyarrow\n    import pandas\n    expected','    import pyarrow\n    expected',1).replace('    if pandas.__version__ != "2.2.3":\n        raise RuntimeError("SEALED_PANDAS_VERSION_MISMATCH: pandas==2.2.3")\n','',1)
    assert inverse == old, 'UNDECLARED_RUNNER_CHANGE'
    verifier = (V2/'verify_bundle.py').read_text().replace('cognix_final_conformal_cal_kaggle_v2.zip',ZIP).replace('READY_FOR_SEPARATELY_AUTHORIZED_FINAL_CONFORMAL_CAL_ATTEMPT_002',READY)
    text(V3/'verify_bundle.py',verifier)
    # Only the old OOV rejection assertion is replaced. All other prior assertions preserved.
    regression = (V2/'regression_verification.py').read_text()
    old_assert = '''        expect_error(lambda: CalProcessor({}, {}, cal, out, "cpu")(c, "segmentation-front/000001.png", b.getvalue()), "segmentation class ids out of range [0, 29): min=29, max=29")
        passed("frozen_segmentation_class_range_refusal_preserved")'''
    new_assert = '''        amended = CalProcessor({}, {}, cal, out, "cpu")
        amended(c, "segmentation-front/000001.png", b.getvalue())
        expected = np.zeros(29, dtype=np.float64); expected[22] = 1.0
        assert np.array_equal(amended.data["seg"][1], expected)
        passed("preregistered_uniform_OOV_to_existing_Other_replaces_v2_OOV_refusal")'''
    assert regression.count(old_assert)==1
    text(V3/'regression_verification.py',regression.replace(old_assert,new_assert))
    write(HERE/'prior_regression_amendment.json',{'source_v2_sha256':sha(V2/'regression_verification.py'),'removed_assertion':old_assert,'replacement_assertion':new_assert,'reason':'Old v2 reject-ID29 behavior is the explicitly amended OOV rule. Every other prior regression assertion is unchanged; frozen extractor still rejects OOV when called directly.'})
    nb = read(V2/'cognix_final_conformal_cal_v2.ipynb')
    for cell in nb['cells']:
        cell['source']=[s.replace('Attempt 002','Attempt 003').replace('(v2)','(v3)').replace('bundle_v2','bundle_v3').replace('attempt002_v1','attempt003_v1') for s in cell['source']]
    nb['cells'][0]['source'][-1] = 'Attempt 002 failed after 92 CAL scenarios; v3 declares uniform OOV-to-Other preprocessing. Partial scores are never reused. Attempt 003 has not occurred and requires separate authorization. Output: '+OUTPUT+'.\n'
    write(V3/'cognix_final_conformal_cal_v3.ipynb',nb)
    sci = read(V2/'scientific_bindings.json')
    sci['git_HEAD']=HEAD
    sci['preparation_context']='v3 post-Attempt-002 amendment; historical source/protocol bindings retain their original provenance'
    sci['segmentation_preprocessing_amendment']=amendment['policy']
    sci['bundle_scientific_file_sha256']['runner.py']=sha(V3/'runner.py')
    write(V3/'scientific_bindings.json',sci)
    unchanged = {rel: digest for rel,digest in sci['bundle_scientific_file_sha256'].items() if rel!='runner.py'}
    for rel,digest in unchanged.items(): assert sha(V3/rel)==sha(V2/rel)==digest
    for rel in ['final_conformal_cal.txt','final_evaluation.txt','partition_binding.json','model_registry.json','runtime_requirements.txt','evidence/protocol/conformal_protocol.json','evidence/protocol/frozen_method_bindings.json']:
        assert sha(V3/rel)==sha(V2/rel)
        unchanged[rel]=sha(V3/rel)
    originals = read(V3/'model_registry.json')['scorers']
    for row in originals: assert sha(ROOT/row['checkpoint'])==row['file_sha256']
    write(V3/'unchanged_scientific_bindings.json',{'v2_binding':bindings,'unchanged_file_sha256':unchanged,'original_checkpoint_bindings':originals,
      'unchanged_semantics':amendment['unchanged'],'amended_scientific_rule':amendment['amended'],'runner_changes':amendment['administrative_changes'],
      'identity_claim':'No claim of v2/v3 scientific identity for OOV inputs','runner_exact_inverse_comparison_passed':True})
    # Append preservation attributes without rewriting any existing attribute bytes.
    attr = ROOT/'.gitattributes'
    original = attr.read_bytes()
    added = b''.join((p+'/** -text\n').encode() for p in ['reports/carla_final_conformal_cal_attempt002_failure_v1','reports/carla_final_conformal_cal_attempt002_oov_amendment_v1','reports/carla_final_conformal_cal_bundle_v3'])
    attr.write_bytes(original+(b'\n' if original and not original.endswith(b'\n') else b'')+added)
    print(json.dumps({'prepared_source':True,'external_evidence_status':evidence_status,'record_count_only':record_count,'TEST_network_requests':0,'TEST_payload_bytes_accessed':0,'Attempt003_executed':False}))

def finalize():
    assert git('rev-parse','HEAD').strip()==HEAD
    assert not (V3/'BUNDLE_SHA256SUMS').exists(), 'REFUSE_RESEAL_FINISHED_BUNDLE'
    baseline=read(HERE/'preservation_baseline.json')
    # Make enforcement-only differences explicit before any preparation seal.
    amendment=read(HERE/'amendment_record.json')
    device_change='require exact Tesla T4 device name, rather than v2 substring match; same frozen device pin'
    if device_change not in amendment['administrative_changes']: amendment['administrative_changes'].append(device_change)
    write(HERE/'amendment_record.json',amendment)
    shutil.copyfile(HERE/'amendment_record.json',V3/'segmentation_oov_amendment.json')
    device_note='\nThe existing device pin is exactly one visible Tesla T4. V3 explicitly enforces the full device name; v2 checked a T4 substring. The pinned hardware is unchanged.\n'
    audit=(HERE/'root_cause_audit.md').read_text()
    if device_note not in audit: audit+=device_note
    text(HERE/'root_cause_audit.md',audit); text(V3/'segmentation_oov_amendment.md',audit)
    policy=read(V3/'protected_eval_policy.json'); policy['output']=OUTPUT
    policy['attempt003_executed']=False; policy['partial_attempt002_scores_reused']=False
    policy['body_accounting_caveat']=read(FAIL/'failure_record.json')['body_accounting_caveat']
    write(V3/'protected_eval_policy.json',policy)
    old=(V2/'runner.py').read_text(); new=(V3/'runner.py').read_text()
    start=new.index('def map_segmentation_oov_to_frozen_vocabulary('); end=new.index('class CalProcessor:',start)
    gate='''                if array.dtype != np.uint8 or array.ndim not in (2, 3) or (array.ndim == 3 and array.shape[2] < 1):
                    raise ValueError("FROZEN_SEGMENTATION_SCHEMA_REQUIRES_UINT8_2D_OR_3D_CHANNEL_MAP")
                feature = segmentation_histogram_features(array)'''
    inverse=(new[:start]+new[end:]).replace('                feature = segmentation_histogram_features(map_segmentation_oov_to_frozen_vocabulary(array))',gate)
    inverse=inverse.replace(OUTPUT,'/kaggle/working/cognix_final_conformal_cal_attempt002_v1')
    inverse=inverse.replace('    import pyarrow\n    import pandas\n    expected','    import pyarrow\n    expected',1).replace('    if pandas.__version__ != "2.2.3":\n        raise RuntimeError("SEALED_PANDAS_VERSION_MISMATCH: pandas==2.2.3")\n','',1)
    inverse=inverse.replace('torch.cuda.get_device_name(0) != "Tesla T4"','"T4" not in torch.cuda.get_device_name(0)',1)
    assert inverse==old, 'UNDECLARED_RUNNER_CHANGE'
    sci=read(V3/'scientific_bindings.json'); sci['bundle_scientific_file_sha256']['runner.py']=sha(V3/'runner.py'); write(V3/'scientific_bindings.json',sci)
    unchanged=read(V3/'unchanged_scientific_bindings.json'); unchanged['runner_changes']=amendment['administrative_changes']; write(V3/'unchanged_scientific_bindings.json',unchanged)
    # Validate historical evidence against the user's JSON values, without score parsing.
    supplied=Path('C:/Users/Aditya/.codex/attachments/7909ff86-5071-4aea-80cb-6c33a402c47b/Pasted text.txt').read_text()
    failure_record=read(FAIL/'failure_record.json')
    for heading,key in [('Actual failure_exception_audit.json:','actual_failure_exception_audit'),('Actual access_ledger.json:','actual_access_ledger'),('Actual result_manifest.json:','actual_result_manifest')]:
        start=supplied.index('{',supplied.index(heading)); obj,_=json.JSONDecoder().raw_decode(supplied[start:]); assert failure_record[key]==obj
    assert read(FAIL/'immutable_output/attempt_started.json')['manifest_sha256']==V2_SHA
    transcripts=read(V3/'offline_test_transcripts.json') if (V3/'offline_test_transcripts.json').exists() else {}
    for version,base,tests in [('v1',V1,['verify_bundle.py','synthetic_verification.py']),('v2',V2,['verify_bundle.py','synthetic_verification.py','regression_verification.py']),('v3',V3,['synthetic_verification.py','regression_verification.py','oov_regression_verification.py'])]:
        for target in tests:
            if version+'/'+target in transcripts: continue
            print(json.dumps({'offline_check_started':version+'/'+target}),flush=True)
            transcripts[version+'/'+target]=offline(base,target,guard_base=V2 if version=='v1' else base)
    cmd=[sys.executable,'-B',str(V3/'offline_checks.py'),'--guard-selftest']
    if 'guard_selftest' not in transcripts:
        r=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True); assert r.returncode==0,r.stderr
        lines=[json.loads(s) for s in r.stdout.splitlines()]
        transcripts['guard_selftest']={'command':cmd,'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'result':lines[0],'network_guard':lines[-1]}
    syn=transcripts['v3/synthetic_verification.py']['result']; reg=transcripts['v3/regression_verification.py']['result']; oov=transcripts['v3/oov_regression_verification.py']['result']
    write(V3/'synthetic_verification_results.json',syn); write(V3/'regression_verification_results.json',reg); write(V3/'oov_regression_verification_results.json',oov)
    # The only previously amended test assertion is documented; prove other bytes identical.
    test_amendment=read(HERE/'prior_regression_amendment.json')
    assert (V3/'regression_verification.py').read_text().replace(test_amendment['replacement_assertion'],test_amendment['removed_assertion'])==(V2/'regression_verification.py').read_text()
    assert sha(V3/'synthetic_verification.py')==sha(V2/'synthetic_verification.py')==sha(V1/'synthetic_verification.py')
    assert inventory(V1)==baseline['v1'] and inventory(V2)==baseline['v2']
    assert sha(ARCHIVE)==EVIDENCE_SHA and sha(FAIL/ARCHIVE.name)==EVIDENCE_SHA
    original_hashes={row['checkpoint']:sha(ROOT/row['checkpoint']) for row in unchanged['original_checkpoint_bindings']}
    assert all(original_hashes[row['checkpoint']]==row['file_sha256'] for row in unchanged['original_checkpoint_bindings'])
    for rel,h in unchanged['unchanged_file_sha256'].items(): assert sha(V2/rel)==sha(V3/rel)==h
    for p,metadata in baseline['unrelated_untracked_metadata'].items():
        stat=(ROOT/p).stat(); assert [stat.st_size,stat.st_mtime_ns]==metadata
    assert git('ls-files','--stage').splitlines()==baseline['index_tree_before']
    assert git('diff','--name-only').splitlines()==['.gitattributes']
    write(V3/'offline_test_transcripts.json',transcripts)
    import platform,numpy,PIL,pyarrow,pandas,torch
    actual={'Python':platform.python_version(),'NumPy':numpy.__version__,'Pillow':PIL.__version__,'PyArrow':pyarrow.__version__,'pandas':pandas.__version__,'PyTorch':torch.__version__}
    validation={'readiness':READY,'preparation_passed':True,'Attempt003_authorized':False,'Attempt003_executed':False,'Kaggle_notebook_executed':False,
        'CAL_count':125,'EVAL_count':500,'frozen_scorer_states':15,'overlap':0,'union':625,'all_frozen_states_strictly_loadable':True,
        'TEST_network_requests':0,'TEST_payload_bytes_accessed':0,'no_fitting_training_tuning':True,
        'synthetic_verification_groups_passed':syn['synthetic_verification_groups_passed'],'synthetic_verification_groups_failed':0,
        'regression_groups_passed':reg['regression_groups_passed'],'regression_groups_failed':0,
        'oov_regression_groups_passed':oov['oov_regression_groups_passed'],'oov_regression_groups_failed':0,
        'v1_all_bytes_unchanged':True,'v2_all_bytes_unchanged':True,'Attempt002_evidence_unchanged':True,
        'local_offline_runtime':actual,'runtime_version_pins_unchanged':True,
        'runtime_enforcement_amendments':['existing pandas==2.2.3 pin now explicitly checked','exact existing Tesla T4 name checked'],
        'runtime_limitation':'Local synthetic CPU checks do not certify locked Kaggle GPU numerics. Full exact runtime_check must pass before any future separately authorized network access.',
        'network_proof_limit':'Process-local Python audit hooks in all verification subprocesses, explicit refusal of network/child-process events; no OS-wide packet capture or native-library networking proof.',
        'class32_semantic_identity':'UNKNOWN','partial_CAL_score_values_used':False,'scientific_amendment':amendment['amended'],
        'unchanged_binding_record':'unchanged_scientific_bindings.json','scientific_identity_claim':False,'scientific_execution_success':False,
        'historical_traceback':'Actual immutable evidence; distinguished from synthetic corroboration','git_staged':False,'commit':False,'push':False,
        'unrelated_untracked_preservation_evidence':'Write operations confined to new directories and .gitattributes; all existing unrelated untracked file size/mtime and Git index unchanged. No prohibited payload opened for preservation hashing.'}
    write(V3/'pre_access_validation.json',validation)
    readme='''# COGNIX CARLA final conformal CAL v3 — preparation only

READY_FOR_SEPARATELY_AUTHORIZED_FINAL_CONFORMAL_CAL_ATTEMPT_003

This readiness means offline preparation passed. It does not authorize TEST access and does not mean Attempt 003 occurred. No scientific runner execution, network request, TEST payload access, fitting, tuning, staging, commit or push occurred during preparation.

Attempt 002 is INCOMPLETE_NO_RETRY: 92/125 CAL completed, 0 EVAL decoded, one GET, 62,739,644,426 compressed bytes, 455 roots, no final thresholds. Its immutable historical evidence is sealed separately. Never retry/resume Attempt 002 or reuse its partial CAL scores. Class 32 semantic identity, failing scenario/tick and unrecorded properties remain UNKNOWN.

V3 explicitly amends segmentation preprocessing after the failure: accept uint8 2-D/3-D, select channel 0 for 3-D, preserve IDs 0..28, collapse every ID >28 to existing CARLA Other=22, then call the byte-identical 29-bin extractor. No special cases, frequencies, outcomes, class meaning, labels or membership inform mapping. The 92 partial scores were not parsed or used for repair selection. This is a post-failure scientific compatibility amendment, not a v2/v3 science-identity claim. It can change OOV scores.

CAL/EVAL membership (125/500), exclusions, all 15 checkpoints/state hashes, fitted parameters, Camera/IMU features, graph/weights/UQ, label semantics, conformal definition and runtime pins are unchanged. Scores are 1-P_t(Y_t), scenario max over t>=1, alpha=.05, augmented 125 scores plus infinity, k=120, inclusive <=, one pooled threshold per scorer. No method ranking or EVAL metrics. See unchanged_scientific_bindings.json for every hash and segmentation_oov_amendment.json for the declared change.

Runtime lock: Python 3.12.13, NumPy 2.0.2, Pillow 12.3.0, PyArrow 25.0.1, pandas 2.2.3, PyTorch 2.10.0+cu128, CUDA 12.8, cuDNN 91002, exactly one visible Tesla T4. Lock and requirements files are byte-identical to v2; v3 now explicitly enforces the existing pandas pin and exact Tesla T4 name. Local CPU tests do not certify those GPU numerics.

Attach the upload ZIP as a Kaggle input and check the externally recorded ZIP SHA256 before extraction. Extract into /kaggle/working/cognix_cal_bundle_v3. Notebook execution defaults to False. Preflight is offline:

    python -B offline_checks.py runner.py --preflight

Only a later separately authorized human action may use the existing --execute-authorized-final-cal switch. No automatic execution is provided by preparation. Output is fixed to /kaggle/working/cognix_final_conformal_cal_attempt003_v1; existence refuses execution. A permanent attempt marker precedes access; exactly one full byte-zero sequential GET, no redirects, Range, retry, resume or raw-archive retention. Full expected size 91538225599 and SHA256 267e48f2249deb0269ad950aa81bca57dc02e3bbdf2d73acc172af267b18254a and complete 125-CAL membership are required before thresholds. Failures seal access/failure/result evidence and forbid partial thresholds. Every CAL scenario is recomputed; no Attempt-002 partial scores are read.

EVAL/excluded bodies physically transit and decompress but remain opaque, never reaching image/Feather/label/feature/model decoders. Historical opaque_body_bytes_discarded_by_role counts CAL bytes before decode/discard decisions; its CAL value is not entirely opaque/discarded. EVAL/EXCLUDED values are opaque discard. Original streaming code/ledger is unchanged.

Verification uses only synthetic fixtures and frozen TRAIN artifacts, with process-local Python audit hooks. No OS-wide capture is claimed. Historical evidence/*.txt and evidence/v2_scientific_bindings.json retain explicitly historical provenance and are never execution targets. BUNDLE_SHA256SUMS covers all bundle files except itself, detached seal, upload ZIP and detached ZIP digest; ZIP includes manifest and its detached seal, and excludes itself. v1/v2 remain byte-identical.
'''
    text(V3/'README.md',readme)
    write(HERE/'verification_summary.json',{'all_groups_passed':sum(r['result'].get('synthetic_verification_groups_passed',r['result'].get('regression_groups_passed',r['result'].get('oov_regression_groups_passed',0))) for r in transcripts.values()),
        'v3_groups_passed':syn['synthetic_verification_groups_passed']+reg['regression_groups_passed']+oov['oov_regression_groups_passed'],
        'groups_failed':0,'fitted_parameter_file_sha256':sha(V3/'upstream/fitted_oneclass_parameters.npz'),
        'fitted_parameter_array_sha256':oov['fitted_parameter_array_sha256'],'runner_exact_inverse_comparison_passed':True,'prior_regression_exact_inverse_comparison_passed':True,
        'v1_v2_inventory_before_after_identical':True,'external_evidence_verified_and_preserved':True,'user_supplied_evidence_JSON_matches_actual':True,'original_checkpoint_sha256':original_hashes})
    seal(HERE,'AMENDMENT_SHA256SUMS')
    write(V3/'amendment_binding.json',{'HEAD':HEAD,'amendment_manifest_sha256':sha(HERE/'AMENDMENT_SHA256SUMS'),'failure_manifest_sha256':sha(FAIL/'FAILURE_SHA256SUMS'),
        'external_evidence_sha256':EVIDENCE_SHA,'v2_manifest_sha256':V2_SHA,'v2_ZIP_sha256':V2_ZIP_SHA})
    seal(V3,'BUNDLE_SHA256SUMS')
    transcripts['v3/final_integrity']=offline(V3,'verify_bundle.py')
    transcripts['v3/final_preflight']=offline(V3,'runner.py','--preflight')
    # Deterministic ZIP metadata and sorted paths; no resealing of prior versions.
    with zipfile.ZipFile(V3/ZIP,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(V3.rglob('*')):
            if not p.is_file() or p.name in {ZIP,ZIP+'.sha256'}: continue
            info=zipfile.ZipInfo(p.relative_to(V3).as_posix(),date_time=(2026,10,4,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED; info.external_attr=0o100644<<16; info.create_system=3
            z.writestr(info,p.read_bytes(),compresslevel=9)
    text(V3/(ZIP+'.sha256'),sha(V3/ZIP)+'  '+ZIP+'\n')
    with zipfile.ZipFile(V3/ZIP) as z:
        assert z.testzip() is None
        expected_files={p.relative_to(V3).as_posix() for p in V3.rglob('*') if p.is_file() and p.name not in {ZIP,ZIP+'.sha256'}}
        assert set(z.namelist())==expected_files and len(z.namelist())==len(expected_files)
        for rel in z.namelist(): assert hashlib.sha256(z.read(rel)).hexdigest()==sha(V3/rel)
    assert inventory(V1)==baseline['v1'] and inventory(V2)==baseline['v2']
    review_commands=['git rev-parse HEAD','git diff -- .gitattributes',
        "Get-Content 'reports/carla_final_conformal_cal_attempt002_failure_v1/report.md'",
        "Get-Content 'reports/carla_final_conformal_cal_attempt002_oov_amendment_v1/root_cause_audit.md'",
        "Get-Content 'reports/carla_final_conformal_cal_bundle_v3/unchanged_scientific_bindings.json'",
        "Get-Content 'reports/carla_final_conformal_cal_bundle_v3/pre_access_validation.json'",
        "Get-Content 'reports/carla_final_conformal_cal_v3_review.json'",
        "& '"+sys.executable+"' -B reports/carla_final_conformal_cal_bundle_v3/offline_checks.py reports/carla_final_conformal_cal_bundle_v3/verify_bundle.py",
        "& '"+sys.executable+"' -B reports/carla_final_conformal_cal_bundle_v3/offline_checks.py reports/carla_final_conformal_cal_bundle_v3/runner.py --preflight",
        'git status --short']
    stage_commands=['git add -- .gitattributes','git add -- reports/carla_final_conformal_cal_attempt002_failure_v1','git add -- reports/carla_final_conformal_cal_attempt002_oov_amendment_v1','git add -- reports/carla_final_conformal_cal_bundle_v3','git add -- reports/carla_final_conformal_cal_v3_review.json']
    review_path=ROOT/'reports/carla_final_conformal_cal_v3_review.json'
    write(review_path,{})
    status=git('status','--porcelain=v1')
    created=[FAIL.relative_to(ROOT).as_posix(),HERE.relative_to(ROOT).as_posix(),V3.relative_to(ROOT).as_posix(),review_path.relative_to(ROOT).as_posix()]
    review={**validation,'HEAD':HEAD,'git_subject':git('log','-1','--format=%s').strip(),'created_paths':created,
        'created_files':[p.relative_to(ROOT).as_posix() for d in (FAIL,HERE,V3) for p in sorted(d.rglob('*')) if p.is_file()]+[review_path.relative_to(ROOT).as_posix()],
        'tracked_files_modified':['.gitattributes'],'git_status':status.splitlines(),'unrelated_untracked_files_preserved':True,
        'unrelated_untracked_count':len(baseline['unrelated_untracked_metadata']),'git_index_unchanged':True,
        'Attempt002_evidence':failure_record,'root_cause':'unchanged frozen 29-bin vocabulary rejects OOV; v2 representation handling is not this failure',
        'OOV_policy_decision':amendment,'unchanged_bindings_and_hashes':unchanged,'verification_results':read(HERE/'verification_summary.json'),
        'offline_verification_transcripts':transcripts,'failure_manifest_sha256':sha(FAIL/'FAILURE_SHA256SUMS'),'amendment_manifest_sha256':sha(HERE/'AMENDMENT_SHA256SUMS'),
        'v3_manifest_sha256':sha(V3/'BUNDLE_SHA256SUMS'),'v3_ZIP_sha256':sha(V3/ZIP),'v3_ZIP_bytes':(V3/ZIP).stat().st_size,
        'v2_manifest_sha256':V2_SHA,'v2_ZIP_sha256':V2_ZIP_SHA,'v3_ZIP_complete_file_and_byte_verification_passed':True,
        'human_review_commands':review_commands,'human_exact_path_staging_commands_NOT_EXECUTED':stage_commands,
        'staging_policy':'Commands supplied only for later human review. No git add, commit or push executed.'}
    write(review_path,review)
    # Review is outside bundle/amendment to avoid circular seals; record all final transcript/hash claims here.
    print(json.dumps({k:review[k] for k in ['HEAD','created_paths','tracked_files_modified','root_cause','class32_semantic_identity','partial_CAL_score_values_used','verification_results','v3_manifest_sha256','v3_ZIP_sha256','TEST_network_requests','TEST_payload_bytes_accessed','Attempt003_executed','readiness','human_review_commands','human_exact_path_staging_commands_NOT_EXECUTED']},sort_keys=True,indent=2))

if __name__=='__main__':
    if sys.argv[1:]==['--prepare']: prepare()
    elif sys.argv[1:]==['--seal']: finalize()
    else: raise RuntimeError('Explicit offline builder mode required; never execute runner')
