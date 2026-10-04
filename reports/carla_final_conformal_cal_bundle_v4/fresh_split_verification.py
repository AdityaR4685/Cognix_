"""Offline synthetic four-role and fresh-conformal regression; no execute()."""
import ast
from contextlib import ExitStack
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(n):return json.loads((HERE/n).read_text())
def ids(n):return (HERE/n).read_text().splitlines()
def source_function(s,name):
    node=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name==name)
    return ast.get_source_segment(s,node)
def run():
    import numpy as np
    import torch
    import pandas as pd
    import pyarrow as pa
    import pyarrow.feather as feather
    from PIL import Image
    from frozen_runtime import load,recursive_content_hash
    models,agents=load()
    from cognix.adapters.carla import real_features as features
    from runner import CalProcessor,map_segmentation_oov_to_frozen_vocabulary as mapping,require_fresh_cal_completion
    from streaming import scan
    from calibration_adapter import threshold
    from synthetic_verification import fake_archive
    results=[]
    def passed(n):results.append({'group':n,'status':'PASSED'})
    def rejects(fn,message=None):
        try:fn()
        except (ValueError,RuntimeError,EOFError) as exc:
            if message:assert str(exc)==message,(str(exc),message)
            return
        raise AssertionError('Expected refusal')
    active=(HERE/'runner.py').read_text();precursor=(HERE/'evidence/v3_runner.py.txt').read_text()
    assert source_function(active,'map_segmentation_oov_to_frozen_vocabulary')==source_function(precursor,'map_segmentation_oov_to_frozen_vocabulary')
    assert sha(HERE/'segmentation_oov_amendment.json')==read('unchanged_scorer_bindings.json')['identical_v3_file_sha256']['segmentation_oov_amendment.json']
    passed('01_v3_OOV_function_source_bytes_and_policy_record_identical')
    all_ids=np.arange(256,dtype=np.uint8).reshape(16,16)
    inv=np.arange(29,dtype=np.uint8).reshape(1,29)
    for a in (inv,np.tile(inv,(4,3))):assert np.array_equal(features.segmentation_histogram_features(mapping(a)),features.segmentation_histogram_features(a))
    passed('02_IDs_0_through_28_exact_v3_feature_identity')
    mapped=mapping(all_ids);expected=all_ids.copy();expected[expected>28]=22
    assert np.array_equal(mapped,expected) and np.array_equal(all_ids,np.arange(256,dtype=np.uint8).reshape(16,16))
    passed('03_exhaustive_IDs_29_through_255_to_22_no_mutation')
    v=features.segmentation_histogram_features(mapped);assert v.shape==(29,) and v.dtype==np.float64 and np.isfinite(v).all() and v.sum()==1
    a3=np.stack([all_ids,np.full_like(all_ids,255),np.zeros_like(all_ids)],axis=-1);assert np.array_equal(mapping(a3),mapped)
    passed('04_29D_normalized_histogram_and_channel_zero_semantics')
    for a in (np.zeros((2,2),np.uint16),np.zeros((2,2),np.float32),np.zeros((2,),np.uint8),np.zeros((2,2,0),np.uint8)):
        rejects(lambda a=a:mapping(a),'FROZEN_SEGMENTATION_SCHEMA_REQUIRES_UINT8_2D_OR_3D_CHANNEL_MAP')
    passed('05_invalid_segmentation_dtype_schema_loud_refusal')
    def camera_branch(s):
        tree=ast.parse(s);node=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=="kind == 'camera'")
        return ast.dump(ast.Module(body=node.body,type_ignores=[]),include_attributes=False)
    assert camera_branch(active)==camera_branch(precursor)
    passed('06_camera_path_AST_identical_to_v3')
    registry=read('model_registry.json')['scorers']
    assert len(models)==15
    for row in registry:
        assert sha(HERE/row['bundle_state'])==row['bundle_state_sha256']
        assert recursive_content_hash(models[row['scorer_id']].state_dict())==row['tensor_content_sha256']
    passed('07_all_15_strict_load_and_file_tensor_hash_identity')
    params=HERE/'upstream/fitted_oneclass_parameters.npz';assert sha(params)==read('unchanged_scorer_bindings.json')['identical_v3_file_sha256']['upstream/fitted_oneclass_parameters.npz']
    parameter_hashes={}
    with np.load(params,allow_pickle=False) as z:
        for key in z.files:parameter_hashes[key]=hashlib.sha256(z[key].tobytes()).hexdigest()
    passed('08_all_fitted_one_class_parameters_identical_file_and_array_hashes')
    pool=ids('evidence/fresh_partition/fresh_source_pool.txt');assert sha(HERE/'evidence/fresh_partition/fresh_source_pool.txt')=='85bb93828421c53e54064bfc7de0eb355b5e016427749ada59ee4fffdec157da'
    passed('09_exact_original_500_EVAL_source_pool_SHA256')
    cal,evaluation,prior,excluded=map(ids,['final_conformal_cal.txt','final_evaluation.txt','prior_exposed_cal.txt','historical_exclusions.txt'])
    assert len(cal)==len(set(cal))==100 and len(evaluation)==len(set(evaluation))==400 and not set(cal)&set(evaluation) and set(cal)|set(evaluation)==set(pool)
    membership=read('evidence/fresh_partition/fresh_partition_membership.json');sorted_pool=sorted(pool);ix=membership['permutation_indices']
    assert len(ix)==500 and set(ix)==set(range(500)) and cal==[sorted_pool[i] for i in ix[:100]] and evaluation==[sorted_pool[i] for i in ix[100:]]
    passed('10_fresh_100_400_disjoint_exact_union_and_stored_bijection_no_RNG_replay')
    assert len(prior)==125 and not set(prior)&set(pool) and sha(HERE/'prior_exposed_cal.txt')=='652020b48796d8035d5b78aac9b3d1f869d9ebddcefa3963b2bea02f092bb858'
    passed('11_original_CAL125_exact_and_zero_overlap_permanently_retired')
    assert len(excluded)==2 and not set(excluded)&(set(pool)|set(prior))
    passed('12_historical_exclusions2_zero_overlap')
    assert len(set(cal)|set(evaluation)|set(prior)|set(excluded))==627
    passed('13_exact_four_role_total627')
    c,e,p,x='test/normal/Town01/scenario-1','test/anomaly/Town02/change-weather/scenario-2','test/normal/Town03/scenario-3','test/anomaly/Town01/change-weather/scenario-10'
    b=io.BytesIO();Image.fromarray(all_ids).save(b,format='PNG');png=b.getvalue()
    def forbidden(*args,**kwargs):raise AssertionError('Protected scientific path reached')
    import calibration_adapter
    for role,sid,role_key in [('prior',p,'PRIOR_EXPOSED_CAL_OPAQUE_DISCARD'),('eval',e,'FINAL_EVAL_OPAQUE_DISCARD'),('excluded',x,'HISTORICAL_EXCLUDED_OPAQUE_DISCARD')]:
        protected={role:{sid}};calls=[];ledger={}
        data=fake_archive([(c+'/unneeded.bin',b'f'),(sid+'/segmentation-front/000001.png',png),(sid+'/imu.feather',b'INVALID'),(sid+'/anomaly-observation.feather',b'INVALID')])
        with ExitStack() as stack:
            for obj,key in [(Image,'open'),(feather,'read_table'),(features,'segmentation_histogram_features'),(features,'camera_embedding_features'),(features,'imu_window_features'),(calibration_adapter,'labels_from_table')]:stack.enter_context(patch.object(obj,key,forbidden))
            for agent in agents.values():stack.enter_context(patch.object(agent,'estimate_uncertainty',forbidden))
            for model in models.values():stack.enter_context(patch.object(model,'forward',forbidden))
            scan(io.BytesIO(data),{c},protected.get('eval',set()),protected.get('excluded',set()),lambda *a:calls.append(a),ledger,len(data),prior_exposed_cal=protected.get('prior',set()))
        assert calls==[(c,None,None)] and ledger['scenario_roots_seen_by_role'][role_key]==1 and ledger['eval_scenarios_decoded']==ledger['prior_exposed_cal_scenarios_decoded']==ledger['historical_excluded_scenarios_decoded']==0
        assert ledger['opaque_body_bytes_actually_discarded_by_role'][role_key]==len(png)+14
        passed({'prior':'14_PRIOR_EXPOSED_CAL_all_decoder_UQ_model_spies_unreachable','eval':'15_FINAL_EVAL_all_decoder_UQ_model_spies_unreachable','excluded':'16_HISTORICAL_EXCLUDED_all_decoder_UQ_model_spies_unreachable'}[role])
    calls=[];data=fake_archive([(c+'/imu.feather',b'FAKE_FRESH'),(e+'/imu.feather',b'E'),(p+'/imu.feather',b'P'),(x+'/imu.feather',b'X')])
    scan(io.BytesIO(data),{c},{e},{x},lambda *a:calls.append(a),{},len(data),prior_exposed_cal={p});assert calls==[(c,'imu.feather',b'FAKE_FRESH'),(c,None,None)]
    passed('17_only_FRESH_CAL_reaches_scientific_callback')
    # Full synthetic catalogue: all627 roots, no real contents or source ID semantics.
    synthetic_ids=[f'test/normal/Town01/scenario-{i}' for i in range(1,628)]
    sc,se,sp,sx=map(set,[synthetic_ids[:100],synthetic_ids[100:500],synthetic_ids[500:625],synthetic_ids[625:]])
    data=fake_archive([(sid+'/unneeded.bin',b'SYNTHETIC') for sid in synthetic_ids]);ledger={};calls=[]
    receipt=scan(io.BytesIO(data),sc,se,sx,lambda *a:calls.append(a),ledger,len(data),prior_exposed_cal=sp)
    assert receipt['tar_complete'] and ledger['scenario_roots_seen']==627 and ledger['scenario_roots_seen_by_role']=={'FRESH_CAL':100,'FINAL_EVAL_OPAQUE_DISCARD':400,'PRIOR_EXPOSED_CAL_OPAQUE_DISCARD':125,'HISTORICAL_EXCLUDED_OPAQUE_DISCARD':2}
    assert {a[0] for a in calls}==sc and all(a[1:] == (None,None) for a in calls)
    passed('18_full_synthetic_four_role_627_archive_completes_exact_union')
    assert threshold([i/100 for i in range(100)])['rank_1_based']==96 and threshold([i/100 for i in range(100)])['Q']==.95
    passed('19_conformal100_mechanical_rank96')
    assert threshold([])['quantile_is_infinite'] and threshold([.1])['quantile_is_infinite'] and threshold([])['Q'] is None
    passed('20_augmented_infinity_behavior_preserved')
    q=threshold([.4]*100);assert q['Q']==.4 and q['comparison']=='<=' and .4<=q['Q'] and not .40001<=q['Q']
    assert '<= q for sid in cal' in active
    passed('21_inclusive_less_equal_tie_behavior')
    require_fresh_cal_completion(dict.fromkeys(cal),set(cal))
    for completed,members in [(dict.fromkeys(cal[:-1]),set(cal)),(dict.fromkeys(cal[:99]+['foreign']),set(cal)),(dict.fromkeys(cal+['extra']),set(cal)),(dict.fromkeys(prior),set(prior))]:rejects(lambda completed=completed,members=members:require_fresh_cal_completion(completed,members),'FRESH_CAL_INCOMPLETE_NO_THRESHOLDS_FINALIZED')
    segment=source_function(active,'execute');assert segment.index('ARCHIVE_INTEGRITY_REFUSAL_NO_THRESHOLDS_FINALIZED')<segment.index('require_fresh_cal_completion(processor.completed, cal)')<segment.index('threshold([r')
    passed('22_finalization_refuses_not_exact100_or_membership_after_archive_integrity')
    assert 'cal_scenario_processing_ledger.jsonl' not in segment and 'attempt002' not in segment and 'per_scorer_scenario_scores.json' in segment
    assert 'read_text' not in source_function(active,'map_segmentation_oov_to_frozen_vocabulary')
    passed('23_active_runner_no_historical_partial_score_import_or_read')
    with tempfile.TemporaryDirectory(prefix='cognix_v4_synthetic_') as tmp:
        processor=CalProcessor({}, {}, set(cal),Path(tmp),'cpu')
        with patch.object(Image,'open',forbidden):
            for sid in (prior[0],evaluation[0],excluded[0]):rejects(lambda sid=sid:processor(sid,'segmentation-front/000001.png',b'UNDECODED'),'SCIENTIFIC_CALLBACK_REJECTS_NON_FRESH_CAL')
        # Real code-path scoring on synthetic4-role bytes only, never execute().
        entries=[];n=3
        for name,table in [('imu.feather',pa.table({'acceleration_x':[0.,.1,.2],'acceleration_y':[0.]*n,'acceleration_z':[9.8]*n})),('anomaly-observation.feather',pa.table({'tick':list(range(n)),'anomaly':[False,True,False]}))]:
            b=io.BytesIO();feather.write_feather(table,b);entries.append((c+'/'+name,b.getvalue()))
        for t in range(n):
            b=io.BytesIO();Image.fromarray(np.full((16,16,3),120,np.uint8)).save(b,format='JPEG');entries.extend([(f'{c}/rgb-front/{t:06d}.jpg',b.getvalue()),(f'{c}/segmentation-front/{t:06d}.png',png)])
        entries.extend([(e+'/imu.feather',b'PROTECTED'),(p+'/segmentation-front/000001.png',b'PROTECTED'),(x+'/anomaly-observation.feather',b'PROTECTED')])
        data=fake_archive(entries);processor=CalProcessor(models,agents,{c},Path(tmp),'cpu');ledger={}
        scan(io.BytesIO(data),{c},{e},{x},processor,ledger,len(data),prior_exposed_cal={p})
        assert set(processor.completed)=={c} and len(processor.completed[c])==15 and all(a.shape==(2,) and np.isfinite(a).all() for a in processor.completed[c].values())
    passed('24_old_actual_CAL_IDs_rejected_before_decoder_and_synthetic_four_role_15_scorer_integration')
    nb=read('cognix_final_conformal_cal_v4.ipynb');assert all(cell.get('execution_count') is None and not cell.get('outputs') for cell in nb['cells'])
    assert 'EXECUTE_SEPARATELY_AUTHORIZED_FINAL_CAL = False' in ''.join(''.join(cell['source']) for cell in nb['cells'])
    passed('25_notebook_execution_default_false_new_bundle_output_paths')
    assert segment.count('opener.open(')==1 and 'method="GET"' in segment and 'Accept-Encoding' in segment and 'identity' in segment and 'Range' not in segment
    assert 'redirect_request' in segment and 'REDIRECT_FORBIDDEN_SINGLE_REQUEST' in segment and 'automatic_retry": False' in segment and 'range_requests": 0' in segment
    assert segment.index('attempt_started.json')<segment.index('opener.open(') and segment.index('if OUTPUT.exists()')<segment.index('OUTPUT.mkdir')
    assert '/kaggle/working/cognix_final_conformal_cal_attempt003_v2' in active
    assert active.count('parser.add_argument(')==2 and '--execute-authorized-final-cal' in active
    passed('26_single_GET_identity_no_redirect_Range_retry_resume_marker_output_refusal_preserved')
    assert source_function(active,'runtime_check')==source_function(precursor,'runtime_check') and 'pandas.__version__ != "2.2.3"' in active
    assert sha(HERE/'runtime_lock.json')==read('unchanged_scorer_bindings.json')['identical_v3_file_sha256']['runtime_lock.json']
    passed('27_runtime_pins_and_enforcement_function_exact_v3_identity')
    assert 'torch.cuda.device_count() != 1' in active and 'torch.cuda.get_device_name(0) != "Tesla T4"' in active
    passed('28_exactly_one_Tesla_T4_enforced')
    guard_source=(HERE/'offline_checks.py').read_text();assert 'sys.addaudithook(deny_network)' in guard_source and 'socket.connect' in guard_source and 'subprocess.Popen' in guard_source and 'OFFLINE_CHECK_ONLY_NO_SCIENTIFIC_EXECUTION' in guard_source
    passed('29_offline_guard_network_child_and_scientific_execution_refusal')
    assert read('evidence/fresh_partition/fresh_partition_execution_receipt.json')['TEST_network_requests']==0
    passed('30_preparation_TEST_network_requests_zero')
    assert read('evidence/fresh_partition/fresh_partition_execution_receipt.json')['TEST_payload_bytes_accessed']==0
    passed('31_preparation_TEST_payload_bytes_accessed_zero')
    assert read('fresh_split_amendment_record.json')['Attempt003_executed'] is False
    passed('32_Attempt003_unexecuted_protocol_v4_generation_v2')
    rejects(lambda:scan(io.BytesIO(data),{c},{e},{x},lambda *a:None,{},len(data),prior_exposed_cal={c}),'INVALID_MEMBERSHIP')
    incomplete=fake_archive([(c+'/unneeded.bin',b'X')]);rejects(lambda:scan(io.BytesIO(incomplete),{c},{e},{x},lambda *a:None,{},len(incomplete),prior_exposed_cal={p}),'ARCHIVE_SCENARIO_SET_MISMATCH')
    passed('33_overlap_and_missing_provenance_role_fail_closed')
    seed=read('evidence/fresh_partition/fresh_partition_seed.json');receipt=read('evidence/fresh_partition/fresh_partition_execution_receipt.json');prereg=read('evidence/fresh_partition/fresh_partition_preregistration.json')
    assert seed['seed_spent'] and seed['seed_generated_once'] and not seed['redraw_allowed'] and not seed['seed_search_allowed']
    assert receipt['permutation_count']==1 and prereg['permutations_at_preregistration']==0 and prereg['seed_generated_at_preregistration'] is False
    script=(HERE/'evidence/fresh_partition/fresh_partition.py').read_text();calls=[ast.unparse(n.func) for n in ast.walk(ast.parse(script)) if isinstance(n,ast.Call)]
    assert calls.count('secrets.randbits')==1 and calls.count('rng.permutation')==1
    passed('34_single_seed_single_permutation_chronology_and_no_redraw_guard')
    return {'status':'PASSED','fresh_split_groups_passed':len(results),'groups_failed':0,'groups':results,'fitted_parameter_array_sha256':parameter_hashes,
      'TEST_network_requests':0,'TEST_payload_bytes_accessed':0,'Attempt003_executed':False,'scientific_execute_function_called':False,'fixtures':'synthetic bytes only; preserved scenario ID metadata only'}
if __name__=='__main__':print(json.dumps(run(),sort_keys=True))
