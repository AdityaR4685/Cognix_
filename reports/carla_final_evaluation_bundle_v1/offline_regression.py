"""Synthetic/offline regressions. Runtime/HTTP modules use in-memory mocks."""
import ast
import contextlib
import copy
import gzip
import hashlib
import importlib.util
import io
import itertools
import json
import math
from pathlib import Path
import statistics
import sys
import tarfile
import tempfile
import types
from unittest.mock import patch
import numpy as np

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
from metrics import classification, prediction_sets, conformal, conformal_summary, strict_macro, paired_summary, aggregate, SEEDS, METRICS
from output_store import read, sha, require_completion, record_scenario, reconstruct, seal_results
from verify_bundle import verify, function
from streaming import scan
from scientific_processor import EvalProcessor
from scientific_runtime import runtime_check, map_segmentation_oov_to_frozen_vocabulary
from label_adapter import labels_from_table

class SyntheticTable:
    """Minimal table protocol for label checks; fabricated arrays only."""
    def __init__(self,columns):self.columns={k:np.asarray(v) for k,v in columns.items()}
    def __getitem__(self,key):return types.SimpleNamespace(to_numpy=lambda:self.columns[key])

def check(value):
    if not value:raise AssertionError('Regression expectation failed')
def close(a,b):check(math.isclose(a,b,rel_tol=1e-13,abs_tol=1e-13))
def rejects(fn,code=None):
    try:fn()
    except Exception as exc:
        if code:check(code in str(exc))
        return
    raise AssertionError('Expected fail-closed refusal')
def fake_archive(entries):
    raw=io.BytesIO()
    with tarfile.open(fileobj=raw,mode='w',format=tarfile.USTAR_FORMAT) as tf:
        for name,data in entries:
            member=tarfile.TarInfo(name);member.size=len(data);tf.addfile(member,io.BytesIO(data))
    return gzip.compress(raw.getvalue(),mtime=0)

def runtime_fake(**overrides):
    pins={'Python':'3.12.13','NumPy':'2.0.2','PyTorch':'2.10.0+cu128','Pillow':'12.3.0','PyArrow':'25.0.1','pandas':'2.2.3','CUDA':'12.8','cuDNN':91002,'count':1,'device':'Tesla T4','available':True,**overrides}
    calls=[];ns=types.SimpleNamespace
    torch=ns(__version__=pins['PyTorch'],cuda=ns(is_available=lambda:pins['available'],device_count=lambda:pins['count'],get_device_name=lambda i:pins['device']),
      version=ns(cuda=pins['CUDA']),backends=ns(cudnn=ns(version=lambda:pins['cuDNN'],deterministic=False,benchmark=True,allow_tf32=True),cuda=ns(matmul=ns(allow_tf32=True))),
      set_num_threads=lambda n:calls.append(('threads',n)),use_deterministic_algorithms=lambda x:calls.append(('deterministic',x)))
    modules={'torch':torch,'numpy':ns(__version__=pins['NumPy']),'PIL':ns(__version__=pins['Pillow']),'pyarrow':ns(__version__=pins['PyArrow']),'pandas':ns(__version__=pins['pandas'])}
    with patch.dict(sys.modules,modules),patch('platform.python_version',return_value=pins['Python']):actual=runtime_check()
    check(torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark and not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32)
    check(calls==[('threads',1),('deterministic',True)])
    return actual

def run():
    results=[]
    def group(n,fn):
        try:fn()
        except Exception as exc:
            results.append({'group':n,'status':'FAILED','exception':type(exc).__name__,'message':str(exc)})
        else:results.append({'group':n,'status':'PASSED'})
    completion=HERE.parent/'carla_final_conformal_cal_attempt003_complete_v1'
    evidence=completion/'evidence/cognix_final_conformal_cal_attempt003_v2'
    ingest_spec=importlib.util.spec_from_file_location('ingestion',completion/'ingest_attempt003.py')
    ingest=importlib.util.module_from_spec(ingest_spec);ingest_spec.loader.exec_module(ingest)
    group('01_Attempt003_external_and_copied_archive_SHA',lambda:check(sha(ingest.ARCHIVE)==sha(completion/ingest.ARCHIVE.name)==ingest.ARCHIVE_SHA))
    group('02_RESULT_detached_integrity',lambda:check((evidence/'RESULT_SHA256SUMS.sha256').read_text().split()==[sha(evidence/'RESULT_SHA256SUMS'),'RESULT_SHA256SUMS']))
    group('03_all_Attempt003_result_bytes_and_complete_seal',lambda:(ingest.seal_check(evidence,'RESULT_SHA256SUMS',exact=True),ingest.seal_check(completion,'ATTEMPT003_COMPLETE_SHA256SUMS',exact=True)))
    group('04_exact_100_CAL_and_15_receipts',lambda:check(ingest.validate(evidence)['unique_completed_fresh_CAL']==100))
    binding=read(HERE/'protocol_bindings.json');registry=read(HERE/'model_registry.json')
    group('05_active_threshold_hash_frozen_no_score_dependency',lambda:(check(sha(HERE/'conformal_thresholds.json')==binding['threshold_sha256']),check(not (HERE/'per_scorer_scenario_scores.json').exists())))
    evals=set((HERE/'final_evaluation.txt').read_text().splitlines());cal=set((HERE/'final_conformal_cal.txt').read_text().splitlines());prior=set((HERE/'prior_exposed_cal.txt').read_text().splitlines());excluded=set((HERE/'historical_exclusions.txt').read_text().splitlines())
    group('06_v4_400_EVAL_exact_hash',lambda:check(len(evals)==400 and sha(HERE/'final_evaluation.txt')=='9410f13cc563362bf5ef7ed52333807131bea4fe5a5d51a13c1924e4c855a60a'))
    group('07_four_fixed_role_counts',lambda:check([len(x) for x in (evals,cal,prior,excluded)]==[400,100,125,2]))
    group('08_zero_overlap_exact_627_inventory_union',lambda:(check(not any(a&b for a,b in itertools.combinations((evals,cal,prior,excluded),2))),check(set.union(evals,cal,prior,excluded)==set((HERE/'evidence/inventory/scenario_inventory_sorted.txt').read_text().splitlines()))))
    # Completely synthetic scenarios/bodies, never reads real TEST fixtures.
    e='test/normal/Town01/scenario-901';c='test/normal/Town02/scenario-902';p='test/normal/Town03/scenario-903';x='test/anomaly/Town01/change-weather/scenario-904'
    def roles_test():
        calls=[];ledger={};entries=[(sid+'/imu.feather',body) for sid,body in [(e,b'FAKE_EVAL'),(c,b'BAD_PROTECTED'),(p,b'BAD_PROTECTED'),(x,b'BAD_PROTECTED')]]
        data=fake_archive(entries);receipt=scan(io.BytesIO(data),{e},{c},{x},lambda *args:calls.append(args),ledger,len(data),prior_exposed_cal={p})
        check(calls==[(e,'imu.feather',b'FAKE_EVAL'),(e,None,None)]);check(receipt['sha256']==hashlib.sha256(data).hexdigest())
        check(ledger['scenario_roots_seen_by_role']=={'FINAL_EVAL':1,'FRESH_CAL_OPAQUE_DISCARD':1,'PRIOR_EXPOSED_CAL_OPAQUE_DISCARD':1,'HISTORICAL_EXCLUDED_OPAQUE_DISCARD':1})
    group('09_only_EVAL_reaches_scanner_callback',roles_test)
    group('10_all_opaque_roles_processor_refused_before_import',lambda:[rejects(lambda sid=sid:EvalProcessor({}, {},{e},Path('.'),'cpu')(sid,'imu.feather',b'INVALID'),'NON_FINAL_EVAL') for sid in (c,p,x)])
    group('11_v4_scorer_and_OOV_source_identical',lambda:(verify(False),check(function((HERE/'scientific_runtime.py').read_text(),'map_segmentation_oov_to_frozen_vocabulary')==function((HERE/'provenance/v4_runner.py.txt').read_text(),'map_segmentation_oov_to_frozen_vocabulary'))))
    group('12_all_15_model_hashes_identical',lambda:check(len(registry['scorers'])==15 and all(sha(HERE/r['bundle_state'])==r['bundle_state_sha256']==binding['identical_v4_files'][r['bundle_state']] for r in registry['scorers'])))
    group('13_fitted_parameter_and_manifest_bytes_identical',lambda:check(all(sha(HERE/n)==binding['identical_v4_files'][n] for n in ['upstream/fitted_oneclass_parameters.npz','upstream/manifest.json'])))
    def label_test():
        check(labels_from_table(SyntheticTable({'tick':[0,1,2],'anomaly':[False,True,False]})).tolist()==[0,1,0])
        for df in [SyntheticTable({'tick':[0,2],'anomaly':[0,1]}),SyntheticTable({'tick':[0,1],'anomaly':[0.,1.]}),SyntheticTable({'tick':[0,1],'anomaly':[0,2]})]:rejects(lambda df=df:labels_from_table(df))
        check(read(HERE/'label_semantics.json')['official_class_order']==['normal=0','anomaly=1'])
    group('14_official_label_class_order_and_alignment',label_test)
    normal=classification([.9,.7],[0,0],.5)
    group('15_normal_only_undefined_convention',lambda:check(normal['values']['AUROC'] is None and normal['values']['AUPRC'] is None and normal['values']['balanced_accuracy'] is None and normal['values']['F1']==0))
    def ranking():
        row=classification([.9,.6,.6,.1],[0,1,0,1],.5)
        close(row['values']['AUROC'],.875);close(row['values']['AUPRC'],5/6)
        row=classification([.5,.5],[0,1],.5);close(row['values']['AUROC'],.5);close(row['values']['AUPRC'],.5)
        close(classification([.8,.2],[0,1],.5)['values']['AUROC'],1)
    group('16_two_class_AUROC_and_tied_stepwise_AP',ranking)
    group('17_hard_threshold_inclusive_greater_equal',lambda:check(classification([.5,.75],[1,0],.5)['confusion']=={'TP':1,'FP':0,'FN':0,'TN':1}))
    group('18_exact_BCE_clipping',lambda:(close(classification([0.,1.],[0,1],.5)['values']['BCE'],-math.log(1e-7)),close(classification([1.,0.],[0,1],.5)['values']['BCE'],-math.log(1-1e-7))))
    def ece():
        q=np.arange(16)/15;y=np.zeros(16,dtype=np.int64);row=classification(q,y,.5)
        check([b['n'] for b in row['ECE_bins']]==[1]*14+[2]);close(row['values']['ECE'],.5)
        close(classification([1.],[0],.5)['values']['ECE'],0)
    group('19_ECE_15_edges_empty_bins_final_endpoint',ece)
    group('20_strict_macro_null_and_exact_reasons',lambda:check(strict_macro({'normal':normal,'both':classification([.9,.1],[0,1],.5)})['AUROC']=={'value':None,'assigned_scenarios':2,'defined_scenarios':1,'undefined_scenario_ids_and_reasons':{'normal':'requires both official tick classes'}}))
    def secondary():
        rows={};arrays={}
        for sid,q,y in [('normal',[.9,.8],[0,0]),('both',[.9,.1],[0,1])]:
            rows[sid]={'classification':{},'conformal':{},'metadata':{'town':'Town01','directory_condition':'normal','anomaly_type':'NORMAL'}}
            arrays[sid]={'labels':np.array(y,dtype=np.int64),'q':{}}
            for r in registry['scorers']:
                s=r['scorer_id'];rows[sid]['classification'][s]=classification(q,y,r['frozen_anomaly_threshold']);rows[sid]['conformal'][s]=conformal(q,y,.8);arrays[sid]['q'][s]=np.array(q)
        out=aggregate(rows,arrays,registry);check(out['secondary_metrics.json']['two_class_eligible_scenario_ids']==['both'])
        check(all(v==1 for v in out['secondary_metrics.json']['two_class_eligible_macro_AUROC'].values()))
        check(out['paired_seed_comparison.json']['epistemic_gat_minus_standard_gat']['mean'] is None)
    group('21_secondary_two_class_common_subset_membership',secondary)
    group('22_conformal_inclusive_less_equal_ties',lambda:check(prediction_sets([.5],.5).tolist()==[[True,True]]))
    group('23_conformal_empty_singleton_two_label_sets',lambda:(check(prediction_sets([.5,.9,.1],.4).tolist()==[[False,False],[True,False],[False,True]]),check(prediction_sets([.5],.6).tolist()==[[True,True]])))
    group('24_simultaneous_indicator_matches_block_max',lambda:(check(conformal([.9,.1],[0,1],.11)['simultaneous_supported']),check(not conformal([.9,.8],[0,1],.11)['simultaneous_supported'])))
    def denominators():
        out=conformal_summary({'a':conformal([.9],[0],.2),'b':conformal([.9,.9,.9],[1,1,1],.2)})
        close(out['scenario_macro_tick_coverage'],.5);close(out['pooled_tick_coverage'],.25);close(out['scenario_simultaneous_coverage'],.5)
        close(out['signed_coverage_gap']['pooled_tick_coverage'],-.7);close(out['undercoverage_gap']['pooled_tick_coverage'],.7)
    group('25_equal_scenario_vs_pooled_denominators',denominators)
    def seeds():
        delta=[.1,.2,.3,.4,.5];a=dict(zip(SEEDS,delta));b={s:0. for s in SEEDS};out=paired_summary(a,b)
        close(out['mean'],.3);close(out['median'],.3);close(out['sample_SD'],statistics.stdev(delta));close(out['exact_two_sided_sign_flip_p'],.0625)
        close(out['conditional_seed_CI'][1],.3+2.776445105*statistics.stdev(delta)/math.sqrt(5))
        check(paired_summary({s:None for s in SEEDS},b)['mean'] is None)
    group('26_five_pairs_SD_CI_exact_32_sign_patterns',seeds)
    group('27_no_best_seed_selection',lambda:check(binding['best_seed_selection'] is False and read(HERE/'seed_aggregation_spec.json')['no_best_seed']))
    group('28_no_prediction_ensemble',lambda:check(binding['prediction_ensemble'] is False and read(HERE/'seed_aggregation_spec.json')['ensemble_used'] is False))
    def no_refit():
        check(binding['threshold_refit'] is False)
        for name in ['runner.py','scientific_processor.py','metrics.py','output_store.py']:
            tree=ast.parse((HERE/name).read_text())
            check(not any(isinstance(n,ast.Call) and ((isinstance(n.func,ast.Attribute) and n.func.attr in {'fit','fit_calibrator','manual_seed','seed','permutation'}) or (isinstance(n.func,ast.Name) and n.func.id=='threshold')) for n in ast.walk(tree)))
    group('29_no_refit_search_or_redraw_active_calls',no_refit)
    good_receipt={'compressed_bytes':91538225599,'sha256':ingest.OFFICIAL_SHA,'gzip_crc_validated':True,'tar_complete':True}
    good_ledger={'scenario_roots_seen':627,'scenario_roots_seen_by_role':binding['role_counts'],'fresh_cal_scenarios_decoded':0,'prior_exposed_cal_scenarios_decoded':0,'historical_excluded_scenarios_decoded':0,
      'http_requests':1,'range_requests':0,'automatic_retry':False,'raw_archive_retained':False,'compressed_bytes_received':91538225599,'compressed_sha256_partial_or_complete':ingest.OFFICIAL_SHA,
      'body_bytes_accounted_by_role':{'FINAL_EVAL':62847018051,'FRESH_CAL_OPAQUE_DISCARD':16288609930,'HISTORICAL_EXCLUDED_OPAQUE_DISCARD':240747299,'PRIOR_EXPOSED_CAL_OPAQUE_DISCARD':19633431385},
      'opaque_body_bytes_actually_discarded_by_role':{'FINAL_EVAL':1,'FRESH_CAL_OPAQUE_DISCARD':16288609930,'HISTORICAL_EXCLUDED_OPAQUE_DISCARD':240747299,'PRIOR_EXPOSED_CAL_OPAQUE_DISCARD':19633431385}}
    group('30_finalization_refuses_399_accepts_exact_400_only',lambda:(require_completion(dict.fromkeys(evals),evals,good_receipt,good_ledger),rejects(lambda:require_completion(dict.fromkeys(sorted(evals)[:-1]),evals,good_receipt,good_ledger),'INCOMPLETE')))
    def missing():
        processor=EvalProcessor({}, {},{e},Path('.'),'cpu');processor.current=e;processor.data={'tables':{},'camera':{},'seg':{}}
        # Import stubs only, no decoder/model is invoked before missing-table refusal.
        ns=types.ModuleType('cognix.adapters.carla.real_features');ns.imu_window_features=lambda _:None
        with patch.dict(sys.modules,{'torch':types.SimpleNamespace(),'cognix.adapters.carla.real_features':ns}):rejects(lambda:processor.finish(e),'MISSING_SCENARIO_TABLES')
        rejects(lambda:classification([float('nan')],[0],.5),'NONFINITE')
        data=fake_archive([(e+'/imu.feather',b'x')]);rejects(lambda:scan(io.BytesIO(data),{e},{c},set(),lambda *a:None,{},len(data)),'SCENARIO_SET_MISMATCH')
    group('31_missing_scenario_modalities_nonfinite_fail',missing)
    def request_mock():
        import transport
        import urllib
        calls=[];handlers=[]
        class Request:
            def __init__(self,url,headers,method):self.headers=headers;self.method=method;check(url==transport.URL)
            def get_method(self):return self.method
            def has_header(self,name):return name in self.headers
            def get_header(self,name):return {k.lower():v for k,v in self.headers.items()}.get(name.lower())
        class Response:
            status=200;headers={'Content-Length':'91538225599','Content-Encoding':'identity'}
            def close(self):pass
        class Opener:
            def open(self,request,timeout):calls.append((request,timeout));return Response()
        fake=types.ModuleType('urllib.request');fake.HTTPRedirectHandler=type('Handler',(),{});fake.Request=Request
        def build(handler):handlers.append(handler);return Opener()
        fake.build_opener=build
        with patch.dict(sys.modules,{'urllib.request':fake}),patch.object(urllib,'request',fake,create=True):
            ledger={'http_requests':0};transport.open_one_shot(ledger)
            rejects(lambda:transport.open_one_shot(ledger),'SECOND_REQUEST');rejects(lambda:handlers[0].redirect_request(),'REDIRECT_FORBIDDEN')
        check(len(calls)==1 and calls[0][0].headers=={'Accept-Encoding':'identity'})
    group('32_single_GET_redirect_Range_retry_guards_mock_only',request_mock)
    def pins():
        runtime_fake()
        for name in ['Python','NumPy','PyTorch','Pillow','PyArrow','pandas','CUDA','cuDNN']:
            rejects(lambda name=name:runtime_fake(**{name:'wrong'}))
        check(function((HERE/'scientific_runtime.py').read_text(),'runtime_check')==function((HERE/'provenance/v4_runner.py.txt').read_text(),'runtime_check'))
    group('33_exact_v4_runtime_pins_and_determinism_mock_tests',pins)
    group('34_exactly_one_visible_Tesla_T4_mock_refusals',lambda:[rejects(lambda kw=kw:runtime_fake(**kw),'REQUIRES_ONE_VISIBLE_T4') for kw in [{'count':0},{'count':2},{'device':'Tesla P100'},{'available':False}]])
    group('35_notebook_default_false_unexecuted_separate_cells',lambda:check('EXECUTE_SEPARATELY_AUTHORIZED_FINAL_EVALUATION = False' in ''.join(read(HERE/'cognix_final_evaluation_v1.ipynb')['cells'][1]['source']) and all(not c.get('outputs') for c in read(HERE/'cognix_final_evaluation_v1.ipynb')['cells'])))
    group('36_preparation_TEST_requests_zero_network_guard',lambda:check(binding['preparation_TEST_requests']==0 and 'sys.addaudithook(guard)' in (HERE/'offline_checks.py').read_text()))
    group('37_preparation_TEST_bytes_zero_synthetic_fixtures',lambda:check(binding['preparation_TEST_payload_bytes']==0))
    group('38_final_evaluation_executed_false_namespace',lambda:check(binding['final_evaluation_executed'] is False and '/kaggle/working/cognix_final_evaluation_attempt001_v1' in (HERE/'runner.py').read_text()))
    def oov():
        a=np.arange(256,dtype=np.uint8).reshape(16,16);out=map_segmentation_oov_to_frozen_vocabulary(a)
        check(np.array_equal(out[a<=28],a[a<=28]) and (out[a>28]==22).all() and np.array_equal(a,np.arange(256,dtype=np.uint8).reshape(16,16)))
        a3=np.stack([a,np.zeros_like(a)],axis=-1);check(np.array_equal(map_segmentation_oov_to_frozen_vocabulary(a3),out))
        rejects(lambda:map_segmentation_oov_to_frozen_vocabulary(a.astype(np.int64)))
    group('39_OOV_all_uint8_IDs_channel0_unchanged_29D',oov)
    def intermediate():
        sid=sorted(evals)[0];q={r['scorer_id']:np.array([.5,.9,.1],dtype=np.float32) for r in registry['scorers']}
        with tempfile.TemporaryDirectory(prefix='cognix_synthetic_eval_') as tmp:
            output=Path(tmp);record=record_scenario(output,sid,np.array([0,0,1],dtype=np.int64),q)
            final=reconstruct(output,{sid:record});check(len(final)==7);seal_results(output)
            check((output/'RESULT_SHA256SUMS.sha256').read_text().split()==[sha(output/'RESULT_SHA256SUMS'),'RESULT_SHA256SUMS'])
            rejects(lambda:record_scenario(output,sid,np.array([0,0,1],dtype=np.int64),q),'DUPLICATE')
            altered=copy.deepcopy(record);altered['classification'][next(iter(q))]['values']['accuracy']=.123
            rejects(lambda:reconstruct(output,{sid:altered}),'RECONSTRUCTION_FAILED')
        rejects(lambda:record_scenario(Path('.'),c,np.array([0]),{}),'NON_FINAL_EVAL')
    group('40_numeric_intermediate_metric_reconstruction_and_tamper_refusal',intermediate)
    def archive_refusals():
        from header_helpers import classify
        for path in ['../test/normal/Town01/scenario-1/a','/test/normal/Town01/scenario-1/a','test/normal/Town01/scenario-1/../a']:
            rejects(lambda path=path:classify(path,b'0'))
        entries=[(e+'/imu.feather',b'a'),(c+'/imu.feather',b'b'),(e+'/rgb-front/000001.jpg',b'c')];data=fake_archive(entries)
        rejects(lambda:scan(io.BytesIO(data),{e},{c},set(),lambda *a:None,{},len(data)),'NONCONTIGUOUS')
        data=fake_archive([(e+'/imu.feather',b'a')]);rejects(lambda:scan(io.BytesIO(data[:-8]),{e},set(),set(),lambda *a:None,{},len(data)))
    group('41_archive_safety_corruption_and_noncontiguous_roles',archive_refusals)
    def atomic():
        source=(HERE/'runner.py').read_text()
        check(source.index("save(OUTPUT/'attempt_started.json'")<source.index('with open_one_shot(ledger)'))
        check(source.index('require_completion(processor.completed')<source.index('final=reconstruct'))
        check('INCOMPLETE_NO_RETRY' in source and 'exist_ok=False' in source)
        for k,v in [('http_requests',2),('range_requests',1),('fresh_cal_scenarios_decoded',1),('raw_archive_retained',True)]:
            rejects(lambda k=k,v=v:require_completion(dict.fromkeys(evals),evals,good_receipt,{**good_ledger,k:v}))
        rejects(lambda:require_completion(dict.fromkeys(evals),evals,{**good_receipt,'sha256':'wrong'},good_ledger))
    group('42_atomic_marker_archive_hash_protected_role_finalization_guards',atomic)
    def pipeline_identity():
        source=(HERE/'provenance/v4_runner.py.txt').read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='CalProcessor')
        expected=ast.get_source_segment(source,node).replace('CalProcessor','EvalProcessor').replace('self.cal','self.evaluation')
        expected=expected.replace('models, agents, cal, output, device','models, agents, evaluation, output, device').replace('frozenset(cal)','frozenset(evaluation)')
        expected=expected.replace('NON_FRESH_CAL','NON_FINAL_EVAL').replace('TOTAL_CAL_TICK_MEMORY_CAP_EXCEEDED','TOTAL_EVAL_TICK_MEMORY_CAP_EXCEEDED')
        expected=expected.replace('from calibration_adapter import labels_from_table, tick_scores','from label_adapter import labels_from_table')
        expected=expected.replace('all_scores[scorer] = tick_scores(np.concatenate(probs), labels[1:])','all_scores[scorer] = np.concatenate(probs)')
        expected=expected[:expected.index('        self.completed[sid] = all_scores')]
        actual=(HERE/'scientific_processor.py').read_text();actual=actual[actual.index('class EvalProcessor:'):actual.index('        from output_store import record_scenario')]
        check(actual==expected)
    group('43_entire_scientific_prefix_exact_v4_mechanical_reuse',pipeline_identity)
    def synthetic_pipeline():
        # Real frozen Camera/Seg/IMU feature functions on fabricated arrays.
        # Decoder and model interfaces are in-memory mocks; no official payload.
        from PIL import Image
        features_spec=importlib.util.spec_from_file_location('cognix.adapters.carla.real_features',HERE/'frozen_sources/cognix/adapters/carla/real_features.py')
        features=importlib.util.module_from_spec(features_spec);features_spec.loader.exec_module(features)
        windows=[];original_window=features.imu_window_features
        def window(a):windows.append(a.copy());return original_window(a)
        features.imu_window_features=window
        n=14;imu=np.arange(n*3,dtype=np.float64).reshape(n,3)
        class IMU:
            def to_numpy(self,dtype):return imu.astype(dtype)
        class FeatherTable:
            column_names=[]
            def __init__(self,label=False):self.label=label
            def to_pandas(self):return SyntheticTable({'tick':np.arange(n),'anomaly':np.arange(n)%2}) if self.label else IMU()
        feather=types.ModuleType('pyarrow.feather')
        feather.read_table=lambda body,columns=None:FeatherTable(body.read()==b'SYNTHETIC_LABEL_TABLE')
        pa=types.ModuleType('pyarrow');pa.feather=feather
        batch_calls=[]
        class Model:
            def __call__(self,batch,epistemic):
                check(batch.dtype==np.float32 and epistemic.dtype==np.float64);check(batch.shape[1:]==(3,3));batch_calls.append(len(batch))
                return types.SimpleNamespace(cpu=lambda:types.SimpleNamespace(numpy=lambda:np.full(len(batch),.5,dtype=np.float32)))
        fake_torch=types.SimpleNamespace(inference_mode=contextlib.nullcontext,float32=np.float32,float64=np.float64,tensor=lambda a,dtype,device:np.asarray(a,dtype=dtype))
        calls=[]
        class Agent:
            def __init__(self,name):self.name=name
            def estimate_uncertainty(self,feature):calls.append(self.name);return types.SimpleNamespace(prediction=.5,epistemic=.01,aleatoric=.2)
        sid=sorted(evals)[0];entries=[(sid+'/imu.feather',b'SYNTHETIC_IMU_TABLE'),(sid+'/anomaly-observation.feather',b'SYNTHETIC_LABEL_TABLE')]
        for t in range(n):
            for folder,ext,fmt,array in [('rgb-front','jpg','JPEG',np.full((16,16,3),120,dtype=np.uint8)),('segmentation-front','png','PNG',np.full((16,16),255,dtype=np.uint8))]:
                image=io.BytesIO();Image.fromarray(array).save(image,format=fmt);entries.append((f'{sid}/{folder}/{t:06d}.{ext}',image.getvalue()))
        data=fake_archive(entries)
        with tempfile.TemporaryDirectory(prefix='cognix_synthetic_pipeline_') as tmp,patch.dict(sys.modules,{'torch':fake_torch,'pyarrow':pa,'pyarrow.feather':feather,'cognix.adapters.carla.real_features':features}):
            processor=EvalProcessor({r['scorer_id']:Model() for r in registry['scorers']},{name:Agent(name) for name in ('Camera','IMU','Seg')},{sid},Path(tmp),'mock')
            with contextlib.redirect_stdout(io.StringIO()):scan(io.BytesIO(data),{sid},set(),set(),processor,{},len(data))
            check(processor.total_ticks==n-1 and set(processor.completed)=={sid} and processor.current is None and processor.data=={})
            check(len(batch_calls)==15 and calls==['Camera','IMU','Seg']*(n-1))
            check(len(windows)==n-1 and all(np.array_equal(a,imu[max(0,t-11):t+1]) for t,a in enumerate(windows,1)))
            reconstruct(Path(tmp),processor.completed)
    group('44_synthetic_feature_pipeline_tick0_causal12_order_15_outputs',synthetic_pipeline)
    out={'groups_passed':sum(r['status']=='PASSED' for r in results),'groups_failed':sum(r['status']=='FAILED' for r in results),'groups':results,
      'actual_network_calls':0,'real_TEST_bytes_used':0,'final_evaluation_executed':False,
      'runtime_validation_scope':'Exact source binding plus mocked pin/GPU/determinism refusal tests. No local CUDA execution or new model inference. Future host must pass actual runtime_check.',
      'checkpoint_validation_scope':'All 15 exact sealed state-file hashes; upstream parameter and source bytes. Existing v4 load/forward evidence remains sealed; no calibration/final execution.'}
    return out

if __name__=='__main__':
    result=run();print(json.dumps(result,sort_keys=True))
    if result['groups_failed']:raise SystemExit(1)
