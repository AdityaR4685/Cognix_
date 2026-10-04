"""Synthetic OOV amendment and invariant checks. No real TEST payload/scoring."""
import ast
from contextlib import ExitStack
import hashlib
import importlib.util
from importlib.machinery import SourceFileLoader
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def run():
    import numpy as np
    import pyarrow as pa
    import pyarrow.feather as feather
    from PIL import Image
    from frozen_runtime import load, recursive_content_hash
    models, agents = load()
    from cognix.adapters.carla import real_features as features
    from runner import CalProcessor, map_segmentation_oov_to_frozen_vocabulary as mapping
    from streaming import scan
    from synthetic_verification import fake_archive
    results=[]
    def passed(name): results.append({'group':name,'status':'PASSED'})
    def reject(fn, match):
        try: fn()
        except (ValueError,RuntimeError) as exc: assert str(exc)==match; return
        raise AssertionError('Expected fail-loud refusal: '+match)
    def png(a):
        b=io.BytesIO(); Image.fromarray(a).save(b,format='PNG'); return b.getvalue()
    def hist(a): return features.segmentation_histogram_features(mapping(a))
    c,e,x='test/normal/Town01/scenario-1','test/anomaly/Town02/change-weather/scenario-2','test/anomaly/Town01/change-weather/scenario-10'
    with tempfile.TemporaryDirectory(prefix='cognix_v3_synthetic_') as tmp:
        out=Path(tmp)
        loader=SourceFileLoader('immutable_v2_runner',str(HERE/'evidence/v2_runner.py.txt'))
        spec=importlib.util.spec_from_loader(loader.name,loader)
        historical=importlib.util.module_from_spec(spec); loader.exec_module(historical)
        a=np.array([[1,32]],dtype=np.uint8)
        reject(lambda:historical.CalProcessor({}, {}, {c}, out,'cpu')(c,'segmentation-front/000001.png',png(a)),
               'segmentation class ids out of range [0, 29): min=1, max=32')
        reject(lambda:features.segmentation_histogram_features(a),'segmentation class ids out of range [0, 29): min=1, max=32')
        passed('synthetic_class32_reproduces_actual_v2_processor_and_frozen_extractor_refusal')
        inv=np.arange(29,dtype=np.uint8).reshape(1,29)
        for a in (inv,np.tile(inv,(7,3))):
            assert np.array_equal(hist(a),features.segmentation_histogram_features(a))
            assert hist(a).dtype==np.float64
        for ident in range(29):
            a=np.full((2,3),ident,np.uint8)
            assert np.array_equal(hist(a),features.segmentation_histogram_features(a))
        passed('exact_in_vocabulary_feature_identity_every_ID_0_through_28')
        a=np.arange(256,dtype=np.uint8).reshape(16,16); before=a.copy()
        mapped=mapping(a)
        assert np.array_equal(a,before) and mapped.dtype==np.uint8 and not np.shares_memory(a,mapped)
        expected=a.copy(); expected[expected>28]=22
        assert np.array_equal(mapped,expected)
        assert np.array_equal(mapped[a<=28],a[a<=28])
        for ident in range(29,256):
            v=hist(np.full((2,3),ident,np.uint8)); q=np.zeros(29); q[22]=1
            assert np.array_equal(v,q)
        passed('exhaustive_all_uint8_OOV_IDs_29_through_255_uniformly_collapse_to_22_no_input_mutation')
        v=hist(a)
        assert v.shape==(29,) and v.dtype==np.float64 and np.isfinite(v).all() and v.sum()==1.0
        expected_hist=np.ones(29)/256; expected_hist[22]=228/256
        assert np.array_equal(v,expected_hist)
        passed('29_dimensions_finite_total_pixel_normalization_exact_expected_histogram')
        for channels in (1,2,3,4,7):
            a3=np.full((16,16,channels),255,np.uint8); a3[...,0]=a
            assert np.array_equal(mapping(a3),expected) and np.array_equal(hist(a3),v)
        passed('channel_zero_selected_before_mapping_all_extra_channels_ignored')
        schema='FROZEN_SEGMENTATION_SCHEMA_REQUIRES_UINT8_2D_OR_3D_CHANNEL_MAP'
        for bad in (np.ones((2,2),np.uint16),np.ones((2,2),np.float32),np.ones((2,2),np.int8),
                    np.ones((2,2),bool),np.ones((2,),np.uint8),np.ones((1,1,1,1),np.uint8),np.ones((2,2,0),np.uint8),[[1]]):
            reject(lambda bad=bad:mapping(bad),schema)
        for bad in (np.empty((0,2),np.uint8),np.empty((2,0,3),np.uint8)):
            reject(lambda bad=bad:mapping(bad),'FROZEN_SEGMENTATION_SCHEMA_REQUIRES_NONEMPTY_SPATIAL_MAP')
        passed('segmentation_invalid_dtype_dimensions_channels_empty_map_fail_loud_no_coercion')
        def camera_ast(path):
            tree=ast.parse(path.read_text())
            cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CalProcessor')
            node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__call__')
            gate=next(n for n in ast.walk(node) if isinstance(n,ast.If) and ast.unparse(n.test)=="kind == 'camera'")
            return ast.dump(ast.Module(body=gate.body,type_ignores=[]),include_attributes=False)
        assert camera_ast(HERE/'runner.py')==camera_ast(HERE/'evidence/v2_runner.py.txt')
        passed('camera_gate_and_feature_path_AST_identical_to_v2')
        proc=CalProcessor({}, {}, {c}, out,'cpu')
        for a3 in (a,np.stack([a,np.zeros_like(a),np.full_like(a,255)],axis=-1)):
            idx=len(proc.data.get('seg',{}))+1
            proc(c,f'segmentation-front/{idx:06d}.png',png(a3))
            assert np.array_equal(proc.data['seg'][idx],v)
        passed('actual_v3_PNG_processor_integration_OOV_2D_and_3D')
        registry=json.loads((HERE/'model_registry.json').read_text())['scorers']
        for row in registry:
            assert sha(HERE/row['bundle_state'])==row['bundle_state_sha256']
            assert recursive_content_hash(models[row['scorer_id']].state_dict())==row['tensor_content_sha256']
        assert len(models)==15
        passed('all_15_strictly_loaded_scorer_state_file_and_tensor_content_hashes_match')
        params=HERE/'upstream/fitted_oneclass_parameters.npz'
        expected_params=json.loads((HERE/'evidence/v2_scientific_bindings.json').read_text())['bundle_scientific_file_sha256']['upstream/fitted_oneclass_parameters.npz']
        assert sha(params)==expected_params
        parameter_hashes={}
        with np.load(params,allow_pickle=False) as z:
            for key in z.files: parameter_hashes[key]=hashlib.sha256(z[key].tobytes()).hexdigest()
            seg_keys=[k for k in z.files if k.lower().startswith('seg')]
            assert seg_keys and any(z[k].shape==(29,) for k in seg_keys)
        try: agents['Seg'].estimate_uncertainty(np.zeros(33))
        except ValueError: pass
        else: raise AssertionError('33-dimensional feature was accepted')
        assert np.isfinite(agents['Seg'].estimate_uncertainty(v).prediction)
        passed('fitted_parameter_file_identical_Seg_requires_29_dimensions_and_accepts_v3_histogram')
        entries=[]; n=4
        for name, table in [('imu.feather',pa.table({'acceleration_x':[0.,.1,.2,.3],'acceleration_y':[0.]*n,'acceleration_z':[9.8]*n})),
                            ('anomaly-observation.feather',pa.table({'tick':list(range(n)),'anomaly':[False,False,True,False]}))]:
            b=io.BytesIO(); feather.write_feather(table,b); entries.append((c+'/'+name,b.getvalue()))
        for t in range(n):
            b=io.BytesIO(); Image.fromarray(np.full((16,16,3),120,np.uint8)).save(b,format='JPEG')
            entries += [(f'{c}/rgb-front/{t:06d}.jpg',b.getvalue()),(f'{c}/segmentation-front/{t:06d}.png',png(a))]
        entries += [(e+'/segmentation-front/000001.png',b'invalid protected image'),(e+'/imu.feather',b'invalid protected feather'),(x+'/anomaly-observation.feather',b'invalid protected label')]
        data=fake_archive(entries); ledger={}; proc=CalProcessor(models,agents,{c},out,'cpu')
        scan(io.BytesIO(data),{c},{e},{x},proc,ledger,len(data))
        assert set(proc.completed)=={c} and len(proc.completed[c])==15 and ledger['eval_scenarios_decoded']==0
        assert all(s.shape==(3,) and np.isfinite(s).all() for s in proc.completed[c].values())
        passed('synthetic_OOV_archive_end_to_end_15_frozen_scorers_CAL_only')
        data=fake_archive([(c+'/unneeded.bin',b'x'),(e+'/segmentation-front/000001.png',png(a)),(e+'/imu.feather',b'x'),(x+'/anomaly-observation.feather',b'x')]); calls=[]
        def forbidden(*args,**kwargs): raise AssertionError('Protected scientific decoder reached')
        import calibration_adapter
        with ExitStack() as s:
            for obj,key in [(Image,'open'),(feather,'read_table'),(features,'segmentation_histogram_features'),(features,'camera_embedding_features'),(features,'imu_window_features'),(calibration_adapter,'labels_from_table')]:
                s.enter_context(patch.object(obj,key,forbidden))
            for model in models.values(): s.enter_context(patch.object(model,'forward',forbidden))
            scan(io.BytesIO(data),{c},{e},{x},lambda *args:calls.append(args),{},len(data))
        assert calls==[(c,None,None)]
        passed('EVAL_and_exclusion_image_Feather_feature_label_model_spies_unreachable')
    # Static protections: never call execute(), even with a mock transport.
    src=(HERE/'runner.py').read_text(); tree=ast.parse(src)
    assert src.count('opener.open(')==1 and 'method="GET"' in src
    assert 'Range' not in src and 'redirect_request' in src and 'REDIRECT_FORBIDDEN_SINGLE_REQUEST' in src
    execute=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='execute')
    segment=ast.get_source_segment(src,execute)
    assert segment.index('if OUTPUT.exists()')<segment.index('OUTPUT.mkdir')<segment.index('opener.open(')
    assert segment.index('attempt_started.json')<segment.index('opener.open(')
    assert segment.index('ARCHIVE_INTEGRITY_REFUSAL_NO_THRESHOLDS_FINALIZED')<segment.index('CAL_INCOMPLETE_NO_THRESHOLDS_FINALIZED')<segment.index('threshold([r')
    assert 'cal_scenario_processing_ledger.jsonl' not in segment
    assert 'INCOMPLETE_NO_RETRY' in segment and 'RESULT_SHA256SUMS' in segment
    assert src.count('parser.add_argument(')==2 and '--execute-authorized-final-cal' in src
    assert '/kaggle/working/cognix_final_conformal_cal_attempt003_v1' in src
    assert '91538225599' in src and '267e48f2249deb0269ad950aa81bca57dc02e3bbdf2d73acc172af267b18254a' in src
    passed('static_single_GET_no_redirect_Range_retry_resume_marker_new_path_complete_before_threshold')
    nb=json.loads((HERE/'cognix_final_conformal_cal_v3.ipynb').read_text())
    assert all(cell.get('execution_count') is None and not cell.get('outputs') for cell in nb['cells'])
    assert any('EXECUTE_SEPARATELY_AUTHORIZED_FINAL_CAL = False' in ''.join(cell['source']) for cell in nb['cells'])
    passed('notebook_unexecuted_default_false_and_separate_authorization')
    lock=json.loads((HERE/'runtime_lock.json').read_text())
    assert all(lock[k]==v for k,v in {'CUDA':'12.8','NumPy':'2.0.2','Pillow':'12.3.0','PyArrow':'25.0.1','PyTorch':'2.10.0+cu128','Python':'3.12.13','cuDNN':91002,'device':'one visible Tesla T4'}.items())
    assert 'pandas==2.2.3' in (HERE/'runtime_requirements.txt').read_text() and 'pandas.__version__ != "2.2.3"' in src
    passed('all_exact_runtime_pins_preserved_existing_pandas_pin_explicitly_enforced')
    # Mock only runtime metadata/control; do not invoke execute() or any transport.
    import platform, torch, pandas, PIL, pyarrow
    from runner import runtime_check
    def runtime_case(changes):
        version_targets={'Python':(platform,'python_version'),'NumPy':(np,'__version__'),
            'PyTorch':(torch,'__version__'),'Pillow':(PIL,'__version__'),'PyArrow':(pyarrow,'__version__'),
            'pandas':(pandas,'__version__')}
        expected={**lock,'pandas':'2.2.3'}
        with ExitStack() as stack:
            for key,(obj,attr) in version_targets.items():
                val=changes.get(key,expected[key])
                stack.enter_context(patch.object(obj,attr,return_value=val) if key=='Python' else patch.object(obj,attr,val))
            for obj,attr,val in [(torch.cuda,'is_available',True),(torch.cuda,'device_count',changes.get('gpu_count',1)),
                (torch.cuda,'get_device_name',changes.get('gpu_name','Tesla T4')),(torch.backends.cudnn,'version',changes.get('cuDNN',91002))]:
                stack.enter_context(patch.object(obj,attr,return_value=val))
            stack.enter_context(patch.object(torch.version,'cuda',changes.get('CUDA','12.8')))
            stack.enter_context(patch.object(torch,'set_num_threads'))
            stack.enter_context(patch.object(torch,'use_deterministic_algorithms'))
            flags=[(obj,attr,getattr(obj,attr)) for obj,attr in [(torch.backends.cudnn,'deterministic'),(torch.backends.cudnn,'benchmark'),(torch.backends.cuda.matmul,'allow_tf32'),(torch.backends.cudnn,'allow_tf32')]]
            try: return runtime_check()
            finally:
                for obj,attr,value in flags: setattr(obj,attr,value)
    runtime_case({})
    for key in ('Python','NumPy','PyTorch','Pillow','PyArrow'):
        reject(lambda key=key:runtime_case({key:'wrong'}), ('FROZEN_KAGGLE_RUNTIME_MISMATCH: ' if key in ('Python','NumPy','PyTorch') else 'SEALED_DECODER_VERSION_MISMATCH: ')+key)
    reject(lambda:runtime_case({'pandas':'wrong'}),'SEALED_PANDAS_VERSION_MISMATCH: pandas==2.2.3')
    for changes in ({'gpu_count':0},{'gpu_count':2},{'gpu_name':'Other T4'}):
        reject(lambda changes=changes:runtime_case(changes),'REQUIRES_ONE_VISIBLE_T4')
    for changes in ({'CUDA':'wrong'},{'cuDNN':0}):
        reject(lambda changes=changes:runtime_case(changes),'FROZEN_CUDA_CUDNN_MISMATCH')
    passed('mocked_runtime_metadata_all_pins_and_exactly_one_Tesla_T4_fail_closed_no_execute')
    return {'oov_regression_groups_passed':len(results),'oov_regression_groups_failed':0,'groups':results,
            'fitted_parameter_array_sha256':parameter_hashes,'real_TEST_bytes_used':0,'TEST_network_requests':0,
            'Attempt003_executed':False,'scientific_runner_execute_called':False}
if __name__=='__main__': print(json.dumps(run(),sort_keys=True))
