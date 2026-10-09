"""One pre-source closure for preparation, read-only preflight, and future data."""
import contextlib
from graph_common import *
import graph_common
from current_upstream import authenticate_gate2,verify_current,same_current
from production_audits import (verify_production_clean_resolution,verify_production_inventory_schema,
    verify_compatibility_prerequisites,verify_non_model_imports,verify_environment,disk_resource_audit)

PRE_SOURCE_STEPS=('guard_installed','external_seal','git_state','historical_preservation','future_namespaces',
    'upstream_small_seals_and_partitions','real_context','production_clean_resolution',
    'production_inventory_schema','compatibility_prerequisites','non_model_imports','environment',
    'disk_resources','barriers')


@contextlib.contextmanager
def read_only_scientific_barrier():
    previous=sys.getprofile();calls={};metadata_cache={}
    audit_scope={'active':True}
    def pre_source_audit(event,args):
        if not audit_scope['active']:return
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            require(norm(args[0])!=norm(SOURCE['path']),'TRAIN archive read before pre-source closure completion')
            mode,flags=args[1:3]
            require(not ((isinstance(mode,str) and any(c in mode for c in 'wax+')) or
                flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)), 'Write inside read-only pre-source closure')
        if event in ('os.remove','os.rmdir','os.mkdir','os.rename','os.chmod','os.utime','os.link','os.symlink','shutil.copyfile'):
            require(False,'Mutation inside read-only pre-source closure')
    sys.addaudithook(pre_source_audit)
    forbidden={'feed','replay_file','build_inventory','metadata_from_memory','generate_pseudo_anomaly',
        'generate_pair','write_scenario','fit','fit_calibrator','predict_normality','prob_normal','node',
        'restore_current','new_agent','restore_agent','assemble_fit','modality_calibration',
        'camera_embedding_features','segmentation_histogram_features','imu_window_features'}
    model_classes=('MahalanobisNormality','BootstrapNormalityEnsemble','EnsemblePredictiveCalibrator',
        'RealCameraAgent','RealSegAgent','RealIMUAgent','RealNormalityAgent','RestoredScorer','TrainReplay','FITSink')
    def observe(frame,event,arg):
        if event!='call':return
        code=frame.f_code
        if code not in metadata_cache:
            filename=code.co_filename
            if not os.path.normcase(filename).startswith(os.path.normcase(str(REPO))+os.sep):
                metadata_cache[code]=None
            else:
                path=Path(filename);name=code.co_name;qualified=code.co_qualname
                invalid=name in forbidden or name=='__init__' and any(cls+'.' in qualified for cls in model_classes)
                key=(path.relative_to(REPO).as_posix(),qualified,code.co_firstlineno) if any(
                    path.is_relative_to(REPORT/root) for root in ('gate1_execution_bundle_v8','gate2_execution_bundle_v3')) else None
                metadata_cache[code]=(invalid,qualified,key)
        metadata=metadata_cache[code]
        if metadata is None:return
        invalid,qualified,key=metadata
        if invalid:require(False,'Semantic source/feature/pseudo/model execution forbidden in pre-source closure: '+qualified)
        if key is not None:calls[key]=calls.get(key,0)+1
    sys.setprofile(observe)
    try:yield calls
    finally:
        sys.setprofile(previous)
        audit_scope['active']=False


def verify_barriers():
    records=[]
    for phase in ('preflight','graph-data'):
        guard,counters=make_guard(phase,source_integrity=True)
        cases=[('socket.connect',(None,)),('open',(r'E:\carlanomaly-base-test.tar.gz','rb',os.O_RDONLY)),
            ('import',('torch',None,None,None,None)),('import',('cognix.graph',None,None,None,None))]
        for version in (1,2):
            for name in (f'graph_development_data_v{version}',f'graph_development_runs_v{version}',
                f'.graph_development_data_v{version}.pending'):
                cases.append(('open',(str(REPORT/name/'forbidden'),'wb',os.O_WRONLY|os.O_CREAT)))
        cases.append(('open',(str(RUNS/'forbidden'),'wb',os.O_WRONLY|os.O_CREAT)))
        for event,args in cases:
            try:guard(event,args)
            except PreparationError:records.append(dict(phase=phase,event=event,target=str(args[0]),blocked=True))
            else:require(False,'TEST/network/model/historical write barrier failed')
        require(counters['TEST_requests']==counters['network_requests']==0,'Barrier probe performed real access')
    return dict(status='PASS',probes=records,simulated_audit_events_only=True,real_operations_performed=False,
        shared_subprocess_implementation='subprocess_policy.subprocess_admission',
        subprocess_policy_byte_identical_to_v2=hash_file(BUNDLE/'subprocess_policy.py')==hash_file(V2_BUNDLE/'subprocess_policy.py'),
        source_semantic_execution_blocked_by_profile=True,activity=dict(ZERO))


def run_ordered_steps(callbacks,trace):
    require(tuple(callbacks)==PRE_SOURCE_STEPS,'Pre-source engineering order changed')
    values={}
    for name,callback in callbacks.items():
        trace.append(dict(step=name,event='BEGIN'))
        values[name]=callback(values)
        trace.append(dict(step=name,event='PASS'))
    return values


def pre_source_closure(external_seal,*,preparing=False,progress=lambda message:None):
    """Steps 1-14. No archive SHA or pending creation can occur here."""
    require(graph_common.ACTIVE_PHASE in ('prepare','synthetic','preflight','graph-data'),'Missing process guard')
    require(not preparing or graph_common.ACTIVE_PHASE in ('prepare','synthetic'),'Unsealed preparation only in reviewed fixture/preparation phase')
    from amendment_evidence import verify_amendment
    from subprocess_policy import policy_report
    def seal_step(values):
        if preparing:
            require(external_seal is None and not (BUNDLE/'SHA256SUMS').exists(),'Invalid unsealed preparation admission')
            return dict(unsealed_preparation_only=True)
        result=verify_seal(BUNDLE,external_seal)
        tests=read_json(BUNDLE/'synthetic_test_results.json')
        actual={p.name:hash_file(p) for p in BUNDLE.glob('*.py')}
        require(tests['passed']==tests['tests_run'] and tests['failed']==tests['errors']==tests['skipped']==0 and
            tests['bundle_python_sha256']==actual and tests['real_activity']==ZERO,
            'Accepted tested implementation/evidence mismatch BEFORE source hash')
        require(read_json(BUNDLE/'source_code_audit.json')['new_bundle_python_sha256']==actual and
            read_json(BUNDLE/'preparation_audit.json')['scientific_invariance']=='PASS',
            'Source/preparation evidence mismatch BEFORE source hash')
        require(all(p.stat().st_file_attributes&stat.FILE_ATTRIBUTE_READONLY for p in BUNDLE.rglob('*') if p.is_file()),
            'V3 immutable attributes changed BEFORE source hash')
        return result
    def git_step(values):
        state=git_state();require(state['HEAD']==HEAD and not state['tracked'] and not state['staged'],'Git state mismatch')
        if (BUNDLE/'preparation_initial.json').exists():
            require(state==read_json(BUNDLE/'preparation_initial.json')['git_state'],'Current Git status changed')
        return state
    def history(values):
        verify_amendment()
        return verify_preservation(read_json(BUNDLE/'preservation_baseline.json'),progress)
    def current(values):
        result=verify_current(full_source_hash=False,progress=progress)
        same_current(result,read_json(BUNDLE/'current_experiment_bindings.json'))
        return result
    def context_step(values):
        modules=authenticate_gate2()
        ctx=modules['gate2_resolution'].real_context()
        modules['gate2_resolution']._CONTEXT=ctx
        return dict(modules=modules,ctx=ctx)
    def clean(values):
        r=values['real_context'];return verify_production_clean_resolution(r['modules'],r['ctx'],progress)
    def inventory(values):
        r=values['real_context'];return verify_production_inventory_schema(r['modules'],r['ctx'],progress)
    callbacks={
        'guard_installed':lambda v:dict(phase=graph_common.ACTIVE_PHASE,shared_policy=policy_report()['policy']['schema']),
        'external_seal':seal_step,'git_state':git_step,'historical_preservation':history,
        'future_namespaces':lambda v:future_namespaces_absent(),
        'upstream_small_seals_and_partitions':current,'real_context':context_step,
        'production_clean_resolution':clean,'production_inventory_schema':inventory,
        'compatibility_prerequisites':lambda v:verify_compatibility_prerequisites(v['real_context']['modules']),
        'non_model_imports':lambda v:verify_non_model_imports(),
        'environment':lambda v:verify_environment(),
        'disk_resources':lambda v:disk_resource_audit(v['production_inventory_schema']),
        'barriers':lambda v:verify_barriers()}
    trace=[]
    with read_only_scientific_barrier() as calls:
        values=run_ordered_steps(callbacks,trace)
    ctx=values['real_context']['ctx'];ctx.unchanged()
    return dict(status='PASS',pre_source_steps=list(PRE_SOURCE_STEPS),trace=trace,
        real_context_status='PASS',resolution_context_identity_sha256=ctx.identity_sha256,
        production_clean_resolution=values['production_clean_resolution'],
        production_inventory_schema=values['production_inventory_schema'],
        compatibility_prerequisites=values['compatibility_prerequisites'],non_model_imports=values['non_model_imports'],
        environment=values['environment'],disk_resources=values['disk_resources'],barriers=values['barriers'],
        historical_preservation=values['historical_preservation'],git_state=values['git_state'],
        source_fingerprint_before_full_hash=source_fingerprint(),
        reached_production_functions=[dict(path=p,qualified_function=fn,first_line=line,calls=count,
            source_sha256=hash_file(REPO/p)) for (p,fn,line),count in sorted(calls.items())],
        full_archive_hash_invoked=False,pending_created=False,features_recomputed=False,
        clean_store_created=False,real_models_instantiated=False,activity=dict(ZERO))


def complete_source_integrity(closure,progress=lambda message:None):
    require(closure['status']=='PASS' and [r['step'] for r in closure['trace'] if r['event']=='PASS']==list(PRE_SOURCE_STEPS),
        'Incomplete pre-source closure')
    before=source_fingerprint()
    require(before==closure['source_fingerprint_before_full_hash'] and before['bytes']==SOURCE['bytes'],
        'Source fingerprint changed before full integrity hash')
    progress('All fourteen pre-source gates PASS; now hashing full TRAIN archive, integrity only')
    sha=hash_file(SOURCE['path'],source_progress=True)
    require(sha==SOURCE['sha256'] and source_fingerprint()==before,'Full source SHA/fingerprint mismatch')
    return dict(status='PASS',bytes=SOURCE['bytes'],sha256=sha,fingerprint_before=before,fingerprint_after=source_fingerprint(),
        fresh_full_archive_hash=True,source_payload_parsed=False,pending_created=False,
        trace=closure['trace']+[dict(step='full_source_hash',event='PASS'),dict(step='source_fingerprint_recheck',event='PASS')])


def admission_then_publication(closure_callback,integrity_callback,publication_callback):
    """Shared order seam; publication is unreachable on closure or SHA failure."""
    closure=closure_callback()
    source=integrity_callback(closure)
    require(source['status']=='PASS' and source['sha256']==SOURCE['sha256'] and
        source['bytes']==SOURCE['bytes'] and source['fingerprint_before']==source['fingerprint_after'],
        'Publication needs successful source integrity')
    return publication_callback(closure,source)
