"""Read-only production consumers; no reconstructed resolver or inventory digest."""
import ast
import importlib
import importlib.util
import shutil
import numpy as np
from graph_common import *
from current_restore import inspect_current_states

RAW_TYPES = {'archive_ordinal':int,'emitted_rows':int,'first_tar_offset':int,
    'image_readability':str,'image_tick_coverage':dict,'last_tar_offset':int,
    'member_count':int,'members':list,'metadata_integrity':dict,'n_ticks':int,
    'parser_member_chain_sha256':str,'scenario_id':str,'schema':str,
    'source_member_set_sha256':str,'state':str,'town':str}


def verify_production_clean_resolution(modules, ctx, progress=lambda message:None):
    frozen=read_json(REPORT/'gate2_execution_bundle_v3/verified_block_resolutions.json')
    expected={r['scenario_id']:r for r in frozen['resolutions']}
    partition=read_json(BUNDLE/'current_partition_audit.json')
    require(set(expected)==set(ctx.scenarios)==set(partition['FIT_NORMAL']+partition['CAL_NORMAL_excluded']),
        'Production clean scenario admission mismatch')
    records=[]
    for index,sid in enumerate(sorted(ctx.scenarios),1):
        # This is the exact load_clean -> resolve_arrays path used by FITSink.
        clean=modules['gate2_data'].load_clean(sid,admitted=ctx)
        require(clean['scenario_id']==sid and clean['source_split']=='train' and clean['role']=='clean' and
            clean['synthetic_corruption'] is False,'Production clean role mismatch')
        provenance=clean['resolved_provenance']
        require(provenance==expected[sid],'Production resolution differs from sealed Gate2 manifest')
        require(not provenance['feature_recomputed'] and not provenance['adopted_feature_files_copied'],
            'Production features recomputed/copied')
        arrays=clean['arrays']; shapes={};dtypes={}
        for modality in NODE_ORDER:
            array=arrays[modality.lower()]
            require(array.shape==(2999,DIMS[modality]) and array.dtype==np.dtype('float64') and
                np.isfinite(array).all(),'Production clean shape/dtype/finite contract failed: '+sid+':'+modality)
            shapes[modality]=list(array.shape);dtypes[modality]=str(array.dtype)
        require(np.array_equal(arrays['tick'],np.arange(1,3000)) and
            np.all(arrays['scenario_id']==sid) and np.all(arrays['source_split']=='train') and
            np.all(arrays['town']==sid.split('/')[0]),'Production row identity mismatch')
        role='FIT_NORMAL' if sid in ctx.membership['FIT_NORMAL'] else 'CAL_NORMAL'
        graph_role='GRAPH_TRAIN' if sid in partition['GRAPH_TRAIN'] else 'GRAPH_VAL' if sid in partition['GRAPH_VAL'] else None
        require((role=='FIT_NORMAL')==(graph_role is not None),'Production partition role mismatch')
        records.append(dict(scenario_id=sid,archive_ordinal=provenance['archive_ordinal'],role=role,
            graph_role=graph_role,storage_form=provenance['storage_form'],emitted_rows=2999,
            shapes=shapes,dtypes=dtypes,all_finite=True,exact_sealed_provenance_equal=True,
            canonical_npz_sha256=clean['block_npz_sha256'],canonical_content_sha256=clean['block_content_sha256'],
            canonical_provenance_sha256=digest(canonical(provenance))))
        if index%10==0:progress('Production clean load: %d / 101'%index)
    counts={kind:sum(r['storage_form']==kind for r in records) for kind in
        ('V8_NATIVE_EXTRACTED','V7_TO_V8_IDENTITY_ADOPTION')}
    totals=dict(total_loaded=len(records),native_count=counts['V8_NATIVE_EXTRACTED'],
        adopted_count=counts['V7_TO_V8_IDENTITY_ADOPTION'],FIT_loaded=sum(r['role']=='FIT_NORMAL' for r in records),
        CAL_loaded=sum(r['role']=='CAL_NORMAL' for r in records),
        GRAPH_TRAIN_loaded=sum(r['graph_role']=='GRAPH_TRAIN' for r in records),
        GRAPH_VAL_loaded=sum(r['graph_role']=='GRAPH_VAL' for r in records))
    require(totals==dict(total_loaded=101,native_count=33,adopted_count=68,FIT_loaded=76,CAL_loaded=25,
        GRAPH_TRAIN_loaded=61,GRAPH_VAL_loaded=15),'Incomplete production clean resolution')
    return dict(status='PASS',**totals,resolution_counts=counts,records=records,failures=[],
        feature_recomputation=False,new_clean_store_created=False,features_copied=False,
        resolver='authenticated gate2_data.load_clean -> gate2_resolution.resolve_arrays',
        resolution_manifest_sha256=hash_file(REPORT/'gate2_execution_bundle_v3/verified_block_resolutions.json'),
        resolver_source_sha256=hash_file(Path(modules['gate2_resolution'].__file__)),
        loader_source_sha256=hash_file(Path(modules['gate2_data'].__file__)),activity=dict(ZERO))


def validate_inventory_record(raw,ordinal,replay):
    require(type(raw) is dict and set(raw)==set(RAW_TYPES) and
        all(type(raw[key]) is kind for key,kind in RAW_TYPES.items()),'Production raw inventory schema/type mismatch')
    require('record_sha256' not in raw and raw['schema']==replay.index['scenarios'][ordinal-1]['schema'] and
        raw['archive_ordinal']==ordinal and raw['n_ticks']==3000 and raw['emitted_rows']==2999 and
        raw['town']==raw['scenario_id'].split('/')[0] and raw['member_count']==len(raw['members']),
        'Production raw inventory alignment/count mismatch')
    legacy=replay.legacy
    summary=legacy.summary(raw)  # Only the exact sealed implementation.
    record_digest=legacy.digest(raw)  # Never graph_common.digest or a compact-JSON substitute.
    indexed=replay.index['scenarios'][ordinal-1]
    require(dict(summary,record_sha256=record_digest)==indexed,'Production canonical summary/digest disagreement')
    require(replay.bind(raw,ordinal)==record_digest,'Production ReplayInventory binding mismatch')
    n,_=legacy.check_layout(raw['scenario_id'],raw['members'])
    require(n==raw['n_ticks'] and legacy.member_digest(raw['members'])==raw['source_member_set_sha256'],
        'Production member-set binding mismatch')
    return summary,record_digest


def verify_production_inventory_schema(modules,ctx,progress=lambda message:None):
    replay=modules['gate2_replay_inventory'].load_verified_replay_inventory(ctx.source)
    require(Path(replay.legacy.__file__).resolve()==REPORT/'gate1_execution_bundle_v8/local_inventory.py',
        'Foreign production inventory API')
    seen=set(); records=[]; max_scratch=0; max_member=0
    for ordinal,raw in enumerate(replay.legacy.inventory_records(REPORT/'gate1_execution_bundle_v8',replay.index),1):
        require(ordinal<=101 and raw.get('scenario_id') not in seen,'Extra/duplicate production raw record')
        summary,sha=validate_inventory_record(raw,ordinal,replay)
        seen.add(raw['scenario_id'])
        scratch=sum(m['size'] for m in raw['members'] if m['type']!='5')
        largest=max(m['size'] for m in raw['members'])
        max_scratch=max(max_scratch,scratch);max_member=max(max_member,largest)
        records.append(dict(scenario_id=raw['scenario_id'],archive_ordinal=ordinal,canonical_summary=summary,
            canonical_digest=sha,indexed_digest=replay.index['scenarios'][ordinal-1]['record_sha256'],
            raw_record_sha256_present=False,summary_exact=True,digest_exact=True,
            all_source_member_fields_canonically_bound=True,whole_scenario_raw_bytes_upper_bound=scratch,
            largest_raw_member_bytes=largest))
        if ordinal%20==0:progress('Production raw inventory: %d / 101'%ordinal)
    replay.finish([r['archive_ordinal'] for r in records])
    require(len(records)==101 and seen==set(ctx.scenarios),'Production raw inventory coverage mismatch')
    return dict(status='PASS',production_records_checked=101,canonical_summaries_checked=101,
        canonical_digests_checked=101,fake_record_sha256_dependency=False,
        caller_supplied_record_sha256_required=False,mismatches=[],records=records,
        exact_raw_schema={key:kind.__name__ for key,kind in RAW_TYPES.items()},
        canonical_summary='sealed local_inventory.summary(raw)',canonical_digest='sealed local_inventory.digest(raw)',
        consumer='sealed gate2_replay_inventory.ReplayInventory.bind(raw,ordinal)',
        alternative_digest_implementation_introduced=False,inventory_rebuilt=False,
        TRAIN_sensor_payload_parsed=False,index_or_ledger_modified=False,
        index_sha256=hash_file(REPORT/'gate1_execution_bundle_v8/source_inventory.json'),
        ledger_sha256=hash_file(REPORT/'gate1_execution_bundle_v8/inventory_scenarios.jsonl.gz'),
        API_sha256=hash_file(Path(replay.legacy.__file__)),
        maximum_whole_scenario_scratch_bytes=max_scratch,maximum_actual_raw_member_bytes=max_member,
        activity=dict(ZERO))


def verify_compatibility_prerequisites(modules):
    science=modules['gate2_science'].load_science()
    required=('sanitize','generate_pseudo_anomaly','camera_embedding_features',
        'segmentation_histogram_features','imu_window_features','MahalanobisNormality',
        'BootstrapNormalityEnsemble','EnsemblePredictiveCalibrator','ensemble_to_uncertainty')
    require(all(callable(science.get(key)) for key in required),'Graph replay scientific compatibility missing')
    states=inspect_current_states()
    require(states==read_json(BUNDLE/'current_gate2_restore_audit.json'),'Restoration prerequisites changed')
    return dict(status='PASS',required_definitions=list(required),states_and_mapping_inspection=states,
        real_objects_instantiated=False,real_features_scored=False,fit_or_calibration_executed=False)


def verify_non_model_imports():
    names=('numpy','scipy','pandas','pyarrow','pyarrow.feather','PIL.Image','graph_replay','current_restore','train_replay')
    records={}
    for name in names:
        module=importlib.import_module(name)
        path=Path(module.__file__).resolve();records[name]=dict(origin=str(path),sha256=hash_file(path))
        if name=='train_replay':require(path==REPORT/'gate1_execution_bundle_v8/train_replay.py','Foreign production parser')
        if name in ('graph_replay','current_restore'):require(path==BUNDLE/(name+'.py'),'Foreign graph module')
    return dict(status='PASS',imports=records,parser_constructed=False,models_instantiated=False)


def verify_environment():
    name='gate2_preflight';root=REPORT/'gate2_execution_bundle_v3'
    existing=sys.modules.get(name);require(existing is None or Path(existing.__file__).resolve()==root/(name+'.py'),
        'Foreign production environment verifier')
    spec=importlib.util.find_spec(name);require(spec and Path(spec.origin).resolve()==root/(name+'.py'),
        'Foreign production environment discovery')
    verifier=importlib.import_module(name)
    actual=verifier.environment();expected=read_json(root/'execution_bindings.json')['environment']
    require(actual==expected,'Production Python/native/library environment changed')
    return dict(status='PASS',canonical_environment_sha256=digest(canonical(actual)),
        environment_verifier_sha256=hash_file(Path(verifier.__file__)),python=actual['python'],
        executable_sha256=actual['executable_sha256'],libraries=actual['libraries'],
        all_installed_library_file_manifests_equal=True,all_python_native_runtime_hashes_equal=True)


def disk_resource_audit(inventory_audit,*,free_bytes=None):
    spec=read_json(BUNDLE/'graph_export_spec.json')
    reserve=spec['free_disk_reserve_bytes'];cap=spec['raw_member_cap_bytes']
    require(reserve==20<<30 and cap==128<<20,'Existing raw scratch limits changed')
    require(inventory_audit['maximum_actual_raw_member_bytes']<=cap,'Production raw member exceeds existing cap')
    scratch=inventory_audit['maximum_whole_scenario_scratch_bytes']
    pairs=sum(spec['maximum_pairs_before_skips'].values())
    # Conservative size budget: 64 KiB per metadata row and per ledger record,
    # exact float64 node bytes, plus 1 GiB for NPZ/JSON/unit seals and global manifest.
    artifact=pairs*(3*(64<<10)+2*3*3*8)+(1<<30)
    margin=8<<30
    # Rename publication does not duplicate payloads; nevertheless reserve both.
    required=reserve+scratch+2*artifact+margin
    usage=shutil.disk_usage(REPORT);free=usage.free if free_bytes is None else free_bytes
    require(free>=required,'Insufficient disk/resource margin BEFORE full TRAIN hash')
    return dict(status='PASS',target_drive=REPORT.anchor,total_bytes=usage.total,free_bytes=free,
        raw_scratch_reserve_bytes=reserve,maximum_permitted_retained_raw_member_bytes=cap,
        maximum_actual_raw_member_bytes=inventory_audit['maximum_actual_raw_member_bytes'],
        one_scenario_scratch_upper_bound_bytes=scratch,maximum_pairs=pairs,
        graph_data_artifact_upper_bound_bytes=artifact,pending_and_final_artifact_budget_bytes=2*artifact,
        additional_margin_bytes=margin,required_free_bytes=required,remaining_margin_bytes=free-required,
        estimate_basis='64 KiB per each of two graph metadata rows and one candidate ledger; exact nodes; 1 GiB containers/manifests; whole raw scenario upper bound',
        historical_evidence_deleted=False,simulated_free_space=free_bytes is not None)


def segmentation_compatibility_audit(modules):
    science=modules['gate2_science'].load_science()
    raw=np.arange(256,dtype=np.uint8).reshape(16,16);before=raw.tobytes()
    sanitized,counts=science['sanitize'](raw)
    expected=np.arange(256,dtype=np.uint8);expected[29:]=22
    require(np.array_equal(sanitized.reshape(-1),expected) and raw.tobytes()==before,'Frozen Seg compatibility mismatch')
    vector=science['segmentation_histogram_features'](sanitized)
    require(vector.shape==(29,) and vector.dtype==np.float64,'Seg29 expansion or dtype change')
    # Call the unchanged generate_pair through its sanitizer branch, stopping
    # exactly at the pseudo call. No pseudo implementation, node scorer or graph runs.
    from graph_export import generate_pair
    observed=[]
    class StopBeforePseudo(Exception):pass
    class FixtureAdmission:
        def role(self,sid,source_split):
            require(sid=='fixture/SegCompatibility' and source_split=='train','Foreign Seg probe')
            return 'GRAPH_TRAIN'
    def stop(sample,**kwargs):
        require(np.array_equal(sample['Seg'],sanitized),'Sanitizer did not run before Seg recipe')
        observed.append(dict(recipe_id=kwargs['recipe_id'],seed=kwargs['seed'],severity=kwargs['severity']))
        raise StopBeforePseudo()
    probe=dict(science,generate_pseudo_anomaly=stop)
    try:generate_pair(probe,{m:None for m in NODE_ORDER},FixtureAdmission(),'fixture/SegCompatibility',2,raw,
        {m:np.zeros(DIMS[m],dtype=np.float64) for m in NODE_ORDER})
    except StopBeforePseudo:pass
    require(observed==[dict(recipe_id='seg_region_corruption',seed=2,severity=.2)],'Seg pre-recipe runtime probe incomplete')
    return dict(status='PASS',ids_0_through_28_unchanged=True,ids_29_through_255_mapped_to_bin_22=True,
        exact_mapping=expected.tolist(),sanitizer_applied_before_Seg_pseudo_generation=True,
        runtime_probe='synthetic 0..255 array, stops immediately before real pseudo implementation',
        real_pseudo_implementation_called=False,semantic_interpretation_added_for_31_through_255=False,
        Seg29_expanded=False,adaptive_replacement_or_tuning=False,raw_input_mutated=False,
        sanitizer_source_sha256=hash_file(REPORT/'gate1_execution_bundle_v8/segmentation_oov.py'),
        graph_export_sha256=hash_file(BUNDLE/'graph_export.py'),counts=counts,activity=dict(ZERO))
