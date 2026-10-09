"""Current bindings only; no partition RNG and no historical result parsing."""
import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path
from graph_common import *

SCIENTIFIC_FILES = (
 'cognix/core/types.py','cognix/core/interfaces.py','cognix/adapters/carla/normality.py',
 'cognix/adapters/carla/real_agents.py','cognix/adapters/carla/real_features.py',
 'cognix/adapters/carla/pseudo_anomalies.py','cognix/adapters/carla/carlanomaly_loader.py',
 'cognix/graph/epistemic_gat.py')


def partition_audit(original, graph):
    fit,cal=original['FIT_NORMAL'],original['CAL_NORMAL']
    train,val=graph['GRAPH_TRAIN'],graph['GRAPH_VAL']
    for ids,count,role in ((fit,76,'FIT'),(cal,25,'CAL'),(train,61,'GRAPH_TRAIN'),(val,15,'GRAPH_VAL')):
        require(isinstance(ids,list) and len(ids)==len(set(ids))==count,'Wrong unique count: '+role)
        require(all(type(v) is str and re.fullmatch(r'Town(?:0[1-7]|10HD)/scenario-[1-9][0-9]*',v) for v in ids),
                'Noncanonical current scenario ID')
    require(not set(fit)&set(cal) and not set(train)&set(val) and set(train)|set(val)==set(fit) and
            not (set(train)|set(val))&set(cal),'Graph roles, exact FIT union or CAL exclusion failed')
    require(graph['frozen_FIT_NORMAL']==fit and graph['excluded_CAL_NORMAL']==cal and
            graph['permutation_output']==val+train and graph['sorted_input_ids']==sorted(fit) and
            graph['seed']==2027 and graph['permutation_calls_for_membership']==1,
            'Immutable graph partition record mismatch')
    # Read/assert the frozen record only. No generator, permutation or replay.
    return dict(status='PASS',partition_sha256=GRAPH_PARTITION_SHA,partition_seal=SEALS['graph_partition_freeze_v1'],
        GRAPH_TRAIN=train,GRAPH_VAL=val,FIT_NORMAL=fit,CAL_NORMAL_excluded=cal,
        GRAPH_TRAIN_count=61,GRAPH_VAL_count=15,union_equals_frozen_FIT_76=True,
        CAL_intersections=[],role_intersection=[],membership_recomputed=False)


def protocol_identity():
    raw=PROTOCOL.read_bytes()
    blob=subprocess.check_output(['git','show','HEAD:reports/carla_gat_preregistration_v1/protocol.json'],cwd=REPO,
        env=dict(os.environ,GIT_OPTIONAL_LOCKS='0',GIT_TERMINAL_PROMPT='0'))
    require(digest(blob)==PROTOCOL_SHA and raw.replace(b'\r\n',b'\n')==blob,
            'Historical protocol exact committed SHA/EOL equivalence failed')
    return dict(path=PROTOCOL.relative_to(REPO).as_posix(),expected_scientific_sha256=PROTOCOL_SHA,
        committed_LF_blob_sha256=digest(blob),working_file_sha256=digest(raw),
        working_CRLF_to_LF_equals_committed_bytes=True,source_modified=False),read_json(PROTOCOL)


def source_manifest():
    paths={REPO / p for p in SCIENTIFIC_FILES}
    for name in ('gate2_execution_bundle_v3','gate1_execution_bundle_v8','gate1_execution_bundle_v7'):
        paths.update((REPORT/name).glob('*.py'))
    return {p.relative_to(REPO).as_posix():hash_file(p) for p in sorted(paths)}


def authenticate_gate2():
    root=REPORT/'gate2_execution_bundle_v3'
    verify_seal(root,SEALS[root.name])
    for name in ('gate2_common','gate2_science','gate2_resolution','gate2_data','gate2_replay_inventory'):
        existing=sys.modules.get(name)
        require(existing is None or Path(existing.__file__).resolve()==root/(name+'.py'),'Ambiguous Gate-2 import')
    if str(root) not in sys.path: sys.path.append(str(root))
    for name in ('gate2_common','gate2_science','gate2_resolution','gate2_data','gate2_replay_inventory'):
        spec=importlib.util.find_spec(name)
        require(spec is not None and Path(spec.origin).resolve()==root/(name+'.py'),'Shadow Gate-2 source')
    return {name:importlib.import_module(name) for name in
            ('gate2_science','gate2_resolution','gate2_data','gate2_replay_inventory')}


def verify_current(*, full_source_hash=False, progress=lambda message:None):
    state=git_state()
    require(state['HEAD']==HEAD and not state['tracked'] and not state['staged'],'HEAD/tracked/staged mismatch')
    seals={}
    for name,seal in SEALS.items():
        progress('Independently hashing upstream: '+name)
        seals[name]=verify_seal(REPORT/name,seal)
    original_path=REPORT/'partition_freeze_v1/partition.json'
    graph_path=REPORT/'graph_partition_freeze_v1/partition.json'
    require(hash_file(original_path)==ORIGINAL_PARTITION_SHA and hash_file(graph_path)==GRAPH_PARTITION_SHA,
            'Original/graph partition SHA mismatch')
    for path,sha in ((original_path,ORIGINAL_PARTITION_SHA),(graph_path,GRAPH_PARTITION_SHA)):
        require(path.with_name('partition.json.sha256').read_bytes()==(sha+'  partition.json\n').encode(),
                'Detached partition hash mismatch')
    original,graph=read_json(original_path),read_json(graph_path)
    partition=partition_audit(original,graph)
    final_path=REPORT/'gate2_train_health_v3/FINAL.json'
    require(hash_file(final_path)==FINAL_SHA,'Gate-2 FINAL hash mismatch')
    final=read_json(final_path)
    require(final['status']=='TRAIN_ONLY_SCIENTIFIC_HEALTH_PASS' and
            final['modality_status']==dict(Camera='VALID',IMU='VALID',Seg='VALID'),'Gate-2 PASS/VALID required')
    for key,value in dict(TEST_requests=0,network_requests=0,graph_constructed=False,GAT_executed=False).items():
        require(type(final[key]) is type(value) and final[key]==value,'Gate-2 forbidden activity flag')
    manifest=read_json(REPORT/'gate2_train_health_v3/run_manifest.json')
    bindings=read_json(REPORT/'gate2_execution_bundle_v3/execution_bindings.json')
    require(manifest['HEAD']==bindings['HEAD']==HEAD and manifest['bindings']==bindings and
            manifest['bundle_seal']==SEALS['gate2_execution_bundle_v3'] and
            manifest['membership']==bindings['membership']==dict(FIT_NORMAL=partition['FIT_NORMAL'],
              CAL_NORMAL=partition['CAL_NORMAL_excluded']),'Current Gate-2 binding/membership mismatch')
    for name,seal in final['committed_unit_seals'].items(): verify_seal(REPORT/'gate2_train_health_v3/units'/name,seal)
    source=read_json(REPORT/'gate1_execution_bundle_v8/source_verification.json')
    require(source['identity']==SOURCE and source['independently_measured_full_sha256']==SOURCE['sha256'] and
            source_fingerprint()==source['fingerprint'],'Local TRAIN source identity changed')
    measured=None
    if full_source_hash:
        progress('Independently hashing full TRAIN archive; integrity only, no parser or payload replay')
        measured=hash_file(SOURCE['path'],source_progress=True)
        require(measured==SOURCE['sha256'] and source_fingerprint()==source['fingerprint'],'Full TRAIN SHA mismatch')
    protocol,definitions=protocol_identity()
    resolutions=read_json(REPORT/'gate2_execution_bundle_v3/verified_block_resolutions.json')
    require(resolutions['counts']=={'V8_NATIVE_EXTRACTED':33,'V7_TO_V8_IDENTITY_ADOPTION':68} and
            resolutions['feature_recomputed'] is False and resolutions['adopted_feature_files_copied'] is False,
            'Current canonical storage counts/mechanism mismatch')
    require(resolutions==manifest['block_resolutions'],'Resolution manifest mismatch')
    require(graph['upstream_bindings']['Gate2_v3_final_runtime_seal']==SEALS['gate2_train_health_v3'] and
            graph['upstream_bindings']['Gate2_FINAL_json_sha256']==FINAL_SHA,'Graph partition upstream mismatch')
    current_sources=source_manifest()
    require(all(hash_file(REPO/r['path'])==r['working_file_sha256'] for r in
                read_json(REPORT/'gate1_execution_bundle_v8/scientific_source_hashes.json')['source_files']),
            'Current scientific dependency differs from accepted Gate-1 source')
    import numpy as np
    require(sys.version==bindings['environment']['python'] and np.__version__==bindings['environment']['libraries']['numpy'] and
            hash_file(sys.executable)==bindings['environment']['executable_sha256'],'Bound Python/NumPy mismatch')
    return dict(schema='Experiment-2B-current-graph-bindings-v1',experiment='Experiment 2B',HEAD=HEAD,
        source_identity=SOURCE,source_fingerprint=source_fingerprint(),
        source_full_sha_independently_measured=measured,upstream_seals=seals,
        original_partition_sha256=ORIGINAL_PARTITION_SHA,graph_partition_sha256=GRAPH_PARTITION_SHA,
        graph_partition_seal=SEALS['graph_partition_freeze_v1'],Gate2_FINAL_sha256=FINAL_SHA,
        Gate2_status=final['status'],modality_status=final['modality_status'],partition=partition,
        historical_protocol_identity=protocol,scientific_source_hashes=current_sources,
        canonical_storage=dict(native_v8=33,identity_adopted_v7=68,
          resolution_manifest_sha256=hash_file(REPORT/'gate2_execution_bundle_v3/verified_block_resolutions.json'),
          resolver='authenticated gate2_resolution.resolve_arrays + gate2_data.load_clean',
          clean_features_recomputed=False,clean_features_copied_to_new_store=False),
        environment=dict(python=sys.version,numpy=np.__version__,python_executable=sys.executable,
          python_executable_sha256=hash_file(sys.executable)),activity=dict(ZERO))


def same_current(measured, frozen):
    # A cached authenticated full-hash receipt can be used for the sealing pass;
    # every future preflight and future data replay obtains a fresh full hash.
    a,b=dict(measured),dict(frozen)
    if a['source_full_sha_independently_measured'] is None:
        a['source_full_sha_independently_measured']=b['source_full_sha_independently_measured']
    require(a==b,'Current Experiment-2B binding changed')
