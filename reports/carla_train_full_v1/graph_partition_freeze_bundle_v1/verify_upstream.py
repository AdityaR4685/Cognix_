"""Independently hash upstream seals and read ONLY identity/membership/status metadata."""
from freeze_common import (REPORT, HEAD, UPSTREAM_SEALS, PARTITION_SHA, FINAL_SHA, SOURCE,
    ZERO_STATE, require, verify_seal, read_json, hash_file, source_fingerprint, git_state, environment)
from partition_algorithm import validate_input


def validate_final(final):
    require(final['status'] == 'TRAIN_ONLY_SCIENTIFIC_HEALTH_PASS', 'Gate-2 status is not PASS')
    require(final['modality_status'] == {'Camera': 'VALID', 'IMU': 'VALID', 'Seg': 'VALID'},
            'Gate-2 modalities not all VALID')
    for key, value in ZERO_STATE.items():
        require(type(final[key]) is type(value) and final[key] == value, 'Gate-2 state mismatch: ' + key)


def verify_upstream(progress=lambda message: None):
    state = git_state()
    require(state['HEAD'] == HEAD and not state['tracked_changes'] and not state['staged_changes'],
            'HEAD, tracked or staged state mismatch')
    seals = {}
    for name, expected in UPSTREAM_SEALS.items():
        progress('Hashing immutable upstream seal: ' + name)
        seals[name] = verify_seal(REPORT / name, expected)
    ppath = REPORT / 'partition_freeze_v1' / 'partition.json'
    require(hash_file(ppath) == PARTITION_SHA, 'Original partition SHA mismatch')
    require((ppath.parent / 'partition.json.sha256').read_bytes() ==
            (PARTITION_SHA + '  partition.json\n').encode(), 'Original detached partition SHA mismatch')
    partition = read_json(ppath)
    fit, cal = partition['FIT_NORMAL'], partition['CAL_NORMAL']
    validate_input(fit, cal)
    require(partition['n_fit'] == 76 and partition['n_cal'] == 25 and partition['N'] == 101,
            'Original frozen counts mismatch')
    require(partition['upstream_bindings']['HEAD'] == HEAD and
            partition['upstream_bindings']['source_identity'] == SOURCE, 'Original upstream identity mismatch')
    require(partition['sorted_canonical_ids'] == sorted(fit + cal), 'Original canonical universe mismatch')
    final_path = REPORT / 'gate2_train_health_v3' / 'FINAL.json'
    require(hash_file(final_path) == FINAL_SHA, 'Gate-2 FINAL.json SHA mismatch')
    final = read_json(final_path)
    validate_final(final)
    bindings = read_json(REPORT / 'gate2_execution_bundle_v3' / 'execution_bindings.json')
    manifest = read_json(REPORT / 'gate2_train_health_v3' / 'run_manifest.json')
    membership = {'FIT_NORMAL': fit, 'CAL_NORMAL': cal}
    require(bindings['membership'] == manifest['membership'] == membership, 'Gate-2/frozen membership mismatch')
    require(manifest['bindings'] == bindings, 'Gate-2 runtime/bundle bindings differ')
    for record in (bindings, manifest):
        require(record['HEAD'] == HEAD and record['v8_seal'] == UPSTREAM_SEALS['gate1_execution_bundle_v8'] and
                record['partition_sha256'] == PARTITION_SHA and
                record['partition_runtime_seal'] == UPSTREAM_SEALS['partition_freeze_v1'],
                'Gate-2 upstream seal/HEAD mismatch')
    require(manifest['bundle_seal'] == UPSTREAM_SEALS['gate2_execution_bundle_v3'] and
            bindings['source_identity'] == SOURCE, 'Gate-2 bundle/source identity mismatch')
    for key, value in ZERO_STATE.items():
        require(type(manifest[key]) is type(value) and manifest[key] == value, 'Gate-2 manifest state mismatch')
    source = read_json(REPORT / 'gate1_execution_bundle_v8' / 'source_verification.json')
    require(source['identity'] == SOURCE and source['independently_measured_full_sha256'] == SOURCE['sha256'],
            'Gate-1 source identity mismatch')
    fingerprint = source_fingerprint()
    require(fingerprint == source['fingerprint'], 'Local TRAIN archive fingerprint changed')
    # Verify the committed-unit seals without parsing audits, metrics, arrays or performance.
    for name, seal in final['committed_unit_seals'].items():
        verify_seal(REPORT / 'gate2_train_health_v3' / 'units' / name, seal)
    env = environment()
    require(env['python'] == bindings['environment']['python'] and
            env['python_executable_sha256'] == bindings['environment']['executable_sha256'] and
            env['numpy_version'] == bindings['environment']['libraries']['numpy'], 'Bound Python/NumPy mismatch')
    upstream = {'HEAD': HEAD, 'original_TRAIN_archive_local_identity': SOURCE,
                'archive_current_fingerprint': fingerprint,
                'archive_identity_verification': 'SHA256 from authenticated Gate-1 full-hash evidence; '
                    'current path/size/device/inode/mtime/ctime match; archive content not reread in this task',
                'Gate1_v8_seal': UPSTREAM_SEALS['gate1_execution_bundle_v8'],
                'original_partition_sha256': PARTITION_SHA,
                'original_partition_runtime_seal': UPSTREAM_SEALS['partition_freeze_v1'],
                'original_partition_bundle_seal': UPSTREAM_SEALS['partition_freeze_bundle_v1'],
                'Gate2_v3_bundle_seal': UPSTREAM_SEALS['gate2_execution_bundle_v3'],
                'Gate2_v3_final_runtime_seal': UPSTREAM_SEALS['gate2_train_health_v3'],
                'Gate2_FINAL_json_sha256': FINAL_SHA, 'Gate2_status': final['status'],
                'Gate2_modality_status': final['modality_status'], **ZERO_STATE}
    return {'schema': 'graph-partition-upstream-verification-v1', 'upstream_bindings': upstream,
            'upstream_seal_verifications': seals, 'frozen_FIT_NORMAL': fit, 'excluded_CAL_NORMAL': cal,
            'original_partition_recomputed': False, 'performance_metadata_read_for_membership': False,
            'committed_unit_seals_verified': final['committed_unit_seals'], 'environment': env}
