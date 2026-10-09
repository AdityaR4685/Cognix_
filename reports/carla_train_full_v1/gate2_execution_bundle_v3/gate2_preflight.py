"""Read-only admission: remeasure the full TRAIN archive; never compute membership."""
import argparse
import importlib.metadata
import json
import platform
import re
import sys
import time
from pathlib import Path
from gate2_common import (BUNDLE, REPO, REPORT, V8, STORE, PARTITION, TARGET, HEAD,
    V8_SEAL, PARTITION_SHA, PARTITION_SEAL, PARTITION_BUNDLE_SEAL, SOURCE, MODALITIES,
    DIMS, RECIPES, SEVERITIES, V1, V1_RUNTIME, V1_SEAL, V2, V2_RUNTIME, V2_SEAL, V7_SEAL, TOKEN,
    require, read_json, hash_file, verify_seal, canonical,
    authenticate_upstream, install_guard, git, safe_path)


def environment():
    installed = {}
    for name in ('numpy', 'scipy', 'pandas', 'pyarrow', 'Pillow'):
        distribution = importlib.metadata.distribution(name)
        entries = []
        for relative in sorted(distribution.files, key=str):
            path = Path(distribution.locate_file(relative))
            require(path.is_file(), 'Missing installed dependency file: ' + str(path))
            entries.append({'path': str(relative).replace('\\', '/'), 'bytes': path.stat().st_size,
                            'sha256': hash_file(path)})
        installed[name] = {'version': distribution.version,
                           'actual_installed_file_hashes': entries,
                           'file_manifest_sha256': __import__('gate2_common').digest(canonical(entries))}
    runtime = {}
    prefix = Path(sys.base_prefix)
    for path in sorted([*prefix.glob('python*.dll'), *prefix.glob('vcruntime*.dll'),
                        *prefix.glob('DLLs/*.pyd'), *prefix.glob('DLLs/*.dll')]):
        runtime[str(path.relative_to(prefix)).replace('\\', '/')] = hash_file(path)
    return {'python': sys.version, 'executable': str(Path(sys.executable).resolve()),
            'executable_sha256': hash_file(sys.executable), 'platform': sys.platform,
            'windows_version': list(sys.getwindowsversion()) if sys.platform == 'win32' else None,
            'libraries': {name: importlib.metadata.version(name) for name in
                          ('numpy', 'scipy', 'pandas', 'pyarrow', 'Pillow')},
            'installed_library_file_manifests': installed, 'python_native_runtime_hashes': runtime}


def validate_partition(record, expected, completion):
    require(record['N'] == 101 and record['n_fit'] == 76 and record['n_cal'] == 25 and
            record['experiment'] == 'Experiment 2B' and record['seed'] == 2026,
            'Partition counts/experiment mismatch')
    for role in ('FIT_NORMAL', 'CAL_NORMAL'):
        require(record[role] == expected[role], 'Exact partition membership binding mismatch')
    fit, cal = record['FIT_NORMAL'], record['CAL_NORMAL']
    require(len(fit) == len(set(fit)) == 76 and len(cal) == len(set(cal)) == 25 and
            not set(fit) & set(cal) and set(fit + cal) ==
            {r['scenario_id'] for r in completion['completed_blocks']}, 'Role isolation mismatch')
    require(record['v8_COMPLETE_GATE1'] == completion and
            [r['v8_block_or_adoption_identity'] for r in record['scenario_records_in_archive_order']] ==
            completion['completed_blocks'], 'Partition block/adoption binding mismatch')
    require(record['fitting_executed'] is False and record['pseudo_rows'] == 0 and
            record['TEST_requests'] == record['network_requests'] == 0 and
            not record['graph_constructed'] and not record['GAT_executed'], 'Partition scientific state mismatch')


def namespace_check(preparation=False):
    safe_path(TARGET)
    allowed = {BUNDLE.name, TARGET.name, V1.name, V1_RUNTIME.name, V2.name, V2_RUNTIME.name}
    for path in REPORT.iterdir():
        name = path.name.lower()
        require(not ((name.startswith('gate2') or name.startswith('.gate2')) and path.name not in allowed),
                'Previous conflicting Gate-2 namespace')
        require(not re.search(r'(^|[_-])(graph|gat|nograph|standardgat|epistemicgat)($|[_\-.0-9])', name),
                'Graph state present in Experiment-2 namespace')
    require(not preparation or not TARGET.exists(), 'Real runtime already exists during preparation')


def validate_bindings(bindings):
    expected = {'HEAD': HEAD, 'v8_seal': V8_SEAL, 'partition_sha256': PARTITION_SHA,
                'partition_runtime_seal': PARTITION_SEAL,
                'partition_bundle_seal': PARTITION_BUNDLE_SEAL, 'source_identity': SOURCE,
                'active_modalities': list(MODALITIES), 'dimensions': DIMS, 'window_ticks': 12,
                'bootstrap_members': 5, 'bootstrap_seed': 42,
                'pseudo_rotation': list(RECIPES), 'active_pseudo_severities': SEVERITIES,
                'max_pseudo_per_tick': 1, 'pseudo_base_seed': 0, 'GNSS_downstream': 'EXCLUDED'}
    expected.update(schema='Experiment-2B-Gate2-engineering-amendment-v3',
                    real_runtime=TARGET.relative_to(REPO).as_posix(), authorization_token=TOKEN,
                    v7_seal=V7_SEAL, v1_bundle_seal=V1_SEAL, v2_bundle_seal=V2_SEAL,
                    FIT_policy='Recompute frozen FIT under v3 manifest; no cross-runtime adoption',
                    expected_storage_counts={'V8_NATIVE_EXTRACTED': 33, 'V7_TO_V8_IDENTITY_ADOPTION': 68},
                    preparation_only=True, real_execution_authorized_now=False)
    require(all(bindings.get(k) == value for k, value in expected.items()), 'Execution binding mismatch')
    from gate2_forensics import verify_v1_failure, verify_v2_failure
    require(verify_v1_failure() == read_json(BUNDLE / 'v1_failure_evidence.json'),
            'Preserved v1 failure evidence changed')
    require(hash_file(BUNDLE / 'v1_failure_evidence.json') == bindings['v1_failure_evidence_sha256'],
            'V1 forensic binding changed')
    require(verify_v2_failure() == read_json(BUNDLE/'v2_failure_evidence.json') and
            hash_file(BUNDLE/'v2_failure_evidence.json') == bindings['v2_failure_evidence_sha256'],
            'Preserved v2 stopped execution evidence changed')
    for relative, sha in bindings['sealed_validation_sources'].items():
        require(hash_file(REPO / relative) == sha, 'Sealed validation source changed: ' + relative)
    for name,sha in bindings['canonical_inventory_source_hashes'].items():
        require(hash_file(V8/name) == sha, 'Canonical inventory implementation/evidence changed: ' + name)
    for name, sha in bindings['unchanged_scientific_implementation'].items():
        require(hash_file(BUNDLE / name) == sha == hash_file(V2 / name),
                'Scientific implementation changed: ' + name)
    require(bindings['preprocessing_policy_sha256'] ==
            bindings['scientific_hashes']['preprocessing_policy_sha256'] ==
            read_json(V8 / 'scientific_source_hashes.json')['preprocessing_policy_sha256'],
            'Preprocessing-policy hash mismatch')
    for name, sha in bindings['protocol_hashes'].items():
        require(hash_file(REPORT / name) == sha and
                (REPORT / name).read_bytes() == (BUNDLE / name).read_bytes(), 'Frozen protocol mismatch')
    for path, sha in bindings['extra_source_hashes'].items():
        require(hash_file(REPO / path) == sha, 'Additional source mismatch: ' + path)
    actual_environment = environment()
    require(actual_environment == bindings['environment'] and
            __import__('gate2_common').digest(canonical(actual_environment)) == bindings['environment_sha256'],
            'Python/library/executable environment mismatch')


def progress_printer():
    last = [0.0]
    def progress(count):
        if time.monotonic() - last[0] >= 30:
            print('Read-only full TRAIN source hash verification: %s bytes' % count,
                  file=sys.stderr, flush=True)
            last[0] = time.monotonic()
    return progress


def verify_inputs(progress=None, preparation=False):
    require(git('rev-parse', 'HEAD') == HEAD and not git('diff', '--name-only') and
            not git('diff', '--cached', '--name-only'), 'HEAD/tracked/staged mismatch')
    namespace_check(preparation)
    bindings = read_json(BUNDLE / 'execution_bindings.json')
    validate_bindings(bindings)
    upstream = authenticate_upstream()
    evidence = upstream.verify_inputs(progress)
    verify_seal(PARTITION, PARTITION_SEAL)
    require(hash_file(PARTITION / 'partition.json') == PARTITION_SHA and
            (PARTITION / 'partition.json.sha256').read_bytes() ==
            (PARTITION_SHA + '  partition.json\n').encode(), 'Frozen partition SHA mismatch')
    record = read_json(PARTITION / 'partition.json')
    require((PARTITION / 'partition.json').read_bytes() == canonical(record), 'Noncanonical partition')
    validate_partition(record, bindings['membership'], evidence['completion'])
    gate = read_json(PARTITION / 'DATA_INTEGRITY_GATE_PASS.json')
    require(gate['status'] == 'DATA_INTEGRITY_GATE_PASS' and
            gate['partition_sha256'] == PARTITION_SHA and gate['all_required_integrity_predicates_verified'],
            'Gate-1 PASS binding mismatch')
    require(hash_file(PARTITION / 'member_evidence.jsonl.gz') == evidence['index']['ledger_sha256'],
            'Exact partition member evidence mismatch')
    require(record['scientific_source_bindings'] == evidence['science'] == bindings['scientific_hashes'] and
            record['upstream_bindings'] == evidence['bindings'], 'Scientific/partition upstream binding mismatch')
    require(bindings['preprocessing_policy_sha256'] == evidence['science']['preprocessing_policy_sha256'],
            'Preprocessing-policy mismatch')
    from partition_common import verify_preservation
    preservation = verify_preservation(read_json(BUNDLE / 'preservation_baseline.json'))
    from gate2_resolution import verify_all_resolutions
    resolutions = verify_all_resolutions()
    require(resolutions['counts'] == bindings['expected_storage_counts'] and
            resolutions['membership'] == bindings['membership'], 'Block resolution counts/roles changed')
    from gate2_replay_inventory import verify_all_replay_bindings
    replay_inventory = verify_all_replay_bindings(evidence['index']['source_binding'])
    require(replay_inventory['scenario_count'] == 101, 'Incomplete canonical replay inventory binding proof')
    namespace_check(preparation)
    return {'bindings': bindings, 'membership': bindings['membership'], 'environment': environment(),
            'completion': evidence['completion'], 'source_binding': evidence['index']['source_binding'],
            'preservation': preservation, 'block_resolutions': resolutions,
            'replay_inventory_bindings': replay_inventory,
            'summary': {'HEAD': HEAD, 'v8_seal_verified': V8_SEAL,
                'v8_COMPLETE_GATE1_verified': True, 'all_101_blocks_adoptions_verified': True,
                'partition_SHA256_verified': PARTITION_SHA,
                'partition_runtime_seal_verified': PARTITION_SEAL,
                'FIT_scenarios': 76, 'CAL_scenarios': 25,
                'active_modalities': list(MODALITIES), 'GNSS_downstream': 'EXCLUDED',
                'source_identity_verified': SOURCE, 'source_full_sha256_remeasured': True,
                'partition_recomputed': False, 'preprocessing_policy_sha256': bindings['preprocessing_policy_sha256'],
                'scientific_hashes': evidence['science'], 'environment_sha256': bindings['environment_sha256'],
                'Python_version': sys.version, 'library_versions': bindings['environment']['libraries'],
                'fitting_executed_now': False, 'calibration_executed_now': False,
                'block_resolution_counts': resolutions['counts'],
                'block_resolution_role_counts': resolutions['role_counts'],
                'block_resolution_manifest_sha256': resolutions['resolution_manifest_sha256'],
                'all_101_gate2_loader_resolutions_independently_verified': True,
                'adopted_v7_feature_bytes_identical': True, 'native_v8_feature_bytes_unchanged': True,
                'feature_recomputed': False, 'adopted_feature_files_copied': False,
                'v1_failure_evidence_verified': True, 'v1_committed_units': {},
                'v2_failure_evidence_verified': True,
                'v2_committed_FIT_unit_seals': read_json(BUNDLE/'v2_failure_evidence.json')['committed_units'],
                'all_101_canonical_raw_ledger_digests_cross_checked_with_sealed_index':True,
                'raw_ledger_record_sha256_field_present':False,
                'pseudo_generated_now': False, 'graph_constructed': False, 'GAT_executed': False,
                'TEST_requests': 0, 'network_requests': 0, 'action': 'STOP FOR HUMAN REVIEW'}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-seal', required=True)
    args = parser.parse_args()
    counters = install_guard(allow_source=True)
    try:
        verify_seal(BUNDLE, args.bundle_seal)
        evidence = verify_inputs(progress_printer())
        state = 'absent'
        if TARGET.exists():
            from gate2_runtime import verify_runtime
            state = verify_runtime(TARGET, args.bundle_seal, evidence)['state']
        require(all(v == 0 for v in counters.values()), 'Guard violation')
        print(json.dumps(dict(evidence['summary'], runtime_state=state, guard_counters=counters), indent=2))
        return 0
    except Exception as exc:
        print('STOP FOR HUMAN REVIEW: ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
