"""Preparation only; immutable new v2 evidence, synthetic tests and sealing."""
import argparse
import datetime
import json
import sys
import time
import traceback
from gate2_common import *
from gate2_forensics import verify_v1_failure, verify_v2_failure
from gate2_preflight import namespace_check, validate_bindings, verify_inputs, progress_printer


def initialize():
    require(not (BUNDLE / 'execution_bindings.json').exists(), 'Already initialized')
    counters = install_guard(writable=(BUNDLE,))
    namespace_check(preparation=True)
    failure = verify_v1_failure()
    write_new(BUNDLE / 'v1_failure_evidence.json', pretty(failure))
    v2_failure = verify_v2_failure()
    write_new(BUNDLE/'v2_failure_evidence.json',pretty(v2_failure))
    authenticate_upstream()
    from partition_common import evidence_files
    roots = read_json(V2 / 'preservation_baseline.json')['roots'] + [
        V2.relative_to(REPO).as_posix(), V2_RUNTIME.relative_to(REPO).as_posix()]
    baseline = evidence_files(roots)
    write_new(BUNDLE / 'preservation_baseline.json', pretty(baseline))
    bindings = read_json(V2 / 'execution_bindings.json')
    bindings.update(schema='Experiment-2B-Gate2-engineering-amendment-v3',
        prepared_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        real_runtime=TARGET.relative_to(REPO).as_posix(), authorization_token=TOKEN,
        v1_bundle_seal=V1_SEAL, v2_bundle_seal=V2_SEAL, v7_seal=V7_SEAL,
        v2_failure_evidence_sha256=hash_file(BUNDLE/'v2_failure_evidence.json'),
        v1_failure_evidence_sha256=hash_file(BUNDLE / 'v1_failure_evidence.json'),
        expected_storage_counts={'V8_NATIVE_EXTRACTED': 33, 'V7_TO_V8_IDENTITY_ADOPTION': 68},
        amendment_scope='Engineering canonical CAL inventory-record evidence binding only; unchanged Experiment 2B science',
        FIT_policy='Recompute frozen FIT under v3 manifest; no cross-runtime adoption',
        sealed_validation_sources={p.relative_to(REPO).as_posix(): hash_file(p)
                                   for p in sorted(V8.glob('*.py'))},
        unchanged_scientific_implementation={n: hash_file(V2 / n) for n in
            ('gate2_science.py','gate2_audit.py','execute_gate2.py','gate2_data.py','gate2_resolution.py')},
        canonical_inventory_source_hashes={n:hash_file(V8/n) for n in
            ('local_inventory.py','atomic_publication.py','source_inventory.json','inventory_scenarios.jsonl.gz')})
    write_new(BUNDLE / 'execution_bindings.json', pretty(bindings))
    validate_bindings(bindings)
    require(all(v == 0 for v in counters.values()), 'Preparation initialization guard violation')
    write_new(BUNDLE / 'initial_preparation_audit.json', pretty({
        'v1_failure_proved_before_v3_bindings': True, 'v1_committed_units': {},
        'v2_stopped_failure_and_three_FIT_units_verified_before_v3_bindings':True,
        'v2_committed_units':v2_failure['committed_units'],
        'baseline_files': len(baseline['files']), 'baseline_directories': len(baseline['directories']),
        'guard_counters': counters, 'real_fitting_executed': False,
        'real_calibration_executed': False, 'real_pseudo_generated': False,
        'graph_constructed': False, 'GAT_executed': False, 'TEST_requests': 0,
        'network_requests': 0, 'real_runtime_created': False, 'action': 'STOP FOR HUMAN REVIEW'}))
    print('V3 bindings initialized; v1/v2 stopped failures verified; baseline: %s files, %s directories' %
          (len(baseline['files']), len(baseline['directories'])), flush=True)


def verify():
    require(not (BUNDLE / 'SHA256SUMS').exists(), 'Already sealed')
    counters = install_guard(writable=(BUNDLE,), allow_source=True)
    started = time.monotonic()
    number = 1 + len(list(BUNDLE.glob('read_only_verification_attempt_*_results.json')))
    try:
        evidence = verify_inputs(progress_printer(), preparation=True)
        require(all(v == 0 for v in counters.values()), 'Read-only verification guard violation')
        audit = dict(evidence['summary'], success=True, guard_counters=counters,
                     preservation=evidence['preservation'],
                     verification_elapsed_seconds=time.monotonic() - started)
        write_new(BUNDLE / f'read_only_verification_attempt_{number}_results.json', pretty(audit))
        write_new(BUNDLE / 'read_only_upstream_verification.json', pretty(audit))
        write_new(BUNDLE / 'verified_block_resolutions.json', pretty(evidence['block_resolutions']))
        write_new(BUNDLE/'verified_replay_inventory_bindings.json',pretty(evidence['replay_inventory_bindings']))
        print(json.dumps({k:v for k,v in audit.items() if k != 'scientific_hashes'}, indent=2), flush=True)
    except BaseException as exc:
        path = BUNDLE / f'read_only_verification_attempt_{number}_failure.txt'
        if not path.exists():
            write_new(path, traceback.format_exc().encode())
        result = BUNDLE / f'read_only_verification_attempt_{number}_results.json'
        if not result.exists():
            write_new(result, pretty({'success': False, 'exception': repr(exc),
                'guard_counters': counters, 'action': 'STOP FOR HUMAN REVIEW'}))
        raise


def seal():
    counters = install_guard(writable=(BUNDLE,))
    require(not (BUNDLE / 'SHA256SUMS').exists(), 'Already sealed')
    namespace_check(preparation=True)
    test = read_json(BUNDLE / 'synthetic_test_results.json')
    require(test['success'] and test['failed'] == test['errors'] == test['skipped'] == 0,
            'Preparation tests did not all pass')
    require(test['tested_python_sha256'] == {p.name: hash_file(p) for p in BUNDLE.glob('*.py')},
            'Code changed since final tests')
    authenticate_upstream()
    from partition_common import verify_preservation, fingerprint
    preservation = verify_preservation(read_json(BUNDLE / 'preservation_baseline.json'))
    bindings = read_json(BUNDLE / 'execution_bindings.json')
    validate_bindings(bindings)
    audit = read_json(BUNDLE / 'read_only_upstream_verification.json')
    resolutions = read_json(BUNDLE / 'verified_block_resolutions.json')
    replay = read_json(BUNDLE/'verified_replay_inventory_bindings.json')
    require(audit['success'] and audit['source_full_sha256_remeasured'] and
            audit['all_101_gate2_loader_resolutions_independently_verified'] and
            audit['block_resolution_counts'] == bindings['expected_storage_counts'] == resolutions['counts'] and
            resolutions['membership'] == bindings['membership'] and
            digest(canonical(resolutions['resolutions'])) == resolutions['resolution_manifest_sha256'] ==
            audit['block_resolution_manifest_sha256'], 'Incomplete verification/resolution evidence')
    require(audit['all_101_canonical_raw_ledger_digests_cross_checked_with_sealed_index'] and
            replay['scenario_count'] == 101 and
            replay['index_file_sha256'] == bindings['canonical_inventory_source_hashes']['source_inventory.json'] and
            replay['ledger_file_sha256'] == bindings['canonical_inventory_source_hashes']['inventory_scenarios.jsonl.gz'] and
            all(r['canonical_record_sha256'] == r['indexed_record_sha256'] and
                not r['raw_ledger_contains_record_sha256'] for r in replay['bindings']),
            'Incomplete canonical replay inventory proof')
    source = read_json(V8 / 'source_verification.json')
    require(fingerprint(SOURCE['path']) == source['fingerprint'], 'Source changed after full verification')
    require(git('rev-parse', 'HEAD') == HEAD and not git('diff', '--name-only') and
            not git('diff', '--cached', '--name-only'), 'HEAD/tracked/staged changes')
    require(all(v == 0 for v in counters.values()), 'Seal guard violation')
    write_new(BUNDLE / 'preservation_audit.json', pretty(dict(preservation,
        v1_failure_evidence_verified=True, v1_bundle_seal=V1_SEAL,
        v2_failure_evidence_verified=True,v2_bundle_seal=V2_SEAL,v2_runtime_unchanged=True,
        v2_FIT_and_raw_pending_failed_attempt_seals_unchanged=True,
        v1_runtime_unchanged=True, all_historical_failure_pending_attempt_evidence_unchanged=True,
        baseline_sha256=hash_file(BUNDLE / 'preservation_baseline.json'))))
    names = sorted([p.name for p in BUNDLE.iterdir()] +
                   ['preparation_audit.json', 'bundle_inventory.json', 'SHA256SUMS', 'SHA256SUMS.sha256'])
    write_new(BUNDLE / 'bundle_inventory.json', pretty({'files': names, 'directories': [],
        'payload_hash_listing': 'SHA256SUMS', 'detached_listing_sha256': 'SHA256SUMS.sha256'}))
    audit.update(status='SEALED_PREPARATION_ONLY', tests_passed=test['passed'],
        tests_failed=test['failed'], test_errors=test['errors'], tests_skipped=test['skipped'],
        preservation=preservation, real_runtime_created=False, tracked_changes=[], staged_changes=[],
        real_fitting_executed=False, real_calibration_executed=False, real_pseudo_generated=False,
        feature_recomputed=False, adopted_feature_files_copied=False,
        graph_constructed=False, GAT_executed=False, TEST_requests=0, network_requests=0,
        scientific_repair_performed=False, dependency_installation_performed=False,
        commit_performed=False, push_performed=False,
        synthetic_tests_note='Modeling/calibration/pseudo in tests use artificial fixtures only; no real fitting, calibration or pseudo generation occurred.',
        bundle_files=names, bundle_seal_record='Detached SHA256SUMS.sha256; no self-referential hash',
        future_read_only_preflight_command='python -B reports/carla_train_full_v1/gate2_execution_bundle_v3/gate2_preflight.py --bundle-seal <REVIEWED_V3_SEAL>',
        future_authorized_execution_command='python -B reports/carla_train_full_v1/gate2_execution_bundle_v3/execute_gate2.py --bundle-seal <REVIEWED_V3_SEAL> --authorize-gate2 ' + TOKEN,
        action='STOP FOR HUMAN REVIEW')
    write_new(BUNDLE / 'preparation_audit.json', pretty(audit))
    seal = seal_tree(BUNDLE)
    require(sorted(p.name for p in BUNDLE.iterdir()) == names, 'Final inventory mismatch')
    print(json.dumps({'seal': seal, 'tests_passed': test['passed'], 'failed': test['failed'],
        'errors': test['errors'], 'skipped': test['skipped'], 'counts': resolutions['counts'],
        'preservation': preservation, 'action': 'STOP FOR HUMAN REVIEW'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('initialize', 'verify', 'seal'))
    args = parser.parse_args()
    {'initialize': initialize, 'verify': verify, 'seal': seal}[args.mode]()
