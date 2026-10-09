"""One-time preparation, synthetic tests and sealing. Never compute a real partition."""
import argparse
import datetime
import json
import sys
import time
import unittest
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True
from partition_common import (BUNDLE, REPO, REPORT, V8, STORE, TARGET, HEAD, V8_SEAL,
    SOURCE, TOKEN, require, read_json, write_new, pretty_bytes, sha_bytes, hash_file,
    verify_manifest, evidence_files, install_guard, git)
from partition_evidence import load_v8, verify_inputs, preflight_summary
from partition_artifact import check_target_namespace
from synthetic_fixtures import SyntheticDirectory


def initialize():
    require(not (BUNDLE / 'execution_bindings.json').exists() and
            not (BUNDLE / 'SHA256SUMS.sha256').exists(), 'Preparation already initialized/sealed')
    install_guard(writable=(BUNDLE,))
    require(git('rev-parse', 'HEAD') == HEAD and not git('diff', '--name-only') and
            not git('diff', '--cached', '--name-only'), 'HEAD/tracked/staged mismatch')
    check_target_namespace(TARGET)
    require(not TARGET.exists(), 'Real partition must be absent during preparation')
    legacy, inventory, extraction, blocks, adoption = load_v8()
    legacy.verify_science()
    # Independently preserve and validate the pre-existing v7 failure/raw evidence.
    import prepare_seal
    preserved = prepare_seal.preservation()
    completion = read_json(STORE / 'complete_inventory_and_clean_store.json')
    require(hash_file(STORE / 'complete_inventory_and_clean_store.json') ==
            '8f483022518f7cc42d2ca722e326ef56a3073b87dac500bcc4dcde6125f723ba',
            'Expected completed v8 record changed')
    protocols = []
    for name in ('full_train_preregistered_protocol.json', 'development_gate_spec.json'):
        original = REPORT / name
        raw = original.read_bytes()
        write_new(BUNDLE / name, raw)
        protocols.append({'name': name, 'sha256': sha_bytes(raw), 'bytes': len(raw)})
    paths = [V8 / name for name in ('SHA256SUMS', 'SHA256SUMS.sha256', 'source_inventory.json',
        'inventory_scenarios.jsonl.gz', 'source_verification.json', 'experiment_2b_registration.json',
        'oov_policy.json', 'scientific_source_hashes.json', 'recovery_bindings.json',
        'v7_failure_and_adoption_proof.json', 'historical_preservation_baseline.json')]
    paths += [STORE / name for name in ('state.json', 'complete_inventory_and_clean_store.json',
        'complete_oov_ledger.jsonl.gz', 'complete_Gate1_SHA256SUMS', 'complete_Gate1_SHA256SUMS.sha256')]
    files = [{'path': p.relative_to(REPO).as_posix(), 'bytes': p.stat().st_size,
              'sha256': hash_file(p)} for p in paths]
    roots = read_json(V8 / 'historical_preservation_baseline.json')['roots'] + [
        V8.relative_to(REPO).as_posix(), STORE.relative_to(REPO).as_posix()]
    baseline = evidence_files(roots)
    write_new(BUNDLE / 'historical_preservation_baseline.json', pretty_bytes(baseline))
    state = read_json(STORE / 'state.json')['payload']
    bindings = {'schema': 'Experiment-2B-partition-preparation-bindings-v1',
        'experiment': 'Experiment 2B', 'HEAD': HEAD, 'v8_bundle_seal': V8_SEAL,
        'v8_runtime': STORE.relative_to(REPO).as_posix(),
        'v8_completion_record_sha256': hash_file(STORE / 'complete_inventory_and_clean_store.json'),
        'v8_COMPLETE_GATE1_state_sha256': hash_file(STORE / 'state.json'),
        'v8_completion_seal': hash_file(STORE / 'complete_Gate1_SHA256SUMS'),
        'v8_clean_store_sha256': completion['clean_store_sha256'],
        'v8_global_inventory_and_store_sha256': completion['global_inventory_and_store_sha256'],
        'v8_complete_oov_ledger': completion['complete_oov_ledger'],
        'inventory_binding': completion['inventory_binding'], 'source_identity': SOURCE,
        'v8_preprocessing_policy_sha256': completion['preprocessing_policy_sha256'],
        'scientific_source_bindings': state['scientific_extractor_hashes'],
        'v8_scenario_block_adoption_identities': completion['completed_blocks'],
        'files': files, 'protocols': protocols,
        'split_rule': read_json(REPORT / 'full_train_preregistered_protocol.json')['split'],
        'numpy_version': np.__version__, 'expected_N': 101, 'expected_CAL': 25, 'expected_FIT': 76,
        'prepared_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'original_experiment_2A_protocol_mutated': False,
        'Gate_2': 'Separate human authorization after partition review',
        'future_scientific_repair': 'Separately registered Experiment 2C or later',
        'old_preservation_verified': preserved,
        'preparation_only': True, 'real_partition_computation_authorized_now': False}
    write_new(BUNDLE / 'execution_bindings.json', pretty_bytes(bindings))
    print(json.dumps({'status': 'INITIALIZED_PREPARATION_ONLY',
        'historical_files_bound': len(baseline['files']), 'real_partition': 'absent'}, indent=2))


def tests():
    require(not (BUNDLE / 'SHA256SUMS.sha256').exists(), 'Sealed bundle cannot record new tests')
    with SyntheticDirectory(REPO) as tmp:
        # All data fixtures are synthetic, source opens/network/TEST remain forbidden.
        counters = install_guard(writable=(BUNDLE, Path(tmp)))
        import test_partition
        test_partition.FIXTURE_ROOT = Path(tmp)
        suite = unittest.defaultTestLoader.loadTestsFromModule(test_partition)
        with (BUNDLE / 'test_transcript.txt').open('x', encoding='utf-8') as stream:
            result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        report = {'tests_run': result.testsRun, 'passed': result.testsRun - len(result.failures) -
            len(result.errors) - len(result.skipped), 'failed': len(result.failures),
            'errors': len(result.errors), 'skipped': len(result.skipped),
            'success': result.wasSuccessful(), 'synthetic_fixtures_only': True,
            'guard_counters': counters, 'numpy_version': np.__version__,
            'tested_python_sha256': {p.name: hash_file(p) for p in sorted(BUNDLE.glob('*.py'))}}
        write_new(BUNDLE / 'synthetic_test_results.json', pretty_bytes(report))
        print(json.dumps(report, indent=2, sort_keys=True))
        require(report['success'] and not report['skipped'], 'Synthetic tests did not all pass')


def seal():
    require(not (BUNDLE / 'SHA256SUMS.sha256').exists() and not
            (BUNDLE / 'preparation_audit.json').exists(), 'Already sealed/audited; rewrite refused')
    counters = install_guard(writable=(BUNDLE,), allow_source=True)
    test = read_json(BUNDLE / 'synthetic_test_results.json')
    require(test['success'] and test['failed'] == test['errors'] == test['skipped'] == 0, 'Tests failed')
    require(test['tested_python_sha256'] == {p.name: hash_file(p) for p in BUNDLE.glob('*.py')},
            'Python code changed after synthetic tests')
    require(all(v == 0 for v in test['guard_counters'].values()), 'Unexpected synthetic guard violation')
    from partition_preflight import progress_printer
    started = time.monotonic()
    evidence = verify_inputs(progress_printer())
    check_target_namespace(TARGET)
    require(not TARGET.exists(), 'Real partition exists during preparation')
    audit = preflight_summary(evidence, 'absent', counters)
    require(all(v == 0 for v in counters.values()), 'Preparation guard violation')
    audit.update(status='SEALED_PREPARATION_ONLY', source_verification='independent full archive SHA256 measured during this preparation',
        verification_elapsed_seconds=time.monotonic() - started,
        real_partition_computed=False, real_partition_frozen=False,
        tests_passed=test['passed'], tests_failed=test['failed'], test_errors=test['errors'],
        tests_skipped=test['skipped'], tracked_changes=git('diff', '--name-only').splitlines(),
        staged_changes=git('diff', '--cached', '--name-only').splitlines(),
        retained_test_attempts=sorted(p.name for p in BUNDLE.glob('synthetic_test_attempt_*')),
        test_harness_repairs='Windows tempfile 0700 ACL replaced by workspace fixtures with inherited permissions; missing synthetic OOV fixture field supplied. Prior attempts retained; final suite verifies exact sealed Python hashes.',
        future_preflight_command='python -B reports/carla_train_full_v1/partition_freeze_bundle_v1/partition_preflight.py --bundle-seal <REVIEWED_BUNDLE_SEAL>',
        future_authorized_partition_freeze_command='python -B reports/carla_train_full_v1/partition_freeze_bundle_v1/partition_freeze.py --bundle-seal <REVIEWED_BUNDLE_SEAL> --authorize-partition-freeze ' + TOKEN,
        bundle_seal_record='SHA256SUMS.sha256 (detached to avoid self-referential hashing)',
        bundle_files=sorted([p.name for p in BUNDLE.iterdir()] + ['preparation_audit.json', 'SHA256SUMS', 'SHA256SUMS.sha256']))
    write_new(BUNDLE / 'preparation_audit.json', pretty_bytes(audit))
    require(all(p.is_file() for p in BUNDLE.iterdir()), 'Unexpected preparation directory')
    listing = ''.join(hash_file(p) + '  ' + p.name + '\n' for p in sorted(BUNDLE.iterdir())).encode()
    write_new(BUNDLE / 'SHA256SUMS', listing)
    bundle_seal = sha_bytes(listing)
    write_new(BUNDLE / 'SHA256SUMS.sha256', (bundle_seal + '  SHA256SUMS\n').encode())
    verify_manifest(BUNDLE, expected_seal=bundle_seal)
    audit['bundle_seal'] = bundle_seal
    for name in ('future_preflight_command', 'future_authorized_partition_freeze_command'):
        audit[name] = audit[name].replace('<REVIEWED_BUNDLE_SEAL>', bundle_seal)
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('initialize', 'tests', 'seal'))
    args = parser.parse_args()
    {'initialize': initialize, 'tests': tests, 'seal': seal}[args.mode]()
