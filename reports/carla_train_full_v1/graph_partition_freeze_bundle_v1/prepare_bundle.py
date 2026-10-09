"""New preparation lifecycle: initialize, tests, seal. Never replace phase evidence."""
import argparse
import contextlib
import io
import sys
import unittest
sys.dont_write_bytecode = True
from freeze_common import (BUNDLE, HEAD, TARGET, PENDING, require, read_json, write_json, write_new,
    assert_absent, install_guard, preservation_snapshot, verify_preservation, git_state,
    hash_file, canonical, digest, seal_tree)
from verify_upstream import verify_upstream


def run(mode):
    counters = install_guard((BUNDLE,))
    require(not (BUNDLE / 'SHA256SUMS').exists(), 'Preparation already sealed')
    assert_absent()
    if mode == 'initialize':
        require(not (BUNDLE / 'preservation_baseline.json').exists(), 'Initialization already exists')
        print('Taking exact historical preservation baseline', flush=True)
        baseline = preservation_snapshot()
        write_json(BUNDLE / 'preservation_baseline.json', baseline)
        state = git_state()
        require(state['HEAD'] == HEAD and not state['tracked_changes'] and not state['staged_changes'], 'Git state mismatch')
        write_json(BUNDLE / 'git_state_initial.json', state)
        upstream = verify_upstream(lambda m: print(m, flush=True))
        write_json(BUNDLE / 'upstream_verification.json', upstream)
        write_json(BUNDLE / 'execution_bindings.json', {
            'schema': 'graph-development-partition-freeze-bindings-v1',
            'scope': 'Partition freeze only; no graph artifacts or execution',
            'upstream': upstream, 'target': TARGET.relative_to(BUNDLE.parents[2]).as_posix(),
            'seed': 2027, 'fit_count': 76, 'graph_train_count': 61, 'graph_val_count': 15,
            'human_review_after_freeze': True, 'authorization_basis': 'Current explicit user partition-freeze request',
            'overwrite_allowed': False, 'original_partition_recomputation_allowed': False,
            'performance_informed_selection_allowed': False, 'split_changes_after_graph_results_allowed': False})
        print('Initialization PASS; no real graph partition computed', flush=True)
    elif mode == 'complete-initialization':
        # Retain the first baseline/Git record after a verifier-only correction.
        # Refuse to overwrite anything; no real membership is computed here.
        require((BUNDLE / 'preservation_baseline.json').exists() and
                (BUNDLE / 'git_state_initial.json').exists() and
                (BUNDLE / 'initialization_attempt_1.json').exists(), 'Missing retained attempt evidence')
        require(not (BUNDLE / 'upstream_verification.json').exists() and
                not (BUNDLE / 'execution_bindings.json').exists(), 'Initialization already completed')
        require(git_state() == read_json(BUNDLE / 'git_state_initial.json'), 'Git changed since baseline')
        upstream = verify_upstream(lambda m: print(m, flush=True))
        write_json(BUNDLE / 'upstream_verification.json', upstream)
        write_json(BUNDLE / 'execution_bindings.json', {
            'schema': 'graph-development-partition-freeze-bindings-v1',
            'scope': 'Partition freeze only; no graph artifacts or execution',
            'upstream': upstream, 'target': TARGET.relative_to(BUNDLE.parents[2]).as_posix(),
            'seed': 2027, 'fit_count': 76, 'graph_train_count': 61, 'graph_val_count': 15,
            'human_review_after_freeze': True, 'authorization_basis': 'Current explicit user partition-freeze request',
            'overwrite_allowed': False, 'original_partition_recomputation_allowed': False,
            'performance_informed_selection_allowed': False, 'split_changes_after_graph_results_allowed': False})
        print('Initialization completed with original retained baseline; no membership computation', flush=True)
    elif mode == 'tests':
        require((BUNDLE / 'execution_bindings.json').exists(), 'Initialize first')
        require(not (BUNDLE / 'synthetic_test_results.json').exists(), 'Test results already exist')
        import test_partition_freeze
        suite = unittest.defaultTestLoader.loadTestsFromModule(test_partition_freeze)
        transcript = io.StringIO()
        with contextlib.redirect_stdout(transcript), contextlib.redirect_stderr(transcript):
            result = unittest.TextTestRunner(stream=transcript, verbosity=2).run(suite)
        record = {'schema': 'synthetic-tests-v1', 'synthetic_only': True,
            'passed': result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped),
            'failed': len(result.failures), 'errors': len(result.errors), 'skipped': len(result.skipped),
            'tests_run': result.testsRun, 'success': result.wasSuccessful(),
            'source_hashes': {p.name: hash_file(p) for p in sorted(BUNDLE.glob('*.py'))},
            'guard_counters': counters,
            'synthetic_guard_probes': 'Guard callbacks tested directly; no network/TEST/model access attempted'}
        write_new(BUNDLE / 'test_transcript.txt', transcript.getvalue().encode())
        write_json(BUNDLE / 'synthetic_test_results.json', record)
        print(transcript.getvalue())
        require(result.wasSuccessful() and not result.skipped, 'Tests failed/errors/skipped')
    elif mode == 'seal':
        tests = read_json(BUNDLE / 'synthetic_test_results.json')
        require(tests['success'] and tests['failed'] == tests['errors'] == tests['skipped'] == 0, 'Tests not clean')
        require(tests['source_hashes'] == {p.name: hash_file(p) for p in sorted(BUNDLE.glob('*.py'))},
                'Source changed after tests')
        review = read_json(BUNDLE / 'procedure_review.json')
        require(review['status'] == 'PASS' and review['review_kind'] == 'Codex source/procedure review' and
                review['human_review_completed'] is False and review['source_hashes'] == tests['source_hashes'],
                'Procedure review does not bind tested source')
        require(verify_upstream(lambda m: print(m, flush=True)) == read_json(BUNDLE / 'upstream_verification.json'),
                'Upstream verification changed')
        print('Rechecking all historical preservation hashes', flush=True)
        preservation = verify_preservation(read_json(BUNDLE / 'preservation_baseline.json'))
        state = git_state()
        require(state == read_json(BUNDLE / 'git_state_initial.json'), 'Git state changed during preparation')
        write_json(BUNDLE / 'preparation_audit.json', {'status': 'PASS', 'preservation': preservation,
            'git_state': state, 'tests': {k: tests[k] for k in ('passed','failed','errors','skipped')},
            'procedure_review_sha256': hash_file(BUNDLE / 'procedure_review.json'),
            'guard_counters': counters, 'real_partition_computed_during_preparation': False,
            'bundle_seal_reference': 'SHA256SUMS.sha256 (no self-referential hash)',
            'next_authorized_action': 'Only fixed graph-development partition freeze; stop for human review'})
        print('PREPARATION_BUNDLE_SEAL=' + seal_tree(BUNDLE), flush=True)
    else:
        require(False, 'Unknown mode')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('initialize', 'complete-initialization', 'tests', 'seal'))
    args = parser.parse_args()
    try:
        run(args.mode)
    except Exception as exc:
        print('STOP FOR HUMAN REVIEW: ' + str(exc), file=sys.stderr)
        sys.exit(2)
