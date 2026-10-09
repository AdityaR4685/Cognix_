"""Exclusive fixed membership freeze; explicit partition-only scope; no graph execution."""
import argparse
import json
import os
import stat
import subprocess
import sys
sys.dont_write_bytecode = True
from freeze_common import (BUNDLE, TARGET, PENDING, ZERO_STATE, require, read_json, write_json,
    write_new, assert_absent, install_guard, verify_seal, verify_preservation, git_state,
    canonical, digest, hash_file, seal_tree, verify_git_preserved)
from verify_upstream import verify_upstream
from partition_algorithm import derive
from partition_artifact import make_artifact, validate_artifact, methodology


def run(bundle_seal, scope):
    require(scope == 'partition-only', 'Explicit partition-only scope required')
    assert_absent()
    counters = install_guard((TARGET, PENDING))
    verify_seal(BUNDLE, bundle_seal)
    review = read_json(BUNDLE / 'procedure_review.json')
    tests = read_json(BUNDLE / 'synthetic_test_results.json')
    require(review['status'] == 'PASS' and review['source_hashes'] == tests['source_hashes'] and
            tests['success'] and tests['failed'] == tests['errors'] == tests['skipped'] == 0,
            'Reviewed/tested preparation required')
    require(tests['source_hashes'] == {p.name: hash_file(p) for p in sorted(BUNDLE.glob('*.py'))},
            'Tested source mismatch')
    evidence = verify_upstream(lambda message: print(message, file=sys.stderr, flush=True))
    require(evidence == read_json(BUNDLE / 'upstream_verification.json'), 'Prepared upstream bindings changed')
    baseline = read_json(BUNDLE / 'preservation_baseline.json')
    print('Checking historical preservation before one membership derivation', file=sys.stderr, flush=True)
    before = verify_preservation(baseline)
    state = git_state()
    require(state == read_json(BUNDLE / 'git_state_initial.json'), 'Git state changed before freeze')
    assert_absent()
    PENDING.mkdir(exist_ok=False)
    # This is the sole real membership-selection call. No automatic retry exists.
    derivation = derive(evidence['frozen_FIT_NORMAL'], evidence['excluded_CAL_NORMAL'])
    artifact = make_artifact(derivation, evidence, bundle_seal, counters)
    validate_artifact(artifact, evidence, bundle_seal)
    raw = canonical(artifact)
    partition_sha = digest(raw)
    write_new(PENDING / 'partition.json', raw)
    write_new(PENDING / 'partition.json.sha256', (partition_sha + '  partition.json\n').encode())
    # Fresh-process replay has a separate generator only for read-only verification.
    replay = subprocess.run([sys.executable, '-B', str(BUNDLE / 'replay_partition.py')],
        input=canonical({'frozen_FIT_NORMAL': evidence['frozen_FIT_NORMAL'], 'expected_derivation': derivation}),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    write_new(PENDING / 'deterministic_replay_stdout.txt', replay.stdout)
    write_new(PENDING / 'deterministic_replay_stderr.txt', replay.stderr)
    require(replay.returncode == 0, 'Deterministic replay failed; pending evidence retained')
    replay_record = json.loads(replay.stdout)
    require(replay_record['status'] == 'PASS' and replay_record['permutation_output'] == artifact['permutation_output'] and
            replay_record['python'] == evidence['environment']['python'] and
            replay_record['numpy_version'] == evidence['environment']['numpy_version'], 'Replay evidence mismatch')
    write_json(PENDING / 'deterministic_replay.json', replay_record)
    write_new(PENDING / 'methodology.md', methodology(artifact))
    print('Checking every historical hash after artifact/replay creation', file=sys.stderr, flush=True)
    after = verify_preservation(baseline)
    verify_git_preserved(git_state(), state)
    write_json(PENDING / 'preservation_audit.json', {'status': 'PASS', 'before': before, 'after': after,
        'guard_counters': dict(counters), 'git_HEAD': state['HEAD'],
        'tracked_changes': state['tracked_changes'], 'staged_changes': state['staged_changes'],
        'all_prior_evidence_read_only': True, 'original_partition_recomputed': False,
        'tests': {k: tests[k] for k in ('passed', 'failed', 'errors', 'skipped')},
        'procedure_review_sha256': hash_file(BUNDLE / 'procedure_review.json'),
        'publication': 'Exclusive pending files; sealed before Windows os.rename; fail closed on existing target',
        'human_review_completed': False, **ZERO_STATE})
    partition_seal = seal_tree(PENDING)
    require(not TARGET.exists(), 'Target appeared before publication; pending evidence retained')
    require(sys.platform == 'win32', 'Publication requires Windows no-overwrite rename semantics')
    os.rename(PENDING, TARGET)
    for path in TARGET.rglob('*'):
        if path.is_file():
            os.chmod(path, stat.S_IREAD)
    verify_seal(TARGET, partition_seal)
    validate_artifact(read_json(TARGET / 'partition.json'), evidence, bundle_seal)
    require(hash_file(TARGET / 'partition.json') == partition_sha, 'Published partition SHA mismatch')
    print('Checking historical preservation after immutable publication', file=sys.stderr, flush=True)
    published_preservation = verify_preservation(baseline)
    final_git = git_state()
    verify_git_preserved(final_git, state)
    return {'status': 'GRAPH_DEVELOPMENT_PARTITION_FROZEN', 'preparation_bundle_path': str(BUNDLE),
        'preparation_bundle_seal': bundle_seal, 'immutable_graph_partition_path': str(TARGET),
        'partition_sha256': partition_sha, 'partition_seal': partition_seal,
        'GRAPH_VAL': artifact['GRAPH_VAL'], 'GRAPH_TRAIN': artifact['GRAPH_TRAIN'], 'proof': artifact['proof'],
        'deterministic_replay_status': replay_record['status'], 'preservation': published_preservation,
        'guard_counters': counters, **ZERO_STATE,
        'tests': {k: tests[k] for k in ('passed','failed','errors','skipped')}, 'git_state': final_git,
        'future_read_only_verification_command': f'python -B "{BUNDLE / "verify_partition.py"}" '
             f'--bundle-seal {bundle_seal} --partition-seal {partition_seal} --partition-sha256 {partition_sha}',
        'action': 'STOP FOR HUMAN REVIEW'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-seal', required=True)
    parser.add_argument('--scope', required=True, choices=('partition-only',))
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.bundle_seal, args.scope), indent=2, sort_keys=True))
    except Exception as exc:
        print('STOP FOR HUMAN REVIEW: ' + str(exc), file=sys.stderr)
        sys.exit(2)
