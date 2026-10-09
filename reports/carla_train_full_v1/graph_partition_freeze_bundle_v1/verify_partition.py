"""Future READ-ONLY partition verification/preflight; no training or output writes."""
import argparse
import json
import stat
import subprocess
import sys
sys.dont_write_bytecode = True
from freeze_common import (BUNDLE, TARGET, PENDING, ZERO_STATE, require, read_json, verify_seal,
    install_guard, verify_preservation, git_state, canonical, hash_file, digest, verify_git_preserved)
from verify_upstream import verify_upstream
from partition_artifact import validate_artifact, methodology


def run(bundle_seal, partition_seal, partition_sha):
    counters = install_guard()
    verify_seal(BUNDLE, bundle_seal)
    verify_seal(TARGET, partition_seal)
    require(not PENDING.exists() and not list(TARGET.parent.glob('.' + TARGET.name + '.pending*')),
            'Unreviewed pending namespace exists')
    require(hash_file(TARGET / 'partition.json') == partition_sha and
            (TARGET / 'partition.json.sha256').read_bytes() == (partition_sha + '  partition.json\n').encode(),
            'Externally bound partition hash mismatch')
    artifact = read_json(TARGET / 'partition.json')
    require(canonical(artifact) == (TARGET / 'partition.json').read_bytes(), 'Noncanonical partition JSON')
    evidence = verify_upstream(lambda message: print(message, file=sys.stderr, flush=True))
    require(evidence == read_json(BUNDLE / 'upstream_verification.json'), 'Prepared upstream evidence changed')
    validate_artifact(artifact, evidence, bundle_seal)
    require((TARGET / 'methodology.md').read_bytes() == methodology(artifact), 'Methodology mismatch')
    saved_replay = read_json(TARGET / 'deterministic_replay.json')
    require(saved_replay['permutation_output'] == artifact['permutation_output'] and saved_replay['status'] == 'PASS',
            'Saved replay mismatch')
    result = subprocess.run([sys.executable, '-B', str(BUNDLE / 'replay_partition.py')],
        input=canonical({'frozen_FIT_NORMAL': evidence['frozen_FIT_NORMAL'], 'expected_derivation': artifact}),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    require(result.returncode == 0, 'Read-only independent replay failed')
    require(json.loads(result.stdout) == saved_replay, 'Fresh replay differs from sealed replay evidence')
    print('Verifying all historical preservation SHA256 hashes (read-only)', file=sys.stderr, flush=True)
    preservation = verify_preservation(read_json(BUNDLE / 'preservation_baseline.json'))
    state = git_state()
    initial = read_json(BUNDLE / 'git_state_initial.json')
    require(state['HEAD'] == initial['HEAD'] and not state['tracked_changes'] and not state['staged_changes'],
            'HEAD/tracked/staged state mismatch')
    verify_git_preserved(state, initial)
    require(all(getattr(p.stat(), 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_READONLY
                for root in (BUNDLE, TARGET) for p in root.rglob('*') if p.is_file()), 'Read-only attribute absent')
    return {'status': 'READ_ONLY_GRAPH_PARTITION_PREFLIGHT_PASS', 'partition_sha256': partition_sha,
        'preparation_bundle_seal': bundle_seal, 'partition_seal': partition_seal,
        'counts': {'GRAPH_TRAIN': 61, 'GRAPH_VAL': 15}, 'proof': artifact['proof'],
        'deterministic_replay': 'PASS', 'historical_preservation': preservation,
        'guard_counters': counters, 'git_state': state, **ZERO_STATE,
        'files_written': 0, 'graph_execution_authorized': False, 'action': 'STOP FOR HUMAN REVIEW'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-seal', required=True)
    parser.add_argument('--partition-seal', required=True)
    parser.add_argument('--partition-sha256', required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.bundle_seal, args.partition_seal, args.partition_sha256), indent=2, sort_keys=True))
    except Exception as exc:
        print('STOP FOR HUMAN REVIEW: ' + str(exc), file=sys.stderr)
        sys.exit(2)
