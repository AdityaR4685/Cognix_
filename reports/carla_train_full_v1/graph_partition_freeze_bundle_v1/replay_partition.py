"""Independent fresh-process, read-only replay; never publishes/reselects membership."""
import json
import sys
sys.dont_write_bytecode = True
from freeze_common import install_guard, require, digest, canonical


def main():
    counters = install_guard()
    import numpy as np
    payload = json.loads(sys.stdin.buffer.read())
    fit = payload['frozen_FIT_NORMAL']
    require(len(fit) == len(set(fit)) == 76, 'Replay input count/uniqueness mismatch')
    sorted_ids = sorted(fit)
    # A separate verification process: one fixed generator, one fixed permutation.
    generator = np.random.Generator(np.random.PCG64(2027))
    initial = generator.bit_generator.state
    ids = generator.permutation(sorted_ids).tolist()
    final = generator.bit_generator.state
    n_val = max(1, round(0.20 * 76))
    expected = payload['expected_derivation']
    require(n_val == 15 and ids == expected['permutation_output'] and
            ids[:n_val] == expected['GRAPH_VAL'] and ids[n_val:] == expected['GRAPH_TRAIN'] and
            initial == expected['initial_rng_state'] and final == expected['final_rng_state'],
            'Independent replay mismatch')
    print(json.dumps({'status': 'PASS', 'fresh_process': True, 'verification_only': True,
        'generator_instances_in_replay': 1, 'permutation_calls_in_replay': 1, 'seed': 2027,
        'python': sys.version, 'numpy_version': np.__version__, 'sorted_input_ids': sorted_ids,
        'permutation_output': ids, 'GRAPH_VAL': ids[:n_val], 'GRAPH_TRAIN': ids[n_val:],
        'permutation_output_sha256': digest(canonical(ids)), 'guard_counters': counters}, sort_keys=True))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('STOP FOR HUMAN REVIEW: ' + str(exc), file=sys.stderr)
        sys.exit(2)
