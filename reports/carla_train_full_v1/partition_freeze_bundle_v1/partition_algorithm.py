"""The unchanged preregistered split. Only canonical IDs enter membership computation."""
import re
from collections import Counter
import numpy as np
from partition_common import require


def validate_ids(ids, expected_n=101):
    require(len(ids) == expected_n and len(set(ids)) == len(ids), 'Scenario count/duplicate mismatch')
    require(all(isinstance(sid, str) and re.fullmatch(
        r'(Town01|Town02|Town03|Town04|Town05|Town10HD)/scenario-(0|[1-9][0-9]*)', sid) for sid in ids),
        'Noncanonical scenario ID')
    return sorted(ids)


def cal_count(n):
    return max(1, round(0.25 * n))


def compute_partition(ids):
    ordered = validate_ids(ids)
    generator = np.random.Generator(np.random.PCG64(2026))
    indices = generator.permutation(len(ordered)).tolist()
    permuted = [ordered[i] for i in indices]
    n_cal = cal_count(len(ordered))
    require(n_cal == 25 and len(ordered) - n_cal == 76, 'Unexpected FIT/CAL size')
    return {'N': len(ordered), 'n_cal': n_cal, 'n_fit': len(ordered) - n_cal,
            'seed': 2026, 'RNG': 'NumPy PCG64', 'numpy_version': np.__version__,
            'sorted_canonical_ids': ordered, 'permutation_indices': indices,
            'CAL_NORMAL': permuted[:n_cal], 'FIT_NORMAL': permuted[n_cal:]}


def town_counts(ids):
    return dict(sorted(Counter(sid.split('/')[0] for sid in ids).items()))
