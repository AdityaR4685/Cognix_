"""Membership uses ONLY frozen FIT IDs; CAL IDs are exclusion assertions."""
import re
from freeze_common import require, digest, canonical


def validate_input(fit, cal):
    for role, values, count in (('FIT_NORMAL', fit, 76), ('CAL_NORMAL', cal, 25)):
        require(isinstance(values, list) and len(values) == count, role + ' count mismatch')
        require(all(type(v) is str and re.fullmatch(r'Town(?:0[1-7]|10HD)/scenario-[1-9][0-9]*', v)
                    for v in values), role + ' noncanonical scenario ID')
        require(len(set(values)) == count, role + ' duplicate scenario ID')
    require(not set(fit) & set(cal), 'Frozen FIT/CAL overlap')
    return sorted(fit)


def derive(fit, cal):
    import numpy as np
    sorted_ids = validate_input(fit, cal)
    # Exactly one generator and exactly one permutation for this derivation.
    rng = np.random.Generator(np.random.PCG64(2027))
    initial_state = rng.bit_generator.state
    permuted = rng.permutation(sorted_ids).tolist()
    final_state = rng.bit_generator.state
    n_graph_val = max(1, round(0.20 * 76))
    require(n_graph_val == 15, 'Unexpected Python round result')
    positions = {v: i for i, v in enumerate(sorted_ids)}
    result = {'sorted_input_ids': sorted_ids, 'seed': 2027, 'bit_generator': 'PCG64',
              'generator_expression': 'numpy.random.Generator(numpy.random.PCG64(2027))',
              'generator_instances_for_membership': 1, 'permutation_calls_for_membership': 1,
              'permutation_output': permuted, 'permutation_indices': [positions[v] for v in permuted],
              'initial_rng_state': initial_state, 'final_rng_state': final_state,
              'n_graph_val_formula': 'max(1, round(0.20 * 76))', 'round_semantics': 'Python built-in ties-to-even',
              'n_graph_val': n_graph_val, 'n_graph_train': 76 - n_graph_val,
              'GRAPH_VAL': permuted[:n_graph_val], 'GRAPH_TRAIN': permuted[n_graph_val:],
              'sorted_input_sha256': digest(canonical(sorted_ids)),
              'permutation_output_sha256': digest(canonical(permuted))}
    result['proof'] = prove(result, fit, cal)
    return result


def prove(result, fit, cal):
    sorted_ids = validate_input(fit, cal)
    train, val = result['GRAPH_TRAIN'], result['GRAPH_VAL']
    require(len(train) == len(set(train)) == 61, 'GRAPH_TRAIN count/uniqueness mismatch')
    require(len(val) == len(set(val)) == 15, 'GRAPH_VAL count/uniqueness mismatch')
    require(not set(train) & set(val), 'GRAPH role overlap')
    require(set(train) | set(val) == set(fit), 'Graph union differs from frozen FIT_NORMAL')
    require(not (set(train) | set(val)) & set(cal), 'CAL_NORMAL leakage')
    require(result['sorted_input_ids'] == sorted_ids, 'Input ordering mismatch')
    require(result['permutation_output'] == val + train, 'Permutation/role order mismatch')
    require(result['permutation_indices'] == [sorted_ids.index(v) for v in val + train], 'Index order mismatch')
    return {'GRAPH_TRAIN_count': 61, 'GRAPH_VAL_count': 15, 'union_count': 76,
            'union_equals_frozen_FIT_NORMAL': True, 'union_sorted_ids': sorted_ids,
            'missing_FIT_NORMAL_ids': [], 'extra_graph_ids': [], 'GRAPH_TRAIN_GRAPH_VAL_intersection': [],
            'CAL_NORMAL_GRAPH_TRAIN_intersection': [], 'CAL_NORMAL_GRAPH_VAL_intersection': [],
            'zero_CAL_NORMAL_overlap': True, 'ordered_roles_equal_single_permutation': True}
