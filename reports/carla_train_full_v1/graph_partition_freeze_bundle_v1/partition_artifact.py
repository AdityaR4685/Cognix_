"""Content validation and human-readable methodology, without graph execution."""
from datetime import datetime, timezone
from freeze_common import (BUNDLE, ZERO_STATE, LATER_SPEC, require, read_json, hash_file,
                           digest, canonical)
from partition_algorithm import prove


def make_artifact(derivation, evidence, bundle_seal, counters):
    return {'schema': 'immutable-experiment-2b-graph-development-partition-v1',
        'experiment': 'Experiment 2B', 'status': 'GRAPH_DEVELOPMENT_PARTITION_FROZEN',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'immutable_after_freeze': True, 'preparation_bundle_seal': bundle_seal,
        'upstream_bindings': evidence['upstream_bindings'], 'environment': evidence['environment'],
        'frozen_FIT_NORMAL': evidence['frozen_FIT_NORMAL'], 'excluded_CAL_NORMAL': evidence['excluded_CAL_NORMAL'],
        **derivation, 'later_graph_specification': LATER_SPEC, **ZERO_STATE,
        'guard_counters_at_derivation': dict(counters),
        'membership_inputs': 'Only 76 frozen canonical FIT_NORMAL IDs; CAL used only for exclusion assertions',
        'original_76_25_partition_recomputed': False, 'performance_used_for_membership': False,
        'balancing': False, 'stratification': False, 'manual_edits': False, 'seed_search': False,
        'alternative_partition': False, 'membership_selection_retries': 0,
        'split_fixed_even_if_later_graph_results_are_poor': True,
        'human_review_completed': False, 'action': 'STOP FOR HUMAN REVIEW'}


def validate_artifact(artifact, evidence, bundle_seal):
    require(artifact['schema'] == 'immutable-experiment-2b-graph-development-partition-v1' and
            artifact['experiment'] == 'Experiment 2B' and
            artifact['status'] == 'GRAPH_DEVELOPMENT_PARTITION_FROZEN' and
            artifact['immutable_after_freeze'] is True, 'Artifact schema/status mismatch')
    require(artifact['preparation_bundle_seal'] == bundle_seal and
            artifact['upstream_bindings'] == evidence['upstream_bindings'] and
            artifact['environment'] == evidence['environment'] and
            artifact['frozen_FIT_NORMAL'] == evidence['frozen_FIT_NORMAL'] and
            artifact['excluded_CAL_NORMAL'] == evidence['excluded_CAL_NORMAL'], 'Artifact upstream binding mismatch')
    require(artifact['seed'] == 2027 and artifact['bit_generator'] == 'PCG64' and
            artifact['generator_expression'] == 'numpy.random.Generator(numpy.random.PCG64(2027))' and
            artifact['generator_instances_for_membership'] == artifact['permutation_calls_for_membership'] == 1 and
            artifact['n_graph_val_formula'] == 'max(1, round(0.20 * 76))' and
            artifact['round_semantics'] == 'Python built-in ties-to-even' and
            artifact['n_graph_val'] == 15 and artifact['n_graph_train'] == 61, 'Fixed rule mismatch')
    require(artifact['proof'] == prove(artifact, evidence['frozen_FIT_NORMAL'], evidence['excluded_CAL_NORMAL']),
            'Artifact proof mismatch')
    require(artifact['sorted_input_sha256'] == digest(canonical(artifact['sorted_input_ids'])) and
            artifact['permutation_output_sha256'] == digest(canonical(artifact['permutation_output'])),
            'Input/permutation digest mismatch')
    require(artifact['later_graph_specification'] == LATER_SPEC, 'Later graph specification mismatch')
    for key, value in ZERO_STATE.items():
        require(type(artifact[key]) is type(value) and artifact[key] == value, 'Artifact zero state mismatch: ' + key)
    for key in ('original_76_25_partition_recomputed', 'performance_used_for_membership', 'balancing',
                'stratification', 'manual_edits', 'seed_search', 'alternative_partition', 'human_review_completed'):
        require(artifact[key] is False, 'Forbidden action claimed: ' + key)
    require(artifact['membership_selection_retries'] == 0 and
            artifact['split_fixed_even_if_later_graph_results_are_poor'] is True and
            artifact['action'] == 'STOP FOR HUMAN REVIEW', 'Immutability/review state mismatch')
    require(datetime.fromisoformat(artifact['created_utc']).utcoffset().total_seconds() == 0,
            'Invalid UTC creation timestamp')


def methodology(artifact):
    lines = [
        '# Experiment 2B immutable graph-development partition', '',
        'This freeze is limited to whole-scenario GRAPH_TRAIN / GRAPH_VAL membership after Gate-2 PASS. '
        'It creates no graph training data, edges, node tensors, models or optimization results.', '',
        'The authenticated original partition remains unchanged. Read its existing 76 unique canonical '
        'FIT_NORMAL IDs, sort with ordinary Python lexical string ordering, instantiate exactly one '
        '`numpy.random.Generator(numpy.random.PCG64(2027))`, and apply exactly one permutation to that list. '
        '`max(1, round(0.20 * 76))` uses Python built-in ties-to-even round and equals 15. '
        'The first 15 permuted IDs are ordered GRAPH_VAL; the remaining 61 are ordered GRAPH_TRAIN.', '',
        'The 25 CAL_NORMAL IDs are excluded. There is no retry, balancing, stratification, town balancing, '
        'seed search, manual change or model/performance-informed selection. Membership remains fixed '
        'even if later graph results are poor. TEST is prohibited.', '',
        'Membership selection has one generator and one permutation. Deterministic replay runs in a '
        'separate read-only verification process with the same fixed rule; it cannot select or publish an '
        'alternative partition. Synthetic tests use only synthetic IDs.', '',
        'Upstream payloads were independently SHA256-hashed, including all Gate-2 final runtime payloads '
        'and committed unit seals. Only FINAL status/flags, run/bundle identity metadata and the original '
        'partition membership were semantically read. Audit metrics, fitted arrays and performance '
        'summaries were not parsed. Raw hashing is integrity verification and cannot affect membership.', '',
        'The TRAIN archive identity binds the authenticated Gate-1 full SHA256 and the unchanged current '
        'path, byte count, device, inode, modification time and creation time. The 146 GB archive is not '
        'rehashed or opened in this partition-only task. This is an identity/metadata preservation check; '
        'it is not a newly measured archive-content digest.', '',
        'The later node order is Camera, IMU, Seg. The later feature order is prob_normal, epistemic, '
        'aleatoric. Conditions remain NoGraph, StandardGAT, EpistemicGAT. The epistemic prior remains '
        '`w_j = 1 / (1 + E_j)` and `e_ij^epi = e_ij - log(1 + E_j)`. These are declarations only; '
        'later graph execution requires separate human review.', '',
        'Preparation source review and synthetic tests precede sealing the preparation bundle. The '
        'current explicit user request authorizes this partition freeze. Codex procedure review is '
        'recorded as such; no completed human review is claimed. Exclusive pending-directory/file '
        'creation and Windows rename refuse existing target/pending evidence. Failures preserve partial '
        'new evidence and block retry. Exact inventories, detached seals and read-only file attributes '
        'support immutable verification; an authorized filesystem owner can still change attributes, '
        'so future use must verify externally supplied seals.', '',
        'JSON encoding is UTF-8, sorted keys, compact comma/colon separators, no nonfinite numbers and '
        'no trailing newline. `partition.json.sha256` hashes these exact bytes. `artifact_inventory.json` '
        'lists payload hashes/sizes excluding itself and seal metadata to avoid a hash cycle. '
        '`SHA256SUMS` covers every payload and the inventory itself; its detached SHA256 is the namespace '
        'seal. Preflight checks the exact file and directory sets and all historical preservation hashes.', '',
        f"Python: {artifact['environment']['python']}",
        f"NumPy: {artifact['environment']['numpy_version']}", '',
        'Proof: GRAPH_TRAIN and GRAPH_VAL are unique/disjoint; their union exactly equals frozen '
        'FIT_NORMAL (76), and both CAL intersections are empty. Counts are 61/15.', '',
        '## Exact ordered GRAPH_VAL (15)', '', *artifact['GRAPH_VAL'], '',
        '## Exact ordered GRAPH_TRAIN (61)', '', *artifact['GRAPH_TRAIN'], '',
        '## Sorted input FIT_NORMAL (76)', '', *artifact['sorted_input_ids'], '',
        '**STOP FOR HUMAN REVIEW. No graph model was executed.**', '']
    return '\n'.join(lines).encode('utf-8')
