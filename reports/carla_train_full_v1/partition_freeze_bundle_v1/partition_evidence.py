"""Read-only admission of sealed complete Experiment-2B evidence before a split."""
import gzip
import hashlib
import sys
from collections import Counter
import numpy as np
from partition_common import (BUNDLE, REPO, REPORT, V8, STORE, HEAD, V8_SEAL, SOURCE,
    require, read_json, hash_file, verify_manifest, verify_source, verify_preservation, fingerprint, git)
from partition_algorithm import validate_ids


def load_v8():
    # Authenticate every imported legacy verifier before executing its module code.
    verify_manifest(V8, expected_seal=V8_SEAL)
    sys.path.append(str(V8))
    import recovery_common
    import local_inventory
    import local_extraction
    import clean_block_store
    import v7_adoption
    require(recovery_common.BUNDLE == V8, 'Unexpected legacy verifier import location')
    return recovery_common, local_inventory, local_extraction, clean_block_store, v7_adoption


def verify_bound_files(records):
    for record in records:
        path = REPO / record['path']
        require(path.stat().st_size == record['bytes'] and hash_file(path) == record['sha256'],
                'Frozen evidence mismatch: ' + record['path'])


def validate_completion(completion, state, index, completed, science, proof, old_digest):
    require(state['status'] == 'COMPLETE_GATE1' and state['gate1_complete'] is True and
            state['completed_scenarios'] == 101, 'Incomplete Gate-1')
    require(completion['schema'] == 'complete-local-Gate1-v8' and
            completion['experiment'] == state['experiment'] == 'Experiment 2B', 'Experiment/schema mismatch')
    require(completed == state['completed_blocks'] == completion['completed_blocks'], 'Block/adoption identity mismatch')
    require(len(completed) == completion['scenario_count'] == index['scenario_count'] == 101,
            'Missing/extra completed scenario')
    validate_ids([r['scenario_id'] for r in completed])
    require([r['archive_ordinal'] for r in completed] == list(range(1, 102)), 'Archive ordinal mismatch')
    require(all(r['rows'] == 2999 and r['state'] == 'COMPLETE_EXTRACTED' for r in completed) and
            sum(r['rows'] for r in completed) == completion['emitted_rows'] == 302899, 'Clean row count mismatch')
    require(index['total_tar_members'] == 606707 and all(s['emitted_rows'] == 2999 and
            s['n_ticks'] == 3000 for s in index['scenarios']), 'Inventory total mismatch')
    require(Counter(r['kind'] for r in completed) == {'V7_TO_V8_IDENTITY_ADOPTION': 68, 'NEW_V8_EXTRACTION': 33},
            'Adoption/extraction kind mismatch')
    require(completion['dimensions'] == {'camera': 18, 'seg': 29, 'imu': 10}, 'Dimensions mismatch')
    for evidence in (completion, state):
        require(evidence['partition'] is None and evidence['fitting_executed'] is False and
                evidence['pseudo_rows'] == 0 and evidence['GAT_executed'] is False and
                evidence['TEST_requests'] == evidence['network_requests'] == 0, 'Scientific/network state mismatch')
    require(state['TRAIN_network_requests'] == 0, 'TRAIN network state mismatch')
    require(completion['source_binding'] == state['source_binding'] == index['source_binding'], 'Source/store mismatch')
    inv = {k: index[k] for k in ('schema', 'ledger_sha256', 'ledger_logical_sha256',
        'scenario_index_sha256', 'scenario_count', 'total_tar_members', 'train_contract_sha256')}
    require(completion['inventory_binding'] == state['inventory_binding'] == inv, 'Inventory/store mismatch')
    require(state['scientific_extractor_hashes'] == science and
            state['v7_publication_proof_sha256'] == old_digest(proof) and
            completion['preprocessing_policy_sha256'] == state['preprocessing_policy_sha256'] ==
            science['preprocessing_policy_sha256'], 'Policy/scientific/proof mismatch')
    require(completion['scientific_limitation'] == state['scientific_limitation'], 'Amendment limitation mismatch')
    require(completion['clean_store_sha256'] == old_digest(completed) and
            completion['global_inventory_and_store_sha256'] == old_digest({'inventory': inv, 'blocks': completed}),
            'Global store digest mismatch')


def verify_global_ledger(completion, completed):
    expected = completion['complete_oov_ledger']
    path = STORE / 'complete_oov_ledger.jsonl.gz'
    require(hash_file(path) == expected['sha256'], 'Global OOV compressed digest mismatch')
    logical = hashlib.sha256()
    count = 0
    with gzip.open(path, 'rb') as global_stream:
        for block in completed:
            name = block['scenario_id'].replace('/', '__')
            with gzip.open(STORE / 'verified_clean_blocks' / name / 'oov_ledger.jsonl.gz', 'rb') as stream:
                for line in stream:
                    require(global_stream.readline() == line, 'Global OOV ledger differs from complete block ledgers')
                    logical.update(line)
                    count += 1
        require(global_stream.read() == b'', 'Extra global OOV frame')
    require(count == expected['frames'] == 303000 and logical.hexdigest() == expected['logical_sha256'],
            'Global OOV completion mismatch')
    require(expected['schema'] == 'complete-Experiment-2B-OOV-ledger-v8' and
            expected['descriptive_only'] is True and
            expected['count_remapped'] == sum(r['oov_ledger']['count_remapped'] for r in completed) and
            expected['observed_total_semantic_pixels'] == sum(r['oov_ledger']['total_semantic_pixels'] or 0 for r in completed) and
            expected['adopted_pixel_totals'] == 'unmeasured; accepted-domain zero-OOV proof only',
            'OOV summary mismatch')


def protocol_bindings(bindings):
    for record in bindings['protocols']:
        original = REPORT / record['name']
        copy = BUNDLE / record['name']
        require(original.read_bytes() == copy.read_bytes() and hash_file(original) == record['sha256'],
                'Frozen protocol changed')
    protocol = read_json(BUNDLE / 'full_train_preregistered_protocol.json')
    require(protocol['split'] == bindings['split_rule'] and protocol['experiment'] == 'COGNIX Experiment 2A',
            'Original 2A split preregistration changed')
    require(protocol['split']['seed'] == 2026 and protocol['split']['rng'] == 'NumPy PCG64' and
            protocol['split']['CAL_formula'] == 'max(1, round(0.25 * N))', 'Unreviewed split algorithm')
    return protocol


def descriptors(index):
    # Inventory contains column/row/member-hash provenance, but no weather values.
    result = {}
    import local_inventory
    for record in local_inventory.inventory_records(V8, index):
        weather = record['metadata_integrity'].get('weather.feather')
        result[record['scenario_id']] = {
            'town': record['town'], 'weather_metadata': weather,
            'weather_values': {column: None for column in weather['columns']} if weather else None,
            'weather_values_missing_reason': 'Values not retained in the sealed inventory/clean store; no source metadata replay in this partition step',
            'environment': None, 'environment_missing_reason': 'No additional environment descriptors retained in sealed evidence'}
    return result


def verify_inputs(progress=None):
    bindings = read_json(BUNDLE / 'execution_bindings.json')
    require(git('rev-parse', 'HEAD') == HEAD == bindings['HEAD'], 'HEAD mismatch')
    require(not git('diff', '--name-only') and not git('diff', '--cached', '--name-only'), 'Tracked/staged changes')
    require(np.__version__ == bindings['numpy_version'], 'NumPy version mismatch')
    protocol = protocol_bindings(bindings)
    legacy, inventory, extraction, blocks, adoption = load_v8()
    legacy.verify_science()
    verify_bound_files(bindings['files'])
    binding = extraction.initial_binding()
    require(binding['identity'] == SOURCE == bindings['source_identity'], 'Source binding mismatch')
    measured = verify_source(SOURCE, binding, progress)
    index = inventory.verify_inventory(V8, measured)
    science = blocks.scientific_hashes()
    proof = adoption.historical_proof()
    adoption.audit_candidates(index, measured, proof)
    state = extraction.read_progress(STORE)
    require(state is not None, 'Missing Gate-1 state')
    completed = extraction.verify_store(STORE, index, measured, science, state, proof)
    completion = read_json(STORE / 'complete_inventory_and_clean_store.json')
    validate_completion(completion, state, index, completed, science, proof, inventory.digest)
    verify_manifest(STORE, 'complete_Gate1_SHA256SUMS', bindings['v8_completion_seal'], exact=False)
    verify_global_ledger(completion, completed)
    preservation = verify_preservation(read_json(BUNDLE / 'historical_preservation_baseline.json'))
    verify_bound_files(bindings['files'])
    require(fingerprint(SOURCE['path']) == binding['fingerprint'] and
            git('rev-parse', 'HEAD') == HEAD and not git('diff', '--name-only') and
            not git('diff', '--cached', '--name-only'), 'Source/HEAD/tracked state changed during verification')
    return {'bindings': bindings, 'protocol': protocol, 'index': index, 'science': science,
            'completion': completion, 'completed': completed, 'descriptors': descriptors(index),
            'preservation': preservation}


def preflight_summary(evidence, target_state, counters):
    return {'HEAD': HEAD, 'experiment': 'Experiment 2B', 'v8_seal_verified': V8_SEAL,
            'source_identity_verified': evidence['bindings']['source_identity'],
            'completed_v8_scenarios': 101, 'total_rows': 302899,
            'existing_real_partition': target_state, 'fitting_executed': False, 'pseudo_rows': 0,
            'GAT_executed': False, 'graph_constructed': False, 'calibration_executed': False,
            'TEST_requests': counters['TEST_requests'], 'network_requests': counters['network_requests'],
            'expected_N': 101, 'expected_CAL': 25, 'expected_FIT': 76, 'split_seed': 2026,
            'RNG': 'PCG64', 'numpy_version': np.__version__, 'guard_counters': counters,
            'global_inventory_and_store_sha256': evidence['completion']['global_inventory_and_store_sha256'],
            'complete_oov_ledger': evidence['completion']['complete_oov_ledger'],
            'historical_preservation': evidence['preservation'],
            'DATA_INTEGRITY_GATE': 'AWAITING_AUTHORIZED_PARTITION_FREEZE' if target_state == 'absent' else 'FROZEN_PARTITION_VERIFIED',
            'action': 'STOP FOR HUMAN REVIEW'}
