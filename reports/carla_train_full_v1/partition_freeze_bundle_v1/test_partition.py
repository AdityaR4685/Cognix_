"""Synthetic fixtures only. No real source, TEST, network or historical writes."""
import ast
import copy
import datetime
import gzip
import inspect
import json
import os
import unittest
from pathlib import Path
from unittest import mock
import numpy as np
import partition_algorithm as algorithm
import partition_artifact as artifact
import partition_common as common
import partition_evidence as admission
import partition_freeze as execution
import partition_preflight as preflight
from synthetic_fixtures import SyntheticDirectory

FIXTURE_ROOT = None


def old_digest(value):
    # Legacy v8 digests deliberately retain their ORIGINAL indented encoding.
    return common.sha_bytes(common.pretty_bytes(value))


def fixture():
    ids = ['Town01/scenario-' + str(i) for i in range(101)]
    records = [{'scenario_id': sid, 'archive_ordinal': i + 1, 'rows': 2999,
                'state': 'COMPLETE_EXTRACTED',
                'kind': 'V7_TO_V8_IDENTITY_ADOPTION' if i < 68 else 'NEW_V8_EXTRACTION',
                'manifest_sha256': common.sha_bytes(('manifest' + sid).encode()),
                'npz_sha256': common.sha_bytes(('npz' + sid).encode()),
                'array_content_sha256': common.sha_bytes(('array' + sid).encode()),
                'oov_ledger': {'count_remapped': 0, 'total_semantic_pixels': None}}
               for i, sid in enumerate(ids)]
    index_records = [{'scenario_id': sid, 'archive_ordinal': i + 1, 'town': 'Town01',
                      'emitted_rows': 2999, 'n_ticks': 3000, 'member_count': 6007,
                      'record_sha256': common.sha_bytes(('record' + sid).encode()),
                      'source_member_set_sha256': common.sha_bytes(('members' + sid).encode()),
                      'parser_member_chain_sha256': common.sha_bytes(('chain' + sid).encode())}
                     for i, sid in enumerate(ids)]
    inv = {'schema': 'frozen-local-TRAIN-inventory-v7', 'ledger_sha256': 'a' * 64,
           'ledger_logical_sha256': 'b' * 64, 'scenario_index_sha256': 'c' * 64,
           'scenario_count': 101, 'total_tar_members': 606707, 'train_contract_sha256': 'd' * 64}
    source = {'identity': {'path': 'SYNTHETIC_TRAIN'}, 'fingerprint': {}}
    science = {'preprocessing_policy_sha256': 'e' * 64, 'source_files': [],
               'frozen_segmentation_function_sha256': 'f' * 64}
    proof = {'synthetic_preserved_v7_proof': True}
    state = {'status': 'COMPLETE_GATE1', 'gate1_complete': True, 'completed_scenarios': 101,
             'completed_blocks': records, 'experiment': 'Experiment 2B', 'source_binding': source,
             'inventory_binding': inv, 'scientific_extractor_hashes': science,
             'v7_publication_proof_sha256': old_digest(proof),
             'preprocessing_policy_sha256': 'e' * 64, 'scientific_limitation': 'synthetic',
             'partition': None, 'fitting_executed': False, 'pseudo_rows': 0,
             'GAT_executed': False, 'TEST_requests': 0, 'network_requests': 0, 'TRAIN_network_requests': 0}
    completion = {k: copy.deepcopy(v) for k, v in state.items() if k not in
                  ('status', 'gate1_complete', 'completed_scenarios', 'scientific_extractor_hashes')}
    completion.update(schema='complete-local-Gate1-v8', scenario_count=101, emitted_rows=302899,
                      dimensions={'camera': 18, 'seg': 29, 'imu': 10},
                      complete_oov_ledger={'schema': 'synthetic-complete-OOV', 'sha256': '9' * 64,
                                           'logical_sha256': '8' * 64, 'frames': 303000},
                      clean_store_sha256=old_digest(records),
                      global_inventory_and_store_sha256=old_digest({'inventory': inv, 'blocks': records}))
    index = dict(inv, scenarios=index_records, source_binding=source)
    protocol = {'schema_version': 1, 'experiment': 'COGNIX Experiment 2A',
                'split': {'seed': 2026, 'rng': 'NumPy PCG64',
                          'steps': ['synthetic exact rule'], 'CAL_formula': 'max(1, round(0.25 * N))'}}
    evidence = {'index': index, 'completed': records, 'completion': completion,
                'science': science, 'protocol': protocol,
                'bindings': {'source_identity': source['identity'],
                             'protocols': [{'name': 'full_train_preregistered_protocol.json', 'sha256': '1' * 64}]},
                'descriptors': {sid: {'town': 'Town01', 'weather_values': None,
                                     'environment': None} for sid in ids}, 'preservation': {'synthetic': True}}
    return ids, evidence, state, proof


class SyntheticBase(unittest.TestCase):
    def setUp(self):
        self.ids, self.evidence, self.state, self.proof = fixture()
        self.temp = SyntheticDirectory(FIXTURE_ROOT)
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def validate(self):
        e = self.evidence
        admission.validate_completion(e['completion'], self.state, e['index'], e['completed'],
                                      e['science'], self.proof, old_digest)

    def runtime_fixture(self):
        source = self.root / 'synthetic_bundle'
        source.mkdir()
        raw = gzip.compress(b'{"synthetic_member_sha256":"' + b'0' * 64 + b'"}\n', mtime=0)
        (source / 'inventory_scenarios.jsonl.gz').write_bytes(raw)
        self.evidence['index']['ledger_sha256'] = common.sha_bytes(raw)
        patch = mock.patch.object(artifact, 'V8', source)
        patch.start()
        self.addCleanup(patch.stop)
        return self.root / 'synthetic_partition'


class SplitTests(SyntheticBase):
    def test_N101_CAL25_FIT76(self):
        r = algorithm.compute_partition(self.ids)
        self.assertEqual((r['N'], r['n_cal'], r['n_fit']), (101, 25, 76))

    def test_python_ties_to_even(self):
        self.assertEqual([algorithm.cal_count(n) for n in (1, 2, 6, 10, 14, 18, 101)], [1, 1, 2, 2, 4, 4, 25])
        self.assertEqual(round(2.5), 2)
        self.assertEqual(round(3.5), 4)

    def test_exact_lexical_sorting(self):
        r = algorithm.compute_partition(list(reversed(self.ids)))
        self.assertEqual(r['sorted_canonical_ids'], sorted(self.ids))
        self.assertLess(r['sorted_canonical_ids'].index('Town01/scenario-10'),
                        r['sorted_canonical_ids'].index('Town01/scenario-2'))

    def test_one_PCG64_generator_one_permutation(self):
        real_generator = np.random.Generator
        real_bitgen = np.random.PCG64
        calls = []
        class Tracked:
            def __init__(self, bitgen):
                calls.append('generator')
                self.generator = real_generator(bitgen)
            def permutation(self, n):
                calls.append(('permutation', n))
                return self.generator.permutation(n)
        with mock.patch.object(np.random, 'Generator', side_effect=Tracked) as generator, \
             mock.patch.object(np.random, 'PCG64', wraps=real_bitgen) as pcg:
            algorithm.compute_partition(self.ids)
        pcg.assert_called_once_with(2026)
        self.assertEqual(generator.call_count, 1)
        self.assertEqual(calls, ['generator', ('permutation', 101)])

    def test_deterministic_repeated_synthetic_computation(self):
        self.assertEqual(algorithm.compute_partition(self.ids), algorithm.compute_partition(self.ids))

    def test_input_order_does_not_affect_membership(self):
        self.assertEqual(algorithm.compute_partition(self.ids), algorithm.compute_partition(self.ids[::2] + self.ids[1::2]))

    def test_preregistered_numpy_reference(self):
        expected = np.random.Generator(np.random.PCG64(2026)).permutation(sorted(self.ids)).tolist()
        r = algorithm.compute_partition(self.ids)
        self.assertEqual(r['CAL_NORMAL'] + r['FIT_NORMAL'], expected)

    def test_disjoint_FIT_CAL(self):
        r = algorithm.compute_partition(self.ids)
        self.assertFalse(set(r['FIT_NORMAL']) & set(r['CAL_NORMAL']))

    def test_union_all_accepted(self):
        r = algorithm.compute_partition(self.ids)
        self.assertEqual(set(r['CAL_NORMAL']) | set(r['FIT_NORMAL']), set(self.ids))

    def test_no_duplicate_output(self):
        r = algorithm.compute_partition(self.ids)
        self.assertEqual(len(set(r['CAL_NORMAL'] + r['FIT_NORMAL'])), 101)
        self.assertEqual(sorted(r['permutation_indices']), list(range(101)))

    def test_duplicate_input_stops(self):
        with self.assertRaises(common.ReviewRequired):
            algorithm.compute_partition(self.ids[:-1] + [self.ids[0]])

    def test_omitted_ID_stops(self):
        with self.assertRaises(common.ReviewRequired):
            algorithm.compute_partition(self.ids[:-1])

    def test_extra_ID_stops(self):
        with self.assertRaises(common.ReviewRequired):
            algorithm.compute_partition(self.ids + ['Town02/scenario-1'])

    def test_alias_noncanonical_ID_stops(self):
        for alias in ('town01/scenario-1', 'Town01/scenario-01', 'train/Town01/scenario-1', 'Town01/scenario-1 '):
            with self.subTest(alias=alias), self.assertRaises(common.ReviewRequired):
                algorithm.compute_partition(self.ids[:-1] + [alias])

    def test_no_historical_membership_input(self):
        self.assertEqual(list(inspect.signature(algorithm.compute_partition).parameters), ['ids'])
        with self.assertRaises(TypeError):
            algorithm.compute_partition(self.ids, historical_membership=self.ids[:20])

    def test_no_town_weather_OOV_performance_balancing(self):
        split = algorithm.compute_partition(self.ids)
        changed = copy.deepcopy(self.evidence)
        for i, sid in enumerate(self.ids):
            changed['descriptors'][sid].update(weather_values={'rain': i}, performance=-i, OOV=i * 999)
        a = artifact.build_record(self.evidence, split, '2026-10-07T00:00:00+00:00', '2' * 64)
        b = artifact.build_record(changed, split, '2026-10-07T00:00:00+00:00', '2' * 64)
        self.assertEqual(a['CAL_NORMAL'], b['CAL_NORMAL'])
        self.assertEqual(a['FIT_NORMAL'], b['FIT_NORMAL'])
        self.assertEqual(a['town_counts'], b['town_counts'])


class IntegrityTests(SyntheticBase):
    def test_valid_synthetic_completion(self):
        self.validate()

    def test_source_SHA_mismatch_stops(self):
        path = self.root / 'synthetic_train.bin'
        path.write_bytes(b'synthetic TRAIN bytes')
        identity = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': '0' * 64}
        bound = {'identity': identity, 'fingerprint': common.fingerprint(path),
                 'independently_measured_full_sha256': '0' * 64}
        with self.assertRaises(common.ReviewRequired):
            common.verify_source(identity, bound)

    def test_source_size_mismatch_stops(self):
        path = self.root / 'synthetic_train.bin'
        path.write_bytes(b'abcd')
        identity = {'path': str(path), 'bytes': 5, 'sha256': common.sha_bytes(b'abcd')}
        with self.assertRaises(common.ReviewRequired):
            common.verify_source(identity, {'identity': identity, 'fingerprint': common.fingerprint(path)})

    def test_source_identity_verifies_full_bytes(self):
        path = self.root / 'synthetic_train.bin'
        raw = b'synthetic TRAIN bytes'
        path.write_bytes(raw)
        identity = {'path': str(path), 'bytes': len(raw), 'sha256': common.sha_bytes(raw)}
        bound = {'identity': identity, 'fingerprint': common.fingerprint(path),
                 'independently_measured_full_sha256': identity['sha256']}
        self.assertEqual(common.verify_source(identity, bound), bound)

    def test_HEAD_mismatch_stops_before_source_access(self):
        with mock.patch.object(admission, 'read_json', return_value={'HEAD': common.HEAD}), \
             mock.patch.object(admission, 'git', return_value='wrong_HEAD'), \
             mock.patch.object(admission, 'verify_source') as source:
            with self.assertRaises(common.ReviewRequired):
                admission.verify_inputs()
            source.assert_not_called()

    def test_numpy_version_mismatch_stops_before_source_access(self):
        bindings = {'HEAD': common.HEAD, 'numpy_version': 'synthetic_wrong_version'}
        with mock.patch.object(admission, 'read_json', return_value=bindings), \
             mock.patch.object(admission, 'git', side_effect=[common.HEAD, '', '']), \
             mock.patch.object(admission, 'verify_source') as source:
            with self.assertRaises(common.ReviewRequired):
                admission.verify_inputs()
            source.assert_not_called()

    def test_incomplete_or_changed_global_OOV_ledger_stops(self):
        runtime = self.root / 'synthetic_v8'
        block = runtime / 'verified_clean_blocks' / 'Town01__scenario-0'
        block.mkdir(parents=True)
        (block / 'oov_ledger.jsonl.gz').write_bytes(gzip.compress(b'expected-synthetic-frame\n', mtime=0))
        global_raw = gzip.compress(b'different-synthetic-frame\n', mtime=0)
        (runtime / 'complete_oov_ledger.jsonl.gz').write_bytes(global_raw)
        completion = {'complete_oov_ledger': {'sha256': common.sha_bytes(global_raw)}}
        with mock.patch.object(admission, 'STORE', runtime), self.assertRaises(common.ReviewRequired):
            admission.verify_global_ledger(completion, [{'scenario_id': 'Town01/scenario-0'}])

    def test_v8_expected_seal_mismatch_stops(self):
        (self.root / 'dummy.txt').write_bytes(b'synthetic sealed v8')
        raw = (common.hash_file(self.root / 'dummy.txt') + '  dummy.txt\n').encode()
        (self.root / 'SHA256SUMS').write_bytes(raw)
        (self.root / 'SHA256SUMS.sha256').write_bytes((common.sha_bytes(raw) + '  SHA256SUMS\n').encode())
        with self.assertRaises(common.ReviewRequired):
            common.verify_manifest(self.root, expected_seal='0' * 64)

    def test_sealed_file_tampering_stops(self):
        path = self.root / 'manifest.json'
        path.write_bytes(b'original')
        record = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': common.hash_file(path)}
        path.write_bytes(b'tampered')
        with self.assertRaises(common.ReviewRequired):
            admission.verify_bound_files([record])

    def test_block_mismatch_stops(self):
        self.evidence['completed'] = copy.deepcopy(self.evidence['completed'])
        self.evidence['completed'][100]['manifest_sha256'] = '0' * 64
        with self.assertRaises(common.ReviewRequired):
            self.validate()

    def test_adoption_mismatch_stops(self):
        self.evidence['completed'] = copy.deepcopy(self.evidence['completed'])
        self.evidence['completed'][0]['npz_sha256'] = '0' * 64
        with self.assertRaises(common.ReviewRequired):
            self.validate()

    def test_incomplete_Gate1_stops(self):
        self.state['status'] = 'EXTRACTING'
        with self.assertRaises(common.ReviewRequired):
            self.validate()

    def test_rows_mismatch_stops(self):
        self.evidence['completion']['emitted_rows'] -= 1
        with self.assertRaises(common.ReviewRequired):
            self.validate()

    def test_ordinal_mismatch_stops(self):
        self.evidence['completed'][5]['archive_ordinal'] = 99
        with self.assertRaises(common.ReviewRequired):
            self.validate()

    def test_policy_mismatch_stops(self):
        self.evidence['completion']['preprocessing_policy_sha256'] = '0' * 64
        with self.assertRaises(common.ReviewRequired):
            self.validate()

    def test_dimensions_mismatch_stops(self):
        self.evidence['completion']['dimensions']['seg'] = 30
        with self.assertRaises(common.ReviewRequired):
            self.validate()

    def test_scientific_digest_mismatch_stops(self):
        self.evidence['completion']['clean_store_sha256'] = '0' * 64
        with self.assertRaises(common.ReviewRequired):
            self.validate()

    def test_fitting_pseudo_GAT_TEST_network_states_stop(self):
        for key, value in (('fitting_executed', True), ('pseudo_rows', 1), ('GAT_executed', True),
                           ('TEST_requests', 1), ('network_requests', 1), ('partition', {})):
            old = self.evidence['completion'][key]
            self.evidence['completion'][key] = value
            with self.subTest(key=key), self.assertRaises(common.ReviewRequired):
                self.validate()
            self.evidence['completion'][key] = old


class ArtifactTests(SyntheticBase):
    def test_canonical_json_bytes_deterministic(self):
        self.assertEqual(common.canonical_bytes({'b': [1, 2], 'a': 3}), b'{"a":3,"b":[1,2]}')
        self.assertEqual(common.canonical_bytes({'a': 3, 'b': [1, 2]}), b'{"a":3,"b":[1,2]}')

    def test_canonical_json_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            common.canonical_bytes({'value': float('nan')})

    def test_partition_detached_SHA_verifies(self):
        expected = {'synthetic': True}
        raw = common.canonical_bytes(expected)
        sha = common.sha_bytes(raw)
        self.assertEqual(artifact.verify_partition_bytes(raw, expected, (sha + '  partition.json\n').encode()), sha)

    def test_partition_detached_SHA_mismatch_stops(self):
        with self.assertRaises(common.ReviewRequired):
            artifact.verify_partition_bytes(b'{}', {}, b'0  partition.json\n')

    def test_semantic_equal_noncanonical_bytes_stop(self):
        raw = b'{ "synthetic": true }\n'
        with self.assertRaises(common.ReviewRequired):
            artifact.verify_partition_bytes(raw, {'synthetic': True},
                (common.sha_bytes(raw) + '  partition.json\n').encode())

    def test_existing_partition_mismatch_never_overwritten(self):
        target = self.runtime_fixture()
        artifact.freeze(self.evidence, target, '2' * 64)
        (target / 'partition.json').write_bytes(b'{"tampered":true}')
        before = {p.name: p.read_bytes() for p in target.iterdir()}
        with self.assertRaises(common.ReviewRequired):
            artifact.freeze(self.evidence, target, '2' * 64)
        self.assertEqual(before, {p.name: p.read_bytes() for p in target.iterdir()})

    def test_existing_same_partition_byte_identical(self):
        target = self.runtime_fixture()
        first = artifact.freeze(self.evidence, target, '2' * 64)
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in target.iterdir()}
        second = artifact.freeze(self.evidence, target, '2' * 64)
        self.assertEqual(first['partition_sha256'], second['partition_sha256'])
        self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in target.iterdir()})

    def test_semantically_changed_membership_stops_even_with_rehashed_runtime(self):
        target = self.runtime_fixture()
        artifact.freeze(self.evidence, target, '2' * 64)
        record = common.read_json(target / 'partition.json')
        record['CAL_NORMAL'][0], record['FIT_NORMAL'][0] = record['FIT_NORMAL'][0], record['CAL_NORMAL'][0]
        raw = common.canonical_bytes(record)
        (target / 'partition.json').write_bytes(raw)
        (target / 'partition.json.sha256').write_bytes((common.sha_bytes(raw) + '  partition.json\n').encode())
        listing = ''.join(common.hash_file(p) + '  ' + p.name + '\n' for p in sorted(target.iterdir())
                          if p.name not in ('SHA256SUMS', 'SHA256SUMS.sha256')).encode()
        (target / 'SHA256SUMS').write_bytes(listing)
        (target / 'SHA256SUMS.sha256').write_bytes((common.sha_bytes(listing) + '  SHA256SUMS\n').encode())
        with self.assertRaisesRegex(common.ReviewRequired, 'byte/semantic mismatch'):
            artifact.freeze(self.evidence, target, '2' * 64)

    def test_new_freeze_one_permutation_including_readbacks(self):
        target = self.runtime_fixture()
        with mock.patch.object(artifact, 'compute_partition', wraps=algorithm.compute_partition) as split:
            artifact.freeze(self.evidence, target, '2' * 64)
            split.assert_called_once()

    def test_existing_freeze_one_permutation(self):
        target = self.runtime_fixture()
        artifact.freeze(self.evidence, target, '2' * 64)
        with mock.patch.object(artifact, 'compute_partition', wraps=algorithm.compute_partition) as split:
            artifact.freeze(self.evidence, target, '2' * 64)
            split.assert_called_once()

    def test_runtime_exact_member_evidence_copy(self):
        target = self.runtime_fixture()
        artifact.freeze(self.evidence, target, '2' * 64)
        self.assertEqual((target / artifact.MEMBER_FILE).read_bytes(),
                         (artifact.V8 / 'inventory_scenarios.jsonl.gz').read_bytes())

    def test_no_DATA_GATE_PASS_before_publication(self):
        target = self.runtime_fixture()
        with mock.patch.object(artifact.os, 'rename', side_effect=OSError('synthetic publication failure')):
            with self.assertRaises(OSError):
                artifact.freeze(self.evidence, target, '2' * 64)
        self.assertFalse(target.exists())
        pending = list(target.parent.glob('.' + target.name + '.pending-*'))
        self.assertEqual(len(pending), 1)
        self.assertTrue((pending[0] / 'partition.json').is_file())
        with self.assertRaises(common.ReviewRequired):
            artifact.check_target_namespace(target)

    def test_partial_target_stops(self):
        target = self.root / 'synthetic_partition'
        target.mkdir()
        with self.assertRaises(common.ReviewRequired):
            artifact.freeze(self.evidence, target, '2' * 64)
        self.assertEqual(list(target.iterdir()), [])

    def test_experiment2B_and_null_descriptors(self):
        split = algorithm.compute_partition(self.ids)
        record = artifact.build_record(self.evidence, split, '2026-10-07T00:00:00+00:00', '2' * 64)
        self.assertEqual(record['experiment'], 'Experiment 2B')
        self.assertEqual(record['split_rule_version']['original_experiment'], 'COGNIX Experiment 2A')
        self.assertIsNone(record['scenario_records_in_archive_order'][0]['descriptors']['weather_values'])

    def test_gate2_separate_authorization_and_2C_repair(self):
        record = artifact.gate_record('1' * 64, '2' * 64)
        self.assertIn('Separate human authorization', record['Gate_2_authorization'])
        self.assertIn('Experiment 2C', record['future_scientific_repair'])
        self.assertEqual(record['action'], 'STOP FOR HUMAN REVIEW')

    def test_duplicate_JSON_keys_rejected(self):
        path = self.root / 'synthetic.json'
        path.write_bytes(b'{"key":1,"key":2}')
        with self.assertRaises(common.ReviewRequired):
            common.read_json(path)


class ScopeTests(SyntheticBase):
    def test_missing_authorization_stops_before_inputs(self):
        with mock.patch.object(execution, 'verify_inputs') as inputs:
            with self.assertRaises(common.ReviewRequired):
                execution.run('WRONG_TOKEN', '2' * 64)
            inputs.assert_not_called()

    def test_absent_preflight_never_computes_partition(self):
        target = self.root / 'synthetic_partition'
        counters = {k: 0 for k in ('TEST_requests', 'network_requests')}
        with mock.patch.object(preflight, 'install_guard', return_value=counters), \
             mock.patch.object(preflight, 'verify_manifest'), \
             mock.patch.object(preflight, 'verify_inputs', return_value=self.evidence), \
             mock.patch.object(preflight, 'TARGET', target), \
             mock.patch.object(artifact, 'compute_partition', side_effect=AssertionError('real split computed')):
            result = preflight.run('2' * 64)
        self.assertEqual(result['existing_real_partition'], 'absent')
        self.assertFalse(target.exists())

    def test_readonly_guard_rejects_write(self):
        guard, counters = common.make_guard()
        with self.assertRaises(common.ReviewRequired):
            guard('open', (str(self.root / 'file'), 'w', os.O_WRONLY | os.O_CREAT))
        self.assertEqual(counters['blocked_mutation_attempts'], 1)

    def test_v7_v8_mutations_rejected(self):
        guard, counters = common.make_guard(writable=(common.TARGET,))
        for path in (common.STORE / 'state.json', common.V8 / 'README.md',
                     common.REPORT / 'gate1_local_extraction_v7' / 'active.lock'):
            for event, args in (('open', (str(path), 'w', os.O_WRONLY)), ('os.remove', (str(path), -1)),
                                ('os.utime', (str(path), None, None)), ('os.rename', (str(path), str(self.root / 'new'), -1, -1))):
                with self.subTest(path=path, event=event), self.assertRaises(common.ReviewRequired):
                    guard(event, args)

    def test_TEST_access_blocked_without_open(self):
        guard, counters = common.make_guard()
        with self.assertRaises(common.ReviewRequired):
            guard('open', (str(self.root / 'test' / 'normal' / 'data.bin'), 'r', 0))
        self.assertEqual(counters['TEST_requests'], 0)
        self.assertEqual(counters['blocked_TEST_attempts'], 1)

    def test_network_blocked_without_request(self):
        guard, counters = common.make_guard()
        with self.assertRaises(common.ReviewRequired):
            guard('socket.connect', (None, ('synthetic.invalid', 443)))
        self.assertEqual(counters['network_requests'], 0)
        self.assertEqual(counters['blocked_network_attempts'], 1)

    def test_source_open_forbidden_in_synthetic_tests(self):
        guard, counters = common.make_guard()
        with self.assertRaises(common.ReviewRequired):
            guard('open', (common.SOURCE['path'], 'rb', 0))

    def test_freeze_guard_rejects_historical_write(self):
        guard, counters = execution.make_freeze_guard()
        with self.assertRaises(common.ReviewRequired):
            guard('open', (str(common.STORE / 'state.json'), 'w', os.O_WRONLY))

    def test_no_fitting(self):
        self.assertFalse(artifact.build_record(self.evidence, algorithm.compute_partition(self.ids),
                         '2026-10-07T00:00:00+00:00', '2' * 64)['fitting_executed'])

    def test_no_pseudo(self):
        self.assertEqual(artifact.build_record(self.evidence, algorithm.compute_partition(self.ids),
                         '2026-10-07T00:00:00+00:00', '2' * 64)['pseudo_rows'], 0)

    def test_no_graph_or_GAT(self):
        record = artifact.build_record(self.evidence, algorithm.compute_partition(self.ids),
                                       '2026-10-07T00:00:00+00:00', '2' * 64)
        self.assertFalse(record['graph_constructed'])
        self.assertFalse(record['GAT_executed'])

    def test_no_model_network_TEST_imports_or_calls(self):
        forbidden_calls = {'fit', 'fit_transform', 'calibrate', 'train_graph', 'generate_pseudo_anomalies',
                           'precompute_pseudo', 'build_train_cache', 'partition_train_scenarios'}
        forbidden_imports = ('torch', 'sklearn', 'requests', 'urllib', 'http', 'socket', 'cognix.graph', 'cognix.adapters')
        for name in ('partition_algorithm.py', 'partition_artifact.py', 'partition_common.py',
                     'partition_evidence.py', 'partition_preflight.py', 'partition_freeze.py'):
            tree = ast.parse((common.BUNDLE / name).read_text())
            calls = {getattr(n.func, 'id', getattr(n.func, 'attr', None)) for n in ast.walk(tree) if isinstance(n, ast.Call)}
            self.assertFalse(calls & forbidden_calls, name)
            imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)] + \
                      [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
            self.assertFalse(any(n and n.startswith(forbidden_imports) for n in imports), name)

    def test_synthetic_evidence_not_modified_by_publication(self):
        target = self.runtime_fixture()
        before = copy.deepcopy(self.evidence)
        source_bytes = (artifact.V8 / 'inventory_scenarios.jsonl.gz').read_bytes()
        artifact.freeze(self.evidence, target, '2' * 64)
        self.assertEqual(self.evidence, before)
        self.assertEqual((artifact.V8 / 'inventory_scenarios.jsonl.gz').read_bytes(), source_bytes)
