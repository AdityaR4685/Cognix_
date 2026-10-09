"""Synthetic/unit checks only. No real archive, models, metrics, TEST or network."""
import copy
import json
import os
import sys
import unittest
from unittest import mock
from pathlib import Path

from freeze_common import (BUNDLE, FreezeError, ZERO_STATE, canonical, digest, make_guard,
                           assert_absent, write_new, verify_seal, seal_tree, verify_git_preserved, TARGET, REPO)
from partition_algorithm import derive, prove, validate_input
from verify_upstream import validate_final
from partition_artifact import make_artifact, validate_artifact


def fixture():
    return ([f'Town01/scenario-{i}' for i in range(1, 77)],
            [f'Town02/scenario-{i}' for i in range(1, 26)])


class PartitionTests(unittest.TestCase):
    def setUp(self):
        self.fit, self.cal = fixture()

    def test_counts_and_python_round(self):
        d = derive(self.fit, self.cal)
        self.assertEqual((len(d['GRAPH_TRAIN']), len(d['GRAPH_VAL'])), (61, 15))
        self.assertEqual(max(1, round(0.20 * 76)), 15)
        self.assertEqual(round(2.5), 2)

    def test_lexical_input_not_numeric_order(self):
        d = derive(self.fit[::-1], self.cal)
        self.assertEqual(d['sorted_input_ids'][:5], ['Town01/scenario-1', 'Town01/scenario-10',
            'Town01/scenario-11', 'Town01/scenario-12', 'Town01/scenario-13'])

    def test_determinism_and_input_order_invariance(self):
        self.assertEqual(derive(self.fit, self.cal), derive(self.fit[::-1], self.cal[::-1]))

    def test_exact_one_generator_one_permutation_fixed_seed(self):
        import numpy as np
        real_pcg, real_generator = np.random.PCG64, np.random.Generator
        calls = []
        class Spy:
            def __init__(self, bit):
                calls.append(('Generator',))
                self.inner = real_generator(bit)
                self.bit_generator = self.inner.bit_generator
            def permutation(self, ids):
                calls.append(('permutation', list(ids)))
                return self.inner.permutation(ids)
        def pcg(seed):
            calls.append(('PCG64', seed))
            return real_pcg(seed)
        with mock.patch.object(np.random, 'PCG64', pcg), mock.patch.object(np.random, 'Generator', Spy):
            d = derive(self.fit, self.cal)
        self.assertEqual([v[0] for v in calls], ['PCG64', 'Generator', 'permutation'])
        self.assertEqual(calls[0][1], 2027)
        self.assertEqual(calls[2][1], sorted(self.fit))
        self.assertEqual(d['permutation_output'], d['GRAPH_VAL'] + d['GRAPH_TRAIN'])

    def test_roles_disjoint_union_exact_no_cal(self):
        d = derive(self.fit, self.cal)
        train, val = set(d['GRAPH_TRAIN']), set(d['GRAPH_VAL'])
        self.assertFalse(train & val)
        self.assertEqual(train | val, set(self.fit))
        self.assertFalse((train | val) & set(self.cal))

    def test_wrong_input_count_rejected(self):
        with self.assertRaises(FreezeError): validate_input(self.fit[:-1], self.cal)

    def test_duplicate_fit_rejected(self):
        with self.assertRaises(FreezeError): validate_input(self.fit[:-1] + [self.fit[0]], self.cal)

    def test_noncanonical_ids_rejected(self):
        for bad in ('Town01/scenario-01', 'Town01\\scenario-1', '../TEST/scenario-1',
                    'Town01/scenario-0', 'Town08/scenario-1', 1):
            with self.subTest(bad=bad), self.assertRaises(FreezeError):
                validate_input([bad] + self.fit[1:], self.cal)

    def test_duplicate_cal_rejected(self):
        with self.assertRaises(FreezeError): validate_input(self.fit, self.cal[:-1] + [self.cal[0]])

    def test_fit_cal_leakage_rejected(self):
        with self.assertRaises(FreezeError): validate_input(self.fit, [self.fit[0]] + self.cal[1:])

    def test_mutated_membership_rejected(self):
        d = derive(self.fit, self.cal)
        d['GRAPH_TRAIN'][0] = self.cal[0]
        with self.assertRaises(FreezeError): prove(d, self.fit, self.cal)

    def test_mutated_role_order_rejected(self):
        d = derive(self.fit, self.cal)
        d['GRAPH_VAL'] = d['GRAPH_VAL'][::-1]
        with self.assertRaises(FreezeError): prove(d, self.fit, self.cal)

    def test_changed_input_sort_rejected(self):
        d = derive(self.fit, self.cal)
        d['sorted_input_ids'] = d['sorted_input_ids'][::-1]
        with self.assertRaises(FreezeError): prove(d, self.fit, self.cal)

    def test_missing_fit_union_rejected(self):
        d = derive(self.fit, self.cal)
        d['GRAPH_TRAIN'][0] = d['GRAPH_VAL'][0]
        with self.assertRaises(FreezeError): prove(d, self.fit, self.cal)

    def test_permutation_indices_rejected_if_changed(self):
        d = derive(self.fit, self.cal)
        d['permutation_indices'] = d['permutation_indices'][::-1]
        with self.assertRaises(FreezeError): prove(d, self.fit, self.cal)

    def test_status_all_valid_and_zero_flags(self):
        final = dict(ZERO_STATE, status='TRAIN_ONLY_SCIENTIFIC_HEALTH_PASS',
                     modality_status={'Camera': 'VALID', 'IMU': 'VALID', 'Seg': 'VALID'})
        validate_final(final)
        for key, value in {'status': 'FAIL', 'TEST_requests': 1, 'network_requests': 1,
                           'graph_constructed': True, 'GAT_executed': True, 'TEST_requests_type': False}.items():
            bad = copy.deepcopy(final)
            bad['TEST_requests' if key == 'TEST_requests_type' else key] = value
            with self.subTest(key=key), self.assertRaises(FreezeError): validate_final(bad)
        for modality in ('Camera', 'IMU', 'Seg'):
            bad = copy.deepcopy(final)
            bad['modality_status'][modality] = 'INVALID'
            with self.subTest(modality=modality), self.assertRaises(FreezeError): validate_final(bad)

    def test_artifact_upstream_tamper_rejected(self):
        evidence = {'upstream_bindings': {'HEAD': 'synthetic'}, 'environment': {'python': 'synthetic'},
                    'frozen_FIT_NORMAL': self.fit, 'excluded_CAL_NORMAL': self.cal}
        d = derive(self.fit, self.cal)
        a = make_artifact(d, evidence, 'synthetic-seal', dict(ZERO_STATE))
        validate_artifact(a, evidence, 'synthetic-seal')
        a['upstream_bindings'] = {'HEAD': 'altered'}
        with self.assertRaises(FreezeError): validate_artifact(a, evidence, 'synthetic-seal')

    def test_later_specification_cannot_change(self):
        evidence = {'upstream_bindings': {}, 'environment': {},
                    'frozen_FIT_NORMAL': self.fit, 'excluded_CAL_NORMAL': self.cal}
        a = make_artifact(derive(self.fit, self.cal), evidence, 'synthetic-seal', dict(ZERO_STATE))
        a = copy.deepcopy(a)
        a['later_graph_specification']['node_order'] = ['Camera', 'Seg', 'IMU']
        with self.assertRaises(FreezeError): validate_artifact(a, evidence, 'synthetic-seal')

    def test_membership_has_no_performance_parameter(self):
        import inspect
        self.assertEqual(list(inspect.signature(derive).parameters), ['fit', 'cal'])


class GuardTests(unittest.TestCase):
    def test_no_network_block_before_request(self):
        guard, counters = make_guard()
        with self.assertRaises(FreezeError): guard('socket.connect', (None, ('invalid.synthetic', 443)))
        self.assertEqual(counters['network_requests'], 0)
        self.assertEqual(counters['blocked_network_attempts'], 1)

    def test_no_test_access_block_before_open(self):
        guard, counters = make_guard()
        for name in ('TEST/scenario-1/data', 'TEST_NORMAL/data', 'carlanomaly-base-test.tar.gz',
                     'anomaly-observation.feather'):
            with self.subTest(name=name), self.assertRaises(FreezeError): guard('open', (name, 'r', 0))
        self.assertEqual(counters['TEST_requests'], 0)
        self.assertEqual(counters['blocked_TEST_attempts'], 4)

    def test_upstream_writes_blocked(self):
        guard, counters = make_guard((BUNDLE,))
        with self.assertRaises(FreezeError): guard('open', (BUNDLE.parent / 'gate2_train_health_v3/FINAL.json', 'w', os.O_WRONLY))
        self.assertEqual(counters['blocked_mutation_attempts'], 1)

    def test_raw_archive_open_blocked(self):
        from freeze_common import SOURCE
        guard, counters = make_guard()
        with self.assertRaises(FreezeError): guard('open', (SOURCE['path'], 'r', 0))
        self.assertEqual(counters['blocked_raw_archive_attempts'], 1)

    def test_model_import_blocked(self):
        guard, counters = make_guard()
        for name in ('torch', 'cognix.graph', 'tensorflow', 'jax'):
            with self.subTest(name=name), self.assertRaises(FreezeError): guard('import', (name,))
        self.assertEqual(counters['blocked_model_attempts'], 4)

    def test_mutating_subprocess_blocked(self):
        guard, counters = make_guard()
        with self.assertRaises(FreezeError): guard('subprocess.Popen', ('git', ['git', 'commit'], None, None))
        self.assertEqual(counters['blocked_subprocess_attempts'], 1)

    def test_existing_target_fails_closed(self):
        with self.assertRaises(FreezeError): assert_absent(BUNDLE, BUNDLE.parent / '.synthetic.pending')

    def test_existing_pending_fails_closed(self):
        with self.assertRaises(FreezeError): assert_absent(BUNDLE / 'does_not_exist', BUNDLE)

    def test_prior_git_state_preserved_with_only_new_partition(self):
        initial = {'HEAD': 'fixed', 'tracked_changes': '', 'staged_changes': '',
                   'status_porcelain': '?? prior-evidence/'}
        current = dict(initial, status_porcelain='?? prior-evidence/\n?? ' + TARGET.relative_to(REPO).as_posix() + '/')
        verify_git_preserved(current, initial)
        with self.assertRaises(FreezeError):
            verify_git_preserved(dict(current, status_porcelain=current['status_porcelain'] + '\n?? other/'), initial)

    def test_exclusive_write_no_overwrite(self):
        # This existing source file must survive exclusive-create refusal.
        p = Path(__file__)
        before = p.read_bytes()
        with self.assertRaises(FileExistsError): write_new(p, b'cannot overwrite')
        self.assertEqual(p.read_bytes(), before)

    def test_seal_tamper_extra_missing_and_bad_path(self):
        # All filesystem behavior simulated in memory; no prior evidence mutated.
        from freeze_common import tree_inventory
        raw = b'payload'
        listing = (digest(raw) + '  payload.json\n').encode()
        expected = digest(listing)
        root = BUNDLE / 'synthetic_seal'
        fs = {root/'SHA256SUMS': listing, root/'SHA256SUMS.sha256': (expected+'  SHA256SUMS\n').encode(),
              root/'payload.json': raw}
        actual = {'files': [{'path': 'payload.json', 'bytes': len(raw), 'sha256': digest(raw)}], 'directories': []}
        with mock.patch('freeze_common.safe_path', lambda p: Path(p)), \
             mock.patch.object(Path, 'read_bytes', lambda p: fs[p]), \
             mock.patch.object(Path, 'exists', lambda p: p in fs), \
             mock.patch('freeze_common.hash_file', lambda p: digest(fs[Path(p)])), \
             mock.patch('freeze_common.tree_inventory', lambda *a: actual):
            verify_seal(root, expected)
            fs[root/'payload.json'] = b'tampered'
            with self.assertRaises(FreezeError): verify_seal(root, expected)
            fs[root/'payload.json'] = raw
            actual['files'].append({'path': 'extra.json', 'bytes': 0, 'sha256': digest(b'')})
            with self.assertRaises(FreezeError): verify_seal(root, expected)
            actual['files'] = []
            with self.assertRaises(FreezeError): verify_seal(root, expected)
            bad_listing = (digest(raw)+'  ../escape.json\n').encode()
            fs[root/'SHA256SUMS'] = bad_listing
            fs[root/'SHA256SUMS.sha256'] = (digest(bad_listing)+'  SHA256SUMS\n').encode()
            with self.assertRaises(FreezeError): verify_seal(root, digest(bad_listing))

    def test_upstream_directory_inventory_uses_string_lexical_order(self):
        root = BUNDLE / 'synthetic_directory_order'
        payloads = {'DIRECTORY_INVENTORY.json': canonical(['units', 'units/Town01', 'units/source']),
                    'units/source/data.bin': b'x', 'units/Town01/data.bin': b'y'}
        rows = [{'path': n, 'bytes': len(v), 'sha256': digest(v)} for n, v in payloads.items()]
        listing = ''.join(f"{r['sha256']}  {r['path']}\n" for r in rows).encode()
        expected = digest(listing)
        fs = {root / k: v for k, v in payloads.items()}
        fs[root/'SHA256SUMS'] = listing
        fs[root/'SHA256SUMS.sha256'] = (expected+'  SHA256SUMS\n').encode()
        actual = {'files': rows, 'directories': ['units', 'units/source', 'units/Town01']}
        with mock.patch('freeze_common.safe_path', lambda p: Path(p)), \
             mock.patch.object(Path, 'read_bytes', lambda p: fs[p]), \
             mock.patch.object(Path, 'exists', lambda p: p in fs), \
             mock.patch('freeze_common.hash_file', lambda p: digest(fs[Path(p)])), \
             mock.patch('freeze_common.tree_inventory', lambda *a: actual):
            verify_seal(root, expected)


if __name__ == '__main__':
    unittest.main()
