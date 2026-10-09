"""Synthetic contract tests. Real TRAIN feature blocks and archive are never opened."""
import copy
import datetime
import hashlib
import inspect
import json
import os
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch
import numpy as np
import gate2_common as common
from gate2_common import (BUNDLE, REPO, DIMS, MODALITIES, RECIPES, SEVERITIES, TOKEN,
    ReviewRequired, require, canonical, hash_file, read_json, write_new, verify_seal, seal_tree, make_guard)
from gate2_science import load_science, new_agent, member_arrays, restore_agent, check_features
from gate2_data import (assemble_fit, recipe_for_tick, validate_cal_row, npz_bytes, load_npz,
                       modality_calibration, clean_cal_rows, array_digest)
from gate2_audit import (classify, audit_complete, audit_modality, constants, numeric_predicates,
                        gate_decision, json_evidence)
from gate2_pseudo import generate_tick, CALSink, read_pseudo
from gate2_preflight import validate_partition, validate_bindings, namespace_check
from gate2_runtime import Runtime, verify_runtime, manifest_for, expected_units
from execute_gate2 import authorization

SCRATCH = None
SCIENCE = None


def finite_fixture():
    scores = np.repeat([.1, .3, .5, .7, .9], 10)
    labels = np.concatenate([np.r_[np.ones(n), np.zeros(10 - n)] for n in (1, 3, 5, 7, 9)])
    return np.repeat(scores[:, None], 5, axis=1), labels


def clean_record(sid, n=3):
    arrays = {'tick': np.arange(1, n + 1), 'scenario_id': np.array([sid] * n),
              'source_split': np.array(['train'] * n)}
    for modality, dim in DIMS.items():
        # Stable numeric marker makes contamination visible to a fitting spy.
        arrays[modality.lower()] = np.full((n, dim), int(hashlib.sha256(sid.encode()).hexdigest()[:4], 16))
    return {'scenario_id': sid, 'source_split': 'train', 'role': 'clean', 'synthetic_corruption': False,
            'arrays': arrays, 'block_npz_sha256': 'synthetic', 'block_content_sha256': 'synthetic'}


def pseudo_row(sid='Town01/scenario-1', tick=1):
    rid = recipe_for_tick(tick)
    return {'source_split': 'train', 'partition': 'CAL_NORMAL', 'parent_scenario': sid,
            'parent_tick': tick, 'window_start_tick': max(0, tick - 11), 'window_end_tick': tick,
            'target_normal': 0, 'synthetic_corruption': True, 'recipe_id': rid,
            'severity': SEVERITIES[rid], 'seed': tick, 'modality': rid.split('_')[0],
            'graph_role': None, 'upstream_fit': False}


class SyntheticCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global SCIENCE
        if SCIENCE is None:
            SCIENCE = load_science()

    def directory(self):
        path = SCRATCH / uuid.uuid4().hex
        path.mkdir()
        return path


class IsolationTests(SyntheticCase):
    def setUp(self):
        self.membership = read_json(BUNDLE / 'execution_bindings.json')['membership']

    def test_exact_all_76_upstream_FIT_clean_only_no_CAL(self):
        reads = []
        def reader(sid):
            reads.append(sid)
            return clean_record(sid)
        x, bound = assemble_fit(self.membership, 'Camera', reader, 3)
        self.assertEqual(reads, sorted(self.membership['FIT_NORMAL']))
        self.assertFalse(set(reads) & set(self.membership['CAL_NORMAL']))
        self.assertEqual(len(x), 76 * 3)
        self.assertEqual(bound['pseudo_rows'], 0)
        self.assertEqual(bound['TEST_rows'], 0)
        agent = new_agent(SCIENCE, 'Camera')
        with patch.object(agent, 'fit', wraps=agent.fit) as spy:
            agent.fit(x)
            np.testing.assert_array_equal(spy.call_args.args[0], x)
        self.assertEqual(agent._ensemble.n_fitted_members, 5)

    def test_CAL_record_cannot_enter_oneclass_fitting(self):
        foreign = clean_record(self.membership['CAL_NORMAL'][0])
        with self.assertRaises(ReviewRequired):
            assemble_fit(self.membership, 'Camera', lambda sid: foreign, 3)

    def test_pseudo_record_cannot_enter_oneclass_fitting(self):
        def reader(sid):
            r = clean_record(sid)
            r['role'] = 'pseudo'
            r['synthetic_corruption'] = True
            return r
        with self.assertRaises(ReviewRequired):
            assemble_fit(self.membership, 'Seg', reader, 3)

    def test_TEST_record_cannot_enter_upstream_fit(self):
        def reader(sid):
            r = clean_record(sid)
            r['source_split'] = 'test_normal'
            return r
        with self.assertRaises(ReviewRequired):
            assemble_fit(self.membership, 'IMU', reader, 3)

    def test_CAL_pseudo_parent_isolation(self):
        sid = self.membership['CAL_NORMAL'][0]
        row = pseudo_row(sid)
        validate_cal_row(row, self.membership, 'Camera')
        row['parent_scenario'] = self.membership['FIT_NORMAL'][0]
        with self.assertRaises(ReviewRequired):
            validate_cal_row(row, self.membership)

    def test_no_FIT_graph_pseudo(self):
        row = pseudo_row(self.membership['CAL_NORMAL'][0])
        row['graph_role'] = 'graph_train'
        with self.assertRaises(ReviewRequired):
            validate_cal_row(row, self.membership)

    def test_only_modality_matched_pseudo_used_by_mapping(self):
        sid = 'Town01/scenario-1'
        member = {'FIT_NORMAL': ['Town02/scenario-1'], 'CAL_NORMAL': [sid]}
        rows = [pseudo_row(sid, t) for t in (1, 2, 5)]
        arrays = {m.lower(): np.stack([np.full(dim, i) for i in range(3)]) for m, dim in DIMS.items()}
        for modality, index in (('Camera', 0), ('Seg', 1), ('IMU', 2)):
            x, y, provenance = modality_calibration(member, modality,
                lambda parent: (arrays, rows), clean_record)
            self.assertEqual(y.tolist(), [1, 1, 1, 0])
            self.assertEqual(provenance[-1]['modality'], modality.lower())
            np.testing.assert_array_equal(x[-1], arrays[modality.lower()][index])

    def test_pseudo_TEST_provenance_rejected_by_unchanged_generator(self):
        with self.assertRaises(SCIENCE['PseudoAnomalyError']):
            SCIENCE['generate_pseudo_anomaly']({'Camera': np.ones((8, 8, 3), np.uint8)},
                'camera_occlusion', 'test_normal', source_scenario='synthetic', source_tick=1)

    def test_GNSS_excluded_from_agents_recipes_and_dimensions(self):
        self.assertEqual(set(DIMS), {'Camera', 'Seg', 'IMU'})
        self.assertFalse(any(r.startswith('gnss') for r in SCIENCE['_RECIPE_BY_ID']))
        with self.assertRaises(ReviewRequired):
            new_agent(SCIENCE, 'GNSS')
        with self.assertRaises(ReviewRequired):
            assemble_fit(self.membership, 'GNSS', clean_record, 3)


class ScienceTests(SyntheticCase):
    def test_exact_Camera18_Seg29_IMU10_dimensions(self):
        samples = {'Camera': SCIENCE['camera_embedding_features'](np.ones((8, 8, 3), np.uint8)),
                   'Seg': SCIENCE['segmentation_histogram_features'](np.arange(29, dtype=np.uint8)[None, :]),
                   'IMU': SCIENCE['imu_window_features'](np.arange(36).reshape(12, 3))}
        for modality, vector in samples.items():
            self.assertEqual(vector.shape, (DIMS[modality],))
            check_features(vector[None, :], modality)
            with self.assertRaises(ReviewRequired):
                check_features(np.ones((3, DIMS[modality] + 1)), modality)

    def test_bootstrap_members_5_seed42_and_state_roundtrip(self):
        x = np.random.default_rng(100).normal(size=(20, 10))
        a, b = new_agent(SCIENCE, 'IMU').fit(x), new_agent(SCIENCE, 'IMU').fit(x)
        self.assertEqual((a.n_members, a.seed), (5, 42))
        arrays = member_arrays(a)
        for key, value in arrays.items():
            np.testing.assert_array_equal(value, member_arrays(b)[key])
        restored = restore_agent(SCIENCE, 'IMU', arrays)
        np.testing.assert_array_equal(a.predict_normality(x[0]), restored.predict_normality(x[0]))

    def test_causal_window12_and_warmup_preserved(self):
        member = {'FIT_NORMAL': ['other'], 'CAL_NORMAL': ['Town01/scenario-1']}
        for tick in (1, 5, 12, 100):
            if recipe_for_tick(tick) is None:
                continue
            row = pseudo_row(tick=tick)
            validate_cal_row(row, member)
            row['window_start_tick'] = max(0, tick - 12)
            if tick >= 12:
                with self.assertRaises(ReviewRequired):
                    validate_cal_row(row, member)

    def test_exact_frozen_pseudo_severities(self):
        frozen = read_json(BUNDLE / 'full_train_preregistered_protocol.json')['frozen_modeling']['pseudo_definitions_and_default_severities']
        self.assertEqual(SEVERITIES, {k: v for k, v in frozen['default_severities'].items() if not k.startswith('gnss')})
        self.assertEqual(RECIPES, tuple(frozen['recipes']))
        self.assertEqual(frozen['max_pseudo_per_tick'], 1)
        self.assertEqual(frozen['seed'], 0)

    def test_max_one_per_tick_rotation_seed_unchanged_GNSS_slots_skipped(self):
        expected = ['camera_occlusion', 'seg_region_corruption', None, None,
                    'imu_spike', 'imu_bias_scale', 'camera_brightness_shift']
        self.assertEqual([recipe_for_tick(t) for t in range(1, 8)], expected)
        self.assertEqual([recipe_for_tick(t) for t in range(8, 15)], expected)

    def test_deterministic_pseudo_behavior_and_source_immutability(self):
        membership = {'FIT_NORMAL': ['other'], 'CAL_NORMAL': ['Town01/scenario-1']}
        for tick in (1, 2, 5, 6, 7):
            rid = recipe_for_tick(tick)
            modality = 'Camera' if rid.startswith('camera') else 'Seg' if rid.startswith('seg') else 'IMU'
            payload = (np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3) if modality == 'Camera'
                       else np.tile(np.arange(16, dtype=np.uint8), (16, 1)) if modality == 'Seg'
                       else np.arange(18, dtype=float).reshape(6, 3))
            fn = SCIENCE[{'Camera': 'camera_embedding_features', 'Seg': 'segmentation_histogram_features',
                          'IMU': 'imu_window_features'}[modality]]
            original = payload.copy()
            parent = {m.lower(): np.ones(d) for m, d in DIMS.items()}
            parent[modality.lower()] = fn(payload)
            a, ar = generate_tick(SCIENCE, membership, 'Town01/scenario-1', tick, payload, parent)
            b, br = generate_tick(SCIENCE, membership, 'Town01/scenario-1', tick, payload, parent)
            self.assertEqual(ar, br)
            if a is not None:
                for key in a:
                    np.testing.assert_array_equal(a[key], b[key])
                self.assertEqual(ar['seed'], tick)
                self.assertEqual(ar['severity'], SEVERITIES[rid])
            np.testing.assert_array_equal(payload, original)

    def test_seg_amendment_is_upstream_of_pseudo_recipe(self):
        payload = np.array([[0, 29, 30, 255], [1, 2, 22, 28]], dtype=np.uint8)
        sanitized, counts = SCIENCE['sanitize'](payload)
        parent = {m.lower(): np.ones(d) for m, d in DIMS.items()}
        parent['seg'] = SCIENCE['segmentation_histogram_features'](sanitized)
        membership = {'FIT_NORMAL': ['other'], 'CAL_NORMAL': ['Town01/scenario-1']}
        seen = []
        original = SCIENCE['generate_pseudo_anomaly']
        def spy(observation, **kwargs):
            seen.append(observation['Seg'].copy())
            return original(observation, **kwargs)
        with patch.dict(SCIENCE, generate_pseudo_anomaly=spy):
            generate_tick(SCIENCE, membership, 'Town01/scenario-1', 2, payload, parent)
        np.testing.assert_array_equal(seen[0], sanitized)
        self.assertEqual(counts['count_remapped'], 3)
        np.testing.assert_array_equal(payload, [[0, 29, 30, 255], [1, 2, 22, 28]])

    def test_uncertainty_equations_nats_no_amplification(self):
        p = np.array([.1, .2, .5, .7, .9])
        entropy = lambda v: -(v * np.log(v) + (1 - v) * np.log1p(-v))
        u = SCIENCE['ensemble_to_uncertainty'](p)
        self.assertAlmostEqual(u.total, entropy(p.mean()))
        self.assertAlmostEqual(u.aleatoric, entropy(p).mean())
        self.assertAlmostEqual(u.epistemic, u.total - u.aleatoric)
        identical = SCIENCE['ensemble_to_uncertainty'](np.full(5, .5))
        self.assertEqual(identical.epistemic, 0.)
        tiny = SCIENCE['ensemble_to_uncertainty'](np.array([.5, .5, .5, .5, .5 + 1e-10]))
        self.assertLess(tiny.epistemic, 1e-12)
        self.assertAlmostEqual(identical.total, np.log(2))

    def test_finite_probability_UQ_parameter_domain_checks(self):
        x, s, p, uq, params = np.ones((2, 18)), np.ones((2, 5)) * .5, np.ones((2, 5)) * .5, np.tile([.2, .1, .1], (2, 1)), np.array([1., 0., .5, .1])
        self.assertTrue(numeric_predicates(x, s, p, uq, params)['pass'])
        for target in (x, s, p, uq, params):
            original = target.flat[0]
            target.flat[0] = np.nan
            self.assertFalse(numeric_predicates(x, s, p, uq, params)['pass'])
            target.flat[0] = original
        for target, value in ((s, -1.), (p, 1.01), (uq, -1.), (uq, 1.)):
            original = target.flat[0]
            target.flat[0] = value
            self.assertFalse(numeric_predicates(x, s, p, uq, params)['pass'])
            target.flat[0] = original

    def test_exact_constant_detection_preserves_tiny_ranges(self):
        self.assertTrue(constants(np.ones(5))['exact_constant_output'])
        a = np.array([.5, .5, np.nextafter(.5, 1.)])
        self.assertFalse(constants(a)['exact_constant_output'])
        self.assertEqual(constants(np.array([[1., 2.], [1., 3.]]))['exact_constant_columns'], [True, False])


class DecisionTests(SyntheticCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        M, y = finite_fixture()
        cls.info = SCIENCE['EnsemblePredictiveCalibrator']().fit(M, y).optimization_info

    def decide(self, info=None, **changes):
        args = dict(numeric_ok=True, provenance_ok=True, supported_classes=True,
                    constant_output=False, require_valid_ok=True)
        args.update(changes)
        return classify(self.info if info is None else info, **args)

    def test_VALID_requires_complete_unchanged_audit(self):
        self.assertTrue(audit_complete(self.info))
        self.assertEqual(self.decide(), 'VALID')
        self.assertEqual(self.decide(require_valid_ok=False), 'INVALID_OTHER')

    def test_numeric_INVALID_has_first_priority(self):
        info = dict(self.info, suspected_separation=True)
        self.assertEqual(self.decide(info, numeric_ok=False), 'INVALID_NUMERIC')

    def test_structural_invalid_strict_and_weak(self):
        for scores in ([.1, .2, .8, .9], [.1, .5, .5, .9]):
            c = SCIENCE['EnsemblePredictiveCalibrator']().fit(np.array(scores), np.array([0, 0, 1, 1]))
            self.assertEqual(self.decide(c.optimization_info, require_valid_ok=False), 'INVALID_STRUCTURAL')
            with self.assertRaises(RuntimeError):
                c.require_valid()

    def test_structural_suspected_boundary_direction_precedes_OTHER(self):
        for changes in ({'suspected_separation': True}, {'boundary_trend': 'decreasing'},
                        {'lower_profile_direction': True}):
            info = dict(self.info, **changes)
            self.assertEqual(self.decide(info, provenance_ok=False), 'INVALID_STRUCTURAL')

    def test_mean_separation_alone_is_descriptive_not_structural_proof(self):
        info = copy.deepcopy(self.info)
        info['separation']['ensemble_mean']['classification'] = 'completely_separated'
        self.assertEqual(self.decide(info), 'VALID')

    def test_OTHER_missing_audit_provenance_classes_constant(self):
        self.assertEqual(self.decide({}), 'INVALID_OTHER')
        self.assertEqual(self.decide(dict(self.info, finite_optimum=False)), 'INVALID_OTHER')
        for flag in ('provenance_ok', 'supported_classes'):
            self.assertEqual(self.decide(**{flag: False}), 'INVALID_OTHER')
        self.assertEqual(self.decide(constant_output=True), 'INVALID_OTHER')

    def test_any_invalid_active_modality_fails_gate(self):
        for modality in MODALITIES:
            for invalid in ('INVALID_STRUCTURAL', 'INVALID_NUMERIC', 'INVALID_OTHER'):
                statuses = dict.fromkeys(MODALITIES, 'VALID')
                statuses[modality] = invalid
                self.assertEqual(gate_decision(statuses), 'TRAIN_ONLY_SCIENTIFIC_HEALTH_FAIL')

    def test_all_three_VALID_pass_and_no_missing_GNSS_channel(self):
        self.assertEqual(gate_decision(dict.fromkeys(MODALITIES, 'VALID')), 'TRAIN_ONLY_SCIENTIFIC_HEALTH_PASS')
        with self.assertRaises(ReviewRequired):
            gate_decision({'Camera': 'VALID', 'Seg': 'VALID'})
        with self.assertRaises(ReviewRequired):
            gate_decision(dict.fromkeys((*MODALITIES, 'GNSS'), 'VALID'))

    def test_complete_audit_emits_every_diagnostic_on_synthetic_only(self):
        M, labels = finite_fixture()
        x = np.random.default_rng(9).normal(size=(len(M), 18))
        agent = new_agent(SCIENCE, 'Camera').fit(x)
        rows = clean_cal_rows('Town01/scenario-1', np.arange(1, len(M) + 1))
        for i, label in enumerate(labels):
            if label == 0:
                rows[i] = pseudo_row(tick=1 + 7 * i)
        binding = {'total_rows': 20, 'pseudo_rows': 0, 'TEST_rows': 0}
        with patch.object(agent, 'predict_normality', side_effect=list(M)):
            report, arrays = audit_modality(SCIENCE, agent, 'Camera', x, labels, rows, binding,
                {'FIT_NORMAL': ['other'], 'CAL_NORMAL': ['Town01/scenario-1']})
        self.assertEqual(report['status'], 'VALID')
        self.assertEqual(report['sample_counts']['CAL_combined'], len(x))
        for field in ('optimization_info', 'numeric_predicates', 'class_distributions', 'recipe_roles',
                      'uncertainty', 'exact_constant_checks', 'member_score_disagreement',
                      'member_probability_disagreement', 'probability_saturation', 'FIT_binding'):
            self.assertIn(field, report)
        self.assertEqual(arrays['member_scores'].shape, (50, 5))
        self.assertEqual(report['uncertainty']['epistemic']['max'], 0.)

    def test_missing_diagnostic_outputs_are_OTHER_without_fabricated_numeric_failure(self):
        M, labels = finite_fixture()
        x = np.random.default_rng(9).normal(size=(len(M), 18))
        agent = new_agent(SCIENCE, 'Camera').fit(x)
        rows = clean_cal_rows('Town01/scenario-1', np.arange(1, len(M) + 1))
        for i, label in enumerate(labels):
            if label == 0:
                rows[i] = pseudo_row(tick=1 + 7 * i)
        with patch.object(agent, 'predict_normality', side_effect=list(M)), \
             patch.dict(SCIENCE, ensemble_to_uncertainty=lambda p: (_ for _ in ()).throw(RuntimeError('synthetic missing diagnostic'))):
            report, arrays = audit_modality(SCIENCE, agent, 'Camera', x, labels, rows,
                {'total_rows': 20, 'pseudo_rows': 0, 'TEST_rows': 0},
                {'FIT_NORMAL': ['other'], 'CAL_NORMAL': ['Town01/scenario-1']})
        self.assertEqual(report['status'], 'INVALID_OTHER')
        self.assertTrue(report['numeric_predicates']['pass'])
        self.assertEqual(report['uncertainty']['epistemic']['count'], 0)
        self.assertFalse(arrays['UQ_row_available'].any())


class GuardAndBindingTests(SyntheticCase):
    def test_no_network_guard_and_no_TEST_guard(self):
        guard, counters = make_guard(writable=(SCRATCH,))
        for event, args in (('socket.connect', (None, 'example.invalid')),
                            ('open', (str(SCRATCH / 'test' / 'fake'), 'r', 0)),
                            ('open', (str(SCRATCH / 'test_normal' / 'fake'), 'r', 0))):
            with self.assertRaises(ReviewRequired):
                guard(event, args)
        self.assertEqual(counters['network_requests'], 0)
        self.assertEqual(counters['TEST_requests'], 0)
        self.assertEqual(counters['blocked_network_attempts'], 1)
        self.assertEqual(counters['blocked_TEST_attempts'], 2)

    def test_no_graph_no_GAT_import_no_graph_units(self):
        guard, counters = make_guard()
        for name in ('cognix.graph.epistemic_gat', 'torch', 'graph_fit_export', 'graph_training_metrics'):
            with self.assertRaises(ReviewRequired):
                guard('import', (name, None))
        self.assertEqual(counters['blocked_graph_attempts'], 4)
        self.assertNotIn('RealGNSSAgent', SCIENCE)
        self.assertFalse(any(name.startswith('cognix.graph') for name in __import__('sys').modules))

    def test_guard_denies_mutation_of_upstream_and_real_source_in_tests(self):
        guard, counters = make_guard(writable=(SCRATCH,))
        with self.assertRaises(ReviewRequired):
            guard('open', (str(common.PARTITION / 'partition.json'), 'w', os.O_WRONLY))
        with self.assertRaises(ReviewRequired):
            guard('open', (common.SOURCE['path'], 'r', 0))
        self.assertEqual(counters['blocked_mutation_attempts'], 1)
        self.assertEqual(counters['blocked_source_attempts'], 1)

    def test_upstream_partition_exact_binding_mismatch_stops_no_recomputation(self):
        record = read_json(common.PARTITION / 'partition.json')
        expected = read_json(BUNDLE / 'execution_bindings.json')['membership']
        validate_partition(record, expected, record['v8_COMPLETE_GATE1'])
        expected['FIT_NORMAL'] = list(reversed(expected['FIT_NORMAL']))
        with self.assertRaises(ReviewRequired):
            validate_partition(record, expected, record['v8_COMPLETE_GATE1'])

    def test_source_binding_mismatch_stops(self):
        binding = read_json(BUNDLE / 'execution_bindings.json')
        binding['source_identity']['sha256'] = '0' * 64
        with self.assertRaises(ReviewRequired):
            validate_bindings(binding)

    def test_v8_seal_and_partition_runtime_seal_mismatch_stops(self):
        binding = read_json(BUNDLE / 'execution_bindings.json')
        for key in ('v8_seal', 'partition_runtime_seal', 'partition_sha256', 'preprocessing_policy_sha256'):
            altered = copy.deepcopy(binding)
            altered[key] = '0' * 64
            with self.assertRaises(ReviewRequired):
                validate_bindings(altered)
        root = self.directory()
        write_new(root / 'evidence.json', b'{}')
        seal_tree(root)
        with self.assertRaises(ReviewRequired):
            verify_seal(root, '0' * 64)

    def test_independent_source_SHA_mismatch_stops_on_synthetic_file(self):
        from partition_common import fingerprint, verify_source
        path = self.directory() / 'synthetic-train.gz'
        write_new(path, b'synthetic bytes')
        identity = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': hash_file(path)}
        bound = {'identity': identity, 'fingerprint': fingerprint(path),
                 'independently_measured_full_sha256': identity['sha256']}
        verify_source(identity, bound)
        altered = dict(identity, sha256='0' * 64)
        bound = dict(bound, identity=altered, independently_measured_full_sha256='0' * 64)
        with self.assertRaises(Exception):
            verify_source(altered, bound)

    def test_authorization_required_before_fitting_and_creation(self):
        for token in (None, '', TOKEN + '_extra'):
            with self.assertRaises(ReviewRequired):
                authorization(token)
        authorization(TOKEN)

    def test_final_executor_cannot_advance_to_graph(self):
        source = (BUNDLE / 'execute_gate2.py').read_text()
        import ast
        tree = ast.parse(source)
        calls = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        self.assertNotIn('build_graph', calls)
        self.assertNotIn('train_graph', calls)
        self.assertNotIn('NoGraph', calls)
        self.assertNotIn('StandardGAT', calls)
        self.assertNotIn('EpistemicGAT', calls)

    def test_nonfinite_evidence_preserved_in_standard_JSON(self):
        value = json_evidence({'params': [np.nan, np.inf, -np.inf]})
        canonical(value)
        self.assertEqual(value['params'][0], {'nonfinite': 'NaN'})


class RuntimeTests(SyntheticCase):
    def runtime(self):
        parent = self.directory()
        root = parent / 'runtime'
        evidence = {'bindings': {'prediction_semantics': common.SEMANTICS},
                    'membership': {'FIT_NORMAL': ['synthetic-FIT'], 'CAL_NORMAL': ['synthetic-CAL']},
                    'environment': {'synthetic': True}, 'source_binding': {'synthetic': True},
                    'completion': {'synthetic': True}}
        return Runtime(root, 'a' * 64, evidence)

    def test_immutable_committed_unit_no_overwrite(self):
        run = self.runtime()
        with run.lease():
            run.begin_attempt()
            path = run.commit('fit-Camera', lambda p: write_new(p / 'value', b'original'), {'input': 1})
            original = {p.name: p.read_bytes() for p in path.iterdir()}
            run.commit('fit-Camera', lambda p: self.fail('Committed work must not repeat'), {'input': 1})
            self.assertEqual(original, {p.name: p.read_bytes() for p in path.iterdir()})
            with self.assertRaises(ReviewRequired):
                run.commit('fit-Camera', lambda p: None, {'input': 2})

    def test_resume_reuses_verified_units_and_preserves_failed_pending(self):
        run = self.runtime()
        with run.lease():
            run.begin_attempt()
            run.commit('fit-Camera', lambda p: write_new(p / 'value', b'fit'), {})
            def failed(pending):
                write_new(pending / 'partial_evidence', b'preserve me')
                raise RuntimeError('injected publication failure')
            try:
                run.commit('fit-Seg', failed, {})
            except RuntimeError as exc:
                run.finish_attempt('EXECUTION_STOPPED', 'synthetic-fit', exc)
            attempt = run.attempt
            before = {p.relative_to(attempt).as_posix(): p.read_bytes() for p in attempt.rglob('*') if p.is_file()}
        resumed = Runtime(run.root, run.bundle_seal, run.evidence)
        with resumed.lease():
            resumed.begin_attempt()
            self.assertTrue(resumed.has('fit-Camera'))
            resumed.commit('fit-Seg', lambda p: write_new(p / 'value', b'deterministic second attempt'), {})
            resumed.finish_attempt('COMPLETE_SYNTHETIC', 'resume')
        self.assertEqual(before, {p.relative_to(attempt).as_posix(): p.read_bytes() for p in attempt.rglob('*') if p.is_file()})
        self.assertEqual(read_json(attempt / 'failure_ledger.json')['phase'], 'synthetic-fit')

    def test_OS_lease_released_after_interruption(self):
        run = self.runtime()
        with self.assertRaises(KeyboardInterrupt):
            with run.lease():
                raise KeyboardInterrupt()
        with run.lease():
            self.assertTrue(run.verify()['state'].startswith('resumable'))

    def test_completed_runtime_immutable_and_final_failure_retained(self):
        run = self.runtime()
        statuses = {'Camera': 'VALID', 'Seg': 'INVALID_STRUCTURAL', 'IMU': 'VALID'}
        with run.lease():
            run.begin_attempt()
            for name in sorted(expected_units(run.evidence['membership'])):
                def writer(path, name=name):
                    if name.startswith('audit-'):
                        write_new(path / 'audit.json', canonical({'status': statuses[name[6:]]}))
                    else:
                        write_new(path / 'synthetic', b'fixture')
                run.commit(name, writer, {})
            run.finish_attempt('COMPLETE_SYNTHETIC', 'decision')
            final = run.finalize(statuses)
        self.assertEqual(final['status'], 'TRAIN_ONLY_SCIENTIFIC_HEALTH_FAIL')
        self.assertEqual(run.verify()['state'], 'complete_verified_immutable')
        before = hash_file(run.root / 'SHA256SUMS')
        with self.assertRaises(ReviewRequired):
            run.begin_attempt()
        with self.assertRaises(ReviewRequired):
            with run.lease():
                pass
        self.assertEqual(hash_file(run.root / 'SHA256SUMS'), before)

    def test_tampered_committed_evidence_stops_resume(self):
        run = self.runtime()
        with run.lease():
            run.begin_attempt()
            path = run.commit('fit-IMU', lambda p: write_new(p / 'value', b'original'), {})
        (path / 'value').write_bytes(b'tampered synthetic fixture')
        with self.assertRaises(ReviewRequired):
            Runtime(run.root, run.bundle_seal, run.evidence)

    def test_conflicting_runtime_source_binding_stops(self):
        run = self.runtime()
        other = copy.deepcopy(run.evidence)
        other['source_binding'] = {'different_synthetic': True}
        with self.assertRaises(ReviewRequired):
            Runtime(run.root, run.bundle_seal, other)

    def test_graph_unit_is_rejected(self):
        run = self.runtime()
        with run.lease():
            run.begin_attempt()
            with self.assertRaises(ReviewRequired):
                run.commit('graph-NoGraph', lambda p: self.fail('No graph writer'), {})

    def test_simultaneous_OS_lease_is_rejected(self):
        run = self.runtime()
        with run.lease():
            with self.assertRaises(OSError):
                with run.lease():
                    self.fail('Concurrent lease accepted')

    def test_orphan_uncommitted_attempt_is_preserved_on_resume(self):
        run = self.runtime()
        with run.lease():
            run.begin_attempt()
            orphan = run.attempt
            pending = orphan / 'pending-fit-Camera'
            pending.mkdir()
            (pending / 'empty-forensic-directory').mkdir()
            write_new(pending / 'partial', b'crash evidence')
        resumed = Runtime(run.root, run.bundle_seal, run.evidence)
        with resumed.lease():
            resumed.begin_attempt()
            resumed.commit('fit-Camera', lambda p: write_new(p / 'value', b'new same-seed attempt'), {})
        self.assertEqual((pending / 'partial').read_bytes(), b'crash evidence')
        self.assertFalse((orphan / 'SHA256SUMS').exists())
        seal_tree(orphan)
        verify_seal(orphan)
        self.assertTrue((pending / 'empty-forensic-directory').is_dir())


class RawReplayTests(SyntheticCase):
    def test_small_artificial_full_window_replay_CAL_only_GNSS_ignored(self):
        import io
        import tarfile
        import gzip
        import PIL.Image
        import pandas as pd
        from train_replay import TrainReplay
        sid, fit_sid = 'Town01/scenario-1', 'Town01/scenario-2'
        camera = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
        segmentation = np.tile(np.array([0, 1, 2, 29, 30, 28, 255, 3], dtype=np.uint8), (8, 1))
        def image_bytes(array, format):
            stream = io.BytesIO()
            PIL.Image.fromarray(array).save(stream, format=format)
            return stream.getvalue()
        jpg, png = image_bytes(camera, 'JPEG'), image_bytes(segmentation, 'PNG')
        with PIL.Image.open(io.BytesIO(jpg)) as image:
            decoded_camera = np.asarray(image)
        imu = np.arange(9000, dtype=float).reshape(3000, 3) / 1000
        feather = io.BytesIO()
        pd.DataFrame(imu, columns=SCIENCE['IMU_ACCEL_COLUMNS']).to_feather(feather)
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode='w', format=tarfile.USTAR_FORMAT) as tar:
            for name in ('train', 'train/Town01', 'train/' + sid):
                member = tarfile.TarInfo(name)
                member.type = tarfile.DIRTYPE
                tar.addfile(member)
            def add(name, raw):
                member = tarfile.TarInfo('train/' + sid + '/' + name)
                member.size = len(raw)
                tar.addfile(member, io.BytesIO(raw))
            add('imu.feather', feather.getvalue())
            # Invalid GNSS bytes prove this payload is never decoded/processed.
            add('gnss.feather', b'GNSS ignored synthetic payload')
            for tick in range(3000):
                add(f'rgb-front/{tick:06d}.jpg', jpg)
                add(f'segmentation-front/{tick:06d}.png', png)
            member = tarfile.TarInfo('train/' + fit_sid)
            member.type = tarfile.DIRTYPE
            tar.addfile(member)
            member = tarfile.TarInfo('train/' + fit_sid + '/FIT-must-not-be-materialized')
            member.size = 3
            tar.addfile(member, io.BytesIO(b'FIT'))
        source = gzip.compress(buffer.getvalue(), mtime=0)
        observed = TrainReplay()
        observed.feed(source)
        self.assertTrue(observed.finished)
        records = [{'scenario_id': r['scenario_id'], 'archive_ordinal': r['archive_ordinal'],
                    'schema':'frozen-local-TRAIN-scenario-v7','town':r['scenario_id'].split('/')[0],
                    'n_ticks':3000,'emitted_rows':2999,
                    'first_tar_offset': r['first_tar_header_offset'],
                    'parser_member_chain_sha256': r['member_chain_sha256'],
                    'members': [m for m in observed.members if m['scenario_id'] == r['scenario_id']]}
                   for r in observed.completed]
        import local_inventory
        from gate2_replay_inventory import ReplayInventory
        for record in records:
            record['source_member_set_sha256'] = local_inventory.member_digest(record['members'])
        entries = [dict(local_inventory.summary(r),record_sha256=local_inventory.digest(r)) for r in records]
        inventory_index = {'schema':'frozen-local-TRAIN-inventory-v7','source_binding':{'synthetic':True},
            'ledger_name':'inventory_scenarios.jsonl.gz','scenarios':entries,'scenario_count':len(entries),
            'scenario_index_sha256':local_inventory.digest(entries)}
        self.assertTrue(all('record_sha256' not in r for r in records))
        arrays = {'camera': np.repeat(SCIENCE['camera_embedding_features'](decoded_camera)[None, :], 2999, axis=0),
                  'seg': np.repeat(SCIENCE['segmentation_histogram_features'](SCIENCE['sanitize'](segmentation)[0])[None, :], 2999, axis=0),
                  'imu': np.stack([SCIENCE['imu_window_features'](imu[max(0, t - 11):t + 1]) for t in range(1, 3000)]),
                  'tick': np.arange(1, 3000)}
        evidence = {'bindings': {'prediction_semantics': common.SEMANTICS,
                    'preprocessing_policy_sha256': read_json(BUNDLE / 'execution_bindings.json')['preprocessing_policy_sha256']},
                    'membership': {'FIT_NORMAL': [fit_sid], 'CAL_NORMAL': [sid]},
                    'environment': {'synthetic': True}, 'source_binding': {'synthetic': True},
                    'completion': {'synthetic': True}}
        root = self.directory() / 'runtime'
        run = Runtime(root, 'a' * 64, evidence)
        clean = {'arrays': arrays, 'block_npz_sha256': '0' * 64}
        with run.lease():
            run.begin_attempt()
            sink = CALSink(run, SCIENCE, records, evidence['source_binding'],
                           ReplayInventory(inventory_index,evidence['source_binding'],local_inventory))
            replay = TrainReplay(sink=sink, on_member=lambda member: None)
            with patch('gate2_pseudo.load_clean', return_value=clean), \
                 patch('gate2_pseudo.shutil.disk_usage', return_value=type('Disk', (), {'free': 1 << 50})()):
                replay.feed(source)
            self.assertTrue(replay.finished)
            self.assertEqual(sink.ordinals_verified, [1, 2])
            unit = run.unit('cal-' + sid.replace('/', '__'))
            self.assertEqual(read_json(unit/'unit.json')['inputs']['scenario_inventory_record_sha256'],
                             local_inventory.digest(records[0]))
            generated, provenance = read_pseudo(unit)
            self.assertGreater(len(provenance), 0)
            self.assertEqual(len({r['parent_tick'] for r in provenance}), len(provenance))
            self.assertTrue(all(r['parent_scenario'] == sid and r['modality'] != 'gnss' for r in provenance))
            self.assertTrue(all(r['preprocessing_policy_applied_before_recipe'] for r in provenance if r['modality'] == 'seg'))
            for modality in MODALITIES:
                self.assertEqual(generated[modality.lower()].shape, (len(provenance), DIMS[modality]))
            self.assertFalse(any(p.name.startswith('raw-') for p in run.attempt.iterdir()))
            self.assertFalse(run.has('cal-' + fit_sid.replace('/', '__')))
            run.finish_attempt('COMPLETE_SYNTHETIC', 'CAL replay')

    def test_source_member_mismatch_stops_before_pseudo_generation(self):
        run = RuntimeTests.runtime(self)
        expected = {'scenario_id': 'synthetic-CAL', 'archive_ordinal': 1,
                    'schema':'frozen-local-TRAIN-scenario-v7','n_ticks':3000,'emitted_rows':2999,
                    'first_tar_offset': 512, 'members': [{'path': 'train/synthetic-CAL/imu.feather',
                    'sha256': 'a' * 64}], 'parser_member_chain_sha256': 'b' * 64,
                    'source_member_set_sha256':'d'*64}
        import local_inventory
        from gate2_replay_inventory import ReplayInventory
        entries = [dict(local_inventory.summary(expected),record_sha256=local_inventory.digest(expected))]
        inventory_index = {'schema':'frozen-local-TRAIN-inventory-v7','source_binding':{'synthetic':True},
            'ledger_name':'inventory_scenarios.jsonl.gz','scenarios':entries,'scenario_count':1,
            'scenario_index_sha256':local_inventory.digest(entries)}
        with run.lease():
            run.begin_attempt()
            sink = CALSink(run, SCIENCE, [expected], {'synthetic': True},
                           ReplayInventory(inventory_index,{'synthetic':True},local_inventory))
            sink.scenario_begin({'scenario_id': 'synthetic-CAL', 'archive_ordinal': 1,
                                 'first_tar_header_offset': 512})
            with self.assertRaises(ReviewRequired):
                sink.member_complete({'scenario_id': 'synthetic-CAL',
                                      'path': 'train/synthetic-CAL/imu.feather', 'sha256': 'c' * 64})
            self.assertFalse(run.has('cal-synthetic-CAL'))
