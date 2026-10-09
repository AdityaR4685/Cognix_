"""Offline unittest runner plus exact frozen mathematical regression function bodies.

pytest is not installed in the frozen local interpreter. A deliberately limited
assertion adapter supports raises/approx/parametrize for the explicitly selected
untouched regression bodies below; no pytest suite-wide execution is claimed.
"""
import io
import json
import math
import re
import shutil
import sys
import types
import unittest
import uuid
from pathlib import Path
from gate2_common import (BUNDLE, REPO, TARGET, require, install_guard, write_new,
                         pretty, hash_file, authenticate_upstream)
from gate2_science import definitions, load_science

REGRESSION_PATH = 'tests/unit/test_carla_calibration_validity.py'


class Approx:
    def __init__(self, value, rel=1e-6, abs=1e-12):
        self.value, self.rel, self.abs = value, rel, abs
    def __eq__(self, other):
        return math.isclose(float(other), float(self.value), rel_tol=self.rel, abs_tol=self.abs)


class Raises:
    def __init__(self, expected, match=None):
        self.expected, self.match = expected, match
    def __enter__(self):
        return self
    def __exit__(self, kind, value, tb):
        if kind is None or not issubclass(kind, self.expected):
            raise AssertionError('Expected frozen regression exception ' + str(self.expected))
        if self.match and not re.search(self.match, str(value)):
            raise AssertionError('Frozen exception message did not match ' + self.match)
        return True


def frozen_regressions(science):
    cases = [
        ('test_finite_seg_imu_mathematical_fixtures', ('seg',)),
        ('test_finite_seg_imu_mathematical_fixtures', ('imu',)),
        ('test_separated_fixtures_refuse_scientific_probabilities', ('camera',)),
        ('test_separated_fixtures_refuse_scientific_probabilities', ('gnss',)),
        ('test_quasi_separation_exact_shared_boundary', ()),
        ('test_mean_separation_is_not_member_separation_proof', ()),
        ('test_permutation_repeated_fit_and_duplicate_weight_invariance', ()),
        ('test_validity_requires_both_classes', (science['np'].zeros(4),)),
        ('test_validity_requires_both_classes', (science['np'].ones(4),)),
        ('test_invalid_real_agent_blocks_prediction_and_uq_with_auditable_diagnostics', ()),
        ('test_failed_refit_clears_valid_mapping', ()),
        ('test_constant_member_rows_have_no_identifiable_slope', ()),
        ('test_valid_real_agent_default_prediction_and_uq', ()),
        ('test_finite_profile_has_a_turn_instead_of_monotonic_boundary_descent', ()),
        ('test_inverted_target_does_not_validate_zero_slope_boundary', ()),
        ('test_agent_refit_normal_model_invalidates_old_calibrator', ())]
    shim = types.SimpleNamespace(raises=Raises, approx=Approx,
        mark=types.SimpleNamespace(parametrize=lambda *args, **kwargs: lambda function: function))
    ns = dict(science, pytest=shim)
    definitions(REPO / REGRESSION_PATH, {'finite_fixture'} | {name for name, _ in cases}, ns)
    suite = unittest.TestSuite()
    for name, args in cases:
        suite.addTest(unittest.FunctionTestCase(lambda name=name, args=args: ns[name](*args),
                      description=REGRESSION_PATH + '::' + name + str(args if not args or isinstance(args[0], str) else '(synthetic labels)')))
    # Retain the frozen v8 segmentation compatibility policy regressions.
    import test_experiment_2b
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(test_experiment_2b.OOVRuleTests))
    return suite


def main():
    require(not (BUNDLE / 'SHA256SUMS').exists(), 'Cannot write tests into sealed preparation')
    require(not TARGET.exists(), 'Real runtime must remain absent during preparation tests')
    scratch = REPO / ('synthetic-gate2-tests-' + uuid.uuid4().hex)
    scratch.mkdir()  # inherit workspace ACL; no Python-3.14 tempfile 0700 ACL
    counters = install_guard(writable=(BUNDLE, scratch))
    from gate2_common import STORE, V7_STORE
    actual_block_roots = tuple(str(p.absolute()).lower() for p in
                              (STORE / 'verified_clean_blocks', V7_STORE / 'verified_clean_blocks'))
    def synthetic_only(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, __import__('os').PathLike)):
            name = __import__('os').path.abspath(__import__('os').fsdecode(args[0])).lower()
            require(not any(name == root or name.startswith(root + __import__('os').sep)
                            for root in actual_block_roots), 'Real feature block forbidden in synthetic tests')
    sys.addaudithook(synthetic_only)
    authenticate_upstream()
    import partition_evidence
    partition_evidence.load_v8()
    import test_gate2
    test_gate2.SCRATCH = scratch
    test_gate2.SCIENCE = load_science()
    suite = unittest.defaultTestLoader.loadTestsFromModule(test_gate2)
    import test_storage_amendment
    test_storage_amendment.SCRATCH = scratch
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(test_storage_amendment))
    import test_replay_inventory
    test_replay_inventory.SCRATCH = scratch
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(test_replay_inventory))
    suite.addTests(frozen_regressions(test_gate2.SCIENCE))
    transcript = io.StringIO()
    result = unittest.TextTestRunner(stream=transcript, verbosity=2).run(suite)
    number = 1 + len(list(BUNDLE.glob('synthetic_test_attempt_*_results.json')))
    report = {'tests_run': result.testsRun, 'passed': result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped),
              'failed': len(result.failures), 'errors': len(result.errors), 'skipped': len(result.skipped),
              'success': result.wasSuccessful(), 'synthetic_fixtures_only': True,
              'real_fitting_executed': False, 'real_calibration_executed': False,
              'real_pseudo_generated': False, 'graph_constructed': False, 'GAT_executed': False,
              'real_feature_block_access_forbidden_by_audit_hook': True,
              'new_storage_tests_use_exact_authenticated_v8_and_v7_verifiers': True,
              'TEST_requests': 0, 'network_requests': 0, 'guard_counters': counters,
              'tested_python_sha256': {p.name: hash_file(p) for p in BUNDLE.glob('*.py')},
              'frozen_regression_source_hashes': {REGRESSION_PATH: hash_file(REPO / REGRESSION_PATH),
                   'v8/test_experiment_2b.py': hash_file(__import__('gate2_common').V8 / 'test_experiment_2b.py')},
              'frozen_regression_runner': 'Original selected AST function bodies with limited pytest assertion adapter, and original v8 OOVRuleTests under unittest',
              'attempt': number}
    report['success'] &= all(v == 0 for v in counters.values())
    if test_replay_inventory.SCHEMA_EVIDENCE is not None:
        write_new(BUNDLE/f'realistic_schema_regression_attempt_{number}_evidence.json',
                  pretty(test_replay_inventory.SCHEMA_EVIDENCE))
    write_new(BUNDLE / f'synthetic_test_attempt_{number}_transcript.txt', transcript.getvalue().encode())
    write_new(BUNDLE / f'synthetic_test_attempt_{number}_results.json', pretty(report))
    print(transcript.getvalue(), end='')
    print(json.dumps(report, indent=2, sort_keys=True))
    if report['success']:
        if (BUNDLE / 'synthetic_test_results.json').exists():
            os = __import__('os')
            os.rename(BUNDLE / 'synthetic_test_results.json', BUNDLE / f'synthetic_success_checkpoint_before_attempt_{number}.json')
        write_new(BUNDLE / 'synthetic_test_results.json', pretty(report))
    require(scratch.resolve() == scratch and scratch.is_relative_to(REPO) and
            scratch.name.startswith('synthetic-gate2-tests-'), 'Unsafe synthetic scratch cleanup')
    # Synthetic fixtures can be removed; transcripts and failures are retained.
    shutil.rmtree(scratch)
    return 0 if report['success'] else 1


if __name__ == '__main__':
    sys.exit(main())
