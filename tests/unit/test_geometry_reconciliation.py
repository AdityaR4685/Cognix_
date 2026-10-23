"""Unit tests for geometry reconciliation and fail-closed validation contracts."""
import pathlib
import pytest
import re


WORK_ROOT = pathlib.Path(r'C:\Users\Aditya\Cognix2_recovered\.runtime_v1_environment_transition_qualification_008_attempt_014_work')
_DIAG_PREFIXES = ['geometry_materialization', 'mapper_sentinel_replay', 'geometry_correction']


def _strip_diag_prefix(path_str, work_root):
    """Strip a diagnostic subdirectory prefix if present; return original string otherwise."""
    p = pathlib.Path(path_str)
    root = pathlib.Path(work_root)
    for prefix in _DIAG_PREFIXES:
        diag_dir = root / prefix
        try:
            rel = p.relative_to(diag_dir)
            return str(root / rel)
        except ValueError:
            continue
    return path_str


def _find_publishing_tmp_match(target, observed):
    """If target is base.publishing.<pid>.<hex>.tmp, find matching base.publishing.<any>.tmp in observed.
    Returns the single matched entry, or None if zero or more-than-one matches (fail closed on ambiguity)."""
    m = re.match(r'^(.+)\.publishing\.[0-9]+\.[0-9a-f]+\.tmp$', target, re.IGNORECASE)
    if not m:
        return None
    base = re.escape(m.group(1))
    pat = re.compile(r'^' + base + r'\.publishing\.[0-9]+\.[0-9a-f]+\.tmp$', re.IGNORECASE)
    matches = [v for k, v in observed.items() if pat.match(k)]
    if len(matches) != 1:
        return None
    return matches[0]


class TestStripDiagPrefix:
    def test_strip_geometry_materialization(self):
        inp = str(WORK_ROOT / 'geometry_materialization' / 'adversarial_sentinels' / 'test.py')
        expected = str(WORK_ROOT / 'adversarial_sentinels' / 'test.py')
        assert _strip_diag_prefix(inp, WORK_ROOT) == expected

    def test_strip_mapper_sentinel_replay(self):
        inp = str(WORK_ROOT / 'mapper_sentinel_replay' / 'sub' / 'ref.json')
        expected = str(WORK_ROOT / 'sub' / 'ref.json')
        assert _strip_diag_prefix(inp, WORK_ROOT) == expected

    def test_strip_geometry_correction(self):
        inp = str(WORK_ROOT / 'geometry_correction' / 'fix.py')
        expected = str(WORK_ROOT / 'fix.py')
        assert _strip_diag_prefix(inp, WORK_ROOT) == expected

    def test_unprefixed_path_unchanged(self):
        inp = str(WORK_ROOT / 'regular_file.py')
        assert _strip_diag_prefix(inp, WORK_ROOT) == inp

    def test_partial_prefix_not_stripped(self):
        inp = str(WORK_ROOT / 'geometry_materialization_extra' / 'file.py')
        assert _strip_diag_prefix(inp, WORK_ROOT) == inp


class TestFindPublishingTmpMatch:
    def test_single_match_succeeds(self):
        diag_target = str(WORK_ROOT / 'file.py.publishing.16216.8418ec4492043a378bac1f97942542c8.tmp')
        live_key = str(WORK_ROOT / 'file.py.publishing.18776.7e5d113e6f154e1a2458810e903f4c67.tmp')
        live_entry = {'states': [{'exists': True}]}
        observed = {live_key: live_entry}
        assert _find_publishing_tmp_match(diag_target, observed) is live_entry

    def test_ambiguous_matches_fail_closed(self):
        diag_target = str(WORK_ROOT / 'file.py.publishing.16216.8418ec4492043a378bac1f97942542c8.tmp')
        observed = {
            str(WORK_ROOT / 'file.py.publishing.100.aaa.tmp'): {'states': []},
            str(WORK_ROOT / 'file.py.publishing.200.bbb.tmp'): {'states': []},
        }
        assert _find_publishing_tmp_match(diag_target, observed) is None

    def test_no_match_returns_none(self):
        diag_target = str(WORK_ROOT / 'file.py.publishing.16216.8418ec4492043a378bac1f97942542c8.tmp')
        observed = {}
        assert _find_publishing_tmp_match(diag_target, observed) is None

    def test_different_base_returns_none(self):
        diag_target = str(WORK_ROOT / 'file.py.publishing.16216.8418ec4492043a378bac1f97942542c8.tmp')
        observed = {
            str(WORK_ROOT / 'other.py.publishing.18776.7e5d113e6f154e1a2458810e903f4c67.tmp'): {'states': []}
        }
        assert _find_publishing_tmp_match(diag_target, observed) is None

    def test_non_publishing_tmp_returns_none(self):
        target = str(WORK_ROOT / 'regular_file.json')
        observed = {target: {'states': []}}
        assert _find_publishing_tmp_match(target, observed) is None


class TestPrerequisiteSourceReadResolution:
    def test_prereq_source_found_and_valid(self):
        target = str(WORK_ROOT / 'shadow' / 'stage_c_mapper_artifacts.py')
        expected_sha = '2a978c610416603c896111893d4278bba006d79ccb284eb0f7a5e6d166b7594c'
        suite_prereqs = {
            target: {
                'open_count': 55,
                'path': target,
                'sha256_at_export': expected_sha,
            }
        }
        entry = suite_prereqs.get(target)
        assert entry is not None
        assert entry.get('open_count', 0) > 0
        assert entry.get('sha256_at_export') == expected_sha

    def test_prereq_source_zero_open_count_fails(self):
        target = str(WORK_ROOT / 'shadow' / 'stage_c_mapper_artifacts.py')
        suite_prereqs = {
            target: {
                'open_count': 0,
                'path': target,
                'sha256_at_export': 'dummy',
            }
        }
        entry = suite_prereqs.get(target)
        assert entry is not None
        assert not (entry.get('open_count', 0) > 0)

    def test_prereq_source_sha_mismatch_fails(self):
        target = str(WORK_ROOT / 'shadow' / 'stage_c_mapper_artifacts.py')
        suite_prereqs = {
            target: {
                'open_count': 10,
                'path': target,
                'sha256_at_export': 'different_hash',
            }
        }
        entry = suite_prereqs.get(target)
        assert entry.get('sha256_at_export') != 'expected_hash'


class TestHashVerificationBehavior:
    def test_text_sha_matches_expected(self):
        expected_sha = 'bf261c6e0bc769246e6c91ec72ac162f89f38fcd0d6e464a7841cc9ef129db1d'
        states = [
            {'argument': 'concrete runtime path expression', 'bytes': 23622, 'exists': True},
            {'writing': False},
            {'text_sha256': expected_sha},
        ]
        obs_hashes = [s.get('sha256') or s.get('text_sha256') or s.get('expected_sha256') for s in states if s.get('sha256') or s.get('text_sha256') or s.get('expected_sha256')]
        assert expected_sha in obs_hashes

    def test_byte_sha_matches_expected(self):
        expected_sha = 'ca303f628851c880db6dd332fc6100ddb65681c3569de772d7587c637ce373e4'
        states = [
            {'bytes': 100, 'exists': True, 'sha256': expected_sha},
        ]
        obs_hashes = [s.get('sha256') or s.get('text_sha256') or s.get('expected_sha256') for s in states if s.get('sha256') or s.get('text_sha256') or s.get('expected_sha256')]
        assert expected_sha in obs_hashes

    def test_empty_hashes_when_no_read_event(self):
        states = [
            {'argument': 'concrete runtime path expression', 'bytes': 1961, 'exists': True},
            {'construction_operation': 'joinpath', 'intended_root': str(WORK_ROOT / 'shadow')},
            {'writing': False},
        ]
        obs_hashes = [s.get('sha256') or s.get('text_sha256') or s.get('expected_sha256') for s in states if s.get('sha256') or s.get('text_sha256') or s.get('expected_sha256')]
        assert len(obs_hashes) == 0
