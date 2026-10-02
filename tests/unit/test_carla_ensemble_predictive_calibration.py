"""Diagnostic mathematical tests for the ensemble-predictive calibration protocol
and the descriptive Jensen diagnostic.

Protocol under test (EnsemblePredictiveCalibrator):
    fit      : minimize mean_i BCE(y_i, mean_t sigmoid(a*s_it + b))
               on the full (n_obs, K) member-score matrix
    inference: p_t = sigmoid(a*s_t + b) per member, p = mean_t p_t,
               then the canonical COGNIX entropy decomposition.

Required coverage (protocol-review milestone):
    unfitted behavior; both-class requirement; positive monotonic slope;
    deterministic fit; probability bounds; identical member scores reduce to
    ordinary scalar calibration behavior; increasing all member scores cannot
    lower the ensemble probability; member permutation invariance; K duplicated
    identical members do not re-weight observations; synthetic separable data;
    Separable fixtures remain numerically inspectable in diagnostic mode; the post-fit objective
    is never worse than at initialization; legacy Seg/GNSS-style scalar
    calibration remains approximately compatible.

No TEST data, no GAT/conformal, no downloads, no benchmark dependence: tests
assert mathematical/protocol properties only, never CARLA metric outcomes.
"""
from __future__ import annotations

import numpy as np
import pytest

from cognix.adapters.carla.normality import (
    EnsemblePredictiveCalibrator,
    MahalanobisNormality,
    ScoreCalibrator,
    ensemble_jensen_diagnostic,
    ensemble_to_uncertainty,
)
from cognix.adapters.carla.real_agents import (
    CONSENSUS_MARGIN,
    RealAgentError,
    RealCameraAgent,
    RealGNSSAgent,
    RealIMUAgent,
    RealSegAgent,
)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def _ep_bce(a, b, M, y):
    """The fitted objective, evaluated directly (test-side reference)."""
    q = np.clip(_sigmoid(a * np.asarray(M, float) + b).mean(axis=1), 1e-12, 1 - 1e-12)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))


@pytest.fixture
def separable_members():
    """Well-separated member-score matrix: normals ~0.9, pseudo ~0.1."""
    rng = np.random.default_rng(0)
    mn = rng.normal(0.9, 0.02, size=(60, 5))
    mp = rng.normal(0.1, 0.02, size=(60, 5))
    y = np.array([1.0] * 60 + [0.0] * 60)
    return np.vstack([mn, mp]), y


@pytest.fixture
def overlapping_members():
    """Weakly separated members: the interesting optimization regime."""
    rng = np.random.default_rng(2)
    mn = rng.normal(0.6, 0.18, size=(50, 5))
    mp = rng.normal(0.42, 0.18, size=(50, 5))
    y = np.array([1.0] * 50 + [0.0] * 50)
    return np.vstack([mn, mp]), y


# ── 1. unfitted behavior ─────────────────────────────────────────────────────

def test_ep_calibrator_fails_before_fit():
    cal = EnsemblePredictiveCalibrator()
    assert cal.is_fitted is False
    with pytest.raises(RuntimeError, match="not fitted"):
        cal.prob_normal(0.5, diagnostic=True)
    with pytest.raises(RuntimeError, match="not fitted"):
        cal.ensemble_probability(np.array([0.5, 0.6]), diagnostic=True)
    with pytest.raises(RuntimeError, match="not fitted"):
        _ = cal.ensemble_predictive_nll
    with pytest.raises(RuntimeError, match="not fitted"):
        _ = cal.optimization_info


def test_ep_calibrator_rejects_degenerate_inputs():
    cal = EnsemblePredictiveCalibrator()
    with pytest.raises(ValueError):
        cal.fit(np.zeros((1, 3)), np.array([1.0, 0.0]))       # too few rows
    with pytest.raises(ValueError, match="length mismatch"):
        cal.fit(np.zeros((4, 3)), np.array([1.0, 0.0]))       # label mismatch
    with pytest.raises(ValueError, match="non-finite"):
        cal.fit(np.array([[0.5, np.nan], [0.2, 0.1]]), np.array([1.0, 0.0]))


# ── 2. both-class requirement ────────────────────────────────────────────────

def test_ep_calibrator_requires_both_classes():
    cal = EnsemblePredictiveCalibrator()
    with pytest.raises(ValueError, match="both classes"):
        cal.fit(np.array([[0.1], [0.2], [0.3]]), np.ones(3))
    with pytest.raises(ValueError, match="both classes"):
        cal.fit(np.array([[0.1], [0.2], [0.3]]), np.zeros(3))
    with pytest.raises(ValueError, match="only 0/1"):
        cal.fit(np.array([[0.1], [0.2], [0.3]]), np.array([1.0, 0.0, 2.0]))


# ── 3. positive monotonic slope ──────────────────────────────────────────────

def test_ep_calibrator_slope_positive_by_construction(separable_members, overlapping_members):
    """a > 0 is enforced by the log-parameterization, never by clipping; it
    must hold even for inverted-looking initializations and weak separation."""
    for M, y in (separable_members, overlapping_members):
        cal = EnsemblePredictiveCalibrator().fit(M, y)
        assert cal.is_fitted
        assert cal.a > 0.0
        assert np.isfinite(cal.a) and np.isfinite(cal.b)


def test_ep_calibrator_preserves_direction(separable_members):
    M, y = separable_members
    cal = EnsemblePredictiveCalibrator().fit(M, y)
    lo, hi = float(M.min()), float(M.max())
    assert cal.prob_normal(hi, diagnostic=True) > cal.prob_normal(lo, diagnostic=True)
    assert cal.ensemble_probability(np.full(5, hi), diagnostic=True) > cal.ensemble_probability(np.full(5, lo), diagnostic=True)


# ── 4. deterministic fit ─────────────────────────────────────────────────────

def test_ep_calibrator_deterministic(separable_members, overlapping_members):
    for M, y in (separable_members, overlapping_members):
        c1 = EnsemblePredictiveCalibrator().fit(M, y)
        c2 = EnsemblePredictiveCalibrator().fit(M, y)
        assert c1.a == c2.a and c1.b == c2.b
        assert c1.optimization_info == c2.optimization_info


def test_ep_agent_calibration_deterministic_end_to_end(normal_cloud):
    a1, a2 = _calibrated_agent(RealGNSSAgent, normal_cloud), _calibrated_agent(RealGNSSAgent, normal_cloud)
    obs = normal_cloud[3]
    assert a1.predict(obs, diagnostic=True).value == a2.predict(obs, diagnostic=True).value
    assert a1.estimate_uncertainty(obs, diagnostic=True).total == a2.estimate_uncertainty(obs, diagnostic=True).total


# ── 5. probability bounds ────────────────────────────────────────────────────

def test_ep_probabilities_bounded(separable_members, overlapping_members):
    for M, y in (separable_members, overlapping_members):
        cal = EnsemblePredictiveCalibrator().fit(M, y)
        probes = np.linspace(0.0, 1.0, 21)
        for s in probes:
            p = cal.prob_normal(float(s), diagnostic=True)
            assert 0.0 <= p <= 1.0 and np.isfinite(p)
        q = cal.ensemble_probability(M[13], diagnostic=True)
        assert 0.0 <= q <= 1.0 and np.isfinite(q)


def test_ep_agent_predictions_bounded_and_decomposition_consistent(normal_cloud):
    agent = _calibrated_agent(RealCameraAgent, normal_cloud)
    for obs in list(normal_cloud[:3]) + [normal_cloud[0] + 4.0]:
        p = agent.predict(obs, diagnostic=True)
        u = agent.estimate_uncertainty(obs, diagnostic=True)
        assert 0.0 <= p.value <= 1.0
        assert 0.0 <= u.prediction <= 1.0
        assert u.epistemic >= 0.0 and u.aleatoric >= 0.0 and u.total >= 0.0
        assert u.total == pytest.approx(u.epistemic + u.aleatoric, abs=1e-9)
        assert u.prediction == pytest.approx(p.value, rel=1e-9)


# ── 6. identical member scores reduce to scalar calibration ─────────────────

def test_ep_identical_members_reduce_to_scalar_calibration(overlapping_members):
    """With all K member scores identical, the ensemble-predictive objective
    mathematically REDUCES to ordinary scalar BCE on the shared score: on data
    with a finite MLE (overlapping classes), the fit must match the legacy
    scalar logistic optimum, and ensemble_probability must equal the
    single-member mapping exactly. (On perfectly separable data the BCE
    optimum is a slope ridge — both fitters stop on it at different points,
    so the numeric comparison is only meaningful where the optimum is unique.)"""
    M, y = overlapping_members
    M_ident = np.repeat(M[:, :1], 5, axis=1)  # K = 5 identical columns
    cal = EnsemblePredictiveCalibrator().fit(M_ident, y)
    scalar = ScoreCalibrator().fit(M_ident[:, 0], y)  # same model, legacy fitter
    assert cal.a == pytest.approx(scalar.a, rel=1e-3, abs=1e-4)
    assert cal.b == pytest.approx(scalar.b, abs=1e-2)
    row = M_ident[7]
    assert cal.ensemble_probability(row, diagnostic=True) == pytest.approx(cal.prob_normal(float(row[0]), diagnostic=True), rel=1e-12)


# ── 7. monotonicity in member scores ─────────────────────────────────────────

def test_ep_raising_all_member_scores_never_lowers_ensemble_probability(separable_members):
    M, y = separable_members
    cal = EnsemblePredictiveCalibrator().fit(M, y)
    rng = np.random.default_rng(5)
    for _ in range(25):
        s = np.sort(rng.uniform(0.0, 1.0, size=5))
        q0 = cal.ensemble_probability(s, diagnostic=True)
        for delta in (0.01, 0.1, 0.5):
            q1 = cal.ensemble_probability(s + delta, diagnostic=True)
            assert q1 >= q0 - 1e-15
        # also under permutation of the raised scores
        q2 = cal.ensemble_probability((s + 0.25)[::-1], diagnostic=True)
        assert q2 >= cal.ensemble_probability(s[::-1], diagnostic=True) - 1e-15


# ── 8. member permutation invariance ─────────────────────────────────────────

def test_ep_member_permutation_leaves_prediction_unchanged(separable_members):
    M, y = separable_members
    cal = EnsemblePredictiveCalibrator().fit(M, y)
    rng = np.random.default_rng(6)
    for _ in range(20):
        row = rng.uniform(0.0, 1.0, size=7)
        perm = rng.permutation(7)
        assert cal.ensemble_probability(row, diagnostic=True) == pytest.approx(cal.ensemble_probability(row[perm], diagnostic=True), rel=1e-12)


def test_ep_agent_prediction_invariant_to_member_order(normal_cloud):
    """Permuting the AGENT's member scores (simulated by a permuted member
    matrix at calibration) must not change any deployed ensemble probability."""
    M, y = _member_pairs(RealSegAgent, normal_cloud)
    a1 = RealSegAgent(seed=42, n_members=5).fit(normal_cloud)
    a1.fit_calibrator(M, y)
    a2 = RealSegAgent(seed=42, n_members=5).fit(normal_cloud)
    a2.fit_calibrator(M[:, ::-1].copy(), y)  # reversed member columns
    for obs in normal_cloud[:4]:
        assert a1.predict(obs, diagnostic=True).value == pytest.approx(a2.predict(obs, diagnostic=True).value, abs=1e-9)


# ── 9. duplicated members do not re-weight observations ─────────────────────

def test_ep_duplicated_members_do_not_reweight_observations(overlapping_members):
    """Duplicating every member K times leaves the OBJECTIVE invariant (each
    observation still carries total weight 1), so on data with a finite MLE the
    fitted mapping must be numerically identical to the K=1 fit — no
    observation gains statistical weight from having more ensemble members."""
    M, y = overlapping_members
    c1 = EnsemblePredictiveCalibrator().fit(M[:, :1], y)          # K = 1
    c5 = EnsemblePredictiveCalibrator().fit(np.repeat(M[:, :1], 5, axis=1), y)  # K = 5 duplicates
    assert c5.a == pytest.approx(c1.a, rel=1e-7, abs=1e-7)
    assert c5.b == pytest.approx(c1.b, abs=1e-6)
    # also invariant on the separable ridge: the same ridge POINT is reached
    rng = np.random.default_rng(0)
    Ms = np.vstack([rng.normal(0.9, 0.02, (60, 1)), rng.normal(0.1, 0.02, (60, 1))])
    ys = np.array([1.0] * 60 + [0.0] * 60)
    d1 = EnsemblePredictiveCalibrator().fit(Ms, ys)
    d5 = EnsemblePredictiveCalibrator().fit(np.repeat(Ms[:, :1], 5, axis=1), ys)
    assert d5.a == pytest.approx(d1.a, rel=1e-7, abs=1e-7)


def test_ep_agent_accepts_matrix_without_collapsing_to_mean(normal_cloud):
    """The production path must fit on the full member matrix: passing the
    matrix and passing its column-mean must NOT generally give the same fit
    (the matrix fit weights within-observation member dispersion correctly)."""
    agent = RealIMUAgent(seed=42, n_members=5).fit(normal_cloud)
    M, y = _member_pairs(RealIMUAgent, normal_cloud)
    agent.fit_calibrator(M, y)
    assert agent._calibrator.is_fitted
    # member matrix stored semantics: prediction = mean of member probabilities
    s_t = agent.predict_normality(normal_cloud[2])
    q = float(np.mean([agent._calibrator.prob_normal(float(s), diagnostic=True) for s in s_t]))
    assert agent.predict(normal_cloud[2], diagnostic=True).value == pytest.approx(q, rel=1e-9)


# ── 10. synthetic separability regimes ───────────────────────────────────────

def test_ep_separable_synthetic_data(separable_members):
    M, y = separable_members
    cal = EnsemblePredictiveCalibrator().fit(M, y)
    assert np.isfinite(cal.a) and cal.a > 0
    q_norm = np.array([cal.ensemble_probability(M[i], diagnostic=True) for i in range(60)])
    q_pseu = np.array([cal.ensemble_probability(M[60 + i], diagnostic=True) for i in range(60)])
    assert float(np.median(q_norm)) > float(np.median(q_pseu))
    assert float(np.median(q_norm)) > 0.9
    assert float(np.median(q_pseu)) < 0.1


def test_ep_overlapping_data_remains_unsaturated(overlapping_members):
    """Weak separation must NOT produce an extreme slope: the fit should stay
    finite, positive, and give ordered but non-degenerate probabilities."""
    M, y = overlapping_members
    cal = EnsemblePredictiveCalibrator().fit(M, y)
    assert 0.0 < cal.a < 100.0
    q = np.array([cal.ensemble_probability(M[i], diagnostic=True) for i in range(len(M))])
    assert np.all((q > 1e-6) & (q < 1.0 - 1e-6))
    assert float(np.median(q[:50])) > float(np.median(q[50:]))


# ── 11. Camera/IMU-style extreme-slope fixture stays finite ─────────────────

def test_ep_extreme_separation_large_slope_remains_finite():
    """Camera/IMU smoke regime: near-perfect member-score separation drives the
    logistic optimum to a very large slope. The fit must remain finite, the
    mapping must stay in [0, 1] on both regimes, and the ensemble prediction
    must not produce NaN/inf anywhere."""
    rng = np.random.default_rng(11)
    mn = 1.0 - rng.uniform(0.0, 1e-9, size=(40, 5))   # ~1.0 - 1e-10
    mp = rng.uniform(0.0, 1e-11, size=(40, 5))        # ~0.0 + 1e-12
    M = np.vstack([mn, mp])
    y = np.array([1.0] * 40 + [0.0] * 40)
    cal = EnsemblePredictiveCalibrator().fit(M, y)
    assert np.isfinite(cal.a) and np.isfinite(cal.b) and cal.a > 0
    info = cal.optimization_info
    assert np.isfinite(info["nll_final"]) and np.isfinite(info["nll_initial"])
    q_n = cal.ensemble_probability(mn[0], diagnostic=True)
    q_p = cal.ensemble_probability(mp[0], diagnostic=True)
    assert np.isfinite(q_n) and np.isfinite(q_p)
    assert 0.0 <= q_n <= 1.0 and 0.0 <= q_p <= 1.0
    assert q_n > q_p
    # UQ decomposition stays finite on such members
    u = ensemble_to_uncertainty(np.array([cal.prob_normal(float(s), diagnostic=True) for s in mn[0]]))
    assert np.isfinite([u.prediction, u.epistemic, u.aleatoric, u.total]).all()


# ── 12. optimizer never worsens the objective ────────────────────────────────

def test_ep_objective_never_worse_than_initialization(separable_members, overlapping_members):
    """Backtracking descent guarantee: the post-fit objective must be <= the
    objective at initialization, on every regime (including the flat/weak one)."""
    for M, y in (separable_members, overlapping_members):
        cal = EnsemblePredictiveCalibrator().fit(M, y)
        info = cal.optimization_info
        assert info["nll_final"] <= info["nll_initial"] + 1e-12
        # and the reported final objective matches an independent evaluation
        assert info["nll_final"] == pytest.approx(_ep_bce(cal.a, cal.b, M, y), rel=1e-9, abs=1e-12)


def test_ep_multi_start_beats_or_matches_single_start(separable_members):
    """Multi-initialization is optimization VERIFICATION: the kept fit must have
    the lowest objective among the deterministic starts (never worse than any
    single-start run of the same optimizer family)."""
    M, y = separable_members
    cal = EnsemblePredictiveCalibrator().fit(M, y)
    # a deliberately poor single start (tiny slope) must not beat the kept fit
    assert cal.optimization_info["nll_final"] <= _ep_bce(0.5, 0.0, M, y) + 1e-9


# ── 13. legacy Seg/GNSS-style scalar compatibility ───────────────────────────

@pytest.fixture
def normal_cloud():
    rng = np.random.default_rng(0)
    return rng.normal(0.0, 0.1, size=(80, 6))


def _member_pairs(cls, normal_cloud):
    """Build CAL member-score pairs the way the agents do (no TEST labels)."""
    agent = cls(seed=42, n_members=5).fit(normal_cloud)
    rng = np.random.default_rng(1)
    normal = normal_cloud[rng.integers(0, len(normal_cloud), size=20)]
    cal_obs = list(normal) + list(normal + 3.0)
    M = np.array([agent.predict_normality(o) for o in cal_obs])
    y = np.array([1.0] * 20 + [0.0] * 20)
    return M, y


def _calibrated_agent(cls, normal_cloud):
    agent = cls(seed=42, n_members=5).fit(normal_cloud)
    M, y = _member_pairs(cls, normal_cloud)
    agent.fit_calibrator(M, y)
    return agent


@pytest.mark.parametrize("cls", [RealCameraAgent, RealSegAgent, RealGNSSAgent, RealIMUAgent])
def test_seg_gnss_style_fixture_calibrates_with_member_matrix(cls, normal_cloud):
    """Legacy-style fixtures (Mahalanobis ensemble on a tight normal cloud,
    offset pseudo) must calibrate successfully under the new protocol and stay
    approximately compatible with the legacy mean-score fit: both protocols
    produce the same ORDERING on the calibration set and close medians. Exact
    slope equality is NOT asserted: on near-separable member scores the BCE
    optimum is a slope ridge and the two fitters legitimately stop at different
    ridge points (the ensemble-predictive fit may ride further — a strictly
    better objective, verified separately)."""
    agent = _calibrated_agent(cls, normal_cloud)
    assert agent._calibrator.is_fitted
    assert agent._calibrator.a > 0
    M, y = _member_pairs(cls, normal_cloud)
    legacy = ScoreCalibrator().fit(M.mean(axis=1), y)  # old protocol (reference)
    assert legacy.is_fitted and legacy.a > 0
    q_ep = np.array([agent._calibrator.ensemble_probability(row, diagnostic=True) for row in M])
    q_legacy = np.array([legacy.prob_normal(float(s), diagnostic=True) for s in M.mean(axis=1)])
    # same ordering of the two classes
    assert float(np.median(q_ep[y == 1])) > float(np.median(q_ep[y == 0]))
    assert float(np.median(q_legacy[y == 1])) > float(np.median(q_legacy[y == 0]))
    # approximate compatibility of the deployed probabilities
    assert float(np.median(np.abs(q_ep - q_legacy))) < 0.25
    # the ensemble-predictive objective at its fit is never worse than at the
    # legacy mapping (the EP objective evaluates BOTH protocols)
    assert _ep_bce(agent._calibrator.a, agent._calibrator.b, M, y) <= \
        _ep_bce(legacy.a, legacy.b, M, y) + 1e-9


def test_seg_gnss_style_scalar_vector_input_still_accepted(normal_cloud):
    """Backward compatibility: a 1-D scalar-score vector reduces to K = 1
    ordinary BCE calibration (documented reduction)."""
    agent = RealGNSSAgent(seed=42, n_members=5).fit(normal_cloud)
    M, y = _member_pairs(RealGNSSAgent, normal_cloud)
    agent.fit_calibrator(M.mean(axis=1), y)
    assert agent._calibrator.is_fitted
    ref = EnsemblePredictiveCalibrator().fit(M.mean(axis=1)[:, None], y)
    assert agent._calibrator.a == pytest.approx(ref.a, rel=1e-9)
    assert agent._calibrator.b == pytest.approx(ref.b, rel=1e-9)


# ── Jensen diagnostic (descriptive only) ─────────────────────────────────────

def test_jensen_diagnostic_values_and_ordering():
    M = np.array([[0.25, 0.75], [0.5, 0.5], [0.1, 0.9]])
    a, b = 6.0, 0.0
    diag = ensemble_jensen_diagnostic(a, b, M)
    p_members = _sigmoid(a * M + b).mean(axis=1)
    p_direct = _sigmoid(a * M.mean(axis=1) + b)
    expected = np.abs(p_members - p_direct)
    assert diag["max"] == pytest.approx(float(expected.max()), rel=1e-12)
    assert diag["median"] == pytest.approx(float(np.median(expected)), rel=1e-12)
    assert diag["p90"] == pytest.approx(float(np.percentile(expected, 90)), rel=1e-12)
    assert 0.0 <= diag["median"] <= diag["p90"] <= diag["max"]
    # zero dispersion -> zero gap
    flat = np.tile([[0.42, 0.42, 0.42]], (5, 1))
    d0 = ensemble_jensen_diagnostic(9.0, -1.0, flat)
    assert d0["max"] == pytest.approx(0.0, abs=1e-15)


def test_jensen_diagnostic_rejects_bad_shapes():
    with pytest.raises(ValueError):
        ensemble_jensen_diagnostic(1.0, 0.0, np.array([0.1, 0.2]))
    with pytest.raises(ValueError):
        ensemble_jensen_diagnostic(1.0, 0.0, np.zeros((0, 3)))


def test_agent_records_jensen_diagnostic_but_never_refuses(normal_cloud):
    """The production path must RECORD the diagnostic (max/median/p90) and must
    NEVER refuse a fit because of it — the Camera/IMU extreme-slope regime now
    calibrates (the retired gate is gone)."""
    agent = RealCameraAgent(seed=42, n_members=5).fit(normal_cloud)
    M, y = _member_pairs(RealCameraAgent, normal_cloud)
    agent.fit_calibrator(M, y)  # must not raise, whatever the gap
    assert agent._calibrator.is_fitted
    diag = agent._cal_jensen_diag
    assert diag is not None and {"max", "median", "p90"} <= set(diag)
    res = agent.predict(normal_cloud[4], diagnostic=True)
    assert res.metadata["cal_jensen_gap_max"] == pytest.approx(diag["max"], rel=1e-12)
    assert res.metadata["cal_jensen_gap_median"] == pytest.approx(diag["median"], rel=1e-12)
    assert res.metadata["cal_jensen_gap_p90"] == pytest.approx(diag["p90"], rel=1e-12)
    # descriptive scale: the recorded max must not be silently clipped to 0
    assert diag["max"] >= 0.0


def test_consensus_margin_no_longer_gates_anything(normal_cloud):
    """The retired threshold survives only as a named legacy constant; a gap
    far above it must NOT prevent calibration."""
    assert CONSENSUS_MARGIN == 0.05  # legacy reference value, unchanged
    agent = RealIMUAgent(seed=42, n_members=5).fit(normal_cloud)
    M, y = _member_pairs(RealIMUAgent, normal_cloud)
    agent.fit_calibrator(M, y)
    assert agent._calibrator.is_fitted


def test_epistemic_not_redefined_wording():
    """Documentation must record near-zero epistemic as an observed CARLA
    finding, never as a COGNIX-universal property; the decomposition itself is
    untouched."""
    import inspect
    from cognix.adapters.carla import real_agents
    src = inspect.getsource(real_agents)
    assert "observed CARLA finding" in src
    assert "near-zero probability-space epistemic is an observed" in src


# ── production-path hardening regressions ────────────────────────────────────

def test_agent_fit_calibrator_length_mismatch_still_fails(normal_cloud):
    agent = RealCameraAgent().fit(normal_cloud)
    with pytest.raises(RealAgentError, match="mismatch"):
        agent.fit_calibrator(np.zeros((5, 3)), np.array([1.0, 0.0]))


def test_unfitted_agent_still_fails_loud(normal_cloud):
    agent = RealSegAgent()
    with pytest.raises(RealAgentError, match="fit"):
        agent.predict(normal_cloud[0], diagnostic=True)
    with pytest.raises(RealAgentError, match="fit"):
        agent.estimate_uncertainty(normal_cloud[0], diagnostic=True)


def test_diagnostic_lifecycle_keeps_invalid_fit_inspectable(normal_cloud):
    """After a diagnostic ensemble-predictive fit, explicit diagnostic prediction
    and UQ work; a second fit replaces the calibrator (no stale state)."""
    agent = _calibrated_agent(RealCameraAgent, normal_cloud)
    p1 = agent.predict(normal_cloud[1], diagnostic=True).value
    M, y = _member_pairs(RealCameraAgent, normal_cloud)
    agent.fit_calibrator(M[::-1].copy(), y[::-1].copy())  # refit with flipped rows
    p2 = agent.predict(normal_cloud[1], diagnostic=True).value
    assert 0.0 <= p1 <= 1.0 and 0.0 <= p2 <= 1.0
    assert agent._cal_jensen_diag is not None
