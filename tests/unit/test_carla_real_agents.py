"""
Focused tests for the Experiment-1 normality layer and real agents (Stage 1,
hardened).

Covers:
1.  predict_normality() API consistency: raw one-class statistic is the
    primary output; probability appears only via explicit calibration.
2.  Normal-only fit contract; raw-score bounds [0, 1]; ordering.
3.  High-dimensional stability regression (d = 6, 32, 128, 512): normal
    observations must NOT collapse to score ~ 0 due to dimensionality;
    mahalanobis_distance2 provenance stays available.
4.  ScoreCalibrator hardening: {0,1} labels, both classes, fitted-state guard,
    direction preservation, set-size-invariant optimization, determinism.
5.  Ensemble uncertainty decomposition (canonical COGNIX semantics; identical
    member PROBABILITIES -> epistemic ~ 0; disagreeing members -> positive).
6.  Constructed normality probability contract for all four real-agent wrappers.
7.  Observation-dictionary failure behavior (missing key / explicit None).
8.  Ensemble calibration protocol (ensemble-predictive fit + descriptive Jensen diagnostic).
9.  No anomaly-label dependency; frozen synthetic modules untouched;
    `import cognix` must not import CARLA adapters; scientific wording.

No downloads, no pseudo-anomalies, no artifact training, no refits.
"""
from __future__ import annotations

import numpy as np
import pytest

from cognix.core.types import PredictionResult, UncertaintyResult

from cognix.adapters.carla.normality import (
    BootstrapNormalityEnsemble,
    KNNNormality,
    MahalanobisNormality,
    NormalityModel,
    ScoreCalibrator,
    ensemble_to_uncertainty,
)
from cognix.adapters.carla.real_agents import (
    CONSENSUS_MARGIN,
    RealAgentError,
    RealCameraAgent,
    RealGNSSAgent,
    RealIMUAgent,
    RealNormalityAgent,
    RealSegAgent,
)


# ── fixture data (precomputed compact vectors; no raw images) ────────────────

@pytest.fixture
def normal_features():
    rng = np.random.default_rng(0)
    return rng.normal(0.0, 0.1, size=(80, 6))


@pytest.fixture
def cal_pairs(normal_features):
    """CAL pairs: in-cloud rows normal, offset rows pseudo-corrupted (no TEST labels)."""
    rng = np.random.default_rng(1)
    normal = normal_features[rng.integers(0, len(normal_features), size=20)]
    pseudo = normal + 3.0
    model = MahalanobisNormality().fit(normal_features)
    scores = [model.predict_normality(r) for r in normal] + [
        model.predict_normality(r) for r in pseudo
    ]
    target_normal = np.array([1.0] * 20 + [0.0] * 20)
    return np.array(scores), target_normal


def _member_matrix(agent, observations):
    """(n_obs, K) member-score matrix per the calibration protocol."""
    return np.array([agent.predict_normality(o) for o in observations])


def _build(cls, normal_features, cal_observations=None, pseudo_offset=3.0):
    """Build + fit + calibrate an agent per the ensemble calibration protocol:
    fit on the full (n_obs, K) matrix; Jensen gap is descriptive.
    Separable fixture probabilities are requested explicitly in diagnostic mode."""
    agent = cls(seed=42, n_members=5)
    agent.fit(normal_features)
    rng = np.random.default_rng(1)
    normal = normal_features[rng.integers(0, len(normal_features), size=20)]
    pseudo = normal + pseudo_offset
    cal_obs = list(normal) + list(pseudo)
    target_normal = np.array([1.0] * 20 + [0.0] * 20)
    member_matrix = _member_matrix(agent, cal_obs)
    agent.fit_calibrator(member_matrix, target_normal)
    return agent


# ── 1. API consistency ───────────────────────────────────────────────────────

def test_predict_normality_is_primary_model_api(normal_features):
    m = MahalanobisNormality().fit(normal_features)
    assert isinstance(m, NormalityModel)
    s = m.predict_normality(normal_features[0])
    assert isinstance(s, float) and 0.0 <= s <= 1.0
    # score() is a documented backward-compatible alias, not a probability
    assert m.score(normal_features[0]) == s


def test_ensemble_predict_normality_shape_and_aliases(normal_features):
    ens = BootstrapNormalityEnsemble(n_members=5, seed=42).fit(normal_features)
    v = ens.predict_normality(normal_features[0])
    assert v.shape == (5,)
    assert ens.scores(normal_features[0]) is not None  # alias kept
    assert np.array_equal(ens.scores(normal_features[0]), v)
    assert ens.mean_score(normal_features[0]) == pytest.approx(float(v.mean()))


def test_agent_predict_normality_is_raw_not_probability(normal_features):
    a = _build(RealCameraAgent, normal_features)
    v = a.predict_normality(normal_features[3])
    assert v.shape == (5,)
    assert np.all((v >= 0.0) & (v <= 1.0))
    # agent-level raw API must exist and differ in name/semantics from predict()
    assert a.normality_score(normal_features[3]) == pytest.approx(float(v.mean()))


# ── 2. fit contract, bounds, ordering ────────────────────────────────────────

def test_fit_rejects_degenerate_input():
    m = MahalanobisNormality()
    with pytest.raises(ValueError):
        m.fit(np.zeros((1, 4)))
    with pytest.raises(ValueError):
        m.fit(np.zeros(4))
    with pytest.raises(ValueError):
        m.fit(np.full((5, 4), np.nan))
    with pytest.raises(ValueError):
        KNNNormality(k=5).fit(np.zeros((3, 4)))


def test_normality_higher_inside_cloud_than_off_manifold(normal_features):
    inside = normal_features[0]
    outside = normal_features[0] + 5.0
    for m in (
        MahalanobisNormality().fit(normal_features),
        KNNNormality(k=3).fit(normal_features),
    ):
        assert m.predict_normality(inside) > m.predict_normality(outside)


def test_raw_scores_bounded_zero_one(normal_features):
    probes = [normal_features[0], normal_features.mean(axis=0), normal_features[3] + 9.0]
    for m in (
        MahalanobisNormality().fit(normal_features),
        KNNNormality(k=5).fit(normal_features),
    ):
        for p in probes:
            assert 0.0 <= m.predict_normality(p) <= 1.0


def test_mahalanobis_distance2_provenance_available(normal_features):
    m = MahalanobisNormality().fit(normal_features)
    d2 = m.mahalanobis_distance2(normal_features[0] + 5.0)
    assert d2 > 0.0 and np.isfinite(d2)
    with pytest.raises(RuntimeError):
        MahalanobisNormality().mahalanobis_distance2(normal_features[0])


def test_mahalanobis_no_double_centering_nonzero_mean_features():
    """Regression: mahalanobis_distance2 must center ONCE. An earlier version
    pre-centered the input and _distance2 centered again (x - 2*mean), which is
    a no-op on zero-mean fixtures but inflated every query distance by orders of
    magnitude for realistic nonzero-mean features (e.g. GNSS lat ~30 deg).
    A TRAINING ROW (d2_train contributing to m2) must query consistently and a
    fresh near-mean row must score as clearly normal."""
    rng = np.random.default_rng(13)
    base = np.array([100.0, 30.0, 5.0, 0.5, -2.0, 7.0])
    x = base + rng.normal(0.0, 0.1, size=(60, 6))
    m = MahalanobisNormality().fit(x)
    # consistency: querying a training row must give ~the fitted mean scale,
    # not an astronomically inflated value
    d2_self = m.mahalanobis_distance2(x[7])
    assert 0.0 <= d2_self <= 10.0 * m._d2_scale
    # fresh normal draw (same distribution) must score as normal, not ~0
    fresh = base + rng.normal(0.0, 0.1, size=6)
    s = m.predict_normality(fresh)
    assert s > 0.3, f"fresh normal row collapsed at nonzero mean: score={s:.3e}"
    # exact math identity: score == exp(-0.5 * d2 / m2) on the SAME d2
    assert s == pytest.approx(float(np.exp(-0.5 * m.mahalanobis_distance2(fresh) / m._d2_scale)), rel=1e-9)


# ── 3. high-dimensional stability (regression) ──────────────────────────────

@pytest.mark.parametrize("d", [6, 32, 128, 512])
def test_highdim_normal_rows_do_not_collapse_to_zero(d):
    """Regression: with the dimension-stable statistic, fresh NORMAL rows must
    keep non-degenerate scores at any realistic dimension (the un-normalized
    exp(-0.5*d2) gave ~1e-35 at d=128 and ~1e-137 at d=512)."""
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=(4 * d, d))
    m = MahalanobisNormality(shrinkage=0.1).fit(x)
    fresh = np.array([m.predict_normality(rng.normal(0, 1, d)) for _ in range(50)])
    assert np.all(np.isfinite(fresh))
    assert np.all((fresh >= 0.0) & (fresh <= 1.0))
    assert int((fresh == 0.0).sum()) == 0, "normal rows must not underflow to 0"
    assert float(np.median(fresh)) >= 0.15, (
        f"normal-row median score collapsed at d={d}: {np.median(fresh):.3e}"
    )


@pytest.mark.parametrize("d", [32, 128])
def test_highdim_median_score_dimension_stable(d):
    """Median normal-row score must stay in the same band across dimensions
    (no exponential-in-d decay) — the core property of the d2/m2 statistic."""
    meds = []
    for dd in (d, 6):
        rng = np.random.default_rng(0)
        x = rng.normal(0.0, 1.0, size=(4 * dd, dd))
        m = MahalanobisNormality(shrinkage=0.1).fit(x)
        fresh = np.array([m.predict_normality(rng.normal(0, 1, dd)) for _ in range(50)])
        meds.append(float(np.median(fresh)))
    ratio = max(meds) / min(meds)
    assert ratio < 5.0, f"median scores not dimension-stable: {meds}"


def test_highdim_monotonicity_and_provenance_scale():
    d = 128
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=(4 * d, d))
    m = MahalanobisNormality(shrinkage=0.1).fit(x)
    center = m.predict_normality(x.mean(axis=0))
    typical = m.predict_normality(rng.normal(0, 1, d))
    far = m.predict_normality(rng.normal(0, 1, d) + 8.0)
    assert center > typical > far
    # provenance: fresh-normal d2 should be on the order of d (chi-square-like)
    d2 = m.mahalanobis_distance2(rng.normal(0, 1, d))
    assert 0.4 * d < d2 < 4.0 * d


def test_highdim_deterministic(normal_features):
    d = 128
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=(4 * d, d))
    m1 = MahalanobisNormality().fit(x)
    m2 = MahalanobisNormality().fit(x.copy())
    q = rng.normal(0, 1, d)
    assert m1.predict_normality(q) == m2.predict_normality(q)
    assert m1.mahalanobis_distance2(q) == m2.mahalanobis_distance2(q)


def test_ensemble_highdim_agent_end_to_end():
    d = 128
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=(200, d))
    a = RealCameraAgent(seed=42, n_members=3)
    a.fit(x)
    normal = x[:10]
    pseudo = x[:10] + 12.0
    cal_obs = list(normal) + list(pseudo)
    a.fit_calibrator(_member_matrix(a, cal_obs), np.array([1.0] * 10 + [0.0] * 10))
    p_norm = a.predict(x[50], diagnostic=True).value
    p_far = a.predict(x[50] + 12.0, diagnostic=True).value
    assert 0.0 <= p_far < p_norm <= 1.0


# ── 4. ScoreCalibrator hardening ────────────────────────────────────────────

def test_calibrator_rejects_non_binary_labels():
    cal = ScoreCalibrator()
    with pytest.raises(ValueError, match="only 0/1"):
        cal.fit(np.array([0.2, 0.4, 0.6]), np.array([0.0, 1.0, 2.0]))
    with pytest.raises(ValueError, match="only 0/1"):
        cal.fit(np.array([0.2, 0.4]), np.array([0.0, 0.5]))


def test_calibrator_requires_both_classes():
    cal = ScoreCalibrator()
    with pytest.raises(ValueError, match="both classes"):
        cal.fit(np.array([0.1, 0.2, 0.3]), np.ones(3))
    with pytest.raises(ValueError, match="both classes"):
        cal.fit(np.array([0.1, 0.2, 0.3]), np.zeros(3))


def test_calibrator_fails_before_fit():
    cal = ScoreCalibrator()
    assert cal.is_fitted is False
    with pytest.raises(RuntimeError, match="not fitted"):
        cal.prob_normal(0.7, diagnostic=True)


def test_calibrator_preserves_direction(normal_features, cal_pairs):
    scores, target_normal = cal_pairs
    cal = ScoreCalibrator().fit(scores, target_normal)
    assert cal.is_fitted
    lo, hi = float(scores.min()), float(scores.max())
    assert cal.prob_normal(hi, diagnostic=True) > cal.prob_normal(lo, diagnostic=True)


def test_calibrator_refuses_inverted_semantics(normal_features, cal_pairs):
    """Labels flipped (normal = LOW score) must be refused, not silently fit."""
    scores, target_normal = cal_pairs
    with pytest.raises(ValueError, match="inverts the semantic direction"):
        ScoreCalibrator().fit(scores, 1.0 - target_normal)


def test_calibrator_optimization_invariant_to_set_size(normal_features, cal_pairs):
    """Duplicating the calibration set must not change the fitted mapping
    (Newton on a scaled log-likelihood): guards against learning rates that
    scale with calibration-set size."""
    scores, target_normal = cal_pairs
    c1 = ScoreCalibrator().fit(scores, target_normal)
    c2 = ScoreCalibrator().fit(np.tile(scores, 10), np.tile(target_normal, 10))
    assert c1.a == pytest.approx(c2.a, rel=1e-6, abs=1e-9)
    assert c1.b == pytest.approx(c2.b, rel=1e-6, abs=1e-9)


def test_calibrator_deterministic(normal_features, cal_pairs):
    scores, target_normal = cal_pairs
    a = ScoreCalibrator().fit(scores, target_normal)
    b = ScoreCalibrator().fit(scores, target_normal)
    assert a.a == b.a and a.b == b.b


def test_calibrator_rejects_nonfinite_scores():
    with pytest.raises(ValueError, match="non-finite"):
        ScoreCalibrator().fit(np.array([0.5, np.nan]), np.array([1.0, 0.0]))


# ── 5. uncertainty decomposition (canonical semantics) ──────────────────────

def test_uncertainty_bounds_and_finiteness():
    rng = np.random.default_rng(3)
    for _ in range(20):
        p = rng.uniform(0.0, 1.0, size=int(rng.integers(1, 8)))
        u = ensemble_to_uncertainty(p)
        assert isinstance(u, UncertaintyResult)
        assert 0.0 <= u.prediction <= 1.0
        assert u.epistemic >= 0.0 and u.aleatoric >= 0.0 and u.total >= 0.0
        assert np.isfinite([u.prediction, u.epistemic, u.aleatoric, u.total]).all()


def test_identical_member_probabilities_give_near_zero_epistemic():
    u = ensemble_to_uncertainty(np.full(7, 0.83))
    assert u.epistemic == pytest.approx(0.0, abs=1e-9)
    assert u.aleatoric > 0.0


def test_disagreeing_members_give_positive_epistemic():
    p = np.array([0.05, 0.95, 0.4, 0.6])
    u = ensemble_to_uncertainty(p)
    assert u.epistemic > 0.05
    h = lambda q: -(q * np.log(q) + (1 - q) * np.log(1 - q))
    pbar = p.mean()
    assert u.total == pytest.approx(float(h(pbar)), rel=1e-9)
    assert u.aleatoric == pytest.approx(float(np.mean([h(q) for q in p])), rel=1e-9)
    assert u.epistemic == pytest.approx(max(0.0, u.total - u.aleatoric), rel=1e-9)


def test_ensemble_rejects_invalid_probabilities():
    with pytest.raises(ValueError):
        ensemble_to_uncertainty(np.array([0.5, 1.4]))
    with pytest.raises(ValueError):
        ensemble_to_uncertainty(np.array([]))
    with pytest.raises(ValueError):
        ensemble_to_uncertainty(np.array([np.nan, 0.5]))


# ── 6. real-agent wrappers: canonical contract ──────────────────────────────

ALL_AGENT_CLASSES = [RealCameraAgent, RealSegAgent, RealGNSSAgent, RealIMUAgent]


@pytest.fixture
def fitted_agents(normal_features):
    out = {}
    for cls in ALL_AGENT_CLASSES:
        out[cls] = _build(cls, normal_features, None)
    return out


@pytest.mark.parametrize("cls", ALL_AGENT_CLASSES)
def test_all_four_agents_emit_prob_normal(cls, normal_features, fitted_agents):
    agent = fitted_agents[cls]
    res = agent.predict(normal_features[3], diagnostic=True)
    assert isinstance(res, PredictionResult)
    assert 0.0 <= res.value <= 1.0
    assert res.confidence == res.value
    assert res.metadata["prediction_semantics"] == "P(target=normal) under TRAIN-derived clean-vs-pseudo calibration"


@pytest.mark.parametrize("cls", ALL_AGENT_CLASSES)
def test_all_four_agents_emit_genuine_ensemble_uncertainty(cls, normal_features, fitted_agents):
    agent = fitted_agents[cls]
    u = agent.estimate_uncertainty(normal_features[3], diagnostic=True)
    assert isinstance(u, UncertaintyResult)
    assert 0.0 <= u.prediction <= 1.0
    assert u.epistemic >= 0.0 and u.aleatoric >= 0.0 and u.total >= 0.0
    assert u.total == pytest.approx(u.epistemic + u.aleatoric, abs=1e-9)
    assert u.prediction == pytest.approx(agent.predict(normal_features[3], diagnostic=True).value, rel=1e-9)


@pytest.mark.parametrize("cls", ALL_AGENT_CLASSES)
def test_off_manifold_input_scores_less_normal(cls, normal_features, fitted_agents):
    agent = fitted_agents[cls]
    normal_obs = normal_features[3]
    weird_obs = normal_features[3] + 4.0
    assert agent.predict(weird_obs, diagnostic=True).value < agent.predict(normal_obs, diagnostic=True).value


def test_agent_seed_reproducibility(normal_features):
    def make():
        a = RealGNSSAgent(seed=42, n_members=5)
        a.fit(normal_features)
        a.fit_calibrator(*_pair_args(a, normal_features))
        return a

    def _pair_args(agent, x):
        rng = np.random.default_rng(1)
        normal = x[rng.integers(0, len(x), size=20)]
        cal_obs = list(normal) + list(normal + 3.0)
        return (
            _member_matrix(agent, cal_obs).mean(axis=1),
            np.array([1.0] * 20 + [0.0] * 20),
        )

    a1, a2 = make(), make()
    obs = normal_features[7]
    assert a1.predict(obs, diagnostic=True).value == a2.predict(obs, diagnostic=True).value
    assert a1.estimate_uncertainty(obs, diagnostic=True).epistemic == a2.estimate_uncertainty(obs, diagnostic=True).epistemic


def test_agent_metadata_precise_uq_wording(normal_features):
    a = RealGNSSAgent(seed=123, n_members=4)
    md = a.metadata()
    assert md["uq_method"] == "bootstrap_oneclass_ensemble"
    assert md["uq_source"] == "model_disagreement"
    assert md["synthetic_probability_noise"] is False
    assert md["seed"] == 123 and md["n_members"] == 4
    assert md["prediction_semantics"] == "P(target=normal) under TRAIN-derived clean-vs-pseudo calibration"
    assert "genuine_uq" not in md
    # module sources must not over-claim
    import inspect
    from cognix.adapters.carla import normality, real_agents
    for mod in (normality, real_agents):
        assert "genuine_uq" not in inspect.getsource(mod)


# ── 7. observation-dictionary failure behavior ──────────────────────────────

def test_missing_dict_key_fails_loud():
    a = RealGNSSAgent()
    with pytest.raises(RealAgentError, match="absent from the observation"):
        a.extract_vector({"Camera": np.zeros(4)})


def test_explicit_none_slice_fails_loud():
    a = RealGNSSAgent()
    with pytest.raises(RealAgentError, match="slice is None"):
        a.extract_vector({"GNSS": None})


def test_whole_dict_never_reinterpreted_as_input():
    a = RealIMUAgent()
    with pytest.raises(RealAgentError):
        a.extract_vector({"unrelated": "payload"})


def test_scalar_observation_still_supported(normal_features):
    a = RealCameraAgent()
    a.fit(normal_features)
    a.fit_calibrator(*_pair_args(a, normal_features))
    v = a.extract_vector(normal_features[0])
    assert v.shape == (6,)


def _pair_args(agent, x):
    rng = np.random.default_rng(1)
    normal = x[rng.integers(0, len(x), size=20)]
    cal_obs = list(normal) + list(normal + 3.0)
    return (
        _member_matrix(agent, cal_obs).mean(axis=1),
        np.array([1.0] * 20 + [0.0] * 20),
    )


# ── 8. ensemble calibration protocol ────────────────────────────────────────

def test_fit_calibrator_accepts_member_score_matrix(normal_features):
    """Matrix input fits the ensemble-predictive objective and records the
    descriptive Jensen gap. Separable fixtures use explicit diagnostic mode."""
    a = RealCameraAgent(seed=42, n_members=5)
    a.fit(normal_features)
    rng = np.random.default_rng(1)
    normal = normal_features[rng.integers(0, len(normal_features), size=20)]
    cal_obs = list(normal) + list(normal + 3.0)
    a.fit_calibrator(_member_matrix(a, cal_obs), np.array([1.0] * 20 + [0.0] * 20))
    assert a._calibrator.is_fitted
    # PROTOCOL UPDATE: the diagnostic is recorded but is no longer a gate.
    assert a._cal_jensen_gap is not None and a._cal_jensen_gap >= 0.0
    u = a.estimate_uncertainty(normal_features[5], diagnostic=True)
    assert u.prediction == pytest.approx(a.predict(normal_features[5], diagnostic=True).value, rel=1e-9)


def test_jensen_gap_recorded_but_never_refuses(normal_features):
    """PROTOCOL UPDATE (ensemble-predictive calibration): the former
    fit-time Jensen-gap REFUSAL gate is retired — the calibrator now directly
    optimizes the deployed ensemble prediction, so there is no fit/inference
    mismatch left to gate on. The gap is still COMPUTED on the CAL
    distribution and recorded (max/median/p90) as a DESCRIPTIVE diagnostic,
    useful for identifying extreme nonlinear/slope regimes. Member-score
    patterns that historically tripped the gate must now calibrate
    successfully with the diagnostic recorded; the agent stays fully usable."""
    a = RealCameraAgent(seed=42, n_members=2)
    a.fit(normal_features)
    m = np.tile(np.array([[0.0, 1.0]]), (4, 1))  # 4 obs x 2 members
    y = np.array([1.0, 1.0, 0.0, 0.0])
    a.fit_calibrator(m, y)  # must NOT raise: the gate is retired
    assert a._calibrator is not None and a._calibrator.is_fitted
    # the diagnostic is recorded with sane internal ordering (max >= p90 >= median)
    diag = a._cal_jensen_diag
    assert diag is not None and 0.0 <= diag["median"] <= diag["p90"] <= diag["max"]
    assert a._cal_jensen_gap == diag["max"]
    # the lifecycle stays fail-loud and fully usable after the fit
    res = a.predict(normal_features[0], diagnostic=True)
    assert 0.0 <= res.value <= 1.0
    u = a.estimate_uncertainty(normal_features[0], diagnostic=True)
    assert u.total >= 0.0

    # Tight member scores (tiny dispersion) record a tiny gap (gap ~1e-4).
    a2 = RealCameraAgent(seed=42, n_members=2)
    a2.fit(normal_features)
    m_tight = np.tile(np.array([[0.49, 0.51]]), (4, 1))
    a2.fit_calibrator(m_tight, y)
    assert a2._cal_jensen_gap is not None and a2._cal_jensen_gap <= CONSENSUS_MARGIN


def test_register_member_dispersion_returns_observed_dispersion(normal_features):
    a = RealCameraAgent()
    a.fit(normal_features)
    d = a.register_member_dispersion(np.array([[0.5, 0.9], [0.4, 0.8]]))
    assert d == pytest.approx(0.2)
    with pytest.raises(RealAgentError, match="n_obs"):
        a.register_member_dispersion(np.array([0.5, 0.9]))


def test_fit_calibrator_length_mismatch(normal_features):
    a = RealCameraAgent()
    a.fit(normal_features)
    with pytest.raises(RealAgentError, match="mismatch"):
        a.fit_calibrator(np.zeros(5), np.array([1.0, 0.0]))


# ── 9. lifecycle failure modes ──────────────────────────────────────────────

@pytest.mark.parametrize("cls", ALL_AGENT_CLASSES)
def test_unfitted_agent_fails_loud(cls, normal_features):
    a = cls()
    with pytest.raises(RealAgentError, match="fit"):
        a.predict(normal_features[0], diagnostic=True)
    with pytest.raises(RealAgentError, match="fit"):
        a.estimate_uncertainty(normal_features[0], diagnostic=True)
    with pytest.raises(RealAgentError, match="fit"):
        a.predict_normality(normal_features[0])


def test_fitted_but_uncalibrated_agent_fails_loud(normal_features):
    a = RealCameraAgent()
    a.fit(normal_features)
    with pytest.raises(RealAgentError, match="not ready"):
        a.predict(normal_features[0], diagnostic=True)


def test_unhealthy_agent_contract():
    a = RealIMUAgent()
    a.healthy = False
    p = a.predict(np.zeros(4), diagnostic=True)
    u = a.estimate_uncertainty(np.zeros(4), diagnostic=True)
    assert p.value == 0.5 and p.confidence == 0.0
    assert u.prediction == 0.5 and u.epistemic == 1.0 and u.aleatoric == 1.0


def test_fit_rejects_single_row(normal_features):
    a = RealSegAgent()
    with pytest.raises(RealAgentError, match="NORMAL"):
        a.fit(normal_features[:1])


# ── 10. no anomaly-label dependency / frozen paths / purity ─────────────────

def test_fit_path_uses_only_supplied_normal_rows():
    rng = np.random.default_rng(9)
    x = rng.normal(0, 0.1, size=(50, 4))
    a1 = RealCameraAgent(seed=42, n_members=3)
    a1.fit(x)
    a2 = RealCameraAgent(seed=42, n_members=3)
    a2.fit(x.copy())
    assert np.array_equal(a1.predict_normality(x[0]), a2.predict_normality(x[0]))
    import inspect
    sig = inspect.signature(RealNormalityAgent.fit)
    assert list(sig.parameters) == ["self", "normal_features"]


def test_normality_module_has_no_anomaly_label_semantics():
    import inspect
    from cognix.adapters.carla import normality
    src = inspect.getsource(normality)
    for forbidden in ("anomaly_type", "test_anomaly", "y_test"):
        assert forbidden not in src


def test_frozen_synthetic_modules_unmodified():
    import subprocess
    frozen = [
        "cognix/adapters/carla/dataset.py",
        "cognix/adapters/carla/features.py",
        "cognix/adapters/carla/agents.py",
        "cognix/artifacts/build_synthetic_artifacts.py",
    ]
    for rel in frozen:
        r = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel])
        assert r.returncode == 0, f"frozen file modified relative to HEAD: {rel}"


def test_generic_package_import_does_not_pull_adapters():
    import subprocess
    import sys
    code = (
        "import sys; import cognix; "
        "sys.exit(1 if any(m.startswith('cognix.adapters') for m in sys.modules) else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True)
    assert r.returncode == 0, r.stderr.decode(errors="replace")
