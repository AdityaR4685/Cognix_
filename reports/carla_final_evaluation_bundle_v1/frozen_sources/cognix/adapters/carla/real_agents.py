"""
Real CarlAnomaly sensor agents (Experiment 1) — canonical-contract wrappers
around the normal-only modeling layer in normality.py.

ARCHITECTURE GUARDRAIL (pip-installable framework):
    CARLA-specific classes live in the CARLA adapter only. The generic COGNIX
    core sees ONLY the canonical contract:
        PredictionResult.value = P(target=normal) under TRAIN-derived clean-vs-pseudo calibration  (float in [0, 1])
        UncertaintyResult{prediction, epistemic, aleatoric, total}
        metadata["prediction_semantics"] = "P(target=normal) under TRAIN-derived clean-vs-pseudo calibration"
    A future adapter (e.g. cognix/adapters/medical/) can implement the same
    contracts without modifying the core.

MILESTONE SCOPE:
    Agents consume PRECOMPUTED compact vectors/embeddings (not raw
    1920x1080 images). Pretrained backbone extraction, pseudo-anomalies,
    artifact training, conformal/GAT refits, downloads and final evaluation
    are explicitly OUT OF SCOPE for this milestone.

SCIENTIFIC GUARDRAILS:
    - fit() trains ONLY on normal feature rows.
    - Raw normality statistics are preserved for provenance (and the exact
      Mahalanobis distance via mahalanobis_distance2 on members); the
      score-to-probability mapping is EXPLICIT and agents fail loud until it
      has been fitted. A raw one-class score is never presented as a
      probability.
    - UQ source is MODEL DISAGREEMENT of a bootstrap ensemble, decomposed per
      the canonical COGNIX Bernoulli semantics. These are epistemic/aleatoric
      ESTIMATES per that decomposition, not ground-truth physical uncertainty;
      no simulated sensor-noise mechanism is used
      (synthetic_probability_noise = False).
    - No salted hash() anywhere: seeding uses explicit integer seeds.

CALIBRATION PROTOCOL (ensemble-predictive, mathematically consistent)
----------------------------------------------------------
The final agent prediction is  p_mean(x) = mean_t P_t(target=normal)(x), where
each member probability applies the SAME calibrator to that member's raw
score: P_t = sigmoid(a * s_t(x) + b).

The calibrator is ENSEMBLE-PREDICTIVE (EnsemblePredictiveCalibrator): the
fit minimizes directly
    mean_i BCE(y_i, mean_t sigmoid(a * s_it + b))
on the full (n_obs, K) member-score matrix, i.e. the fitted objective IS the
deployed ensemble prediction. There is no mean-score surrogate and therefore
no fit/inference consistency gate: the former CONSENSUS_MARGIN Jensen-gap
refusal is retired as a gate. The diagnostic
    | mean_t sigmoid(a*s_t + b) - sigmoid(a*mean_t s_t + b) |
is still COMPUTED on the CAL distribution and recorded (max/median/p90) in
metadata (cal_jensen_gap_max/median/p90) as a descriptive audit statistic —
useful for identifying extreme nonlinear/slope regimes (e.g. near-perfectly
separable members), but never used to refuse the fit.

Labels: y_i in {0,1} (1 = normal/normal, 0 = pseudo; never TEST labels —
TRAIN/CAL provenance only). Both classes are required. Every observation
carries total weight 1 regardless of K: duplicating members does NOT change
the effective weight of any observation. The slope is positive by
construction (log-parameterization) and the optimizer is deterministic
(damped Newton, analytic gradients, backtracking, fixed multi-start).

NOTE ON EPISTEMIC (observed CARLA finding, not a COGNIX property): with the
current compact handcrafted features and bootstrap ensembles, probability-
space epistemic on real CarlAnomaly CAL data is observed to be near zero
(pseudo-side member scores saturate; member PROBABILITIES agree even where
member SCORES disagree). This is a recorded empirical finding of the CARLA
experiment — NOT an expected universal property of COGNIX, whose canonical
decomposition is unchanged (epistemic = max(0, H(mean p_t) - mean H(p_t))).
Score-space disagreement remains recorded in metadata for auditability.

Target probabilities depend on the constructed pseudo sampling distribution
and class balance. They are not physical driving safety probabilities or
validated posteriors for official anomaly types. Numerical optimizer
termination alone does not validate calibration. Invalid fits are retained
for audit, but prediction/UQ require explicit diagnostic=True to use them.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from cognix.adapters.carla.normality import (
    BootstrapNormalityEnsemble,
    EnsemblePredictiveCalibrator,
    MahalanobisNormality,
    ScoreCalibrator,
    ensemble_jensen_diagnostic,
    ensemble_to_uncertainty,
)
from cognix.core.interfaces import AgentInterface
from cognix.core.types import PredictionResult, UncertaintyResult


class RealAgentError(RuntimeError):
    """Raised when a real agent is queried before fit/calibration or on bad input."""


# Legacy mean-score-protocol artifact, retained ONLY as the named reference
# threshold in historical documentation and tests of the diagnostic scale: the
# former fit-time refusal gate (Jensen gap > margin) was retired when the
# calibrator became ensemble-predictive (the fit now optimizes the deployed
# ensemble prediction directly, so no consistency gate is needed). The gap is
# still recorded as a descriptive diagnostic; CONSENSUS_MARGIN no longer gates
# any production behavior.
CONSENSUS_MARGIN = 0.05


class RealNormalityAgent(AgentInterface):
    """
    One-class normality agent over precomputed compact features.

    Lifecycle (mirrors the frozen builder's TRAIN/CAL separation):
        1. fit(normal_features)               — TRAIN-only normal rows
        2. fit_calibrator(scores, target_normal)    — CAL-only pairs (explicit mapping)
        3. predict / estimate_uncertainty     — fail loud until both are done

    ``vector_fn`` extracts the agent's compact feature vector from an
    observation. Observations follow the established dict-by-agent_id
    convention: a dict missing the agent's key, or mapping it to None,
    raises RealAgentError (absent modalities are never zero-filled and never
    fall back to whole-dict conversion).
    """

    def __init__(
        self,
        agent_id: str,
        vector_fn: Callable,
        n_members: int = 5,
        seed: int = 42,
    ):
        self.agent_id = agent_id
        self.vector_fn = vector_fn
        self.n_members = int(n_members)
        self.seed = int(seed)
        self.healthy = True
        self._ensemble: Optional[BootstrapNormalityEnsemble] = None
        self._calibrator: Optional[EnsemblePredictiveCalibrator] = None
        self._cal_jensen_gap: Optional[float] = None
        self._cal_jensen_diag: Optional[dict] = None
        self._metadata = {
            "agent_id": agent_id,
            "prediction_semantics": "P(target=normal) under TRAIN-derived clean-vs-pseudo calibration",
            "uq_method": "bootstrap_oneclass_ensemble",
            "uq_source": "model_disagreement",
            "synthetic_probability_noise": False,
            "normality_model": "mahalanobis_bootstrap",
            "calibration_protocol": "ensemble_predictive",
            "n_members": self.n_members,
            "seed": self.seed,
        }

    # ── vector extraction ─────────────────────────────────────────────────
    def extract_vector(self, observation) -> np.ndarray:
        if isinstance(observation, dict):
            if self.agent_id not in observation:
                raise RealAgentError(
                    f"Agent '{self.agent_id}': key '{self.agent_id}' absent from the "
                    "observation dictionary; refusing to reinterpret the whole "
                    "dictionary as this agent's input."
                )
            obs = observation[self.agent_id]
        else:
            obs = observation
        if obs is None:
            raise RealAgentError(
                f"Agent '{self.agent_id}': observation slice is None (missing "
                "modality); absent modalities are never zero-filled."
            )
        v = np.asarray(obs if self.vector_fn is None else self.vector_fn(obs), dtype=np.float64).ravel()
        if v.size == 0:
            raise RealAgentError(f"Agent '{self.agent_id}': empty feature vector.")
        if not np.all(np.isfinite(v)):
            raise RealAgentError(f"Agent '{self.agent_id}': non-finite feature values.")
        return v

    # ── fitting (TRAIN / CAL protocol) ────────────────────────────────────
    def fit(self, normal_features: np.ndarray) -> "RealNormalityAgent":
        x = np.asarray(normal_features, dtype=np.float64)
        if x.ndim != 2 or len(x) < 2:
            raise RealAgentError(
                f"Agent '{self.agent_id}': fit expects a (n>=2, d) matrix of "
                "NORMAL feature rows only."
            )
        self._ensemble = BootstrapNormalityEnsemble(
            n_members=self.n_members,
            factory=MahalanobisNormality,
            seed=self.seed,
        )
        self._ensemble.fit(x)
        self._calibrator = None
        self._cal_jensen_gap = self._cal_jensen_diag = None
        return self

    def predict_normality(self, observation) -> np.ndarray:
        """Raw per-member normality statistics, shape (K,). NOT probabilities."""
        if self._ensemble is None:
            raise RealAgentError(f"Agent '{self.agent_id}': fit() required first.")
        return self._ensemble.predict_normality(self.extract_vector(observation))

    def normality_score(self, observation) -> float:
        """Mean raw normality statistic (NOT a probability)."""
        return float(self.predict_normality(observation).mean())

    def fit_calibrator(self, scores: np.ndarray, target_normal: np.ndarray) -> "RealNormalityAgent":
        """
        Fit the ENSEMBLE-PREDICTIVE mapping on CAL pairs.

        ``scores`` is the (n_obs, K>=2) member-score matrix (primary protocol:
        every member's score is calibrated PER MEMBER and the ensemble
        probability is the mean of member probabilities). A 1-D vector of
        scalar scores is also accepted and reduces to ordinary scalar BCE
        calibration (K = 1).

        The fit minimizes mean_i BCE(y_i, mean_t sigmoid(a*s_it + b)) —
        exactly the deployed inference semantics — with a shared (a, b)
        across members, a > 0 by construction, deterministic optimization,
        and total weight 1 per observation regardless of K. No Jensen-gap
        gate is applied; the gap is recorded as a descriptive diagnostic
        (max/median/p90) when a member matrix is supplied.
        """
        if self._ensemble is None:
            raise RealAgentError(f"Agent '{self.agent_id}': fit() must precede fit_calibrator().")
        s = np.asarray(scores, dtype=np.float64)
        y = np.asarray(target_normal, dtype=np.float64)
        member_matrix = None
        if s.ndim == 2 and s.shape[1] >= 2:
            member_matrix = s
        if len(s) != len(y):
            raise RealAgentError("calibration scores/labels length mismatch")
        self._calibrator = None  # prevent stale readiness after an unsuccessful refit
        calibrator = EnsemblePredictiveCalibrator().fit(s, y)
        # Descriptive fit/inference mismatch diagnostic (NOT a gate anymore):
        # identifies extreme nonlinear/slope regimes; recorded for auditability.
        self._cal_jensen_gap = None
        if member_matrix is not None:
            diag = ensemble_jensen_diagnostic(calibrator.a, calibrator.b, member_matrix)
            self._cal_jensen_gap = diag["max"]
            self._cal_jensen_diag = diag
        else:
            self._cal_jensen_diag = None
        self._calibrator = calibrator
        return self

    def register_member_dispersion(self, member_scores: np.ndarray) -> float:
        """Auditability helper: record the max calibration-time member-score
        dispersion from a (n_obs, K>=2) matrix. Returns the recorded value.
        Diagnostic only; Jensen gap is descriptive and calibration validity
        is audited separately."""
        m = np.asarray(member_scores, dtype=np.float64)
        if m.ndim != 2 or m.shape[1] < 2:
            raise RealAgentError("expected a (n_obs, K>=2) member-score matrix")
        return float(m.std(axis=1).max())

    def _require_ready(self, *, diagnostic: bool = False) -> None:
        if self._ensemble is None or self._calibrator is None:
            raise RealAgentError(
                f"Agent '{self.agent_id}' is not ready: fit() and fit_calibrator() "
                "must both complete (TRAIN then CAL) before prediction."
            )
        try:
            self._calibrator.require_valid(diagnostic=diagnostic)
        except RuntimeError as exc:
            raise RealAgentError(str(exc)) from exc

    # ── canonical COGNIX contract ─────────────────────────────────────────
    def _p_samples(self, observation) -> np.ndarray:
        """Per-member P(target=normal) under TRAIN-derived clean-vs-pseudo calibration samples {p_t} (explicit mapping per member)."""
        s_t = self.predict_normality(observation)
        return np.array(
            [self._calibrator.prob_normal(float(s), diagnostic=True) for s in s_t],
            dtype=np.float64,
        )

    def predict(self, observation, *, diagnostic: bool = False) -> PredictionResult:
        if not self.healthy:
            return PredictionResult(value=0.5, confidence=0.0, metadata={
                "status": "unhealthy", "calibration_valid": False,
                "prediction_semantics": self._metadata["prediction_semantics"]})
        self._require_ready(diagnostic=diagnostic)
        # p_mean = mean_t p_t — identical to UncertaintyResult.prediction so the
        # canonical contract is internally exact (mean of member probabilities,
        # each mapped explicitly; never sigmoid-of-mean-score).
        p = float(self._p_samples(observation).mean())
        s_t = self.predict_normality(observation)
        return PredictionResult(
            value=p,
            confidence=p,
            metadata={
                "agent_id": self.agent_id,
                "calibration_valid": self._calibrator.optimization_info["finite_optimum"],
                "calibration_status": self._calibrator.optimization_info["status"],
                "diagnostic_only": diagnostic,
                "prediction_semantics": "P(target=normal) under TRAIN-derived clean-vs-pseudo calibration",
                "normality_score_mean": float(s_t.mean()),
                "member_score_std": float(s_t.std()),
                "cal_jensen_gap_max": (self._cal_jensen_diag or {}).get("max"),
                "cal_jensen_gap_median": (self._cal_jensen_diag or {}).get("median"),
                "cal_jensen_gap_p90": (self._cal_jensen_diag or {}).get("p90"),
                # OBSERVED CARLA finding (not a COGNIX-universal property):
                # probability-space epistemic is near zero on real CarlAnomaly
                # CAL data because member probabilities saturate; the canonical
                # decomposition itself is unchanged and NOT redefined.
                "epistemic_note": (
                    "near-zero probability-space epistemic is an observed "
                    "CARLA finding, not an expected universal property"
                ),
            },
        )

    def estimate_uncertainty(self, observation, *, diagnostic: bool = False) -> UncertaintyResult:
        if not self.healthy:
            return UncertaintyResult(prediction=0.5, epistemic=1.0, aleatoric=1.0, total=2.0)
        self._require_ready(diagnostic=diagnostic)
        return ensemble_to_uncertainty(self._p_samples(observation))

    def metadata(self) -> dict:
        return {**self._metadata,
                "calibration_valid": bool(self._calibrator and
                    self._calibrator.optimization_info["finite_optimum"]),
                "calibration_audit": (self._calibrator.optimization_info
                                      if self._calibrator else None)}


# ── Concrete CARLA agents (precomputed-vector inputs for this milestone) ─────
# The identity vector_fn keeps this stage free of pretrained backbones; a
# backbone later replaces vector_fn without touching the agent contract.
# vector_fn=None means "the observation slice IS the feature vector".

def _identity(x):
    return x


class RealCameraAgent(RealNormalityAgent):
    """Camera agent over precomputed camera embeddings/statistic vectors."""

    def __init__(self, n_members: int = 5, seed: int = 42):
        super().__init__("Camera", _identity, n_members=n_members, seed=seed)


class RealSegAgent(RealNormalityAgent):
    """Segmentation agent over precomputed class-composition vectors."""

    def __init__(self, n_members: int = 5, seed: int = 42):
        super().__init__("Seg", _identity, n_members=n_members, seed=seed)


class RealGNSSAgent(RealNormalityAgent):
    """GNSS agent over precomputed trajectory-consistency feature windows."""

    def __init__(self, n_members: int = 5, seed: int = 42):
        super().__init__("GNSS", _identity, n_members=n_members, seed=seed)


class RealIMUAgent(RealNormalityAgent):
    """IMU agent over precomputed acceleration-based motion features."""

    def __init__(self, n_members: int = 5, seed: int = 42):
        super().__init__("IMU", _identity, n_members=n_members, seed=seed)
