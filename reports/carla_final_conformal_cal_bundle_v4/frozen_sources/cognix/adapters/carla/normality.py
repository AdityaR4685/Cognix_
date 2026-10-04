"""
Normal-only one-class modeling layer for CarlAnomaly Experiment 1.

SCOPE (architecture guardrail):
    Domain-agnostic by construction (abstract feature vectors, opaque scores),
    but placed inside cognix/adapters/carla/ because its only consumer is the
    CARLA adapter. Generalize into a generic module only on second use by
    another domain adapter.

AUDITABILITY REQUIREMENTS:
    1. fit(X_normal) trains ONLY on the supplied normal samples.
    2. predict_normality(X) is the primary API. It returns the RAW one-class
       normality statistic — a bounded, monotone, dimension-stable quantity.
       A raw normality score is NOT a probability: P(target=normal) under TRAIN-derived clean-vs-pseudo calibration appears only
       after the EXPLICIT ScoreCalibrator mapping. Nothing here silently
       renames a distance/score into a probability.
    3. The exact Mahalanobis squared distance is always available for
       provenance via mahalanobis_distance2().
    4. Ensemble members are exactly reproducible under fixed integer seeds
       (numpy PCG64 via np.random.default_rng — never Python's salted hash()).
    5. Uncertainty follows the canonical COGNIX Bernoulli decomposition
       (identical semantics to cognix.uncertainty.decomposition):
           p_mean    = mean(p_t)
           total     = H(p_mean)
           aleatoric = mean_t H(p_t)
           epistemic = max(0, total - aleatoric)
       These are ESTIMATES per that decomposition — not ground-truth physical
       uncertainty. A bootstrap ensemble may legitimately disagree on a
       constant input (members saw different resamples), so epistemic is NOT
       asserted to vanish on identical inputs; it vanishes only when the
       MEMBER PROBABILITIES are identical.

Raw normality statistic definition (numerically stable)
-------------------------------------------------------
MahalanobisNormality fits mu and shrunk precision on normal rows only and
exposes TWO quantities:

    mahalanobis_distance2(x) = (x - mu)^T Sigma_shrunk^{-1} (x - mu)     [exact]

    score(x) = predict_normality(x) = exp(-0.5 * d2(x) / m2)
        m2 = mean training squared distance (a chi-square-style scale for the
             fitted distribution under the shrinkage metric)

    Justification: for d-dimensional roughly Gaussian features, d2 of NORMAL
    observations grows linearly with d (E[d2] ~ d under a matched metric), so
    the un-normalized exp(-0.5 * d2) collapses to numerical zero in high
    dimensions (measured: median ~1e-35 at d=128, ~1e-137 at d=512 for fresh
    NORMAL rows — unusable as a calibrator input). Dividing by the fitted
    mean m2 makes the exponent a DIMENSIONLESS multiple of the training-
    typical distance: score ~ 1 near the distribution center, ~ exp(-0.5)
    for a training-typical point, monotonically decreasing as d2 grows.
    The statistic is bounded in [0, 1], strictly monotone in d2, deterministic,
    and never interpreted as a probability before calibration. The exact d2
    is preserved separately for provenance.

KNNNormality.score(x) = exp(-d_k(x) / s)
    d_k(x) = mean distance to the k nearest training rows
    s = mean training kNN distance (fitted scale, guarded against 0)
    Already dimension-robust (fresh normals score near exp(-1) at any d);
    bounded in [0, 1], monotone, deterministic.

Score-to-probability mapping (EXPLICIT, fitted on calibration pairs)
--------------------------------------------------------------------
Two calibrators share the mapping family p = sigmoid(a * s + b), a > 0:

ScoreCalibrator
    Legacy SCALAR form: damped Newton logistic regression on (s_bar, target_normal)
    pairs where s_bar is a per-observation scalar score. Kept for backward
    compatibility and as the reference scalar-calibration implementation.

EnsemblePredictiveCalibrator (PRODUCTION protocol)
    ENSEMBLE-PREDICTIVE form: the deployed ensemble prediction is
        q_i = mean_t sigmoid(a * s_it + b)
    (member scores mapped PER MEMBER with the shared mapping, then averaged),
    so the fit directly minimizes
        mean_i BCE(y_i, q_i)
    on the full (n_obs, K) member-score matrix — fit and inference are the
    SAME objective (no mean-score surrogate, hence no fit/inference
    consistency gate). See the class docstring for optimizer details.

    Hardening (both calibrators):
        - target_normal must contain only {0, 1} (ints/floats/bools).
        - both classes required (else the mapping is unidentifiable).
        - explicit fitted/unfitted state; prob_normal() fails before fit().
        - deterministic, iteration-bounded optimization.
        - direction guard: higher normality must never map to lower
          P(target=normal) under TRAIN-derived clean-vs-pseudo calibration (enforced by log-parameterization or explicit check).

    The legacy fit/inference mismatch diagnostic (Jensen gap,
    |mean_t sigmoid(a s_t + b) - sigmoid(a mean_t s_t + b)|) is recorded as a
    DESCRIPTIVE statistic (see ensemble_jensen_diagnostic) — useful for
    identifying extreme nonlinear/slope regimes — but is NOT a fit gate under
    the ensemble-predictive protocol.
"""
from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np


# ── one-class normality models ───────────────────────────────────────────────

class NormalityModel:
    """Protocol: fit on normal rows only; predict_normality returns the raw
    bounded normality statistic (NOT a probability)."""

    def fit(self, x_normal: np.ndarray) -> "NormalityModel":
        raise NotImplementedError

    def predict_normality(self, x: np.ndarray) -> float:
        raise NotImplementedError


class MahalanobisNormality(NormalityModel):
    """Gaussian one-class model with shrinkage covariance and a dimension-
    stable normality statistic (see module docstring for the definition)."""

    def __init__(self, shrinkage: float = 0.1):
        if not (0.0 <= shrinkage < 1.0):
            raise ValueError("shrinkage must be in [0, 1)")
        self.shrinkage = float(shrinkage)
        self._mean: Optional[np.ndarray] = None
        self._prec: Optional[np.ndarray] = None
        self._d2_scale: float = 1.0

    def fit(self, x_normal: np.ndarray) -> "MahalanobisNormality":
        x = np.asarray(x_normal, dtype=np.float64)
        if x.ndim != 2 or len(x) < 2:
            raise ValueError("fit expects (n, d) with n >= 2")
        if not np.all(np.isfinite(x)):
            raise ValueError("fit expects finite features")
        self._mean = x.mean(axis=0)
        centered = x - self._mean
        cov = centered.T @ centered / max(len(x) - 1, 1)
        diag = np.diag(np.diag(cov))
        cov = (1.0 - self.shrinkage) * cov + self.shrinkage * diag
        cov += np.eye(cov.shape[0]) * 1e-9
        self._prec = np.linalg.inv(cov)
        # Dimensionless scale: mean squared distance of the TRAINING rows
        # under the fitted metric (chi-square-style reference for the fitted
        # distribution). Guarded against degenerate all-identical rows.
        d2_train = np.array([self._distance2(row) for row in x])
        mean_d2 = float(d2_train.mean())
        self._d2_scale = mean_d2 if mean_d2 > 1e-12 else 1.0
        return self

    def _distance2(self, x: np.ndarray) -> float:
        v = np.asarray(x, dtype=np.float64).ravel() - self._mean
        return float(v @ self._prec @ v)

    def mahalanobis_distance2(self, x: np.ndarray) -> float:
        """Exact squared Mahalanobis distance under the fitted (shrunk) metric.
        Provenance quantity — NOT a normality score and NOT a probability.

        NOTE: the input is passed to _distance2 UNCENTERED — _distance2 performs
        its own centering. (An earlier version pre-centered here, so inputs were
        centered twice: x - 2*mean. With zero-mean fixtures that is a no-op and
        went undetected; with realistic nonzero-mean features (e.g. GNSS
        latitude ~30 deg) the absolute coordinate leaked into the precision
        quadratic and inflated every query distance by orders of magnitude.)"""
        if self._mean is None or self._prec is None:
            raise RuntimeError("MahalanobisNormality must be fitted first.")
        v = np.asarray(x, dtype=np.float64).ravel()
        if not np.all(np.isfinite(v)):
            raise ValueError("non-finite input")
        return max(self._distance2(v), 0.0)

    def predict_normality(self, x: np.ndarray) -> float:
        """Raw normality statistic exp(-0.5 * d2 / m2) in [0, 1].
        Higher = more normal. NOT a probability (see module docstring)."""
        if self._mean is None or self._prec is None:
            raise RuntimeError("MahalanobisNormality must be fitted first.")
        d2 = self.mahalanobis_distance2(x)
        return float(np.exp(-0.5 * d2 / self._d2_scale))

    # Backward-compatible alias used by earlier-stage tests/callers. The
    # ensemble-level legacy protocol this alias once served (fit on the MEAN OF
    # MEMBER SCORES) was replaced by ensemble-predictive calibration on the
    # full (n_obs, K) member matrix; the alias stays a harmless per-row
    # convenience and is no longer part of the production calibration path.
    def score(self, x: np.ndarray) -> float:
        return self.predict_normality(x)


class KNNNormality(NormalityModel):
    """k-nearest-neighbour distance one-class model (clean, dependency-free,
    already dimension-robust)."""

    def __init__(self, k: int = 5):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = int(k)
        self._train: Optional[np.ndarray] = None
        self._scale: float = 1.0

    def fit(self, x_normal: np.ndarray) -> "KNNNormality":
        x = np.asarray(x_normal, dtype=np.float64)
        if x.ndim != 2 or len(x) < self.k + 1:
            raise ValueError(f"fit expects (n, d) with n >= k+1={self.k + 1}")
        if not np.all(np.isfinite(x)):
            raise ValueError("fit expects finite features")
        self._train = x
        dists = np.array([self._knn_distance(row) for row in x])
        mean_dist = float(dists.mean())
        self._scale = mean_dist if mean_dist > 1e-12 else 1.0
        return self

    def _knn_distance(self, row: np.ndarray) -> float:
        diff = self._train - row
        d = np.sqrt((diff * diff).sum(axis=1))
        d = np.sort(d)
        if d.size > self.k and d[0] == 0.0:
            d = d[1:]  # exclude self when the query row is a training row
        kk = min(self.k, d.size)
        return float(d[:kk].mean()) if kk else 0.0

    def predict_normality(self, x: np.ndarray) -> float:
        """Raw normality statistic exp(-d_k / s) in [0, 1]. NOT a probability."""
        if self._train is None:
            raise RuntimeError("KNNNormality must be fitted before scoring.")
        dist = self._knn_distance(np.asarray(x, dtype=np.float64).ravel())
        return float(np.exp(-dist / self._scale))

    # Backward-compatible alias.
    def score(self, x: np.ndarray) -> float:
        return self.predict_normality(x)


# ── explicit score-to-probability mapping ────────────────────────────────────

class ScoreCalibrator:
    """
    EXPLICIT monotone mapping s -> p = sigmoid(a*s + b).

    Fit protocol: (s_bar_i, target_normal_i) pairs where s_bar_i is the ensemble's
    MEAN OF MEMBER SCORES for calibration observation i (see
    real_agents.CALIBRATION_PROTOCOL). Applied pointwise to individual member
    scores at inference; see the protocol discussion there for consistency.

    Hardening:
        - target_normal must contain only {0, 1} (ints/floats/bools).
        - both classes required (else the mapping is unidentifiable).
        - explicit fitted/unfitted state; prob_normal() fails before fit().
        - damped Newton logistic regression on standardized inputs:
          deterministic, iteration-bounded, learning rate does NOT scale
          with calibration-set size.
        - direction guard: the fitted slope must be positive on the raw
          score scale (higher normality => higher P(target=normal) under TRAIN-derived clean-vs-pseudo calibration); a fit whose
          optimum inverts this raises instead of silently flipping semantics.
    """

    def __init__(self):
        self.a: float = 0.0
        self.b: float = 0.0
        self._fitted: bool = False
        self._s_mean: float = 0.0
        self._s_std: float = 1.0

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, scores: np.ndarray, target_normal: np.ndarray) -> "ScoreCalibrator":
        s = np.asarray(scores, dtype=np.float64).ravel()
        y = np.asarray(target_normal, dtype=np.float64).ravel()
        if len(s) == 0 or len(s) != len(y):
            raise ValueError("expected equal-length score/target_normal arrays")
        if not np.all(np.isfinite(s)):
            raise ValueError("non-finite scores")
        yset = set(np.unique(y).tolist())
        if not yset.issubset({0.0, 1.0}):
            raise ValueError(f"target_normal must contain only 0/1, got {sorted(yset)}")
        if yset != {0.0, 1.0}:
            raise ValueError(
                "calibration requires both classes (target_normal 0 and 1); a "
                "single-class set cannot identify the mapping"
            )
        # Standardize the score scale: makes Newton steps scale-free with
        # respect to feature dimension and score magnitude.
        self._s_mean = float(s.mean())
        self._s_std = float(s.std()) or 1.0
        z = (s - self._s_mean) / self._s_std

        # Damped Newton on the MEAN Bernoulli log-likelihood (gradients and
        # Hessians normalized by N): the argmax is identical to the sum-LL
        # optimum, but step sizes no longer scale with the calibration-set
        # size, so duplicating the CAL set yields the identical mapping.
        w1, w0 = 0.0, 0.0  # coefficients on [z, 1]
        lam = 1e-6
        n = len(z)
        for _ in range(100):
            p = self._sigmoid(w1 * z + w0)
            g = np.array([(y - p) @ z, (y - p).sum()]) / n
            wz = p * (1.0 - p)
            H = np.array(
                [[(wz * z * z).sum(), (wz * z).sum()],
                 [(wz * z).sum(), wz.sum()]]
            ) / n
            try:
                step = np.linalg.solve(H + lam * np.eye(2), g)
            except np.linalg.LinAlgError:
                step = g * 1e-3
            w1 += step[0]
            w0 += step[1]
            if max(abs(step[0]), abs(step[1])) < 1e-10:
                break
        # Convert back to raw-score coordinates: a*(s-mean)/std + b
        a = float(w1 / self._s_std)
        b = float(w0 - w1 * self._s_mean / self._s_std)
        if a <= 0.0:
            raise ValueError(
                "fitted mapping inverts the semantic direction (higher "
                "normality would map to lower P(target=normal) under TRAIN-derived clean-vs-pseudo calibration); refusing to fit"
            )
        self.a, self.b = a, b
        self._fitted = True
        return self

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def prob_normal(self, normality_score: float, *, diagnostic: bool = False) -> float:
        if not self._fitted:
            raise RuntimeError(
                "ScoreCalibrator is not fitted; no probability can be emitted."
            )
        return float(self._sigmoid(self.a * float(normality_score) + self.b))


# ── ensemble-predictive calibration (production protocol) ───────────────────

def ensemble_jensen_diagnostic(a: float, b: float, member_matrix: np.ndarray) -> dict:
    """Descriptive fit/inference mismatch diagnostic for a shared mapping
    p = sigmoid(a*s + b) applied to a (n_obs, K) member-score matrix:

        gap_i = | mean_t sigmoid(a*s_it + b) - sigmoid(a*mean_t(s_it) + b) |

    Purely descriptive (max/median/p90): identifies extreme nonlinear/slope
    regimes. It is NOT a consistency gate under the ensemble-predictive
    protocol, because production fitting directly optimizes the ensemble
    prediction. Logged for auditability and cross-protocol comparability.
    """
    M = np.asarray(member_matrix, dtype=np.float64)
    if M.ndim != 2 or M.shape[0] == 0:
        raise ValueError("expected a non-empty (n_obs, K) member-score matrix")
    p_members = 1.0 / (1.0 + np.exp(-np.clip(a * M + b, -500, 500)))
    p_mean_direct = 1.0 / (1.0 + np.exp(-np.clip(a * M.mean(axis=1) + b, -500, 500)))
    gap = np.abs(p_members.mean(axis=1) - p_mean_direct)
    return {
        "max": float(gap.max()),
        "median": float(np.median(gap)),
        "p90": float(np.percentile(gap, 90)),
    }


def score_separation(member_scores: np.ndarray, target_normal: np.ndarray) -> dict:
    """Threshold geometry, separately for row means and pooled member scores.

    Row-mean separation is descriptive: it does not prove separation for the
    ensemble-predictive likelihood. Pooled strict separation is sufficient.
    Equality is exact (no benchmark-dependent tolerance).
    """
    M = np.asarray(member_scores, dtype=float)
    if M.ndim == 1:
        M = M[:, None]
    y = np.asarray(target_normal, dtype=float)

    def geometry(s, labels):
        normal, pseudo = s[labels == 1], s[labels == 0]
        low, high = float(normal.min()), float(pseudo.max())
        unique = np.unique(s)
        thresholds = np.r_[np.nextafter(unique[0], -np.inf),
                           unique[:-1] + np.diff(unique) / 2, unique[-1]]
        ns, ps = np.sort(normal), np.sort(pseudo)
        violations = (np.searchsorted(ns, thresholds, side="right") +
                      len(ps) - np.searchsorted(ps, thresholds, side="right"))
        best = int(np.argmin(violations))
        overlap = [max(float(normal.min()), float(pseudo.min())),
                   min(float(normal.max()), float(pseudo.max()))]
        kind = ("completely_separated" if low > high else
                "quasi_separated" if low == high and np.ptp(s) > 0 else "overlapping")
        return {"normal_range": [float(normal.min()), float(normal.max())],
                "pseudo_range": [float(pseudo.min()), float(pseudo.max())],
                "overlap_interval": overlap if overlap[0] <= overlap[1] else None,
                "min_normal": low, "max_pseudo": high, "gap": low - high,
                "best_threshold": float(thresholds[best]),
                "threshold_rule": "normal iff score > threshold",
                "violations": int(violations[best]), "classification": kind}

    return {"ensemble_mean": geometry(M.mean(axis=1), y),
            "pooled_members": geometry(M.ravel(), np.repeat(y, M.shape[1]))}


def fixed_slope_profile(member_scores, target_normal, slopes) -> list:
    """Diagnostic only: deterministic multi-bracket intercept minimization.

    The ensemble objective need not be convex in b. Search every local basin
    on a fixed grid including score quantiles, then refine each bracket. The
    profile never supplies the deployed slope or intercept.
    """
    from scipy.optimize import minimize_scalar

    M = np.asarray(member_scores, dtype=float)
    if M.ndim == 1:
        M = M[:, None]
    y = np.asarray(target_normal, dtype=float)
    out = []
    for a in slopes:
        a = float(a)
        def loss(b):
            q = np.clip(EnsemblePredictiveCalibrator._sigmoid(a * M + b).mean(axis=1),
                        1e-12, 1 - 1e-12)
            return float(-np.mean(y * np.log(q) + (1 - y) * np.log1p(-q)))
        grid = np.unique(np.r_[np.linspace(-a * M.max() - 40, -a * M.min() + 40, 129),
                               -a * np.quantile(M, np.linspace(0, 1, 129))])
        values = np.array([loss(b) for b in grid])
        best = (float(values.min()), float(grid[np.argmin(values)]))
        for j in range(1, len(grid) - 1):
            if (values[j] <= values[j-1] and values[j] <= values[j+1] and
                    (values[j] < values[j-1] or values[j] < values[j+1])):
                result = minimize_scalar(loss, bounds=(grid[j-1], grid[j+1]),
                                         method="bounded", options={"xatol": 1e-10})
                candidate = (float(result.fun), float(result.x))
                if candidate < best:
                    best = candidate
        out.append({"slope_raw": a, "intercept_raw": best[1], "nll": best[0]})
    return out


class EnsemblePredictiveCalibrator:
    """
    EXPLICIT ensemble-predictive mapping (s_1..s_K) -> p = mean_t sigmoid(a*s_t + b).

    Fit protocol: minimize mean_i BCE(y_i, mean_t sigmoid(a*s_it + b)) over the
    (n_obs, K) member-score matrix — the EXACT objective of the deployed
    ensemble prediction (fit and inference are the same functional). See the
    module docstring for the protocol rationale and hardening contract.

    Implementation notes:
        - scores are standardized internally (deterministic moments over all
          n*K entries); (a, b) are reported on the RAW score scale;
        - slope positivity: the standardized slope is optimized as A = exp(u),
          so a > 0 by construction (no post-hoc clipping);
        - analytic gradients: with q_i = mean_t p_it and w_it = p_it(1-p_it),
              dL/dA = mean_i [ (q_i - y_i) / (q_i (1 - q_i)) * mean_t(w_it z_it) ]
              dL/dB = mean_i [ (q_i - y_i) / (q_i (1 - q_i)) * mean_t(w_it)     ]
          and the chain rule through A = exp(u) gives dL/du = A * dL/dA;
        - the Hessian is formed by central differences of the analytic
          gradient (deterministic), the Newton system is damped, and a
          backtracking line search guarantees monotone objective descent;
        - multiple deterministic initializations are tried and the fit with
          the LOWEST final objective is kept (local-minimum detection, not
          hyperparameter tuning);
        - clipping of q to [1e-12, 1 - 1e-12] is for log/division safety only.
    """

    _LOG_SLOPE_INITS = (0.0, float(np.log(10.0)), float(np.log(2.0)))

    def __init__(self):
        self.a: float = 0.0
        self.b: float = 0.0
        self._fitted: bool = False
        self._s_mean: float = 0.0
        self._s_std: float = 1.0
        self._nll_final: Optional[float] = None
        self._nll_initial: Optional[float] = None
        self._n_iter: int = 0
        self._converged: bool = False

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def _objective(self, A: float, B: float, Z: np.ndarray, y: np.ndarray) -> float:
        q = np.clip(self._sigmoid(A * Z + B).mean(axis=1), 1e-12, 1.0 - 1e-12)
        return float(-np.mean(y * np.log(q) + (1.0 - y) * np.log(1.0 - q)))

    def _grad(self, A: float, B: float, Z: np.ndarray, y: np.ndarray) -> np.ndarray:
        P = self._sigmoid(A * Z + B)                       # (n, K)
        q = np.clip(P.mean(axis=1), 1e-12, 1.0 - 1e-12)    # (n,)
        coef = (q - y) / np.clip(q * (1.0 - q), 1e-12, None)
        w = P * (1.0 - P)                                  # (n, K)
        dA = float(np.mean(coef * (w * Z).mean(axis=1)))
        dB = float(np.mean(coef * w.mean(axis=1)))
        return np.array([A * dA, dB])                      # chain rule A = exp(u)

    def fit(self, member_scores: np.ndarray, target_normal: np.ndarray) -> "EnsemblePredictiveCalibrator":
        self._fitted = False  # a failed refit must never leave a stale mapping
        M = np.asarray(member_scores, dtype=np.float64)
        y = np.asarray(target_normal, dtype=np.float64).ravel()
        if M.ndim == 1:
            M = M[:, None]  # scalar-score form (K = 1): reduces to ordinary BCE
        if M.ndim != 2 or M.shape[0] < 2 or M.shape[1] < 1:
            raise ValueError("member_scores must be a (n>=2, K>=1) matrix")
        if len(M) != len(y):
            raise ValueError("member_scores/target_normal length mismatch")
        if not np.all(np.isfinite(M)):
            raise ValueError("non-finite member scores")
        yset = set(np.unique(y).tolist())
        if not yset.issubset({0.0, 1.0}):
            raise ValueError(f"target_normal must contain only 0/1, got {sorted(yset)}")
        if yset != {0.0, 1.0}:
            raise ValueError(
                "calibration requires both classes (target_normal 0 and 1); a "
                "single-class set cannot identify the mapping"
            )
        # Standardize over ALL n*K entries (deterministic, scale-free Newton).
        self._s_mean = float(M.mean())
        self._s_std = float(M.std()) or 1.0
        Z = (M - self._s_mean) / self._s_std

        eps = 1e-5
        lam = 1e-6
        best = None
        # Multiple deterministic initializations: local-minimum verification,
        # NOT hyperparameter tuning (fixed spread of starting log-slopes).
        for u0 in self._LOG_SLOPE_INITS:
            u, B = float(u0), 0.0
            f = self._objective(float(np.exp(u)), B, Z, y)
            f0 = f
            converged = False
            it = 0
            for it in range(1, 301):
                g = self._grad(float(np.exp(u)), B, Z, y)
                # Deterministic Hessian: central differences of the analytic
                # gradient (dL/du, dL/dB).
                H = np.zeros((2, 2))
                for j, h in enumerate((eps * max(1.0, abs(u)), eps)):
                    up = np.array([u, B], dtype=np.float64); up[j] += h
                    um = np.array([u, B], dtype=np.float64); um[j] -= h
                    H[:, j] = (self._grad(float(np.exp(up[0])), up[1], Z, y)
                               - self._grad(float(np.exp(um[0])), um[1], Z, y)) / (2.0 * h)
                try:
                    step = np.linalg.solve(H + lam * np.eye(2), g)
                except np.linalg.LinAlgError:
                    step = g * 1e-3
                # Backtracking: monotone descent (objective never worsens).
                t = 1.0
                for _ in range(40):
                    cand_u, cand_B = u - t * step[0], B - t * step[1]
                    f_c = self._objective(float(np.exp(cand_u)), cand_B, Z, y)
                    if f_c <= f + 1e-14:
                        break
                    t *= 0.5
                u, B, f_prev, f = cand_u, cand_B, f, f_c
                if abs(f_prev - f) < 1e-12 and float(np.max(np.abs(t * step))) < 1e-10:
                    converged = True
                    break
            if best is None or f < best[0]:
                best = (f, u, B, converged, it, f0)
        f, u, B, converged, it, f0 = best
        A = float(np.exp(u))
        self.a = A / self._s_std
        self.b = float(B - A * self._s_mean / self._s_std)
        self._nll_final = float(f)
        self._nll_initial = float(f0)
        self._n_iter = int(it)
        self._converged = bool(converged)
        geometry = score_separation(M, y)
        profile = fixed_slope_profile(M, y, (1, 2, 5, 10, 20, 50, 100, 200,
                                               500, 1000, 2000, 5000))
        relative = fixed_slope_profile(M, y, (self.a, 2 * self.a, 4 * self.a))
        gradient_norm = float(np.max(np.abs(self._grad(A, B, Z, y))))
        # Objective-scale numerical resolution; never a slope cutoff.
        resolution = 1e-10
        improving_tail = (profile[-2]["nll"] - profile[-1]["nll"] > resolution and
                          profile[-3]["nll"] - profile[-2]["nll"] > resolution)
        lower_direction = min(r["nll"] for r in profile + relative) < f - resolution
        pooled_kind = geometry["pooled_members"]["classification"]
        structural = pooled_kind != "overlapping"
        suspected = structural or (improving_tail and lower_direction)
        # A stationary point alone can be a local minimum or a clipping plateau.
        # Require identifiable curvature and no observed lower profile basin.
        H = np.column_stack([
            (self._grad(float(np.exp(u + eps)), B, Z, y) -
             self._grad(float(np.exp(u - eps)), B, Z, y)) / (2 * eps),
            (self._grad(A, B + eps, Z, y) - self._grad(A, B - eps, Z, y)) / (2 * eps)])
        curvature = float(np.linalg.eigvalsh((H + H.T) / 2).min())
        finite = bool(converged and gradient_norm < 1e-8 and curvature > 0 and
                      np.isfinite([self.a, self.b, f]).all() and not suspected and
                      not lower_direction)
        self._validity = {
            "optimizer_converged": bool(converged), "finite_optimum": finite,
            "suspected_separation": bool(suspected), "gradient_norm": gradient_norm,
            "minimum_curvature": curvature, "slope_raw": self.a,
            "boundary_trend": "decreasing" if improving_tail else "not_decreasing",
            "lower_profile_direction": bool(lower_direction),
            "status": ("finite_stationary_candidate" if finite else
                       "suspected_separation" if suspected else "unvalidated_optimizer_result"),
            "separation": geometry, "slope_profile": profile,
            "relative_slope_profile": relative,
            "validity_scope": "numerical evidence, not a global existence proof",
        }
        self._fitted = True
        return self

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def ensemble_predictive_nll(self) -> float:
        """Final minimized objective mean_i BCE(y_i, mean_t sigmoid(a s_it + b))."""
        if not self._fitted:
            raise RuntimeError("EnsemblePredictiveCalibrator is not fitted.")
        return float(self._nll_final)

    @property
    def optimization_info(self) -> dict:
        """Deterministic optimizer audit trail (initial/final objective,
        iterations, convergence flag). Diagnostic only."""
        if not self._fitted:
            raise RuntimeError("EnsemblePredictiveCalibrator is not fitted.")
        return {
            "nll_initial": float(self._nll_initial),
            "nll_final": float(self._nll_final),
            "n_iter": int(self._n_iter),
            "converged": bool(self._converged),
            **self._validity,
        }

    def prob_normal(self, normality_score: float, *, diagnostic: bool = False) -> float:
        """Single-member mapping p = sigmoid(a*s + b) (the shared member map).
        The ENSEMBLE prediction is the mean of this over member scores —
        identical to the fitted objective's inner term."""
        if not self._fitted:
            raise RuntimeError(
                "EnsemblePredictiveCalibrator is not fitted; no probability can be emitted."
            )
        self.require_valid(diagnostic=diagnostic)
        if not np.isfinite(normality_score):
            raise ValueError("normality score must be finite")
        return float(self._sigmoid(self.a * float(normality_score) + self.b))

    def require_valid(self, *, diagnostic: bool = False) -> None:
        if not self._fitted:
            raise RuntimeError("EnsemblePredictiveCalibrator is not fitted.")
        if not diagnostic and not self._validity["finite_optimum"]:
            raise RuntimeError("Calibration invalid for scientific use: " +
                               self._validity["status"] +
                               "; diagnostic=True permits audit-only evaluation.")

    def ensemble_probability(self, member_scores, *, diagnostic: bool = False) -> float:
        """Deployed ensemble prediction mean_t sigmoid(a*s_t + b) for one
        observation's member scores — the exact quantity the fit optimizes."""
        if not self._fitted:
            raise RuntimeError("EnsemblePredictiveCalibrator is not fitted.")
        self.require_valid(diagnostic=diagnostic)
        s = np.asarray(member_scores, dtype=np.float64).ravel()
        if s.size == 0 or not np.all(np.isfinite(s)):
            raise ValueError("member scores must be a non-empty finite array")
        return float(self._sigmoid(self.a * s + self.b).mean())


# ── bootstrap ensemble ───────────────────────────────────────────────────────

class BootstrapNormalityEnsemble:
    """
    K one-class members on bootstrap resamples of the normal-only training
    rows. Epistemic source: model disagreement (members trained on different
    resamples disagree most where the normal manifold is under-determined).
    """

    def __init__(
        self,
        n_members: int = 5,
        factory: Callable[[], NormalityModel] = MahalanobisNormality,
        seed: int = 42,
    ):
        if n_members < 1:
            raise ValueError("n_members must be >= 1")
        self.n_members = int(n_members)
        self.factory = factory
        self.seed = int(seed)
        self._members: List[NormalityModel] = []

    def fit(self, x_normal: np.ndarray) -> "BootstrapNormalityEnsemble":
        x = np.asarray(x_normal, dtype=np.float64)
        if x.ndim != 2 or len(x) < 2:
            raise ValueError("fit expects (n, d) with n >= 2")
        rng = np.random.default_rng(self.seed)  # PCG64 — never Python hash()
        n = len(x)
        self._members = []
        for _ in range(self.n_members):
            idx = rng.integers(0, n, size=n)
            member = self.factory()
            member.fit(x[idx])
            self._members.append(member)
        return self

    @property
    def n_fitted_members(self) -> int:
        return len(self._members)

    def predict_normality(self, x: np.ndarray) -> np.ndarray:
        """Raw per-member normality statistics, shape (K,). NOT probabilities."""
        if not self._members:
            raise RuntimeError("ensemble must be fitted before scoring")
        return np.array([m.predict_normality(x) for m in self._members], dtype=np.float64)

    # Provenance-preserving aliases (single-observation convenience).
    def scores(self, x: np.ndarray) -> np.ndarray:
        return self.predict_normality(x)

    def mean_score(self, x: np.ndarray) -> float:
        return float(self.predict_normality(x).mean())


# ── uncertainty decomposition (canonical COGNIX semantics) ───────────────────

def _bernoulli_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def ensemble_to_uncertainty(p_samples: np.ndarray):
    """
    Map ensemble P(target=normal) under TRAIN-derived clean-vs-pseudo calibration samples {p_t} to the canonical UncertaintyResult:

        p_mean    = mean(p_t)
        total     = H(p_mean)
        aleatoric = mean_t H(p_t)
        epistemic = max(0, total - aleatoric)

    These are ESTIMATES per the COGNIX Bernoulli decomposition (identical
    semantics to cognix.uncertainty.decomposition) — not ground-truth
    physical uncertainty. Epistemic ~ 0 holds when the member PROBABILITIES
    are identical; a bootstrap ensemble may legitimately disagree on a
    constant input.
    """
    from cognix.core.types import UncertaintyResult

    p = np.asarray(p_samples, dtype=np.float64).ravel()
    if p.size == 0 or not np.all(np.isfinite(p)):
        raise ValueError("p_samples must be a non-empty finite array")
    if np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("probabilities must lie in [0, 1]")
    p_mean = float(p.mean())
    total = float(_bernoulli_entropy(np.array([p_mean]))[0])
    aleatoric = float(_bernoulli_entropy(p).mean())
    epistemic = max(0.0, total - aleatoric)
    return UncertaintyResult(
        prediction=p_mean,
        epistemic=epistemic,
        aleatoric=aleatoric,
        total=total,
    )
