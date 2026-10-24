"""
Extended Metrics for Cognix Universal Benchmark.

Adds metrics beyond what cognix/metrics/evaluation.py provides:
  - Balanced accuracy, Precision, Recall, F1, AUROC (classification)
  - Confusion matrix components (TP/TN/FP/FN)
  - Maximum Calibration Error (MCE)
  - Calibration curve data (for reliability diagrams)
  - Uncertainty quality: AUROC/AUPR for error detection, separation
  - OOD detection: AUROC/AUPR/FPR95 using epistemic as OOD score
  - Selective risk / AURC
  - Attention entropy
  - Holm-Bonferroni correction for multiple tests

All metrics are computed from raw arrays; no fabricated values.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

try:
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, f1_score,
        precision_score, recall_score, balanced_accuracy_score,
        roc_curve, confusion_matrix
    )
    _SKLEARN = True
except ImportError:
    _SKLEARN = False


# ── Predictive Quality ──────────────────────────────────────────────────────

def compute_classification_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute full binary classification metrics from probability outputs.

    Parameters
    ----------
    probs   : (N,) predicted probability for class 1
    labels  : (N,) true binary labels {0, 1}
    threshold: decision threshold

    Returns dict with accuracy, balanced_accuracy, precision, recall, f1,
    auroc, TP, TN, FP, FN, confusion_matrix.
    """
    if len(probs) == 0:
        return {k: float("nan") for k in [
            "accuracy", "balanced_accuracy", "precision", "recall",
            "f1", "auroc", "TP", "TN", "FP", "FN"]}

    preds_bin = (probs >= threshold).astype(int)
    labels_int = labels.astype(int)

    # Basic
    acc = float(np.mean(preds_bin == labels_int))
    tp = int(np.sum((preds_bin == 1) & (labels_int == 1)))
    tn = int(np.sum((preds_bin == 0) & (labels_int == 0)))
    fp = int(np.sum((preds_bin == 1) & (labels_int == 0)))
    fn = int(np.sum((preds_bin == 0) & (labels_int == 1)))

    if _SKLEARN:
        bal_acc = float(balanced_accuracy_score(labels_int, preds_bin))
        prec = float(precision_score(labels_int, preds_bin, zero_division=0))
        rec  = float(recall_score(labels_int, preds_bin, zero_division=0))
        f1   = float(f1_score(labels_int, preds_bin, zero_division=0))
        try:
            auc = float(roc_auc_score(labels_int, probs)) if len(np.unique(labels_int)) > 1 else float("nan")
        except Exception:
            auc = float("nan")
    else:
        # Fallback: manual balanced accuracy
        pos_rate = float(np.mean(preds_bin[labels_int == 1] == 1)) if tp + fn > 0 else 0.0
        neg_rate = float(np.mean(preds_bin[labels_int == 0] == 0)) if tn + fp > 0 else 0.0
        bal_acc = (pos_rate + neg_rate) / 2.0
        prec = tp / (tp + fp + 1e-10)
        rec  = tp / (tp + fn + 1e-10)
        f1   = 2 * prec * rec / (prec + rec + 1e-10)
        auc  = float("nan")

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auroc": auc,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }


# ── Calibration ──────────────────────────────────────────────────────────────

def compute_mce(
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Maximum Calibration Error (worst-bin |acc - conf|)."""
    predictions = np.asarray(predictions, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if len(predictions) == 0:
        return 0.0
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    mce = 0.0
    for i in range(n_bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (predictions >= lower) & (predictions <= upper if i == n_bins - 1 else predictions < upper)
        if np.any(mask):
            gap = abs(float(np.mean(predictions[mask])) - float(np.mean(labels[mask])))
            mce = max(mce, gap)
    return float(mce)


def compute_calibration_curve(
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, List[float]]:
    """
    Reliability diagram data: mean confidence and mean accuracy per bin.
    Returns dict with keys 'mean_confidence', 'mean_accuracy', 'bin_sizes'.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    mean_conf, mean_acc, bin_sizes = [], [], []
    for i in range(n_bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (predictions >= lower) & (predictions <= upper if i == n_bins - 1 else predictions < upper)
        if np.any(mask):
            mean_conf.append(float(np.mean(predictions[mask])))
            mean_acc.append(float(np.mean(labels[mask])))
            bin_sizes.append(int(np.sum(mask)))
        else:
            mean_conf.append(float((lower + upper) / 2))
            mean_acc.append(float("nan"))
            bin_sizes.append(0)
    return {"mean_confidence": mean_conf, "mean_accuracy": mean_acc, "bin_sizes": bin_sizes}


# ── Uncertainty Quality ───────────────────────────────────────────────────────

def compute_uncertainty_quality(
    epistemics: np.ndarray,
    aleatorics: np.ndarray,
    totals: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
) -> Dict[str, Any]:
    """
    Evaluate whether uncertainty signals model errors.

    Computes:
      - mean uncertainty for correct vs incorrect predictions
      - AUROC/AUPR for using epistemic uncertainty to detect errors
      - uncertainty separation (mean_epi_error - mean_epi_correct)
    """
    preds_bin = (probs >= 0.5).astype(int)
    is_wrong = (preds_bin != labels.astype(int)).astype(int)  # 1 = error

    correct_mask = is_wrong == 0
    error_mask   = is_wrong == 1

    result: Dict[str, Any] = {
        "mean_epistemic": float(np.mean(epistemics)),
        "mean_aleatoric": float(np.mean(aleatorics)),
        "mean_total":     float(np.mean(totals)),
        "n_correct": int(np.sum(correct_mask)),
        "n_error":   int(np.sum(error_mask)),
    }

    for name, arr in [("epistemic", epistemics), ("aleatoric", aleatorics), ("total", totals)]:
        result[f"mean_{name}_correct"] = float(np.mean(arr[correct_mask])) if np.any(correct_mask) else float("nan")
        result[f"mean_{name}_error"]   = float(np.mean(arr[error_mask]))   if np.any(error_mask)   else float("nan")

    # Separation
    epi_sep = result["mean_epistemic_error"] - result["mean_epistemic_correct"]
    result["epistemic_separation"] = float(epi_sep) if not (np.isnan(epi_sep)) else float("nan")

    # AUROC/AUPR: use epistemic as score for detecting errors
    if _SKLEARN and len(np.unique(is_wrong)) > 1:
        try:
            result["auroc_error_detection"] = float(roc_auc_score(is_wrong, epistemics))
            result["aupr_error_detection"]  = float(average_precision_score(is_wrong, epistemics))
        except Exception:
            result["auroc_error_detection"] = float("nan")
            result["aupr_error_detection"]  = float("nan")
    else:
        result["auroc_error_detection"] = float("nan")
        result["aupr_error_detection"]  = float("nan")

    # Correlation
    if len(epistemics) > 1:
        try:
            result["epistemic_error_corr"] = float(np.corrcoef(epistemics, is_wrong.astype(float))[0, 1])
        except Exception:
            result["epistemic_error_corr"] = float("nan")
    else:
        result["epistemic_error_corr"] = float("nan")

    return result


# ── OOD Detection ─────────────────────────────────────────────────────────────

def compute_ood_metrics(
    id_epistemics: np.ndarray,
    ood_epistemics: np.ndarray,
) -> Dict[str, Any]:
    """
    Evaluate epistemic uncertainty as OOD score.

    Higher epistemic uncertainty = more OOD.
    ID label = 0, OOD label = 1.

    Parameters
    ----------
    id_epistemics  : epistemic uncertainties from in-distribution samples
    ood_epistemics : epistemic uncertainties from OOD samples
    """
    n_id  = len(id_epistemics)
    n_ood = len(ood_epistemics)
    if n_id == 0 or n_ood == 0:
        return {"auroc": float("nan"), "aupr": float("nan"), "fpr95": float("nan"),
                "n_id": n_id, "n_ood": n_ood}

    scores = np.concatenate([id_epistemics, ood_epistemics])
    labels = np.concatenate([np.zeros(n_id), np.ones(n_ood)])  # 1=OOD

    result: Dict[str, Any] = {"n_id": n_id, "n_ood": n_ood}

    if not _SKLEARN or len(np.unique(labels)) < 2:
        result.update({"auroc": float("nan"), "aupr": float("nan"), "fpr95": float("nan")})
        return result

    try:
        result["auroc"] = float(roc_auc_score(labels, scores))
    except Exception:
        result["auroc"] = float("nan")
    try:
        result["aupr"] = float(average_precision_score(labels, scores))
    except Exception:
        result["aupr"] = float("nan")

    # FPR@95%TPR
    try:
        fpr_arr, tpr_arr, _ = roc_curve(labels, scores)
        idx = np.searchsorted(tpr_arr, 0.95)
        result["fpr95"] = float(fpr_arr[min(idx, len(fpr_arr) - 1)])
    except Exception:
        result["fpr95"] = float("nan")

    return result


# ── Conformal Prediction ──────────────────────────────────────────────────────

def compute_conformal_metrics(
    prediction_sets: List[List[int]],
    labels: np.ndarray,
    target_coverage: float = 0.95,
) -> Dict[str, Any]:
    """
    Conformal prediction quality metrics.

    Parameters
    ----------
    prediction_sets : list of lists; each entry is the set of predicted classes
    labels          : true integer labels
    target_coverage : nominal 1-alpha coverage
    """
    if not prediction_sets or len(labels) == 0:
        return {k: float("nan") for k in [
            "empirical_coverage", "target_coverage", "coverage_gap",
            "abs_coverage_gap", "mean_set_size", "median_set_size"]}

    set_sizes = [len(ps) for ps in prediction_sets]
    covered   = sum(1 for i, ps in enumerate(prediction_sets) if int(labels[i]) in ps)
    emp_cov   = float(covered / len(labels))
    cov_gap   = emp_cov - target_coverage

    # Distribution of set sizes
    sizes_arr = np.array(set_sizes)
    size_counts = {int(k): int(v) for k, v in zip(*np.unique(sizes_arr, return_counts=True))}

    return {
        "empirical_coverage": emp_cov,
        "target_coverage": target_coverage,
        "coverage_gap": float(cov_gap),
        "abs_coverage_gap": float(abs(cov_gap)),
        "mean_set_size": float(np.mean(sizes_arr)),
        "median_set_size": float(np.median(sizes_arr)),
        "set_size_distribution": size_counts,
        "n_samples": len(labels),
        "n_covered": int(covered),
    }


# ── Selective Prediction / AURC ───────────────────────────────────────────────

def compute_selective_risk(
    probs: np.ndarray,
    labels: np.ndarray,
    n_thresholds: int = 50,
) -> Dict[str, Any]:
    """
    Selective prediction: evaluate risk (error rate) vs coverage tradeoff.

    Coverage = fraction of samples acted upon (confidence >= threshold).
    Risk     = error rate among acted-on samples.

    Returns:
      - full_coverage_risk: risk when acting on everything
      - aurc: Area Under the Risk-Coverage curve (lower is better)
      - risk_coverage_curve: list of (coverage, risk) tuples
      - abstention_rate at which risk falls below 5%
    """
    if len(probs) == 0:
        return {"aurc": float("nan"), "full_coverage_risk": float("nan"),
                "risk_coverage_curve": []}

    # Confidence = max(p, 1-p)
    confidence = np.maximum(probs, 1 - probs)
    is_wrong = (np.round(probs) != labels.astype(int)).astype(float)

    thresholds = np.linspace(0.5, 1.0, n_thresholds)
    curve = []
    for tau in thresholds:
        acted = confidence >= tau
        coverage = float(np.mean(acted))
        if coverage > 0:
            risk = float(np.mean(is_wrong[acted]))
        else:
            risk = float("nan")
        curve.append({"threshold": float(tau), "coverage": coverage, "risk": risk})

    # AURC: integrate over [0, 1] coverage
    valid_pts = [(pt["coverage"], pt["risk"]) for pt in curve
                 if not np.isnan(pt["risk"]) and pt["coverage"] > 0]
    if len(valid_pts) > 1:
        covs, risks = zip(*valid_pts)
        _trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))
        aurc = float(_trapezoid(list(risks), list(covs)) * (-1))  # negate since coverage decreases
    else:
        aurc = float("nan")

    return {
        "full_coverage_risk": float(np.mean(is_wrong)),
        "aurc": aurc,
        "risk_coverage_curve": curve,
        "coverage_at_10pct_abstention": float(np.mean(confidence >= np.percentile(confidence, 10))),
    }


# ── Graph Mechanism Metrics ───────────────────────────────────────────────────

def compute_attention_metrics(
    attention_matrix: np.ndarray,
    healthy_indices: List[int],
    degraded_indices: List[int],
) -> Dict[str, Any]:
    """
    Analyze attention weights for a (N, N) attention matrix.

    Parameters
    ----------
    attention_matrix : (N, N) attention weights (row i attends to col j)
    healthy_indices  : indices of healthy agents
    degraded_indices : indices of degraded/failed agents
    """
    N = attention_matrix.shape[0]
    result: Dict[str, Any] = {}

    # Attention entropy per row (how spread the attention is)
    entropies = []
    for i in range(N):
        row = attention_matrix[i]
        row = row[row > 1e-9]  # non-zero attention
        if len(row) > 0:
            # Normalize just in case
            row = row / (row.sum() + 1e-12)
            ent = float(-np.sum(row * np.log(row + 1e-12)))
            entropies.append(ent)

    result["mean_attention_entropy"] = float(np.mean(entropies)) if entropies else float("nan")
    result["max_attention_entropy"]  = float(np.max(entropies))  if entropies else float("nan")

    # Mean attention toward healthy vs degraded senders
    if healthy_indices:
        healthy_attn = attention_matrix[:, healthy_indices].mean()
        result["mean_attn_to_healthy"] = float(healthy_attn)
    else:
        result["mean_attn_to_healthy"] = float("nan")

    if degraded_indices:
        degraded_attn = attention_matrix[:, degraded_indices].mean()
        result["mean_attn_to_degraded"] = float(degraded_attn)
        if healthy_indices:
            result["suppression_ratio"] = float(
                result["mean_attn_to_degraded"] / (result["mean_attn_to_healthy"] + 1e-10)
            )
        else:
            result["suppression_ratio"] = float("nan")
    else:
        result["mean_attn_to_degraded"] = float("nan")
        result["suppression_ratio"] = float("nan")

    return result


# ── Holm-Bonferroni Correction ────────────────────────────────────────────────

def holm_bonferroni(p_values: List[float]) -> List[float]:
    """
    Apply Holm-Bonferroni correction to a list of raw p-values.
    Returns list of adjusted p-values (same order as input).
    """
    n = len(p_values)
    if n == 0:
        return []

    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    max_adj = 0.0

    for rank, (orig_idx, p) in enumerate(indexed):
        adj = p * (n - rank)
        adj = max(adj, max_adj)
        adj = min(adj, 1.0)
        adjusted[orig_idx] = adj
        max_adj = adj

    return adjusted


def holm_bonferroni_dict(named_pvalues: Dict[str, float]) -> Dict[str, float]:
    """Apply Holm-Bonferroni to named p-values, return adjusted dict."""
    keys = list(named_pvalues.keys())
    raw  = [named_pvalues[k] for k in keys]
    adj  = holm_bonferroni(raw)
    return {k: float(v) for k, v in zip(keys, adj)}


# ── Summary Statistics ────────────────────────────────────────────────────────

def compute_summary_stats(values: List[float]) -> Dict[str, float]:
    """
    Compute mean, SD, median, 95% CI from a list of values.
    Uses t-distribution CI (appropriate for small n).
    """
    if not values:
        return {k: float("nan") for k in ["mean", "sd", "median", "ci95_lower", "ci95_upper"]}
    arr = np.array([v for v in values if not np.isnan(v)], dtype=np.float64)
    if len(arr) == 0:
        return {k: float("nan") for k in ["mean", "sd", "median", "ci95_lower", "ci95_upper"]}
    n = len(arr)
    mean = float(np.mean(arr))
    sd   = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    med  = float(np.median(arr))
    if n > 1:
        from scipy import stats as _sp
        ci = _sp.t.interval(0.95, df=n - 1, loc=mean, scale=_sp.sem(arr))
        ci_lo, ci_hi = float(ci[0]), float(ci[1])
    else:
        ci_lo = ci_hi = mean
    return {"mean": mean, "sd": sd, "median": med, "ci95_lower": ci_lo, "ci95_upper": ci_hi, "n": n}


# ── Throughput ────────────────────────────────────────────────────────────────

def compute_throughput(n_decisions: int, elapsed_seconds: float) -> float:
    """Decisions per second."""
    if elapsed_seconds <= 0:
        return float("nan")
    return float(n_decisions / elapsed_seconds)
