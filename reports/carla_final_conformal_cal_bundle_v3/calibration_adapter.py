"""Registered official-label alignment and scenario-block conformal adapter."""
import math


def labels_from_table(table):
    import numpy as np
    ticks = table["tick"].to_numpy()
    labels = table["anomaly"].to_numpy()
    if ticks.dtype.kind not in "iu" or len(ticks) < 2 or not np.array_equal(ticks, np.arange(len(ticks))):
        raise ValueError("LABEL_TICKS_MUST_BE_COMPLETE_CONTIGUOUS_INTEGER_0_TO_N_MINUS_1")
    if labels.dtype.kind not in "biu" or not np.isin(labels, (0, 1)).all():
        raise ValueError("OFFICIAL_LABELS_MUST_BE_BOOLEAN_OR_INTEGER_0_1_NO_NULLS")
    return labels.astype(np.int64)


def tick_scores(q, labels):
    import numpy as np
    q = np.asarray(q)
    labels = np.asarray(labels)
    if q.shape != labels.shape or not np.isfinite(q).all() or not ((q >= 0) & (q <= 1)).all():
        raise ValueError("INVALID_ALIGNED_PROBABILITIES")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("INVALID_LABELS")
    # Preserve the float32 frozen graph probability; candidate arithmetic float64.
    q = q.astype(np.float64)
    return np.where(labels == 0, 1 - q, q)


def threshold(scores):
    scores = list(scores)
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in scores):
        raise ValueError("INVALID_SCENARIO_SCORES")
    n = len(scores)
    k = (19 * (n + 1) + 19) // 20
    infinite = n == 0 or k > n
    return {"alpha": 0.05, "n_cal": n, "rank_1_based": k,
            "Q": None if infinite else sorted(scores)[k - 1],
            "quantile_is_infinite": infinite, "augmentation": "+infinity", "comparison": "<="}
