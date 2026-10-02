"""Frozen constructed-corruption metrics; no threshold uses training/CAL/TEST."""
import itertools

import numpy as np

from cognix.adapters.carla import graph_fit_export as ge

METRICS = ("AUROC", "AUPRC", "F1", "accuracy", "balanced_accuracy", "Brier", "BCE", "ECE")


def validate_predictions(q, target):
    q, y = np.asarray(q, dtype=np.float64), np.asarray(target)
    ge.require(q.ndim == y.ndim == 1 and len(q) == len(y) and len(q) > 0,
               "nonempty aligned probability/target vectors required")
    ge.require(np.isfinite(q).all() and np.all((q >= 0) & (q <= 1))
               and set(y.tolist()) <= {0, 1}, "invalid q_normal/target_normal")
    return q, y.astype(np.int64)


def bce(q, target):
    q, y = validate_predictions(q, target)
    return float(np.mean(-y * np.log(np.clip(q, 1e-7, 1 - 1e-7))
                         - (1 - y) * np.log(np.clip(1 - q, 1e-7, 1 - 1e-7))))


def scenario_macro_bce(q, target, scenarios, expected_scenarios=None):
    q, y = validate_predictions(q, target)
    scenarios = np.asarray(scenarios)
    ge.require(len(scenarios) == len(q), "scenario alignment")
    ids = sorted(set(scenarios.tolist()))
    if expected_scenarios is not None:
        ge.require(ids == sorted(expected_scenarios), "missing/foreign validation scenario")
    per = {s: bce(q[scenarios == s], y[scenarios == s]) for s in ids}
    return float(np.mean(list(per.values()))), per


def ece15(q, target):
    q, y = validate_predictions(q, target)
    edges = np.arange(16, dtype=np.float64) / 15
    membership = np.minimum(np.searchsorted(edges, q, side="right") - 1, 14)
    bins, total = [], 0.0
    for k in range(15):
        selected = membership == k
        n = int(selected.sum())
        mean_p = float(q[selected].mean()) if n else None
        mean_y = float(y[selected].mean()) if n else None
        contribution = n / len(q) * abs(mean_p - mean_y) if n else 0.0
        total += contribution
        bins.append({"bin": k, "lower": float(edges[k]), "upper": float(edges[k + 1]),
                     "count": n, "mean_probability": mean_p, "mean_target": mean_y,
                     "weighted_gap": contribution})
    return float(total), bins


def binary_metrics(q_normal, target_normal, threshold):
    q, target = validate_predictions(q_normal, target_normal)
    ge.require(np.isfinite(threshold) and 0 <= threshold <= 1, "corruption threshold domain")
    score, y = 1 - q, 1 - target
    positive, negative = int(y.sum()), int((1 - y).sum())
    prediction = score >= threshold
    tp = int(np.sum(prediction & (y == 1))); fp = int(np.sum(prediction & (y == 0)))
    fn, tn = positive - tp, negative - fp
    reasons = {}
    # Average ranks/pairwise wins for AUROC, with 0.5 credit on tied scores.
    order = np.argsort(score, kind="stable")
    s, labels = score[order], y[order]
    _, starts, sizes = np.unique(s, return_index=True, return_counts=True)
    group_pos = np.add.reduceat(labels, starts)
    group_neg = sizes - group_pos
    if positive and negative:
        earlier_neg = np.cumsum(group_neg) - group_neg
        auroc = float(np.sum(group_pos * (earlier_neg + 0.5 * group_neg)) / (positive * negative))
    else:
        auroc = None; reasons["AUROC"] = "requires both constructed classes"
    if positive:
        # Descending tie groups: stepwise average precision, no PR trapezoids.
        gpos, count = group_pos[::-1], sizes[::-1]
        ap = float(np.sum((gpos / positive) * (np.cumsum(gpos) / np.cumsum(count))))
    else:
        ap = None; reasons["AUPRC"] = "no corruption-positive observations"
    denominator = 2 * tp + fp + fn
    # This explicit zero is prescribed in metrics_spec.json::threshold.
    f1 = float(2 * tp / denominator) if denominator else 0.0
    balanced = float(0.5 * (tp / positive + tn / negative)) if positive and negative else None
    if balanced is None:
        reasons["balanced_accuracy"] = "requires both constructed classes"
    ece, bins = ece15(q, target)
    return {"AUROC": auroc, "AUPRC": ap, "F1": f1, "accuracy": float((tp + tn) / len(y)),
            "balanced_accuracy": balanced, "Brier": float(np.mean((q - target) ** 2)),
            "BCE": bce(q, target), "ECE": ece, "undefined_reasons": reasons,
            "ECE_bins": bins, "n": len(y), "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}}


def select_validation_threshold(q, target, scenarios, split, expected_scenarios=None):
    q, target = validate_predictions(q, target)
    role = np.asarray(split)
    ge.require(len(role) == len(q) and np.all(role == 1), "threshold selection requires GRAPH_VALIDATION only")
    scenarios = np.asarray(scenarios)
    ge.require(len(scenarios) == len(q), "scenario alignment")
    ids = sorted(set(scenarios.tolist()))
    if expected_scenarios is not None:
        ge.require(ids == sorted(expected_scenarios), "threshold validation scenario identity")
    score, y = 1 - q, 1 - target
    candidates = np.unique(np.r_[0.0, score, 1.0])
    f1s = []
    for sid in ids:
        mask = scenarios == sid
        order = np.argsort(score[mask], kind="stable")
        s, labels = score[mask][order], y[mask][order]
        before = np.r_[0, np.cumsum(labels)]
        positions = np.searchsorted(s, candidates, side="left")
        tp = int(labels.sum()) - before[positions]
        fp = len(labels) - positions - tp
        fn = int(labels.sum()) - tp
        denominator = 2 * tp + fp + fn
        f1s.append(np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator > 0))
    objective = np.mean(f1s, axis=0)
    best = objective.max()
    # Exact objective equality; largest threshold wins ties.
    index = np.flatnonzero(objective == best)[-1]
    return {"threshold": float(candidates[index]), "validation_scenario_macro_F1": float(best),
            "candidate_count": len(candidates), "selection_split": "GRAPH_VALIDATION",
            "tie_rule": "largest_threshold", "classify": "p_corrupt>=threshold"}


def grouped_metrics(q, target, scenarios, threshold):
    q, target = validate_predictions(q, target)
    scenarios = np.asarray(scenarios)
    ge.require(len(scenarios) == len(q), "metric scenario alignment")
    per = {sid: binary_metrics(q[scenarios == sid], target[scenarios == sid], threshold)
           for sid in sorted(set(scenarios.tolist()))}
    macro, reasons = {}, {}
    for name in METRICS:
        undefined = [sid for sid, values in per.items() if values[name] is None]
        if undefined:
            macro[name] = None
            reasons[name] = "undefined in scenarios: " + ",".join(undefined)
        else:
            macro[name] = float(np.mean([v[name] for v in per.values()]))
    macro["undefined_reasons"] = reasons
    return {"pooled": binary_metrics(q, target, threshold), "per_scenario": per, "scenario_macro": macro}


def dataset_metric_report(dataset, q):
    a, scenarios = dataset.arrays, dataset.scenarios
    q, target = validate_predictions(q, a["target_normal"])
    vi = dataset.validation_indices
    threshold = select_validation_threshold(q[vi], target[vi], scenarios[vi], a["split"][vi],
                                             dataset.splits["GRAPH_VALIDATION"])
    tau = threshold["threshold"]
    report = {"threshold_selection": threshold, "data_kind": dataset.data_kind,
              "interpretation": "engineering evidence only" if dataset.data_kind == "synthetic_fixture" else "internal graph-development statistics",
              "per_split": {}}
    for split, indices in (("GRAPH_TRAIN", dataset.train_indices), ("GRAPH_VALIDATION", vi)):
        values = grouped_metrics(q[indices], target[indices], scenarios[indices], tau)
        # A recipe diagnostic uses its pseudo rows AND those pairs' clean parents.
        recipe_metrics = {}
        for k, recipe in enumerate(ge.RECIPES):
            pairs = set(a["pair_id"][indices][a["recipe_index"][indices] == k].tolist())
            selected = np.array([str(pid) in pairs for pid in a["pair_id"][indices]])
            if selected.any():
                ids = indices[selected]
                recipe_metrics[recipe] = grouped_metrics(q[ids], target[ids], scenarios[ids], tau)
            else:
                recipe_metrics[recipe] = {"value": None, "reason": "no retained pairs for recipe"}
        values["per_recipe"] = recipe_metrics
        report["per_split"][split] = values
    return report


def paired_difference_summary(differences):
    """Future five-seed descriptive summary. Missing seeds are never replaced."""
    from cognix.adapters.carla.graph_training_data import SEEDS
    if set(differences) != set(SEEDS) or any(differences.get(s) is None for s in SEEDS):
        return {"complete": False, "value": None, "reason": "all five planned seed pairs are required",
                "per_seed": {str(s): differences.get(s) for s in SEEDS}}
    d = np.array([differences[s] for s in SEEDS], dtype=float)
    ge.require(np.isfinite(d).all(), "nonfinite paired difference")
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    signs = np.array(list(itertools.product((-1, 1), repeat=5)))
    sign_flip = float(np.mean(np.abs((signs * d).mean(axis=1)) >= abs(mean) - 1e-12))
    radius = 2.776445105 * sd / np.sqrt(5)
    return {"complete": True, "per_seed": {str(s): float(v) for s, v in zip(SEEDS, d)},
            "mean": mean, "median": float(np.median(d)), "sample_SD": sd,
            "min": float(d.min()), "max": float(d.max()), "conditional_seed_CI": [mean - radius, mean + radius],
            "paired_dz": mean / sd if sd > 0 else None,
            "undefined_reasons": {} if sd > 0 else {"paired_dz": "zero paired-difference SD"},
            "exact_two_sided_sign_flip_p": sign_flip}


def scenario_bootstrap(differences_by_seed_scenario):
    d = np.asarray(differences_by_seed_scenario, dtype=float)
    ge.require(d.shape == (5, 3) and np.isfinite(d).all(), "requires five paired seeds and three scenarios")
    draws = np.random.Generator(np.random.PCG64(606)).integers(0, 3, size=(10000, 3))
    values = d[:, draws].mean(axis=(0, 2))
    return {"percentile_interval": np.percentile(values, [2.5, 97.5]).tolist(),
            "draws": 10000, "seed": 606, "unit": "whole_validation_scenario", "interpretation": "descriptive fragility only"}
