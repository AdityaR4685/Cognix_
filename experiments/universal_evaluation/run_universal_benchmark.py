"""
Cognix Universal Benchmark — Main Orchestrator.

Evaluation suite covering 17 metric families across:
  - 3 graph modes: NoGraph, StandardGAT, EpistemicGAT
  - 2 ablations: AverageFusion, NoConformal
  - 8 scenarios + robustness/resilience sweeps
  - 20 paired seeds (42-61)

Checkpoints raw results after every seed.
Skips already-completed seed/scenario/mode combinations.

Scientific constraints:
  - NO modification to any core Cognix algorithm
  - No tuning on benchmark results
  - Escalation ground truth: N/A (documented)
  - Energy/GPU: N/A (CPU-only, no NVML — documented)
"""

import sys
import os
import json
import time
import math
import random
import traceback
import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

# ── Cognix imports ────────────────────────────────────────────────────────────
from cognix.evaluation.scenarios import DataGenerator
from cognix.evaluation.statistics import paired_analysis
from cognix.graph.epistemic_gat import EpistemicGAT
from cognix.graph.standard_gat import StandardGAT
from cognix.graph.no_graph import NoGraph
from cognix.belief.fusion import EpistemicWeightedFusion, AverageFusion
from cognix.calibration.conformal import ConformalPredictor
from cognix.config.schema import CognixConfig
from cognix.engine.decision_engine import DecisionEngine
from cognix.core.interfaces import AgentInterface
from cognix.core.types import PredictionResult, UncertaintyResult
from cognix.metrics.evaluation import calculate_ece, accuracy, brier_score, LatencyTracker

# ── Extended metrics ──────────────────────────────────────────────────────────
from metrics_extended import (
    compute_classification_metrics,
    compute_mce,
    compute_calibration_curve,
    compute_uncertainty_quality,
    compute_ood_metrics,
    compute_conformal_metrics,
    compute_selective_risk,
    compute_attention_metrics,
    compute_summary_stats,
    compute_throughput,
    holm_bonferroni_dict,
)
from scenarios_extended import (
    ExtendedDataGenerator,
    NOISE_SEVERITY_LEVELS,
    AGENT_FAILURE_COUNTS,
    get_scenario_config,
)

# ── Output paths ──────────────────────────────────────────────────────────────
RESULTS_DIR = ROOT / "results" / "universal_evaluation"
RAW_DIR      = RESULTS_DIR / "raw"
SUMMARIES_DIR = RESULTS_DIR / "summaries"
STATS_DIR    = RESULTS_DIR / "statistics"
LATENCY_DIR  = RESULTS_DIR / "latency"
ROBUST_DIR   = RESULTS_DIR / "robustness"
OOD_DIR      = RESULTS_DIR / "ood"
CONFORMAL_DIR = RESULTS_DIR / "conformal"
SAFETY_DIR   = RESULTS_DIR / "safety"
SCALABILITY_DIR = RESULTS_DIR / "scalability"
RESOURCES_DIR = RESULTS_DIR / "resources"
ABLATIONS_DIR = RESULTS_DIR / "ablations"

for d in [RESULTS_DIR, RAW_DIR, SUMMARIES_DIR, STATS_DIR, LATENCY_DIR,
          ROBUST_DIR, OOD_DIR, CONFORMAL_DIR, SAFETY_DIR,
          SCALABILITY_DIR, RESOURCES_DIR, ABLATIONS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
SEEDS = list(range(42, 62))  # 20 seeds: 42-61
SCALABILITY_SEEDS = [42, 43, 44, 45, 46]  # 5 seeds for scalability

GRAPH_MODES = ["NoGraph", "StandardGAT", "EpistemicGAT"]
ABLATION_MODES = ["AverageFusion", "NoConformal"]

N_TRAIN  = 100
N_CAL    = 100
N_TEST   = 200
N_WARMUP = 10   # latency warmup runs

MAIN_SCENARIOS = ["NORMAL", "HIGH_NOISE", "MISSING_AGENT", "OOD_SHIFT",
                  "CONFLICTING", "MULTI_FAILURE"]

# ── Reliability tracker ───────────────────────────────────────────────────────
_reliability_log: Dict[str, Any] = {
    "attempted": 0, "successful": 0, "failed": 0, "errors": [],
    "nan_inf_occurrences": []
}


def _log_failure(run_id: str, exc: Exception):
    _reliability_log["failed"] += 1
    _reliability_log["errors"].append({
        "run_id": run_id, "error": str(exc),
        "traceback": traceback.format_exc()
    })


def _check_nan_inf(metrics: Dict, run_id: str):
    for k, v in metrics.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            _reliability_log["nan_inf_occurrences"].append(
                {"run_id": run_id, "metric": k, "value": str(v)}
            )


# ── Agent implementation ──────────────────────────────────────────────────────

class FeatureModel(nn.Module):
    def __init__(self, feature_idx: int):
        super().__init__()
        self.feature_idx = feature_idx
        self.net = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(16, 1), nn.Sigmoid()
        )

    def forward(self, x):
        feat = x[:, self.feature_idx:self.feature_idx + 1]
        return self.net(feat)


class HeterogeneousAgent(AgentInterface):
    """Single-feature MC-Dropout agent (same as evaluation/run.py — not modified)."""

    def __init__(self, agent_id: str, feature_idx: int, T: int = 30):
        self.agent_id  = agent_id
        self.feature_idx = feature_idx
        self.model     = FeatureModel(feature_idx)
        self.healthy   = True
        self.T         = T

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=0.01)
        crit = nn.BCELoss()
        for _ in range(50):
            opt.zero_grad()
            out = self.model(torch.FloatTensor(X))
            loss = crit(out, torch.FloatTensor(y).view(-1, 1))
            loss.backward()
            opt.step()

    def predict(self, x: np.ndarray) -> PredictionResult:
        if not self.healthy:
            return PredictionResult(value=0.5, confidence=0.0)
        self.model.eval()
        with torch.no_grad():
            val = float(self.model(torch.FloatTensor(x).view(1, -1)).item())
        return PredictionResult(value=val, confidence=1.0)

    def estimate_uncertainty(self, x: Any) -> UncertaintyResult:
        if not self.healthy:
            return UncertaintyResult(prediction=0.5, epistemic=1.0,
                                     aleatoric=1.0, total=2.0)
        self.model.train()
        with torch.no_grad():
            t_x = torch.FloatTensor(x).view(1, -1)
            samples = np.array([self.model(t_x).item() for _ in range(self.T)])
        p_bar = float(np.mean(samples))
        epi   = float(np.mean((samples - p_bar) ** 2))
        ale   = float(np.mean(samples * (1.0 - samples)))
        p_bar = float(np.clip(p_bar, 1e-6, 1 - 1e-6))
        return UncertaintyResult(prediction=p_bar, epistemic=epi,
                                 aleatoric=ale, total=epi + ale)

    def metadata(self) -> dict:
        return {"id": self.agent_id, "feature_idx": self.feature_idx}


# ── Pipeline helpers ──────────────────────────────────────────────────────────

def _run_single_sample(
    x_i: np.ndarray,
    agents: List[HeterogeneousAgent],
    gat,
    fuser,
    calibrator: Optional[ConformalPredictor],
    agent_reliabilities: Dict[str, float],
    cgx_config: CognixConfig,
    use_conformal: bool = True,
) -> Dict[str, Any]:
    """
    Run the Cognix pipeline for a single sample and return all outputs needed for metrics.
    Mirrors evaluation/run.py logic exactly (no algorithmic changes).
    """
    agent_order = sorted([a.agent_id for a in agents])
    agents_by_id = {a.agent_id: a for a in agents}

    preds: Dict[str, float] = {}
    ep_uncs: Dict[str, float] = {}
    ale_uncs: Dict[str, float] = {}

    t_unc = time.perf_counter()
    for aid in agent_order:
        a = agents_by_id[aid]
        preds[aid] = float(a.predict(x_i).value)
        u = a.estimate_uncertainty(x_i)
        ep_uncs[aid] = float(u.epistemic)
        ale_uncs[aid] = float(u.aleatoric)
    t_unc_ms = (time.perf_counter() - t_unc) * 1000

    # Node features
    nf = []
    for aid in agent_order:
        nf.append([preds[aid], ep_uncs[aid], ale_uncs[aid]])
    node_features = np.array(nf, dtype=np.float32)

    N = len(agent_order)
    adjacency = (np.ones((N, N)) - np.eye(N)).astype(np.float32)

    t_gat = time.perf_counter()
    g_res = gat.forward(node_features, adjacency, ep_uncs, agent_order)
    t_gat_ms = (time.perf_counter() - t_gat) * 1000

    refined_preds: Dict[str, float] = {}
    for i, aid in enumerate(agent_order):
        val = float(g_res.node_outputs[i, 0]) if len(g_res.node_outputs.shape) > 1 \
              else float(g_res.node_outputs[i])
        p_ref = float(np.clip(1.0 / (1.0 + math.exp(-val)), 1e-7, 1 - 1e-7))
        refined_preds[aid] = p_ref

    t_fuse = time.perf_counter()
    f_res = fuser.fuse(predictions=refined_preds,
                       uncertainties=ep_uncs, reliabilities=agent_reliabilities)
    fused_prob = float(np.clip(f_res.probability, 1e-7, 1 - 1e-7))
    t_fuse_ms = (time.perf_counter() - t_fuse) * 1000

    pred_set: List[int] = []
    t_cal = time.perf_counter()
    if use_conformal and calibrator is not None:
        cal_in = np.array([[1.0 - fused_prob, fused_prob]])
        cp_sets = calibrator.predict(cal_in, alpha=0.05)
        pred_set = cp_sets[0].prediction_set
    t_cal_ms = (time.perf_counter() - t_cal) * 1000

    # Aggregate uncertainties
    total_epi = float(np.mean(list(ep_uncs.values())))
    total_ale = float(np.mean(list(ale_uncs.values())))

    # Last-layer attention
    attn_matrix = None
    if g_res.attention:
        attn_mat = g_res.attention[-1]
        if hasattr(attn_mat, 'numpy'):
            attn_mat = attn_mat.numpy()
        attn_matrix = np.array(attn_mat, dtype=np.float32)

    return {
        "prob": fused_prob,
        "pred_set": pred_set,
        "epistemic": total_epi,
        "aleatoric": total_ale,
        "total_unc": total_epi + total_ale,
        "agent_epistemics": dict(ep_uncs),
        "agent_preds": dict(refined_preds),
        "attention_matrix": attn_matrix,
        "latency_unc_ms": t_unc_ms,
        "latency_gat_ms": t_gat_ms,
        "latency_fuse_ms": t_fuse_ms,
        "latency_cal_ms": t_cal_ms,
    }


# ── Core evaluation function ──────────────────────────────────────────────────

def run_evaluation(
    seed: int,
    graph_type: str,
    scenario_name: str,
    num_agents: int = 4,
    noise_level: float = 0.0,
    ood_severity: float = 0.0,
    missing_agents: List[int] = None,
    conflict_fraction: float = 0.0,
    use_avg_fusion: bool = False,
    use_conformal: bool = True,
    n_train: int = N_TRAIN,
    n_cal: int = N_CAL,
    n_test: int = N_TEST,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run the complete Cognix pipeline for one seed/scenario/mode combination.

    Returns a flat dict with all computed metrics.
    """
    if missing_agents is None:
        missing_agents = []

    # ── Seeding ────────────────────────────────────────────────────────────
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ── Data generation ────────────────────────────────────────────────────
    gen_tr  = DataGenerator(n_samples=n_train, seed=seed)
    gen_cal = DataGenerator(n_samples=n_cal,   seed=seed + 1000)
    gen_te  = DataGenerator(n_samples=n_test,  seed=seed + 2000)

    X_train, y_train = gen_tr.generate_base(num_features=num_agents)
    X_cal,   y_cal   = gen_cal.generate_base(num_features=num_agents)
    X_test_base, y_test_base = gen_te.generate_base(num_features=num_agents)

    # Apply test condition
    if scenario_name == "CONFLICTING":
        ext_gen = ExtendedDataGenerator(n_samples=n_test, seed=seed + 2000)
        X_test, y_test, ood_labels = ext_gen.generate_conflicting(
            num_features=num_agents, flip_fraction=conflict_fraction or 0.5)
    elif scenario_name == "MULTI_FAILURE":
        ext_gen = ExtendedDataGenerator(n_samples=n_test, seed=seed + 2000)
        X_test, y_test, ood_labels = ext_gen.generate_multi_failure(
            num_features=num_agents, noise_level=noise_level or 3.0,
            missing_agents=missing_agents if missing_agents else [1, 2])
    else:
        # Determine condition name for apply_condition
        if scenario_name == "NOISE_SWEEP":
            cond = "HIGH_NOISE" if (noise_level is not None and noise_level > 0.0) else "NORMAL"
        elif scenario_name in ["NORMAL", "HIGH_NOISE", "OOD_SHIFT", "MISSING_AGENT"]:
            cond = scenario_name
        else:
            cond = "NORMAL"

        X_test, y_test, ood_labels = gen_te.apply_condition(
            X_test_base, y_test_base,
            condition=cond,
            noise_level=noise_level,
            ood_severity=ood_severity,
            missing_agent_idx=missing_agents if missing_agents else [1],
        )

    injected_noise_max = float(np.max(np.abs(X_test[:, 1] - X_test_base[:, 1])))
    injected_noise_mean = float(np.mean(np.abs(X_test[:, 1] - X_test_base[:, 1])))

    # ── Agents ────────────────────────────────────────────────────────────
    agents = [HeterogeneousAgent(f"Agent_{i}", i) for i in range(num_agents)]
    for agent in agents:
        agent.fit(X_train, y_train)

    # Set agent health for MISSING_AGENT conditions
    if missing_agents:
        for idx in missing_agents:
            if idx < len(agents):
                agents[idx].healthy = False

    # Reliability (measured on cal set with all healthy)
    agent_reliabilities: Dict[str, float] = {}
    for agent in agents:
        was_healthy = agent.healthy
        agent.healthy = True
        cal_preds = [round(agent.predict(X_cal[i]).value) for i in range(len(X_cal))]
        agent_reliabilities[agent.agent_id] = float(np.mean(np.array(cal_preds) == y_cal))
        agent.healthy = was_healthy

    # Restore health for test
    if missing_agents:
        for idx in missing_agents:
            if idx < len(agents):
                agents[idx].healthy = False

    # ── Graph / GAT ───────────────────────────────────────────────────────
    if graph_type == "EpistemicGAT":
        gat = EpistemicGAT(input_dim=3, hidden_dim=8, output_dim=1)
        gat.fit(agents, X_train, y_train, seed=seed)
    elif graph_type == "StandardGAT":
        gat = StandardGAT(input_dim=3, hidden_dim=8, output_dim=1)
        gat.fit(agents, X_train, y_train, seed=seed)
    elif graph_type == "NoGraph":
        gat = NoGraph()
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")

    # ── Fusion ────────────────────────────────────────────────────────────
    fuser = AverageFusion() if use_avg_fusion else EpistemicWeightedFusion(eps=1e-8)

    # ── Calibration (fit on cal set) ──────────────────────────────────────
    cal_probs_list = []
    for i in range(len(X_cal)):
        res_cal = _run_single_sample(
            X_cal[i], agents, gat, fuser, None, agent_reliabilities,
            CognixConfig(), use_conformal=False
        )
        p = float(np.clip(res_cal["prob"], 1e-7, 1 - 1e-7))
        cal_probs_list.append([1.0 - p, p])

    calibrator: Optional[ConformalPredictor] = None
    if use_conformal:
        calibrator = ConformalPredictor()
        calibrator.fit(np.array(cal_probs_list), y_cal.astype(int))

    # ── Warmup (for latency measurement) ──────────────────────────────────
    for i in range(min(N_WARMUP, len(X_test))):
        _run_single_sample(X_test[i], agents, gat, fuser, calibrator,
                           agent_reliabilities, CognixConfig(), use_conformal=use_conformal)

    # ── Test loop ─────────────────────────────────────────────────────────
    all_probs:    List[float] = []
    all_epistemics: List[float] = []
    all_aleatorics: List[float] = []
    all_totals:   List[float] = []
    all_pred_sets: List[List[int]] = []
    all_attn_matrices: List[np.ndarray] = []
    latency_total: List[float] = []
    latency_unc:  List[float] = []
    latency_gat:  List[float] = []
    latency_fuse: List[float] = []
    latency_cal:  List[float] = []

    t_throughput_start = time.perf_counter()
    for i in range(len(X_test)):
        t0 = time.perf_counter()
        r = _run_single_sample(
            X_test[i], agents, gat, fuser, calibrator,
            agent_reliabilities, CognixConfig(), use_conformal=use_conformal
        )
        latency_total.append((time.perf_counter() - t0) * 1000)

        all_probs.append(r["prob"])
        all_epistemics.append(r["epistemic"])
        all_aleatorics.append(r["aleatoric"])
        all_totals.append(r["total_unc"])
        all_pred_sets.append(r["pred_set"])
        latency_unc.append(r["latency_unc_ms"])
        latency_gat.append(r["latency_gat_ms"])
        latency_fuse.append(r["latency_fuse_ms"])
        latency_cal.append(r["latency_cal_ms"])
        if r["attention_matrix"] is not None:
            all_attn_matrices.append(r["attention_matrix"])

    t_throughput_elapsed = time.perf_counter() - t_throughput_start

    # ── Compute all metrics ────────────────────────────────────────────────
    probs_arr  = np.array(all_probs)
    labels_arr = y_test.astype(int)
    epi_arr    = np.array(all_epistemics)
    ale_arr    = np.array(all_aleatorics)
    tot_arr    = np.array(all_totals)

    # 1. Predictive quality
    clf_metrics = compute_classification_metrics(probs_arr, labels_arr)

    # 2. Probabilistic quality
    nll_val  = float(-np.mean(
        labels_arr * np.log(probs_arr + 1e-15) +
        (1 - labels_arr) * np.log(1 - probs_arr + 1e-15)
    ))
    brier_val = float(brier_score(probs_arr, labels_arr))

    # 3. Calibration
    ece_val   = float(calculate_ece(probs_arr, labels_arr))
    mce_val   = float(compute_mce(probs_arr, labels_arr))
    cal_curve = compute_calibration_curve(probs_arr, labels_arr)
    cal_gap   = abs(float(np.mean(probs_arr)) - float(np.mean(labels_arr)))

    # 4. Uncertainty quality
    unc_quality = compute_uncertainty_quality(epi_arr, ale_arr, tot_arr,
                                              labels_arr, probs_arr)

    # 5. OOD detection — ID vs OOD using epistemic uncertainty
    ood_labels_int = ood_labels.astype(int)
    if ood_labels_int.sum() > 0:
        id_mask  = ood_labels_int == 0
        ood_mask = ood_labels_int == 1
        ood_metrics = compute_ood_metrics(epi_arr[id_mask], epi_arr[ood_mask])
    else:
        ood_metrics = {"auroc": float("nan"), "aupr": float("nan"),
                       "fpr95": float("nan"), "n_id": len(epi_arr), "n_ood": 0,
                       "note": "No OOD labels in this scenario"}

    # 6. Conformal prediction
    conformal_metrics = compute_conformal_metrics(all_pred_sets, labels_arr,
                                                  target_coverage=0.95)

    # 7. Selective / safety
    selective = compute_selective_risk(probs_arr, labels_arr)

    # 8. Graph / attention mechanism
    attn_metrics: Dict[str, Any] = {}
    if all_attn_matrices:
        mean_attn = np.mean(all_attn_matrices, axis=0)
        healthy_idx  = [i for i in range(num_agents) if i not in missing_agents]
        degraded_idx = list(missing_agents)
        attn_metrics = compute_attention_metrics(mean_attn, healthy_idx, degraded_idx)

    # 9. Latency
    lat_total = np.array(latency_total)
    lat_unc   = np.array(latency_unc)
    lat_gat   = np.array(latency_gat)
    lat_fuse  = np.array(latency_fuse)
    lat_cal   = np.array(latency_cal)

    def _perc(arr):
        return {
            "mean": float(np.mean(arr)),
            "p50":  float(np.percentile(arr, 50)),
            "p95":  float(np.percentile(arr, 95)),
            "p99":  float(np.percentile(arr, 99)),
            "sd":   float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        }

    latency_breakdown = {
        "total":   _perc(lat_total),
        "uncertainty": _perc(lat_unc),
        "gat":     _perc(lat_gat),
        "fusion":  _perc(lat_fuse),
        "calibration": _perc(lat_cal),
    }

    # 10. Throughput
    throughput = compute_throughput(len(X_test), t_throughput_elapsed)

    # ── Assemble result dict ───────────────────────────────────────────────
    result = {
        # Identity
        "seed":          seed,
        "graph_type":    graph_type,
        "scenario_name": scenario_name,
        "num_agents":    num_agents,
        "noise_level":   noise_level,
        "ood_severity":  ood_severity,
        "missing_agents": list(missing_agents),
        "injected_noise_max": injected_noise_max,
        "injected_noise_mean": injected_noise_mean,
        "use_avg_fusion": use_avg_fusion,
        "use_conformal":  use_conformal,
        "n_train": n_train, "n_cal": n_cal, "n_test": n_test,

        # Predictive quality
        **{f"clf_{k}": v for k, v in clf_metrics.items()},

        # Probabilistic
        "nll":   nll_val,
        "brier": brier_val,

        # Calibration
        "ece": ece_val,
        "mce": mce_val,
        "calibration_gap": cal_gap,
        "calibration_curve": cal_curve,

        # Uncertainty quality
        **{f"unc_{k}": v for k, v in unc_quality.items()},

        # OOD
        "ood_auroc": ood_metrics.get("auroc"),
        "ood_aupr":  ood_metrics.get("aupr"),
        "ood_fpr95": ood_metrics.get("fpr95"),
        "ood_n_id":  ood_metrics.get("n_id"),
        "ood_n_ood": ood_metrics.get("n_ood"),
        "ood_note":  ood_metrics.get("note", ""),

        # Conformal
        **{f"cp_{k}": v for k, v in conformal_metrics.items()
           if k != "set_size_distribution"},
        "cp_set_size_distribution": conformal_metrics.get("set_size_distribution", {}),

        # Selective / safety
        "selective_full_risk": selective.get("full_coverage_risk"),
        "selective_aurc": selective.get("aurc"),
        "selective_risk_coverage_curve": selective.get("risk_coverage_curve", []),

        # Graph mechanism
        **{f"attn_{k}": v for k, v in attn_metrics.items()},

        # Latency
        "latency": latency_breakdown,

        # Throughput
        "throughput_decisions_per_sec": throughput,
    }

    return result


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _checkpoint_path(scenario: str, graph_type: str, seed: int,
                     extra: str = "") -> Path:
    tag = f"{scenario}_{graph_type}_{seed}"
    if extra:
        tag += f"_{extra}"
    return RAW_DIR / f"{tag}.json"


def _is_done(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 10


def _save_checkpoint(path: Path, data: Dict):
    with open(path, "w") as f:
        # Convert numpy types for JSON
        json.dump(_serialize(data), f, indent=2)


def _serialize(obj):
    """Recursively convert numpy types to Python native for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


# ── Resource measurement ──────────────────────────────────────────────────────

def _get_resource_snapshot() -> Dict[str, Any]:
    """Capture CPU and RAM usage."""
    res: Dict[str, Any] = {}
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        res["ram_rss_mb"] = float(proc.memory_info().rss / 1e6)
        res["ram_vms_mb"] = float(proc.memory_info().vms / 1e6)
        cpu_pct = psutil.cpu_percent(interval=0.5)
        res["cpu_percent"] = float(cpu_pct)
        mem = psutil.virtual_memory()
        res["system_ram_total_gb"] = float(mem.total / 1e9)
        res["system_ram_available_gb"] = float(mem.available / 1e9)
    except Exception as e:
        res["resource_error"] = str(e)
    return res


def _get_environment() -> Dict[str, Any]:
    """Collect hardware / software environment info."""
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT), stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_hash = "unknown"

    env = {
        "git_hash": git_hash,
        "python_version": platform.python_version(),
        "os": platform.system() + " " + platform.release(),
        "cpu": platform.processor() or "unknown",
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda or "N/A",
        "n_seeds": len(SEEDS),
        "seeds": SEEDS,
        "graph_modes": GRAPH_MODES,
        "ablation_modes": ABLATION_MODES,
        "main_scenarios": MAIN_SCENARIOS,
        "n_train": N_TRAIN,
        "n_cal": N_CAL,
        "n_test": N_TEST,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    try:
        import psutil
        mem = psutil.virtual_memory()
        env["system_ram_gb"] = round(mem.total / 1e9, 1)
    except Exception:
        env["system_ram_gb"] = "unknown"

    return env


# ── Main evaluation loops ─────────────────────────────────────────────────────

def run_main_scenarios(verbose: bool = False):
    """Run all main scenarios × graph modes × seeds."""
    print("\n" + "="*70)
    print("PHASE 1: Main scenarios (NoGraph / StandardGAT / EpistemicGAT)")
    print("="*70)

    for scenario_name in MAIN_SCENARIOS:
        scen_cfg = get_scenario_config(scenario_name)

        for graph_type in GRAPH_MODES:
            completed = 0
            for seed in SEEDS:
                run_id = f"{scenario_name}_{graph_type}_{seed}"
                cp = _checkpoint_path(scenario_name, graph_type, seed)

                if _is_done(cp):
                    completed += 1
                    continue

                _reliability_log["attempted"] += 1
                try:
                    result = run_evaluation(
                        seed=seed,
                        graph_type=graph_type,
                        scenario_name=scenario_name,
                        noise_level=scen_cfg["noise_level"],
                        ood_severity=scen_cfg["ood_severity"],
                        missing_agents=scen_cfg["missing_agents"],
                        conflict_fraction=scen_cfg.get("conflict_fraction", 0.0),
                    )
                    _check_nan_inf(
                        {k: v for k, v in result.items() if isinstance(v, (int, float))},
                        run_id
                    )
                    _save_checkpoint(cp, result)
                    _reliability_log["successful"] += 1
                    completed += 1
                    if verbose:
                        print(f"  [{run_id}] acc={result.get('clf_accuracy', 'N/A'):.3f}")

                except Exception as e:
                    _log_failure(run_id, e)
                    print(f"  [FAILED] {run_id}: {e}")

            print(f"  {scenario_name:20s} | {graph_type:12s} | {completed}/{len(SEEDS)} done")


def run_ablations(verbose: bool = False):
    """Run ablations: AverageFusion, NoConformal."""
    print("\n" + "="*70)
    print("PHASE 2: Ablation studies")
    print("="*70)

    # AverageFusion ablation (no epistemic weighting) — use EpistemicGAT otherwise
    for scenario_name in ["NORMAL", "HIGH_NOISE", "MISSING_AGENT", "OOD_SHIFT"]:
        scen_cfg = get_scenario_config(scenario_name)

        for ablation, kwargs in [
            ("AverageFusion", {"use_avg_fusion": True, "use_conformal": True}),
            ("NoConformal",   {"use_avg_fusion": False, "use_conformal": False}),
        ]:
            completed = 0
            for seed in SEEDS:
                run_id = f"{scenario_name}_{ablation}_{seed}"
                cp = _checkpoint_path(scenario_name, ablation, seed)

                if _is_done(cp):
                    completed += 1
                    continue

                _reliability_log["attempted"] += 1
                try:
                    result = run_evaluation(
                        seed=seed,
                        graph_type="EpistemicGAT",
                        scenario_name=scenario_name,
                        noise_level=scen_cfg["noise_level"],
                        ood_severity=scen_cfg["ood_severity"],
                        missing_agents=scen_cfg["missing_agents"],
                        **kwargs,
                    )
                    result["ablation_mode"] = ablation
                    _save_checkpoint(cp, result)
                    _reliability_log["successful"] += 1
                    completed += 1
                except Exception as e:
                    _log_failure(run_id, e)
                    print(f"  [FAILED] {run_id}: {e}")

            print(f"  {scenario_name:20s} | {ablation:14s} | {completed}/{len(SEEDS)} done")


def run_robustness_sweep(verbose: bool = False):
    """Noise severity sweep for robustness curves."""
    print("\n" + "="*70)
    print("PHASE 3: Robustness sweep (noise severity)")
    print("="*70)

    for sev_idx, noise_level in enumerate(NOISE_SEVERITY_LEVELS):
        for graph_type in GRAPH_MODES:
            completed = 0
            for seed in SEEDS:
                run_id = f"NOISE_SWEEP_{graph_type}_{seed}_sev{sev_idx}"
                cp = _checkpoint_path(f"NOISE_SWEEP_sev{sev_idx}", graph_type, seed)

                if _is_done(cp):
                    completed += 1
                    continue

                _reliability_log["attempted"] += 1
                try:
                    scen_cfg = get_scenario_config("NOISE_SWEEP", severity_idx=sev_idx)
                    result = run_evaluation(
                        seed=seed,
                        graph_type=graph_type,
                        scenario_name="NOISE_SWEEP",
                        noise_level=noise_level,
                    )
                    result["noise_severity_idx"] = sev_idx
                    _save_checkpoint(cp, result)
                    _reliability_log["successful"] += 1
                    completed += 1
                except Exception as e:
                    _log_failure(run_id, e)
                    print(f"  [FAILED] {run_id}: {e}")

            print(f"  Noise={noise_level:.1f} | {graph_type:12s} | {completed}/{len(SEEDS)} done")

    # Also save the sweep to ROBUST_DIR
    _aggregate_robustness_sweep()


def _aggregate_robustness_sweep():
    """Collect noise sweep results into summary JSON."""
    sweep_data: Dict[str, Any] = {"severity_levels": NOISE_SEVERITY_LEVELS, "data": {}}

    for graph_type in GRAPH_MODES:
        sweep_data["data"][graph_type] = {}
        for sev_idx, noise_level in enumerate(NOISE_SEVERITY_LEVELS):
            results = []
            for seed in SEEDS:
                cp = _checkpoint_path(f"NOISE_SWEEP_sev{sev_idx}", graph_type, seed)
                if cp.exists():
                    with open(cp) as f:
                        results.append(json.load(f))
            if results:
                sweep_data["data"][graph_type][str(noise_level)] = {
                    "accuracy": compute_summary_stats([r.get("clf_accuracy") for r in results
                                                       if r.get("clf_accuracy") is not None]),
                    "f1":       compute_summary_stats([r.get("clf_f1") for r in results
                                                       if r.get("clf_f1") is not None]),
                    "ece":      compute_summary_stats([r.get("ece") for r in results
                                                       if r.get("ece") is not None]),
                    "brier":    compute_summary_stats([r.get("brier") for r in results
                                                       if r.get("brier") is not None]),
                    "nll":      compute_summary_stats([r.get("nll") for r in results
                                                       if r.get("nll") is not None]),
                    "epistemic": compute_summary_stats([r.get("unc_mean_epistemic") for r in results
                                                        if r.get("unc_mean_epistemic") is not None]),
                    "coverage": compute_summary_stats([r.get("cp_empirical_coverage") for r in results
                                                       if r.get("cp_empirical_coverage") is not None]),
                    "set_size": compute_summary_stats([r.get("cp_mean_set_size") for r in results
                                                       if r.get("cp_mean_set_size") is not None]),
                }

    with open(ROBUST_DIR / "noise_severity_sweep.json", "w") as f:
        json.dump(_serialize(sweep_data), f, indent=2)


def run_resilience_sweep(verbose: bool = False):
    """Multi-agent failure resilience sweep."""
    print("\n" + "="*70)
    print("PHASE 4: Multi-agent resilience sweep")
    print("="*70)

    max_fail = min(2, 3)  # 0, 1, 2 failed agents (cap at num_agents-1 = 3)

    for failed_count in AGENT_FAILURE_COUNTS:
        for graph_type in GRAPH_MODES:
            completed = 0
            for seed in SEEDS:
                run_id = f"AGENT_FAILURE_{graph_type}_{seed}_f{failed_count}"
                cp = _checkpoint_path(f"AGENT_FAILURE_f{failed_count}", graph_type, seed)

                if _is_done(cp):
                    completed += 1
                    continue

                _reliability_log["attempted"] += 1
                try:
                    missing = list(range(failed_count))
                    result = run_evaluation(
                        seed=seed,
                        graph_type=graph_type,
                        scenario_name="MISSING_AGENT" if failed_count > 0 else "NORMAL",
                        missing_agents=missing,
                    )
                    result["failed_agent_count"] = failed_count
                    _save_checkpoint(cp, result)
                    _reliability_log["successful"] += 1
                    completed += 1
                except Exception as e:
                    _log_failure(run_id, e)
                    print(f"  [FAILED] {run_id}: {e}")

            print(f"  Failed={failed_count} | {graph_type:12s} | {completed}/{len(SEEDS)} done")


def run_scalability(verbose: bool = False):
    """Scalability sweep: vary agent count."""
    print("\n" + "="*70)
    print("PHASE 5: Scalability (agent count)")
    print("="*70)

    agent_counts = [2, 4, 8, 16]

    for n_agents in agent_counts:
        for graph_type in GRAPH_MODES:
            completed = 0
            results_acc = []

            for seed in SCALABILITY_SEEDS:
                run_id = f"SCALABILITY_{graph_type}_{seed}_n{n_agents}"
                cp = SCALABILITY_DIR / f"{run_id}.json"

                if _is_done(cp):
                    completed += 1
                    try:
                        with open(cp) as f:
                            results_acc.append(json.load(f))
                    except Exception:
                        pass
                    continue

                _reliability_log["attempted"] += 1
                try:
                    t_start = time.perf_counter()
                    result = run_evaluation(
                        seed=seed,
                        graph_type=graph_type,
                        scenario_name="NORMAL",
                        num_agents=n_agents,
                    )
                    elapsed = time.perf_counter() - t_start
                    result["n_agents_scalability"] = n_agents
                    result["run_elapsed_seconds"] = elapsed

                    # Resource snapshot after run
                    res_snap = _get_resource_snapshot()
                    result["resource_snapshot"] = res_snap

                    _save_checkpoint(cp, result)
                    results_acc.append(result)
                    _reliability_log["successful"] += 1
                    completed += 1
                except Exception as e:
                    _log_failure(run_id, e)
                    print(f"  [FAILED] {run_id}: {e}")

            print(f"  n_agents={n_agents:2d} | {graph_type:12s} | {completed}/{len(SCALABILITY_SEEDS)} done")

            # Save summary
            if results_acc:
                summary = {
                    "n_agents": n_agents,
                    "graph_type": graph_type,
                    "seeds": SCALABILITY_SEEDS,
                    "latency_p50":  compute_summary_stats(
                        [r.get("latency", {}).get("total", {}).get("p50") for r in results_acc
                         if r.get("latency")]),
                    "latency_p95":  compute_summary_stats(
                        [r.get("latency", {}).get("total", {}).get("p95") for r in results_acc
                         if r.get("latency")]),
                    "throughput":   compute_summary_stats(
                        [r.get("throughput_decisions_per_sec") for r in results_acc
                         if r.get("throughput_decisions_per_sec") is not None]),
                    "accuracy":     compute_summary_stats(
                        [r.get("clf_accuracy") for r in results_acc
                         if r.get("clf_accuracy") is not None]),
                    "ece":          compute_summary_stats(
                        [r.get("ece") for r in results_acc if r.get("ece") is not None]),
                    "ram_mb":       compute_summary_stats(
                        [r.get("resource_snapshot", {}).get("ram_rss_mb")
                         for r in results_acc if r.get("resource_snapshot")]),
                }
                with open(SCALABILITY_DIR / f"summary_{graph_type}_n{n_agents}.json", "w") as f:
                    json.dump(_serialize(summary), f, indent=2)


def run_ood_collection(verbose: bool = False):
    """
    Collect OOD detection results from already-run NORMAL vs OOD_SHIFT checkpoints.
    Uses epistemic uncertainty as OOD score.
    ID = NORMAL samples, OOD = OOD_SHIFT samples.
    """
    print("\n" + "="*70)
    print("PHASE 6: OOD detection aggregation (NORMAL vs OOD_SHIFT)")
    print("="*70)

    for graph_type in GRAPH_MODES:
        # Collect epistemics from NORMAL (ID) and OOD_SHIFT (OOD) checkpoint files
        id_epistemics_all  = []
        ood_epistemics_all = []

        for seed in SEEDS:
            cp_normal = _checkpoint_path("NORMAL", graph_type, seed)
            cp_ood    = _checkpoint_path("OOD_SHIFT", graph_type, seed)

            if cp_normal.exists():
                with open(cp_normal) as f:
                    d = json.load(f)
                epi = d.get("unc_mean_epistemic")
                if epi is not None:
                    id_epistemics_all.append(epi)

            if cp_ood.exists():
                with open(cp_ood) as f:
                    d = json.load(f)
                epi = d.get("unc_mean_epistemic")
                if epi is not None:
                    ood_epistemics_all.append(epi)

        if id_epistemics_all and ood_epistemics_all:
            ood_res = compute_ood_metrics(
                np.array(id_epistemics_all), np.array(ood_epistemics_all)
            )
            ood_res["graph_type"] = graph_type
            ood_res["note"] = (
                "OOD detection using mean epistemic uncertainty per run as the OOD score. "
                "ID=NORMAL runs, OOD=OOD_SHIFT runs. "
                "Threshold not tuned on final benchmark results."
            )
            with open(OOD_DIR / f"ood_{graph_type}.json", "w") as f:
                json.dump(_serialize(ood_res), f, indent=2)
            print(f"  {graph_type:12s} | AUROC={ood_res.get('auroc', 'nan'):.3f}  AUPR={ood_res.get('aupr', 'nan'):.3f}  FPR95={ood_res.get('fpr95', 'nan'):.3f}")
        else:
            print(f"  {graph_type:12s} | Insufficient data for OOD evaluation")


# ── Summary aggregation ───────────────────────────────────────────────────────

def aggregate_summaries():
    """Load all checkpoints, compute per-scenario/mode summary statistics."""
    print("\n" + "="*70)
    print("AGGREGATING summaries and statistics ...")
    print("="*70)

    all_summaries: Dict[str, Any] = {}

    for scenario_name in MAIN_SCENARIOS:
        all_summaries[scenario_name] = {}
        for graph_type in GRAPH_MODES + ["AverageFusion", "NoConformal"]:
            results = []
            for seed in SEEDS:
                cp = _checkpoint_path(scenario_name, graph_type, seed)
                if cp.exists():
                    try:
                        with open(cp) as f:
                            results.append(json.load(f))
                    except Exception:
                        pass

            if not results:
                continue

            def _get(key, results=results):
                return [r.get(key) for r in results if r.get(key) is not None]

            summary = {
                "n": len(results),
                "graph_type": graph_type,
                "scenario": scenario_name,
                "accuracy":           compute_summary_stats(_get("clf_accuracy")),
                "balanced_accuracy":  compute_summary_stats(_get("clf_balanced_accuracy")),
                "precision":          compute_summary_stats(_get("clf_precision")),
                "recall":             compute_summary_stats(_get("clf_recall")),
                "f1":                 compute_summary_stats(_get("clf_f1")),
                "auroc_clf":          compute_summary_stats(_get("clf_auroc")),
                "nll":                compute_summary_stats(_get("nll")),
                "brier":              compute_summary_stats(_get("brier")),
                "ece":                compute_summary_stats(_get("ece")),
                "mce":                compute_summary_stats(_get("mce")),
                "calibration_gap":    compute_summary_stats(_get("calibration_gap")),
                "mean_epistemic":     compute_summary_stats(_get("unc_mean_epistemic")),
                "mean_aleatoric":     compute_summary_stats(_get("unc_mean_aleatoric")),
                "mean_total_unc":     compute_summary_stats(_get("unc_mean_total")),
                "epi_separation":     compute_summary_stats(_get("unc_epistemic_separation")),
                "auroc_error_det":    compute_summary_stats(_get("unc_auroc_error_detection")),
                "aupr_error_det":     compute_summary_stats(_get("unc_aupr_error_detection")),
                "cp_coverage":        compute_summary_stats(_get("cp_empirical_coverage")),
                "cp_set_size":        compute_summary_stats(_get("cp_mean_set_size")),
                "cp_coverage_gap":    compute_summary_stats(_get("cp_coverage_gap")),
                "selective_risk":     compute_summary_stats(_get("selective_full_risk")),
                "selective_aurc":     compute_summary_stats(_get("selective_aurc")),
                "throughput":         compute_summary_stats(_get("throughput_decisions_per_sec")),
                "latency_mean":       compute_summary_stats(
                    [r.get("latency", {}).get("total", {}).get("mean") for r in results
                     if r.get("latency")]),
                "latency_p50":        compute_summary_stats(
                    [r.get("latency", {}).get("total", {}).get("p50") for r in results
                     if r.get("latency")]),
                "latency_p95":        compute_summary_stats(
                    [r.get("latency", {}).get("total", {}).get("p95") for r in results
                     if r.get("latency")]),
                "latency_p99":        compute_summary_stats(
                    [r.get("latency", {}).get("total", {}).get("p99") for r in results
                     if r.get("latency")]),
            }

            all_summaries[scenario_name][graph_type] = summary

            csv_out = SUMMARIES_DIR / f"{scenario_name}_{graph_type}.json"
            with open(csv_out, "w") as f:
                json.dump(_serialize(summary), f, indent=2)

    with open(SUMMARIES_DIR / "all_summaries.json", "w") as f:
        json.dump(_serialize(all_summaries), f, indent=2)

    return all_summaries


def compute_statistical_tests(all_summaries: Dict):
    """
    Compute paired Wilcoxon tests and Cohen's d for all main metric families.
    Apply Holm-Bonferroni correction within families.
    """
    print("  Computing statistical tests ...")

    stat_results: Dict[str, Any] = {}

    metrics_to_test = [
        "clf_accuracy", "clf_f1", "nll", "brier", "ece", "mce",
        "unc_mean_epistemic", "unc_epistemic_separation",
        "unc_auroc_error_detection",
        "cp_empirical_coverage", "cp_mean_set_size",
    ]

    comparisons = [
        ("NoGraph", "StandardGAT"),
        ("NoGraph", "EpistemicGAT"),
        ("StandardGAT", "EpistemicGAT"),
    ]

    for scenario_name in MAIN_SCENARIOS:
        stat_results[scenario_name] = {}
        p_values_raw: Dict[str, float] = {}

        for base_graph, exp_graph in comparisons:
            comp_key = f"{base_graph}_vs_{exp_graph}"
            stat_results[scenario_name][comp_key] = {}

            for metric in metrics_to_test:
                # Collect paired per-seed values
                base_vals, exp_vals = [], []
                for seed in SEEDS:
                    cp_base = _checkpoint_path(scenario_name, base_graph, seed)
                    cp_exp  = _checkpoint_path(scenario_name, exp_graph, seed)
                    if cp_base.exists() and cp_exp.exists():
                        try:
                            with open(cp_base) as f:
                                db = json.load(f)
                            with open(cp_exp) as f:
                                de = json.load(f)
                            vb = db.get(metric)
                            ve = de.get(metric)
                            if vb is not None and ve is not None:
                                base_vals.append(float(vb))
                                exp_vals.append(float(ve))
                        except Exception:
                            pass

                if len(base_vals) >= 5:
                    try:
                        res = paired_analysis(base_vals, exp_vals)
                        stat_results[scenario_name][comp_key][metric] = res
                        p_values_raw[f"{comp_key}_{metric}"] = res.get("p_value_wilcoxon", 1.0) or 1.0
                    except Exception as e:
                        stat_results[scenario_name][comp_key][metric] = {"error": str(e)}
                else:
                    stat_results[scenario_name][comp_key][metric] = {
                        "note": f"Insufficient paired data (n={len(base_vals)})"
                    }

        # Holm-Bonferroni correction over all tests in this scenario
        if p_values_raw:
            adj_p = holm_bonferroni_dict(p_values_raw)
            stat_results[scenario_name]["holm_bonferroni_adjusted_p"] = adj_p

    with open(STATS_DIR / "statistical_tests.json", "w") as f:
        json.dump(_serialize(stat_results), f, indent=2)

    return stat_results


def compute_latency_comparison(all_summaries: Dict):
    """Build the latency comparison table (across graph modes, NORMAL scenario)."""
    print("  Building latency comparison table ...")
    lat_table: Dict[str, Any] = {"scenario": "NORMAL"}

    for graph_type in GRAPH_MODES:
        if "NORMAL" in all_summaries and graph_type in all_summaries["NORMAL"]:
            s = all_summaries["NORMAL"][graph_type]
            mean_val = s.get("latency_mean", {}).get("mean") if s.get("latency_mean") else s.get("latency_p50", {}).get("mean")
            lat_table[graph_type] = {
                "mean_ms":   mean_val,
                "p50_ms":    s.get("latency_p50", {}).get("mean"),
                "p95_ms":    s.get("latency_p95", {}).get("mean"),
                "p99_ms":    s.get("latency_p99", {}).get("mean"),
                "throughput": s.get("throughput", {}).get("mean"),
            }
        else:
            lat_table[graph_type] = {k: None for k in ["mean_ms", "p50_ms", "p95_ms", "p99_ms", "throughput"]}

    # Compute deltas
    def _delta(base, exp, key):
        bv = lat_table.get(base, {}).get(key)
        ev = lat_table.get(exp, {}).get(key)
        if bv and ev:
            delta = ev - bv
            pct   = (delta / bv * 100) if bv != 0 else float("nan")
            return {"delta_ms": delta, "pct_change": pct,
                    "direction": "increase" if delta > 0 else "decrease"}
        return {"delta_ms": None, "pct_change": None, "direction": None}

    lat_table["comparisons"] = {
        "StandardGAT_vs_NoGraph":    _delta("NoGraph", "StandardGAT", "p50_ms"),
        "EpistemicGAT_vs_StandardGAT": _delta("StandardGAT", "EpistemicGAT", "p50_ms"),
        "EpistemicGAT_vs_NoGraph":   _delta("NoGraph", "EpistemicGAT", "p50_ms"),
    }
    lat_table["mean_comparisons"] = {
        "StandardGAT_vs_NoGraph":    _delta("NoGraph", "StandardGAT", "mean_ms"),
        "EpistemicGAT_vs_StandardGAT": _delta("StandardGAT", "EpistemicGAT", "mean_ms"),
        "EpistemicGAT_vs_NoGraph":   _delta("NoGraph", "EpistemicGAT", "mean_ms"),
    }

    with open(LATENCY_DIR / "latency_comparison.json", "w") as f:
        json.dump(_serialize(lat_table), f, indent=2)

    return lat_table


# ── Main entrypoint ───────────────────────────────────────────────────────────

def main(smoke: bool = False):
    print("\n" + "="*70)
    print("COGNIX UNIVERSAL BENCHMARK — Starting")
    if smoke:
        print("[SMOKE TEST MODE: 1 seed, 50 samples, all 3 modes]")
    print("="*70)

    # Save environment metadata
    env = _get_environment()
    with open(RESULTS_DIR / "metadata.json", "w") as f:
        json.dump(env, f, indent=2)

    t_total_start = time.time()

    if smoke:
        # Smoke test: 1 seed, 50 samples, 3 modes, NORMAL only
        global SEEDS, N_TRAIN, N_CAL, N_TEST, N_WARMUP
        SEEDS = [42]
        N_TRAIN = 50; N_CAL = 50; N_TEST = 50; N_WARMUP = 5

        for graph_type in GRAPH_MODES:
            cp = _checkpoint_path("NORMAL", graph_type, 42)
            cp.unlink(missing_ok=True)  # Force re-run for smoke test

        run_main_scenarios(verbose=True)
        print("\n[SMOKE TEST] Checking saved files...")
        ok = True
        for graph_type in GRAPH_MODES:
            cp = _checkpoint_path("NORMAL", graph_type, 42)
            if not cp.exists():
                print(f"  FAIL: {cp} not found!")
                ok = False
            else:
                with open(cp) as f:
                    d = json.load(f)
                acc = d.get("clf_accuracy")
                ece = d.get("ece")
                lat = d.get("latency", {}).get("total", {}).get("p50")
                nan_found = any(v is None for v in [acc, ece, lat])
                status = "PASS" if not nan_found else "WARN (NaN/None detected)"
                print(f"  {graph_type:12s}: acc={acc:.3f}, ece={ece:.3f}, lat_p50={lat:.2f}ms - {status}")
                if nan_found:
                    ok = False
        if ok:
            print("\n[SMOKE TEST PASSED] [OK] All metrics computed, outputs saved.")
        else:
            print("\n[SMOKE TEST WARNINGS] Check above output for issues.")
        return

    # Full run
    run_main_scenarios()
    run_ablations()
    run_robustness_sweep()
    run_resilience_sweep()
    run_scalability()
    run_ood_collection()

    # Aggregate
    all_summaries = aggregate_summaries()
    stat_results  = compute_statistical_tests(all_summaries)
    lat_table     = compute_latency_comparison(all_summaries)

    # Save reliability log
    _reliability_log["success_rate"] = (
        _reliability_log["successful"] / max(_reliability_log["attempted"], 1)
    )
    with open(RESULTS_DIR / "reliability_log.json", "w") as f:
        json.dump(_serialize(_reliability_log), f, indent=2)

    try:
        from report_generator import generate_report
        generate_report()
    except Exception as e:
        print(f"Error generating final report: {e}")

    elapsed = time.time() - t_total_start
    print(f"\n{'='*70}")
    print(f"BENCHMARK COMPLETE in {elapsed/60:.1f} minutes")
    print(f"Successful: {_reliability_log['successful']} / {_reliability_log['attempted']}")
    print(f"Failed:     {_reliability_log['failed']}")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Cognix Universal Benchmark")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test only")
    args = parser.parse_args()
    main(smoke=args.smoke)
