#!/usr/bin/env python3
"""
COGNIX Experiment 2B — Phase-B TEST label reader and frozen scorer.

Safety properties:
- --self-test never opens the TEST archive.
- --execute-phase-b requires an explicit authorization token.
- Only exact */anomaly-observation.feather members may be extractfile()'d.
- anomaly-front/* and every non-label payload are never opened.
- Phase-A predictions are authenticated before label access.
- No model execution, training, calibration, threshold tuning, or checkpoint selection.
- TEST thresholds are never selected/recomputed.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import time
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(r"C:\Users\Aditya\Cognix2_recovered")
ROOT = REPO / "reports" / "carla_train_full_v1"

ARCHIVE = Path(r"E:\carlanomaly-base-test.tar.gz")
ARCHIVE_BYTES = 91538225599
ARCHIVE_SHA256 = "267e48f2249deb0269ad950aa81bca57dc02e3bbdf2d73acc172af267b18254a"

PHASE_A_DIR = ROOT / "graph_test_phase_A_predictions_v1"
PHASE_A_SEAL = "abec3d65cd52ae569751eb81d3dee7027eaaffa4283197cf307e806ab6f5267b"

PROTOCOL_DIR = ROOT / "graph_test_evaluation_protocol_freeze_bundle_v1"
PROTOCOL_SEAL = "d4ceb25023f49acc96ab2a62f05f7fdc92c88c608705ce5fbc34dfe64367691e"

PHASE_B_PREP = ROOT / "graph_test_phase_B_scoring_bundle_v1_staging"
PHASE_B_OUTPUT_STAGING = ROOT / "graph_test_phase_B_results_v1_staging"
PHASE_B_OUTPUT_FINAL = ROOT / "graph_test_phase_B_results_v1"

EXPECTED_SCENARIOS = 627
EXPECTED_TICKS = 300
SCORED_TICKS = 299
EXPECTED_PREDICTION_ROWS = EXPECTED_SCENARIOS * SCORED_TICKS
EXPECTED_RUNS = 15

AUTHORIZATION = "PHASE_B_TEST_LABEL_ACCESS_AUTHORIZED_V1"


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verify_sealed_dir(path: Path, expected_seal: str) -> dict[str, str]:
    sums = path / "SHA256SUMS"
    detached = path / "SHA256SUMS.sha256"
    require(path.is_dir() and sums.is_file() and detached.is_file(),
            f"Missing sealed bundle files: {path}")
    actual_seal = hashlib.sha256(sums.read_bytes()).hexdigest()
    recorded = detached.read_text(encoding="utf-8").strip().split()[0]
    require(actual_seal == recorded == expected_seal,
            f"Bundle seal mismatch for {path}: actual={actual_seal} recorded={recorded}")
    entries: dict[str, str] = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        h, rel = line.split("  ", 1)
        fp = path / rel
        require(fp.is_file(), f"Missing sealed payload: {fp}")
        require(sha256_file(fp) == h, f"Payload hash mismatch: {fp}")
        entries[rel] = h
    return entries


def load_phase_a_inputs() -> dict[str, Any]:
    entries = verify_sealed_dir(PHASE_A_DIR, PHASE_A_SEAL)
    require("predictions_and_nodes.npz" in entries, "Missing sealed predictions NPZ")
    require("scenario_stream_order.json" in entries, "Missing sealed scenario order")
    require("manifest.json" in entries, "Missing sealed Phase-A manifest")

    manifest = json.loads((PHASE_A_DIR / "manifest.json").read_text(encoding="utf-8"))
    require(manifest.get("status") == "PHASE_A_LABEL_BLIND_PREDICTIONS_COMPLETE",
            "Phase-A completion status mismatch")
    require(manifest.get("scenario_count") == EXPECTED_SCENARIOS, "Phase-A scenario count mismatch")
    require(manifest.get("prediction_rows") == EXPECTED_PREDICTION_ROWS, "Phase-A row count mismatch")
    require(manifest.get("run_count") == EXPECTED_RUNS, "Phase-A run count mismatch")
    require(manifest.get("test_labels_opened") is False, "Phase-A says TEST labels were opened")
    require(manifest.get("metrics_computed") is False, "Phase-A says metrics were computed")

    order_obj = json.loads((PHASE_A_DIR / "scenario_stream_order.json").read_text(encoding="utf-8"))
    roots = order_obj.get("scenario_roots")
    require(isinstance(roots, list) and len(roots) == EXPECTED_SCENARIOS,
            "Scenario stream order must contain exactly 627 roots")
    require(len(set(roots)) == EXPECTED_SCENARIOS, "Duplicate scenario root in Phase-A order")

    with np.load(PHASE_A_DIR / "predictions_and_nodes.npz", allow_pickle=False) as z:
        required = {
            "nodes", "scenario_index", "tick", "run_id", "threshold",
            "q_normal", "p_corruption", "predicted_corruption"
        }
        require(set(z.files) == required, f"Unexpected Phase-A NPZ keys: {z.files}")
        scenario_index = z["scenario_index"].copy()
        tick = z["tick"].copy()
        run_id = z["run_id"].copy()
        threshold = z["threshold"].copy()
        q_normal = z["q_normal"].copy()
        p_corruption = z["p_corruption"].copy()
        predicted_corruption = z["predicted_corruption"].copy()

    require(scenario_index.shape == (EXPECTED_PREDICTION_ROWS,), "scenario_index shape mismatch")
    require(tick.shape == (EXPECTED_PREDICTION_ROWS,), "tick shape mismatch")
    require(run_id.shape == (EXPECTED_RUNS,), "run_id shape mismatch")
    require(threshold.shape == (EXPECTED_RUNS,), "threshold shape mismatch")
    require(q_normal.shape == (EXPECTED_PREDICTION_ROWS, EXPECTED_RUNS), "q_normal shape mismatch")
    require(p_corruption.shape == q_normal.shape, "p_corruption shape mismatch")
    require(predicted_corruption.shape == q_normal.shape, "predicted_corruption shape mismatch")
    require(np.isfinite(q_normal).all() and np.isfinite(p_corruption).all(), "Non-finite predictions")
    require(np.all((q_normal >= 0) & (q_normal <= 1)), "q_normal outside [0,1]")
    require(np.array_equal(p_corruption, 1.0 - q_normal), "p_corruption != 1-q_normal")
    require(np.array_equal(predicted_corruption, p_corruption >= threshold.reshape(1, -1)),
            "Sealed predicted_corruption does not match sealed thresholds")

    vals, counts = np.unique(scenario_index, return_counts=True)
    require(np.array_equal(vals, np.arange(EXPECTED_SCENARIOS, dtype=vals.dtype)),
            "scenario_index is not exactly 0..626")
    require(np.all(counts == SCORED_TICKS), "Each scenario must have exactly 299 prediction rows")
    for i in range(EXPECTED_SCENARIOS):
        st = np.sort(tick[scenario_index == i])
        require(np.array_equal(st, np.arange(1, 300, dtype=st.dtype)),
                f"Scenario index {i} does not have ticks 1..299 exactly")

    return {
        "scenario_roots": roots,
        "scenario_index": scenario_index,
        "tick": tick,
        "run_id": run_id,
        "threshold": threshold,
        "q_normal": q_normal,
        "p_corruption": p_corruption,
        "predicted_corruption": predicted_corruption,
        "phase_a_manifest": manifest,
    }


def bce_samples(q: np.ndarray, y: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    require(q.shape == y.shape and q.ndim == 1 and q.size != 0, "BCE alignment")
    require(np.isfinite(q).all() and np.all((q >= 0) & (q <= 1)), "BCE q domain")
    require(np.all((y == 0) | (y == 1)), "BCE y domain")
    return -(y * np.log(np.clip(q, 1e-7, 1 - 1e-7)) +
             (1 - y) * np.log(np.clip(1 - q, 1e-7, 1 - 1e-7)))


def corruption_f1(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    require(y.shape == pred.shape, "F1 alignment")
    tp = int(np.count_nonzero(y & pred))
    fp = int(np.count_nonzero(~y & pred))
    fn = int(np.count_nonzero(y & ~pred))
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom else 0.0


def auroc_corruption(p: np.ndarray, labels: np.ndarray) -> float | None:
    p = np.asarray(p, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    require(p.shape == labels.shape and p.ndim == 1 and p.size != 0, "AUROC alignment")
    require(np.isfinite(p).all() and np.all((labels == 0) | (labels == 1)), "AUROC domain")
    pos = int(labels.sum())
    neg = int((1 - labels).sum())
    if not pos or not neg:
        return None
    order = np.argsort(p, kind="stable")
    _, starts, counts = np.unique(p[order], return_index=True, return_counts=True)
    gp = np.add.reduceat(labels[order], starts)
    gn = counts - gp
    return float(np.sum(gp * (np.cumsum(gn) - gn + 0.5 * gn)) / (pos * neg))


def paired_summary(values: dict[int, float]) -> dict[str, Any]:
    seeds = [101, 202, 303, 404, 505]
    require(set(values) == set(seeds), "All five paired seeds required")
    a = np.asarray([values[s] for s in seeds], dtype=np.float64)
    require(np.isfinite(a).all(), "All five paired differences must be finite")
    return {
        "per_seed": {str(s): float(values[s]) for s in seeds},
        "mean": float(a.mean()),
        "sample_SD": float(a.std(ddof=1)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def _parse_run_id(run_id: str) -> tuple[str, int]:
    method, seed_s = run_id.rsplit("_", 1)
    seed = int(seed_s)
    require(method in {"NoGraph", "StandardGAT", "EpistemicGAT"}, f"Unexpected method: {method}")
    require(seed in {101, 202, 303, 404, 505}, f"Unexpected seed: {seed}")
    return method, seed


def score_all(phase_a: dict[str, Any], target_corruption: np.ndarray) -> dict[str, Any]:
    target_corruption = np.asarray(target_corruption, dtype=np.int64)
    require(target_corruption.shape == (EXPECTED_PREDICTION_ROWS,), "Target shape mismatch")
    require(np.all((target_corruption == 0) | (target_corruption == 1)), "Target domain mismatch")
    target_normal = 1 - target_corruption

    si = phase_a["scenario_index"]
    run_ids = [str(x) for x in phase_a["run_id"].tolist()]
    q = phase_a["q_normal"]
    p = phase_a["p_corruption"]
    pred = phase_a["predicted_corruption"]

    per_run: dict[str, Any] = {}
    for j, rid in enumerate(run_ids):
        per_scenario: dict[str, Any] = {}
        bces: list[float] = []
        f1s: list[float] = []
        aucs: list[float | None] = []
        undefined: list[str] = []
        for sidx, root in enumerate(phase_a["scenario_roots"]):
            m = si == sidx
            yn = target_normal[m]
            yc = target_corruption[m]
            bce = float(bce_samples(q[m, j], yn).mean())
            f1 = corruption_f1(yc, pred[m, j])
            auc = auroc_corruption(p[m, j], yc)
            per_scenario[root] = {
                "BCE": bce,
                "F1": f1,
                "AUROC": auc,
                "normal_count": int(np.sum(yn == 1)),
                "corruption_count": int(np.sum(yc == 1)),
            }
            bces.append(bce)
            f1s.append(f1)
            aucs.append(auc)
            if auc is None:
                undefined.append(root)

        macro_auc = None if undefined else float(np.mean(np.asarray(aucs, dtype=np.float64)))
        per_run[rid] = {
            "BCE": float(np.mean(np.asarray(bces, dtype=np.float64))),
            "F1": float(np.mean(np.asarray(f1s, dtype=np.float64))),
            "AUROC": macro_auc,
            "AUROC_undefined_scenarios": undefined,
            "per_scenario": per_scenario,
        }

    parsed = {rid: _parse_run_id(rid) for rid in run_ids}
    lookup = {(method, seed): rid for rid, (method, seed) in parsed.items()}
    require(len(lookup) == EXPECTED_RUNS, "Run ID uniqueness mismatch")

    comparisons: dict[str, Any] = {}
    for metric in ("BCE", "F1", "AUROC"):
        metric_out: dict[str, Any] = {}
        for label, a, b in (
            ("StandardGAT_minus_NoGraph", "StandardGAT", "NoGraph"),
            ("EpistemicGAT_minus_StandardGAT", "EpistemicGAT", "StandardGAT"),
        ):
            vals: dict[int, float] = {}
            unavailable: dict[str, Any] | None = None
            for seed in (101, 202, 303, 404, 505):
                av = per_run[lookup[(a, seed)]][metric]
                bv = per_run[lookup[(b, seed)]][metric]
                if av is None or bv is None:
                    unavailable = {
                        "reason": "paired comparison unavailable because macro-AUROC is undefined",
                        "metric": metric,
                    }
                    break
                vals[seed] = float(av - bv)
            metric_out[label] = unavailable if unavailable is not None else paired_summary(vals)
        comparisons[metric] = metric_out

    all_undefined = sorted(set(
        root
        for rid in run_ids
        for root in per_run[rid]["AUROC_undefined_scenarios"]
    ))
    return {
        "schema": "COGNIX-TEST-phase-B-frozen-scoring-results-v1",
        "status": "PHASE_B_TEST_SCORING_COMPLETE",
        "scenario_count": EXPECTED_SCENARIOS,
        "prediction_rows": EXPECTED_PREDICTION_ROWS,
        "run_count": EXPECTED_RUNS,
        "macro_AUROC_fail_closed": bool(all_undefined),
        "macro_AUROC_undefined_scenarios_union": all_undefined,
        "per_run": per_run,
        "paired_comparisons": comparisons,
        "threshold_selection_on_TEST": False,
        "model_execution": False,
        "training_executed": False,
        "optimizer_executed": False,
        "recalibration_executed": False,
        "representation_fitting_executed": False,
        "checkpoint_selection_executed": False,
        "result_dependent_retry": False,
    }


def parse_label_feather(raw: bytes, scenario_root: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_feather(io.BytesIO(raw))
    require("tick" in df.columns and "anomaly" in df.columns,
            f"{scenario_root}: required label columns tick/anomaly missing")
    ticks = df["tick"].to_numpy(dtype=np.int64)
    require(len(ticks) == EXPECTED_TICKS, f"{scenario_root}: expected 300 label rows")
    require(np.array_equal(ticks, np.arange(EXPECTED_TICKS, dtype=np.int64)),
            f"{scenario_root}: label ticks must be exactly 0..299")
    lab_raw = df["anomaly"].to_numpy()
    if lab_raw.dtype == bool:
        labels = lab_raw.astype(np.int64)
    else:
        labels = lab_raw.astype(np.int64)
        require(np.all((labels == 0) | (labels == 1)),
                f"{scenario_root}: anomaly must be boolean or 0/1")
    return ticks, labels


def collect_labels_from_archive(phase_a: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    require(ARCHIVE.is_file(), f"TEST archive missing: {ARCHIVE}")
    require(ARCHIVE.stat().st_size == ARCHIVE_BYTES, "TEST archive byte count mismatch")

    wanted = {
        root.rstrip("/") + "/anomaly-observation.feather": idx
        for idx, root in enumerate(phase_a["scenario_roots"])
    }
    require(len(wanted) == EXPECTED_SCENARIOS, "Wanted label member set mismatch")

    labels_by_scenario: list[np.ndarray | None] = [None] * EXPECTED_SCENARIOS
    members_seen = 0
    label_payload_opens = 0
    duplicate_label_members: list[str] = []
    unknown_label_members: list[str] = []
    forbidden_payload_opens = 0
    started = time.time()
    last_report = started

    with tarfile.open(ARCHIVE, mode="r|gz") as tf:
        for member in tf:
            members_seen += 1
            name = member.name.rstrip("/")
            now = time.time()
            if now - last_report >= 30:
                done = sum(x is not None for x in labels_by_scenario)
                print(f"Phase B label stream: {members_seen} members; {done} / {EXPECTED_SCENARIOS} labels read",
                      flush=True)
                last_report = now

            if not member.isfile():
                continue

            if name.endswith("/anomaly-observation.feather"):
                if name not in wanted:
                    unknown_label_members.append(name)
                    continue
                idx = wanted[name]
                if labels_by_scenario[idx] is not None:
                    duplicate_label_members.append(name)
                    continue
                f = tf.extractfile(member)
                require(f is not None, f"Could not open label payload: {name}")
                raw = f.read()
                label_payload_opens += 1
                _, labels = parse_label_feather(raw, phase_a["scenario_roots"][idx])
                labels_by_scenario[idx] = labels
                continue

            # Every other file is metadata-skipped only; extractfile is never called.
            # In particular anomaly-front/* remains unopened.
            continue

    require(not duplicate_label_members, f"Duplicate label members: {duplicate_label_members[:3]}")
    require(not unknown_label_members, f"Unknown label members: {unknown_label_members[:3]}")
    require(label_payload_opens == EXPECTED_SCENARIOS,
            f"Expected 627 label payload opens, got {label_payload_opens}")
    require(all(x is not None for x in labels_by_scenario), "Missing one or more scenario label tables")
    require(forbidden_payload_opens == 0, "Forbidden payload was opened")

    target = np.empty(EXPECTED_PREDICTION_ROWS, dtype=np.int8)
    si = phase_a["scenario_index"]
    ticks = phase_a["tick"]
    for idx in range(EXPECTED_SCENARIOS):
        labels = labels_by_scenario[idx]
        assert labels is not None
        m = si == idx
        scenario_ticks = ticks[m].astype(np.int64)
        target[m] = labels[scenario_ticks].astype(np.int8)

    audit = {
        "schema": "COGNIX-TEST-phase-B-label-access-audit-v1",
        "archive_path": str(ARCHIVE),
        "archive_bytes": ARCHIVE_BYTES,
        "archive_sha256_bound": ARCHIVE_SHA256,
        "members_seen": members_seen,
        "scenario_label_payloads_opened": label_payload_opens,
        "forbidden_payload_opens": forbidden_payload_opens,
        "anomaly_front_opened": False,
        "other_sensor_payloads_opened": False,
        "labels_validated_ticks_0_through_299": True,
        "labels_scored_ticks_1_through_299": True,
        "elapsed_seconds": float(time.time() - started),
    }
    return target, audit


def seal_output_dir(path: Path, payload_names: list[str]) -> str:
    lines: list[str] = []
    for rel in sorted(payload_names):
        fp = path / rel
        require(fp.is_file(), f"Missing output payload before seal: {fp}")
        lines.append(f"{sha256_file(fp)}  {rel}")
    sums = "\n".join(lines) + "\n"
    (path / "SHA256SUMS").write_text(sums, encoding="utf-8")
    seal = hashlib.sha256((path / "SHA256SUMS").read_bytes()).hexdigest()
    (path / "SHA256SUMS.sha256").write_text(seal + "  SHA256SUMS\n", encoding="utf-8")
    return seal


def self_test() -> None:
    # No archive access in this function.
    q = np.array([0.9, 0.8, 0.3, 0.2], dtype=np.float64)
    yn = np.array([1, 1, 0, 0], dtype=np.int64)
    pc = 1.0 - q
    yc = 1 - yn
    require(np.isfinite(bce_samples(q, yn)).all(), "Synthetic BCE failed")
    require(abs(corruption_f1(yc, pc >= 0.5) - 1.0) < 1e-15, "Synthetic F1 failed")
    auc = auroc_corruption(pc, yc)
    require(auc is not None and abs(auc - 1.0) < 1e-15, "Synthetic AUROC failed")
    require(auroc_corruption(pc[:2], np.array([0, 0], dtype=np.int64)) is None,
            "Single-class AUROC fail-closed self-test failed")

    buf = io.BytesIO()
    pd.DataFrame({
        "tick": np.arange(300, dtype=np.int64),
        "anomaly": np.r_[np.zeros(150, dtype=bool), np.ones(150, dtype=bool)],
    }).to_feather(buf)
    ticks, labels = parse_label_feather(buf.getvalue(), "synthetic/scenario")
    require(np.array_equal(ticks, np.arange(300)), "Synthetic feather ticks failed")
    require(labels.shape == (300,) and labels.sum() == 150, "Synthetic feather labels failed")

    print(json.dumps({
        "status": "SELF_TEST_PASS_NO_TEST_LABEL_ACCESS",
        "test_archive_opened": False,
        "test_label_payload_opened": False,
        "metrics_on_TEST_computed": False,
        "synthetic_BCE_finite": True,
        "synthetic_F1": 1.0,
        "synthetic_AUROC": 1.0,
        "synthetic_single_class_AUROC": None,
        "synthetic_label_rows": 300,
    }, sort_keys=True))


def execute_phase_b(authorization: str | None) -> None:
    require(authorization == AUTHORIZATION, "Phase-B authorization token mismatch")
    require(PHASE_B_PREP.is_dir(), "Phase-B preparation staging bundle missing")
    require((PHASE_B_PREP / "phase_B_design.json").is_file(), "phase_B_design.json missing")
    require((PHASE_B_PREP / "phase_B_metric_contract.json").is_file(),
            "phase_B_metric_contract.json missing")
    require(not PHASE_B_OUTPUT_STAGING.exists(), "Phase-B results staging already exists")
    require(not PHASE_B_OUTPUT_FINAL.exists(), "Phase-B final results already exists")

    verify_sealed_dir(PROTOCOL_DIR, PROTOCOL_SEAL)
    phase_a = load_phase_a_inputs()

    design = json.loads((PHASE_B_PREP / "phase_B_design.json").read_text(encoding="utf-8"))
    contract = json.loads((PHASE_B_PREP / "phase_B_metric_contract.json").read_text(encoding="utf-8"))
    require(design.get("current_state", {}).get("label_payload_opened") is False,
            "Design does not certify pre-label state")
    require(design.get("current_state", {}).get("metrics_computed") is False,
            "Design does not certify pre-metric state")
    require(contract.get("threshold_policy", {}).get("TEST_threshold_selection_allowed") is False,
            "Metric contract permits TEST threshold selection")

    PHASE_B_OUTPUT_STAGING.mkdir(parents=True)
    start = {
        "schema": "COGNIX-TEST-phase-B-execution-start-v1",
        "status": "STARTED_PHASE_B_TEST_LABEL_ACCESS",
        "authorization": AUTHORIZATION,
        "phase_A_prediction_bundle_seal": PHASE_A_SEAL,
        "test_protocol_seal": PROTOCOL_SEAL,
        "archive_path": str(ARCHIVE),
        "archive_bytes": ARCHIVE_BYTES,
        "archive_sha256_bound": ARCHIVE_SHA256,
        "threshold_selection_on_TEST": False,
        "model_execution": False,
    }
    (PHASE_B_OUTPUT_STAGING / "execution_start.json").write_text(
        json.dumps(start, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    try:
        target_corruption, access_audit = collect_labels_from_archive(phase_a)
        target_normal = (1 - target_corruption).astype(np.int8)

        np.savez_compressed(
            PHASE_B_OUTPUT_STAGING / "labels_scored_ticks_1_299.npz",
            scenario_index=phase_a["scenario_index"].astype(np.int32),
            tick=phase_a["tick"].astype(np.int16),
            target_corruption=target_corruption.astype(np.int8),
            target_normal=target_normal.astype(np.int8),
        )

        results = score_all(phase_a, target_corruption)
        (PHASE_B_OUTPUT_STAGING / "label_access_audit.json").write_text(
            json.dumps(access_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (PHASE_B_OUTPUT_STAGING / "metrics.json").write_text(
            json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )

        summary = {
            "schema": "COGNIX-TEST-phase-B-completion-v1",
            "status": "PHASE_B_TEST_SCORING_COMPLETE",
            "scenario_count": EXPECTED_SCENARIOS,
            "prediction_rows": EXPECTED_PREDICTION_ROWS,
            "run_count": EXPECTED_RUNS,
            "test_label_payloads_opened": access_audit["scenario_label_payloads_opened"],
            "anomaly_front_opened": False,
            "other_sensor_payloads_opened": False,
            "threshold_selection_on_TEST": False,
            "model_execution": False,
            "training_executed": False,
            "recalibration_executed": False,
            "checkpoint_selection_executed": False,
            "macro_AUROC_fail_closed": results["macro_AUROC_fail_closed"],
            "macro_AUROC_undefined_scenario_count":
                len(results["macro_AUROC_undefined_scenarios_union"]),
        }
        (PHASE_B_OUTPUT_STAGING / "manifest.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        payloads = [
            "execution_start.json",
            "label_access_audit.json",
            "labels_scored_ticks_1_299.npz",
            "manifest.json",
            "metrics.json",
        ]
        seal = seal_output_dir(PHASE_B_OUTPUT_STAGING, payloads)

        os.replace(PHASE_B_OUTPUT_STAGING, PHASE_B_OUTPUT_FINAL)
        print(json.dumps({
            "status": "PHASE_B_TEST_SCORING_COMPLETE",
            "output_path": str(PHASE_B_OUTPUT_FINAL),
            "output_seal": seal,
            "scenario_count": EXPECTED_SCENARIOS,
            "prediction_rows": EXPECTED_PREDICTION_ROWS,
            "run_count": EXPECTED_RUNS,
            "test_label_payloads_opened": EXPECTED_SCENARIOS,
            "anomaly_front_opened": False,
            "other_sensor_payloads_opened": False,
            "threshold_selection_on_TEST": False,
            "macro_AUROC_fail_closed": results["macro_AUROC_fail_closed"],
            "macro_AUROC_undefined_scenario_count":
                len(results["macro_AUROC_undefined_scenarios_union"]),
        }, sort_keys=True))

    except Exception as exc:
        failure = {
            "schema": "COGNIX-TEST-phase-B-failure-v1",
            "status": "FAILED_PRESERVED_PHASE_B_TEST_SCORING",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "retry_requires_review": True,
            "result_dependent_retry": False,
        }
        try:
            (PHASE_B_OUTPUT_STAGING / "failure.json").write_text(
                json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        finally:
            raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--execute-phase-b", action="store_true")
    parser.add_argument("--authorization")
    args = parser.parse_args()

    require(args.self_test != args.execute_phase_b,
            "Choose exactly one of --self-test or --execute-phase-b")
    if args.self_test:
        self_test()
    else:
        execute_phase_b(args.authorization)


if __name__ == "__main__":
    main()
