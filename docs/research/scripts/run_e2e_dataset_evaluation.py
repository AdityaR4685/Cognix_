"""
Command-Line Runner for COGNIX End-to-End Dataset Evaluation.

Executes reproducible evaluation of Cognix across recorded CARLA TEST scenarios
and synthetic perturbation baselines, saving structured evidence and a markdown report.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from cognix.evaluation.e2e_dataset_eval import (
    evaluate_recorded_carlanomaly,
    evaluate_synthetic_demonstration,
    aggregate_carlanomaly_metrics,
    build_evaluation_report,
    get_git_commit
)


DEFAULT_SCENARIOS = [
    "test/anomaly/Town01/change-weather/scenario-1",
    "test/anomaly/Town01/change-weather/scenario-10",
    "test/anomaly/Town01/running-pedestrian/scenario-1",
    "test/anomaly/Town01/spawn-props/scenario-1",
]


def main():
    parser = argparse.ArgumentParser(description="COGNIX End-to-End Dataset Evaluation")
    parser.add_argument("--archive", default="E:/carlanomaly-base-test.tar.gz",
                        help="Path to CarlAnomaly Base TEST tar.gz archive")
    parser.add_argument("--output-dir", default=str(ROOT / "reports/carla_runtime_v1/evaluation_runs/e2e_dataset_evaluation_v1"),
                        help="Directory for evaluation artifacts")
    parser.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS,
                        help="List of scenarios to evaluate")
    parser.add_argument("--methods", nargs="+", default=["NoGraph", "StandardGAT", "EpistemicGAT"],
                        choices=["NoGraph", "StandardGAT", "EpistemicGAT"],
                        help="Graph refinement methods to evaluate")
    parser.add_argument("--seed", type=int, default=101, choices=[101, 202, 303, 404, 505],
                        help="Frozen demonstration seed")
    parser.add_argument("--max-samples", type=int, default=299,
                        help="Ticks per scenario to evaluate (default: 299)")
    parser.add_argument("--skip-synthetic", action="store_true",
                        help="Skip synthetic demonstration benchmark")
    args = parser.parse_args()

    archive_path = Path(args.archive).resolve()
    if not archive_path.is_file():
        print(f"ERROR: Archive not found at {archive_path}")
        sys.exit(1)

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("COGNIX End-to-End Dataset Evaluation")
    print(f"Archive:       {archive_path}")
    print(f"Scenarios:     {len(args.scenarios)} scenarios")
    print(f"Methods:       {args.methods}")
    print(f"Seed:          {args.seed}")
    print(f"Ticks/Scen:    {args.max_samples}")
    print(f"Output Dir:    {out_dir}")
    print("=" * 70)

    # 1. Run Recorded CARLA Simulator Evaluation
    print("\n[Step 1/3] Running recorded CARLA evaluation...")
    eval_results, scenario_provenance = evaluate_recorded_carlanomaly(
        repo_path=ROOT,
        archive_path=archive_path,
        scenarios=args.scenarios,
        methods=args.methods,
        seeds=[args.seed],
        max_samples=args.max_samples
    )

    carlanomaly_agg = aggregate_carlanomaly_metrics(eval_results)

    # 2. Run Synthetic Demonstration Baseline
    synthetic_agg = {}
    if not args.skip_synthetic:
        print("\n[Step 2/3] Running synthetic demonstration baseline...")
        synthetic_agg = evaluate_synthetic_demonstration(seed=args.seed, n_frames=20)
    else:
        print("\n[Step 2/3] Skipping synthetic benchmark as requested.")

    # 3. Assemble Manifest & Save Artifacts
    print("\n[Step 3/3] Assembling reproducible artifacts & report...")
    git_commit = get_git_commit(ROOT)
    git_branch = "gat-testing-complete-checkpoint"

    manifest = {
        "evaluation_id": "COGNIX_E2E_DATASET_EVALUATION_V1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "git_branch": git_branch,
        "archive_path": str(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
        "scenarios": args.scenarios,
        "methods": args.methods,
        "seed": args.seed,
        "max_samples_per_scenario": args.max_samples,
        "total_evaluated_samples": len(args.scenarios) * args.max_samples,
        "scenario_metadata": scenario_provenance,
        "calibration_status": {
            "runtime_calibration_available": False,
            "reason": "No collective conformal artifact is bound to frozen upstream + node refinement + runtime epistemic-weighted fusion",
            "fail_closed_verified": True
        },
        "hardware_environment": {
            "os": os.name,
            "python_version": sys.version,
            "platform": sys.platform
        }
    }

    # Save manifest
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Save aggregate metrics
    metrics_bundle = {
        "manifest": manifest,
        "carlanomaly_results": carlanomaly_agg,
        "synthetic_results": synthetic_agg
    }
    (out_dir / "metrics_summary.json").write_text(json.dumps(metrics_bundle, indent=2), encoding="utf-8")

    # Save synthetic summary
    (out_dir / "synthetic_benchmark_summary.json").write_text(json.dumps(synthetic_agg, indent=2), encoding="utf-8")

    # Save per-tick predictions
    per_tick_data = {}
    for r in eval_results:
        key = f"{r.scenario}__{r.method}_{r.seed}"
        per_tick_data[key] = {
            "scenario": r.scenario,
            "method": r.method,
            "seed": r.seed,
            "bce": r.bce,
            "f1": r.f1_detection,
            "brier": r.brier,
            "ece": r.ece,
            "records": r.per_tick_records
        }
    (out_dir / "per_tick_predictions.json").write_text(json.dumps(per_tick_data, indent=2), encoding="utf-8")

    # Generate Markdown Report
    report_md = build_evaluation_report(manifest, carlanomaly_agg, synthetic_agg)
    (out_dir / "E2E_DATASET_EVALUATION_REPORT.md").write_text(report_md, encoding="utf-8")

    # Also write report to docs/research/ for tracking
    doc_report_path = ROOT / "docs/research/E2E_DATASET_EVALUATION_REPORT.md"
    doc_report_path.write_text(report_md, encoding="utf-8")

    print("\n" + "=" * 70)
    print("EVALUATION RUN COMPLETE")
    print(f"Artifacts saved to: {out_dir}")
    print(f"Report saved to:    {doc_report_path}")
    print("=" * 70)

    # Print summary table
    print("\nSUMMARY METRICS (Recorded CARLA Simulator, Seed 101):")
    print(f"{'Method':<14} | {'Mean BCE':<10} | {'Mean Brier':<10} | {'Mean ECE':<10} | {'Mean F1':<10} | {'Escalate %':<10}")
    print("-" * 75)
    for m in ["NoGraph", "StandardGAT", "EpistemicGAT"]:
        if m in carlanomaly_agg["methods"]:
            sm = carlanomaly_agg["methods"][m]
            print(f"{m:<14} | {sm['mean_bce']:<10.6f} | {sm['mean_brier']:<10.6f} | {sm['mean_ece']:<10.6f} | {sm['mean_f1']:<10.6f} | {sm['mean_escalation_rate']*100:<10.1f}%")

    print("\nPAIRED DIFFERENCES (Mean Delta):")
    paired = carlanomaly_agg["paired_comparisons"]
    if "StandardGAT_vs_NoGraph" in paired:
        p1 = paired["StandardGAT_vs_NoGraph"]
        print(f"StandardGAT - NoGraph:       BCE Delta = {p1['bce']['mean_diff']:+.6f}, F1 Delta = {p1['f1']['mean_diff']:+.6f}, Brier Delta = {p1['brier']['mean_diff']:+.6f}")
    if "EpistemicGAT_vs_StandardGAT" in paired:
        p2 = paired["EpistemicGAT_vs_StandardGAT"]
        print(f"EpistemicGAT - StandardGAT:  BCE Delta = {p2['bce']['mean_diff']:+.6f}, F1 Delta = {p2['f1']['mean_diff']:+.6f}, Brier Delta = {p2['brier']['mean_diff']:+.6f}")


if __name__ == "__main__":
    main()
