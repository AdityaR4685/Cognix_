"""Validate reporting against sealed summaries without executing experiment code.

Supply the pre-edit read-only filesystem baseline with --baseline to also verify
that every experiment artifact and scientific source remains unchanged. No raw
TEST labels/predictions are parsed; raw bytes are read only for integrity hashes.
The output receipt is documentation, not an experiment result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
from plot_experiment_2b import (  # Report-only helper; no scientific runtime imports.
    CONCLUSION_SEAL, CONCLUSION_SHA256, EXPERIMENT, PHASE_B_SEAL, ROOT,
    authenticated_data, sealed_file, sha256,
)

MODIFIED = [
    "CONTINUE_FROM_HERE.md", "README.md", "docs/STATISTICAL_METHODOLOGY.md",
    "docs/architecture/overview.md", "docs/research/ABLATION_PLAN.md",
    "docs/research/COGNIX_FINAL_STATUS.md", "docs/research/METRIC_PROVENANCE.md",
    "docs/research/REPRODUCIBILITY.md",
]
CREATED = [
    "docs/research/EXPERIMENT_2B_PACKAGING_AUDIT.md",
    "docs/research/experiment_2b_packaging_validation.json",
    "docs/research/scripts/plot_experiment_2b.py",
    "docs/research/scripts/validate_experiment_2b_packaging.py",
    "docs/research/figures/experiment_2b/figure_data.json",
    "docs/research/figures/experiment_2b/paired_bce_differences.svg",
    "docs/research/figures/experiment_2b/paired_bce_differences.png",
    "docs/research/figures/experiment_2b/paired_f1_differences.svg",
    "docs/research/figures/experiment_2b/paired_f1_differences.png",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def heading_slug(heading: str) -> str:
    return re.sub(r"[^\w\s-]", "", heading.lower()).replace(" ", "-")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--baseline-sha256", help="Authenticate a rebuilt canonical baseline using its retained pre-edit digest")
    args = parser.parse_args()
    data = authenticated_data()
    require(json.loads((ROOT / "docs/research/figures/experiment_2b/figure_data.json").read_text(encoding="utf-8")) == data,
            "Figure data is not an exact copy of authenticated stored summaries")
    metrics = json.loads(sealed_file(EXPERIMENT / "graph_test_phase_B_results_v1", PHASE_B_SEAL, "metrics.json").read_text(encoding="utf-8"))
    conclusion = json.loads(sealed_file(EXPERIMENT / "graph_test_final_conclusion_bundle_v1", CONCLUSION_SEAL, "final_conclusion.json").read_text(encoding="utf-8"))
    require(len(metrics["per_run"]) == 15, "Frozen run count differs")
    for method, summary in data["seed_mean_metrics"].items():
        for metric in ("BCE", "F1"):
            # Cross-check stored run-level metrics, never calculate metrics from labels.
            stored_mean = math.fsum(metrics["per_run"][f"{method}_{seed}"][metric] for seed in data["seeds"]) / 5
            require(math.isclose(stored_mean, summary[metric], rel_tol=0, abs_tol=1e-15),
                    f"Stored seed mean differs from frozen conclusion: {method}/{metric}")

    undefined = set(metrics["macro_AUROC_undefined_scenarios_union"])
    require(len(undefined) == 107 and all(name.startswith("test/normal/") for name in undefined),
            "Undefined AUROC set is not exactly 107 normal scenarios")
    for run_id, run in metrics["per_run"].items():
        require(run["AUROC"] is None and set(run["AUROC_undefined_scenarios"]) == undefined,
                f"AUROC availability differs: {run_id}")
        scenarios = run["per_scenario"]
        require(len(scenarios) == 627, f"Scenario count differs: {run_id}")
        normal = {name for name in scenarios if name.startswith("test/normal/")}
        anomaly = set(scenarios) - normal
        require(normal == undefined and len(anomaly) == 520, f"Frozen class structure differs: {run_id}")
        require(all(scenarios[name]["normal_count"] == 299 and scenarios[name]["corruption_count"] == 0
                    and scenarios[name]["AUROC"] is None for name in normal), "Normal scenario counts differ")
        require(all(scenarios[name]["normal_count"] > 0 and scenarios[name]["corruption_count"] > 0
                    and scenarios[name]["AUROC"] is not None for name in anomaly), "Anomaly class support differs")
        for count, bounds in (("normal_count", [192, 298]), ("corruption_count", [1, 107])):
            counts = [scenarios[name][count] for name in anomaly]
            require([min(counts), max(counts)] == bounds, f"Stored anomaly counts differ: {run_id}/{count}")
    require(all(value is False for key, value in conclusion["protocol_integrity"].items()
                if key != "negative_or_null_results_retained")
            and conclusion["protocol_integrity"]["negative_or_null_results_retained"] is True,
            "Unexpected frozen protocol integrity flags")

    report_path = ROOT / "docs/research/COGNIX_FINAL_STATUS.md"
    report = report_path.read_text(encoding="utf-8")
    require("| Method | BCE ↓ | Corruption F1 ↑ | Macro-AUROC |" in report, "Metric directions missing")
    for method, summary in data["seed_mean_metrics"].items():
        bce = f"{summary['BCE']:.6f}"
        f1 = f"{summary['F1']:.6f}"
        if method == "StandardGAT":
            bce = f"**{bce}**"
        if method == "NoGraph":
            f1 = f"**{f1}**"
        require(f"| {method} | {bce} | {f1} | Undefined |" in report, f"Performance table differs: {method}")
    for comparison, values in data["paired_effects"].items():
        for metric, summary in values.items():
            fields = [f"{summary['mean']:+.6f}", f"{summary['sample_SD']:.6f}",
                      f"{summary['min']:+.6f}", f"{summary['max']:+.6f}"]
            row = "| " + " | ".join([comparison.replace("_minus_", " - "), metric] + fields) + " |"
            require(row in report, f"Paired table differs: {comparison}/{metric}")
    qualifier = "represents P(target = normal) under the TRAIN-derived constructed clean-vs-pseudo task and must not be interpreted as a probability of physical driving safety."
    require(qualifier in report and "p_corruption = 1 - q_normal" in report, "Probability semantics missing")
    for value in (CONCLUSION_SEAL, CONCLUSION_SHA256, PHASE_B_SEAL):
        require(value in report, "Source identity missing in report")
    require(all(f"### 6.{number} " in report for number in range(1, 10)), "Required results subsection missing")
    require(all(needle in report for needle in (
        "all 107 normal TEST scenarios were single-class", "no inferential significance test was prespecified or performed",
        "all 76 FIT scenarios before the 15 GRAPH_VAL scenarios were separated", "transferred unchanged to TEST",
        "rather than confirmed causes", "no descriptive evidence of improvement over StandardGAT")), "Required claim or limitation missing")

    link_count = 0
    for name in MODIFIED + [CREATED[0]]:
        path = ROOT / name
        text = path.read_text(encoding="utf-8")
        for link in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text):
            if not link or "://" in link:
                continue
            target_name, _, fragment = link.partition("#")
            target = (path.parent / target_name).resolve() if target_name else path
            require(target.exists(), f"Broken local link: {name}: {link}")
            if fragment and target.suffix == ".md":
                headings = re.findall(r"^#+\s+(.+)$", target.read_text(encoding="utf-8"), flags=re.M)
                require(fragment in {heading_slug(heading) for heading in headings}, f"Broken heading link: {name}: {link}")
            link_count += 1

    for metric in ("bce", "f1"):
        svg = ROOT / f"docs/research/figures/experiment_2b/paired_{metric}_differences.svg"
        text = svg.read_text(encoding="utf-8")
        require(all(str(seed) in text for seed in data["seeds"]), f"Seed label missing in {svg}")
        require("StandardGAT" in text and "EpistemicGAT" in text, f"Legend missing in {svg}")
        require('width="252pt" height="201.6pt"' in text, f"Publication dimensions differ in {svg}")
        require(svg.with_suffix(".png").exists(), f"PNG missing for {svg}")

    preservation = {"status": "not_checked_without_pre_edit_baseline"}
    protected_roots = ["reports", "cognix", "experiments", "configs", "tests", "examples", "artifacts"]
    if args.baseline:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        require(Path(baseline["root"]).resolve() == ROOT, "Baseline belongs to another repository")
        current_paths = {p.relative_to(ROOT).as_posix() for folder in baseline["protected_roots"]
                         for p in (ROOT / folder).rglob("*") if p.is_file()}
        require(current_paths == set(baseline["files"]), "Protected file inventory changed")
        for name, expected in baseline["files"].items():
            path = ROOT / name
            stat = path.stat()
            require(stat.st_size == expected["bytes"] and stat.st_mtime_ns == expected["mtime_ns"]
                    and sha256(path) == expected["sha256"], f"Protected file changed: {name}")
        preservation = {
            "status": "all_protected_files_unchanged",
            "protected_roots": baseline["protected_roots"],
            "file_count": len(baseline["files"]),
            "byte_count": sum(entry["bytes"] for entry in baseline["files"].values()),
            "baseline_sha256": sha256(args.baseline),
            "content_sizes_and_modification_times_unchanged": True,
            "historical_failed_and_superseded_evidence_preserved": True,
        }
    elif args.baseline_sha256:
        # The pre-edit temporary baseline may expire with its sandbox. Its digest
        # authenticates the identical canonical record, including every file's
        # path, content hash, size and modification time, without a stored copy.
        paths = sorted(p for folder in protected_roots for p in (ROOT / folder).rglob("*") if p.is_file())
        records = {}
        for path in paths:
            content_sha256 = sha256(path)
            records[path.relative_to(ROOT).as_posix()] = {
                "sha256": content_sha256, "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns,
            }
        rebuilt = {"root": str(ROOT), "protected_roots": protected_roots, "files": records}
        rebuilt_sha256 = hashlib.sha256(json.dumps(rebuilt, sort_keys=True).encode("utf-8")).hexdigest()
        require(rebuilt_sha256 == args.baseline_sha256,
                f"Protected inventory differs from pre-edit baseline: {rebuilt_sha256}")
        preservation = {
            "status": "all_protected_files_unchanged", "protected_roots": protected_roots,
            "file_count": len(records), "byte_count": sum(record["bytes"] for record in records.values()),
            "baseline_sha256": args.baseline_sha256, "rebuilt_baseline_sha256": rebuilt_sha256,
            "content_sizes_and_modification_times_unchanged": True,
            "historical_failed_and_superseded_evidence_preserved": True,
            "verification_method": "canonical inventory rebuilt byte-identically to authenticated pre-edit digest",
        }

    receipt = {
        "schema": "COGNIX-Experiment-2B-report-packaging-validation-v1",
        "status": "PASS", "source_identities": data["source"],
        "checks": {"source_seals": True, "stored_seed_means": True, "stored_paired_summaries": True,
                   "report_tables_and_directions": True, "full_precision_figure_data": True,
                   "AUROC_fail_closed_structure": True, "probability_semantics": True,
                   "required_sections_and_limitations": True, "local_links": link_count,
                   "SVG_seed_labels_legends_dimensions": True},
        "preservation": preservation,
        "modified_files": MODIFIED, "created_files": CREATED,
        "compilation": {"format": "Markdown", "status": "not_applicable_no_LaTeX_source",
                        "latexmk_available": False, "pdflatex_available": False},
        "execution": {"TEST_rerun": False, "Phase_A_or_B_rerun": False, "training": False,
                      "threshold_tuning": False, "recalibration": False, "checkpoint_reselection": False,
                      "TEST_metrics_recomputed_from_raw_labels": False,
                      "figures_use_only_authenticated_frozen_numbers": True},
        "report_file_sha256": {name: sha256(ROOT / name) for name in MODIFIED + CREATED if name != CREATED[1]},
    }
    output = ROOT / CREATED[1]
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "receipt": str(output), "preservation": preservation,
                      "local_links_checked": link_count}, indent=2))


if __name__ == "__main__":
    main()
