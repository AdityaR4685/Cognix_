"""Write the milestone report from completed export/test/integrity evidence."""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from cognix.adapters.carla import graph_fit_export as ge


def main():
    artifact = Path(os.environ["TEMP"]) / "carla_graph_fit_export_v1"
    m = json.loads((artifact / "manifest.json").read_text())
    c = m["counts"]
    schema = json.loads((artifact / "schema.json").read_text())
    integrity = json.loads((OUT / "integrity_results.json").read_text())
    ge.require(integrity["status"] == "passed", "integrity audit incomplete")
    focused = (OUT / "focused_tests.txt").read_text()
    broad = (OUT / "broad_tests.txt").read_text()
    ge.require(re.search(r"43 passed", focused) and re.search(r"548 passed", broad)
               and not re.search(r"\d+ failed|\d+ errors", broad), "requested tests have not passed")
    initial = json.loads((OUT / "initial_audit.json").read_text())
    old = set(initial["tracked_file_hashes"]) | set(initial["intentional_untracked_files"])
    now = set(subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, text=True).splitlines())
    new = sorted((now - old) | {"reports/carla_graph_fit_export_v1/report.md", "reports/carla_graph_fit_export_v1/file_inventory.json"})
    inventory = {"changed_preexisting_files": integrity["changed_preexisting_files"],
        "preserved_existing_untracked_inventory": "initial_audit.json::intentional_untracked_files",
        "new_repository_files": new,
        "new_compact_artifact_files": [str(p) for p in sorted(artifact.iterdir()) if p.is_file()],
        "new_checkpoint_files": [str(p) for p in sorted(artifact.with_name(artifact.name + "_work").iterdir()) if p.is_file()],
        "runtime_caches": "ignored Python/pytest caches are incidental and do not enter the scientific artifact"}
    ge.save_json(OUT / "file_inventory.json", inventory)
    lines = []
    def add(s=""):
        lines.append(s)
    def heading(title):
        add("## " + title); add()
    add("# Frozen FIT Graph Compact Export v1"); add()
    heading("Frozen Protocol Verification")
    add(f"Protocol SHA-256: `{ge.PROTOCOL_SHA256}`. All sealed preregistration files and checksum-list seal verified unchanged before editing and after export. Existing N20 cache/handoff/source hashes match the sealed bindings. The 117 pre-existing untracked files and 193 tracked files remain byte-identical; all 20 frozen raw scenario directories remain available."); add()
    heading("Exporter Implementation")
    add("Added only `cognix/adapters/carla/graph_fit_export.py`, `tests/unit/test_carla_graph_fit_export.py`, and this report directory. Production entry point: `run_export.py --workers 3`. Three bounded local processes performed one full corrupted-image pass. Per-scenario compact checkpoints allow restart with identical dependency and code bindings. Existing outputs are never overwritten. No upstream fitting, training, acquisition, calibration changes or generic-framework edits."); add()
    heading("Clean Graph Construction")
    add("Every frozen FIT parent tick 1..2999 was considered. Accepted clean graphs copy the sealed N20 clean checkpoint in canonical node order Camera/IMU/Seg (handoff indices 0/3/1), with features [prob_normal, epistemic, aleatoric]. All retained clean node blocks were compared byte for byte with N20. The single schema-bound adjacency is [[0,1,1],[1,0,1],[1,1,0]], with no self-loops or spatial topology. Clean target_normal=1."); add()
    heading("FIT Pseudo Generation")
    add("Used unchanged `_pseudo_for_scenario(raw, cached_parent_features, scenario_id, frozen_five_recipe_tuple, 0, 1, None)`. Rotation is recipes[t % 5]; base seed 0, call seed t, existing defaults and causal window 12. Only the 15 sealed FIT scenarios were passed to raw/pseudo generation; no CAL pseudo arrays were reused. IMU corruption consumes [max(0,t-11),t]; Camera/Seg corruption consumes tick t. Pseudo target_normal=0. GNSS contributes neither primary nodes nor recipes nor calibrated exported quantities."); add()
    heading("Untouched-Node Integrity")
    add("Restored five production Mahalanobis bootstrap members and the accepted shared calibrator for each primary modality from hash-verified state, with no `.fit` call. The affected compact block passes through the unchanged production agent normality scores, member calibration and canonical Bernoulli UQ. Untouched compact blocks are verified byte-identical to cached parents; untouched p/E/A nodes are copied directly and verified byte-identical for every pair. Focused spies prove five member-score and five calibration calls per accepted pseudo, only for its affected modality. Clean inference anchors and a ten-parent real FIT recipe replay across train/validation agree with the export."); add()
    heading("No-Effect / Pairing Results")
    add(f"The unchanged compact-feature np.allclose(rtol=1e-12,atol=1e-12) guard skipped {c['skipped_parents']:,} parents, removing both pair members without replacements. Skip decisions precede node inference and do not depend on predictions. {c['node_output_collisions']:,} material-feature corruptions had identical affected-node p/E/A; these remain in the dataset and collision ledger. Every retained pair has exactly one clean target 1 and one pseudo target 0, a stable protocol-bound SHA256 pair ID, the same parent scenario/tick/split, and one affected modality."); add()
    heading("Graph Dataset Counts")
    add("| Quantity | Count |"); add("|---|---:|")
    for key in ("candidate_parents", "retained_parents", "skipped_parents", "clean_rows", "pseudo_rows", "graph_rows", "node_output_collisions"):
        add(f"| {key} | {c[key]:,} |")
    for split in ("GRAPH_TRAIN", "GRAPH_VALIDATION"):
        add(f"| {split} rows | {c['per_split'][split]['graph_rows']:,} |")
    add(); add("Preregistered row maxima are 71,976 train and 17,994 validation. Realized counts follow only eligibility and the frozen guard; maxima are never forced."); add()
    heading("Per-Recipe / Per-Scenario Counts")
    add("Counts below are retained pseudo graphs (one matching clean graph each)."); add()
    add("| Recipe | Train | Validation | Total |"); add("|---|---:|---:|---:|")
    for recipe in ge.RECIPES:
        add(f"| {recipe} | {c['per_split']['GRAPH_TRAIN']['per_recipe'].get(recipe,0):,} | {c['per_split']['GRAPH_VALIDATION']['per_recipe'].get(recipe,0):,} | {c['per_recipe'].get(recipe,0):,} |")
    add(); add("| Modality | Pseudo graphs |"); add("|---|---:|")
    for name in ge.NODE_ORDER:
        add(f"| {name} | {c['per_modality'].get(name,0):,} |")
    add(); add("| Scenario | Split | Candidates | Retained pairs | Skipped | Clean | Pseudo | Rows |"); add("|---|---|---:|---:|---:|---:|---:|---:|")
    role = {d['scenario_id']:d['graph_split'] for d in schema['scenario_dictionary']}
    for sid, v in c['per_scenario'].items():
        add(f"| {sid} | {role[sid]} | {v['candidate_parents']:,} | {v['retained_parents']:,} | {v['skipped_parents']:,} | {v['clean_rows']:,} | {v['pseudo_rows']:,} | {v['graph_rows']:,} |")
    add(); add("`counts.json` also records per-recipe/per-modality counts within each scenario and split."); add()
    heading("Compact Artifact Schema")
    add(f"Artifact directory: `{artifact}`. `graphs.npz` contains no pickle/object arrays, raw frames, archive bytes, GNSS calibrated values, model parameters or raw feature blocks. Provenance common to all rows and dictionaries live in `schema.json`; graph row fields reconstruct using array indices and parent scenario/tick. Scientific storage remains float64. The sealed per-graph hash retains its specified float32 model-view bytes plus float64 canonical epistemic side channel; the separate artifact content hash covers the full float64 storage."); add()
    add("| Array | Dtype | Shape |"); add("|---|---|---|")
    for key, v in schema['arrays'].items():
        add(f"| {key} | {v['dtype']} | {v['shape']} |")
    add(); add("Clean recipe/seed/node indices and actual corruption bounds use -1; clean severity uses NaN. Reconstructed scientific recipe/severity/seed/modality are null. Window bounds always describe parent causal provenance. For the sealed graph hash, clean row provenance carries the deterministically selected recipe scope in corruption_start_tick/corruption_end_tick (the same scope as its paired pseudo); actual compact clean corruption bounds remain -1. `validate_export.py` reconstructs and checks every row hash from these rules."); add()
    heading("Artifact Hashes")
    add(f"Scientific artifact content SHA-256: `{m['artifact_content_sha256']}`."); add()
    add(f"Schema file SHA-256: `{m['schema_sha256']}`."); add()
    add(f"Manifest file SHA-256: `{ge.sha256_file(artifact / 'manifest.json')}`."); add()
    add("| Artifact file | SHA-256 |"); add("|---|---|")
    for name, record in m['files'].items():
        add(f"| {name} | `{record['sha256']}` |")
    add(); add(f"N20 cache scientific content SHA-256: `{integrity['cache_content_sha256']}`. Complete upstream dependency hashes are in `upstream_dependencies.json`; sealed protocol identities are embedded in schema common provenance."); add()
    heading("Focused Tests")
    add("43 passed, zero failures. Covers every requested exporter integrity category, using small local fixtures, deterministic recipe replay, fit-blocking spies, invalid/null mapping rejection, exact untouched nodes, collision retention, causal scope and serialization/content hash checks. Evidence: `focused_tests.txt`."); add()
    heading("Broad Regression")
    add("Ran the requested command with PYTHONHASHSEED=0: `py -3.12 -m pytest tests/unit tests/regression tests/integration tests/test_interfaces.py tests/test_mathematics.py -q`. Result: 548 passed (505 accepted baseline + 43 exporter tests), zero failures. Evidence: `broad_tests.txt`."); add()
    heading("Core / N=20 Integrity")
    add("All 193 existing tracked files and 117 pre-existing untracked files were preserved byte for byte. Sealed preregistration, N20 cache/handoff, calibration, feature, recipe, generic graph/core and frozen synthetic sources remain unchanged. FIT whitelist gates precede raw reads; official TRAIN raw paths only. No TEST, CAL graph rows, training, conformal, acquisition, scientific tuning, commits or pushes. Full file inventory is in `file_inventory.json`; original intentional untracked inventory is in `initial_audit.json`. Generated compact/checkpoint files are included explicitly."); add()
    add("New repository files:"); add()
    for path in new:
        add("- `" + path + "`")
    add()
    heading("Ready / Not Ready for Trainer Implementation")
    add("READY for the next separately authorized trainer implementation milestone: exported FIT data and integrity gates pass."); add()
    heading("Ready / Not Ready for Kaggle Training")
    add("NOT READY: no graph trainer, minibatching/early-stopping implementation or locked Kaggle execution environment has been implemented or validated. No upload or execution was performed."); add()
    heading("Recommended Next Step")
    add("Authorize only the frozen adapter/report-side trainers and their integrity tests against this validated artifact. Keep TEST and conformal outside that milestone, and do not execute Kaggle training before trainer validation and separate execution authorization.")
    with (OUT / "report.md").open("x", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps({"report": str(OUT / 'report.md'), "new_repository_files": len(new),
                      "artifact_content_sha256": m['artifact_content_sha256']}))


if __name__ == "__main__":
    main()
