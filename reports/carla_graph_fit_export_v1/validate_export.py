"""Read-only artifact audit, representative real-input replay and preservation."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import numpy as np
from cognix.adapters.carla import graph_fit_export as ge
from cognix.adapters.carla import cache_builder as cb


def main():
    artifact = Path(os.environ["TEMP"]) / "carla_graph_fit_export_v1"
    p, splits, dependencies = ge.verify_frozen_dependencies(ROOT)
    schema = json.loads((artifact / "schema.json").read_text())
    manifest = json.loads((artifact / "manifest.json").read_text())
    counts = json.loads((artifact / "counts.json").read_text())
    ledger = json.loads((artifact / "skip_ledger.json").read_text())
    collisions = json.loads((artifact / "node_output_collisions.json").read_text())
    for name, record in manifest["files"].items():
        ge.require(ge.sha256_file(artifact / name) == record["sha256"], "artifact file hash: " + name)
    with np.load(artifact / "graphs.npz", allow_pickle=False) as z:
        a = dict(z)
    expected_keys = {"node_features", "target_normal", "target_normal_per_node", "is_pseudo", "split",
        "scenario_index", "town_index", "tick", "pair_id", "row_key", "parent_scenario_index", "parent_tick",
        "corrupted_node_index", "recipe_index", "severity", "recipe_seed", "window_start_tick", "window_end_tick",
        "corruption_start_tick", "corruption_end_tick", "graph_content_sha256", "calibration_valid"}
    ge.require(set(a) == expected_keys and not any(v.dtype.kind == "O" for v in a.values()),
               "unexpected raw/model/GNSS payload in compact export")
    for name, arr in a.items():
        ge.require(schema["arrays"][name] == {"dtype": str(arr.dtype), "shape": list(arr.shape)},
                   "actual array disagrees with compact schema")
    ge.validate_artifact(a, schema)
    ge.require(ge.scientific_content_hash(a, schema) == manifest["artifact_content_sha256"], "scientific content hash")
    ge.require([s["scenario_id"] for s in schema["scenario_dictionary"]] == splits["sorted_FIT_ids"], "exact sealed FIT dictionary")
    ge.require(counts["candidate_parents"] == 44985, "every frozen parent candidate")
    ge.require(counts["clean_rows"] == counts["pseudo_rows"] == len(a["tick"]) // 2, "exact pairing")
    ge.require(counts["per_split"]["GRAPH_TRAIN"]["graph_rows"] <= 71976 and counts["per_split"]["GRAPH_VALIDATION"]["graph_rows"] <= 17994, "preregistered maxima")
    b = p["bindings"]
    parameter_path = next(Path(s) for s in b["all_cache_handoff_file_sha256"] if s.endswith("fitted_oneclass_parameters.npz"))
    cache_path = next(Path(s) for s in b["all_cache_handoff_file_sha256"] if s.endswith("carla_train_cache.npz"))
    hm = json.loads((parameter_path.parent / "manifest.json").read_text())
    with np.load(parameter_path.parent / "compact_training_checkpoint.npz", allow_pickle=False) as z:
        fit = (z["partition"] == 0) & ~z["is_pseudo"]
        clean = np.stack([z[k][fit][:, [0, 3, 1]] for k in ge.FEATURE_ORDER], axis=-1)
        h_ticks = z["tick"][fit]
        h_sids = np.array([hm["scenario_dictionary"][int(i)]["scenario_id"] for i in z["scenario_index"][fit]])
    h_index = {(str(s), int(t)): i for i, (s, t) in enumerate(zip(h_sids, h_ticks))}
    dictionary = schema["scenario_dictionary"]
    row_index = {}
    for i in range(0, len(a["tick"]), 2):
        sid = dictionary[int(a["scenario_index"][i])]["scenario_id"]
        t = int(a["tick"][i])
        ge.frozen_split(sid, splits)
        ge.require(a["node_features"][i].tobytes() == clean[h_index[sid, t]].tobytes(), "clean node changed from N20")
        row_index[sid, t] = i
    missing = {(s, t) for s in splits["sorted_FIT_ids"] for t in range(1, 3000)} - set(row_index)
    ge.require(missing == {(s["scenario_id"], s["tick"]) for s in ledger}
               and len(missing) == len(ledger), "all omitted parents have exactly one skip")
    for record in ledger:
        ge.require(record["reason"] == "no_effect" and record["recipe_id"] == ge.selected_recipe(record["tick"]), "fixed no-effect ledger")
    observed_collisions = set()
    for i in range(0, len(a["tick"]), 2):
        j = int(a["corrupted_node_index"][i + 1])
        if np.array_equal(a["node_features"][i, j], a["node_features"][i + 1, j]):
            observed_collisions.add(str(a["pair_id"][i]))
    ge.require(observed_collisions == {r["pair_id"] for r in collisions}
               and len(observed_collisions) == len(collisions), "collision ledger exact and retained")
    # Reconstruct every sealed model-view graph hash from compact arrays/schema.
    for i, node in enumerate(a["node_features"]):
        sid_entry = dictionary[int(a["scenario_index"][i])]
        t = int(a["tick"][i]); pseudo = bool(a["is_pseudo"][i])
        selected = ge.selected_recipe(t)
        r = {"scenario_id": sid_entry["scenario_id"], "town": sid_entry["town"],
            "parent_tick": t, "graph_split": schema["split_dictionary"][str(int(a["split"][i]))],
            "is_pseudo": pseudo, "target_normal": int(a["target_normal"][i]), "pair_key": str(a["pair_id"][i]),
            "recipe_id": selected if pseudo else None,
            "severity": float(a["severity"][i]) if pseudo else None,
            "generator_seed": int(a["recipe_seed"][i]) if pseudo else None,
            "corrupted_agent": ge.NODE_ORDER[int(a["corrupted_node_index"][i])] if pseudo else None,
            "window_start_tick": max(0, t - 11), "window_end_tick": t,
            # Build rows record the selected recipe's scope in hash provenance for
            # both pair members; actual compact clean corruption-bound fields are -1.
            "corruption_start_tick": max(0, t - 11) if selected.startswith("imu") else t,
            "corruption_end_tick": t, **schema["common_provenance"],
            "source_split": "train", "N20_partition": "FIT_NORMAL", "generator_base_seed": 0,
            "node_order": list(ge.NODE_ORDER), "source_hash_bundle_sha256": sid_entry["source_hash_bundle_sha256"],
            "target_normal_per_node": a["target_normal_per_node"][i].tolist(), "row_key": str(a["row_key"][i])}
        ge.require(ge.graph_content_hash(r, node) == str(a["graph_content_sha256"][i]), "row graph hash reconstruction")
    # Replay five rotated recipes on real FIT inputs from one scenario per split.
    agents = ge.restore_agents(dict(np.load(parameter_path, allow_pickle=False)), hm)
    dataset = json.loads((ROOT / "reports/carla_train_expand20_v1/frozen_dataset_manifest.json").read_text())
    source = {s["scenario_id"]: s for s in dataset["scenarios"]}
    replay_records = []
    with np.load(cache_path, allow_pickle=False) as z:
        sids, ticks = z["scenario_id"], z["tick"]
        for split in ("GRAPH_TRAIN", "GRAPH_VALIDATION"):
            sid = splits[split][0]
            sel = (sids == sid) & (ticks <= 5)
            parents = tuple(z[k][sel] for k in ("camera", "seg", "gnss", "imu"))
            clean_subset = np.array([clean[h_index[sid, t]] for t in range(1, 6)])
            path = Path(source[sid]["actual_path"])
            ge.require("train" in path.parts, "raw path is not official train")
            replay = ge.build_scenario_pairs(sid, np.arange(1, 6), parents, clean_subset,
                {"scenario_path": path, "imu": cb._imu_accel_table(path)}, agents, splits)
            for k in range(0, len(replay["rows"]), 2):
                t = replay["rows"][k]["parent_tick"]
                i = row_index[sid, t]
                ge.require(replay["node_features"][k:k + 2].tobytes() == a["node_features"][i:i + 2].tobytes(), "real FIT replay differs")
            replay_records.append({"scenario_id": sid, "graph_split": split, "candidate_parents": 5,
                                   "replayed_pairs": len(replay["rows"]) // 2, "byte_identical": True})
    audit = json.loads((OUT / "initial_audit.json").read_text())
    changed = []
    for group in ("tracked_file_hashes", "intentional_untracked_files"):
        for rel, h in audit[group].items():
            if not (ROOT / rel).is_file() or ge.sha256_file(ROOT / rel) != h:
                changed.append(rel)
    ge.require(not changed, "pre-existing work changed: " + repr(changed))
    ge.require(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() == audit["git_head"], "git HEAD changed")
    result = {"status": "passed", "frozen_dependencies_verified": True, "all_clean_nodes_byte_identical_to_N20": True,
        "all_untouched_nodes_byte_identical": True, "exact_sealed_FIT_only": True,
        "all_graph_hashes_reconstructed": len(a["tick"]), "all_pairs_validated": len(a["tick"]) // 2,
        "real_input_replay": replay_records, "changed_preexisting_files": changed,
        "preserved_tracked_files": len(audit["tracked_file_hashes"]),
        "preserved_untracked_files": len(audit["intentional_untracked_files"]),
        "artifact_content_sha256": manifest["artifact_content_sha256"],
        "compact_array_whitelist_verified": True, "no_raw_or_model_or_GNSS_payload_exported": True,
        "protocol_sha256": ge.PROTOCOL_SHA256, "cache_content_sha256": b["N20_cache_content_sha256"],
        "no_TEST_access": True, "no_CAL_pseudo_reuse": True, "no_training": True,
        "no_conformal": True, "no_commit_push": True, "no_new_acquisition": True}
    ge.save_json(OUT / "integrity_results.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
