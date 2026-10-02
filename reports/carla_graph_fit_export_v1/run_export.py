"""One authorized FIT-only export; safe checkpoints avoid repeated image work."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from cognix.adapters.carla import graph_fit_export as ge
from cognix.adapters.carla import cache_builder as cb


def run_scenario(job):
    sid, ticks, parents, clean, raw_path, parameter_path, handoff, splits = job
    agents = ge.restore_agents(dict(np.load(parameter_path, allow_pickle=False)), handoff)
    # Replay canonical inference on a small clean subset, never refit or re-extract.
    for i in sorted({0, min(11, len(ticks) - 1), len(ticks) - 1}):
        for j, name in enumerate(ge.NODE_ORDER):
            features = parents[("camera", "seg", "gnss", "imu").index(name.lower())][i]
            replay = ge.recompute_node(agents[name], features)
            ge.require(np.allclose(replay, clean[i, j], rtol=1e-12, atol=1e-14),
                       "loaded agent disagrees with sealed clean output")
    # No loader enumeration, TEST data, labels, clean feature extraction or GNSS raw access.
    raw = {"scenario_path": Path(raw_path), "imu": cb._imu_accel_table(Path(raw_path))}
    ge.require(len(raw["imu"]) == 3000, "frozen IMU coverage")
    started = time.monotonic()
    result = ge.build_scenario_pairs(sid, ticks, parents, clean, raw, agents, splits)
    return sid, result, round(time.monotonic() - started, 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(os.environ["TEMP"]) / "carla_graph_fit_export_v1")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    ge.require(1 <= args.workers <= 4, "use 1..4 bounded image workers")
    ge.require(not args.output.exists(), "refusing to overwrite an existing compact artifact")
    protocol, splits, dependencies = ge.verify_frozen_dependencies(ROOT)
    b = protocol["bindings"]
    n20 = ROOT / "reports/carla_train_expand20_v1"
    dataset = json.loads((n20 / "frozen_dataset_manifest.json").read_text())
    cache_path = next(Path(p) for p in b["all_cache_handoff_file_sha256"] if p.endswith("carla_train_cache.npz"))
    parameter_path = next(Path(p) for p in b["all_cache_handoff_file_sha256"] if p.endswith("fitted_oneclass_parameters.npz"))
    handoff_path = parameter_path.parent / "manifest.json"
    handoff = json.loads(handoff_path.read_text())
    ge.require(handoff["agent_order"] == ["Camera", "Seg", "GNSS", "IMU"], "handoff agent order")
    cache_manifest = json.loads(cache_path.with_suffix(".json").read_text())
    ge.require(cache_manifest["partition"]["TRAIN_NORMAL"] == splits["sorted_FIT_ids"], "N20 FIT identity")
    # Load only cached clean blocks, never any cal_* pseudo arrays.
    with np.load(cache_path, allow_pickle=False) as z:
        cache = {k: z[k] for k in ("scenario_id", "town", "tick", "source_split", "partition", "camera", "seg", "gnss", "imu")}
    with np.load(parameter_path.parent / "compact_training_checkpoint.npz", allow_pickle=False) as z:
        fit = (z["partition"] == 0) & ~z["is_pseudo"]
        ge.require(z["calibration_valid"].tolist() == [True, True, False, True], "N20 validity mask")
        scenario_idx, handoff_ticks = z["scenario_index"][fit], z["tick"][fit]
        clean = np.stack([z[k][fit][:, [0, 3, 1]] for k in ge.FEATURE_ORDER], axis=-1)
    ge.validate_nodes(clean)
    handoff_sids = np.array([handoff["scenario_dictionary"][int(i)]["scenario_id"] for i in scenario_idx])
    ge.require(set(handoff_sids) == set(splits["sorted_FIT_ids"]), "only clean FIT handoff rows")
    source = {s["scenario_id"]: s for s in dataset["scenarios"]}
    work = args.output.with_name(args.output.name + "_work")
    work.mkdir(exist_ok=True)
    # Resume only this exporter version with exactly the same frozen dependencies.
    job_binding = {"dependencies": dependencies, "exporter_sha256": ge.sha256_file(Path(ge.__file__)),
                   "runner_sha256": ge.sha256_file(Path(__file__))}
    if (work / "binding.json").exists():
        ge.require(json.loads((work / "binding.json").read_text()) == job_binding, "checkpoint dependency mismatch")
    else:
        ge.save_json(work / "binding.json", job_binding)
    results, jobs = {}, []
    dictionary = []
    for sid in splits["sorted_FIT_ids"]:
        split = ge.frozen_split(sid, splits)
        entry = source[sid]
        ge.require(Path(entry["actual_path"]).is_dir(), "missing frozen raw FIT scenario")
        select = cache["scenario_id"] == sid
        ge.require(set(cache["partition"][select]) == {"TRAIN_NORMAL"}
                   and set(cache["source_split"][select]) == {"train"}, "FIT/TRAIN gate")
        ticks = cache["tick"][select]
        ge.require(np.array_equal(ticks, np.arange(1, 3000)), "frozen tick coverage")
        hs = handoff_sids == sid
        ge.require(np.array_equal(handoff_ticks[hs], ticks), "clean handoff/cache alignment")
        dictionary.append({"scenario_id": sid, "town": sid.split("/")[0], "partition": "FIT_NORMAL",
                           "graph_split": split, "source_hash_bundle_sha256": entry["source_hash_bundle_sha256"]})
        stem = sid.replace("/", "__")
        if (work / (stem + ".json")).exists():
            stored = json.loads((work / (stem + ".json")).read_text())
            ge.require(ge.sha256_file(work / (stem + ".npz")) == stored["tensor_file_sha256"], "checkpoint hash mismatch")
            with np.load(work / (stem + ".npz"), allow_pickle=False) as z:
                stored["node_features"] = z["node_features"]
            results[sid] = stored
            print(json.dumps({"phase": "checkpoint_reused", "scenario": sid}), flush=True)
        else:
            jobs.append((sid, ticks, tuple(cache[k][select] for k in ("camera", "seg", "gnss", "imu")),
                         clean[hs], entry["actual_path"], str(parameter_path), handoff, splits))
    print(json.dumps({"phase": "FIT_image_processing", "scenarios": len(jobs), "workers": args.workers}), flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(run_scenario, job) for job in jobs}
        while pending:
            done, pending = concurrent.futures.wait(pending, timeout=30, return_when=concurrent.futures.FIRST_COMPLETED)
            if not done:
                print(json.dumps({"phase": "FIT_image_processing", "completed": len(results), "remaining": len(pending)}), flush=True)
            for f in done:
                sid, result, wall_s = f.result()
                results[sid] = result
                stem = sid.replace("/", "__")
                with (work / (stem + ".npz")).open("xb") as out:
                    np.savez_compressed(out, node_features=result["node_features"])
                ge.save_json(work / (stem + ".json"), {**{k: v for k, v in result.items() if k != "node_features"},
                    "tensor_file_sha256": ge.sha256_file(work / (stem + ".npz"))})
                print(json.dumps({"phase": "scenario_complete", "scenario": sid, "seconds": wall_s,
                                  "retained": len(result["rows"]) // 2, "skips": len(result["skips"]),
                                  "completed": len(results)}), flush=True)
    bindings = {"protocol_sha256": ge.PROTOCOL_SHA256, "cache_content_sha256": b["N20_cache_content_sha256"],
        "cache_file_sha256": b["N20_cache_file_sha256"], "upstream_model_sha256": b["fitted_oneclass_parameters_sha256"],
        "upstream_handoff_sha256": b["N20_handoff_manifest_sha256"],
        "calibration_parameters_sha256": b["agent_calibration_results_sha256"],
        "feature_extractor_sha256": b["source_code_sha256"]["cognix/adapters/carla/real_features.py"],
        "pseudo_registry_sha256": b["source_code_sha256"]["cognix/adapters/carla/pseudo_anomalies.py"],
        "graph_schema_sha256": b["preregistration_dependencies_sha256"]["graph_schema.json"],
        "split_manifest_sha256": b["preregistration_dependencies_sha256"]["split_manifest.json"],
        "metrics_spec_sha256": b["preregistration_dependencies_sha256"]["metrics_spec.json"]}
    validity = {n: {"scientific_valid": True, "status": handoff["calibration_status"][n]} for n in ge.NODE_ORDER}
    arrays, schema, counts, skips, collisions = ge.assemble_artifact(results, bindings, dictionary, validity)
    # Verify deterministic scientific serialization without a second image-processing pass.
    content_hash = ge.scientific_content_hash(arrays, schema)
    ge.require(content_hash == ge.scientific_content_hash(dict(reversed(list(arrays.items()))), json.loads(ge.canonical_json(schema))), "hash determinism")
    ge.verify_frozen_dependencies(ROOT)
    manifest = ge.write_artifact(args.output, arrays, schema, counts, skips, collisions, job_binding)
    with np.load(args.output / "graphs.npz", allow_pickle=False) as z:
        reloaded = {k: z[k] for k in z.files}
    ge.validate_artifact(reloaded, schema)
    ge.require(ge.scientific_content_hash(reloaded, schema) == content_hash, "serialized scientific roundtrip")
    ge.save_json(Path(__file__).parent / "export_result.json", {"output": str(args.output), "manifest_sha256": ge.sha256_file(args.output / "manifest.json"), **manifest})
    print(json.dumps({"phase": "complete", "artifact": str(args.output), "content_sha256": content_hash, "counts": counts}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Fail closed and preserve completed work; never replace a tick/recipe/scenario.
        print(json.dumps({"phase": "failed_closed", "error": repr(exc)}), flush=True)
        raise
