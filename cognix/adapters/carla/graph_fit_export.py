"""Frozen FIT graph export, isolated from the generic COGNIX framework.

No fitting or calibration occurs here. Clean outputs come from the sealed N20
checkpoint; only the corrupted node invokes a restored production agent.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from cognix.adapters.carla import cache_builder as cb
from cognix.adapters.carla.normality import (
    BootstrapNormalityEnsemble, EnsemblePredictiveCalibrator, MahalanobisNormality,
)
from cognix.adapters.carla.real_agents import RealCameraAgent, RealIMUAgent, RealSegAgent

PROTOCOL_SHA256 = "68f035acda758bed0e7c3799a116f58a03719fee4e39f6468fdeaac0ef6ab203"
NODE_ORDER = ("Camera", "IMU", "Seg")
FEATURE_ORDER = ("prob_normal", "epistemic", "aleatoric")
RECIPES = ("camera_brightness_shift", "camera_occlusion", "seg_region_corruption",
           "imu_spike", "imu_bias_scale")
SEVERITIES = (0.35, 0.25, 0.2, 15.0, 0.5)
ADJACENCY = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.uint8)
OFFSETS = {"camera": (0, 18), "seg": (18, 47), "gnss": (47, 55), "imu": (55, 65)}


class GraphExportError(RuntimeError):
    """A frozen dependency or scientific contract failed; never substitute data."""


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    with Path(path).open("xb") as f:
        f.write(canonical_json(value) + b"\n")


def require(condition, message):
    if not condition:
        raise GraphExportError(message)


def frozen_split(scenario_id, splits):
    require(scenario_id not in splits["AGENT_CAL_ONLY"], "CAL scenario rejected")
    roles = [r for r in ("GRAPH_TRAIN", "GRAPH_VALIDATION") if scenario_id in splits[r]]
    require(len(roles) == 1, "scenario must belong to exactly one frozen FIT split")
    return roles[0]


def selected_recipe(tick):
    require(isinstance(tick, (int, np.integer)) and 1 <= tick <= 2999,
            "parent tick must be in 1..2999")
    return RECIPES[int(tick) % 5]


def pair_id(scenario_id, tick, protocol_sha256=PROTOCOL_SHA256):
    return hashlib.sha256(canonical_json({"scenario_id": scenario_id,
        "parent_tick": int(tick), "protocol_sha256": protocol_sha256})).hexdigest()


def validate_nodes(nodes):
    a = np.asarray(nodes)
    require(a.shape[-2:] == (3, 3) and a.dtype == np.float64,
            "node tensor must have float64 scientific [Camera,IMU,Seg] x [p,E,A]")
    require(np.isfinite(a).all(), "invalid/null primary upstream probability/UQ")
    require(np.all((a[..., 0] >= 0) & (a[..., 0] <= 1)) and np.all(a[..., 1:] >= 0),
            "invalid primary probability/UQ domain")


def restore_agents(parameters, handoff):
    """Restore unchanged production classes from saved state, without any fit call."""
    result = {}
    for name, cls in zip(NODE_ORDER, (RealCameraAgent, RealIMUAgent, RealSegAgent)):
        mapping = handoff["calibration_parameters_raw"][name]
        audit = handoff["calibration_audit"][name]
        require(mapping.get("scientific_valid") is True and audit.get("finite_optimum") is True,
                f"{name}: invalid frozen calibration")
        require(all(isinstance(mapping.get(k), (int, float)) and np.isfinite(mapping[k])
                    for k in ("slope", "intercept")) and mapping["slope"] > 0,
                f"{name}: invalid/null mapping parameters")
        agent = cls(seed=42, n_members=5)
        ensemble = BootstrapNormalityEnsemble(n_members=5, seed=42)
        lo, hi = OFFSETS[name.lower()]
        for k in range(5):
            prefix = f"{name.lower()}_{k}_"
            member = MahalanobisNormality()
            member._mean = np.array(parameters[prefix + "mean"], dtype=np.float64, copy=True)
            member._prec = np.array(parameters[prefix + "precision"], dtype=np.float64, copy=True)
            member._d2_scale = float(parameters[prefix + "distance_scale"])
            require(member._mean.shape == (hi - lo,) and member._prec.shape == (hi - lo, hi - lo)
                    and np.isfinite(member._mean).all() and np.isfinite(member._prec).all()
                    and np.isfinite(member._d2_scale) and member._d2_scale > 0,
                    f"{name}: invalid member state")
            ensemble._members.append(member)
        calibrator = EnsemblePredictiveCalibrator()
        calibrator.a, calibrator.b = mapping["slope"], mapping["intercept"]
        calibrator._fitted = True
        calibrator._validity = dict(audit)
        calibrator._nll_initial = audit["nll_initial"]
        calibrator._nll_final = audit["nll_final"]
        calibrator._n_iter = audit["n_iter"]
        calibrator._converged = audit["converged"]
        agent._ensemble, agent._calibrator = ensemble, calibrator
        result[name] = agent
    return result


def recompute_node(agent, features):
    # This is the production frozen path: feature vector -> fitted members ->
    # accepted member calibrator -> canonical Bernoulli entropy decomposition.
    u = agent.estimate_uncertainty(features)
    node = np.array([u.prediction, u.epistemic, u.aleatoric], dtype=np.float64)
    validate_nodes(np.tile(node, (3, 1)))
    require(np.isfinite(u.total) and u.total >= 0, "invalid canonical total")
    return node


def build_scenario_pairs(scenario_id, ticks, parent_features, clean_nodes, raw, agents, splits):
    """Use the frozen production recipe/feature/no-effect path once per parent.

    Missing pseudo ticks are identified from the production no-effect ledger;
    any other error fails the scenario. Only accepted corrupted blocks are
    scored, and the untouched cached p/E/A blocks are copied bit for bit.
    """
    split = frozen_split(scenario_id, splits)  # before raw access/generation
    ticks = np.asarray(ticks)
    require(np.array_equal(ticks, np.arange(1, len(ticks) + 1)) and len(ticks) <= 2999,
            "parents must be canonical consecutive cached ticks, with no replacement")
    validate_nodes(clean_nodes)
    require(clean_nodes.shape == (len(ticks), 3, 3), "clean output alignment")
    for block, name in zip(parent_features, ("camera", "seg", "gnss", "imu")):
        lo, hi = OFFSETS[name]
        require(block.shape == (len(ticks), hi - lo) and np.isfinite(block).all(),
                "invalid cached parent features")
    require(cb.WINDOW_LENGTH == 12, "frozen causal window changed")
    meta, features, attempts, skips = cb._pseudo_for_scenario(
        raw, parent_features, scenario_id, RECIPES, 0, 1, None)
    require(len(meta) == len(features), "pseudo metadata/feature alignment")
    require(attempts == dict(Counter(selected_recipe(int(t)) for t in ticks)), "recipe frequency changed")
    require(all(v.get("parent_uncached", 0) == v.get("feature_extraction_error", 0) == 0
                for v in skips.values()), "unexpected upstream skip; fail closed")
    accepted = {int(m["parent_tick"]): (m, f) for m, f in zip(meta, features)}
    require(len(accepted) == len(meta) and set(accepted) <= set(ticks.tolist()), "duplicate/foreign parent")
    ledger, rows, nodes, collisions = [], [], [], []
    missing = Counter()
    for t in ticks.tolist():
        rid = selected_recipe(t)
        modality = rid.split("_")[0]
        if t not in accepted:
            missing[rid] += 1
            ledger.append({"scenario_id": scenario_id, "tick": t, "recipe_id": rid,
                           "modality": modality, "reason": "no_effect"})
            continue
        m, feature = accepted[t]
        feature = np.asarray(feature)
        require(feature.shape == (65,) and feature.dtype == np.float64 and np.isfinite(feature).all(),
                "invalid production corrupted compact features")
        require(m["parent_scenario"] == scenario_id and m["source_split"] == "train"
                and m["recipe_id"] == rid and m["modality"] == modality
                and m["seed"] == t and m["severity"] == SEVERITIES[t % 5]
                and m["window_start_tick"] == max(0, t - 11) and m["window_end_tick"] == t,
                "frozen pseudo provenance mismatch")
        for name, block in zip(("camera", "seg", "gnss", "imu"), parent_features):
            lo, hi = OFFSETS[name]
            if name != modality:
                require(feature[lo:hi].tobytes() == block[t - 1].tobytes(), "untouched compact block changed")
        lo, hi = OFFSETS[modality]
        parent = parent_features[("camera", "seg", "gnss", "imu").index(modality)][t - 1]
        require(not np.allclose(feature[lo:hi], parent, rtol=1e-12, atol=1e-12), "no-effect pseudo escaped guard")
        j = tuple(n.lower() for n in NODE_ORDER).index(modality)
        clean = clean_nodes[t - 1].copy()
        pseudo = clean.copy()
        pseudo[j] = recompute_node(agents[NODE_ORDER[j]], feature[lo:hi])
        for intact in set(range(3)) - {j}:
            require(pseudo[intact].tobytes() == clean[intact].tobytes(), "untouched node changed")
        pid = pair_id(scenario_id, t)
        if np.array_equal(pseudo[j], clean[j]):
            collisions.append({"scenario_id": scenario_id, "tick": t, "recipe_id": rid,
                               "modality": modality, "pair_id": pid,
                               "reason": "material_features_changed_but_p_E_A_identical"})
        for is_pseudo, node in ((False, clean), (True, pseudo)):
            rows.append({"scenario_id": scenario_id, "town": scenario_id.split("/")[0],
                "parent_tick": t, "graph_split": split, "is_pseudo": is_pseudo,
                "target_normal": int(not is_pseudo), "pair_key": pid,
                "recipe_id": rid if is_pseudo else None,
                "severity": m["severity"] if is_pseudo else None,
                "generator_seed": t if is_pseudo else None,
                "corrupted_agent": NODE_ORDER[j] if is_pseudo else None,
                "window_start_tick": max(0, t - 11), "window_end_tick": t,
                "corruption_start_tick": max(0, t - 11) if modality == "imu" else t,
                "corruption_end_tick": t})
            nodes.append(node)
    require(dict(missing) == {rid: v["no_effect"] for rid, v in skips.items() if v["no_effect"]},
            "missing pseudo lacks an exact production no-effect reason")
    tensor = np.array(nodes, dtype=np.float64).reshape(-1, 3, 3)
    return {"rows": rows, "node_features": tensor, "skips": ledger, "collisions": collisions,
            "candidate_parents": len(ticks)}


def graph_content_hash(provenance, nodes):
    """Sealed model-view hash. Float64 scientific storage has a separate hash."""
    h = hashlib.sha256(canonical_json(provenance))
    h.update(np.ascontiguousarray(nodes, dtype="<f4").tobytes())
    h.update(np.ascontiguousarray(nodes[:, 1], dtype="<f8").tobytes())
    h.update(ADJACENCY.tobytes())
    return h.hexdigest()


def scientific_content_hash(arrays, schema):
    h = hashlib.sha256(canonical_json(schema))
    for key in sorted(arrays):
        a = np.asarray(arrays[key])
        # Unicode hashes are canonical UTF8 JSON, avoiding platform UTF32 endian differences.
        if a.dtype.kind == "U":
            raw, dtype = canonical_json(a.tolist()), "unicode_utf8"
        else:
            a = np.ascontiguousarray(a, dtype=a.dtype.newbyteorder("<"))
            raw, dtype = a.tobytes(), a.dtype.str
        h.update(canonical_json({"name": key, "dtype": dtype, "shape": list(a.shape)}))
        h.update(raw)
    return h.hexdigest()


def assemble_artifact(results, bindings, scenario_dictionary, calibration_metadata):
    rows, tensors, ledger, collisions = [], [], [], []
    for sid in sorted(results):
        block = results[sid]
        require(all(r["scenario_id"] == sid for r in block["rows"]), "foreign scenario result")
        rows.extend(block["rows"])
        tensors.append(block["node_features"])
        ledger.extend(block["skips"])
        collisions.extend(block["collisions"])
    nodes = np.concatenate(tensors)
    validate_nodes(nodes)
    scenario_ids = [s["scenario_id"] for s in scenario_dictionary]
    require(scenario_ids == sorted(results), "FIT scenario dictionary mismatch")
    towns = sorted({s.split("/")[0] for s in scenario_ids})
    source_by_sid = {s["scenario_id"]: s["source_hash_bundle_sha256"] for s in scenario_dictionary}
    for r in rows:
        r.update(bindings)
        r.update(source_split="train", N20_partition="FIT_NORMAL", generator_base_seed=0,
                 node_order=list(NODE_ORDER), source_hash_bundle_sha256=source_by_sid[r["scenario_id"]])
        r["target_normal_per_node"] = [int(n != r["corrupted_agent"]) for n in NODE_ORDER]
        r["row_key"] = hashlib.sha256(canonical_json({k: r[k] for k in
            ("pair_key", "is_pseudo", "recipe_id", "severity", "generator_seed")})).hexdigest()
    # Explicit dtype/sentinel contract: float scientific null is NaN; integer null -1.
    arrays = {
        "node_features": nodes,
        "target_normal": np.array([r["target_normal"] for r in rows], dtype=np.uint8),
        "target_normal_per_node": np.array([r["target_normal_per_node"] for r in rows], dtype=np.uint8),
        "is_pseudo": np.array([r["is_pseudo"] for r in rows], dtype=bool),
        "split": np.array([int(r["graph_split"] == "GRAPH_VALIDATION") for r in rows], dtype=np.uint8),
        "scenario_index": np.array([scenario_ids.index(r["scenario_id"]) for r in rows], dtype=np.uint16),
        "town_index": np.array([towns.index(r["town"]) for r in rows], dtype=np.uint8),
        "tick": np.array([r["parent_tick"] for r in rows], dtype=np.int32),
        "pair_id": np.array([r["pair_key"] for r in rows], dtype="U64"),
        "row_key": np.array([r["row_key"] for r in rows], dtype="U64"),
        "parent_scenario_index": np.array([scenario_ids.index(r["scenario_id"]) for r in rows], dtype=np.uint16),
        "parent_tick": np.array([r["parent_tick"] for r in rows], dtype=np.int32),
        "corrupted_node_index": np.array([NODE_ORDER.index(r["corrupted_agent"]) if r["is_pseudo"] else -1 for r in rows], dtype=np.int8),
        "recipe_index": np.array([RECIPES.index(r["recipe_id"]) if r["is_pseudo"] else -1 for r in rows], dtype=np.int8),
        "severity": np.array([r["severity"] if r["is_pseudo"] else np.nan for r in rows], dtype=np.float64),
        "recipe_seed": np.array([r["generator_seed"] if r["is_pseudo"] else -1 for r in rows], dtype=np.int64),
        "window_start_tick": np.array([r["window_start_tick"] for r in rows], dtype=np.int32),
        "window_end_tick": np.array([r["window_end_tick"] for r in rows], dtype=np.int32),
        "corruption_start_tick": np.array([r["corruption_start_tick"] if r["is_pseudo"] else -1 for r in rows], dtype=np.int32),
        "corruption_end_tick": np.array([r["corruption_end_tick"] if r["is_pseudo"] else -1 for r in rows], dtype=np.int32),
        "graph_content_sha256": np.array([graph_content_hash(r, n) for r, n in zip(rows, nodes)], dtype="U64"),
        "calibration_valid": np.ones(3, dtype=bool),
    }
    schema = {"artifact_version": "carla_graph_fit_export_v1", "schema_version": 1,
        "node_order": list(NODE_ORDER), "feature_order": list(FEATURE_ORDER),
        "scientific_dtype": "float64", "adjacency": ADJACENCY.tolist(),
        "split_dictionary": {"0": "GRAPH_TRAIN", "1": "GRAPH_VALIDATION"},
        "scenario_dictionary": scenario_dictionary, "town_dictionary": towns,
        "recipe_dictionary": list(RECIPES), "corrupted_node_dictionary": {"-1": None, "0": "Camera", "1": "IMU", "2": "Seg"},
        "null_encoding": "NaN severity, -1 recipe/seed/corrupted node/corruption bounds; reconstructed row provenance uses null",
        "canonical_order": "scenario_id,parent_tick,is_pseudo; clean before pseudo",
        "common_provenance": bindings, "source_split": "train", "N20_partition": "FIT_NORMAL",
        "generator_base_seed": 0, "calibration_validity": calibration_metadata,
        "content_hash": "SHA256 canonical sorted compact UTF8 schema JSON then sorted arrays: canonical name/dtype/shape header plus little-endian C bytes; Unicode as canonical UTF8 JSON",
        "graph_content_hash": "sealed graph_schema.json model-view algorithm; scientific storage remains float64",
        "raw_data_present": False, "graph_model_parameters_present": False,
        "arrays": {k: {"dtype": str(v.dtype), "shape": list(v.shape)} for k, v in arrays.items()}}
    validate_artifact(arrays, schema)
    def counts_subset(subset):
        pseudo = [r for r in subset if r["is_pseudo"]]
        return {"retained_parents": len(pseudo), "clean_rows": len(subset) - len(pseudo),
                "pseudo_rows": len(pseudo), "graph_rows": len(subset),
                "per_recipe": dict(Counter(r["recipe_id"] for r in pseudo)),
                "per_modality": dict(Counter(r["corrupted_agent"] for r in pseudo))}
    counts = counts_subset(rows)
    counts.update(candidate_parents=sum(v["candidate_parents"] for v in results.values()),
                  skipped_parents=len(ledger), node_output_collisions=len(collisions))
    counts["per_split"] = {s: counts_subset([r for r in rows if r["graph_split"] == s]) for s in ("GRAPH_TRAIN", "GRAPH_VALIDATION")}
    counts["per_scenario"] = {s: {**counts_subset([r for r in rows if r["scenario_id"] == s]),
        "candidate_parents": results[s]["candidate_parents"], "skipped_parents": len(results[s]["skips"])} for s in scenario_ids}
    require(counts["candidate_parents"] == counts["retained_parents"] + counts["skipped_parents"], "pair accounting")
    return arrays, schema, counts, ledger, collisions


def validate_artifact(a, schema):
    """Validate pairing, precision, labels, causal provenance and untouched nodes."""
    require(schema["node_order"] == list(NODE_ORDER) and schema["feature_order"] == list(FEATURE_ORDER)
            and schema["adjacency"] == ADJACENCY.tolist(), "schema contract mismatch")
    validate_nodes(a["node_features"])
    n = len(a["tick"])
    require(n % 2 == 0 and len(set(a["pair_id"].tolist())) == n // 2, "exactly two rows per pair")
    require(len(set(a["row_key"].tolist())) == n, "duplicate graph row keys")
    require(np.array_equal(a["is_pseudo"], np.tile([False, True], n // 2))
            and np.array_equal(a["target_normal"], np.tile([1, 0], n // 2)), "clean/pseudo targets/order")
    for key in ("pair_id", "scenario_index", "town_index", "tick", "split", "parent_tick", "parent_scenario_index", "window_start_tick", "window_end_tick"):
        require(np.array_equal(a[key][::2], a[key][1::2]), "pair provenance mismatch: " + key)
    require(np.array_equal(a["parent_tick"], a["tick"]) and np.array_equal(a["parent_scenario_index"], a["scenario_index"]), "foreign parent")
    require(np.all((a["tick"] >= 1) & (a["tick"] <= 2999)) and set(a["split"].tolist()) <= {0, 1}, "tick/split domain")
    require(np.all(a["recipe_index"][::2] == -1) and np.all(a["corrupted_node_index"][::2] == -1)
            and np.all(a["recipe_seed"][::2] == -1) and np.isnan(a["severity"][::2]).all(), "clean sentinel fields")
    require(np.array_equal(a["recipe_index"][1::2], a["tick"][1::2] % 5)
            and np.array_equal(a["recipe_seed"][1::2], a["tick"][1::2]), "recipe rotation/seed")
    require(np.array_equal(a["window_end_tick"], a["tick"])
            and np.array_equal(a["window_start_tick"], np.maximum(0, a["tick"] - 11)), "causal window")
    require(a["calibration_valid"].tolist() == [True, True, True], "invalid mapping mask")
    previous = None
    for i in range(0, n, 2):
        sid = schema["scenario_dictionary"][int(a["scenario_index"][i])]["scenario_id"]
        entry = schema["scenario_dictionary"][int(a["scenario_index"][i])]
        require(entry["partition"] == "FIT_NORMAL" and entry["graph_split"] == schema["split_dictionary"][str(int(a["split"][i]))], "CAL/foreign split in export")
        key = (sid, int(a["tick"][i]))
        require(previous is None or previous < key, "noncanonical ordering")
        previous = key
        require(a["pair_id"][i] == pair_id(*key, schema["common_provenance"]["protocol_sha256"]), "unstable pair identity")
        recipe_idx = int(a["recipe_index"][i + 1])
        j = int(a["corrupted_node_index"][i + 1])
        require(j == (0, 0, 2, 1, 1)[recipe_idx] and a["severity"][i + 1] == SEVERITIES[recipe_idx], "single affected modality/severity")
        require(np.all(a["target_normal_per_node"][i] == 1)
                and a["target_normal_per_node"][i + 1].tolist() == [int(k != j) for k in range(3)], "node targets")
        require(a["corruption_start_tick"][i + 1] == (max(0, key[1] - 11) if j == 1 else key[1])
                and a["corruption_end_tick"][i + 1] == key[1], "corruption scope")
        for intact in set(range(3)) - {j}:
            require(a["node_features"][i, intact].tobytes() == a["node_features"][i + 1, intact].tobytes(), "untouched node invariant")


def write_artifact(output, arrays, schema, counts, skips, collisions, dependencies):
    output = Path(output)
    output.mkdir(exist_ok=False)
    # No object arrays, raw payload, cached feature vectors or model parameters.
    with (output / "graphs.npz").open("xb") as f:
        np.savez_compressed(f, **arrays)
    for name, value in (("schema.json", schema), ("counts.json", counts),
                        ("skip_ledger.json", skips), ("node_output_collisions.json", collisions),
                        ("upstream_dependencies.json", dependencies)):
        save_json(output / name, value)
    manifest = {"artifact_version": "carla_graph_fit_export_v1",
        "artifact_content_sha256": scientific_content_hash(arrays, schema),
        "schema_sha256": sha256_file(output / "schema.json"), "counts": counts,
        "files": {p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size}
                  for p in sorted(output.iterdir())},
        "raw_data_present": False, "training_performed": False,
        "TEST_access": False, "CAL_pseudo_reused": False}
    save_json(output / "manifest.json", manifest)
    return manifest


def verify_frozen_dependencies(root):
    root = Path(root)
    seal = root / "reports/carla_gat_preregistration_v1"
    require(sha256_file(seal / "protocol.json") == PROTOCOL_SHA256, "protocol hash mismatch")
    protocol = json.loads((seal / "protocol.json").read_text())
    dependencies = {"protocol.json": PROTOCOL_SHA256}
    for line in (seal / "SHA256SUMS").read_text().splitlines():
        h, name = line.split("  ", 1)
        require(sha256_file(seal / name) == h, "sealed file changed: " + name)
        dependencies[str(seal / name)] = h
    require(sha256_file(seal / "SHA256SUMS") == (seal / "SHA256SUMS.sha256").read_text().strip().split()[0], "seal checksum list")
    b = protocol["bindings"]
    for path, h in b["all_cache_handoff_file_sha256"].items():
        require(sha256_file(path) == h, "frozen handoff/cache mismatch: " + path)
        dependencies[path] = h
    for group in ("source_code_sha256", "generic_graph_source_sha256"):
        for rel, h in b[group].items():
            require(sha256_file(root / rel) == h, "upstream source changed: " + rel)
            dependencies[rel] = h
    n20 = root / "reports/carla_train_expand20_v1"
    for name, field in (("frozen_dataset_manifest.json", "N20_dataset_manifest_sha256"),
                        ("frozen_acquisition_config.json", "N20_acquisition_config_sha256"),
                        ("calibration_results.json", "agent_calibration_results_sha256")):
        require(sha256_file(n20 / name) == b[field], "N20 dependency changed: " + name)
        dependencies[str(n20 / name)] = b[field]
    cache = next(Path(p) for p in b["all_cache_handoff_file_sha256"] if p.endswith("carla_train_cache.npz"))
    require(cb._content_sha256(cache) == b["N20_cache_content_sha256"], "N20 scientific cache content changed")
    splits = json.loads((seal / "split_manifest.json").read_text())
    require(protocol["node_order"] == list(NODE_ORDER) and protocol["FIT_graph_pseudo_policy"]["recipe_order"] == list(RECIPES), "frozen constants mismatch")
    require(protocol["FIT_graph_pseudo_policy"]["severity"] == dict(zip(RECIPES, SEVERITIES)), "frozen severity mismatch")
    require(splits["sorted_FIT_ids"] == sorted(splits["GRAPH_TRAIN"] + splits["GRAPH_VALIDATION"])
            and not set(splits["sorted_FIT_ids"]) & set(splits["AGENT_CAL_ONLY"]), "FIT/CAL split integrity")
    return protocol, splits, dependencies
