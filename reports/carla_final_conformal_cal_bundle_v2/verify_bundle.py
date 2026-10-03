"""Offline pre-access verifier. No HTTP, raw TEST, training, or fitting."""
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PARTITION_SEAL = "e3047f2032896cc062f4b90eab9f7488151b9680e1b3c1234a81c112dcce577c"
CAL_SHA = "652020b48796d8035d5b78aac9b3d1f869d9ebddcefa3963b2bea02f092bb858"
EVAL_SHA = "85bb93828421c53e54064bfc7de0eb355b5e016427749ada59ee4fffdec157da"
MEMBERSHIP_SHA = "e41392264aae01e47c67b27fa8ad3a1c28c7c589c55e35e778d0502d4e567ada"
EXCLUSIONS = {"test/anomaly/Town01/change-weather/scenario-1", "test/anomaly/Town01/change-weather/scenario-10"}


def require(ok, message):
    if not ok:
        raise RuntimeError("PRE_ACCESS_REFUSAL: " + message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def verify(require_ready=True):
    manifest = HERE / "BUNDLE_SHA256SUMS"
    digest = sha(manifest)
    require((HERE / "BUNDLE_SHA256SUMS.sha256").read_text().split() == [digest, "BUNDLE_SHA256SUMS"], "manifest seal")
    seen = set()
    for line in manifest.read_text().splitlines():
        expected, rel = line.split("  ", 1)
        path = (HERE / rel).resolve()
        require(path.is_relative_to(HERE) and rel not in seen and path.is_file(), "manifest paths")
        require(sha(path) == expected, "file hash: " + rel)
        seen.add(rel)
    allowed_unsealed = {"BUNDLE_SHA256SUMS", "BUNDLE_SHA256SUMS.sha256", "cognix_final_conformal_cal_kaggle_v2.zip",
                        "cognix_final_conformal_cal_kaggle_v2.zip.sha256"}
    actual = {p.relative_to(HERE).as_posix() for p in HERE.rglob("*") if p.is_file()}
    require(actual - allowed_unsealed == seen, "unsealed/missing files")
    require(sha(HERE / "evidence/partition/SHA256SUMS") == PARTITION_SEAL, "frozen partition seal")
    for line in (HERE / "evidence/partition/SHA256SUMS").read_text().splitlines():
        expected, rel = line.split("  ", 1)
        require(sha(HERE / "evidence/partition" / (rel + ".txt" if rel.endswith(".py") else rel)) == expected,
                "partition original file: " + rel)
    require(sha(HERE / "final_conformal_cal.txt") == CAL_SHA, "CAL membership")
    require(sha(HERE / "final_evaluation.txt") == EVAL_SHA, "EVAL membership")
    require(sha(HERE / "evidence/partition/partition_membership.json") == MEMBERSHIP_SHA, "membership JSON")
    cal = (HERE / "final_conformal_cal.txt").read_text().splitlines()
    evaluation = (HERE / "final_evaluation.txt").read_text().splitlines()
    require(len(cal) == len(set(cal)) == 125 and len(evaluation) == len(set(evaluation)) == 500, "role counts")
    require(not set(cal) & set(evaluation) and len(set(cal) | set(evaluation)) == 625, "disjoint union")
    require(not (set(cal) | set(evaluation)) & EXCLUSIONS, "historical exclusions")
    eligible = (HERE / "evidence/inventory/eligible_inventory_sorted.txt").read_text().splitlines()
    require(set(eligible) == set(cal) | set(evaluation), "exact eligible inventory union")
    for name, seal in (("SOURCE_INVENTORY_SHA256SUMS", "e5219b9dae9447bf8d818828e5ef9ee7987cc5d375f345c4be7d52ad045a333c"),
                       ("ELIGIBLE_INVENTORY_SHA256SUMS", "9c08a8cc404b857d8030d316eac12f26fd9145e301ecdce1c081a4c82e981218")):
        require(sha(HERE / "evidence/inventory" / name) == seal, "inventory seal")
        for line in (HERE / "evidence/inventory" / name).read_text().splitlines():
            expected, rel = line.split("  ", 1)
            require(sha(HERE / "evidence/inventory" / rel) == expected, "inventory binding")
    rows = read("model_registry.json")["scorers"]
    require(len(rows) == 15 and {(r["method"], r["seed"]) for r in rows} == {
        (m, s) for m in ("nograph", "standard_gat", "epistemic_gat") for s in (101, 202, 303, 404, 505)}, "15 scorer identities")
    frozen = read("evidence/protocol/frozen_method_bindings.json")["runs"]
    originals = {(r["method"], r["seed"]): r for r in frozen}
    for row in rows:
        original = originals[(row["method"], row["seed"])]
        for key in ("checkpoint", "file_sha256", "content_sha256", "selected_epoch", "frozen_anomaly_threshold"):
            require(row[key] == original[key], "registered checkpoint identity")
        require(row["strict_load_passed"] and row["original_content_hash_verified"], "loadability evidence")
        require(sha(HERE / row["bundle_state"]) == row["bundle_state_sha256"], "compact state")
    protocol = read("evidence/protocol/conformal_protocol.json")
    specification = read("conformal_specification.json")
    for key in ("alpha", "numerics", "score", "scenario_score", "eligible_ticks", "prediction_set", "calibration"):
        require(specification[key] == protocol[key], "conformal preregistration: " + key)
    require(specification["n_cal"] == 125 and specification["rank_1_based"] == 120, "exact rank")
    from calibration_adapter import threshold
    require(threshold([0.5] * 125)["rank_1_based"] == 120, "adapter exact rank")
    require(read("label_semantics.json")["resolved_before_TEST_access"] is True, "official label semantics")
    scientific = read("scientific_bindings.json")
    for rel, expected in scientific["bundle_scientific_file_sha256"].items():
        require(sha(HERE / rel) == expected, "scientific binding")
    policy = read("protected_eval_policy.json")
    require(policy["membership_override"] is False and policy["training"] is False and policy["tuning"] is False,
            "inference/protection policy")
    require(policy["expected_compressed_size"] == 91538225599 and policy["expected_archive_sha256"] ==
            "267e48f2249deb0269ad950aa81bca57dc02e3bbdf2d73acc172af267b18254a", "archive fixed binding")
    validation = read("pre_access_validation.json")
    require(validation["synthetic_verification_groups_failed"] == 0 and validation["synthetic_verification_groups_passed"] > 0,
            "synthetic guards")
    if require_ready:
        require(validation["readiness"] == "READY_FOR_SEPARATELY_AUTHORIZED_FINAL_CONFORMAL_CAL_ATTEMPT_002", "bundle is not ready")
    return {"status": "PASSED", "manifest_sha256": digest, "sealed_files": len(seen), "cal": 125, "eval": 500,
            "scorers": 15, "TEST_access_performed": False, "readiness": validation["readiness"]}


if __name__ == "__main__":
    print(json.dumps(verify(), sort_keys=True))
