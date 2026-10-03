"""Amendment 003: one OS-random seed receipt; protocol text changes only.

generate reserves the receipt with exclusive creation BEFORE its sole draw.
If a receipt exists (even incomplete), never draw again. amend reads that
receipt; it does not generate randomness or instantiate NumPy/PCG64.
"""
import argparse
import hashlib
import json
import os
import platform
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE / "amendment_history/pre_amendment_003"
OLD_HASH = "c6516d507a7b0189c4fc11d0ad429a7dc080ebb49ea127623a87e1c4ed39482b"
ID = "003_partition_randomization_clarification_only"
RECEIPT = HERE / "partition_randomization_receipt.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def generate():
    assert hashlib.sha256((OLD / "SHA256SUMS").read_bytes()).hexdigest() == OLD_HASH
    assert not (HERE / "amendment_003_record.json").exists()
    # Exclusive reservation prevents a retry/redraw if any subsequent step fails.
    with RECEIPT.open("x", encoding="utf-8", newline="\n") as handle:
        seed = secrets.randbits(128)  # Exactly one OS-backed 128-bit seed draw.
        now = datetime.now(timezone.utc)
        receipt = {
            "amendment_id": ID,
            "purpose": "Partition randomization only; not a model/training seed",
            "seed": seed,
            "seed_decimal": str(seed),
            "bits": 128,
            "generation_method": "Python secrets.randbits(128), backed by OS cryptographic randomness via secrets.SystemRandom",
            "draw_count": 1,
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "generated_at_utc": now.isoformat(),
            "generated_at_Asia_Calcutta": now.astimezone(timezone(timedelta(hours=5, minutes=30))).isoformat(),
            "client_timezone": "Asia/Calcutta",
            "before_any_TEST_inventory_access_in_this_milestone": True,
            "historical_exposure_qualification": "Earlier documented historical TEST exposures remain excluded; this receipt does not deny those prior exposures.",
            "inventory_accessed": False,
            "inventory_executed": False,
            "partition_executed": False,
            "PCG64_instantiated": False,
            "permutations_executed": 0,
            "immutable_rule": "Accept this first draw unconditionally, including any coincidence with an earlier seed. Never regenerate, redraw, balance, search or change it based on inventory composition, labels, predictions, metrics or conformal results. Lost/corrupt receipt means STOP; no replacement draw.",
            "future_use": "After eligible inventory is independently sealed: NumPy Generator(PCG64(SEALED_RANDOM_SEED)), exactly one permutation of lexicographically sorted eligible full IDs; n_cal=ceil(N_eligible/5); first n_cal calibration, remainder evaluation.",
            "predecessor_manifest_sha256": OLD_HASH
        }
        handle.write(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({"seed_decimal": str(seed), "generated_at_utc": now.isoformat(),
                      "python_version": receipt["python_version"], "draw_count": 1,
                      "receipt_sha256": hashlib.sha256(RECEIPT.read_bytes()).hexdigest()}))


def amend():
    assert not (HERE / "amendment_003_record.json").exists(), "Already amended; do not overwrite"
    receipt = read(RECEIPT)
    seed = receipt["seed"]
    assert type(seed) is int and 0 <= seed < 2**128 and str(seed) == receipt["seed_decimal"]
    receipt_hash = hashlib.sha256(RECEIPT.read_bytes()).hexdigest()
    partition = read(OLD / "test_partition_protocol.json")
    partition["seed"] = seed
    partition["allocation"]["seed"] = seed
    partition.pop("seed_distinct_from")
    partition["seed_receipt"] = {"path": RECEIPT.name, "sha256": receipt_hash}
    partition["seed_collision_policy"] = "The first OS-random draw is accepted unconditionally, even if it equals a historical seed; no rejection sampling or seed search."
    partition["amendment_id"] = ID
    partition["status"] = "SEALED_AMENDMENT_003_DESIGN_ONLY"
    partition["algorithm_only"][3] = "The single OS-cryptographic seed was drawn and sealed under Amendment 003 BEFORE any TEST inventory access in this milestone. Never regenerate or change it. Later seal complete source inventory, exclusions and eligible inventory hash before instantiating the partition Generator; record/pin NumPy version and UTF-8 bytewise lexicographic ID ordering."
    partition["algorithm_only"][5] = "Later, after eligible inventory is independently sealed, create exactly one NumPy Generator(PCG64(SEALED_RANDOM_SEED)) using the integer in partition_randomization_receipt.json; call permutation exactly once on the sorted complete eligible list (or its ordered integer indices). No further seed draws, rejection of seed collisions, stratum loops, redraws, balancing or seed search."
    partition["algorithm_only"][7] = partition["algorithm_only"][7].replace("RNG/version/protocol hashes", "sealed seed receipt, RNG/version/protocol hashes")
    partition["randomization_interpretation"] = "A single 128-bit seed is drawn once using local OS cryptographic randomness BEFORE any TEST inventory access in this milestone, then sealed irrevocably. Later the independently sealed eligible inventory is permuted once with Generator(PCG64(SEALED_RANDOM_SEED)). The primary finite-catalogue interpretation is marginal over preregistered randomized assignment and a uniformly selected held-out eligible catalogue scenario, conditional on frozen scorer, complete eligible inventory, uniform scenario assignment and pre-randomization exclusions; not conditional on the realized seed/partition or each calibration sample. No population/new-route/independent-family guarantee. Separately any population interpretation requires appropriate scenario-score exchangeability, which the random split does not establish."
    partition["seed_draw_timing"] = "Receipt sealed now, before any later TEST inventory access; partition Generator creation and its sole permutation remain prohibited until the inventory is independently sealed and execution is separately authorized. Historical documented exposure is not erased."
    write("test_partition_protocol.json", partition)

    cp = read(OLD / "conformal_protocol.json")
    cp["amendment_id"] = ID
    cp["status"] = "SEALED_AMENDMENT_003_DESIGN_ONLY"
    cp["primary_guarantee"] = "The primary finite-catalogue interpretation is marginal over the preregistered randomized assignment (one OS-random seed drawn and sealed before any later TEST inventory access; later one PCG64 permutation) and a uniformly selected held-out eligible released catalogue scenario. The augmented-rank >=0.95 statement remains conditional on frozen scorer, complete eligible inventory, uniform scenario assignment and exclusion rules fixed before randomization. It is not conditional on the realized seed/partition/calibration sample and is not a population, new-route or independent-family guarantee. S_new<=Q implies simultaneous inclusion of all eligible true tick labels for that one scenario."
    cp["population_interpretation"] = "Separately, population-style coverage requires an appropriate scenario-score exchangeability assumption conditional on frozen development/scorer. The random split does not establish that assumption. No population/new-route/independent-family coverage is established here."
    cp["randomization_receipt"] = {"path": RECEIPT.name, "sha256": receipt_hash}
    cp["proof"][-1] = "This statement is marginal over the preregistered randomized assignment (including the single OS-random seed draw) and held-out eligible catalogue scenario selection; never conditional on the realized seed, partition, calibration sample, cutoff, town/type, features or singleton selection. Any separate population argument still needs appropriate scenario-score exchangeability; the random split does not establish it."
    cp["randomization_assumption"] = "The rank argument retains its uniform-assignment condition. OS entropy makes the seed a probabilistic draw rather than a predetermined constant; seeded PCG64 is the registered pseudorandom implementation, not by itself a proof of exact uniformity over all catalogue permutations or of population exchangeability."
    write("conformal_protocol.json", cp)

    inventory = read(OLD / "inventory_access_protocol.json")
    inventory["amendment_id"] = ID
    inventory["gates_before_randomization"][-1] = "The OS-random seed receipt is already sealed before any later TEST inventory access. Seal source inventory hash, exact exclusion supplement and eligible inventory hash independently before instantiating Generator(PCG64(SEALED_RANDOM_SEED)) and its sole permutation. Never redraw/change the seed. Reconciliation failure or unresolved exclusion decision means STOP, no partition."
    inventory["procedure_only"][-1] = inventory["procedure_only"][-1].replace("before RNG creation", "before partition Generator creation (the OS-random seed is already sealed before inventory access)")
    write("inventory_access_protocol.json", inventory)
    ready = read(OLD / "readiness.json")
    ready["amendment_id"] = ID
    ready["blockers"][1] = ready["blockers"][1].replace("before RNG creation", "before partition Generator creation; the OS-random seed is already sealed")
    ready["next_step"] = "Stop after sealing Amendment 003. The one OS-random partition seed is sealed now; never redraw it. Later separately authorized inventory acquisition must retain all Amendment 002 gates, independently seal eligible inventory before Generator creation/permutation, and stop on failed reconciliation. No inventory, partition or scores now."
    ready["partition_seed_receipt"] = {"path": RECEIPT.name, "sha256": receipt_hash}
    write("readiness.json", ready)

    report = (OLD / "report.md").read_text(encoding="utf-8")
    report = report.replace("**PCG64 seed 2028**", "**Generator(PCG64(SEALED_RANDOM_SEED)) using the single OS-random seed in partition_randomization_receipt.json**")
    report = report.replace("**before RNG creation**", "**before partition Generator creation (the OS-random seed is sealed before inventory access)**")
    report = report.replace("PCG64(2028) is the committed reproducible realization; the guarantee is over the registered assignment mechanism and held-out scenario selection, not conditional on that realized seed/partition or each calibration sample.", "Amendment 003 draws a single 128-bit seed locally using OS cryptographic randomness before any TEST inventory access in this milestone and seals its receipt now; a predetermined constant alone was not a probabilistic randomization draw. Later, after independently sealing the eligible inventory, exactly one Generator(PCG64(SEALED_RANDOM_SEED)) permutation implements the registered assignment. The primary interpretation is marginal over preregistered randomized assignment and a held-out eligible catalogue scenario, not conditional on the realized seed/partition or each calibration sample. A population-style interpretation separately requires appropriate scenario-score exchangeability, which the random split does not establish. The rank argument retains the uniform-assignment condition; randomized seeded PCG64 is the registered implementation, not itself a proof of exact uniformity over all permutations.")
    report = report.replace("# Protocol Hashes\n\n", "# Protocol Hashes\n\n**Amendment 003 — partition randomization clarification only:** the complete 122-file Amendment 002 state, including every prior history and both seal files, is preserved byte-for-byte in `amendment_history/pre_amendment_003/`. Old manifest: `" + OLD_HASH + "`. `partition_randomization_receipt.json` records seed **" + str(seed) + "**, one `secrets.randbits(128)` draw, UTC/local timestamp and Python version. This seed is sealed before any later TEST inventory access; never redraw, change, balance or search it. Generator construction and its sole permutation remain later authorized steps after independently sealing eligible inventory. `amendment_003_record.json` and `amendment_003_integrity.json` document exact changes; all other Amendment 002 components remain frozen.\n\n", 1)
    report = report.replace("# Recommended Next Step\n\nStop after sealing Amendment 002.", "# Recommended Next Step\n\nStop after sealing Amendment 003. The partition seed was drawn once and sealed; no partition Generator or permutation was executed.")
    (HERE / "report.md").write_text(report, encoding="utf-8")
    inventory_md = (OLD / "inventory_access_protocol.md").read_text(encoding="utf-8")
    inventory_md = inventory_md.replace("one PCG64(2028) permutation", "one Generator(PCG64(SEALED_RANDOM_SEED)) permutation using the irrevocable seed receipt sealed under Amendment 003 before any later TEST inventory access")
    inventory_md = inventory_md.replace("**before RNG creation**", "**before partition Generator creation (the OS-random seed is already sealed before inventory access)**")
    (HERE / "inventory_access_protocol.md").write_text(inventory_md, encoding="utf-8")
    verifier = (OLD / "verify_protocol.py").read_text(encoding="utf-8")
    verifier = verifier.replace('if __name__ == "__main__":\n    main()', 'if __name__ == "__main__":\n    if (HERE / "amendment_003_record.json").exists():\n        import runpy\n        runpy.run_path(str(HERE / "verify_protocol_003.py"), run_name="__main__")\n    else:\n        main()')
    (HERE / "verify_protocol.py").write_text(verifier, encoding="utf-8")
    write("amendment_003_record.json", {
        "amendment_id": ID, "title": "Amendment 003 — partition randomization clarification only",
        "client_date": "2026-10-03", "client_timezone": "Asia/Calcutta",
        "authorization": "Explicit user request; one local OS-random seed generation and protocol reseal only",
        "old_manifest_sha256": OLD_HASH, "previous_archive": "amendment_history/pre_amendment_003/",
        "previous_files_archived": len(read(HERE / "amendment_003_initial_state.json")["prior_files"]),
        "receipt": {"path": RECEIPT.name, "sha256": receipt_hash, "seed": seed},
        "reason": "Fixed predetermined PCG64(2028) was reproducible but was not itself a probabilistic randomization draw; retain finite-catalogue randomization interpretation by an irrevocable OS-random seed drawn and sealed before later TEST inventory access.",
        "changes": ["Single local OS-cryptographic 128-bit seed draw and exclusive receipt", "Partition seed and randomization timing/semantics only", "Guarantee marginal over preregistered assignment and held-out eligible catalogue scenario; population exchangeability remains a separate unestablished assumption", "Current verifier dispatch and amendment validation/integrity/seals"],
        "unchanged": ["20/80 allocation ratio", "complete scenario calibration unit", "candidate score and block maximum", "pooled conformal and alpha=.05", "15 method/model-seed scorers", "all metrics", "historical exclusions", "subgroup reporting", "scientific pipeline", "inventory bounds/parser/access-ledger/output protocol"],
        "no_redraw_rule": receipt["immutable_rule"],
        "seed_generated": True, "seed_draw_count": 1, "PCG64_instantiated": False,
        "activity_attestation": {name: False for name in ["TEST_access", "TEST_scenario_enumeration", "sensor_or_label_access", "inventory_execution", "partition_execution", "inference", "conformal_fitting", "training", "tuning", "commit", "push"]},
        "new_manifest_reference": "SHA256SUMS.sha256; no self-referential hash",
        "historical_records": "Amendment 002 records/scripts/changelog remain byte-identical as historical artifacts and refer to their own checkpoint. Current randomization references are Amendment 003 and the seed receipt. Do not regenerate earlier amendments.",
        "attestation_scope": "Executed tools in this milestone, not an OS-wide access trace; archived metadata is not sensor/label access."
    })
    print(json.dumps({"amendment_written": ID, "seed_decimal": str(seed), "seal_pending": True}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["generate", "amend"])
    args = parser.parse_args()
    if args.mode == "generate":
        generate()
    else:
        amend()
