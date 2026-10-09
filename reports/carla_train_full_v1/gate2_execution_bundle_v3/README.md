# Experiment 2B Gate-2 engineering execution amendment v3

PREPARATION ONLY. No real Gate-2 fitting, CAL pseudo generation, calibration,
scientific audit, graph, GAT, TEST or final evaluation is authorized or executed
by preparation. STOP FOR HUMAN REVIEW.

V2 stopped at `CAL_pseudo_only` with `KeyError('record_sha256')`. It had committed
exactly three immutable FIT units and no CAL, cal-source-verification, audit,
decision or FINAL result. These FIT units are intermediate execution evidence;
no scientific modality validity or Gate-2 PASS/FAIL is inferred. The failed
attempt and all 1,288 raw CAL files (329,415,042 bytes) remain intact, bound by
their verified stopped-attempt seal and the exact preservation baseline.

Preserved v2 FIT-unit seals:

| Unit | SHA256 seal |
| --- | --- |
| fit-Camera | 531ec474585318ac91ba8d7f281690c432a4a444fee96cf743a5a44222780bb6 |
| fit-IMU | 3ca96bf8a5aa1042619ab721c3f517bb38df0e592ff5f60aa509e946de4d3bd4 |
| fit-Seg | ee72ee058a38ca87c451e26f4c814d7db92a1ac000d6cbbe278ebcded22f8075 |

This is an engineering evidence-binding defect in Experiment 2B. The raw full
scenario records yielded by sealed `local_inventory.inventory_records(V8)`
do not contain `record_sha256`. The separately sealed index stores
`dict(local_inventory.summary(full_record), record_sha256=local_inventory.digest(full_record))`.
V2's synthetic replay fixture inserted a dummy hash in its raw record, masking
the production-schema difference. Neither preserved v2 code nor its evidence
has been edited to correct that fixture.

V3 uses the exact authenticated sealed `local_inventory.summary` and
`local_inventory.digest` implementations. Their digest hashes the original
formatted JSON representation defined by sealed `atomic_publication.json_bytes`;
compact JSON or a newly invented algorithm is not substituted.
`gate2_replay_inventory.py` first authenticates the v8 bundle and independently
verifies its full inventory through the canonical sealed verifier. It then
binds the fixed index/ledger paths and their seals and checks the index schema,
source, count, alignment, uniqueness, hashes and exact summary representation.
For each replayed scenario it derives the digest from the full raw record and
requires exact equality of the reconstructed canonical summary plus digest
with the indexed entry at `archive_ordinal - 1`. This covers scenario identity,
ordinal, first TAR offset, source-member set, parser member chain, n_ticks,
emitted_rows and every other canonical summary field. Changes to excluded
summary fields such as full members or metadata still change the full-record
digest. Caller-provided raw `record_sha256` fields are rejected and never added.

`CALSink` stores the verified digest separately from `self.current`, preserving
the raw ledger representation. It validates the binding at scenario begin and
independently rechecks it at scenario completion before pseudo consumption.
Only that derived value enters the CAL unit's
`scenario_inventory_record_sha256` input binding. Exact existing live replay,
member, source, tick, role, raw-parent, pseudo and complete-archive checks remain.
Missing/extra/duplicate/unaligned entries, malformed or modified evidence,
changed index/ledger/seals, path escape, links, reparse points and junctions
fail closed.

The future runtime is exclusively
`reports/carla_train_full_v1/gate2_train_health_v3/`, with schema
`transactional-Experiment-2B-Gate2-v3` and a new v3 authorization token. It is
absent during preparation. Its manifest binds its own reviewed v3 bundle seal,
both historical failures, all block resolutions and the canonical inventory
binding proof. V1/v2 runtime namespaces are admitted only as protected historical
evidence. No v2 FIT directory, unit manifest or fitted-state NPZ is copied or
adopted into v3. Future v3 execution recomputes the same frozen FIT work in its
own manifest using the same clean bytes, sorted membership, environment,
unchanged fitting implementation, five members and seed42. No cross-runtime
adoption or scientific selection machinery is introduced.

Scientific invariants remain unchanged:

- Experiment 2B; P(target = normal) under the TRAIN-derived constructed
  clean-vs-pseudo calibration task.
- Camera18, Seg29, IMU10; window12; GNSS downstream EXCLUDED.
- Exactly 76 frozen FIT_NORMAL scenarios, clean TRAIN only; no CAL/pseudo/TEST
  in fitting. Exactly 25 frozen CAL_NORMAL scenarios, clean plus matched pseudo.
- Five bootstrap members, seed42; unchanged one-class, scoring and calibration
  validity/uncertainty code and model architecture.
- Natural-log entropy: total = H(mean p), aleatoric = mean H(member p),
  epistemic = max(0, total - aleatoric).
- Severities: brightness .35, occlusion .25, Seg corruption .2, IMU spike 15.0,
  IMU bias scale .5; max one pseudo/tick. Original tick seeds and seven-slot
  recipe rotation retain skipped GNSS slots.
- Status priority INVALID_NUMERIC, INVALID_STRUCTURAL, INVALID_OTHER, VALID.
  PASS requires Camera, Seg and IMU all VALID. Otherwise FAIL. Both outcomes
  stop for human review; no automatic graph/GAT/TEST/final evaluation.
- No tuning, search, rescue or scientific repair; this is not Experiment 2C.

`gate2_science.py`, `gate2_audit.py`, `execute_gate2.py`, `gate2_data.py` and
`gate2_resolution.py` remain byte-identical to v2. Frozen protocol files and
scientific parameters, memberships, dependency/runtime file hashes and source
bindings are unchanged. `scientific_invariance_audit.json` also proves equality
of all pseudo-generation scientific function bodies and the CAL scientific
body after normalizing only the explicit engineering binding edits.
`engineering_patch.diff` and `code_change_audit.json` enumerate every changed
and added Python file relative to v2. Changes to preparation/tests, forensic
bindings, namespace/version and inventory admission are reported transparently.

The mandatory realistic fixture is built by the exact sealed
`local_inventory.build_inventory` from a tiny valid artificial TRAIN archive,
with full source members and valid Feather metadata. It reads the resulting
raw gzip ledger and separate index through their actual sealed APIs, verifies
their canonical digest equality, and extracts/evaluates the exact preserved
v2 failing expression. That expression raises `KeyError('record_sha256')`,
while v3 succeeds. `realistic_schema_regression_attempt_*_evidence.json`
contains the full raw sample, indexed summary, hashes and observed old failure.
Synthetic tests also retain the full-window CAL replay test with the corrected
raw/index representation, both feature-storage forms, mixed FIT/CAL isolation,
unchanged scientific regressions and transactional resume/failure tests.
Adversarial tests cover each requested inventory mismatch and ambiguity.
Link/reparse tests inject Windows path indicators in synthetic paths; no link
is created in historical evidence.

Preparation independently rehashes the complete local TRAIN archive and checks
all 101 actual clean block resolutions: 33 native v8 and 68 adopted v7; FIT
24/52 and CAL 9/16. It also cross-checks all 101 real raw-ledger scenario digests
with the sealed index without generating features or pseudo. Both historical
Gate-2 bundles/runtimes, Gate-1 v7/v8, partition, and all earlier failure/raw/
pending/attempt evidence are preserved by exact inventory, hash, size and mtime.
Engineering readback of historical intermediate states makes no scientific
validity claim. Synthetic fixtures may exercise modeling, calibration, audit
and pseudo primitives on artificial data only; the runner forbids opening
real feature blocks or the real TRAIN archive. Actual preparation flags are
fitting=false, calibration=false, pseudo=false, scientific_audit=false,
graph=false, GAT=false, TEST_requests=0 and network_requests=0.
No dependency installation, commit or push occurs.

The package includes immutable bindings, v1/v2 forensic audits, preservation
baseline/audit, real read-only verification and inventory binding proofs,
realistic schema regression evidence, synthetic results/transcripts, the review
patch, exact bundle inventory, SHA256SUMS and detached SHA256SUMS.sha256.
Failed preparation attempts, if any, are retained. Owned synthetic scratch is
removed after its results/transcripts are preserved. The initial preparation
audit checkpoint retains a legacy field name; the current audit labels its
actual v3 preparation facts correctly without changing any execution fact.

## Future commands after review

Run from `C:\Users\Aditya\Cognix2_recovered` in the bound existing Python 3.14
environment. Replace `<REVIEWED_V3_SEAL>` with the reviewed detached 64-character
seal. The value remains external to payload files to avoid self-reference.

Read-only preflight: no runtime creation, pseudo, fitting, calibration or
scientific audit. Independently validates the whole source archive, all 101
blocks/adoptions and canonical replay bindings, exact partition/science/
environment, v1/v2 failure and FIT seals, and all protected preservation.

```powershell
python -B reports/carla_train_full_v1/gate2_execution_bundle_v3/gate2_preflight.py --bundle-seal <REVIEWED_V3_SEAL>
```

Real execution only after separate explicit human authorization; recomputes FIT
in v3 and repeats independent preflight before runtime creation or modeling:

```powershell
python -B reports/carla_train_full_v1/gate2_execution_bundle_v3/execute_gate2.py --bundle-seal <REVIEWED_V3_SEAL> --authorize-gate2 HUMAN_REVIEWED_EXPERIMENT_2B_GATE2_TRAIN_HEALTH_V3
```

Neither command resumes, repairs, cleans or reseals v1/v2. V3 retains immutable
transactional units, verified resume within v3, OS lease, no-replace publication,
preserved failed/pending attempts, and fail-closed finalization. Both PASS and
FAIL stop before any graph/GAT/TEST/final evaluation.

STOP FOR HUMAN REVIEW.
