# Experiment 2B graph-development partition freeze only

This new bundle prepares and verifies the immutable 61 GRAPH_TRAIN / 15 GRAPH_VAL
partition derived solely from the already frozen 76 FIT_NORMAL scenario IDs.
The original 76/25 FIT/CAL partition is read, authenticated and never recomputed.
The 25 CAL_NORMAL scenarios are excluded. TEST, network, model execution, graph
artifacts, optimization, dependency installation, commit and push are prohibited.

The frozen rule is Python lexical sort of the 76 IDs, exactly one
`numpy.random.Generator(numpy.random.PCG64(2027))`, exactly one permutation, and
`max(1, round(0.20 * 76)) == 15` using Python built-in round. The first 15 permuted
IDs are ordered GRAPH_VAL; the remaining 61 are ordered GRAPH_TRAIN. No retry,
balancing, stratification, town balancing, manual edit, seed search or performance
selection is allowed. The split remains fixed even if later results are poor.

The later specification is declarations only: node order Camera, IMU, Seg;
feature order prob_normal, epistemic, aleatoric; conditions NoGraph, StandardGAT,
EpistemicGAT; prior `w_j = 1 / (1 + E_j)` and
`e_ij^epi = e_ij - log(1 + E_j)`. No later graph execution is authorized here.

Independent upstream verification hashes complete sealed payloads and checks
HEAD, original partition SHA/runtime seal, Gate-1 v8, Gate-2 v3 bundle/runtime,
FINAL.json hash and PASS/VALID/zero counters/false graph flags. Fitted arrays,
audit metrics and performance summaries are never semantically parsed. Original
TRAIN archive identity uses authenticated Gate-1 full-hash evidence plus current
matching local metadata; the raw archive is never opened/rehashed in this task.

The preservation baseline covers every pre-existing file and directory under
`reports/carla_train_full_v1/`, including historical/failure evidence, with exact
SHA256, size, modification/creation times and device/inode. Read access times
are excluded because reads can change them. Git HEAD/tracked/staged state is
also checked. Only the two new namespaces and the exclusive pending publication
namespace are excluded. An existing output or pending path always fails closed.
The retained first initialization attempt stopped on a new verifier's Windows
path/string directory-order comparison. Its baseline and failure record remain
in this bundle. The corrected verifier uses the original lexical string order;
no membership was computed in either initialization phase.

Preparation order is `prepare_bundle.py initialize`, `tests`, source/procedure
review, then `seal`. Tests use synthetic IDs and direct simulated guard probes;
they make no TEST/network/model requests. `procedure_review.json` binds all tested
source hashes and explicitly records Codex source review, with human review
pending. The user's current explicit request authorizes the partition freeze.
After sealing this bundle, the only freeze procedure is:

```powershell
python -B reports/carla_train_full_v1/graph_partition_freeze_bundle_v1/freeze_partition.py --bundle-seal <PREPARATION_SEAL> --scope partition-only
```

The procedure verifies this reviewed/tested seal, all upstream bindings and
preservation before one membership selection. It exclusively creates a pending
namespace, writes canonical JSON, proves disjointness/counts/union/CAL exclusion,
and launches a separate read-only deterministic replay. Replay uses one identical
fixed generator/permutation for verification; it cannot publish alternative
membership. The procedure rechecks preservation, seals the new pending namespace,
and uses Windows no-overwrite rename. Partial evidence is preserved and blocks
retry. All new sealed files receive Windows read-only attributes. Cryptographic
seals must still be checked because a filesystem owner can change attributes.

The immutable namespace contains `partition.json`, its detached SHA256, exact
ordered input/permutation/roles, RNG initial/final state, upstream bindings,
Python/NumPy identity, replay evidence, methodology and preservation audit.
`artifact_inventory.json` excludes itself and seal metadata to avoid circular
hashing. `SHA256SUMS` covers every payload and the inventory; its detached
SHA256SUMS.sha256 is the namespace seal. Verifiers reject unexpected/missing
files and directories, corrupt content and externally supplied seal mismatches.

Future verification is read-only and must receive the three actual frozen values:

```powershell
python -B reports/carla_train_full_v1/graph_partition_freeze_bundle_v1/verify_partition.py --bundle-seal <PREPARATION_SEAL> --partition-seal <PARTITION_SEAL> --partition-sha256 <PARTITION_JSON_SHA256>
```

It writes no files, revalidates all upstream/historical hashes and exact bindings,
and verifies another independent replay. Bytecode writes are disabled. Audit
guards deny TEST/network/raw-archive opens, model imports, unrelated subprocesses
and writes outside new outputs (all writes in verification). No earlier runner
or original partition algorithm is imported or executed.

**STOP FOR HUMAN REVIEW after this partition freeze.**
