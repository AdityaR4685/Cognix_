# Experiment 2B deterministic FULL-TRAIN partition: preparation only

This new bundle prepares the unchanged whole-scenario FIT/CAL split in the original
`full_train_preregistered_protocol.json`. It does not compute or freeze the real
partition. The Experiment 2A protocol and all v7/v8 evidence remain immutable.
The accepted clean store belongs to Experiment 2B under the separately sealed v8
segmentation compatibility amendment.

The completed extraction is `COMPLETE_GATE1` clean-store completion. The original
DATA / INTEGRITY gate still awaits the real partition. After a future authorized
freeze, the new runtime publishes `DATA_INTEGRITY_GATE_PASS` and **STOP FOR HUMAN
REVIEW**. Gate 2 fitting requires separate human authorization after partition
review. Any later scientific repair requires separately registered Experiment 2C
or later; it cannot be silently folded into 2B.

## Frozen split

Collect all 101 accepted complete canonical TRAIN IDs. Sort using unchanged Python
string ordering. Instantiate `numpy.random.Generator(numpy.random.PCG64(2026))`
exactly once and call its permutation exactly once. Compute
`n_cal = max(1, round(0.25 * N))` with Python ties-to-even rounding: 25 CAL and
76 FIT. The first 25 permuted IDs are ordered CAL_NORMAL; the rest are ordered
FIT_NORMAL. No historical N20 membership, seed search, manual edits, balancing,
feature/OOV values, weather values or model outcomes enter this calculation.

## Real read-only preflight

From the repository root, substitute the reviewed SHA256 from this bundle's
`SHA256SUMS.sha256` for `<REVIEWED_BUNDLE_SEAL>`:

```powershell
python -B reports/carla_train_full_v1/partition_freeze_bundle_v1/partition_preflight.py --bundle-seal <REVIEWED_BUNDLE_SEAL>
```

This command writes no files, performs no partition computation while the target
is absent, and independently hashes the complete local 146,453,559,283-byte TRAIN
archive. It authenticates HEAD, v8/v7/v6 seals and source bindings; independently
reads every block/adoption and all original adopted arrays; validates exact
canonical IDs, ordinals, dimensions, scientific bindings, 2999 rows per scenario,
302899 total rows, global inventory/store digests, all 303000 OOV frame records,
COMPLETE_GATE1, original protocol copies and preserved historical evidence.
Python bytecode writes are disabled. Network, TEST and writes are blocked by an
audit guard. A mismatch stops for human review without repair or overwrite.

## Future authorized execution

Only after explicit human review/authorization:

```powershell
python -B reports/carla_train_full_v1/partition_freeze_bundle_v1/partition_freeze.py --bundle-seal <REVIEWED_BUNDLE_SEAL> --authorize-partition-freeze HUMAN_REVIEWED_EXPERIMENT_2B_PARTITION_FREEZE_V1
```

The token is mandatory and is checked before real admission. The future freeze
repeats all verification before computation and writes only to the new
`reports/carla_train_full_v1/partition_freeze_v1/` runtime. Publication uses an
exclusive pending directory, exclusive file creation, file fsyncs and Windows
`os.rename`, which refuses an existing target. It independently verifies before
and after publication. Pending/partial evidence is preserved on failure and
blocks retry pending human review. No v8 runtime state is updated.

The authoritative `partition.json` uses exact UTF-8 JSON bytes with
`sort_keys=True`, `separators=(',', ':')`, `allow_nan=False`, and no trailing
newline. `partition.json.sha256` is the detached hash. The runtime also contains
an exact copy of the sealed v8 member inventory, `member_evidence.jsonl.gz`,
which retains every source/member SHA256 and scenario record with TAR offsets.
The partition JSON binds that file by compressed/logical digest and each
scenario's exact inventory-record/member-set digest. It binds all 101 original
block/adoption hashes, v8 completion/state/seal, OOV ledger, scientific sources,
preprocessing policy, frozen Seg29 function, original split text and NumPy version.

Town counts are descriptive and are calculated only after membership. Weather
metadata column names, row counts and source hashes are retained; values absent
from sealed inventory/store are explicitly null with a reason. Additional
environment descriptors are explicitly null. No source metadata replay or
feature extraction is performed by this step.

An existing runtime is verified byte-for-byte against the canonical expected
record, including ordered membership and all bindings, and semantically through
the authenticated upstream store. Its original validated UTC creation timestamp
is reused for comparison. It is never overwritten. Extra files, missing files,
changed membership, bindings or detached hashes stop for human review.

## Preparation evidence

`execution_bindings.json` freezes exact inputs without a real split.
`historical_preservation_baseline.json` and the final audit bind exact file and
directory inventories, hashes, sizes and modification times for all historical
recovery bundles and both v7/v8 runtimes, including failure/raw evidence.
`synthetic_test_results.json` and `test_transcript.txt` record tests using only
synthetic data. Original protocol copies are byte-identical. `SHA256SUMS` lists
every bundle file; `SHA256SUMS.sha256` is the detached preparation seal. The audit
references the detached seal to avoid self-referential hashing; the preparation
tool prints the actual seal and exact future commands after sealing.
Prior synthetic-run logs are retained. The initial fixture harness was corrected
for Windows temporary-directory ACLs and a missing synthetic OOV field; the final
suite records the exact Python hashes and must pass with zero failures/errors.

The preparation lifecycle is `prepare_bundle.py initialize`, `tests`, then `seal`.
Each mode refuses replacement of its records. Do not rerun a completed mode or
edit this sealed bundle. A change requires a separately reviewed bundle version.
No fitting, pseudo anomalies, calibration, graph construction, GAT training,
TEST, network, commit or push is authorized here.

**STOP FOR HUMAN REVIEW.**
