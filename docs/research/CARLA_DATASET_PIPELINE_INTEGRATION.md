# CarlAnomaly replay integration

Engineering design recorded before runtime implementation, 2026-10-11.

## Audit and preservation

The dashboard previously assembled six synthetic sensor inputs, invoked DecisionEngine,
and built its payload in the same function. DecisionEngine already accepts an agent-keyed
input dictionary. Its generic pipeline supplies uncertainty, trust, graph refinement,
fusion, calibration when configured, risk, decision and uncertainty Shapley attribution.
No core rewrite is needed. A narrow generic fix permits node refinement plugins that
return no attention, reporting no communication rather than indexing an empty list.
It also takes refinement and upstream uncertainty method names from generic plugin
metadata, preserving legacy names when plugins omit them. Bootstrap ensemble uncertainty
must not be reported as MC dropout. This missing contract was
found by the first bounded archive smoke check. The agent registry stores interfaces, not fitted model state.
Synthetic persistence under artifacts/synthetic is a separate six-agent chain.

All reports are read-only. The preservation receipt records the initial Git state,
hashes of existing reporting edits, and a content/size/mtime/directory inventory digest
of every report (18,036 files). No scientific executor or result-scoring script is run.

## Boundaries and contracts

`cognix.runtime.replay` will define immutable ObservationEnvelope, a ReplaySource protocol,
and ReplayController. Envelopes carry source/domain/sequence/sample identity, optional
timestamp, per-agent inputs, metadata and provenance. They contain no sensor-specific
fields. Sources expose sequence listing, sample count and indexed reads. Controllers
own selection, play/pause, step/previous, restart and a bounded playback rate.

`cognix.adapters.carla.replay_source` owns archive paths, member names, RGB/segmentation
decoding, causal IMU windows and feature extraction. It emits Camera, IMU, Seg in that
order, with dimensions 18, 10, 29; tick 0 is synchronization only. It uses existing
real_features functions and the authenticated existing segmentation sanitization.
It streams gzip tar files without extracting them, retaining only compact features
for one bounded replay window. Seeking a different scenario may rescan gzip. Archive
paths are configuration/environment inputs, never an engine concern.

`cognix.runtime.bridge` validates the active agent contract, evaluates each agent once,
and gives cached PredictionResult/UncertaintyResult interfaces to the ordinary
DecisionEngine. It forwards only agent_inputs, never envelope metadata or labels.
`dashboard.payload` transforms DecisionResult generically, supporting arbitrary names
and counts. Application assembly selects an adapter; UI consumes that generic payload.

## Artifact compatibility findings

Gate2 TRAIN health v3 identifies valid five-member Mahalanobis ensembles and fitted
ensemble predictive score calibrators for Camera/IMU/Seg. These upstream calibrators
are not collective conformal calibrators. Restore their existing NPZ state with source
hashes, committed unit seals, finite/dimension validation and no fit calls.

The checkpoint freeze retains NoGraph, StandardGAT and EpistemicGAT for seeds
101/202/303/404/505. Default demonstration configuration: NoGraph, seed 101.
Configuration exposes strategy and seed; no TEST results inform that choice.
Authenticate the checkpoint hash and identity against the frozen bindings.
The sealed standalone graph definitions differ from current graph_training_models
parameter naming. Reuse their authenticated class ASTs, with a GraphRefinement wrapper
exposing per-node logits and actual attention. NoGraph executes its frozen node MLP
without inter-agent communication. GATs genuinely execute their frozen layers.
Feature columns remain prob_normal, epistemic, aleatoric; priors use original float64
epistemic values and frozen float32 conversion rules. Reject any schema mismatch.

Ordinary runtime epistemic-weighted fusion is different from the sealed benchmark's
mean pooling. No collective conformal artifact is bound to this runtime chain.
Do not reuse synthetic conformal scores or the experiment's corruption thresholds.
DATASET exposes a supported partial chain with calibration unavailable, null set,
null calibrated confidence and a stated reason. A request requiring a fully calibrated
chain must fail closed. Preparing TRAIN-only collective conformal artifacts with an
explicit frozen runtime contract is a separate future task, not part of this work.
Reliabilities are explicit uniform demonstration assumptions, not fitted health scores.
Risk/ACT/ESCALATE outputs are framework policy demonstrations, not vehicle commands.
Attribution uses the engine's collective epistemic uncertainty coalition function;
it does not recompute graph/fusion for every coalition.

## Label isolation and semantics

Only RGB-front, segmentation-front and IMU payloads are inference inputs. Never open
anomaly-front or official label files for this integration. Optional annotations live
outside envelopes and are attached by the payload builder after a DecisionResult exists.
Tests change annotations and assert that inference output does not change.
Dataset member paths are display provenance only, never engine context.
q_normal means P(target=normal) under the TRAIN-derived clean-vs-pseudo task;
p_corruption = 1-q_normal. The UI uses constructed-task normality terminology.

## Controls, provenance and errors

SYNTHETIC remains the existing six-agent demonstration. DATASET always means real static
CarlAnomaly replay. SIMULATION and LIVE remain future modes and reject activation.
Source status, mode configuration, sequence selection, play, pause, next/previous,
restart and rate endpoints feed a small UI control strip. New mode/scenario selection
clears stale history. Sources and models must load successfully before inference;
errors stay explicit, without any synthetic fallback presented as DATASET.

Provenance includes mode/domain/source, dataset split, sequence/sample, archive stat
identity, observed member hashes, artifact/checkpoint/source hashes, strategy/seed,
runtime session identity, fusion/reliability assumptions and label_visible_to_inference=false.
Previously measured archive hashes are identified as historical evidence rather than
claimed as freshly rehashed. Source I/O/feature time is separate from measured runtime
time; unavailable modules and metrics are null. No scientific metric suite is computed.

## Verification and limitations

Focused tests cover a non-CARLA toy source, arbitrary three/six agent sets, immutable
envelopes, navigation, label isolation, module status preservation, archive errors,
schema/checkpoint rejection, genuine tiny tar replay, API behavior and AST-based core
import independence. Retain existing synthetic dashboard tests and relevant feature,
loader, agent and full-pipeline integration tests. Test fixture fitting, if exercised,
is isolated synthetic testing; there is no fitting of dataset runtime artifacts.
Run at least one bounded real archive replay, inspect payload and verify preservation.

The interface is designed to remain domain-neutral and is exercised with CARLA-specific
and generic test sources; this is interface testing, not proof of domain generalization.
A future synchronized simulator source can emit the same envelope into the same bridge.
simulation/carla/adapter.py stays untouched: its fixed GO/BRAKE skeleton is not a
completed simulator integration. Gzip seeking can be slow; replay is not real-time sensor
operation. Collective conformal calibration is the explicit remaining artifact blocker.

## Completion and validation

The implementation is complete for the supported partial DATASET chain. Both real
archives are present locally; the demonstration reads the TEST archive without opening
official labels or anomaly masks. The first catalog sequence is
`test/anomaly/Town01/change-weather/scenario-1`. A bounded two-sample replay executed all
three frozen strategies with predeclared seed 101 through the ordinary DecisionEngine:

| Strategy | Fused confidence, first sample | Refinement | Communication | Decision | Collective calibration |
| --- | ---: | --- | --- | --- | --- |
| NoGraph | 0.5983577270747344 | Executed | None | REQUEST_INFORMATION | Unavailable |
| StandardGAT | 0.6733284835738085 | Executed | Executed | REQUEST_INFORMATION | Unavailable |
| EpistemicGAT | 0.6733159281568293 | Executed | Executed | REQUEST_INFORMATION | Unavailable |

These are engineering smoke outputs, not benchmark metrics or evidence for selecting a
strategy. Each cycle used the three real agents, frozen bootstrap ensemble uncertainty,
ordinary runtime fusion and genuine uncertainty Shapley attribution. Cached next/previous
navigation returned ticks 2/1. The API smoke exercised configuration, catalog, selection,
play/pause, next/previous, restart, rate, latest and history. Browser inspection verified
the three-agent table, upstream/refined outputs, actual module statuses, unavailable
calibration and replay controls; `carla_dataset_dashboard.jpg` records the result.

The following commands ran successfully with Python 3.12 from the repository root:

```cmd
C:\Users\Aditya\AppData\Local\Programs\Python\Python312\python.exe -X utf8 -B -m pytest tests/unit/test_carlanomaly_loader.py tests/unit/test_carla_real_agents.py -q -p no:cacheprovider --basetemp=C:/Users/Aditya/Cognix2_recovered/.integration_work/pytest-existing-1
C:\Users\Aditya\AppData\Local\Programs\Python\Python312\python.exe -X utf8 -B -m pytest tests/unit/test_replay_integration.py tests/unit/test_dashboard.py tests/unit/test_carla_real_features.py tests/test_carla_adapter.py tests/integration/test_pipeline.py tests/integration/test_full_pipeline_integration.py -q -p no:cacheprovider --basetemp=C:/Users/Aditya/Cognix2_recovered/.integration_work/pytest-final-1
C:\Users\Aditya\AppData\Local\Programs\Python\Python312\python.exe -X utf8 -B -m pytest tests/unit/test_replay_integration.py tests/unit/test_dashboard.py tests/integration/test_full_pipeline_integration.py -q -p no:cacheprovider --basetemp=C:/Users/Aditya/Cognix2_recovered/.integration_work/pytest-module-metadata-1
C:\Users\Aditya\AppData\Local\Programs\Python\Python312\python.exe -X utf8 -B dashboard/replay_smoke.py --archive E:/carlanomaly-base-test.tar.gz --max-samples 2 --output docs/research/carla_dataset_replay_validation.json
C:\Users\Aditya\AppData\Local\Programs\Python\Python312\python.exe -X utf8 -B dashboard/test_live.py --archive E:/carlanomaly-base-test.tar.gz --method StandardGAT --output docs/research/carla_dataset_http_validation.json
git -c core.whitespace=cr-at-eol diff --check
```

The first two test commands passed 108 and 73 tests respectively (181 distinct selected
tests). The final upstream-method metadata change passed a further 34-test recheck.
Existing test helpers fit and calibrate isolated temporary synthetic fixtures; no
research or dataset runtime artifact was fitted, trained, tuned or recalibrated.
Fourteen existing Matplotlib/Pyparsing deprecation warnings appeared in the larger
suites. Initial default pytest temporary-directory and sandbox socket attempts failed
because of environment permissions; explicit workspace test directories and approved
loopback execution resolved those limitations. No package installation was needed.

The tracked HTML already uses CRLF. Plain `git diff --check` treats those existing line
endings as whitespace, so the successful check uses command-scoped `cr-at-eol`; no Git
configuration or whole-file line-ending conversion was made. Task test directories are
removed only after resolving and checking their absolute paths against the workspace.

The preservation receipt verifies unchanged contents, sizes, mtimes and directory
membership for all 18,036 report files, totaling 5,914,657,424 bytes in 634 directories.
Its before/after aggregate SHA-256 is
`fca2deb52752ddfdefb11a2fd26dab50d9e3a5cc83ceb6fa83ebca4bfe97cc5f`.
All pre-existing modified reporting files and untracked reporting artifacts remain
byte-identical. The change manifest identifies this task's five modified tracked files
and sixteen new files separately from the eight previously modified tracked files and
15,426 previously untracked files. Nothing is staged.

No Phase A/Phase B execution, TEST scoring, new experiment metric, research training,
runtime recalibration, threshold tuning or TEST-dependent model selection occurred.
No protected evidence or seal was rewritten. No Git staging, commit, push, pull, reset,
restore, stash or clean operation occurred. The next separately specified artifact task
would define and prepare TRAIN-only collective conformal calibration for the frozen
runtime graph/fusion contract, then bind and validate it before claiming a calibrated
chain. This integration deliberately reports that missing module as unavailable.

## Follow-up core and TRAIN calibration audit, 2026-10-11

The previous integration validation above remains its original engineering record.
This follow-up has separate evidence: `COGNIX_CORE_PIPELINE_INTEGRATION_AUDIT.md`,
`carla_runtime_calibration_audit.json`, `carla_runtime_calibration_contract.json`,
`carla_runtime_calibration_validation.json` and `carla_runtime_calibration_preservation.json`.
Previous preservation, change, screenshot and replay receipts are not rewritten.

The core diff contains generic attention compatibility and method provenance changes,
with no CARLA vocabulary/import/agent schema. Optional method metadata initially sat
inside the numeric UQ exception boundary. `_optional_method` now isolates absent or
failing metadata from successful numeric stages, preserving the base research and
production failure behavior. Scientific helper methods remain AST-identical to the
base, and reference-output comparisons include configured failures and attribution.

The recorded runtime configuration is the existing NoGraph/101 demonstration default,
Camera/IMU/Seg dimensions 18/10/29, frozen five-member ensembles and their fitted
probability mappings, node features `[prob_normal, epistemic, aleatoric]`, float32
node execution and float64 upstream epistemic values. The epoch-97 checkpoint is
SHA-256 `9069f465e169c0e6e057957fe12953dae994ea2ae875c99f755051d4c57f9863`.
Critical source hashes, unit seals, software versions and archive fingerprint are
bound in the contract snapshot. Uniform reliabilities remain explicit assumptions;
no compatible fitted runtime reliability artifact was identified.

Fusion remains the existing EpistemicWeightedFusion with epsilon 1e-6:
`raw_i = 1/(e_i+epsilon)`, `w_i = raw_i/(sum(raw)+epsilon)`, then
`fused_confidence = clip(sum(w_i*refined_p_i),0,1)`. This is different from benchmark
mean pooling. The existing ConformalPredictor receives exactly float64
`[[1-fused_confidence,fused_confidence]]`, class order constructed pseudo/non-normal=0,
constructed normal=1; nonconformity is `1-P(true_class)`. Alpha is fixed at 0.05,
with the augmented finite-sample rank and inclusive comparison already implemented.
The method can produce empty sets. Its `calibrated_confidence` field preserves the
original fused confidence; it does not transform the probability magnitude.

Calibration preparation stopped before fitting. Authenticated partition
`706c819ec568fd2dd2a1a89d9d7cc13eed7ba86466d8b5aa4ad4de277a863230`
has 76 FIT and 25 CAL parents. FIT fitted the upstream ensembles; every CAL parent
fitted each upstream probability mapping, including 74,975 clean rows per modality
and frozen modality-matched pseudo rows. No independent scenario remains in the
101-parent TRAIN catalogue. Constructed labels are clean=1/pseudo=0, with ticks
1..2999 and causal windows; actual CAL feature/label arrays were not decoded here.
Reusing those parents cannot establish the independent split-conformal source
required by this frozen composite scorer. A new retrospective partition or model
refit would require separately specified scientific work.

Historical final conformal bundles v1..v4 are TEST-derived, use official labels and
older scorer artifacts, different class ordering, benchmark pooling and scenario-max
scores. They were audited as metadata/code, not executed or bound. The artifact
identity, n_cal, fitted quantile and seal for this runtime remain null. No package
under artifacts/carla_runtime was created or sealed. All graph configurations continue
to reject requests requiring an unavailable calibrator; no mismatched calibrator is
loaded. A future loader must bind strategy/seed/checkpoint, upstream order/dimensions,
fusion/reliability, class ordering, partition and source hashes, then independently
verify the manifest and finite conformal state before enabling calibration.

Tests compare offline score construction with the actual runtime calibrator input,
exercise genuine conformal execution on fixed synthetic reference state, and retain
post-inference label isolation across the conformal result. That reference state is
not a CARLA calibration artifact. No calibrated real TEST smoke occurred because
the required TRAIN-only artifact could not be validly frozen. The next prerequisite
is a separately specified untouched TRAIN calibration source and scenario-aware
constructed-task protocol, or a separately authorized new runtime training chain.
