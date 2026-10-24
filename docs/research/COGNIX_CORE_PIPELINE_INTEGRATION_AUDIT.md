# COGNIX core integration audit

Audit date: 2026-10-11. Interpreter: Python 3.12.10 at
`C:\Users\Aditya\AppData\Local\Programs\Python\Python312\python.exe`.
The base is Git HEAD `b5899cd53408b88f0bf4ee5b5c204d12188ae8d8`.
The original integration diff (22 additions / 15 removals) and pre-task source hash
are retained in `carla_runtime_calibration_audit.json` before further code changes.

## Every original changed block

| Block | Classification | Reason and effect |
| --- | --- | --- |
| Last-layer attention reporting | A: generic correctness; C: compatibility | A node refinement plugin can legitimately return no attention. Empty/None attention now means no communication. Supplied attention must be finite and N-by-N. Reported edges use positive off-diagonal weights instead of asserting every possible edge. Adjacency provenance is `provided_adjacency`. |
| Graph module method | B: metadata/provenance | Use the graph result's declared method, retaining `epistemic_gat` as the legacy default. This distinguishes independent refinement from attention without a domain-specific branch. |
| Collect agent UQ method metadata | B: metadata/provenance, with an unintended failure effect | Real uncertainty estimators need not be MC dropout. However, the original metadata call was inside the successful UQ exception boundary: a metadata exception or malformed metadata could replace a valid estimate with production heuristic UQ or fail research execution. |
| Report combined UQ methods | B: metadata/provenance | Join declared methods deterministically; preserve the existing confidence-heuristic fallback label and legacy default where metadata is absent. |

No D (CARLA leakage) or unnecessary domain-specific block was found. No scientific
formula was replaced. The metadata exception coupling was a real generic regression;
this task fixes it with `_optional_method`, used by UQ and graph status reporting.
Missing, malformed, empty or failing optional metadata retains the legacy method
label while preserving a successful numeric result. Required stage failures still
use the existing research exception and production fallback boundaries. No behavior
needed moving into a domain adapter.

## Fourteen audit answers

1. No CARLA-specific vocabulary or imports entered executable pipeline code.
2. No Camera/IMU/Seg identity, count, dimension or feature assumption entered the core.
3. No archive, split, dataset layout or replay behavior entered the core.
4. Public input/output signatures are unchanged. Communication statistics may now be
   null for independent node refinement; attention edges and method metadata are more accurate.
5. Valid graph numeric execution and sigmoid extraction are unchanged. Independent
   refinement without attention is newly supported. Malformed attention now fails
   validation through the existing failure boundary; this is an intentional generic
   compatibility/correctness change, not a graph mathematics change.
6. Fusion mathematics and reliability handling are unchanged.
7. Calibration mathematics, alpha 0.05 and input `[1-fused_confidence, fused_confidence]`
   are unchanged. The existing `calibrated_confidence` equals the original fused
   confidence; split conformal does not rescale that probability.
8. Risk and decision formulas/thresholds are unchanged.
9. Attribution mathematics and Monte Carlo permutation policy are unchanged.
10. Configured failure fallback policies are unchanged. This task removes the
    accidental dependency of numeric success on optional metadata success.
11. Research mode still raises on configured stage failures. An omitted optional
    calibrator remains unexecuted, as in the base; research mode never implied that
    an absent calibrator was fitted or that a partial chain was fully calibrated.
12. Production fallback outputs and failure statuses match the base on the tested
    UQ, graph, fusion, calibration and attribution failure paths.
13. Non-CARLA users benefit from independent refinement and accurate method reporting.
    Generic three-agent, Unicode and arbitrary-name fixtures remain supported.
14. The initial attention and metadata changes are useful generic support. The
    optional-metadata isolation fix is necessary to preserve prior numeric behavior.

AST comparison confirms that only `run` and `_estimate_uncertainties` differ from the
base class. Trust weighting, aggregation, fusion, calibration formulas, risk,
decision and attribution were reviewed; the scientific helper methods are AST
identical. Tests execute both base and current classes against the same reference
inputs and seeded test-only Shapley permutations. No production RNG policy changed.

## Collective conformal compatibility stop

The frozen TRAIN partition is SHA-256
`706c819ec568fd2dd2a1a89d9d7cc13eed7ba86466d8b5aa4ad4de277a863230`,
with 76 FIT and 25 CAL scenarios. All 76 FIT scenarios fitted the upstream ensembles.
All 25 CAL scenarios fitted each upstream `EnsemblePredictiveCalibrator` probability
mapping. The authenticated Gate2 specification explicitly assigns CAL to mapping
fitting, and its audit executor calls `agent.fit_calibrator(scores, labels)`.

| Agent | CAL parents already used | CAL clean rows | CAL pseudo rows | CAL combined rows |
| --- | ---: | ---: | ---: | ---: |
| Camera | 25 | 74,975 | 21,425 | 96,400 |
| IMU | 25 | 74,975 | 21,400 | 96,375 |
| Seg | 25 | 74,975 | 10,725 | 85,700 |

CAL labels here are constructed clean=1/pseudo=0 labels, not official TEST labels.
They correspond to synchronized ticks 1..2999, tick 0 excluded, with causal IMU
windows `[max(0,t-11),t]`; pseudo rows use the frozen modality-matched recipe,
severity and seed rules. This task inspected metadata/code only and did not decode
these feature/label arrays or generate new pseudo samples.

There is no untouched parent scenario in the frozen 101-scenario TRAIN catalogue.
Refitting split conformal on these same parents would not establish an independent
calibration source for the current frozen composite scorer. A retrospective split,
dropping the fitted probability mapping, selecting unused-looking ticks or replacing
the upstream artifacts would change the scientific contract and is outside this task.
Temporal dependence also requires an explicit scenario-aware coverage protocol;
frame count alone cannot establish exchangeability.

Historical final conformal bundles v1..v4 were inspected as code/metadata only, never
executed. They use the TEST archive, official normal/anomaly labels, older upstream
artifacts/checkpoints, benchmark mean pooling, class 0=normal / class 1=anomaly and
scenario-maximum nonconformity scores. They are incompatible with the requested
TRAIN-only runtime contract. Their scores/threshold outcomes were not imported.

The runtime contract snapshot and audit access ledger record verified upstream
references, NoGraph/101 epoch-97 checkpoint SHA-256
`9069f465e169c0e6e057957fe12953dae994ea2ae875c99f755051d4c57f9863`,
uniform reliability, runtime fusion and conformal semantics, source hashes, partition
identity and software versions. It is explicitly a blocked contract snapshot, not
an executable or sealed calibration package. Calibration generation never started;
no runtime calibration artifact, sample count, quantile or seal is claimed.

The TRAIN archive exists. Its current size/device/inode/mtime/ctime exactly match
authenticated historical identity; its historical full SHA-256 is
`6cf22ecf7d2910b45ed71127525834d1a4a26a65cc65918e7b5e27da95a38683`.
The 146 GB archive was not reopened or freshly rehashed. Protected report files were
hashed as opaque bytes for preservation; no official TEST label payload was decoded.

Following the user's stop conditions, collective calibration stays unavailable.
Every strategy requiring calibration continues to fail closed. No after-freeze real
TEST smoke was run because a valid calibration freeze could not occur. A separate
task must specify untouched TRAIN-only calibration data and its constructed-task,
scenario-aware protocol, or explicitly authorize a new runtime training/partition
chain outside Experiment 2B. CARLASim work remains blocked on that prerequisite.
## Validation result

Final selected suite: 63 passed (29 new core/calibration audit tests, 19 replay
integration tests, 5 dashboard tests, 10 existing full-pipeline integration tests),
with 14 existing Matplotlib/Pyparsing deprecation warnings. The first attempt had
52 passed / 11 failed: the regression fixture compared different Monte Carlo Shapley
permutations and used `sorted_ids` instead of the exact `sorted_canonical_ids` field.
Only the test fixtures were corrected; no probability, threshold or runtime RNG
behavior was tuned. Existing integration tests fit isolated temporary synthetic
fixtures; no research/runtime upstream, graph or conformal artifact was fitted.

Commands, all using the explicitly named installed Python 3.12 interpreter:

```cmd
C:\Users\Aditya\AppData\Local\Programs\Python\Python312\python.exe -X utf8 -B docs/research/scripts/audit_runtime_calibration.py baseline
C:\Users\Aditya\AppData\Local\Programs\Python\Python312\python.exe -X utf8 -B docs/research/scripts/audit_runtime_calibration.py capture_core
C:\Users\Aditya\AppData\Local\Programs\Python\Python312\python.exe -X utf8 -B docs/research/scripts/audit_runtime_calibration.py audit
C:\Users\Aditya\AppData\Local\Programs\Python\Python312\python.exe -X utf8 -B -m pytest tests/unit/test_runtime_calibration_audit.py tests/unit/test_replay_integration.py tests/unit/test_dashboard.py tests/integration/test_full_pipeline_integration.py -q -p no:cacheprovider --basetemp=C:/Users/Aditya/Cognix2_recovered/.runtime_calibration_work/pytest-1
C:\Users\Aditya\AppData\Local\Programs\Python\Python312\python.exe -X utf8 -B -m pytest tests/unit/test_runtime_calibration_audit.py tests/unit/test_replay_integration.py tests/unit/test_dashboard.py tests/integration/test_full_pipeline_integration.py -q -p no:cacheprovider --basetemp=C:/Users/Aditya/Cognix2_recovered/.runtime_calibration_work/pytest-2 --junitxml=docs/research/carla_runtime_calibration_tests.xml
C:\Users\Aditya\AppData\Local\Programs\Python\Python312\python.exe -X utf8 -B docs/research/scripts/audit_runtime_calibration.py cleanup
C:\Users\Aditya\AppData\Local\Programs\Python\Python312\python.exe -X utf8 -B docs/research/scripts/audit_runtime_calibration.py verify
git -c core.whitespace=cr-at-eol diff --check
```

The workspace fixture path was resolved and checked before removal. The command-scoped
CRLF whitespace setting preserves existing tracked line endings without changing Git
configuration. Full JUnit results and the separate validation/preservation/change
receipts are under docs/research, outside closed scientific bundles.
