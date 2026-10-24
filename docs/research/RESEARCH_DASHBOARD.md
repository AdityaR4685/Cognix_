# COGNIX research replay dashboard

The dashboard consumes generic observation envelopes through the ordinary DecisionEngine
and displays DecisionResult. Domain parsing and fitted artifact restoration belong to
adapters; dashboard/sources.py is the application composition boundary. The generic
payload builder and replay controller support arbitrary agent identifiers and counts.

## Implemented modes

- **SYNTHETIC:** the existing six-agent synthetic demonstration. Feature generation uses
  seed 42; probability-noise generators have explicit per-agent seeds. Uncertainty is a
  synthetic simulation, not measured sensor model uncertainty. Synthetic artifacts and
  illustrative persisted benchmark references remain specific to this mode.
- **DATASET:** static CarlAnomaly Base archive replay, using real archived RGB/segmentation
  images and IMU measurements. CarlAnomaly was recorded from CARLA simulation; these
  archive observations are not physical road measurements or a running CARLA server.
  Only Camera, IMU, Seg are active. Real models and genuine graph refinement run through
  DecisionEngine; this engineering demonstration computes no scientific evaluation suite.
- **SIMULATION:** future synchronized CARLA server callbacks. Activation is rejected.
- **LIVE:** future physical sensors. Activation is rejected.

PLAYING/PAUSED describe replay state; CONNECTED/OFFLINE describe browser connectivity.
Neither connection nor periodic updates turn DATASET replay into LIVE or SIMULATION.
The old simulation/carla/adapter.py GO/BRAKE skeleton stays untouched and is not used.

## Scientific availability

Real upstream artifacts are restored from sealed Gate2 TRAIN health v3 state. Their
ensemble predictive score calibrators map normality statistics to constructed-task
probabilities. They are not collective conformal calibrators.

The three frozen graph strategies (NoGraph node MLP, StandardGAT, EpistemicGAT) and all
five frozen seeds are configurable. NoGraph/101 is the predeclared demonstration default.
NoGraph performs independent node refinement; GATs execute genuine attention. These
configurations are not ranked or selected using TEST performance.

The ordinary runtime uses epistemic-weighted fusion with explicit uniform reliability
assumptions, unlike the sealed benchmark's mean pooling. No conformal artifact is bound
to this collective chain: DATASET displays **calibration unavailable**, a null prediction
set and null calibrated confidence. A mode request with require_calibration=true fails
closed. No synthetic scores or experiment corruption thresholds are reused.

The UI distinguishes upstream predictions from node-refined predictions and labels the
collective output **Constructed-task normality probability**. q_normal denotes
P(target=normal) under the TRAIN-derived clean-vs-pseudo task; p_corruption=1-q_normal.
It is not a driving safety probability or official anomaly posterior. Risk/decision
outputs are framework demonstration policies, not physical vehicle commands. Shapley
values describe the engine's collective epistemic uncertainty coalition function;
they do not rerun graph inference for every coalition.

## Run locally (Windows cmd.exe)

This machine's primary Python 3.14 has model libraries but lacks FastAPI/uvicorn/pytest.
The existing Python 3.12 installation includes those dependencies; no packages were
installed for this task. From the repository root:

```bat
set CARLANOMALY_TRAIN_ARCHIVE=E:\carlanomaly-base-train.tar.gz
set CARLANOMALY_TEST_ARCHIVE=E:\carlanomaly-base-test.tar.gz
C:\Users\Aditya\AppData\Local\Programs\Python\Python312\python.exe -X utf8 -B -m uvicorn dashboard.app:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. Choose DATASET, split, an archive path (or configured
environment variable), strategy and seed; apply the source, select a scenario, then
use Next or Play. Pause, Previous, Restart and 0.1–10 samples/s rate are supported.
There is one shared server replay session across browsers. Source changes clear history.
While a sample is processing, reconfiguration returns 409; Pause stops subsequent
samples after current bounded preparation completes.

## API

| Endpoint | Operation |
|---|---|
| GET /api/source/status | Mode, source, selected sequence/index, rate, processing/error state, artifact configuration |
| POST /api/source/mode | mode; optional split, archive, graph_strategy, seed, max_samples, require_calibration |
| GET /api/dataset/scenarios | Adapter-provided scenario catalog |
| POST /api/dataset/select | scenario identifier |
| POST /api/dataset/play, /pause, /restart | Replay state controls |
| POST /api/dataset/step | delta: 1 (next), -1 (previous), 0 (current) |
| POST /api/dataset/rate | rate in samples/s |
| GET /api/latest, /history, /status | Backward-compatible result/history/status APIs |
| POST /api/set_scenario | Backward-compatible alias for selection |
| /ws | Result/error updates while playing |

Missing archives, missing sensors, modified bound code/state, incompatible graph shapes,
unknown configurations and unbound collective calibration requests return explicit
errors. DATASET failures never display a synthetic fallback. A paused DATASET with no
sample returns 409 from /api/latest until Next or Play is used.

## Bounded I/O and provenance

The adapter streams gzip tar members without extractall or any archive extraction to
disk. It decodes only required sensor members, retaining compact features for the
first max_samples of one scenario (default 300, maximum 3000). TEST has 299 valid ticks
after tick-0 warm-up. TRAIN replay is similarly bounded. Previous/restart reuse that
cache; moving to another scenario may require scanning gzip again. Sensor membership,
image/member size and causal IMU windows are validated; absent modalities are never
zero-filled. Current archive size/stat is checked against authenticated historical
inventory. Full 90/146 GB archive hashes are not freshly recomputed; observed required
members are freshly hashed. The UI preserves this distinction in provenance.

No official label or anomaly-front payload is opened. Envelopes keep agent_inputs
separate from display metadata. RuntimeBridge gives DecisionEngine only agent_inputs,
and validates agents before any partial result can be displayed. Optional ground-truth
annotations can be attached to a completed DecisionResult by build_payload; the current
CarlAnomaly dashboard leaves them disabled. Tests prove annotations cannot alter inference.

Payloads preserve module_status, failures, upstream/refined predictions, UQ, trust,
graph strategy/communication, fusion, risk, escalation/abstention and attribution.
They include source mode/domain/split/scenario/sample, archive and member identity,
artifact/source hashes, seed, runtime session and label_visible_to_inference=false.
Absent calibration/metrics are null. Source I/O/feature preparation and agent computation
are timed separately from engine stage timings and total runtime. Cached agent interfaces
prevent UI reads from recomputing stochastic uncertainty. Engine UQ timings consequently
measure cache retrieval; agent_inference_ms measures the real prediction/UQ computation.

See CARLA_DATASET_PIPELINE_INTEGRATION.md for design, compatibility, validation and the
separate future task needed to prepare collective conformal artifacts from TRAIN only.
The core is designed to remain domain-neutral and tested with both real-format CARLA
and non-CARLA toy sources; these checks do not establish universal domain generalization.

## TRAIN collective calibration follow-up

The 2026-10-11 compatibility audit stopped calibration preparation before fitting:
all 25 frozen CAL scenarios already fitted the upstream probability mappings, while
the remaining 76 FIT scenarios fitted the ensembles. No untouched parent scenario
remains in that frozen TRAIN catalogue. Historical final conformal packages are
TEST-derived and use a different scorer/score contract, so they are not bound here.
See COGNIX_CORE_PIPELINE_INTEGRATION_AUDIT.md and the separate
carla_runtime_calibration_audit.json evidence.

DATASET collective calibration therefore remains unavailable. No graph configuration
can reuse a calibrator from another strategy or seed; requests requiring calibration
still fail closed. There is no new calibration artifact ID, n_cal, quantile or seal to
display. Genuine calibrated DATASET support awaits a separately specified independent
TRAIN calibration source and matching artifact verification.

The existing split-conformal method constructs prediction sets without rescaling the
fused probability. When a compatible artifact is eventually bound, its coverage target
must be labeled as a prediction-set target; a copied `calibrated_confidence` scalar is
the unchanged original fused confidence, not a newly calibrated probability magnitude.
