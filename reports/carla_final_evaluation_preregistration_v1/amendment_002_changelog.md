# Amendment 002 — final pre-prediction conformal / partition protocol

2026-10-02, Asia/Calcutta. Explicitly authorized design amendment only. No replacement choice uses model performance, TEST labels, predictions or fitted set sizes.

All 57 prior files, including Amendment 001, the original preregistration archive and both seal files, are archived byte-for-byte in `amendment_history/pre_amendment_002/`. `amendment_record.json` remains Amendment 001; `amendment_002_record.json` records this amendment. Both sealed audits remain unchanged. Their earlier baseline hashes refer to the archived protocol bytes, not this amended protocol; do not regenerate historical audits.

| Component | Previous design | Amendment 002 |
|---|---|---|
| Allocation | town × directory condition × anomaly type; n_CAL,h=ceil(N_h/5) | Uniform complete-scenario split; sorted complete eligible IDs; exactly one PCG64(2028) permutation; n_cal=ceil(N_eligible/5); first n_cal to calibration, remainder to evaluation |
| Exclusions | Two known historical full scenario IDs | Both retained; any future independently authenticated exposure documented and its exclusion sealed before partition execution |
| Conformal | One block score per scenario; Q_h by stratum; Q=max_h Q_h | Same scorer and block maximum, pooled scenario scores; augmented rank k=ceil((n_cal+1)·.95); inclusive comparison; one global Q per frozen method/seed |
| Guarantee | Stratum exchangeability and global envelope | Marginal scenario-level finite-catalogue randomization/exchangeability conditional on frozen scorer, complete eligible inventory, uniform assignment and exclusions fixed before randomization |
| Subgroups | Assignment strata and descriptive diagnostics | Preserve town/condition/type in inventory and evaluation ledger; composition, metrics, empirical coverage/set size by subgroup are descriptive only; sparse results null or qualified |
| Inventory | Complete exact public list unavailable | Mechanically bounded future structure-only header traversal, byte-transit disclosure, exact access ledger and fail-closed gates; no extractor implemented or executed |
| Science and metrics | Frozen | Preserve all scientific components and performance metrics, except necessary conformal stratification field removal/rewording |

## Established reasons

The original design stratified by **town × directory condition × anomaly type**, with **n_CAL,h=ceil(N_h/5)** and **Q=max_h Q_h**. At alpha=.05, finite Q_h requires at least **19** calibration scenarios, hence N_h≥91 under that quota. Publisher totals establish **107 normal TEST scenarios across six towns**. Two strata reaching 91 would require at least 182 normal scenarios, so at least five normal-town strata necessarily fail finite rank. Thus at least one Q_h=+inf, Q=+inf and every conformal set is **{normal, anomaly}**. This follows from metadata/count mathematics alone, not model outcomes.

The sealed public audit found **no recoverable released-scenario → seed/route/base-drive/family mapping**. `scenario-N` has no supported family semantics. Exact family partitioning is unavailable from public information. Family/route dependence remains an external-generalization limitation.

## Scope and unchanged components

Retain s_j,t(y)=1−P_j,t(y), S_j=max_t s_j,t(Y_j,t), every frozen eligible synchronized tick and exactly one calibration score per scenario. Pool all calibration scenario scores and add +infinity for the standard rank. Fit 15 separate global Q values using identical membership, with no prediction ensemble. Remove town/condition/type cutoffs, Q_h and Q=max_h Q_h.

Uniform assignment and a uniformly selected held-out catalogue scenario give symmetric score positions conditional on their union. The rank statement is marginal over assignment and held-out selection; S_new≤Q implies all eligible true tick labels for that one scenario are included. It is not conditional on the realized seed/partition or each calibration sample. A fixed PCG64 seed supplies reproducibility, not population exchangeability.

No independent-family coverage, new-route population coverage, iid drive generation, arbitrary future CARLA coverage, live-distribution coverage, physical-safety certification, conditional town/type/condition coverage, simultaneous coverage of every evaluation scenario, or joint coverage across all 15 scorers is claimed.

Upstream M1, graph architecture, checkpoints, graph readout/fusion, graph seeds, probability calibration, node features, pseudo recipes, threshold selection, target semantics, GNSS exclusion and generic COGNIX remain unchanged. Strict scenario-macro AUROC remains primary where defined with unchanged strict undefined conventions; two-class-eligible macro separately secondary; scenario simultaneous coverage remains the primary conformal diagnostic. Other metric definitions and seed summaries, including the frozen descriptive bootstrap, remain unchanged. No future outcome may change metrics.

## Access, gates and integrity

The exact 627-scenario list remains unavailable publicly. `inventory_access_protocol.json` and its Markdown companion specify later authorized header traversal. Payload bytes may transit sequential gzip decompression, discarded without semantic interpretation or persistence. No image decoding, Feather parsing, timestep annotation inspection, sensor-body retention, sensor statistics or inference is permitted. Output only unique IDs and path-derived town/condition/type, plus a separate exact access ledger.

Before RNG creation require complete unique inventory; published reconciliation to 107 normal and 520 anomaly before exclusions; both known historical IDs found and removed; no duplicate full IDs; every ID path-parseable; all exclusion decisions sealed; and source/eligible inventory hashes sealed. Changed publisher records or failed reconciliation mean STOP and amendment. No stratification, redraw or balancing.

Old protocol manifest SHA-256: `e03da4019c35d53420e42e02d4c0ebe2dd539a24ff5b645fdf5427078122ade1`.

Metadata audit manifest SHA-256: `888a63a0f0bb190eb899cfe900674bea6804b354017993b45c235b41b1c26839`.

Public provenance audit manifest SHA-256: `88242128bd2d4b3564e9c58ef12445b7c44c146f0a7f79b30e14371ed9700c50`.

New digest: `SHA256SUMS.sha256`, avoiding self-reference. The manifest seals current files and all history, excluding itself and its detached digest. `amendment_002_integrity.json` lists exact changed/new files. Static verification checks formulas and frozen fields without scores, RNG or science imports, prior archive identity, both audit manifests, source/config preservation and Git state outside the protocol. Checkpoints, prediction arrays and raw TRAIN/TEST payloads are not rehashed. This is scoped byte-integrity evidence and executed-tool attestation, not an OS-wide trace or full untracked-content checksum.

No TEST access, TEST scenario enumeration, sensor or label access, inventory execution, partition, inference, conformal fitting, training, tuning, commit or push occurred. Stop after sealing.
