# COGNIX public provenance audit v1

**Exact family/group identities for released TEST scenarios cannot be recovered from the inspected public information without TEST access.** The generator mechanism is substantially known, but the released-scenario-to-generation-run-to-family mapping is missing. This is a bounded negative finding about the public sources inspected on 2 October 2026 (Asia/Calcutta), not proof that private or future publisher records cannot exist.

Milestone: **PUBLIC PROVENANCE RECOVERY / GENERATOR AUDIT ONLY**. Parent metadata-audit manifest: `888a63a0f0bb190eb899cfe900674bea6804b354017993b45c235b41b1c26839`; its entries were verified. Generator snapshot: `carlanomaly/carla-gen@9b3b284721b41e7aea87ffe2aa812766775be146`. The existing scientific protocol is preserved.

## Access boundary and reproducibility

Only public software/source/configuration/documentation and the sealed local audit/protocol context were inspected. No official TEST archive was requested, downloaded, listed or decompressed. No TEST sensor payload, timestep annotation, prediction, model output or final partition was opened. No simulator, devkit loader, downloader, training script, evaluator or public source test was executed. Source code describing label writers and model drivers was read solely as software; no actual generated labels or outputs were inspected. Public media `assets/example.gif` was deliberately skipped.

All 74 generator text blobs were preserved; their Git blob hashes match the advertised tree. The sole advertised generator branch has one parentless public commit, with no tags, releases, issues or pull requests in the returned listings. This supplies no earlier public generation or renaming history. The API's resolved tree response identifier and the actual commit tree object are recorded separately.

The owner listing exposed `devkit` and `baselines`. Their current public text sources and six main-history file trees each were inspected. Removed devkit plan/documentation/source files were also inspected. The additional baselines `reproducible-runs` branch was inspected at file-tree level. Historical related-repository blobs and that extra branch's blob contents were not exhaustively audited. No claim is made about deleted, unadvertised or private refs. Source URLs, revisions, sizes, hashes, skipped assets, and search coverage are recorded in `source_inventory.json`, `evidence_index.json` and supporting snapshots.

The [publisher resources page](https://carlanomaly.de/resources/) describes the Hydra/CARLA framework; the [publisher FAQ](https://carlanomaly.de/about/) describes simulator patches. These pages still provide placeholder software links. The user-supplied public repository has matching generator structure, but no inspected release manifest certifies that this exact revision/configuration generated each released clip.

## Distinct knowledge states

| State | What this audit supports |
|---|---|
| **GENERATOR MECHANISM KNOWN** | Hydra composition, per-component seeds, map-dependent spawning, Traffic Manager autopilot, live anomaly callbacks, weather initialization, timestamp output paths and recording/config/log mechanisms. |
| **DATASET-SCENARIO MAPPING KNOWN** | Not established for released run/seed/route/family identities. Only the naming grammar and interpretation of split/condition/town/type fields in a supplied full path are known. |
| **MAPPING NOT RECOVERABLE** | Actual 627 released IDs, their configured/effective simulator seeds, realized routes, retained run timestamps, common base drives, variant parents and exact family groups. |

Publisher totals remain 107 normal plus 520 anomaly TEST scenarios, taken from the parent audit's sealed composition metadata. They are aggregate counts, not an inventory. This audit enumerates zero actual released scenario IDs and constructs zero groups.

## Seeds, routes and reproducibility

The [default configuration](https://github.com/carlanomaly/carla-gen/blob/9b3b284721b41e7aea87ffe2aa812766775be146/config/default.yaml#L3) permits command-line and optional local overrides and sets a top-level seed. [Simulation configuration](https://github.com/carlanomaly/carla-gen/blob/9b3b284721b41e7aea87ffe2aa812766775be146/config/simulation/default.yaml#L17) forwards it to ego, vehicles and pedestrians. Seeded spawning depends on map content, available blueprints and spawn success.

The [ego spawn implementation](https://github.com/carlanomaly/carla-gen/blob/9b3b284721b41e7aea87ffe2aa812766775be146/carlagen/utils/carla.py#L344) selects `seed % len(spawn_points)`; a failed spawn changes the local seed and logs the replacement. Therefore neither equal spawn positions nor equal numeric seeds constitute a unique route or family identity. The same seed has no map-independent spawn meaning.

Traffic Manager uses a hard-coded seed of 12345. The [simulation setup](https://github.com/carlanomaly/carla-gen/blob/9b3b284721b41e7aea87ffe2aa812766775be146/carlagen/simulation.py#L271) loads the world, spawns actors and enables live autopilot. No explicit ego route catalogue, route ID, prescribed waypoint replay or base-drive ID was found in generator sources.

There are further static reproducibility limitations: [base anomaly constructor](https://github.com/carlanomaly/carla-gen/blob/9b3b284721b41e7aea87ffe2aa812766775be146/carlagen/callback/ano.py#L238) parameters are `seed` and `scheduler_seed`, while spawn-props/steer-driver configs use legacy `cb_seed`/`sc_seed`. The inspected call chain does not translate these names into the base RNG seeds. Some steering/traffic-light choices use global RNGs. These are source findings, not tested release behavior. CARLA client version is 0.9.14, while Hydra and some dependencies are unpinned; patched simulator/maps and unpublished overrides are not release-bound.

## Naming, sidecars and logs

The [Hydra output configuration](https://github.com/carlanomaly/carla-gen/blob/9b3b284721b41e7aea87ffe2aa812766775be146/config/hydra/default.yaml#L5) appends date/time directories to town/condition/type prefixes. Multiruns append a Hydra job number. Saved sensor filenames are tick-number based. None of these defaults establishes a released `scenario-N` suffix.

**No stable semantic meaning of `scenario-1` across towns or anomaly types is established.** No inspected generator code creates that suffix, and no public renaming/selection table connects timestamp directories or sweep jobs to it. Sequential numbering, seed equality and shared route semantics are all unverified. Matching numbers are not linkage evidence.

Under [documented Hydra defaults](https://hydra.cc/docs/1.2/tutorials/basic/running_your_app/working_directory/), run directories normally contain composed configuration and override sidecars plus an application log. The repository enables working-directory changes and does not disable those sidecars. Its entrypoint logs configuration; spawning/weather code logs selected settings and adjusted spawn seeds. This establishes possible generation-time provenance storage, not its retention or publication in the release. No run sidecar or log from TEST was accessed.

The [recorder callback](https://github.com/carlanomaly/carla-gen/blob/9b3b284721b41e7aea87ffe2aa812766775be146/carlagen/callback/saving.py#L22) writes `carla.log`. The devkit registry describes a recordings shard with `sim.log`. No public conversion/renaming binding was found. Recordings are payload-bearing sources and were not fetched.

## Normal/anomaly variants, weather and repeated drives

[Normal test configuration](https://github.com/carlanomaly/carla-gen/blob/9b3b284721b41e7aea87ffe2aa812766775be146/config/experiment/test.yaml#L1) and anomaly configurations share the test-clip template. Its [daytime/weather callbacks](https://github.com/carlanomaly/carla-gen/blob/9b3b284721b41e7aea87ffe2aa812766775be146/config/callbacks/test-clip.yaml#L1) receive the same top-level seed. Thus the mechanism supports shared initialization choices under matching town, effective configuration and environment, with live anomaly interventions.

The [generation launcher](https://github.com/carlanomaly/carla-gen/blob/9b3b284721b41e7aea87ffe2aa812766775be146/scripts/generate.sh#L1) explicitly reuses seeds 150–200 across several Town10HD anomaly commands and a training command. Running-pedestrians additionally changes the crossing factor. This is evidence of intended seed reuse; it does not prove retained released pairs. The script invokes no normal-test experiment.

Anomaly runs are fresh simulations in the inspected implementation. Recording exists, but no recorder playback call was found. Therefore “replayed versions of one common drive” is not established. Shared initialization also does not guarantee identical realized trajectories after interventions. Initial weather selection is seed-controlled; change-weather acts during a live drive. No independent weather-variant/base-drive manifest was found.

The release may contain related or repeated drives, but their existence, exact pairs and frequency remain unknown. No sensor similarity, trajectory reconstruction, filename-only linkage or equal-suffix pairing was used.

## Export/release audit and missing inventory

The public generator has generation launchers and rendering utilities, not a release inventory or timestamp-to-scenario exporter. Although `generate.sh` describes dataset generation, its current commands cover Town10HD and reference two absent experiment configurations: `ano-test-06-crazy-cars` and `ano-test-11-driver-steering`. The [training launcher](https://github.com/carlanomaly/carla-gen/blob/9b3b284721b41e7aea87ffe2aa812766775be146/scripts/generate-train.sh#L1) has mostly commented town loops and active Town11/Town12 commands. These snapshots do not specify the documented six-town release selection.

A useful omitted-artifact lead appears in [devkit download source](https://github.com/carlanomaly/devkit/blob/91304c6ca5482789188965164013627be2504e12/carlanomaly/download.py#L43): `package_dataset.sh`, and separately `shasums.md`. Neither is present in the inspected current/historical software file trees or additional branch tree. The references reveal names, not their implementation, inputs or provenance policy. Archive part/split checksums cannot identify families.

The [devkit index](https://github.com/carlanomaly/devkit/blob/91304c6ca5482789188965164013627be2504e12/carlanomaly/index.py#L170) discovers local directories from observation-sidecar paths and counts local RGB files. Its record holds path, timestep count, split, town and anomaly type, without seed, route or family fields. This code is not an embedded public 627-scenario inventory and was not instantiated. Synthetic public test fixtures do not establish actual release membership.

The missing chain is:

1. Released full scenario ID to retained run/timestamp or recorder identity.
2. Retained run to actual composed configuration, overrides, effective seeds and simulator/map/software revision.
3. Variant runs to an explicit shared base-drive/route parent.
4. Complete release inventory and documented selection/renaming/retention policy.

A publisher-issued scenario-only provenance manifest could supply these fields without TEST payload access. A deterministic exporter could help only with its complete input manifest and documented family relation. Bare seed values would still not certify common drives.

## Consequences for final evaluation

**Exact family-level partitioning is not currently feasible.** No safe exact family/group identifiers can be constructed from the inspected public sources. Whole-town or town/type clusters are definable categories once an inventory exists, but do not recover drive families and would change the evaluation target and allocation behavior; this audit adopts none.

Scenario-level randomization is the only currently supportable unit when choosing between individual scenarios and recovered drive families. It is a conditional fallback after a complete eligible inventory is obtained, not an unconditional finding that it is the only defensible design or that the final evaluation is ready. Deferring evaluation pending publisher provenance remains appropriate.

If family identities remain unavailable, any subsequently authorized scenario design would need to state that between-scenario exchangeability within required strata is unverified; related normal/anomaly clips or repeated routes/seeds might cross calibration/evaluation; shared generation mechanisms may affect uncertainty; and independent-family coverage or novel-route generalization is unsupported. A random catalogue split does not itself prove a new-scenario conformal guarantee. Claims must be conditional on the release mix, eligible inventory and appropriate score/calibration exchangeability assumptions.

If explicit publisher relationships later become available, verify their release binding before partitioning, group only supported common-parent relations, and audit cross-stratum families, exposure exclusions and calibration-unit/count consequences. Any design changes require a separate authorized, sealed amendment. No mapping procedure is executed here.

## Seal and stop

The parent count/rank result remains in force: under published counts and the current rule, at least one required stratum has `Q_h=+inf`, so `Q=max_h Q_h=+inf`. Provenance recovery does not repair that mathematics.

No alpha, strata, ratio, score, pooling, envelope, metrics, model, threshold or seed changed. No partition, inference, fitting, training, protocol amendment, commit or push occurred. Existing audit/protocol hashes and Git state are checked in `integrity_verification.json`. New files are confined to this audit directory. `SHA256SUMS` seals every audit file except itself; its SHA-256 is reported on delivery. Work stops after sealing.
