# Amendment 003 — partition randomization clarification only

2026-10-03, Asia/Calcutta. Explicitly authorized protocol amendment and one local random seed draw only.

The complete 122-file Amendment 002 state, including its manifest, detached digest and every earlier history, is preserved byte-for-byte under `amendment_history/pre_amendment_003/`. Earlier amendment records/scripts/changelog remain unchanged historical artifacts. Current randomization instructions are Amendment 003 and `partition_randomization_receipt.json`.

Fixed predetermined PCG64(2028) was reproducible but did not itself constitute a probabilistic randomization draw. Replace it with the following irrevocable draw, made before any TEST inventory access in this milestone:

- Integer seed: **334227055836169582741732795999208677637**.
- Method: one Python `secrets.randbits(128)` call using OS cryptographic randomness.
- Timestamp: **2026-10-03T04:05:19.832037+00:00**, or **2026-10-03T09:35:19.832037+05:30** in Asia/Calcutta.
- Python: **CPython 3.12.14**.
- Receipt SHA-256: `2a6cc338637a6a2b6ce661f6711ac4ea3e45ff6337346d27ff1b4753e3377368`.

The receipt was exclusively created before its sole draw, flushed to disk and independently sealed in `partition_randomization_receipt.sha256`. The generation path refuses to draw if a receipt exists, even if incomplete. Accept the first result unconditionally, including any coincidence with historical seeds. Never regenerate, change, redraw, balance or search the seed based on inventory composition, labels, predictions, metrics or conformal results. A lost/corrupt receipt means STOP, never a replacement draw. Historical documented TEST exposures remain excluded; this timing statement does not erase those earlier exposures.

Later, only after independently sealing the complete eligible inventory and obtaining separate execution authorization, use lexicographically sorted eligible full scenario IDs, NumPy `Generator(PCG64(SEALED_RANDOM_SEED))`, exactly one permutation, n_cal=ceil(N_eligible/5), first n_cal to FINAL_CONFORMAL_CAL and remainder to FINAL_EVALUATION. No PCG64 instance or permutation was created here.

The primary finite-catalogue interpretation is marginal over preregistered randomized assignment and a held-out eligible catalogue scenario. It is not conditional on the realized seed/partition/calibration sample and is not a population/new-route/independent-family guarantee. The augmented-rank proposition retains its uniform-scenario-assignment condition; seeded PCG64 is the registered pseudorandom implementation, not itself a proof of exact uniformity over all permutations. Separately, any population-style interpretation requires appropriate scenario-score exchangeability, which the random split does not establish.

The 20/80 ratio, complete-scenario calibration unit, candidate score, block maximum, pooled conformal construction, alpha=.05, all 15 method/model-seed scorers, all metrics, historical exclusions, subgroup reporting and scientific pipeline remain unchanged. Inventory parser, bounds, output schema and access ledger remain unchanged; only seed timing/Generator wording is clarified. Updated current protocol files and exact JSON field changes are listed in `amendment_003_integrity.json`. The current verifier dispatches to metadata-only Amendment 003 checks; it never draws randomness.

Old manifest SHA-256: `c6516d507a7b0189c4fc11d0ad429a7dc080ebb49ea127623a87e1c4ed39482b`.

New manifest SHA-256: recorded in `SHA256SUMS.sha256`, avoiding self-reference. Every current/new/history file is sealed except the manifest itself and its detached digest. Prior audit manifests and source/config checks remain intact; no raw payload, prediction array or checkpoint is rehashed. Verification is scoped integrity evidence and executed-tool attestation, not an OS-wide trace.

No TEST access or scenario enumeration, sensor/label access, inventory execution, partition, inference, conformal fitting, training, tuning, commit or push occurred. Stop after sealing Amendment 003.
