# Future structure-only TEST inventory acquisition — design only

This procedure is not an implemented extractor and authorizes no TEST access. It was not executed. `inventory_access_protocol.json` is the machine-readable specification. The sealed audits establish publisher aggregates (627 total, 107 normal, 520 anomaly), but no complete exact public list or standalone Base TEST manifest.

## Access boundary

Reaching later headers in the official gzip-compressed Base TEST tar may require sequential decompression. **Payload bytes may transit the decompressor but are discarded without interpretation or persistence.** This is archive byte access, distinct from semantic payload inspection; it is not zero payload-byte reading. Bounded volatile buffers exist solely to advance through bodies/padding to the next header. No compressed archive cache or extracted members are retained. A streamed compressed-byte digest serves release integrity only, not sensor statistics.

Never decode JPEG/PNG, parse Feather, inspect timestep annotations, retain sensor bodies, calculate sensor statistics or invoke inference. Instantiate no dataset loader, devkit index, simulator or model. Derive IDs only from tar-header paths, without interpreting child filenames or tick identifiers as data.

## Mechanically bounded procedure

1. Before opening any archive, separately authorize and seal official URL/release, authenticated checksum/advertised size, parser/tool hash, output schema and caps. Missing size/checksum stops before access. Enumerate no supplementary archives or local TEST directories.
2. Enforce advertised compressed size, 1 TiB decompressed bytes, 10,000,000 headers, 64 GiB per member body, 1 MiB volatile chunks, 627 unique scenarios and 86,400 seconds. Count all discarded bodies/padding and requests. No automatic retries/restarts, expanded caps or fallbacks; failure marks incomplete inventory. Revised access budgets require a separate sealed amendment before another attempt.
3. Validate ordinary 512-byte POSIX/USTAR headers, checksum/size and name/prefix path fields. Accept only documented optional `carlanomaly/` or `./` wrappers; safe POSIX normal/anomaly full scenario grammar. Reject unsafe or ambiguous paths, links/devices/sparse entries and conflicting aliases. STOP for PAX/GNU long-name/link extensions requiring body decoding; do not inspect extension bodies to recover paths.
4. Derive full scenario ID, town, directory condition and anomaly type from header paths. Use NORMAL as a metadata sentinel for normal scenarios, never a tick label. Repeated member headers under a scenario deduplicate to one ID. Duplicate member paths, explicit scenario directory entries, output rows or conflicting aliases halt. Accept structural root directories only; unclassifiable data-bearing entries halt. `scenario-N` supplies no family identity.
5. Discard declared bodies and padding in bounded volatile chunks. Log no sensor contents or complete child paths. Validate full tar termination, gzip trailer/integrity and complete EOF; reject truncation, unexpected concatenated streams or later nonpadding content. Match compressed stream digest to authenticated checksum. Failure stops without inventory sealing or randomization.
6. Emit only sorted unique IDs and path-derived town/condition/type, plus a separate exact access ledger. No tick annotations, labels, statistics, predictions or scores. Complete header traversal does not certify sensor/label/synchronization completeness; a later separately authorized frozen feature/label check must do that.

## Exact access ledger

Record authorization/tool/protocol hashes; source URL/release and size/checksum; every request range, HTTP status and bytes received; start/stop UTC; exact compressed/decompressed/header/body/padding counters; gzip trailer/EOF/digest status; header/unique-scenario counts; canonicalization; cap usage; success/failure reason.

For each header record sequence, decompressed offset, compressed input counter, header type, declared size, discarded body/padding counts, scenario ID or structural root and parse/failure code. Buffered compressed counters measure bytes consumed, not a fictitious exact compressed offset for every decompressed byte. Persist no complete child paths or timestep identifiers. Keep access ledger separate from scenario-only output. The ledger accounts for tool access, not all OS activity.

## Gates before randomization

Require complete unique list, full traversal/integrity/cap checks and **627 total, 107 normal and 520 anomaly before exclusions**. Changed publisher records require amendment. Both `test/anomaly/Town01/change-weather/scenario-1` and `test/anomaly/Town01/change-weather/scenario-10` must be found and removed in full from both roles. No duplicate full inventory IDs; every ID must parse into town/condition/type.

Seal all exclusion decisions, including future independently authenticated historical exposures documented before partition execution. Disclose the incomplete historical file-level ledger; infer no extra IDs or families. Seal source inventory, exact exclusion supplement and eligible inventory hash **before RNG creation**. Failed reconciliation or unresolved exclusion decisions mean STOP.

Only a later authorized partition stage may make one PCG64(2028) permutation of sorted complete eligible IDs, assigning first ceil(N_eligible/5) to calibration and remainder to evaluation. Preserve metadata for descriptive subgroup reporting; no stratification, redraws, balancing or inference belongs in the extractor.
