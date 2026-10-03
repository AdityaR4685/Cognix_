# Header-access amendment 001

This administrative amendment addresses the stopped structure-only attempt's
`UNSUPPORTED_TAR_HEADER_FORMAT` rejection. The exact two initial paths, sizes,
typeflags and `b'ustar '` magic are user-provided historical exposure evidence.
No TEST access is needed for development or Phase A.

The original parser and all stopped-attempt evidence remain unchanged. The new
`header_parser.py` reuses its octal, checksum, path, decompression, counter and
padding rules. An AST comparison constrains changes in `header_fields` to adding
the already-observed magic and consolidating unsupported typeflags under
`STOP_UNSUPPORTED_TAR_TYPEFLAG`. ASCII `0`, NUL regular files and ASCII `5`
directories remain the only permitted types. POSIX ustar and the previously
supported zero-magic form remain accepted. No link, device, FIFO, GNU long-name,
PAX or other extension-body handling is added. Unexpected magic stops closed.

The POSIX representation retains its `00` version and prefix parsing rules.
Space-terminated ustar reads the name field only; its version is recorded without
enabling additional header types or formats. Directory bodies must remain zero.
Bodies remain opaque and may only transit the decoder when advancing requires
it. This validation needs no body advancement: the zero-size directory is
followed immediately by the first regular-file header.

`synthetic_test_results.json` binds deterministic byte-level test results to
parser, test and driver hashes. It covers the authorized forms, all 253 other
typeflag bytes, bad checksums, malformed sizes, unsupported magic, existing
path/version guards, 512-byte advancement, exact counters and immediate stopping
before the second header's body or a third header.

`AMENDMENT_SHA256SUMS` seals Phase A inputs, authorization, source, tests, results
and preserved-state bindings. Its independent digest is verified before the
exclusive validation attempt marker is written and before any network access.
The historical attempt seals and Amendment 003's seal are verified locally;
protected source/configuration bindings and the seed receipt remain unchanged.

Phase B permits exactly one GET of the official `carlanomaly-base-test.tar.gz`,
Range `bytes=0-65535`. There is no additional source-metadata request, redirect,
retry, fallback or larger range. HTTP 206, exact range/length, identity encoding
and the prior ETag must match before the response body is read. The decompressor
is capped at 1,024 output bytes and the driver visits only two headers. Successful
recognition of both historical headers yields `HEADER_ACCESS_VALIDATED` and
stops before decompressing the JPEG body. Any failure spends the attempt and
records the raw structural reason; no second attempt follows.

Evidence distinguishes compressed bytes requested, received and consumed from
decompressed bytes produced. It records raw headers, paths, typeflags,
magic/version fields, checksum verdicts, body/padding counters, request metadata
and terminal reason. Raw member bodies and the volatile compressed prefix are
never saved or interpreted. Received compressed bytes may include opaque body
data, which need not be consumed. The received-prefix digest cannot prove full
archive integrity. Full gzip/tar EOF is intentionally not reached.

`VALIDATION_SHA256SUMS` separately seals the spent-attempt marker and evidence
after the final protected-state recheck. It binds the unchanged Phase A seal.
No scenario/eligible inventory, partition membership, RNG state, CAL/EVAL
assignment, label inspection, Feather parsing, image decoding, features,
inference, conformal fitting, metrics, training, tuning, commit or push is
authorized. No scientific, partition, conformal, metric, model or randomization
rule is changed. Amendment 003 and its seed stay byte-identical. This milestone
ends after sealing this bounded validation and does not proceed to the 91.5 GB
inventory traversal. Attestation covers executed tools and scoped hashes, not
an OS-wide access trace.
