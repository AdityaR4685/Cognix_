# Header-access amendment 001: bounded validation

Terminal state: **HEADER_ACCESS_VALIDATED**. One range GET requested bytes 0-65535 (65,536 compressed bytes).

Received 65,536; decoder consumed 301; decompressed 1,024 bytes. Visited 2 headers; parsed 2. Discarded/transited 0 decompressed member-body bytes and 0 padding bytes.

Exact structural header metadata, checksum verdicts, typeflags, magic/version fields, counters and HTTP response metadata are in validation_evidence.json and header_access_ledger.jsonl. The compressed prefix stayed volatile; neither compressed archive bytes nor member bodies were saved. Compressed data received but not consumed may contain opaque member-body data; it was never decompressed or interpreted.

The amendment and synthetic tests were sealed and verified before this access. The historical stopped-attempt records, Amendment 003, its seed receipt, and protected source/configuration bindings were checked again afterward. This is header validation only; no inventory, eligible inventory, partition membership, RNG state, labels, sensor decoding, scientific execution, commit or push was produced. Full archive integrity and gzip/tar EOF were not validated.

No retry, fallback, further member discovery or full sequential traversal follows. A failure requires separate authorization; this attempt is permanently spent. Attestation covers executed tools and scoped hashes, not an OS-wide access trace.
