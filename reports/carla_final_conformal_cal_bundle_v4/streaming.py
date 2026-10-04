"""Sequential gzip/strict TAR transport; membership gate before any decoder."""
import gzip
import hashlib
import io
from collections import Counter

from header_helpers import header_fields, classify

CHUNK = 65536
MAX_MEMBER = 16 * 1024 * 1024
MAX_UNCOMPRESSED = 512 * 1024**3


class HashReader(io.RawIOBase):
    def __init__(self, source, limit):
        self.source, self.limit = source, limit
        self.count = 0
        self.digest = hashlib.sha256()

    def readable(self):
        return True

    def read(self, n=-1):
        if n < 0:
            raise ValueError("UNBOUNDED_COMPRESSED_READ_FORBIDDEN")
        data = self.source.read(min(n, CHUNK))
        self.count += len(data)
        if self.count > self.limit:
            raise ValueError("COMPRESSED_SIZE_EXCEEDED")
        self.digest.update(data)
        return data


def scan(source, cal, evaluation, exclusions, on_cal, ledger, compressed_limit, *, prior_exposed_cal=frozenset()):
    """on_cal is invoked only behind the immutable CAL membership check.

    Final EVAL/prior-exposed CAL/historical-excluded bodies are chunk-discarded. They never enter on_cal, any
    image/Feather decoder, a feature function, model, or label function.
    """
    roles = (cal, evaluation, prior_exposed_cal, exclusions)
    if any(roles[i] & roles[j] for i in range(4) for j in range(i + 1, 4)):
        raise ValueError("INVALID_MEMBERSHIP")
    raw = HashReader(source, compressed_limit)
    gz = gzip.GzipFile(fileobj=raw, mode="rb")
    seen, finished, aliases = set(), set(), {}
    current = None
    body_bytes = Counter()
    opaque_discarded = Counter()
    seen_roles = {}
    decompressed = 0

    def read_exact(n):
        nonlocal decompressed
        out = bytearray()
        while len(out) < n:
            data = gz.read(min(CHUNK, n - len(out)))
            if not data:
                raise EOFError("TRUNCATED_TAR")
            decompressed += len(data)
            if decompressed > MAX_UNCOMPRESSED:
                raise ValueError("UNCOMPRESSED_CAP_EXCEEDED")
            out.extend(data)
        return bytes(out)

    def discard(n, role=None):
        while n:
            part = min(CHUNK, n)
            read_exact(part)
            if role is not None:
                opaque_discarded[role] += part
            n -= part

    try:
        headers = 0
        while True:
            block = read_exact(512)
            if block == bytes(512):
                if read_exact(512) != bytes(512):
                    raise ValueError("INVALID_TAR_TERMINATOR")
                break
            headers += 1
            if headers > 10_000_000:
                raise ValueError("TAR_HEADER_CAP_EXCEEDED")
            path, kind, size = header_fields(block)
            canonical, metadata, structural, transformation = classify(path, kind)
            if kind == b"5" and size:
                raise ValueError("DATA_BEARING_DIRECTORY")
            if metadata is None:
                if size:
                    raise ValueError("DATA_BEARING_STRUCTURAL_ROOT")
                continue
            sid = metadata["scenario_id"]
            if sid in cal:
                role = "FRESH_CAL"
            elif sid in evaluation:
                role = "FINAL_EVAL_OPAQUE_DISCARD"
            elif sid in prior_exposed_cal:
                role = "PRIOR_EXPOSED_CAL_OPAQUE_DISCARD"
            elif sid in exclusions:
                role = "HISTORICAL_EXCLUDED_OPAQUE_DISCARD"
            else:
                raise ValueError("UNKNOWN_SCENARIO_FAIL_CLOSED")
            seen_roles[sid] = role
            if aliases.setdefault(sid, transformation) != transformation:
                raise ValueError("CONFLICTING_SCENARIO_ALIAS")
            if sid != current:
                if current is not None:
                    if current in cal:
                        on_cal(current, None, None)
                    finished.add(current)
                if sid in finished:
                    raise ValueError("NONCONTIGUOUS_SCENARIO")
                current = sid
                seen.add(sid)
            # No EVAL child paths or tick identifiers are interpreted/retained.
            if kind != b"5":
                body_bytes[role] += size
                if sid in cal:
                    relative = canonical[len(sid) + 1:]
                    # Required modality/annotation selection is CAL-only.
                    needed = relative in ("imu.feather", "anomaly-observation.feather") or (
                        relative.startswith("rgb-front/") or relative.startswith("segmentation-front/"))
                    if needed:
                        if size > MAX_MEMBER:
                            raise ValueError("CAL_MEMBER_CAP_EXCEEDED")
                        on_cal(sid, relative, read_exact(size))
                    else:
                        discard(size, role=role)
                else:
                    discard(size, role=role)
                discard((-size) % 512)
        if current in cal:
            on_cal(current, None, None)
        # Validate gzip CRC/trailer and hash all compressed bytes through EOF.
        # Only zero TAR padding is allowed after the registered terminator.
        while True:
            tail = gz.read(CHUNK)
            if not tail:
                break
            decompressed += len(tail)
            if decompressed > MAX_UNCOMPRESSED or any(tail):
                raise ValueError("INVALID_TAR_TRAILING_DATA")
        while raw.read(CHUNK):
            pass
        if seen != cal | evaluation | prior_exposed_cal | exclusions:
            raise ValueError("ARCHIVE_SCENARIO_SET_MISMATCH")
        return {"compressed_bytes": raw.count, "sha256": raw.digest.hexdigest(),
                "gzip_crc_validated": True, "tar_complete": True, "headers": headers}
    finally:
        ledger.update({"compressed_bytes_received": raw.count,
                       "compressed_sha256_partial_or_complete": raw.digest.hexdigest(),
                       "body_bytes_accounted_by_role": dict(body_bytes),
                       "opaque_body_bytes_actually_discarded_by_role": dict(opaque_discarded),
                       "scenario_roots_seen_by_role": dict(Counter(seen_roles.values())),
                       "prior_exposed_cal_scenarios_decoded": 0,
                       "historical_excluded_scenarios_decoded": 0,
                       "scenario_roots_seen": len(seen), "eval_scenarios_decoded": 0,
                       "transport_is_not_scientific_decoding": True})
        gz.close()
