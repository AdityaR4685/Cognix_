"""Deterministic synthetic byte fixtures only; never official TEST inputs."""
import io
import json
import zlib
from pathlib import Path
import extract_structure_inventory as m


def header(name, kind=b"0", size=0, magic=b"ustar\0"):
    b = bytearray(512)
    encoded = name.encode("utf-8")
    assert len(encoded) <= 100
    b[:len(encoded)] = encoded
    b[100:108] = b"0000755\0"
    b[108:116] = b"0000000\0"
    b[116:124] = b"0000000\0"
    b[124:136] = ("%011o\0" % size).encode()
    b[136:148] = b"00000000000\0"
    b[148:156] = b"        "
    b[156:157] = kind
    b[257:263] = magic
    b[263:265] = b"00"
    b[148:156] = ("%06o\0 " % sum(b)).encode()
    return bytes(b)


def gzip(raw):
    encoder = zlib.compressobj(wbits=31)
    return encoder.compress(raw) + encoder.flush()


def run(raw, caps_override=None, encoded=None):
    packed = encoded if encoded is not None else gzip(raw)
    caps = {"wall_clock_seconds_max": 10, "volatile_chunk_bytes_max": 1048576,
            "compressed_bytes_max": len(packed), "uncompressed_bytes_max": 1000000,
            "tar_header_records_max": 100, "member_body_bytes_max": 100000,
            "unique_scenarios_max": 627}
    caps.update(caps_override or {})
    ledger = {"compressed_bytes_received": 0, "compressed_bytes_consumed": 0,
              "decompressed_bytes_consumed": 0, "requests": [{"bytes_received": 0}],
              "headers_visited": 0, "header_bytes_consumed": 0,
              "tar_termination_bytes_consumed": 0, "tar_end_padding_bytes_discarded": 0,
              "member_body_bytes_discarded": 0, "padding_bytes_discarded": 0,
              "unique_scenario_count": 0, "canonicalization_counts": {}}
    reader = m.BoundedGzip(io.BytesIO(packed), caps, ledger)
    events = io.StringIO()
    try:
        rows = m.walk(reader, ledger, events)
        return rows, ledger, events.getvalue(), None
    except Exception as exc:
        return None, ledger, events.getvalue(), str(exc) if isinstance(exc, m.Stop) else type(exc).__name__


def main():
    tests = []
    root = "test/normal/Town01/scenario-1"
    child = root + "/synthetic-child"
    body = b"opaque uninterpreted synthetic bytes"
    raw = header(root + "/", b"5") + header(child, size=len(body)) + body + b"\0" * ((-len(body)) % 512) + b"\0" * 1024
    rows, ledger, events, error = run(raw)
    assert error is None and len(rows) == 1
    assert ledger["member_body_bytes_discarded"] == len(body)
    assert ledger["padding_bytes_discarded"] == (-len(body)) % 512
    assert ledger["decompressed_bytes_consumed"] == len(raw)
    assert ledger["compressed_bytes_received"] == ledger["compressed_bytes_consumed"] == len(gzip(raw))
    assert child not in events and body.decode() not in events
    assert ledger["gzip_trailer_validated"] and ledger["HTTP_EOF_validated"]
    tests.append("valid traversal, exact counters, deduplication and child/body log exclusion")
    fixtures = [
        ("PAX stops before extension body", header("synthetic-extension", b"x", 23) + b"S" * 23, "UNSUPPORTED_BODY_DEPENDENT_HEADER_EXTENSION"),
        ("GNU long-name stops before body", header("synthetic-extension", b"L", 23) + b"S" * 23, "UNSUPPORTED_BODY_DEPENDENT_HEADER_EXTENSION"),
        ("links rejected", header("synthetic-link", b"2"), "UNSUPPORTED_LINK_DEVICE_OR_HEADER_TYPE"),
        ("duplicate member rejected", header(root + "/", b"5") * 2, "DUPLICATE_MEMBER_PATH"),
        ("alias conflict rejected", header(root + "/", b"5") + header("./" + child), "CONFLICTING_CANONICAL_SCENARIO_ALIASES"),
        ("unsafe traversal rejected", header("test/normal/Town01/../scenario-1/", b"5"), "UNSAFE_OR_AMBIGUOUS_PATH_COMPONENT"),
        ("absolute path rejected", header("/" + root + "/", b"5"), "UNSAFE_HEADER_PATH"),
        ("unsupported town rejected", header("test/normal/Town99/scenario-1/", b"5"), "UNSUPPORTED_TOWN"),
        ("nonpositive scenario rejected", header("test/normal/Town01/scenario-0/", b"5"), "INVALID_SCENARIO_PREFIX"),
        ("unsupported anomaly type rejected", header("test/anomaly/Town01/unknown/scenario-1/", b"5"), "UNSUPPORTED_ANOMALY_TYPE"),
        ("later nonpadding rejected", b"\0" * 1024 + b"X" * 512, "NONPADDING_AFTER_TAR_TERMINATION"),
        ("truncated tar rejected", header(child, size=2) + b"x", "TRUNCATED_TAR_MEMBER_BODY_OR_PADDING"),
    ]
    for title, fixture, expected in fixtures:
        _, counters, _, actual = run(fixture)
        assert actual == expected, (title, actual, expected)
        if "before" in title:
            assert counters["decompressed_bytes_consumed"] == 512 and counters["member_body_bytes_discarded"] == 0
        tests.append(title)
    corrupted = bytearray(header(root + "/", b"5"))
    corrupted[0] ^= 1
    assert run(bytes(corrupted))[3] == "TAR_HEADER_CHECKSUM_MISMATCH"
    tests.append("tar checksum corruption rejected")
    assert run(raw, {"member_body_bytes_max": 1})[3] == "MEMBER_BODY_CAP_EXCEEDED"
    assert run(raw, {"tar_header_records_max": 1})[3] == "HEADER_CAP_EXCEEDED"
    assert run(raw, {"unique_scenarios_max": 0})[3] == "SCENARIO_CAP_EXCEEDED"
    assert run(raw, {"uncompressed_bytes_max": 100})[3] == "DECOMPRESSED_CAP_EXCEEDED"
    tests.append("member, header, scenario and decompressed caps enforced")
    assert run(raw, encoded=gzip(raw) + gzip(b""))[3] == "CONCATENATED_GZIP_OR_TRAILING_COMPRESSED_DATA"
    tests.append("concatenated gzip rejected")
    packed = bytearray(gzip(raw))
    packed[-8] ^= 1
    assert run(raw, encoded=bytes(packed))[3] == "error"
    tests.append("gzip trailer CRC corruption rejected")
    for prefix in ("./", "carlanomaly/"):
        result = run(header(prefix + root + "/", b"5") + b"\0" * 1024)
        assert result[3] is None and root in result[0]
    tests.append("documented single wrappers canonicalized")
    result = {"passed": True, "test_groups": tests, "official_archive_accessed": False,
              "synthetic_only": True, "RNG_instantiated": False}
    (Path(__file__).parent / "synthetic_parser_verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
