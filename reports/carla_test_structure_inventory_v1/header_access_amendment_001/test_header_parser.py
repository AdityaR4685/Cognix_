"""Deterministic synthetic byte tests; no official TEST bytes or network."""
import ast
import io
import json
import zlib
from pathlib import Path
import header_parser as p
import integrity as i
import validate_headers as v

HERE = Path(__file__).resolve().parent


def checksum(block):
    b = bytearray(block)
    b[148:156] = b"        "
    b[148:156] = ("%06o\0 " % sum(b)).encode("ascii")
    return bytes(b)


def header(name="test/normal/Town01/scenario-1/synthetic", kind=b"0", size=0,
           magic=b"ustar ", version=b" \0", prefix=b""):
    b = bytearray(512)
    name = name.encode("utf-8")
    assert len(name) <= 100 and len(kind) == 1 and len(magic) == 6 and len(version) == 2
    b[:len(name)] = name
    b[100:108] = b"0000755\0"
    b[108:116] = b"0000000\0"
    b[116:124] = b"0000000\0"
    b[124:136] = ("%011o\0" % size).encode("ascii")
    b[136:148] = b"00000000000\0"
    b[156:157] = kind
    b[257:263] = magic
    b[263:265] = version
    b[345:345+len(prefix)] = prefix
    return checksum(b)


def expect_stop(block, code, classify=False):
    try:
        path, kind, _ = p.header_fields(block)
        if classify:
            p.classify(path, kind)
    except p.Stop as exc:
        assert str(exc) == code, (str(exc), code)
    else:
        raise AssertionError("Expected " + code)


def reader(raw, override=None):
    compressor = zlib.compressobj(wbits=31)
    packed = compressor.compress(raw) + compressor.flush()
    ledger = v.empty_ledger()
    ledger["requests"] = [{"bytes_received": 0}]
    caps = v.caps()
    caps["compressed_bytes_max"] = len(packed)
    caps["uncompressed_bytes_max"] = len(raw)
    caps.update(override or {})
    return p.BoundedGzip(io.BytesIO(packed), caps, ledger), ledger


def main():
    tests = []
    # AST comparison restricts the amendment to magic/type acceptance and reason.
    old = ast.parse(p.SOURCE.read_text(encoding="utf-8"))
    new = ast.parse((HERE / "header_parser.py").read_text(encoding="utf-8"))
    old_fn = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == "header_fields")
    new_fn = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == "header_fields")
    expected = ast.unparse(old_fn)
    expected = expected.replace("    require(kind not in {b'x', b'g', b'L', b'K', b'S'}, 'UNSUPPORTED_BODY_DEPENDENT_HEADER_EXTENSION')\n", "")
    expected = expected.replace("'UNSUPPORTED_LINK_DEVICE_OR_HEADER_TYPE'", "'STOP_UNSUPPORTED_TAR_TYPEFLAG'")
    expected = expected.replace("{b'ustar\\x00', b'\\x00' * 6}", "{b'ustar\\x00', b'ustar ', b'\\x00' * 6}")
    assert ast.dump(ast.parse(expected).body[0]) == ast.dump(new_fn), "Unapproved parser logic change"
    assert p.octal is p.original.octal and p.text_field is p.original.text_field
    assert p.classify is p.original.classify and p.BoundedGzip is p.original.BoundedGzip
    tests.append("AST: only authorized header acceptance/reason changed; sealed guards and stream reader reused")
    directory = "test/normal/Town01/scenario-1/synthetic/"
    assert p.header_fields(header(directory, b"5")) == (directory, b"5", 0)
    tests.append("directory type 5, size zero, space-terminated ustar")
    for kind in (b"0", b"\0"):
        assert p.header_fields(header(kind=kind, size=731))[1:] == (kind, 731)
    tests.append("ASCII 0 and NUL regular files with positive octal size, space-terminated ustar")
    for magic, version in ((b"ustar\0", b"00"), (b"\0" * 6, b"\0\0")):
        for kind in (b"0", b"\0", b"5"):
            assert p.header_fields(header(kind=kind, magic=magic, version=version))[1] == kind
    assert p.header_fields(header("child", magic=b"ustar\0", version=b"00", prefix=b"test/normal/Town01/scenario-1"))[0] == "test/normal/Town01/scenario-1/child"
    expect_stop(header(magic=b"ustar\0", version=b"01"), "UNSUPPORTED_USTAR_VERSION")
    tests.append("existing POSIX ustar/version/prefix and zero-magic representation preserved")
    corrupted = bytearray(header())
    corrupted[0] ^= 1
    expect_stop(bytes(corrupted), "TAR_HEADER_CHECKSUM_MISMATCH")
    tests.append("checksum corruption rejected")
    for field, code in ((b"00000000009\0", "INVALID_OCTAL_FIELD"),
                        (b"\0" * 12, "INVALID_OCTAL_FIELD"),
                        (b"\x80" + b"\0" * 11, "UNSUPPORTED_BINARY_NUMERIC_FIELD")):
        malformed = bytearray(header())
        malformed[124:136] = field
        expect_stop(checksum(malformed), code)
    expect_stop(header()[:511], "TRUNCATED_TAR_HEADER")
    tests.append("malformed/empty/non-octal/binary sizes and truncated header rejected")
    for flag in range(256):
        if flag not in (0, 48, 53):
            expect_stop(header(kind=bytes([flag]), size=23), "STOP_UNSUPPORTED_TAR_TYPEFLAG")
    tests.append("all 253 unauthorized typeflag bytes rejected, including links/devices/FIFO/GNU/PAX")
    for magic in (b"ustarX", b"other!", b"\xff" * 6):
        expect_stop(header(magic=magic), "UNSUPPORTED_TAR_HEADER_FORMAT")
    tests.append("unexpected magic rejected")
    for path, reason in (("/test/normal/Town01/scenario-1/x", "UNSAFE_HEADER_PATH"),
                         ("C:/test/normal/Town01/scenario-1/x", "DRIVE_PATH"),
                         ("test/normal/Town01/../scenario-1/x", "UNSAFE_OR_AMBIGUOUS_PATH_COMPONENT"),
                         ("test/normal/Town99/scenario-1/x", "UNSUPPORTED_TOWN"),
                         ("test/normal/Town01/scenario-0/x", "INVALID_SCENARIO_PREFIX")):
        expect_stop(header(path), reason, classify=True)
    bad = bytearray(header())
    bad[157] = 65
    expect_stop(checksum(bad), "UNEXPECTED_LINK_NAME")
    bad = bytearray(header())
    bad[50:53] = b"x\0y"
    expect_stop(checksum(bad), "AMBIGUOUS_NUL_FIELD")
    tests.append("existing path restrictions, link-name and ambiguous NUL-field guards preserved")
    for size in (0, 1, 511, 512, 513, 1025):
        raw = header(size=size) + b"S" * size + b"\0" * ((-size) % 512) + header(directory, b"5")
        r, ledger = reader(raw)
        assert p.header_fields(r.exact(512))[2] == size
        event = {"member_body_bytes_discarded": 0, "padding_bytes_discarded": 0}
        r.discard(size, "member_body_bytes_discarded", event)
        r.discard((-size) % 512, "padding_bytes_discarded", event)
        assert ledger["decompressed_bytes_consumed"] == 512 + size + (-size) % 512
        assert p.header_fields(r.exact(512)) == (directory, b"5", 0)
        assert event["member_body_bytes_discarded"] == size and event["padding_bytes_discarded"] == (-size) % 512
    tests.append("exact body discard and 512-byte alignment/padding advancement for six boundary sizes")
    first, second = v.EXPECTED
    raw = header(first[0], first[1], first[2]) + header(second[0], second[1], second[2])
    raw += b"S" * second[2] + b"\0" * ((-second[2]) % 512) + header("test/normal/Town01/scenario-2/", b"5")
    r, ledger = reader(raw, {"uncompressed_bytes_max": 1024})
    events = []
    v.initial_headers(r, ledger, events)
    assert ledger["stop_reason"] == "HEADER_ACCESS_VALIDATED"
    assert ledger["headers_visited"] == ledger["headers_parsed"] == 2
    assert ledger["decompressed_bytes_consumed"] == ledger["header_bytes_consumed"] == 1024
    assert ledger["member_body_bytes_discarded"] == ledger["padding_bytes_discarded"] == 0
    assert len(events) == 2 and all(e["checksum_verdict"] == "PASS" for e in events)
    assert ledger["compressed_bytes_received"] == ledger["requests"][0]["bytes_received"]
    assert ledger["compressed_bytes_consumed"] + len(r.pending) == ledger["compressed_bytes_received"]
    tests.append("validation stops at two known synthetic headers; JPEG-shaped body/third header never decompressed; exact counters")
    for flag, magic, code in ((b"x", b"ustar ", "STOP_UNSUPPORTED_TAR_TYPEFLAG"),
                             (b"5", b"other!", "UNSUPPORTED_TAR_HEADER_FORMAT")):
        r, ledger = reader(header(first[0], flag, 23, magic) + b"S" * 23)
        events = []
        try:
            v.initial_headers(r, ledger, events)
        except p.Stop as exc:
            assert str(exc) == code
        else:
            raise AssertionError("Expected failure")
        assert ledger["decompressed_bytes_consumed"] == 512 and ledger["member_body_bytes_discarded"] == 0
        assert events[0]["typeflag_hex"] == flag.hex() and events[0]["parse_verdict"] == code
        assert len(bytes.fromhex(events[0]["raw_header_hex"])) == 512
    tests.append("failure records exact raw structural metadata and stops before any body")
    result = {"passed": True, "synthetic_only": True, "official_TEST_accessed": False,
              "test_groups": tests, "test_group_count": len(tests), "RNG_instantiated": False,
              "parser_sha256": i.sha(HERE / "header_parser.py"),
              "test_sha256": i.sha(Path(__file__)), "validator_sha256": i.sha(HERE / "validate_headers.py")}
    i.write(HERE / "synthetic_test_results.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
