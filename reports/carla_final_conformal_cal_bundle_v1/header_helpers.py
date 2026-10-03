import re

TOWNS = {"Town01", "Town02", "Town03", "Town04", "Town05", "Town10HD"}

TYPES = {"change-weather", "running-pedestrian", "spawn-props", "steer-driver",
         "street-light-flicker", "traffic-light-flicker", "traffic-light-off",
         "traffic-light-yellow-blinking", "vanish-actor"}

class Stop(Exception):
    pass

def require(ok, code):
    if not ok:
        raise Stop(code)

def text_field(field):
    head, sep, tail = field.partition(b"\0")
    require(not sep or not any(tail), "AMBIGUOUS_NUL_FIELD")
    try:
        return head.decode("utf-8", errors="strict")
    except UnicodeError:
        raise Stop("INVALID_PATH_ENCODING") from None

def octal(field):
    require(not field[0] & 128, "UNSUPPORTED_BINARY_NUMERIC_FIELD")
    trimmed = field.strip(b"\0 ")
    require(bool(trimmed) and all(48 <= c <= 55 for c in trimmed), "INVALID_OCTAL_FIELD")
    return int(trimmed, 8)

def classify(path, kind):
    require(bool(path) and not path.startswith("/") and "\\" not in path,
            "UNSAFE_HEADER_PATH")
    require(not re.match(r"^[A-Za-z]:", path), "DRIVE_PATH")
    original = path
    canonicalization = "NONE"
    if path.startswith("./"):
        path = path[2:]
        canonicalization = "STRIP_SINGLE_DOT_PREFIX"
    elif path.startswith("carlanomaly/"):
        path = path[len("carlanomaly/"):]
        canonicalization = "STRIP_SINGLE_CARLANOMALY_WRAPPER"
    if kind == b"5" and path.endswith("/"):
        path = path[:-1]
    if kind == b"5" and (original in {"./", "carlanomaly/"}):
        return "", None, "", canonicalization
    parts = path.split("/")
    require(all(p and p not in {".", ".."} and "\0" not in p for p in parts), "UNSAFE_OR_AMBIGUOUS_PATH_COMPONENT")
    require(parts[0] == "test", "NON_TEST_OR_UNCLASSIFIABLE_MEMBER")
    structural = len(parts) == 1
    if len(parts) > 1:
        require(parts[1] in {"normal", "anomaly"}, "INVALID_DIRECTORY_CONDITION")
        structural = len(parts) == 2
    if len(parts) > 2:
        require(parts[2] in TOWNS, "UNSUPPORTED_TOWN")
        structural = len(parts) == 3
    n = 4
    anomaly_type = "NORMAL"
    if len(parts) > 3 and parts[1] == "anomaly":
        require(parts[3] in TYPES, "UNSUPPORTED_ANOMALY_TYPE")
        anomaly_type = parts[3]
        structural = len(parts) == 4
        n = 5
    if structural:
        require(kind == b"5", "DATA_BEARING_STRUCTURAL_ROOT")
        return path, None, path, canonicalization
    require(len(parts) >= n and re.fullmatch(r"scenario-[1-9][0-9]*", parts[n - 1]) is not None,
            "INVALID_SCENARIO_PREFIX")
    scenario = "/".join(parts[:n])
    require(len(parts) != n or kind == b"5", "DATA_BEARING_SCENARIO_ROOT")
    # Child path components are used only for safety and duplicate detection.
    # They are never parsed as sensor/tick/annotation information or persisted.
    row = {"scenario_id": scenario, "town": parts[2], "directory_condition": parts[1], "anomaly_type": anomaly_type}
    return path, row, None, canonicalization

def header_fields(block):
    require(len(block) == 512, "TRUNCATED_TAR_HEADER")
    declared = octal(block[124:136])
    checksum = octal(block[148:156])
    require(sum(block[:148]) + 8 * 32 + sum(block[156:]) == checksum, "TAR_HEADER_CHECKSUM_MISMATCH")
    kind = block[156:157]
    # Every other type stops before resolving paths or reading any body.
    require(kind in {b"0", b"\0", b"5"}, "STOP_UNSUPPORTED_TAR_TYPEFLAG")
    magic = block[257:263]
    require(magic in {b"ustar\0", b"ustar ", b"\0" * 6}, "UNSUPPORTED_TAR_HEADER_FORMAT")
    require(magic != b"ustar\0" or block[263:265] == b"00", "UNSUPPORTED_USTAR_VERSION")
    require(not any(block[157:257]), "UNEXPECTED_LINK_NAME")
    name = text_field(block[:100])
    prefix = text_field(block[345:500]) if magic == b"ustar\0" else ""
    return (prefix + "/" if prefix else "") + name, kind, declared
