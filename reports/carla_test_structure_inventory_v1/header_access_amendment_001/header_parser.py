"""Amended header acceptance only; reuse the immutable sealed parser's guards.

Importing the prior module performs no archive access. Never call its walk/main.
Space-terminated ustar uses the name field, not the POSIX prefix field: the
corresponding old-GNU region has different semantics. No extension is supported.
"""
import importlib.util
from pathlib import Path

SOURCE = Path(__file__).resolve().parent.parent / "extract_structure_inventory.py"
spec = importlib.util.spec_from_file_location("sealed_structure_parser", SOURCE)
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
Stop = original.Stop
require = original.require
octal = original.octal
text_field = original.text_field
classify = original.classify
BoundedGzip = original.BoundedGzip


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
