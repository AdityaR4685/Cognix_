"""Phase A only: verify prior metadata, run synthetic tests and seal; no network."""
import subprocess
import sys
from pathlib import Path
import integrity as i

HERE = Path(__file__).resolve().parent


def main():
    assert not (HERE / "AMENDMENT_SHA256SUMS").exists(), "Amendment already sealed; verify only"
    assert not (HERE / "validation_attempt_started.json").exists(), "Access already spent"
    prior_preflight = i.manifest_verify(i.PRIOR, "PREFLIGHT_SHA256SUMS")
    prior_audit = i.manifest_verify(i.PRIOR, "AUDIT_SHA256SUMS")
    protocol_files = i.manifest_verify(i.PROTOCOL, "SHA256SUMS")
    old = i.read(i.PRIOR / "access_preflight.json")
    bindings = {}
    for rel, digest in old["protocol_bindings"].items():
        p = i.PROTOCOL / rel
        assert i.sha(p) == digest
        bindings[p.relative_to(i.ROOT).as_posix()] = digest
    for rel, digest in old["protected_source_config_bindings"].items():
        assert i.sha(i.ROOT / rel) == digest
        bindings[rel] = digest
    prior_files = [p for p in i.PRIOR.iterdir() if p.is_file()]
    for p in prior_files:
        bindings[p.relative_to(i.ROOT).as_posix()] = i.sha(p)
    git_head = subprocess.check_output(["git", "-C", str(i.ROOT), "rev-parse", "HEAD"], text=True).strip()
    assert git_head == old["authorized_execution_HEAD"]
    i.write(HERE / "protected_bindings.json", {"files": bindings, "git_HEAD": git_head,
            "prior_preflight_manifest_entries_verified": prior_preflight,
            "prior_audit_manifest_entries_verified": prior_audit,
            "protocol_manifest_entries_verified": protocol_files,
            "prior_attempt_files_protected": len(prior_files),
            "protected_source_config_bindings": len(old["protected_source_config_bindings"]),
            "scope": "Protocol, prior attempt, source/config metadata only; no raw archive/sensor/model bytes."})
    i.write(HERE / "phase_A_integrity.json", i.protected_verify())
    request = Path("C:/Users/Aditya/.codex/attachments/a6b4459b-7799-4bd5-bf5d-7e96a38b8dae/Pasted text.txt")
    with (HERE / "user_request.txt").open("x", encoding="utf-8", newline="\n") as f:
        f.write(request.read_text(encoding="utf-8-sig"))
    i.write(HERE / "authorization.json", {
        "authorization": "Explicit user request, preserved in user_request.txt",
        "milestone": "HEADER-ACCESS AMENDMENT + BOUNDED VALIDATION ONLY",
        "source_URL": old["source_URL"], "source_ETag": old["source_ETag"],
        "advertised_archive_bytes": old["advertised_bytes"],
        "historical_advertised_archive_sha256": old["advertised_sha256"],
        "source_metadata_basis": "Existing sealed prior attempt only; no new HEAD/checksum request",
        "archive_attempts_allowed": 1, "HTTP_method": "GET", "HTTP_range": "bytes=0-65535",
        "maximum_compressed_response_bytes": 65536, "max_decompressed_bytes": 1024,
        "max_headers": 2, "redirects_retries_fallbacks_larger_ranges_allowed": False,
        "stop_on_recognition_of_both_historical_headers": True,
        "known_headers": [
            {"path": "test/anomaly/Town01/change-weather/scenario-1/rgb-front/", "size": 0, "typeflag_hex": "35", "magic_hex": "757374617220"},
            {"path": "test/anomaly/Town01/change-weather/scenario-1/rgb-front/000000.jpg", "size": 238500, "typeflag_hex": "30", "magic_hex": "757374617220"}],
        "historical_header_source": "User-provided historical official TEST probe evidence; no fresh parser-development exposure",
        "scientific_partition_conformal_metric_model_randomization_rules_changed": False,
        "Amendment_003_modified": False, "seed_modified": False,
        "inventory_RNG_partition_payload_science_commit_push_authorized": False})
    i.write(HERE / "amendment_record.json", {
        "id": "header_access_amendment_001", "created_UTC": i.now(),
        "client_date": "2026-10-03", "client_timezone": "Asia/Calcutta",
        "reason": "Previously sealed parser was too restrictive for the already-observed official b'ustar ' tar-header representation.",
        "prior_parser_sha256": i.sha(i.PRIOR / "extract_structure_inventory.py"),
        "prior_stop_reason": "UNSUPPORTED_TAR_HEADER_FORMAT",
        "tar_header_changes_only": ["Accept b'ustar ' alongside prior b'ustar\\0' and zero-magic forms",
                                    "Reject every non-0/non-NUL/non-5 flag as STOP_UNSUPPORTED_TAR_TYPEFLAG"],
        "regular_typeflags_hex": ["30", "00"], "directory_typeflags_hex": ["35"],
        "directory_semantics": "Structural metadata, zero body permitted; preserve NONZERO_DIRECTORY_BODY guard; no payload decoding",
        "regular_semantics": "Header metadata only; opaque body transit only if needed to advance. This two-header validation stops before its first regular body.",
        "space_ustar_field_interpretation": "Name field only, no POSIX prefix. Preserve POSIX-only 00 version gate; other versions on the newly authorized magic are recorded, not used to enable extensions.",
        "preserved_guards": ["checksum", "octal size", "block alignment/padding", "fail closed",
                             "path parsing", "access counters", "payload noninterpretation", "POSIX version/prefix", "no link name"],
        "other_types_or_formats_authorized": False,
        "unmodified_records": "All prior attempt files and all Amendment 003 files remain byte-identical. No scientific or protocol reseal.",
        "phase_A_TEST_accessed": False, "full_inventory_traversal_authorized": False,
        "seal_requirement": "Parser/tests/driver/authorization and their synthetic results sealed and verified before any new TEST access"})
    if not (HERE / "synthetic_test_results.json").exists():
        subprocess.run([sys.executable, "-B", str(HERE / "test_header_parser.py")], check=True)
    results = i.read(HERE / "synthetic_test_results.json")
    assert results["passed"] and results["synthetic_only"] and not results["official_TEST_accessed"]
    assert results["parser_sha256"] == i.sha(HERE / "header_parser.py")
    assert results["test_sha256"] == i.sha(HERE / "test_header_parser.py")
    assert results["validator_sha256"] == i.sha(HERE / "validate_headers.py")
    names = [p.name for p in HERE.iterdir() if p.is_file()]
    i.seal(HERE, "AMENDMENT_SHA256SUMS", names)
    print(i.amendment_verify())


if __name__ == "__main__":
    main()
