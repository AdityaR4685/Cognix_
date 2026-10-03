"""Exactly one range GET, two historically exposed headers, immediate stop.

No inventory walker, body interpretation, redirect, retry, fallback or cache.
The exclusive attempt marker is irrevocable, including on transport failure.
"""
import json
import time
import urllib.request
from pathlib import Path
import header_parser as p
import integrity as i

HERE = Path(__file__).resolve().parent
EXPECTED = (
    ("test/anomaly/Town01/change-weather/scenario-1/rgb-front/", b"5", 0),
    ("test/anomaly/Town01/change-weather/scenario-1/rgb-front/000000.jpg", b"0", 238500),
)


def empty_ledger():
    return {"compressed_bytes_requested": 65536, "compressed_bytes_received": 0,
            "compressed_bytes_consumed": 0, "decompressed_bytes_consumed": 0,
            "decompressed_bytes_produced": 0, "header_bytes_consumed": 0,
            "headers_visited": 0, "headers_parsed": 0,
            "member_body_bytes_discarded": 0, "padding_bytes_discarded": 0,
            "member_body_bytes_transited": 0, "requests": [],
            "stop_reason": "IN_PROGRESS", "RNG_instantiated": False,
            "PCG64_instantiated": False, "partition_executed": False,
            "inventory_created": False, "payload_decoded_or_inspected": False,
            "member_bodies_persisted": False, "archive_cached": False,
            "science_metrics_model_randomization_rules_changed": False,
            "no_retry_fallback_or_larger_range": True,
            "attestation_scope": "Executed validation and scoped hashes; not an OS-wide access trace."}


def caps():
    return {"compressed_bytes_max": 65536, "uncompressed_bytes_max": 1024,
            "volatile_chunk_bytes_max": 65536, "wall_clock_seconds_max": 90}


def initial_headers(reader, ledger, events):
    for sequence, (expected_path, expected_kind, expected_size) in enumerate(EXPECTED, 1):
        offset = ledger["decompressed_bytes_consumed"]
        block = reader.exact(512)
        ledger["headers_visited"] += 1
        ledger["header_bytes_consumed"] += 512
        event = {"sequence": sequence, "decompressed_header_offset": offset,
                 "compressed_bytes_received": ledger["compressed_bytes_received"],
                 "compressed_bytes_consumed": ledger["compressed_bytes_consumed"],
                 "typeflag_hex": block[156:157].hex(), "typeflag_repr": repr(block[156:157]),
                 "magic_hex": block[257:263].hex(), "magic_repr": repr(block[257:263]),
                 "version_hex": block[263:265].hex(), "version_repr": repr(block[263:265]),
                 "raw_header_hex": block.hex(), "raw_size_field_hex": block[124:136].hex(),
                 "raw_checksum_field_hex": block[148:156].hex(),
                 "checksum_verdict": "NOT_REACHED", "member_body_bytes_discarded": 0,
                 "padding_bytes_discarded": 0, "parse_verdict": "IN_PROGRESS"}
        try:
            event["declared_body_size"] = p.octal(block[124:136])
            stored = p.octal(block[148:156])
            calculated = sum(block[:148]) + 8 * 32 + sum(block[156:])
            event.update(checksum_stored=stored, checksum_calculated=calculated,
                         checksum_verdict="PASS" if stored == calculated else "FAIL")
            path, kind, size = p.header_fields(block)
            ledger["headers_parsed"] += 1
            p.require(kind != b"5" or size == 0, "NONZERO_DIRECTORY_BODY")
            # Invoke the original path restrictions, discard all classification output.
            p.classify(path, kind)
            event["path"] = path
            p.require((path, kind, size) == (expected_path, expected_kind, expected_size),
                      "STOP_HISTORICAL_INITIAL_HEADER_MISMATCH")
            p.require(block[257:263] == b"ustar ", "STOP_HISTORICAL_MAGIC_MISMATCH")
            event["parse_verdict"] = "PASS_HISTORICAL_HEADER_RECOGNIZED"
        except Exception as exc:
            event["parse_verdict"] = str(exc) if isinstance(exc, p.Stop) else type(exc).__name__
            raise
        finally:
            events.append(event)
    # The first directory has no body. Stop before reading the second member body.
    ledger["stop_reason"] = "HEADER_ACCESS_VALIDATED"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise p.Stop("UNAUTHORIZED_SOURCE_REDIRECT")


def main():
    verified = i.amendment_verify()
    auth = i.read(HERE / "authorization.json")
    p.require(auth["archive_attempts_allowed"] == 1 and auth["HTTP_range"] == "bytes=0-65535",
              "INVALID_ATTEMPT_AUTHORIZATION")
    p.require(auth["maximum_compressed_response_bytes"] == 65536, "INVALID_RESPONSE_CAP")
    ledger = empty_ledger()
    ledger.update(start_UTC=i.now(), pre_access_integrity=verified,
                  source_URL=auth["source_URL"], authorization_sha256=i.sha(HERE / "authorization.json"),
                  full_archive_integrity_verified=False, gzip_tar_EOF_validated=False)
    # Durable exclusive marker precedes the only network operation.
    i.write(HERE / "validation_attempt_started.json", {
        "start_UTC": ledger["start_UTC"], "attempt": 1, "no_restart": True,
        "amendment_manifest_sha256": verified["amendment_manifest_sha256"]})
    events = []
    response = None
    reader = None
    started = time.monotonic()
    try:
        opener = urllib.request.build_opener(NoRedirect())
        req = urllib.request.Request(auth["source_URL"], headers={
            "Range": "bytes=0-65535", "Accept-Encoding": "identity",
            "User-Agent": "CognixHeaderAccessValidation/001", "If-Match": auth["source_ETag"]})
        request_record = {"method": "GET", "range": "bytes=0-65535",
                          "bytes_requested": 65536, "bytes_received": 0,
                          "requested_UTC": i.now(), "HTTP_status": None}
        ledger["requests"].append(request_record)
        response = opener.open(req, timeout=45)
        request_record.update(HTTP_status=response.status, response_headers=dict(response.headers.items()))
        p.require(response.status == 206 and response.geturl() == auth["source_URL"], "UNEXPECTED_HTTP_STATUS_OR_URL")
        p.require(response.headers.get("Content-Range") == "bytes 0-65535/" + str(auth["advertised_archive_bytes"]),
                  "HTTP_CONTENT_RANGE_MISMATCH")
        p.require(response.headers.get("Content-Length") == "65536", "HTTP_RESPONSE_SIZE_MISMATCH")
        p.require(response.headers.get("ETag") == auth["source_ETag"], "HTTP_RELEASE_ETAG_MISMATCH")
        p.require(response.headers.get("Content-Encoding", "identity") == "identity", "HTTP_CONTENT_ENCODING_MISMATCH")
        reader = p.BoundedGzip(response, caps(), ledger)
        initial_headers(reader, ledger, events)
    except Exception as exc:
        ledger["stop_reason"] = str(exc) if isinstance(exc, p.Stop) else type(exc).__name__
        ledger["failure_type"] = type(exc).__name__
        # External exception text can contain payload: persist only its class.
        if isinstance(exc, urllib.error.HTTPError):
            ledger["requests"][0]["HTTP_status"] = exc.code
            exc.close()
    finally:
        if response is not None:
            response.close()
        ledger["stop_UTC"] = i.now()
        ledger["elapsed_seconds"] = time.monotonic() - started
        ledger["decompressed_bytes_produced"] = ledger["decompressed_bytes_consumed"]
        if reader is not None:
            ledger["compressed_prefix_sha256"] = reader.digest.hexdigest()
            ledger["digest_scope"] = "Received compressed prefix only; no release-integrity claim"
            ledger["volatile_compressed_bytes_not_consumed"] = len(reader.pending)
        ledger["headers"] = events
        try:
            ledger["post_access_integrity"] = i.amendment_verify()
        except Exception as exc:
            ledger["validation_stop_reason_before_integrity_failure"] = ledger["stop_reason"]
            ledger["stop_reason"] = "POST_ACCESS_INTEGRITY_FAILURE"
            ledger["integrity_failure_type"] = type(exc).__name__
        i.write(HERE / "validation_evidence.json", ledger)
        with (HERE / "header_access_ledger.jsonl").open("x", encoding="utf-8", newline="\n") as f:
            for event in events:
                f.write(json.dumps(event, sort_keys=True) + "\n")
        result = ledger["stop_reason"]
        report = (
            "# Header-access amendment 001: bounded validation\n\n"
            f"Terminal state: **{result}**. One range GET requested bytes 0-65535 (65,536 compressed bytes).\n\n"
            f"Received {ledger['compressed_bytes_received']:,}; decoder consumed {ledger['compressed_bytes_consumed']:,}; "
            f"decompressed {ledger['decompressed_bytes_produced']:,} bytes. "
            f"Visited {ledger['headers_visited']} headers; parsed {ledger['headers_parsed']}. "
            f"Discarded/transited {ledger['member_body_bytes_discarded']} decompressed member-body bytes and "
            f"{ledger['padding_bytes_discarded']} padding bytes.\n\n"
            "Exact structural header metadata, checksum verdicts, typeflags, magic/version fields, counters and HTTP "
            "response metadata are in validation_evidence.json and header_access_ledger.jsonl. "
            "The compressed prefix stayed volatile; neither compressed archive bytes nor member bodies were saved. "
            "Compressed data received but not consumed may contain opaque member-body data; it was never decompressed or interpreted.\n\n"
            "The amendment and synthetic tests were sealed and verified before this access. "
            "The historical stopped-attempt records, Amendment 003, its seed receipt, and protected source/configuration "
            "bindings were checked again afterward. This is header validation only; no inventory, eligible inventory, "
            "partition membership, RNG state, labels, sensor decoding, scientific execution, commit or push was produced. "
            "Full archive integrity and gzip/tar EOF were not validated.\n\n"
            "No retry, fallback, further member discovery or full sequential traversal follows. "
            "A failure requires separate authorization; this attempt is permanently spent. "
            "Attestation covers executed tools and scoped hashes, not an OS-wide access trace.\n")
        with (HERE / "report.md").open("x", encoding="utf-8", newline="\n") as f:
            f.write(report)
        i.seal(HERE, "VALIDATION_SHA256SUMS", ["validation_attempt_started.json", "validation_evidence.json",
               "header_access_ledger.jsonl", "report.md", "AMENDMENT_SHA256SUMS", "AMENDMENT_SHA256SUMS.sha256"])
        print(json.dumps({k: ledger[k] for k in ("stop_reason", "compressed_bytes_requested",
              "compressed_bytes_received", "compressed_bytes_consumed", "decompressed_bytes_produced",
              "headers_parsed", "member_body_bytes_discarded", "padding_bytes_discarded")}))
    return 0 if ledger["stop_reason"] == "HEADER_ACCESS_VALIDATED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
