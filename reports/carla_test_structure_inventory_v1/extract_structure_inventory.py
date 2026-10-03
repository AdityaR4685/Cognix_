"""One bounded official Base TEST traversal. Header paths only; no RNG or science.

No tarfile/dataset loader, extraction, archive cache, sensor decoder or body log.
Synthetic tests are separate. Existing protocol and amendment seals are read-only.
"""
import hashlib
import json
import re
import subprocess
import time
import urllib.request
import zlib
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROTOCOL = ROOT / "reports/carla_final_evaluation_preregistration_v1"
CHUNK = 65536
TOWNS = {"Town01", "Town02", "Town03", "Town04", "Town05", "Town10HD"}
TYPES = {"change-weather", "running-pedestrian", "spawn-props", "steer-driver",
         "street-light-flicker", "traffic-light-flicker", "traffic-light-off",
         "traffic-light-yellow-blinking", "vanish-actor"}


class Stop(Exception):
    pass


def require(ok, code):
    if not ok:
        raise Stop(code)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


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


def header_fields(block):
    require(len(block) == 512, "TRUNCATED_TAR_HEADER")
    declared = octal(block[124:136])
    checksum = octal(block[148:156])
    require(sum(block[:148]) + 8 * 32 + sum(block[156:]) == checksum, "TAR_HEADER_CHECKSUM_MISMATCH")
    kind = block[156:157]
    # Check extension type before resolving paths or reading any extension body.
    require(kind not in {b"x", b"g", b"L", b"K", b"S"}, "UNSUPPORTED_BODY_DEPENDENT_HEADER_EXTENSION")
    require(kind in {b"0", b"\0", b"5"}, "UNSUPPORTED_LINK_DEVICE_OR_HEADER_TYPE")
    magic = block[257:263]
    require(magic in {b"ustar\0", b"\0" * 6}, "UNSUPPORTED_TAR_HEADER_FORMAT")
    require(magic != b"ustar\0" or block[263:265] == b"00", "UNSUPPORTED_USTAR_VERSION")
    require(not any(block[157:257]), "UNEXPECTED_LINK_NAME")
    name = text_field(block[:100])
    prefix = text_field(block[345:500]) if magic == b"ustar\0" else ""
    return (prefix + "/" if prefix else "") + name, kind, declared


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


class BoundedGzip:
    def __init__(self, response, caps, ledger):
        self.response = response
        self.caps = caps
        self.ledger = ledger
        self.started = time.monotonic()
        self.decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
        self.pending = b""
        self.digest = hashlib.sha256()

    def clock_gate(self):
        require(time.monotonic() - self.started <= self.caps["wall_clock_seconds_max"], "WALL_CLOCK_CAP_EXCEEDED")

    def read(self, n):
        require(0 < n <= self.caps["volatile_chunk_bytes_max"], "VOLATILE_CHUNK_CAP_EXCEEDED")
        result = bytearray()
        while len(result) < n:
            self.clock_gate()
            if self.decoder.eof:
                self.ledger["gzip_trailer_validated"] = True
                require(not self.pending and not self.decoder.unused_data, "CONCATENATED_GZIP_OR_TRAILING_COMPRESSED_DATA")
                # HTTP's advertised Content-Length frames the single response.
                require(self.ledger["compressed_bytes_received"] == self.caps["compressed_bytes_max"], "COMPRESSED_SIZE_MISMATCH_AT_GZIP_EOF")
                require(self.response.read(1) == b"", "EXTRA_COMPRESSED_BYTE_AFTER_ADVERTISED_SIZE")
                self.ledger["HTTP_EOF_validated"] = True
                break
            if not self.pending:
                remaining = self.caps["compressed_bytes_max"] - self.ledger["compressed_bytes_received"]
                require(remaining > 0, "TRUNCATED_GZIP_AT_COMPRESSED_CAP")
                self.pending = self.response.read(min(CHUNK, remaining))
                self.clock_gate()
                require(bool(self.pending), "TRUNCATED_COMPRESSED_HTTP_STREAM")
                self.digest.update(self.pending)
                self.ledger["compressed_bytes_received"] += len(self.pending)
                self.ledger["requests"][0]["bytes_received"] += len(self.pending)
            before = len(self.pending)
            piece = self.decoder.decompress(self.pending, n - len(result))
            unused = self.decoder.unused_data if self.decoder.eof else b""
            self.pending = self.decoder.unconsumed_tail
            consumed = before - len(self.pending) - len(unused)
            self.ledger["compressed_bytes_consumed"] += consumed
            self.ledger["decompressed_bytes_consumed"] += len(piece)
            require(self.ledger["decompressed_bytes_consumed"] <= self.caps["uncompressed_bytes_max"], "DECOMPRESSED_CAP_EXCEEDED")
            result.extend(piece)
            if unused:
                raise Stop("CONCATENATED_GZIP_OR_TRAILING_COMPRESSED_DATA")
            require(piece or consumed or self.decoder.eof, "GZIP_NO_PROGRESS")
        return bytes(result)

    def exact(self, n):
        block = self.read(n)
        require(len(block) == n, "TRUNCATED_TAR_RECORD_OR_BODY")
        return block

    def discard(self, n, counter, event):
        while n:
            amount = min(n, CHUNK)
            chunk = self.read(amount)
            self.ledger[counter] += len(chunk)
            event[counter] += len(chunk)
            require(len(chunk) == amount, "TRUNCATED_TAR_MEMBER_BODY_OR_PADDING")
            n -= len(chunk)
            del chunk


def walk(reader, ledger, events):
    rows = {}
    seen_members = set()
    scenario_aliases = {}
    explicit_scenarios = set()
    while True:
        offset = ledger["decompressed_bytes_consumed"]
        block = reader.exact(512)
        if not any(block):
            ledger["tar_termination_bytes_consumed"] += 512
            require(not any(reader.exact(512)), "INVALID_TWO_BLOCK_TAR_TERMINATION")
            ledger["tar_termination_bytes_consumed"] += 512
            while True:
                padding = reader.read(512)
                ledger["tar_end_padding_bytes_discarded"] += len(padding)
                require(not any(padding), "NONPADDING_AFTER_TAR_TERMINATION")
                if not padding:
                    break
                require(len(padding) == 512, "MISALIGNED_TAR_END_PADDING")
            ledger["tar_full_termination_validated"] = True
            return rows
        ledger["headers_visited"] += 1
        ledger["header_bytes_consumed"] += 512
        event = {"sequence_number": ledger["headers_visited"], "decompressed_header_offset": offset,
                 "compressed_input_counter": ledger["compressed_bytes_consumed"],
                 "compressed_bytes_received_counter": ledger["compressed_bytes_received"],
                 "header_type": None, "declared_body_size": None,
                 "member_body_bytes_discarded": 0, "padding_bytes_discarded": 0,
                 "scenario_ID_or_structural_root": None, "parse_or_failure_code": "IN_PROGRESS"}
        try:
            require(ledger["headers_visited"] <= reader.caps["tar_header_records_max"], "HEADER_CAP_EXCEEDED")
            event["header_type"] = block[156:157].hex()
            event["declared_body_size"] = octal(block[124:136])
            path, kind, size = header_fields(block)
            require(size <= reader.caps["member_body_bytes_max"], "MEMBER_BODY_CAP_EXCEEDED")
            require(kind != b"5" or size == 0, "NONZERO_DIRECTORY_BODY")
            canonical, row, structural, transformation = classify(path, kind)
            event["canonicalization_rule"] = transformation
            ledger["canonicalization_counts"][transformation] = ledger["canonicalization_counts"].get(transformation, 0) + 1
            require(canonical not in seen_members, "DUPLICATE_MEMBER_PATH")
            seen_members.add(canonical)
            if row is not None:
                scenario = row["scenario_id"]
                event["scenario_ID_or_structural_root"] = scenario
                prior_alias = scenario_aliases.setdefault(scenario, transformation)
                require(prior_alias == transformation, "CONFLICTING_CANONICAL_SCENARIO_ALIASES")
                require(scenario not in rows or rows[scenario] == row, "CONFLICTING_SCENARIO_METADATA")
                if canonical == scenario and kind == b"5":
                    require(scenario not in explicit_scenarios, "DUPLICATE_EXPLICIT_SCENARIO_DIRECTORY")
                    explicit_scenarios.add(scenario)
                rows[scenario] = row
                ledger["unique_scenario_count"] = len(rows)
                require(len(rows) <= reader.caps["unique_scenarios_max"], "SCENARIO_CAP_EXCEEDED")
            else:
                event["scenario_ID_or_structural_root"] = structural
            del block, path, canonical
            reader.discard(size, "member_body_bytes_discarded", event)
            reader.discard((-size) % 512, "padding_bytes_discarded", event)
            event["parse_or_failure_code"] = "PASS"
        except Exception as exc:
            event["parse_or_failure_code"] = str(exc) if isinstance(exc, Stop) else type(exc).__name__
            raise
        finally:
            events.write(json.dumps(event, sort_keys=True) + "\n")
            events.flush()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise Stop("UNAUTHORIZED_SOURCE_REDIRECT")


def preflight():
    seal = (HERE / "PREFLIGHT_SHA256SUMS").read_text(encoding="utf-8")
    expected_seal = (HERE / "PREFLIGHT_SHA256SUMS.sha256").read_text().split()
    require(expected_seal == [sha(HERE / "PREFLIGHT_SHA256SUMS"), "PREFLIGHT_SHA256SUMS"], "PREFLIGHT_SEAL_MISMATCH")
    for line in seal.splitlines():
        digest, rel = line.split("  ", 1)
        p = (HERE / rel).resolve()
        require(p.is_relative_to(HERE) and sha(p) == digest, "PREFLIGHT_FILE_HASH_MISMATCH")
    authorization = read_json(HERE / "access_preflight.json")
    for rel, digest in authorization["protocol_bindings"].items():
        require(sha(PROTOCOL / rel) == digest, "PROTOCOL_BINDING_MISMATCH")
    for rel, digest in authorization["protected_source_config_bindings"].items():
        require(sha(ROOT / rel) == digest, "PROTECTED_SOURCE_CONFIG_MISMATCH")
    require(subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip() == authorization["authorized_execution_HEAD"], "EXECUTION_HEAD_MISMATCH")
    require(not subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain=v1", "--untracked-files=no"], text=True).strip(), "TRACKED_WORKTREE_MODIFIED")
    require(read_json(HERE / "amendment_003_integrity_recheck.json")["passed"], "AMENDMENT_003_RECHECK_FAILED")
    require(authorization["archive_attempts_allowed"] == 1, "INVALID_ATTEMPT_BUDGET")
    require(not (HERE / "archive_attempt_started.json").exists(), "ARCHIVE_ATTEMPT_ALREADY_STARTED_NO_RESTART")
    return authorization


def main():
    auth = preflight()
    ledger = {"authorization_reference": "access_preflight.json", "tool_source_sha256": sha(Path(__file__)),
              "protocol_sha256": auth["protocol_bindings"]["inventory_access_protocol.json"],
              "preflight_manifest_sha256": sha(HERE / "PREFLIGHT_SHA256SUMS"),
              "source_URL": auth["source_URL"], "publisher_release": auth["publisher_release"],
              "advertised_checksum_and_size": {"sha256": auth["advertised_sha256"], "bytes": auth["advertised_bytes"]},
              "start_UTC": now(), "stop_UTC": None, "requests": [],
              "compressed_bytes_received": 0, "compressed_bytes_consumed": 0,
              "decompressed_bytes_consumed": 0, "header_bytes_consumed": 0,
              "member_body_bytes_discarded": 0, "padding_bytes_discarded": 0,
              "tar_termination_bytes_consumed": 0, "tar_end_padding_bytes_discarded": 0,
              "headers_visited": 0, "unique_scenario_count": 0, "canonicalization_counts": {},
              "gzip_trailer_validated": False, "HTTP_EOF_validated": False,
              "tar_full_termination_validated": False, "compressed_stream_digest_validated": False,
              "complete": False, "failure_reason_or_success": "IN_PROGRESS",
              "no_retry_restart_or_fallback": True, "archive_cache_or_extracted_members_created": False,
              "RNG_instantiated": False, "partition_executed": False,
              "sensor_or_label_interpretation": False, "inference_conformal_training_tuning": False,
              "attestation_scope": "Executed extractor access; not an OS-wide access trace."}
    with (HERE / "archive_attempt_started.json").open("x", encoding="utf-8") as f:
        json.dump({"started_UTC": ledger["start_UTC"], "preflight_manifest_sha256": ledger["preflight_manifest_sha256"], "no_restart": True}, f, indent=2)
        f.write("\n")
    rows = {}
    reader = None
    response = None
    try:
        opener = urllib.request.build_opener(NoRedirect())
        req = urllib.request.Request(auth["source_URL"], headers={"Accept-Encoding": "identity", "User-Agent": "CognixStructureInventory/1", "If-Match": auth["source_ETag"]})
        request_record = {"method": "GET", "range": None, "HTTP_status": None, "bytes_received": 0, "requested_UTC": now()}
        ledger["requests"].append(request_record)
        response = opener.open(req, timeout=45)
        request_record.update(HTTP_status=response.status, response_headers=dict(response.headers.items()))
        require(response.status == 200 and response.geturl() == auth["source_URL"], "UNEXPECTED_HTTP_STATUS_OR_URL")
        require(response.headers.get("Content-Length") == str(auth["advertised_bytes"]), "HTTP_SOURCE_SIZE_MISMATCH")
        require(response.headers.get("ETag") == auth["source_ETag"], "HTTP_RELEASE_ETAG_MISMATCH")
        require(response.headers.get("Content-Encoding", "identity") == "identity", "HTTP_CONTENT_ENCODING_MISMATCH")
        reader = BoundedGzip(response, auth["mechanical_bounds"], ledger)
        with (HERE / "header_access_ledger.jsonl").open("x", encoding="utf-8") as events:
            rows = walk(reader, ledger, events)
        ledger["compressed_stream_digest"] = reader.digest.hexdigest()
        require(ledger["compressed_stream_digest"] == auth["advertised_sha256"], "ARCHIVE_SHA256_MISMATCH")
        ledger["compressed_stream_digest_validated"] = True
        require(len(rows) == 627 and sum(r["directory_condition"] == "normal" for r in rows.values()) == 107 and sum(r["directory_condition"] == "anomaly" for r in rows.values()) == 520, "PUBLISHER_COUNT_MISMATCH")
        require(set(auth["known_exact_exclusions"]).issubset(rows), "HISTORICAL_EXCLUSION_ID_MISSING")
        source = [rows[k] for k in sorted(rows)]
        eligible = [r for r in source if r["scenario_id"] not in auth["known_exact_exclusions"]]
        write_json(HERE / "source_scenario_inventory.json", source)
        write_json(HERE / "historical_exclusion_decisions.json", {"exact_exclusions": auth["known_exact_exclusions"], "excluded_from_both_final_roles": True, "historical_file_level_ledger_complete": False, "additional_authenticated_exposures_established_this_milestone": [], "no_guessed_IDs_or_families": True, "source_record_sha256": auth["protocol_bindings"]["test_exposure_record.json"]})
        write_json(HERE / "eligible_scenario_inventory.json", eligible)
        ledger["complete"] = True
        ledger["failure_reason_or_success"] = "SUCCESS_STRUCTURE_ONLY"
        ledger["source_count"] = len(source)
        ledger["eligible_count"] = len(eligible)
        ledger["inventory_hashes"] = {n: sha(HERE / n) for n in ("source_scenario_inventory.json", "historical_exclusion_decisions.json", "eligible_scenario_inventory.json")}
        # Independent seal only after all traversal/digest/count/exclusion gates.
        manifest = "".join(d + "  " + n + "\n" for n, d in sorted(ledger["inventory_hashes"].items()))
        (HERE / "INVENTORY_SHA256SUMS").write_text(manifest, encoding="utf-8")
        (HERE / "INVENTORY_SHA256SUMS.sha256").write_text(sha(HERE / "INVENTORY_SHA256SUMS") + "  INVENTORY_SHA256SUMS\n", encoding="utf-8")
    except Exception as exc:
        ledger["failure_reason_or_success"] = str(exc) if isinstance(exc, Stop) else type(exc).__name__
        ledger["failure_type"] = type(exc).__name__
        # Exception messages from external libraries can contain payload bytes;
        # persist only the safe class/code, never their arbitrary text.
    finally:
        if response is not None:
            response.close()
        if reader is not None:
            ledger["compressed_stream_digest"] = reader.digest.hexdigest()
            ledger["digest_scope"] = "Entire received compressed stream" if ledger["complete"] else "Received compressed prefix only; not release integrity proof"
            ledger["elapsed_seconds"] = time.monotonic() - reader.started
        ledger["stop_UTC"] = now()
        ledger["budget_usage"] = {"caps": auth["mechanical_bounds"], "compressed_bytes": ledger["compressed_bytes_received"], "decompressed_bytes": ledger["decompressed_bytes_consumed"], "headers": ledger["headers_visited"], "unique_scenarios": ledger["unique_scenario_count"]}
        ledger["no_eligible_inventory_seal_on_failure"] = ledger["complete"] or not (HERE / "INVENTORY_SHA256SUMS").exists()
        write_json(HERE / "access_ledger.json", ledger)
        print(json.dumps({"complete": ledger["complete"], "status": ledger["failure_reason_or_success"], "compressed_bytes_received": ledger["compressed_bytes_received"], "decompressed_bytes_consumed": ledger["decompressed_bytes_consumed"], "headers_visited": ledger["headers_visited"], "unique_scenario_count": ledger["unique_scenario_count"]}))
    return 0 if ledger["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
