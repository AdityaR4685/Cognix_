"""TRAIN-only, bounded streaming gzip/TAR acquisition for CarlAnomaly.

No model fitting, calibration or recipe decisions. The caller supplies a
frozen protocol. Completed archive members are counted only after payload and
padding consumption; scenario completion is delegated to a validation callback
at a subsequent scenario header (or valid archive termination).
"""
from __future__ import annotations

import hashlib
import re
import tarfile
import zlib
from pathlib import Path, PurePosixPath


class AcquisitionError(RuntimeError):
    """Unsafe transport/archive state; preserve staged bytes and fail loudly."""


def validate_range_response(status, headers, expected_start, expected_total=None):
    """Reject full responses, wrong offsets, encoded bodies and changing size."""
    if status != 206:
        raise AcquisitionError(f"Range request returned {status}, expected 206")
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", headers.get("Content-Range", ""))
    if not match:
        raise AcquisitionError("Missing/malformed Content-Range")
    start, end, total = map(int, match.groups())
    if start != expected_start or end < start or end >= total:
        raise AcquisitionError("Content-Range does not match requested offset")
    if expected_total is not None and total != expected_total:
        raise AcquisitionError("Archive total changed during acquisition")
    if headers.get("Content-Encoding", "identity").lower() != "identity":
        raise AcquisitionError("HTTP content encoding would change compressed offsets")
    length = headers.get("Content-Length")
    if length is not None and int(length) != end - start + 1:
        raise AcquisitionError("Content-Length disagrees with Content-Range")
    return end, total


def train_member_path(name):
    """Canonical TRAIN member components; no traversal, links or TEST paths."""
    if "\\" in name or name.startswith("/"):
        raise AcquisitionError(f"Unsafe archive path: {name!r}")
    raw = name.rstrip("/").split("/")
    if any(p in ("", ".", "..") or ":" in p for p in raw):
        raise AcquisitionError(f"Unsafe archive components: {name!r}")
    parts = PurePosixPath(name).parts
    if not parts or parts[0] != "train":
        raise AcquisitionError(f"Non-TRAIN archive member refused: {name!r}")
    return parts


class TrainTarStream:
    """Incremental TAR state machine. No growing whole-stream buffer.

    `on_boundary(record, next_header)` validates the just-finished scenario.
    Existing scenarios have paths in `reuse` and are never written. New files
    use exclusive creation or verified reuse of a previously staged file on
    process recovery. No existing artifact is truncated or deleted.
    """
    def __init__(self, extraction_root, reuse, on_boundary, target=10):
        self.root = Path(extraction_root)
        self.reuse = {k: Path(v) for k, v in reuse.items()}
        self.on_boundary = on_boundary
        self.target = target
        self.state = "header"
        self.header = bytearray()
        self.offset = 0
        self.remaining = self.padding = 0
        self.member = None
        self.file = None
        self.destination = None
        self.digest = None
        self.meta_payload = bytearray()
        self.pax_global = {}
        self.pax_local = {}
        self.long_name = None
        self.current = None
        self.scenarios = []
        self.done = False
        self.tar_end = False
        self.zero_headers = 0
        self.members = 0
        self.trailing_boundary = None

    def _finish_scenario(self, boundary):
        if self.current is not None:
            self.on_boundary(self.current, boundary)
            self.current["complete"] = True
            self.current["completion_evidence"] = boundary
            if len(self.scenarios) == self.target:
                self.done = True
                self.trailing_boundary = boundary

    def _begin(self, block, header_offset):
        if block == bytes(512):
            self.zero_headers += 1
            if self.zero_headers == 2:
                self.tar_end = True
            return
        if self.zero_headers or self.tar_end:
            raise AcquisitionError("Unexpected data after TAR end marker")
        try:
            info = tarfile.TarInfo.frombuf(block, "utf-8", "strict")
        except (tarfile.TarError, UnicodeError, ValueError) as exc:
            raise AcquisitionError(f"Bad TAR header at {header_offset}: {exc}") from exc
        if info.size < 0:
            raise AcquisitionError("Negative TAR member size")
        self.member = info
        self.members += 1
        self.remaining = info.size
        self.padding = (-info.size) % 512
        self.digest = hashlib.sha256()
        self.meta_payload = bytearray()
        if info.type in (tarfile.XHDTYPE, tarfile.XGLTYPE, tarfile.GNUTYPE_LONGNAME):
            if info.size > 16 * 1024 * 1024:
                raise AcquisitionError("Oversized TAR extension record")
        else:
            info.name = self.long_name or self.pax_local.get("path", self.pax_global.get("path", info.name))
            declared_size = self.pax_local.get("size", self.pax_global.get("size"))
            if declared_size is not None:
                if int(declared_size) < 0:
                    raise AcquisitionError("Negative PAX size")
                info.size = self.remaining = int(declared_size)
                self.padding = (-info.size) % 512
            self.long_name = None
            self.pax_local = {}
            if not (info.isfile() or info.isdir()):
                raise AcquisitionError(f"Archive links/special members refused: {info.name}")
            if info.isdir() and info.size:
                raise AcquisitionError("Nonempty directory payload refused")
            parts = train_member_path(info.name)
            if len(parts) >= 3:
                sid = "/".join(parts[1:3])
                if self.current is None or sid != self.current["scenario_id"]:
                    if any(s["scenario_id"] == sid for s in self.scenarios):
                        raise AcquisitionError("Scenario reappeared after its archive boundary")
                    boundary = {"kind": "subsequent_scenario_header", "next_scenario_id": sid,
                                "tar_header_offset": header_offset, "member": info.name}
                    self._finish_scenario(boundary)
                    if self.done:
                        return
                    actual = self.reuse.get(sid, self.root.joinpath(*parts[:3]))
                    self.current = {"scenario_id": sid, "town": parts[1],
                                    "archive_ordinal": len(self.scenarios) + 1,
                                    "first_tar_member_offset": header_offset,
                                    "actual_path": str(actual), "reused": sid in self.reuse,
                                    "rgb_ticks": set(), "seg_ticks": set(), "source_files": {}}
                    self.scenarios.append(self.current)
                if info.isfile():
                    if len(parts) < 4:
                        raise AcquisitionError("Scenario path is a regular file")
                    self.destination = Path(self.current["actual_path"]).joinpath(*parts[3:])
                    self.verify_existing = self.destination.exists()
                    # Validate resolved destination containment before any write.
                    actual = Path(self.current["actual_path"]).resolve()
                    if not self.destination.resolve().is_relative_to(actual):
                        raise AcquisitionError("Extraction destination escapes scenario root")
                    if sid not in self.reuse and not self.destination.exists():
                        self.destination.parent.mkdir(parents=True, exist_ok=True)
                        self.file = self.destination.open("xb")
        self.state = "payload" if self.remaining else "padding"
        if not self.remaining and not self.padding:
            self._finish_member()

    @staticmethod
    def _pax(payload):
        result = {}
        pos = 0
        while pos < len(payload):
            space = payload.find(b" ", pos)
            if space < 0:
                raise AcquisitionError("Malformed PAX length")
            try:
                length = int(payload[pos:space])
            except ValueError as exc:
                raise AcquisitionError("Malformed PAX length") from exc
            end = pos + length
            if length <= space - pos + 1 or end > len(payload) or payload[end-1:end] != b"\n":
                raise AcquisitionError("Malformed PAX record")
            key, sep, value = payload[space+1:end-1].partition(b"=")
            if not sep:
                raise AcquisitionError("Malformed PAX key/value")
            result[key.decode("utf-8")] = value.decode("utf-8")
            pos = end
        return result

    def _finish_member(self):
        info = self.member
        if info.type in (tarfile.XHDTYPE, tarfile.XGLTYPE):
            parsed = self._pax(bytes(self.meta_payload))
            if info.type == tarfile.XGLTYPE:
                self.pax_global.update(parsed)
            else:
                self.pax_local = parsed
        elif info.type == tarfile.GNUTYPE_LONGNAME:
            self.long_name = bytes(self.meta_payload).rstrip(b"\0").decode("utf-8")
        elif info.isfile() and self.current is not None and self.destination is not None:
            if self.file:
                self.file.close()
                self.file = None
            relative = self.destination.relative_to(Path(self.current["actual_path"])).as_posix()
            digest = self.digest.hexdigest()
            if relative in self.current["source_files"]:
                raise AcquisitionError(f"Duplicate archive member: {info.name}")
            # Verify reused or recovered files; never overwrite a mismatch.
            if self.destination.stat().st_size != info.size:
                raise AcquisitionError(f"Existing extraction size mismatch: {self.destination}")
            if self.current["reused"] or getattr(self, "verify_existing", False):
                h = hashlib.sha256()
                with self.destination.open("rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
                if h.hexdigest() != digest:
                    raise AcquisitionError(f"Existing extraction content mismatch: {self.destination}")
            self.current["source_files"][relative] = {"bytes": info.size, "sha256": digest}
            for folder, suffix, key in [("rgb-front", ".jpg", "rgb_ticks"),
                                         ("segmentation-front", ".png", "seg_ticks")]:
                if relative.startswith(folder + "/") and relative.endswith(suffix):
                    stem = Path(relative).stem
                    if len(stem) != 6 or not stem.isdigit():
                        raise AcquisitionError(f"Unexpected frame name: {relative}")
                    self.current[key].add(int(stem))
        self.destination = None
        self.member = None
        self.state = "header"

    def feed(self, data):
        view = memoryview(data)
        pos = 0
        while pos < len(view) and not self.done:
            if self.tar_end:
                if any(view[pos:]):
                    raise AcquisitionError("Nonzero trailing TAR data")
                self.offset += len(view) - pos
                return
            if self.state == "header":
                take = min(512 - len(self.header), len(view) - pos)
                self.header.extend(view[pos:pos+take]); pos += take; self.offset += take
                if len(self.header) == 512:
                    block = bytes(self.header); self.header.clear()
                    self._begin(block, self.offset - 512)
            elif self.state == "payload":
                take = min(self.remaining, len(view) - pos)
                payload = view[pos:pos+take]
                if self.file:
                    self.file.write(payload)
                self.digest.update(payload)
                if self.member.type in (tarfile.XHDTYPE, tarfile.XGLTYPE, tarfile.GNUTYPE_LONGNAME):
                    self.meta_payload.extend(payload)
                self.remaining -= take; pos += take; self.offset += take
                if not self.remaining:
                    self.state = "padding"
                    if not self.padding:
                        self._finish_member()
            else:
                take = min(self.padding, len(view) - pos)
                if any(view[pos:pos+take]):
                    raise AcquisitionError("Nonzero TAR member padding")
                self.padding -= take; pos += take; self.offset += take
                if not self.padding:
                    self._finish_member()

    def finish_gzip(self):
        if not self.tar_end or self.state != "header" or self.header:
            raise AcquisitionError("Gzip ended without valid TAR termination")
        self._finish_scenario({"kind": "valid_gzip_and_tar_termination", "tar_offset": self.offset})

    def close(self):
        if self.file:
            self.file.close()
            self.file = None


class GzipTrainStream:
    """One live gzip decoder with at most 1 MiB decompressed output per call."""
    def __init__(self, parser):
        self.parser = parser
        self.decoder = zlib.decompressobj(31)
        self.compressed_bytes = 0

    def feed(self, chunk):
        self.compressed_bytes += len(chunk)
        pending = chunk
        while pending and not self.parser.done:
            try:
                raw = self.decoder.decompress(pending, 1 << 20)
            except zlib.error as exc:
                raise AcquisitionError(f"Gzip stream corrupt: {exc}") from exc
            self.parser.feed(raw)
            pending = self.decoder.unconsumed_tail
        if self.decoder.eof and not self.parser.done:
            if any(self.decoder.unused_data):
                raise AcquisitionError("Additional gzip members refused")
            self.parser.finish_gzip()
