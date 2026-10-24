"""Bounded, label-blind CarlAnomaly archive replay and legacy synthetic source."""
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import time
import numpy as np
from PIL import Image
from cognix.runtime.replay import ObservationEnvelope, SourceError
from .real_features import camera_embedding_features, segmentation_histogram_features, imu_window_features
from .runtime_artifacts import authenticated, load_sanitizer


@dataclass(frozen=True)
class ArchiveConfig:
    archive: Path
    split: str = "TEST"
    max_samples: int = 300
    max_member_bytes: int = 16 * 1024 * 1024
    max_pixels: int = 16_000_000
    max_scan_seconds: float = 600


def archive_catalog(repo, split):
    report = Path(repo) / "reports/carla_train_full_v1"
    if split == "TRAIN":
        path = report / "gate1_local_extraction_v8/complete_inventory_and_clean_store.json"
        authenticated(path, "8f483022518f7cc42d2ca722e326ef56a3073b87dac500bcc4dcde6125f723ba")
        value = json.loads(path.read_text())
        rows = {"train/" + b["scenario_id"]: b["rows"] for b in value["completed_blocks"]}
        identity = value["source_binding"]["identity"]
        return rows, {"bytes": identity["bytes"], "historical_archive_sha256": identity["sha256"]}
    if split == "TEST":
        path = report / "graph_test_structure_inspection_v1/full_metadata_inventory.json"
        authenticated(path, "8aa1afac7b8042b83560207def8a887c760eb9cfd53f2c2413757a33f0bd2651")
        value = json.loads(path.read_text())
        return {s: 299 for s in value["scenario_file_counts"]}, {
            "bytes": value["archive_bytes"], "historical_archive_sha256": value["archive_sha256"]}
    raise SourceError("Dataset split must be TRAIN or TEST")


class CarlAnomalyReplaySource:
    def __init__(self, config: ArchiveConfig, catalog, sanitizer, identity=None):
        self.config = config
        self.archive = Path(config.archive).resolve()
        if not self.archive.is_file():
            raise SourceError(f"CarlAnomaly archive not configured/found: {self.archive}")
        if config.split not in ("TRAIN", "TEST") or not 1 <= config.max_samples <= 3000:
            raise SourceError("Invalid split or bounded replay window (1..3000 samples)")
        self._stat = self.archive.stat()
        if identity and identity["bytes"] != self._stat.st_size:
            raise SourceError("CarlAnomaly archive size does not match authenticated inventory")
        self._catalog = dict(catalog)
        self._sanitize = sanitizer
        self._cache_sequence = None
        self._cache = []
        self._provenance = {}
        self.info = {"mode": "DATASET", "domain": "autonomous_driving", "source_id": "CarlAnomaly Base",
                     "dataset": "CarlAnomaly Base", "split": config.split, "archive": str(self.archive),
                     "replay_window_limit": config.max_samples, "agent_order": ["Camera", "IMU", "Seg"]}
        self._identity = {"archive_path": str(self.archive), "archive_bytes": self._stat.st_size,
                          "archive_mtime_ns": self._stat.st_mtime_ns, **(identity or {}),
                          "archive_hash_verification": "historical inventory binding and current size/stat; full archive not rehashed"}

    @classmethod
    def configured(cls, repo, split="TEST", archive=None, max_samples=300):
        path = archive or os.environ.get(f"CARLANOMALY_{split}_ARCHIVE")
        if not path:
            raise SourceError(f"Set CARLANOMALY_{split}_ARCHIVE or supply an explicit archive path")
        # Check existence before looking at artifact metadata, for a useful error.
        if not Path(path).is_file():
            raise SourceError(f"CarlAnomaly archive not configured/found: {path}")
        catalog, identity = archive_catalog(repo, split)
        return cls(ArchiveConfig(Path(path), split, max_samples), catalog, load_sanitizer(Path(repo)), identity)

    def sequences(self):
        return sorted(self._catalog)

    def sample_count(self, sequence_id):
        if sequence_id not in self._catalog:
            raise SourceError(f"Scenario missing from frozen inventory: {sequence_id}")
        return min(self.config.max_samples, self._catalog[sequence_id])

    def _image(self, payload):
        with Image.open(io.BytesIO(payload)) as image:
            if image.width * image.height > self.config.max_pixels:
                raise SourceError("Image exceeds replay pixel limit")
            image.load()
            return np.asarray(image)

    def _prepare(self, sequence_id):
        count = self.sample_count(sequence_id)
        prefix = sequence_id + "/"
        camera, seg, hashes, imu = {}, {}, {}, None
        start = time.perf_counter()
        started = False
        current = self.archive.stat()
        if (current.st_size, current.st_mtime_ns) != (self._stat.st_size, self._stat.st_mtime_ns):
            raise SourceError("Archive changed since source configuration")
        # No extract/extractall. Clear tar header inventory to keep memory bounded.
        with tarfile.open(self.archive, mode="r|gz") as tar:
            for member in tar:
                tar.members.clear()
                if time.perf_counter() - start > self.config.max_scan_seconds:
                    raise SourceError("Archive scan exceeded configured time limit")
                name = member.name.removeprefix("./")
                if not name.startswith(prefix):
                    if started:
                        break  # Authenticated archives group each scenario contiguously.
                    continue
                started = True
                parts = PurePosixPath(name).parts
                if ".." in parts or name.startswith("/"):
                    raise SourceError("Invalid archive member path")
                relative = name[len(prefix):]
                sensor, _, frame = relative.partition("/")
                tick = None
                if sensor in ("rgb-front", "segmentation-front") and frame:
                    stem, extension = os.path.splitext(frame)
                    if not stem.isdigit() or extension.lower() not in (".jpg", ".png"):
                        continue
                    tick = int(stem)
                    if not 1 <= tick <= count:
                        continue
                elif relative != "imu.feather":
                    continue  # Includes all label and anomaly-front members.
                if not member.isfile() or not 0 < member.size <= self.config.max_member_bytes:
                    raise SourceError(f"Required sensor member invalid/too large: {name}")
                if name in hashes:
                    raise SourceError(f"Duplicate sensor member: {name}")
                stream = tar.extractfile(member)
                if stream is None:
                    raise SourceError(f"Cannot read required member: {name}")
                with stream:
                    payload = stream.read(self.config.max_member_bytes + 1)
                if len(payload) != member.size:
                    raise SourceError(f"Truncated member: {name}")
                hashes[name] = hashlib.sha256(payload).hexdigest()
                if relative == "imu.feather":
                    import pandas as pd
                    table = pd.read_feather(io.BytesIO(payload), columns=["acceleration_x", "acceleration_y", "acceleration_z"])
                    if not count < len(table) <= 20000:
                        raise SourceError("IMU table cannot supply causal replay windows")
                    imu = table.to_numpy(dtype=np.float64)[:count + 1]
                    if not np.isfinite(imu).all():
                        raise SourceError("Nonfinite IMU samples")
                elif sensor == "rgb-front":
                    camera[tick] = camera_embedding_features(self._image(payload))
                else:
                    sanitized, _ = self._sanitize(self._image(payload))
                    seg[tick] = segmentation_histogram_features(sanitized)
                del payload
                if len(camera) == len(seg) == count and imu is not None:
                    break
        required = set(range(1, count + 1))
        if set(camera) != required or set(seg) != required or imu is None:
            raise SourceError(f"Required sensor member missing for {sequence_id}: "
                              f"Camera={len(camera)}/{count}, Seg={len(seg)}/{count}, IMU={imu is not None}")
        # Build in a local variable; never expose a partially populated source cache.
        cache = [{"Camera": camera[t], "IMU": imu_window_features(imu[max(0, t - 11):t + 1]),
                  "Seg": seg[t]} for t in range(1, count + 1)]
        self._cache, self._cache_sequence = cache, sequence_id
        self._provenance = {**self._identity, "member_sha256": hashes,
                            "read_members": len(hashes), "label_payloads_opened": False,
                            "anomaly_front_payloads_opened": False, "imu_window": "causal, up to 12 rows",
                            "timestamp_source": "unavailable; no physical sampling rate inferred",
                            "prepared_sample_count": count, "archive_prepare_ms": (time.perf_counter() - start) * 1000}

    def read(self, sequence_id, index):
        if not 0 <= index < self.sample_count(sequence_id):
            raise SourceError("Replay sample out of range")
        start = time.perf_counter()
        cached = self._cache_sequence == sequence_id
        if not cached:
            self._prepare(sequence_id)
        return ObservationEnvelope("CarlAnomaly Base", "autonomous_driving", sequence_id,
                                   str(index + 1), self._cache[index],
                                   metadata={"tick": index + 1, "source_read_ms": (time.perf_counter() - start) * 1000,
                                             "cache_hit": cached},
                                   provenance={**self._provenance, **self.info, "label_visible_to_inference": False})


class SyntheticReplaySource:
    def __init__(self, dataset):
        self.dataset = dataset
        self.info = {"mode": "SYNTHETIC", "domain": "autonomous_driving", "source_id": "CarlAnomalyDataset (Synthetic Simulation)",
                     "dataset": "Synthetic demonstration", "seed": 42}
        self._cache = {}

    def sequences(self):
        return list(self.dataset.anomalies)

    def sample_count(self, sequence_id):
        return 100

    def read(self, sequence_id, index):
        start = time.perf_counter()
        if sequence_id not in self.sequences():
            raise SourceError("Unknown synthetic scenario")
        key = (sequence_id, index)
        if key not in self._cache:
            frame = self.dataset.generate_frames(sequence_id)[0]
            self._cache[key] = {"Camera": frame.rgb, "Depth": frame.depth, "LiDAR": frame.lidar,
                                "GNSS": frame.gnss, "IMU": frame.imu, "Seg": frame.segmentation}
        # Only one selected scenario's small window is kept.
        self._cache = {k: v for k, v in self._cache.items() if k[0] == sequence_id}
        affected = {"CAMERA_BLACKOUT": ["Camera", "Depth"], "GPS_DRIFT": ["GNSS"],
                    "HEAVY_RAIN": ["Camera", "LiDAR", "Depth"], "MULTI_FAILURE": ["Camera", "GNSS", "IMU"]}.get(sequence_id, [])
        return ObservationEnvelope(self.info["source_id"], self.info["domain"], sequence_id, str(index + 1),
                                   self._cache[key], metadata={"tick": index + 1,
                                   "source_read_ms": (time.perf_counter() - start) * 1000,
                                   "agent_health": {k: k not in affected for k in self._cache[key]},
                                   "scenario_details": {"severity": "Synthetic", "affected": ", ".join(affected) or "None",
                                                        "expected": "Synthetic perturbation demonstration"}}, provenance=self.info)
