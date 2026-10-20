"""Indexed observation sources; no dataset or sensor knowledge belongs here."""
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Protocol
import copy
import math
import numpy as np


class SourceError(RuntimeError):
    pass


def freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({k: freeze(v) for k, v in value.items()})
    if isinstance(value, np.ndarray):
        # A bytes-backed view cannot be made writable again by a consumer.
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject:
            raise SourceError("Object arrays are not observation inputs")
        return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
    if isinstance(value, (list, tuple)):
        return tuple(freeze(v) for v in value)
    return copy.deepcopy(value)


def plain(value):
    """Convert immutable provenance/status structures to JSON-compatible values."""
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True)
class ObservationEnvelope:
    source_id: str
    domain_id: str
    sequence_id: str
    sample_id: str
    agent_inputs: Mapping[str, Any]
    timestamp: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.agent_inputs or any(not isinstance(k, str) or not k for k in self.agent_inputs):
            raise SourceError("Observations require named agent inputs")
        for name in ("agent_inputs", "metadata", "provenance"):
            object.__setattr__(self, name, freeze(getattr(self, name)))


class ReplaySource(Protocol):
    info: Mapping[str, Any]
    def sequences(self) -> list[str]: ...
    def sample_count(self, sequence_id: str) -> int: ...
    def read(self, sequence_id: str, index: int) -> ObservationEnvelope: ...


class ReplayController:
    def __init__(self, source: ReplaySource):
        self.source = source
        self.sequence_id: str | None = None
        self.index = -1
        self.playing = False
        self.rate = 1.0
        self.error: str | None = None

    def select(self, sequence_id: str):
        if sequence_id not in self.source.sequences():
            raise SourceError(f"Unknown sequence: {sequence_id}")
        if self.source.sample_count(sequence_id) < 1:
            raise SourceError("Sequence contains no replay samples")
        self.sequence_id, self.index, self.error = sequence_id, -1, None
        self.playing = False

    def step(self, delta: int = 1) -> ObservationEnvelope:
        if self.sequence_id is None:
            raise SourceError("Select a sequence before replay")
        if delta not in (-1, 0, 1):
            raise SourceError("Step must be -1, 0 or 1")
        count = self.source.sample_count(self.sequence_id)
        index = max(0, min(count - 1, self.index + delta))
        try:
            observation = self.source.read(self.sequence_id, index)
        except Exception as exc:
            self.playing, self.error = False, str(exc)
            raise
        self.index, self.error = index, None
        if index == count - 1:
            self.playing = False
        return observation

    def restart(self):
        self.index, self.error = -1, None
        self.playing = False

    def set_rate(self, rate: float):
        if not math.isfinite(rate) or not 0.1 <= rate <= 10:
            raise SourceError("Playback rate must be between 0.1 and 10 samples/s")
        self.rate = float(rate)

    def status(self):
        return {**plain(self.source.info), "sequence_id": self.sequence_id, "index": self.index,
                "sample_count": self.source.sample_count(self.sequence_id) if self.sequence_id else None,
                "playing": self.playing, "rate": self.rate, "error": self.error}
