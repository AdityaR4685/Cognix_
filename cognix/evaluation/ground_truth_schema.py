"""Ground-Truth Annotation Schema and Validator (Milestone 8).

Defines the formal schema, coordinate conventions, lifecycle confirmation states,
and validation logic for 2D object bounding boxes and temporal track identities
on recorded CARLA driving sequences.

Coordinate Convention:
- Standard 2D pixel coordinates: (x1, y1, x2, y2)
- Origin (0, 0) is top-left of the image.
- 0 <= x1 < x2 <= image_width, 0 <= y1 < y2 <= image_height.
- All coordinates are in original unscaled image pixels (e.g., 1920x1080).
"""
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


class ObjectCategory(str, Enum):
    """Evaluation target categories."""
    PERSON = "person"
    VEHICLE = "vehicle"
    BICYCLE_OR_MOTORCYCLE = "bicycle_or_motorcycle"
    ROAD_DEBRIS_OR_PROP = "road_debris_or_prop"
    IGNORE_AMBIGUOUS = "ignore_ambiguous"


class OcclusionStatus(str, Enum):
    """Visual occlusion state of the object."""
    NONE = "none"
    PARTIAL = "partial"
    HEAVY = "heavy"


class VisibilityStatus(str, Enum):
    """Environmental visibility state."""
    CLEAR = "clear"
    DEGRADED_WEATHER = "degraded_weather"
    GLARE_OR_SHADOW = "glare_or_shadow"


class ConfirmationStatus(str, Enum):
    """Review confirmation lifecycle state."""
    UNCONFIRMED_DRAFT = "unconfirmed_draft"
    PENDING_REVIEW = "pending_review"
    HUMAN_CONFIRMED = "human_confirmed"


class FrameReviewState(str, Enum):
    """Machine-checkable review completeness states for a frame (Milestone 10)."""
    UNREVIEWED = "unreviewed"
    CONFIRMED_WITH_OBJECTS = "confirmed_with_objects"
    CONFIRMED_EMPTY = "confirmed_empty"
    REQUIRES_ADJUDICATION = "requires_adjudication"


class ValidationError(ValueError):
    """Raised when an annotation violates schema constraints."""
    pass


@dataclass
class BoundingBox:
    """Bounding box in pixel coordinates (xyxy)."""
    x1: float
    y1: float
    x2: float
    y2: float

    def validate(self, image_width: int = 1920, image_height: int = 1080) -> None:
        """Validate box bounds and non-degenerate area."""
        if not (0.0 <= self.x1 < self.x2 <= float(image_width)):
            raise ValidationError(
                f"Invalid horizontal coordinates: x1={self.x1}, x2={self.x2} "
                f"for image width {image_width}"
            )
        if not (0.0 <= self.y1 < self.y2 <= float(image_height)):
            raise ValidationError(
                f"Invalid vertical coordinates: y1={self.y1}, y2={self.y2} "
                f"for image height {image_height}"
            )
        if (self.x2 - self.x1) < 2.0 or (self.y2 - self.y1) < 2.0:
            raise ValidationError(
                f"Degenerate box area: width={self.x2 - self.x1}, height={self.y2 - self.y1}"
            )

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def centroid(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def iou(self, other: "BoundingBox") -> float:
        """Compute Intersection-over-Union against another bounding box."""
        inter_x1 = max(self.x1, other.x1)
        inter_y1 = max(self.y1, other.y1)
        inter_x2 = min(self.x2, other.x2)
        inter_y2 = min(self.y2, other.y2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        union_area = self.area + other.area - inter_area
        if union_area <= 0.0:
            return 0.0
        return inter_area / union_area

    def to_dict(self) -> Dict[str, float]:
        return {
            "x1": round(self.x1, 2),
            "y1": round(self.y1, 2),
            "x2": round(self.x2, 2),
            "y2": round(self.y2, 2),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BoundingBox":
        return cls(
            x1=float(data["x1"]),
            y1=float(data["y1"]),
            x2=float(data["x2"]),
            y2=float(data["y2"]),
        )


@dataclass
class AnnotatedObject:
    """A single annotated object observation within a frame."""
    object_id: str
    track_id: int
    category: ObjectCategory
    bbox: BoundingBox
    occlusion: OcclusionStatus = OcclusionStatus.NONE
    visibility: VisibilityStatus = VisibilityStatus.CLEAR
    is_ambiguous: bool = False
    confirmation_status: ConfirmationStatus = ConfirmationStatus.PENDING_REVIEW
    review_notes: Optional[str] = None
    confidence: Optional[float] = None
    provenance: Optional[str] = None

    def validate(self, image_width: int = 1920, image_height: int = 1080) -> None:
        """Validate object metadata and bounding box."""
        if self.track_id < 0:
            raise ValidationError(f"Invalid track_id: {self.track_id} (must be non-negative)")
        if not isinstance(self.category, ObjectCategory):
            try:
                self.category = ObjectCategory(self.category)
            except ValueError:
                raise ValidationError(f"Invalid category: {self.category}")
        self.bbox.validate(image_width=image_width, image_height=image_height)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "object_id": self.object_id,
            "track_id": self.track_id,
            "category": self.category.value if isinstance(self.category, ObjectCategory) else str(self.category),
            "bbox": self.bbox.to_dict(),
            "occlusion": self.occlusion.value if isinstance(self.occlusion, OcclusionStatus) else str(self.occlusion),
            "visibility": self.visibility.value if isinstance(self.visibility, VisibilityStatus) else str(self.visibility),
            "is_ambiguous": self.is_ambiguous,
            "confirmation_status": self.confirmation_status.value if isinstance(self.confirmation_status, ConfirmationStatus) else str(self.confirmation_status),
            "review_notes": self.review_notes,
        }
        if self.confidence is not None:
            d["confidence"] = self.confidence
        if self.provenance is not None:
            d["provenance"] = self.provenance
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnnotatedObject":
        try:
            category = ObjectCategory(data["category"])
        except ValueError:
            raise ValidationError(f"Invalid category: {data.get('category')}")

        return cls(
            object_id=str(data["object_id"]),
            track_id=int(data["track_id"]),
            category=category,
            bbox=BoundingBox.from_dict(data["bbox"]),
            occlusion=OcclusionStatus(data.get("occlusion", "none")),
            visibility=VisibilityStatus(data.get("visibility", "clear")),
            is_ambiguous=bool(data.get("is_ambiguous", False)),
            confirmation_status=ConfirmationStatus(data.get("confirmation_status", "pending_review")),
            review_notes=data.get("review_notes"),
            confidence=float(data["confidence"]) if "confidence" in data and data["confidence"] is not None else None,
            provenance=data.get("provenance"),
        )


@dataclass
class FrameAnnotation:
    """Annotations for a single recorded frame."""
    scenario_id: str
    tick: int
    frame_hash_sha256: str
    image_width: int = 1920
    image_height: int = 1080
    objects: List[AnnotatedObject] = field(default_factory=list)
    reviewer_id: Optional[str] = None
    confirmed_at_utc: Optional[str] = None
    explicit_review_state: Optional[FrameReviewState] = None

    def validate(self) -> None:
        """Enforce per-frame constraints, including uniqueness of track_ids."""
        seen_track_ids: Set[int] = set()
        seen_object_ids: Set[str] = set()

        for obj in self.objects:
            obj.validate(image_width=self.image_width, image_height=self.image_height)

            if obj.object_id in seen_object_ids:
                raise ValidationError(f"Duplicate object_id '{obj.object_id}' in tick {self.tick}")
            seen_object_ids.add(obj.object_id)

            # In a single frame, each distinct object must have a distinct track_id
            if obj.track_id in seen_track_ids:
                raise ValidationError(
                    f"Duplicate track_id {obj.track_id} in scenario {self.scenario_id} tick {self.tick}"
                )
            seen_track_ids.add(obj.track_id)

    @property
    def review_state(self) -> FrameReviewState:
        """Compute the machine-checkable review completeness state."""
        if self.explicit_review_state is not None:
            return self.explicit_review_state
        if self.reviewer_id is None:
            return FrameReviewState.UNREVIEWED
        # Check if any object flags unresolved conflict or active object ambiguity
        for obj in self.objects:
            if obj.review_notes and any(k in obj.review_notes.lower() for k in ("adjudication", "conflict", "uncertain")):
                return FrameReviewState.REQUIRES_ADJUDICATION
            if obj.category != ObjectCategory.IGNORE_AMBIGUOUS and obj.is_ambiguous:
                return FrameReviewState.REQUIRES_ADJUDICATION
        if any(obj.confirmation_status != ConfirmationStatus.HUMAN_CONFIRMED for obj in self.objects):
            return FrameReviewState.UNREVIEWED
        if len(self.objects) == 0:
            return FrameReviewState.CONFIRMED_EMPTY
        return FrameReviewState.CONFIRMED_WITH_OBJECTS

    @property
    def is_confirmed(self) -> bool:
        return self.review_state in (
            FrameReviewState.CONFIRMED_WITH_OBJECTS,
            FrameReviewState.CONFIRMED_EMPTY,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "tick": self.tick,
            "frame_hash_sha256": self.frame_hash_sha256,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "review_state": self.review_state.value,
            "objects": [obj.to_dict() for obj in self.objects],
            "reviewer_id": self.reviewer_id,
            "confirmed_at_utc": self.confirmed_at_utc,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FrameAnnotation":
        explicit_state = None
        if "review_state" in data and data["review_state"]:
            try:
                explicit_state = FrameReviewState(data["review_state"])
            except ValueError:
                explicit_state = None
        return cls(
            scenario_id=str(data["scenario_id"]),
            tick=int(data["tick"]),
            frame_hash_sha256=str(data["frame_hash_sha256"]),
            image_width=int(data.get("image_width", 1920)),
            image_height=int(data.get("image_height", 1080)),
            objects=[AnnotatedObject.from_dict(obj_dict) for obj_dict in data.get("objects", [])],
            reviewer_id=data.get("reviewer_id"),
            confirmed_at_utc=data.get("confirmed_at_utc"),
            explicit_review_state=explicit_state,
        )


@dataclass
class AnnotationDataset:
    """Collection of frame annotations across scenarios."""
    schema_version: str = "1.0.0"
    created_at_utc: str = ""
    annotation_mode: Optional[str] = None
    frames: List[FrameAnnotation] = field(default_factory=list)

    def validate(self) -> None:
        """Validate entire dataset, ensuring scenario isolation."""
        for frame in self.frames:
            frame.validate()

    def get_scenario_frames(self, scenario_id: str) -> List[FrameAnnotation]:
        return [f for f in self.frames if f.scenario_id == scenario_id]

    @property
    def total_objects(self) -> int:
        return sum(len(f.objects) for f in self.frames)

    @property
    def confirmed_frames_count(self) -> int:
        return sum(1 for f in self.frames if f.is_confirmed)

    @property
    def review_state_counts(self) -> Dict[str, int]:
        counts = {state.value: 0 for state in FrameReviewState}
        for f in self.frames:
            counts[f.review_state.value] += 1
        return counts

    @property
    def is_fully_confirmed(self) -> bool:
        if not self.frames:
            return False
        return all(f.is_confirmed for f in self.frames)

    def compute_sha256(self) -> str:
        """Compute deterministic SHA-256 hash of serialized content."""
        canonical_json = json.dumps(self.to_dict(), sort_keys=True, indent=2).encode("utf-8")
        return hashlib.sha256(canonical_json).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "schema_version": self.schema_version,
            "created_at_utc": self.created_at_utc,
            "total_frames": len(self.frames),
            "total_objects": self.total_objects,
            "confirmed_frames": self.confirmed_frames_count,
            "is_fully_confirmed": self.is_fully_confirmed,
            "review_state_counts": self.review_state_counts,
            "frames": [f.to_dict() for f in self.frames],
        }
        if self.annotation_mode is not None:
            d["annotation_mode"] = self.annotation_mode
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnnotationDataset":
        dataset = cls(
            schema_version=data.get("schema_version", "1.0.0"),
            created_at_utc=data.get("created_at_utc", ""),
            annotation_mode=data.get("annotation_mode"),
            frames=[FrameAnnotation.from_dict(f) for f in data.get("frames", [])],
        )
        return dataset

    def save_json(self, path: Path) -> str:
        canonical_json = json.dumps(self.to_dict(), sort_keys=True, indent=2).encode("utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_json)
        return hashlib.sha256(canonical_json).hexdigest()

    @classmethod
    def load_json(cls, path: Path) -> "AnnotationDataset":
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
        dataset = cls.from_dict(data)
        dataset.validate()
        return dataset
