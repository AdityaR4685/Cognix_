from dataclasses import dataclass, field
from itertools import count


@dataclass
class BBoxAnnotation:
    """
    Container for bounding box results
    """

    identifier: int = field(
        default_factory=count().__next__
    )  # auto increments to product unique ids
    x_min: int = None
    x_max: int = None
    y_min: int = None
    y_max: int = None
    image_id: int = None
    category_name: str = None
    visible: bool = None
    bbox_ratio: float = None
    bbox_ratio_ok: bool = None
    category_id: int = None
    actor_type: str = None
    actor_id: int = None
    distance: float = None
    iscrowd: int = 0

    def has_coodinates(self) -> bool:
        return self.x_max is not None

    def get_bbox(self):
        return [self.x_min, self.y_min, self.x_max, self.y_max]

    def area(self):
        return abs(self.x_max - self.x_min) * abs(self.y_max - self.y_min)

    def to_coco(self):
        return {
            "area": self.area(),
            "iscrowded": 0,
            "image_id": self.image_id,
            "bbox": self.get_bbox(),
            "category_id": self.category_id,
            "image_id": self.image_id,
            "id": self.identifier,
        }
