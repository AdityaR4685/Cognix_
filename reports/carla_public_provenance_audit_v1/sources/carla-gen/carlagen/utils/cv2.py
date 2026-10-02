import cv2
import numpy as np
from ..annotation import BBoxAnnotation

import logging

log = logging.getLogger(__name__)


def draw_dotted_bb(img, box, **kwargs):
    draw_dotted_bounding_box(img, box.x_max, box.x_min, box.y_max, box.y_min, **kwargs)


def draw_bb(img, box, **kwargs):
    draw_bounding_box(img, box.x_max, box.x_min, box.y_max, box.y_min, **kwargs)


def draw_dotted_bounding_box(
    img, x_max, x_min, y_max, y_min, color=(255, 255, 255, 255), thickness=2
):
    drawline_dotted_line(img, (x_min, y_min), (x_max, y_min), color, thickness)
    drawline_dotted_line(img, (x_min, y_max), (x_max, y_max), color, thickness)
    drawline_dotted_line(img, (x_min, y_min), (x_min, y_max), color, thickness)
    drawline_dotted_line(img, (x_max, y_min), (x_max, y_max), color, thickness)


def draw_bounding_box(
    img, x_max, x_min, y_max, y_min, color=(255, 255, 255, 255), thickness=2
):
    cv2.line(img, (x_min, y_min), (x_max, y_min), color, thickness)
    cv2.line(img, (x_min, y_max), (x_max, y_max), color, thickness)
    cv2.line(img, (x_min, y_min), (x_min, y_max), color, thickness)
    cv2.line(img, (x_max, y_min), (x_max, y_max), color, thickness)


def draw_bbox(img, a: BBoxAnnotation, color=(255, 255, 255, 255), thickness=2):
    x_max, y_min, x_min, y_max = a.x_max, a.y_min, a.x_min, a.y_max

    cv2.line(img, (x_min, y_min), (x_max, y_min), color, thickness)
    cv2.line(img, (x_min, y_max), (x_max, y_max), color, thickness)
    cv2.line(img, (x_min, y_min), (x_min, y_max), color, thickness)
    cv2.line(img, (x_max, y_min), (x_max, y_max), color, thickness)


def drawline_dotted_line(img, pt1, pt2, color, thickness=1, gap=3):
    try:
        dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5

        pts = []

        for i in np.arange(0, dist, gap):
            r = i / dist
            x = int((pt1[0] * (1 - r) + pt2[0] * r) + 0.5)
            y = int((pt1[1] * (1 - r) + pt2[1] * r) + 0.5)
            p = (x, y)
            pts.append(p)

        for p in pts:
            cv2.circle(img, p, 1, color, -1)
    except TypeError as e:
        log.exception(e)
        log.error(f"{pt1} {pt2}")
        pass
