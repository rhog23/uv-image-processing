"""Drawing utilities"""

import cv2
from typing import Tuple, List, Dict, Any
from . import config


def calculate_centroid(rect: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """Calculate the centroid of a rectangle (face's bounding box)"""
    x, y, w, h = rect
    centroid_x = x + w // 2
    centroid_y = y + h // 2
    return centroid_x, centroid_y


def draw_detections(
    frame: cv2.typing.MatLike, detections: List[Dict[str, Any]]
) -> None:
    """Draws bounding boxes, centroids, and coordinates for detected faces.

    Args:
        frame: The image frame to draw on.
        detections: A list of dictionaries, where each dictionary contains: `rect (x, y, w, h)`, `centroid (cx, cy)`, and optionally `type`.
    """
    for detection in detections:
        x, y, w, h = detection["rect"]
        centroid_x, centroid_y = detection["centroid"]
        detection_type = detection.get("type", "frontal")  # defaults to frontal

        box_color = config.BOX_COLOR
        if detection_type == "profile":
            box_color = config.BOX_COLOR_PROFILE

        # draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, config.BOX_THICKNESS)

        # draw centroid
        cv2.circle(
            frame,
            (centroid_x, centroid_y),
            config.CENTROID_RADIUS,
            config.CENTROID_COLOR,
            -1,  # filled circle
        )

        # display coordinates text
        label = f"({centroid_x, centroid_y})"

        if detection_type == "profile":
            label = f"P: {label}"  # optional: prefix for profile detection

        cv2.putText(
            frame,
            label,
            (x + 5, y - 7),
            config.TEXT_FONT,
            config.TEXT_FONT_SCALE,
            config.TEXT_COLOR,
            config.TEXT_THICKNESS,
            cv2.LINE_AA,
        )
