# smart_pan_tilt_system/vision_processor.py

"""
Handles camera operations, frame processing, and unified face detection (frontal & profile)
using OpenCV, including Non-Maximum Suppression (NMS).
"""
import cv2
import numpy as np
import os
from typing import Tuple, List, Optional, Dict

# Import configuration (assuming this file is in the same directory level as config.py or config is in PYTHONPATH)
import config


class VisionProcessor:
    """
    Manages camera input, unified face detection, and related calculations.
    """

    def __init__(
        self,
        camera_id: int = config.CAMERA_ID,
        frame_size: Tuple[int, int] = config.FRAME_SIZE,
        camera_fov_h_deg: float = config.CAMERA_FOV_H_DEG,
        dead_zone_px: int = config.DEAD_ZONE_PX,
        frontal_cascade_filename: str = config.HAAR_CASCADE_FRONTAL_FILENAME,
        profile_cascade_filename: str = config.HAAR_CASCADE_PROFILE_FILENAME,
        frontal_scale_factor: float = config.FRONTAL_SCALE_FACTOR,
        frontal_min_neighbors: int = config.FRONTAL_MIN_NEIGHBORS,
        profile_scale_factor: float = config.PROFILE_SCALE_FACTOR,
        profile_min_neighbors: int = config.PROFILE_MIN_NEIGHBORS,
        min_face_size: Tuple[int, int] = config.MIN_FACE_SIZE,
        nms_overlap_threshold: float = config.NMS_OVERLAP_THRESHOLD,
    ) -> None:
        """
        Initializes the VisionProcessor with unified detection capabilities.
        """
        self.camera_id: int = camera_id
        self.frame_width: int = frame_size[0]
        self.frame_height: int = frame_size[1]
        self.camera_fov_h_deg: float = camera_fov_h_deg
        self.dead_zone_px: int = dead_zone_px

        self.cap: cv2.VideoCapture = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open webcam ID {self.camera_id}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(
            f"Camera requested: {self.frame_width}x{self.frame_height}, Actual: {actual_width}x{actual_height}"
        )
        if actual_width != self.frame_width or actual_height != self.frame_height:
            print("Warning: Camera resolution mismatch. Updating internal dimensions.")
            self.frame_width = actual_width
            self.frame_height = actual_height
            # Update dead_zone_px if frame_width changed and it's based on percentage
            # Assuming config.DEAD_ZONE_PERCENT is the source of truth for the ratio
            self.dead_zone_px = int(self.frame_width * config.DEAD_ZONE_PERCENT / 2)

        # Load Haar Cascades
        self.frontal_face_cascade = self._load_cascade_safe(frontal_cascade_filename)
        self.profile_face_cascade = self._load_cascade_safe(profile_cascade_filename)

        # Detection Parameters
        self.frontal_scale_factor = frontal_scale_factor
        self.frontal_min_neighbors = frontal_min_neighbors
        self.profile_scale_factor = profile_scale_factor
        self.profile_min_neighbors = profile_min_neighbors
        self.min_face_size = min_face_size
        self.nms_overlap_threshold = nms_overlap_threshold

        print("VisionProcessor initialized with unified face detection.")

    def _load_cascade_safe(self, filename: str) -> cv2.CascadeClassifier:
        """Safely loads a Haar Cascade file from standard locations."""
        cv2_path = os.path.join(cv2.data.haarcascades, filename)
        if os.path.exists(cv2_path):
            cascade_path = cv2_path
        else:
            # Fallback: check project root (if VisionProcessor is in a subfolder like 'core')
            # This might need adjustment based on your project structure
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root_path = os.path.join(
                os.path.dirname(script_dir), filename
            )  # if vision_processor is one level down
            # If vision_processor.py is at the root with config.py:
            # project_root_path = os.path.join(script_dir, filename)
            if os.path.exists(project_root_path):
                cascade_path = project_root_path
            else:
                raise FileNotFoundError(
                    f"Haar Cascade file '{filename}' not found in cv2.data or project root."
                )

        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            raise IOError(f"Failed to load Haar Cascade from: {cascade_path}")
        print(
            f"Successfully loaded {os.path.basename(cascade_path)} from {cascade_path}"
        )
        return cascade

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Reads a frame from the camera."""
        ret, frame = self.cap.read()
        if not ret:
            return False, None
        # Ensure frame is of the expected size, some cameras might ignore initial set
        if frame.shape[1] != self.frame_width or frame.shape[0] != self.frame_height:
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        return True, frame

    def _non_max_suppression(
        self, boxes: np.ndarray, scores: np.ndarray, overlap_thresh: float
    ) -> List[Tuple[int, int, int, int]]:
        """Applies Non-Maximum Suppression to filter overlapping bounding boxes."""
        if len(boxes) == 0:
            return []

        # Convert (x, y, w, h) to (x1, y1, x2, y2)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        # Sort by score in descending order
        indices = np.argsort(scores)[::-1]

        keep_indices = []
        while len(indices) > 0:
            i = indices[0]
            keep_indices.append(i)

            # Calculate IoU with remaining boxes
            xx1 = np.maximum(x1[i], x1[indices[1:]])
            yy1 = np.maximum(y1[i], y1[indices[1:]])
            xx2 = np.minimum(x2[i], x2[indices[1:]])
            yy2 = np.minimum(y2[i], y2[indices[1:]])

            w_intersect = np.maximum(0, xx2 - xx1 + 1)
            h_intersect = np.maximum(0, yy2 - yy1 + 1)
            intersection_area = w_intersect * h_intersect

            iou = intersection_area / (
                areas[i] + areas[indices[1:]] - intersection_area
            )

            # Keep boxes with IoU less than the threshold
            indices_to_keep_mask = iou <= overlap_thresh
            indices = indices[1:][indices_to_keep_mask]

        return [tuple(map(int, box)) for box in boxes[keep_indices]]

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detects faces using a unified approach (frontal, profile, flipped profile)
        and applies Non-Maximum Suppression.

        Args:
            frame (np.ndarray): The input frame (BGR).

        Returns:
            List[Tuple[int, int, int, int]]: A list of bounding boxes (x, y, w, h)
                                               for detected faces after NMS.
        """
        gray: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_flipped: np.ndarray = cv2.flip(gray, 1)  # Flip horizontally

        all_detected_boxes = []
        all_detected_scores = []

        # 1. Frontal Face Detection
        frontal_rects, _, frontal_scores = self.frontal_face_cascade.detectMultiScale3(
            gray,
            scaleFactor=self.frontal_scale_factor,
            minNeighbors=self.frontal_min_neighbors,
            minSize=self.min_face_size,
            outputRejectLevels=True,
        )
        if len(frontal_rects) > 0:
            all_detected_boxes.extend(frontal_rects)
            all_detected_scores.extend(frontal_scores.flatten())

        # 2. Profile Face Detection (Original Image)
        profile_rects, _, profile_scores = self.profile_face_cascade.detectMultiScale3(
            gray,
            scaleFactor=self.profile_scale_factor,
            minNeighbors=self.profile_min_neighbors,
            minSize=self.min_face_size,
            outputRejectLevels=True,
        )
        if len(profile_rects) > 0:
            all_detected_boxes.extend(profile_rects)
            all_detected_scores.extend(profile_scores.flatten())

        # 3. Profile Face Detection (Flipped Image)
        profile_flipped_rects, _, profile_flipped_scores = (
            self.profile_face_cascade.detectMultiScale3(
                gray_flipped,
                scaleFactor=self.profile_scale_factor,
                minNeighbors=self.profile_min_neighbors,
                minSize=self.min_face_size,
                outputRejectLevels=True,
            )
        )
        if len(profile_flipped_rects) > 0:
            # Convert coordinates back to original image space
            rects_to_add = []
            for r_idx in range(len(profile_flipped_rects)):
                x, y, w, h = profile_flipped_rects[r_idx]
                original_x = self.frame_width - x - w
                rects_to_add.append([original_x, y, w, h])
            all_detected_boxes.extend(rects_to_add)  # Add as list of lists/np.array
            all_detected_scores.extend(profile_flipped_scores.flatten())

        # Apply Non-Maximum Suppression
        if len(all_detected_boxes) > 0:
            np_boxes = np.array(
                all_detected_boxes, dtype=np.int32
            )  # Ensure correct dtype
            np_scores = np.array(all_detected_scores, dtype=np.float64)

            # Ensure scores and boxes have the same length
            if len(np_boxes) != len(np_scores):
                print(
                    f"Warning: Mismatch in boxes ({len(np_boxes)}) and scores ({len(np_scores)}) count before NMS. This should not happen."
                )
                # Attempt to reconcile, e.g. by taking the minimum length, or skip NMS for this frame
                min_len = min(len(np_boxes), len(np_scores))
                np_boxes = np_boxes[:min_len]
                np_scores = np_scores[:min_len]
                if min_len == 0:
                    return []

            final_boxes = self._non_max_suppression(
                np_boxes, np_scores, self.nms_overlap_threshold
            )
            return final_boxes

        return []  # No detections

    def get_largest_face(
        self, faces: List[Tuple[int, int, int, int]]
    ) -> Optional[Tuple[int, int, int, int]]:
        """Finds the largest face from a list of detected faces by area."""
        if not faces:
            return None
        return max(faces, key=lambda f: f[2] * f[3])  # Max by area (w*h)

    def calculate_pan_offset_angle(self, face_bbox: Tuple[int, int, int, int]) -> float:
        """Calculates the angular offset of the detected face from the frame center."""
        x, _, w, _ = face_bbox
        frame_center_x: int = self.frame_width // 2
        object_center_x: int = x + w // 2
        pixel_offset: int = object_center_x - frame_center_x

        if abs(pixel_offset) <= self.dead_zone_px:
            return 0.0

        angle_offset_deg: float = (
            pixel_offset * self.camera_fov_h_deg
        ) / self.frame_width
        return angle_offset_deg

    def draw_tracking_info(
        self,
        frame: np.ndarray,
        tracked_face_bbox: Optional[Tuple[int, int, int, int]],
        all_detected_faces: List[Tuple[int, int, int, int]],  # All faces after NMS
        servo_positions: Dict[int, float],
    ) -> None:
        """Draws tracking information on the frame."""
        # Draw dead zone
        cv2.rectangle(
            frame,
            (self.frame_width // 2 - self.dead_zone_px, 0),
            (self.frame_width // 2 + self.dead_zone_px, self.frame_height),
            (0, 0, 255),
            1,  # Red for dead zone
        )

        # Draw all detected faces (e.g., in blue)
        for x, y, w, h in all_detected_faces:
            if tracked_face_bbox and (x, y, w, h) == tracked_face_bbox:
                continue  # Skip drawing if it's the tracked face, handled next
            cv2.rectangle(
                frame, (x, y), (x + w, y + h), (255, 0, 0), 1
            )  # Blue for other detections

        # Highlight the tracked face (e.g., in green)
        if tracked_face_bbox:
            x, y, w, h = tracked_face_bbox
            cv2.rectangle(
                frame, (x, y), (x + w, y + h), (0, 255, 0), 2
            )  # Green for tracked face
            cv2.putText(
                frame,
                "Tracking",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        # Display servo positions and other info
        text_y_offset = 20
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (200, 255, 200)  # Light green/cyan
        thickness = 1

        for pin, pos in servo_positions.items():
            cv2.putText(
                frame,
                f"Servo {pin}: {pos:.0f} us",
                (10, text_y_offset),
                font,
                font_scale,
                font_color,
                thickness,
            )
            text_y_offset += 25

        cv2.putText(
            frame,
            f"Dead Zone: +/-{self.dead_zone_px} px",
            (10, text_y_offset),
            font,
            font_scale,
            (0, 0, 255),
            thickness,
        )
        text_y_offset += 25
        cv2.putText(
            frame,
            f"Detections: {len(all_detected_faces)}",
            (10, text_y_offset),
            font,
            font_scale,
            font_color,
            thickness,
        )

    def release(self) -> None:
        """Releases the camera resource."""
        if self.cap.isOpened():
            self.cap.release()
            print("Camera released.")
