"""
Face Detector
"""

import cv2
import cv2.data
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import logging
from . import config


logger = logging.getLogger(__name__)


class FaceDetector:
    """A class to detect human face using haar cascades, including frontal and profile views."""

    def __init__(
        self, frontal_cascade_path: str, profile_cascade_path: Optional[str] = None
    ):
        """Initializes the FaceDetector.

        Args:
            frontal_cascade_path (str): Path to the Haar cascade XML file for frontal faces.
            profile_cascade_path (Optional[str], optional): Optional path to the Haar cascade XML file for profile faces. Defaults to None.
        """
        self.frontal_cascade_path = frontal_cascade_path
        self.profile_cascade_path = profile_cascade_path

        self.frontal_face_cascade = cv2.CascadeClassifier()
        self.profile_face_cascade = (
            cv2.CascadeClassifier() if self.profile_cascade_path else None
        )

        self._load_cascades()

    def _load_cascades(self) -> None:
        """Loads the Haar cascade classifiers."""
        if not self.frontal_face_cascade.load(
            f"{cv2.data.haarcascades}{self.frontal_cascade_path}"
        ):
            error_msg = f"Error loading frontal cascade: {self.frontal_cascade_path}"
            logger.error(error_msg)
            raise IOError(error_msg)
        logger.info(f"Successfully loaded frontal cascade: {self.frontal_cascade_path}")

        if self.profile_cascade_path and self.profile_face_cascade:
            if not self.profile_face_cascade.load(
                f"{cv2.data.haarcascades}{self.profile_cascade_path}"
            ):
                error_msg = (
                    f"Error loading profile cascade file: {self.profile_cascade_path}"
                )
                logger.warning(
                    error_msg
                )  # warn instead of raise, can proceed with frontal face only
                self.profile_face_cascade = (
                    None  # disable profile face if loading is failed
                )
            else:
                logger.info(
                    f"Successfully loaded profile cascade: {self.profile_cascade_path}"
                )
        elif self.profile_cascade_path is None:
            logger.info(
                "Profile cascade path not provided. Profile detection will be skipped."
            )

    def _rects_to_detections(
        self, rects: np.ndarray, detection_type: str
    ) -> List[Dict[str, Any]]:
        """Converts raw rectangles from detectMultiScale to the detection dictionary format

        Args:
            rects (np.ndarray): raw rectangles
            detection_type (str): face detection type

        Returns:
            List[Dict[str, Any]]: a list of dictionary(ies) of rectangles, centroids, and detection type
        """
        detections = []
        for x, y, w, h in rects:
            centroid_x, centroid_y = x + w // 2, y + h // 2
            detections.append(
                {
                    "rect": (x, y, w, h),
                    "centroid": (centroid_x, centroid_y),
                    "type": detection_type,
                }
            )
        return detections

    def _remove_overlapping_detections(
        self, detections: List[Dict[str, Any]], overlap_thresh: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Simple non-maximum suppression (NMS) to remove highly overlapping detections.
        This is a basic version. More sophisticated NMS might be needed for complex scenes.
        It prioritizes frontal detections if they overlap with profile detections significantly.

        Args:
            detections (List[Dict[str, Any]]): list of dictionary(ies) of rectangles, centroids, and detection type
            overlap_thresh (float, optional): boxes' overlap threshold. Defaults to 0.3.

        Returns:
            List[Dict[str, Any]]: cleaned list of detections
        """
        if not detections:
            return []

        # convert rect to [x1, y1, x2, y2] format
        boxes = np.array([d["rect"] for d in detections])
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]
        x2 = x1 + w
        y2 = y1 + h
        areas = w * h
        types = [d["type"] for d in detections]

        # Sort by confidence (not available from detectMultiScale, so using area as a proxy, or just original order)
        # For simplicity, we'll iterate and mark for removal.
        indices = np.arange(len(detections))

        keep_indices = []

        # =========== TODO ===========
        # Sort by area (larger detections first, can be a simple heuristic)
        # Or sort by type (frontal first)
        # For now, let's iterate and if a highly overlapping detection is found, prefer frontal or larger

        # A simpler approach for now: iterate and remove if Jaccard index is high
        # This is not a full NMS, but a greedy approach for this example.

        # Let's just return combined detections and handle overlaps by drawing order or visually.
        # A full NMS implementation is more involved.
        # For now, we will combine and potentially have overlaps.
        # If precise counting is needed, NMS is crucial.
        #
        # Alternative: if a frontal and profile detection are very similar, keep the frontal one.
        # This gets complex quickly. For now, we just combine. User can implement NMS if needed.

        # No NMS for simplicity in this example, just return all.
        # If you add NMS, it would go here.
        # Example placeholder for where NMS would be:
        # valid_detections_indices = non_max_suppression_function(boxes, scores_if_available, overlap_thresh)
        # final_detections = [detections[i] for i in valid_detections_indices]
        # return final_detections
        # =========== TODO ===========

        return detections  # returning combined without NMS for now

    def detect_faces(
        self,
        image: cv2.typing.MatLike,
        scale_factor: float = config.DETECTOR_SCALE_FACTOR,
        min_neighbors_frontal: int = config.DETECTOR_MIN_NEIGHBORS_FRONTAL,
        min_neighbors_profile: int = config.DETECTOR_MIN_NEIGHBORS_PROFILE,
        min_size: Tuple[int, int] = config.DETECTOR_MIN_SIZE,
        max_size: Optional[Tuple[int, int]] = config.DETECTOR_MAX_SIZE,
    ) -> List[Dict[str, Any]]:
        """Detects human faces in an image using frontal and optionally profile cascades.

        Args:
            image (cv2.typing.MatLike): input image / frame
            scale_factor (float, optional): detection scale factor. Defaults to config.DETECTOR_SCALE_FACTOR.
            min_neighbors_frontal (int, optional): minimum neighbors for frontal face detection. Defaults to config.DETECTOR_MIN_NEIGHBORS_FRONTAL.
            min_neighbors_profile (int, optional): minimum neighbors for profile face detection. Defaults to config.DETECTOR_MIN_NEIGHBORS_PROFILE.
            min_size (Tuple[int, int], optional): minimum size of the face. Defaults to config.DETECTOR_MIN_SIZE.
            max_size (Optional[Tuple[int, int]], optional): maximum size of the face. Defaults to config.DETECTOR_MAX_SIZE.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a detected face.
        """
        if image is None or image.size == 0:
            logger.warning("Received an empty image for detection.")
            return []

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if config.HIST_EQUALIZATION:
            gray_image = cv2.equalizeHist(
                gray_image
            )  # TODO: add other equalization methods

        all_detections = []

        # 1. Detect frontal faces
        frontal_detection_params = {
            "scaleFactor": scale_factor,
            "minNeighbors": min_neighbors_frontal,
            "minSize": min_size,
        }

        if max_size:
            frontal_detection_params["maxSize"] = max_size

        frontal_face_rects = self.frontal_face_cascade.detectMultiScale(
            gray_image, **frontal_detection_params
        )
        all_detections.extend(self._rects_to_detections(frontal_face_rects, "frontal"))

        # 2. Detect profile faces
        if self.profile_face_cascade:
            profile_detection_params = {
                "scaleFactor": scale_factor,  # try different params for profile
                "minNeighbors": min_neighbors_profile,
                "minSize": min_size,
            }
            if max_size:
                profile_detection_params["maxSize"] = max_size

            profile_face_rects = self.profile_face_cascade.detectMultiScale(
                gray_image, **profile_detection_params
            )
            all_detections.extend(
                self._rects_to_detections(profile_face_rects, "profile")
            )

            # For better profile detection, also try flipping the image horizontally and detecting again
            # as haarcascade_profileface.xml is typically trained for one profile direction (e.g. left)
            # and relies on the multiscale search to catch the other to some extent.
            # This doubles profile detection time but can improve recall.
            flipped_gray_image = cv2.flip(gray_image, 1)
            profile_face_rects_flipped = self.profile_face_cascade.detectMultiScale(
                flipped_gray_image, **profile_detection_params
            )

            # Convert flipped rects back to original image coordinates
            original_flipped_rects = []
            img_width = gray_image.shape[1]
            for x, y, w, h in profile_face_rects_flipped:
                original_x = img_width - (x + w)  # Crucial conversion
                original_flipped_rects.append((original_x, y, w, h))

            all_detections.extend(
                self._rects_to_detections(np.array(original_flipped_rects), "profile")
            )

            # ============ TODO ============
            # 3. Optional: Apply Non-Maximum Suppression (NMS) to remove highly overlapping boxes
            # For simplicity, we'll skip NMS here, but in a production system, it would be important!
            # if frontal and profile detectors trigger on the same face, or multiple scales do.
            # final_detections = self._remove_overlapping_detections(all_detections)
            # ============ TODO ============

            if not all_detections:
                logger.debug("No faces detected in the current frame.")
            else:
                logger.debug(
                    f"Detected {len(all_detections)} potential face(s) before NMS."
                )
                # NMS might reduce this count.

            return all_detections  # or final_detections if NMS is implemented

    def process_image_file(
        self, image_path: str
    ) -> Tuple[cv2.typing.MatLike, List[Dict[str, Any]]]:
        """Loads an image from file, detects faces, and returns the image with detections."""
        frame = cv2.imread(image_path)
        if frame is None:
            logger.error(f"Could not read image from {image_path}")
            raise FileNotFoundError(f"Could not read image from {image_path}")

        detections = self.detect_faces(frame)  # Uses configured parameters
        return frame, detections
