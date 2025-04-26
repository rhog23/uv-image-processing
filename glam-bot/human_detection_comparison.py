import cv2
import numpy as np
from skimage.feature import hog
from skimage import exposure, color
import matplotlib.pyplot as plt
import time
import sys


class ImagePreprocessor:
    """Advanced image preprocessing techniques to improve detection accuracy"""

    @staticmethod
    def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
        """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Split the LAB channels
        l, a, b = cv2.split(lab)

        # Apply CLAHE to the L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_clahe = clahe.apply(l)

        # Merge the CLAHE enhanced L channel with the original A and B channels
        lab_clahe = cv2.merge((l_clahe, a, b))

        # Convert back to BGR color space
        enhanced_image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

        return enhanced_image

    @staticmethod
    def apply_fast_adaptive_histogram_equalization(image, adaptation_radius=15):
        """
        A faster alternative to CLAHE using a simplified approach
        that adapts the histogram locally without the full computational cost
        """
        # Convert to grayscale if it's a color image
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Create output image
        output = np.zeros_like(gray)

        # Get image dimensions
        height, width = gray.shape

        # Process the image in blocks
        for y in range(0, height, adaptation_radius):
            for x in range(0, width, adaptation_radius):
                # Define region bounds
                y_end = min(y + adaptation_radius, height)
                x_end = min(x + adaptation_radius, width)

                # Extract block
                block = gray[y:y_end, x:x_end]

                # Skip empty blocks
                if block.size == 0:
                    continue

                # Apply histogram equalization to the block
                block_eq = cv2.equalizeHist(block)

                # Place equalized block back
                output[y:y_end, x:x_end] = block_eq

        # If original was color, convert back to color
        if len(image.shape) > 2:
            # Create a 3-channel output
            output_color = image.copy()
            # Apply the enhanced luminance while preserving color
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv[:, :, 2] = output  # Replace V channel with enhanced version
            output_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            return output_color

        return output

    @staticmethod
    def apply_unsharp_masking(image, sigma=1.0, strength=1.5):
        """Apply unsharp masking to enhance edges"""
        # Create a blurred version of the image
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)

        # Calculate the unsharp mask
        unsharp_mask = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)

        return unsharp_mask

    @staticmethod
    def apply_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
        """Apply bilateral filter to reduce noise while preserving edges"""
        filtered = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        return filtered

    @staticmethod
    def enhance_image(image, enhancement_level="medium"):
        """Apply a combination of preprocessing techniques based on enhancement level"""
        # Make a copy of the original image
        enhanced = image.copy()

        if enhancement_level == "low":
            # Basic enhancement
            enhanced = ImagePreprocessor.apply_fast_adaptive_histogram_equalization(
                enhanced
            )

        elif enhancement_level == "medium":
            # Medium enhancement
            enhanced = ImagePreprocessor.apply_clahe(enhanced)
            enhanced = ImagePreprocessor.apply_bilateral_filter(enhanced, d=7)

        elif enhancement_level == "high":
            # High enhancement
            enhanced = ImagePreprocessor.apply_clahe(enhanced, clip_limit=3.0)
            enhanced = ImagePreprocessor.apply_bilateral_filter(enhanced, d=9)
            enhanced = ImagePreprocessor.apply_unsharp_masking(enhanced, strength=1.2)

        return enhanced


def non_max_suppression(boxes, weights, overlap_threshold=0.5):
    """
    Apply non-maximum suppression to eliminate overlapping detections

    Args:
        boxes: numpy array of shape (n, 4) where each row is [x, y, w, h]
        weights: detection confidence scores
        overlap_threshold: IoU threshold for considering boxes as overlapping

    Returns:
        indices of boxes to keep
    """
    # If no boxes, return empty list
    if len(boxes) == 0:
        return []

    # Convert from [x, y, w, h] to [x1, y1, x2, y2]
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    # Compute area of each box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    # Sort indices by confidence score (descending order)
    idxs = np.argsort(weights)[::-1]

    # Initialize list to store indices of boxes to keep
    keep = []

    # Process boxes in order of confidence
    while len(idxs) > 0:
        # Keep the box with highest confidence
        last = len(idxs) - 1
        i = idxs[0]
        keep.append(i)

        # Find intersection with remaining boxes
        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])

        # Compute width and height of intersection
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Compute intersection area
        intersection = w * h

        # Compute IoU
        overlap = intersection / (area[i] + area[idxs[1:]] - intersection)

        # Delete all other boxes with IoU > threshold
        idxs = np.delete(
            idxs, np.concatenate(([0], np.where(overlap > overlap_threshold)[0] + 1))
        )

    return keep


class HumanDetector:
    """Base class for human detection methods"""

    def __init__(self, name):
        self.name = name
        self.processing_times = []
        self.preprocessor = ImagePreprocessor()
        self.enhancement_level = "medium"  # Default enhancement level

    def detect(self, image):
        """Detect humans in an image"""
        raise NotImplementedError

    def get_avg_processing_time(self):
        """Return average processing time"""
        if not self.processing_times:
            return 0
        return sum(self.processing_times) / len(self.processing_times)

    def set_enhancement_level(self, level):
        """Set the image enhancement level (low, medium, high)"""
        if level in ["low", "medium", "high"]:
            self.enhancement_level = level
        else:
            print(f"Invalid enhancement level: {level}. Using 'medium' instead.")
            self.enhancement_level = "medium"


class HOGDetector(HumanDetector):
    """Human detection using Histogram of Oriented Gradients with improved preprocessing"""

    def __init__(self):
        super().__init__("Enhanced HOG Detector")
        # Initialize the HOG descriptor/person detector
        self.hog_detector = cv2.HOGDescriptor()
        self.hog_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # Detection parameters
        self.win_stride = (8, 8)
        self.padding = (16, 16)
        self.scale = 1.05
        self.nms_threshold = 0.45

    def set_parameters(
        self, win_stride=None, padding=None, scale=None, nms_threshold=None
    ):
        """Update detection parameters"""
        if win_stride:
            self.win_stride = win_stride
        if padding:
            self.padding = padding
        if scale:
            self.scale = scale
        if nms_threshold:
            self.nms_threshold = nms_threshold

    def detect(self, image):
        start_time = time.time()

        # Enhance image using preprocessing techniques
        enhanced_image = self.preprocessor.enhance_image(image, self.enhancement_level)

        # Resize image for faster detection
        height, width = enhanced_image.shape[:2]
        scale = min(1.0, 800 / max(height, width))
        if scale < 1.0:
            enhanced_image = cv2.resize(
                enhanced_image, (int(width * scale), int(height * scale))
            )

        # Detect people
        boxes, weights = self.hog_detector.detectMultiScale(
            enhanced_image,
            winStride=self.win_stride,
            padding=self.padding,
            scale=self.scale,
        )

        # Scale boxes back to original size
        if scale < 1.0 and len(boxes) > 0:
            boxes = boxes / scale

        # Apply non-maximum suppression
        if len(boxes) > 0:
            keep_indices = non_max_suppression(boxes, weights, self.nms_threshold)
            boxes = boxes[keep_indices]
            weights = weights[keep_indices]

        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)

        return boxes, weights


class HaarCascadeDetector(HumanDetector):
    """Human detection using Haar Cascades with improved preprocessing"""

    def __init__(self):
        super().__init__("Enhanced Haar Cascade Detector")
        # Load the human detector cascades
        self.body_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_fullbody.xml"
        )
        self.upper_body_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_upperbody.xml"
        )
        self.lower_body_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_lowerbody.xml"
        )

        # Detection parameters
        self.scale_factor = 1.1
        self.min_neighbors = 5
        self.min_size = (30, 80)
        self.nms_threshold = 0.5

    def set_parameters(
        self, scale_factor=None, min_neighbors=None, min_size=None, nms_threshold=None
    ):
        """Update detection parameters"""
        if scale_factor:
            self.scale_factor = scale_factor
        if min_neighbors:
            self.min_neighbors = min_neighbors
        if min_size:
            self.min_size = min_size
        if nms_threshold:
            self.nms_threshold = nms_threshold

    def detect(self, image):
        start_time = time.time()

        # Enhance image using preprocessing techniques
        enhanced_image = self.preprocessor.enhance_image(image, self.enhancement_level)

        # Convert to grayscale
        gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)

        # Apply unsharp masking to enhance edges for haar detection
        gray = self.preprocessor.apply_unsharp_masking(gray, sigma=0.5, strength=1.0)

        # Detect bodies
        bodies = self.body_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        # Detect upper bodies
        upper_bodies = self.upper_body_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        # Detect lower bodies
        lower_bodies = self.lower_body_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        # Combine detections
        all_boxes = []
        all_weights = []

        # Add full body detections with higher confidence
        if len(bodies) > 0:
            for box in bodies:
                all_boxes.append(box)
                all_weights.append(
                    0.9
                )  # Assign high confidence to full body detections

        # Add upper body detections
        if len(upper_bodies) > 0:
            for box in upper_bodies:
                all_boxes.append(box)
                all_weights.append(0.7)  # Medium confidence

        # Add lower body detections
        if len(lower_bodies) > 0:
            for box in lower_bodies:
                all_boxes.append(box)
                all_weights.append(0.6)  # Lower confidence

        # Convert to numpy arrays
        all_boxes = np.array(all_boxes)
        all_weights = np.array(all_weights)

        # Apply non-maximum suppression if we have detections
        if len(all_boxes) > 0:
            keep_indices = non_max_suppression(
                all_boxes, all_weights, self.nms_threshold
            )
            all_boxes = all_boxes[keep_indices]
            all_weights = all_weights[keep_indices]

        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)

        return all_boxes, all_weights


class BackgroundSubtractionDetector(HumanDetector):
    """Human detection using background subtraction and contour analysis with improved preprocessing"""

    def __init__(self):
        super().__init__("Enhanced Background Subtraction Detector")
        # Create background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )
        self.prev_frame = None
        self.min_area = 500  # Minimum contour area to be considered a person
        self.nms_threshold = 0.3

    def set_parameters(self, min_area=None, nms_threshold=None):
        """Update detection parameters"""
        if min_area:
            self.min_area = min_area
        if nms_threshold:
            self.nms_threshold = nms_threshold

    def detect(self, image):
        start_time = time.time()

        # Enhance image using preprocessing techniques
        enhanced_image = self.preprocessor.enhance_image(image, self.enhancement_level)

        # Apply background mask
        fg_mask = self.bg_subtractor.apply(enhanced_image)

        # Remove shadows (gray pixels)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Apply morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter contours by size and shape to identify humans
        boxes = []
        weights = []

        for contour in contours:
            # Calculate contour area
            area = cv2.contourArea(contour)

            if area > self.min_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)

                # Human-like aspect ratio (height > width)
                aspect_ratio = h / float(w)

                # If shape resembles a human (taller than wide)
                if aspect_ratio > 1.5 and h > 80:
                    boxes.append([x, y, w, h])
                    weights.append(area / 10000)  # Normalized area as confidence score

        # Convert to numpy arrays
        boxes = np.array(boxes)
        weights = np.array(weights)

        # Apply non-maximum suppression if we have detections
        if len(boxes) > 0:
            keep_indices = non_max_suppression(boxes, weights, self.nms_threshold)
            boxes = boxes[keep_indices]
            weights = weights[keep_indices]

        self.prev_frame = enhanced_image.copy()

        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)

        return boxes, weights


class HybridDetector(HumanDetector):
    """Advanced hybrid approach combining HOG, Haar, and motion-based detection"""

    def __init__(self):
        super().__init__("Advanced Hybrid Detector")
        self.hog_detector = HOGDetector()
        self.haar_detector = HaarCascadeDetector()
        self.bg_detector = BackgroundSubtractionDetector()
        self.confidence_threshold = 0.3
        self.nms_threshold = 0.4

    def set_parameters(self, confidence_threshold=None, nms_threshold=None):
        """Update detection parameters"""
        if confidence_threshold:
            self.confidence_threshold = confidence_threshold
        if nms_threshold:
            self.nms_threshold = nms_threshold

    def set_enhancement_level(self, level):
        """Set the image enhancement level for all sub-detectors"""
        super().set_enhancement_level(level)
        self.hog_detector.set_enhancement_level(level)
        self.haar_detector.set_enhancement_level(level)
        self.bg_detector.set_enhancement_level(level)

    def detect(self, image):
        start_time = time.time()

        # Get detections from all methods
        hog_boxes, hog_weights = self.hog_detector.detect(image)
        haar_boxes, haar_weights = self.haar_detector.detect(image)
        bg_boxes, bg_weights = self.bg_detector.detect(image)

        # Combine all detections
        all_boxes = []
        all_weights = []

        # Add HOG detections with weight adjustment
        if len(hog_boxes) > 0:
            for box, weight in zip(hog_boxes, hog_weights):
                # Normalize HOG confidence
                norm_weight = min(weight, 1.0) * 1.2  # Boost HOG confidence by 20%
                if norm_weight > self.confidence_threshold:
                    all_boxes.append(box)
                    all_weights.append(norm_weight)

        # Add Haar detections
        if len(haar_boxes) > 0:
            for box, weight in zip(haar_boxes, haar_weights):
                if weight > self.confidence_threshold:
                    all_boxes.append(box)
                    all_weights.append(weight)

        # Add background subtraction detections
        if len(bg_boxes) > 0:
            for box, weight in zip(bg_boxes, bg_weights):
                if weight > self.confidence_threshold:
                    all_boxes.append(box)
                    all_weights.append(weight)

        # Convert to numpy arrays
        all_boxes = np.array(all_boxes)
        all_weights = np.array(all_weights)

        # Apply non-maximum suppression if we have detections
        if len(all_boxes) > 0:
            keep_indices = non_max_suppression(
                all_boxes, all_weights, self.nms_threshold
            )
            all_boxes = all_boxes[keep_indices]
            all_weights = all_weights[keep_indices]

        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)

        return all_boxes, all_weights


class KCFTracker:
    """KCF (Kernelized Correlation Filters) tracker for continuous tracking of detected humans"""

    def __init__(self):
        self.trackers = []
        self.track_ids = []
        self.next_id = 0
        self.max_disappeared = 30  # Maximum frames a tracker can disappear
        self.disappeared_counts = {}
        self.tracked_boxes = {}

    def update(self, frame, detected_boxes):
        """Update trackers with new frame and detected boxes"""
        current_boxes = {}

        # If no trackers yet, initialize with detected boxes
        if not self.trackers:
            for box in detected_boxes:
                x, y, w, h = [int(v) for v in box]
                tracker = cv2.TrackerKCF_create()
                tracker.init(frame, (x, y, w, h))
                self.trackers.append(tracker)
                self.track_ids.append(self.next_id)
                current_boxes[self.next_id] = (x, y, w, h)
                self.disappeared_counts[self.next_id] = 0
                self.next_id += 1
        else:
            # First, update existing trackers
            tracked_ids_to_keep = []

            for i, (tracker, track_id) in enumerate(zip(self.trackers, self.track_ids)):
                success, box = tracker.update(frame)

                if success:
                    x, y, w, h = [int(v) for v in box]
                    current_boxes[track_id] = (x, y, w, h)
                    self.disappeared_counts[track_id] = 0
                    tracked_ids_to_keep.append(i)
                else:
                    # Increment disappeared count
                    self.disappeared_counts[track_id] += 1

                    # Keep tracker if it hasn't disappeared for too long
                    if self.disappeared_counts[track_id] <= self.max_disappeared:
                        tracked_ids_to_keep.append(i)

            # Retain only successful trackers
            self.trackers = [self.trackers[i] for i in tracked_ids_to_keep]
            self.track_ids = [self.track_ids[i] for i in tracked_ids_to_keep]

            # Match detected boxes with existing trackers
            if len(detected_boxes) > 0:
                # For each detected box, see if it matches an existing tracked box
                for box in detected_boxes:
                    x, y, w, h = [int(v) for v in box]

                    # Check if this detection overlaps significantly with an existing tracker
                    is_new_detection = True

                    for track_id, tracked_box in current_boxes.items():
                        tx, ty, tw, th = tracked_box

                        # Calculate IoU
                        # Convert to x1, y1, x2, y2 format
                        box1 = [x, y, x + w, y + h]
                        box2 = [tx, ty, tx + tw, ty + th]

                        # Calculate intersection
                        x_left = max(box1[0], box2[0])
                        y_top = max(box1[1], box2[1])
                        x_right = min(box1[2], box2[2])
                        y_bottom = min(box1[3], box2[3])

                        if x_right < x_left or y_bottom < y_top:
                            intersection_area = 0
                        else:
                            intersection_area = (x_right - x_left) * (y_bottom - y_top)

                        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
                        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

                        iou = intersection_area / float(
                            box1_area + box2_area - intersection_area
                        )

                        # If significant overlap, consider it the same object
                        if iou > 0.45:
                            is_new_detection = False
                            break

                    # If this is a new detection, create a new tracker
                    if is_new_detection:
                        tracker = cv2.TrackerKCF_create()
                        tracker.init(frame, (x, y, w, h))
                        self.trackers.append(tracker)
                        self.track_ids.append(self.next_id)
                        current_boxes[self.next_id] = (x, y, w, h)
                        self.disappeared_counts[self.next_id] = 0
                        self.next_id += 1

        # Update tracked boxes
        self.tracked_boxes = current_boxes

        return current_boxes


class CSRTTracker:
    """CSRT (Channel and Spatial Reliability Tracking) tracker for more accurate but slower tracking"""

    def __init__(self):
        self.trackers = []
        self.track_ids = []
        self.next_id = 0
        self.max_disappeared = 30  # Maximum frames a tracker can disappear
        self.disappeared_counts = {}
        self.tracked_boxes = {}

    def update(self, frame, detected_boxes):
        """Update trackers with new frame and detected boxes"""
        current_boxes = {}

        # If no trackers yet, initialize with detected boxes
        if not self.trackers:
            for box in detected_boxes:
                x, y, w, h = [int(v) for v in box]
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, (x, y, w, h))
                self.trackers.append(tracker)
                self.track_ids.append(self.next_id)
                current_boxes[self.next_id] = (x, y, w, h)
                self.disappeared_counts[self.next_id] = 0
                self.next_id += 1
        else:
            # First, update existing trackers
            tracked_ids_to_keep = []

            for i, (tracker, track_id) in enumerate(zip(self.trackers, self.track_ids)):
                success, box = tracker.update(frame)

                if success:
                    x, y, w, h = [int(v) for v in box]
                    current_boxes[track_id] = (x, y, w, h)
                    self.disappeared_counts[track_id] = 0
                    tracked_ids_to_keep.append(i)
                else:
                    # Increment disappeared count
                    self.disappeared_counts[track_id] += 1

                    # Keep tracker if it hasn't disappeared for too long
                    if self.disappeared_counts[track_id] <= self.max_disappeared:
                        tracked_ids_to_keep.append(i)

            # Retain only successful trackers
            self.trackers = [self.trackers[i] for i in tracked_ids_to_keep]
            self.track_ids = [self.track_ids[i] for i in tracked_ids_to_keep]

            # Match detected boxes with existing trackers
            if len(detected_boxes) > 0:
                # For each detected box, see if it matches an existing tracked box
                for box in detected_boxes:
                    x, y, w, h = [int(v) for v in box]

                    # Check if this detection overlaps significantly with an existing tracker
                    is_new_detection = True

                    for track_id, tracked_box in current_boxes.items():
                        tx, ty, tw, th = tracked_box

                        # Calculate IoU
                        # Convert to x1, y1, x2, y2 format
                        box1 = [x, y, x + w, y + h]
                        box2 = [tx, ty, tx + tw, ty + th]

                        # Calculate intersection
                        x_left = max(box1[0], box2[0])
                        y_top = max(box1[1], box2[1])
                        x_right = min(box1[2], box2[2])
                        y_bottom = min(box1[3], box2[3])

                        if x_right < x_left or y_bottom < y_top:
                            intersection_area = 0
                        else:
                            intersection_area = (x_right - x_left) * (y_bottom - y_top)

                        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
                        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

                        iou = intersection_area / float(
                            box1_area + box2_area - intersection_area
                        )

                        # If significant overlap, consider it the same object
                        if iou > 0.45:
                            is_new_detection = False
                            break

                    # If this is a new detection, create a new tracker
                    if is_new_detection:
                        tracker = cv2.TrackerCSRT_create()
                        tracker.init(frame, (x, y, w, h))
                        self.trackers.append(tracker)
                        self.track_ids.append(self.next_id)
                        current_boxes[self.next_id] = (x, y, w, h)
                        self.disappeared_counts[self.next_id] = 0
                        self.next_id += 1

        # Update tracked boxes
        self.tracked_boxes = current_boxes

        return current_boxes


# class DetectionTrackingSystem:
#     """Integrated system for human detection and tracking"""

#     def __init__(
#         self, detector_type="hybrid", tracker_type="kcf", detection_interval=5
#     ):
#         """
#         Initialize the detection and tracking system

#         Args:
#             detector_type: Type of detector to use ('hog', 'haar', 'hybrid')
#             tracker_type: Type of tracker to use ('kcf', 'csrt')
#             detection_interval: Run detection every N frames for efficiency
#         """
#         # Initialize detector
#         if detector_type == "hog":
#             self.detector = HOGDetector()
#         elif detector_type == "haar":
#             self.detector = HaarCascadeDetector()
#         elif detector_type == "hybrid":
#             self.detector = HybridDetector()
#         else:
#             print(f"Unknown detector type: {detector_type}. Using hybrid detector.")
#             self.detector = HybridDetector()

#         # Initialize tracker
#         if tracker_type == "kcf":
#             self.tracker = KCFTracker()
#         elif tracker_type == "csrt":
#             self.tracker = CSRTTracker()
#         else:
#             print(f"Unknown tracker type: {tracker_type}. Using KCF tracker.")
#             self.tracker = KCFTracker()

#         self.detection_interval = detection_interval
#         self.frame_count = 0
#         self.tracked_ids = {}  # Store persistent IDs for visualization

#     def process_frame(self, frame):
#         """Process a single frame"""
#         # Make a copy for drawing
#         display_frame = frame.copy()

#         # Increment frame counter
#         self.frame_count += 1

#         # Run detection at intervals for efficiency
#         if self.frame_count % self.detection_interval == 0:
#             # Run detector
#             boxes, weights = self.detector.detect(frame)

#             # Update tracker with new detections
#             tracked_boxes = self.tracker.update(frame, boxes)
#         else:
#             # Just update tracker without new detections
#             tracked_boxes = self.tracker.update(frame, np.array([]))

#         # Draw tracked boxes
#         for track_id, (x, y, w, h) in tracked_boxes.items():
#             # Assign persistent color for each ID
#             if track_id not in self.tracked_ids:
#                 # Generate a color based on the ID
#                 color = (
#                     (track_id * 50) % 255,
#                     (track_id * 100) % 255,
#                     (track_id * 150) % 255,
#                 )
#                 self.tracked_ids[track_id] = color
#             else:
#                 color = self.tracked_ids[track_id]

#             # Draw rectangle
#             cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)

#             # Display ID
#             cv2.putText(
#                 display_frame,
#                 f"ID: {track_id}",
#                 (x, y - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.5,
#                 color,
#                 2,
#             )

#         # Display detection status
#         if self.frame_count % self.detection_interval == 0:
#             cv2.putText(
#                 display_frame,
#                 "Detecting...",
#                 (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.7,
#                 (0, 0, 255),
#                 2,
#             )
#         else:
#             cv2.putText(
#                 display_frame,
#                 "Tracking...",
#                 (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.7,
#                 (0, 255, 0),
#                 2,
#             )

#         # Add frame counter
#         cv2.putText(
#             display_frame,
#             f"Frame: {self.frame_count}",
#             (10, 60),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.7,
#             (255, 0, 0),
#             2,
#         )

#         return display_frame, tracked_boxes


class SingleObjectTrackingSystem:
    """System for detecting and tracking a single human in video"""

    def __init__(
        self, detector_type="hybrid", tracker_type="kcf", detection_interval=5
    ):
        """
        Initialize the single object detection and tracking system

        Args:
            detector_type: Type of detector to use ('hog', 'haar', 'hybrid')
            tracker_type: Type of tracker to use ('kcf', 'csrt')
            detection_interval: Run detection every N frames for efficiency
        """
        # Initialize detector
        if detector_type == "hog":
            self.detector = HOGDetector()
        elif detector_type == "haar":
            self.detector = HaarCascadeDetector()
        elif detector_type == "hybrid":
            self.detector = HybridDetector()
        else:
            print(f"Unknown detector type: {detector_type}. Using hybrid detector.")
            self.detector = HybridDetector()

        # Save tracker type for reinitialization
        self.tracker_type = tracker_type

        # Initialize single tracker
        if tracker_type == "kcf":
            self.tracker = cv2.TrackerKCF_create()
        elif tracker_type == "csrt":
            self.tracker = cv2.TrackerCSRT_create()
        else:
            print(f"Unknown tracker type: {tracker_type}. Using KCF tracker.")
            self.tracker_type = "kcf"
            self.tracker = cv2.TrackerKCF_create()

        self.detection_interval = detection_interval
        self.frame_count = 0
        self.tracking_initialized = False
        self.tracked_box = None
        self.tracker_lost = False
        self.last_detection_frame = 0
        self.detection_confidence = 0.0

    def process_frame(self, frame):
        """Process a single frame"""
        # Make a copy for drawing
        display_frame = frame.copy()

        # Increment frame counter
        self.frame_count += 1

        # Run detection at intervals or if tracker is lost
        if (
            (self.frame_count % self.detection_interval == 0)
            or (not self.tracking_initialized)
            or self.tracker_lost
        ):
            # Run detector
            boxes, weights = self.detector.detect(frame)

            # Reset tracker lost flag
            self.tracker_lost = False

            # If objects detected, initialize or reinitialize tracker with highest confidence detection
            if len(boxes) > 0:
                # Find detection with highest confidence
                best_idx = np.argmax(weights) if len(weights) > 0 else 0
                x, y, w, h = [int(v) for v in boxes[best_idx]]

                # Create new tracker instance
                if self.tracker_type == "kcf":
                    self.tracker = cv2.TrackerKCF_create()
                else:
                    self.tracker = cv2.TrackerCSRT_create()

                # Initialize tracker with best detection
                self.tracker.init(frame, (x, y, w, h))
                self.tracking_initialized = True
                self.tracked_box = (x, y, w, h)
                self.last_detection_frame = self.frame_count
                self.detection_confidence = (
                    weights[best_idx] if len(weights) > 0 else 1.0
                )

                # Display detection status
                cv2.putText(
                    display_frame,
                    "Target Acquired",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
            else:
                # No objects detected
                cv2.putText(
                    display_frame,
                    "No Target Found",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

        # Update tracker if initialized
        elif self.tracking_initialized:
            success, box = self.tracker.update(frame)

            if success:
                # Update tracked box
                self.tracked_box = tuple(int(v) for v in box)

                # Display tracking status
                cv2.putText(
                    display_frame,
                    "Tracking",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
            else:
                # Tracker lost target
                self.tracker_lost = True
                cv2.putText(
                    display_frame,
                    "Target Lost",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

        # Draw tracked box if available
        if self.tracked_box and not self.tracker_lost:
            x, y, w, h = self.tracked_box

            # Draw rectangle
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display confidence
            cv2.putText(
                display_frame,
                f"Conf: {self.detection_confidence:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            # Display tracking duration
            frames_tracked = self.frame_count - self.last_detection_frame
            cv2.putText(
                display_frame,
                f"Frames tracked: {frames_tracked}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )

        return display_frame, (
            self.tracked_box if self.tracked_box and not self.tracker_lost else None
        )


def draw_detections(image, boxes, weights=None, color=(0, 255, 0), thickness=2):
    """Draw bounding boxes around detected humans"""
    img_copy = image.copy()

    if len(boxes) > 0:
        for i, (x, y, w, h) in enumerate(boxes):
            confidence_text = (
                f"{weights[i]:.2f}" if weights is not None and i < len(weights) else ""
            )

            # Draw rectangle
            cv2.rectangle(
                img_copy, (int(x), int(y)), (int(x + w), int(y + h)), color, thickness
            )

            # Draw confidence text if available
            if confidence_text:
                cv2.putText(
                    img_copy,
                    confidence_text,
                    (int(x), int(y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )

    return img_copy


def evaluate_detector(detector, test_images, enhancement_level="medium"):
    """Evaluate detector performance on test images"""
    results = []

    # Set enhancement level
    detector.set_enhancement_level(enhancement_level)

    for image_path in test_images:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            continue

        # Show original and enhanced images
        enhanced_image = detector.preprocessor.enhance_image(image, enhancement_level)

        # Detect humans
        boxes, weights = detector.detect(image)

        # Draw detections
        result_image = draw_detections(image, boxes, weights)

        # Store results
        results.append(
            {
                "image_path": image_path,
                "detections": len(boxes),
                "boxes": boxes,
                "weights": weights,
                "original_image": image,
                "enhanced_image": enhanced_image,
                "result_image": result_image,
                "processing_time": (
                    detector.processing_times[-1] if detector.processing_times else 0
                ),
            }
        )

    return results


def compare_detectors(detectors, test_images, enhancement_level="medium"):
    """Compare multiple detectors on test images"""
    all_results = {}

    for detector in detectors:
        print(f"Evaluating {detector.name} with {enhancement_level} enhancement...")
        results = evaluate_detector(detector, test_images, enhancement_level)
        all_results[detector.name] = results

        # Calculate average processing time and detection count
        avg_time = detector.get_avg_processing_time()
        total_detections = sum(r["detections"] for r in results)

        print(f"  Average processing time: {avg_time:.4f} seconds")
        print(f"  Total detections: {total_detections}")
        print()

    return all_results


def visualize_comparisons(all_results, test_images):
    """Visualize detection results for comparison"""
    for i, image_path in enumerate(test_images):
        plt.figure(figsize=(20, 15))

        # Track detector count to adjust subplot layout
        detector_count = len(all_results)
        cols = min(3, detector_count + 1)
        rows = (detector_count + cols) // cols + 1  # +1 for original image row

        # Plot original image
        plt.subplot(rows, cols, 1)
        original = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        plt.imshow(original)
        plt.title("Original Image")
        plt.axis("off")

        # Plot enhanced images in first row
        for j, (detector_name, results) in enumerate(all_results.items(), 2):
            if i < len(results):
                enhanced_img = cv2.cvtColor(
                    results[i]["enhanced_image"], cv2.COLOR_BGR2RGB
                )

                plt.subplot(rows, cols, j)
                plt.imshow(enhanced_img)
                plt.title(f"{detector_name}\nEnhanced Image")
                plt.axis("off")

        # Plot detection results in second row
        for j, (detector_name, results) in enumerate(all_results.items(), 1):
            if i < len(results):
                result = results[i]
                result_img = cv2.cvtColor(result["result_image"], cv2.COLOR_BGR2RGB)

                plt.subplot(rows, cols, cols + j)
                plt.imshow(result_img)
                plt.title(
                    f"{detector_name}\nDetections: {result['detections']}\nTime: {result['processing_time']:.3f}s"
                )
                plt.axis("off")

        plt.tight_layout()
        plt.show()


def run_video_demo(
    video_path=0,
    detector_type="hybrid",
    tracker_type="kcf",
    enhancement_level="medium",
    detection_interval=5,
):
    """
    Run a real-time or video file demo of the detection and tracking system

    Args:
        video_path: Path to video file or camera index (0 for default webcam)
        detector_type: Type of detector to use ('hog', 'haar', 'hybrid')
        tracker_type: Type of tracker to use ('kcf', 'csrt')
        enhancement_level: Level of image enhancement ('low', 'medium', 'high')
        detection_interval: Run detection every N frames for efficiency
    """
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_path}")
        return

    # Initialize detection and tracking system
    system = SingleObjectTrackingSystem(detector_type, tracker_type, detection_interval)

    # Set enhancement level
    system.detector.set_enhancement_level(enhancement_level)

    # Setup display window
    window_name = f"Human Detection ({detector_type} + {tracker_type})"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_times = []

    try:
        while True:
            # Start timing
            start_time = time.time()

            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("End of video stream")
                break

            # Process frame
            display_frame, tracked_boxes = system.process_frame(frame)

            # Calculate and display FPS
            frame_time = time.time() - start_time
            frame_times.append(frame_time)

            # Calculate average FPS over last 30 frames
            if len(frame_times) > 30:
                frame_times.pop(0)
            avg_frame_time = sum(frame_times) / len(frame_times)
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0

            cv2.putText(
                display_frame,
                f"FPS: {fps:.1f}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            # Add enhancement level info
            cv2.putText(
                display_frame,
                f"Enhancement: {enhancement_level}",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            # Add tracking info
            cv2.putText(
                display_frame,
                f"Tracker: {tracker_type.upper()}",
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 255),
                2,
            )

            # Display frame
            cv2.imshow(window_name, display_frame)

            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break
            elif key == ord("1"):  # Change enhancement level
                enhancement_level = "low"
                system.detector.set_enhancement_level(enhancement_level)
                print(f"Changed enhancement level to {enhancement_level}")
            elif key == ord("2"):
                enhancement_level = "medium"
                system.detector.set_enhancement_level(enhancement_level)
                print(f"Changed enhancement level to {enhancement_level}")
            elif key == ord("3"):
                enhancement_level = "high"
                system.detector.set_enhancement_level(enhancement_level)
                print(f"Changed enhancement level to {enhancement_level}")

    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()


def main():
    # Example usage with test images
    """test_images = ["path/to/your/test_image1.jpg", "path/to/your/test_image2.jpg"]

    # Create detectors with enhanced preprocessing
    detectors = [HOGDetector(), HaarCascadeDetector(), HybridDetector()]

    # Compare detectors with different enhancement levels
    print("Testing with LOW enhancement level:")
    low_results = compare_detectors(detectors, test_images, "low")

    print("Testing with MEDIUM enhancement level:")
    medium_results = compare_detectors(detectors, test_images, "medium")

    print("Testing with HIGH enhancement level:")
    high_results = compare_detectors(detectors, test_images, "high")

    # Visualize comparisons with medium enhancement (most balanced)
    visualize_comparisons(medium_results, test_images)"""

    # Run video demo with best configuration
    print(
        "\nRunning video demo with Hybrid detector, CSRT tracker and medium enhancement"
    )
    run_video_demo(
        video_path=0,  # Use webcam (change to video file path if needed)
        detector_type="hog",
        tracker_type="csrt",  # CSRT is more accurate, KCF is faster
        enhancement_level="high",
        detection_interval=10,  # Detect every 10 frames (adjust based on CPU performance)
    )


if __name__ == "__main__":
    # Check OpenCV version - some functions require OpenCV 3.4.1 or higher
    major, minor, _ = cv2.__version__.split(".")
    if int(major) < 3 or (int(major) == 3 and int(minor) < 4):
        print(
            f"Warning: Your OpenCV version ({cv2.__version__}) may not support all tracker types."
        )
        print("Consider upgrading to OpenCV 3.4.1 or higher for full functionality.\n")

    main()


# Example preprocessing benchmark function
def benchmark_preprocessing_methods(image_path):
    """Benchmark the performance of different preprocessing methods"""
    original = cv2.imread(image_path)
    if original is None:
        print(f"Could not read image: {image_path}")
        return

    preprocessor = ImagePreprocessor()
    methods = {
        "Original": lambda img: img.copy(),
        "CLAHE": preprocessor.apply_clahe,
        "Fast Adaptive Histogram": preprocessor.apply_fast_adaptive_histogram_equalization,
        "Unsharp Masking": preprocessor.apply_unsharp_masking,
        "Bilateral Filter": preprocessor.apply_bilateral_filter,
        "CLAHE + Bilateral": lambda img: preprocessor.apply_bilateral_filter(
            preprocessor.apply_clahe(img)
        ),
        "CLAHE + Unsharp": lambda img: preprocessor.apply_unsharp_masking(
            preprocessor.apply_clahe(img)
        ),
        "Fast Adaptive + Bilateral": lambda img: preprocessor.apply_bilateral_filter(
            preprocessor.apply_fast_adaptive_histogram_equalization(img)
        ),
    }

    results = {}

    # Process with each method and measure time
    for name, method in methods.items():
        start_time = time.time()
        result = method(original)
        processing_time = time.time() - start_time
        results[name] = {"image": result, "time": processing_time}
        print(f"{name}: {processing_time:.4f} seconds")

    # Display results
    plt.figure(figsize=(20, 15))
    rows = (len(methods) + 3) // 4
    cols = min(4, len(methods))

    for i, (name, result) in enumerate(results.items(), 1):
        plt.subplot(rows, cols, i)
        plt.imshow(cv2.cvtColor(result["image"], cv2.COLOR_BGR2RGB))
        plt.title(f"{name}\nTime: {result['time']:.4f}s")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

    return results


def find_optimal_parameters(detector, test_images, enhancement_level="medium"):
    """Find optimal parameters for a detector through grid search"""
    best_params = {}
    best_detection_count = 0

    # Set enhancement level
    detector.set_enhancement_level(enhancement_level)

    if isinstance(detector, HOGDetector):
        # Grid search for HOG parameters
        win_strides = [(4, 4), (8, 8), (16, 16)]
        paddings = [(8, 8), (16, 16), (32, 32)]
        scales = [1.03, 1.05, 1.07]
        nms_thresholds = [0.3, 0.45, 0.6]

        for win_stride in win_strides:
            for padding in paddings:
                for scale in scales:
                    for nms_threshold in nms_thresholds:
                        # Set parameters
                        detector.set_parameters(
                            win_stride=win_stride,
                            padding=padding,
                            scale=scale,
                            nms_threshold=nms_threshold,
                        )

                        # Evaluate
                        results = evaluate_detector(
                            detector, test_images, enhancement_level
                        )
                        total_detections = sum(r["detections"] for r in results)

                        print(
                            f"HOG params: win_stride={win_stride}, padding={padding}, "
                            f"scale={scale}, nms={nms_threshold} -> {total_detections} detections"
                        )

                        if total_detections > best_detection_count:
                            best_detection_count = total_detections
                            best_params = {
                                "win_stride": win_stride,
                                "padding": padding,
                                "scale": scale,
                                "nms_threshold": nms_threshold,
                            }

    elif isinstance(detector, HaarCascadeDetector):
        # Grid search for Haar parameters
        scale_factors = [1.05, 1.1, 1.15]
        min_neighbors_list = [3, 5, 7]
        min_sizes = [(20, 60), (30, 80), (40, 100)]
        nms_thresholds = [0.3, 0.5, 0.7]

        for scale_factor in scale_factors:
            for min_neighbors in min_neighbors_list:
                for min_size in min_sizes:
                    for nms_threshold in nms_thresholds:
                        # Set parameters
                        detector.set_parameters(
                            scale_factor=scale_factor,
                            min_neighbors=min_neighbors,
                            min_size=min_size,
                            nms_threshold=nms_threshold,
                        )

                        # Evaluate
                        results = evaluate_detector(
                            detector, test_images, enhancement_level
                        )
                        total_detections = sum(r["detections"] for r in results)

                        print(
                            f"Haar params: scale_factor={scale_factor}, min_neighbors={min_neighbors}, "
                            f"min_size={min_size}, nms={nms_threshold} -> {total_detections} detections"
                        )

                        if total_detections > best_detection_count:
                            best_detection_count = total_detections
                            best_params = {
                                "scale_factor": scale_factor,
                                "min_neighbors": min_neighbors,
                                "min_size": min_size,
                                "nms_threshold": nms_threshold,
                            }

    elif isinstance(detector, HybridDetector):
        # Grid search for Hybrid parameters
        confidence_thresholds = [0.2, 0.3, 0.4]
        nms_thresholds = [0.3, 0.4, 0.5]

        for confidence_threshold in confidence_thresholds:
            for nms_threshold in nms_thresholds:
                # Set parameters
                detector.set_parameters(
                    confidence_threshold=confidence_threshold,
                    nms_threshold=nms_threshold,
                )

                # Evaluate
                results = evaluate_detector(detector, test_images, enhancement_level)
                total_detections = sum(r["detections"] for r in results)

                print(
                    f"Hybrid params: confidence={confidence_threshold}, nms={nms_threshold} "
                    f"-> {total_detections} detections"
                )

                if total_detections > best_detection_count:
                    best_detection_count = total_detections
                    best_params = {
                        "confidence_threshold": confidence_threshold,
                        "nms_threshold": nms_threshold,
                    }

    print(f"\nBest parameters for {detector.name}: {best_params}")
    print(f"Total detections with best parameters: {best_detection_count}")

    return best_params
