#!/usr/bin/env python3
"""
Human Body Detection using classical image processing techniques
Optimized for Raspberry Pi 4 with 4GB RAM (no GPU)
Motion detection using background subtraction (MOG2)
Skin Color Detection (optional): Adds a supplementary method to detect skin-like regions, which can help in specific scenarios (e.g., detecting exposed skin like faces or hands).
"""

import cv2
import numpy as np
import time
from collections import deque
import argparse


class HumanDetector:
    def __init__(
        self,
        use_hog=True,
        use_motion=True,
        use_skin=False,
        history_size=10,
        confidence_threshold=0.5,
    ):
        """
        Initialize the human detector with multiple detection methods

        Args:
            use_hog: Whether to use HOG+SVM detection
            use_motion: Whether to use motion detection
            use_skin: Whether to use skin color detection
            history_size: Number of frames to keep for temporal filtering
            confidence_threshold: Minimum confidence for detection
        """
        self.use_hog = use_hog
        self.use_motion = use_motion
        self.use_skin = use_skin
        self.history_size = history_size
        self.confidence_threshold = confidence_threshold

        # Initialize HOG detector
        if self.use_hog:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # Initialize background subtractor for motion detection
        if self.use_motion:
            # Using MOG2 background subtractor which adapts to changing backgrounds
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=16, detectShadows=False
            )

        # Detection history for temporal filtering
        self.detection_history = deque(maxlen=history_size)

        # Performance metrics
        self.fps_history = deque(maxlen=30)
        self.last_frame_time = time.time()

    def detect_humans(self, frame):
        """
        Detect humans in the given frame using multiple techniques

        Args:
            frame: Input image frame

        Returns:
            List of bounding boxes (x, y, width, height, confidence)
        """
        start_time = time.time()

        # Scale down for faster processing
        scale = 0.8
        frame_small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        # Store original dimensions for later scaling back
        orig_h, orig_w = frame.shape[:2]

        all_detections = []

        # 1. HOG + SVM detection
        if self.use_hog:
            hog_detections = self._detect_with_hog(frame_small)
            # Scale back to original size
            hog_detections = [
                (int(x / scale), int(y / scale), int(w / scale), int(h / scale), conf)
                for x, y, w, h, conf in hog_detections
            ]
            all_detections.extend(hog_detections)

        # 2. Motion-based detection
        if self.use_motion:
            motion_detections = self._detect_with_motion(frame_small)
            # Scale back to original size
            motion_detections = [
                (int(x / scale), int(y / scale), int(w / scale), int(h / scale), conf)
                for x, y, w, h, conf in motion_detections
            ]
            all_detections.extend(motion_detections)

        # 3. Skin color detection (optional, often used as a supplementary method)
        if self.use_skin:
            skin_detections = self._detect_with_skin_color(frame_small)
            # Scale back to original size
            skin_detections = [
                (int(x / scale), int(y / scale), int(w / scale), int(h / scale), conf)
                for x, y, w, h, conf in skin_detections
            ]
            all_detections.extend(skin_detections)

        # Apply non-maximum suppression to remove overlapping detections
        final_detections = self._apply_nms(all_detections)

        # Apply temporal filtering for more stability
        final_detections = self._apply_temporal_filtering(final_detections)

        # Update performance metrics
        current_time = time.time()
        fps = (
            1.0 / (current_time - self.last_frame_time)
            if (current_time - self.last_frame_time) > 0
            else 0
        )
        self.fps_history.append(fps)
        self.last_frame_time = current_time

        return final_detections

    def _detect_with_hog(self, frame):
        """
        Detect humans using HOG features and SVM classifier
        """
        # Convert to grayscale for HOG
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply histogram equalization to improve contrast
        gray = cv2.equalizeHist(gray)

        # Detect people
        boxes, weights = self.hog.detectMultiScale(
            gray,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
            hitThreshold=0,  # Default threshold for detection
            useMeanshiftGrouping=False,
        )

        # Format detections as (x, y, width, height, confidence)
        detections = []
        for (x, y, w, h), weight in zip(boxes, weights):
            confidence = float(weight)
            if confidence > self.confidence_threshold:
                detections.append((x, y, w, h, confidence))

        return detections

    def _detect_with_motion(self, frame):
        """
        Detect potential humans using motion detection
        """
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)

        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(blurred)

        # Apply morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # Dilate to merge adjacent blobs
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

        # Find contours of moving objects
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections = []
        for contour in contours:
            # Filter small contours
            if cv2.contourArea(contour) < 500:
                continue

            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Filter based on aspect ratio and size
            aspect_ratio = float(h) / w

            # Human-like aspect ratio check (height > width)
            if 1.2 < aspect_ratio < 3.0 and h > 80:
                confidence = min(
                    cv2.contourArea(contour) / 5000, 0.9
                )  # Normalize confidence
                detections.append((x, y, w, h, confidence))

        return detections

    def _detect_with_skin_color(self, frame):
        """
        Detect potential humans using skin color detection
        """
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define range for skin color in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        # Create skin mask
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(
            skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections = []
        for contour in contours:
            # Filter small contours
            if cv2.contourArea(contour) < 1000:
                continue

            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # This is a weak signal, so assign lower confidence
            confidence = min(cv2.contourArea(contour) / 10000, 0.6)
            detections.append((x, y, w, h, confidence))

        return detections

    def _apply_nms(self, detections, overlap_threshold=0.4):
        """
        Apply non-maximum suppression to remove overlapping detections
        """
        if not detections:
            return []

        # Convert to numpy array for easier processing
        boxes = np.array(
            [[x, y, x + w, y + h, conf] for x, y, w, h, conf in detections]
        )

        # Get coordinates of bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        scores = boxes[:, 4]

        # Calculate area of bounding boxes
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        # Sort by confidence score
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            # Calculate intersection areas
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h

            # Calculate IoU
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            # Keep detections with IoU less than threshold
            inds = np.where(ovr <= overlap_threshold)[0]
            order = order[inds + 1]

        # Return filtered detections
        return [
            (
                int(boxes[i, 0]),
                int(boxes[i, 1]),
                int(boxes[i, 2] - boxes[i, 0]),
                int(boxes[i, 3] - boxes[i, 1]),
                float(boxes[i, 4]),
            )
            for i in keep
        ]

    def _apply_temporal_filtering(self, detections):
        """
        Apply temporal filtering to stabilize detections across frames
        """
        # Add current detections to history
        self.detection_history.append(detections)

        # Not enough history yet
        if len(self.detection_history) < 3:
            return detections

        # Create a list of all recent detections
        all_recent = []
        for frame_detections in self.detection_history:
            all_recent.extend(frame_detections)

        # Group similar detections
        merged_detections = []
        processed = set()

        for i, det_i in enumerate(all_recent):
            if i in processed:
                continue

            x_i, y_i, w_i, h_i, conf_i = det_i
            center_i = (x_i + w_i // 2, y_i + h_i // 2)

            # Find similar detections
            similar = []
            similar_idxs = []

            for j, det_j in enumerate(all_recent):
                if j in processed:
                    continue

                x_j, y_j, w_j, h_j, conf_j = det_j
                center_j = (x_j + w_j // 2, y_j + h_j // 2)

                # Calculate distance between centers
                dist = np.sqrt(
                    (center_i[0] - center_j[0]) ** 2 + (center_i[1] - center_j[1]) ** 2
                )

                # If centers are close, consider them the same object
                if dist < (w_i + w_j) / 3:
                    similar.append(det_j)
                    similar_idxs.append(j)
                    processed.add(j)

            # Calculate average detection
            if similar:
                avg_x = sum(det[0] for det in similar) / len(similar)
                avg_y = sum(det[1] for det in similar) / len(similar)
                avg_w = sum(det[2] for det in similar) / len(similar)
                avg_h = sum(det[3] for det in similar) / len(similar)
                avg_conf = sum(det[4] for det in similar) / len(similar)

                # Weigh recent detections more heavily
                if len(similar) > 1:
                    avg_conf = min(avg_conf * 1.1, 1.0)

                merged_detections.append(
                    (int(avg_x), int(avg_y), int(avg_w), int(avg_h), float(avg_conf))
                )

        return merged_detections

    def get_performance_metrics(self):
        """
        Return current performance metrics
        """
        avg_fps = (
            sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        )
        return {"fps": avg_fps}

    def visualize_detections(self, frame, detections):
        """
        Draw detection boxes on the frame
        """
        result = frame.copy()

        for x, y, w, h, confidence in detections:
            # Color based on confidence (green for high, yellow for medium, red for low)
            if confidence > 0.7:
                color = (0, 255, 0)  # Green
            elif confidence > 0.5:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 0, 255)  # Red

            # Draw rectangle
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)

            # Display confidence
            text = f"{confidence:.2f}"
            cv2.putText(
                result, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

        # Display FPS
        metrics = self.get_performance_metrics()
        fps_text = f"FPS: {metrics['fps']:.1f}"
        cv2.putText(
            result, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

        return result


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Human Detection using Image Processing"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source (0 for webcam, path for video file)",
    )
    parser.add_argument("--display", action="store_true", help="Display output window")
    parser.add_argument("--output", type=str, default="", help="Output video file path")
    parser.add_argument("--no-hog", action="store_true", help="Disable HOG detection")
    parser.add_argument(
        "--no-motion", action="store_true", help="Disable motion detection"
    )
    parser.add_argument(
        "--skin", action="store_true", help="Enable skin color detection"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Detection confidence threshold"
    )

    args = parser.parse_args()

    # Initialize the human detector
    detector = HumanDetector(
        use_hog=not args.no_hog,
        use_motion=not args.no_motion,
        use_skin=args.skin,
        confidence_threshold=args.threshold,
    )

    # Open video source
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize video writer if output is specified
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (frame_width, frame_height))

    print("Human Detector started")
    print(f"HOG detection: {'Enabled' if not args.no_hog else 'Disabled'}")
    print(f"Motion detection: {'Enabled' if not args.no_motion else 'Disabled'}")
    print(f"Skin detection: {'Enabled' if args.skin else 'Disabled'}")
    print(f"Confidence threshold: {args.threshold}")

    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break

            # Detect humans
            detections = detector.detect_humans(frame)

            # Visualize detections
            result = detector.visualize_detections(frame, detections)

            # Display result if requested
            if args.display:
                cv2.imshow("Human Detection", result)

                # Break on 'q' press
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # Write frame if output is specified
            if writer:
                writer.write(result)

            # Print performance stats every 30 frames
            if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 30 == 0:
                metrics = detector.get_performance_metrics()
                print(f"FPS: {metrics['fps']:.1f}")

    finally:
        # Release resources
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("Human Detector stopped")


if __name__ == "__main__":
    main()
