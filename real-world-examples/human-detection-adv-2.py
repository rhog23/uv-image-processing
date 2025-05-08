#!/usr/bin/env python3
"""
Robust Human Body Detection using Classical Image Processing Techniques
Optimized for Raspberry Pi 4 with 4GB RAM (no GPU)
"""

import cv2
import numpy as np
import time
from collections import deque
import argparse
import threading
import queue
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class HumanDetector:
    def __init__(
        self,
        use_hog=True,
        use_motion=True,
        use_skin=False,
        history_size=10,
        confidence_threshold=0.6,
    ):
        """
        Initialize the human detector with enhanced robustness.

        Args:
            use_hog: Enable HOG+SVM detection.
            use_motion: Enable motion detection.
            use_skin: Enable skin color detection as a refining step.
            history_size: Number of frames for temporal filtering.
            confidence_threshold: Minimum confidence for detections.
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

        # Initialize background subtractor
        if self.use_motion:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=300, varThreshold=20, detectShadows=False
            )

        # Threading queues
        self.frame_queue = queue.Queue(maxsize=3)
        self.hog_result_queue = queue.Queue(maxsize=3)
        self.motion_result_queue = queue.Queue(maxsize=3)
        self.stop_event = threading.Event()

        # Detection history
        self.detection_history = deque(maxlen=history_size)

        # Performance metrics
        self.fps_history = deque(maxlen=30)
        self.last_frame_time = time.time()
        self.frame_counter = 0
        self.DETECTION_INTERVAL = 3  # Process HOG every 3 frames

    def start_workers(self):
        """Start worker threads for HOG and motion detection."""
        if self.use_hog:
            self.hog_thread = threading.Thread(target=self._hog_worker)
            self.hog_thread.daemon = True
            self.hog_thread.start()
        if self.use_motion:
            self.motion_thread = threading.Thread(target=self._motion_worker)
            self.motion_thread.daemon = True
            self.motion_thread.start()

    def _hog_worker(self):
        """Worker thread for HOG detection."""
        while not self.stop_event.is_set():
            try:
                frame, frame_id = self.frame_queue.get(timeout=0.1)
                detections = self._detect_with_hog(frame)
                self.hog_result_queue.put((detections, frame_id))
            except queue.Empty:
                continue

    def _motion_worker(self):
        """Worker thread for motion detection."""
        while not self.stop_event.is_set():
            try:
                frame, frame_id = self.frame_queue.get(timeout=0.1)
                detections = self._detect_with_motion(frame)
                self.motion_result_queue.put((detections, frame_id))
            except queue.Empty:
                continue

    def detect_humans(self, frame):
        """Detect humans in the frame using multiple techniques."""
        self.frame_counter += 1
        start_time = time.time()

        # Apply CLAHE for better contrast
        frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(frame_lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        frame_lab = cv2.merge((l, a, b))
        frame = cv2.cvtColor(frame_lab, cv2.COLOR_LAB2BGR)

        # Downscale frame
        scale = 0.3  # 30% for speed
        frame_small = cv2.resize(
            frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
        orig_h, orig_w = frame.shape[:2]
        frame_id = self.frame_counter

        all_detections = []

        # Queue frame for detection
        if (
            self.frame_counter % self.DETECTION_INTERVAL == 0
            and self.frame_queue.empty()
        ):
            self.frame_queue.put((frame_small, frame_id), block=False)

        # Collect HOG detections
        if self.use_hog:
            while not self.hog_result_queue.empty():
                detections, det_frame_id = self.hog_result_queue.get_nowait()
                if det_frame_id == frame_id:
                    detections = [
                        (
                            int(x / scale),
                            int(y / scale),
                            int(w / scale),
                            int(h / scale),
                            conf,
                        )
                        for x, y, w, h, conf in detections
                    ]
                    all_detections.extend(detections)

        # Collect motion detections
        if self.use_motion:
            while not self.motion_result_queue.empty():
                detections, det_frame_id = self.motion_result_queue.get_nowait()
                if det_frame_id == frame_id:
                    detections = [
                        (
                            int(x / scale),
                            int(y / scale),
                            int(w / scale),
                            int(h / scale),
                            conf,
                        )
                        for x, y, w, h, conf in detections
                    ]
                    all_detections.extend(detections)

        # Refine with skin detection
        if self.use_skin and all_detections:
            all_detections = self._refine_with_skin(frame_small, all_detections, scale)

        # Weighted fusion
        final_detections = self._weighted_fusion(all_detections)

        # Apply NMS and temporal filtering
        final_detections = self._apply_nms(final_detections, overlap_threshold=0.5)
        final_detections = self._apply_temporal_filtering(final_detections)

        # Log performance
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
        """Detect humans using HOG+SVM."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        boxes, weights = self.hog.detectMultiScale(
            gray,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.03,  # Slightly tighter scale for stricter detection
            useMeanshiftGrouping=False,
        )
        detections = [
            (x, y, w, h, float(weight))
            for (x, y, w, h), weight in zip(boxes, weights)
            if weight > self.confidence_threshold
        ]
        return detections

    def _detect_with_motion(self, frame):
        """Detect humans using motion detection with shape analysis."""
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        fg_mask = self.bg_subtractor.apply(blurred)
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        detections = []

        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(h) / w

            # Shape analysis: Check for human-like contours
            if 1.2 < aspect_ratio < 3.0 and h > 60:
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) > 5:  # Human contours typically have complex shapes
                    confidence = min(cv2.contourArea(contour) / 5000, 0.9)
                    detections.append((x, y, w, h, confidence))

        return detections

    def _refine_with_skin(self, frame, detections, scale):
        """Refine detections using skin color."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Dynamic HSV range based on scene statistics
        h_mean = np.mean(hsv[:, :, 0])
        lower_skin = np.array([max(0, h_mean - 10), 20, 70], dtype=np.uint8)
        upper_skin = np.array([min(20, h_mean + 10), 255, 255], dtype=np.uint8)

        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        kernel = np.ones((5, 5), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.dilate(skin_mask, kernel, iterations=1)

        refined_detections = []
        for x, y, w, h, conf in detections:
            # Scale to small frame
            x_small, y_small = int(x * scale), int(y * scale)
            w_small, h_small = int(w * scale), int(h * scale)

            # Check skin presence in ROI
            roi = skin_mask[y_small : y_small + h_small, x_small : x_small + w_small]
            skin_ratio = (
                cv2.countNonZero(roi) / (w_small * h_small)
                if w_small * h_small > 0
                else 0
            )

            # Boost confidence if skin is present
            if skin_ratio > 0.1:  # At least 10% skin pixels
                conf = min(conf * 1.2, 1.0)
            refined_detections.append((x, y, w, h, conf))

        return refined_detections

    def _weighted_fusion(self, detections):
        """Fuse detections with weights: HOG (0.6), Motion (0.3), Skin (0.1)."""
        if not detections:
            return []

        # Group detections by proximity
        grouped = []
        processed = set()

        for i, det_i in enumerate(detections):
            if i in processed:
                continue

            x_i, y_i, w_i, h_i, conf_i = det_i
            center_i = (x_i + w_i // 2, y_i + h_i // 2)
            group = [det_i]
            group_idxs = [i]

            for j, det_j in enumerate(detections[i + 1 :], start=i + 1):
                if j in processed:
                    continue

                x_j, y_j, w_j, h_j, conf_j = det_j
                center_j = (x_j + w_j // 2, y_j + h_j // 2)
                dist = np.sqrt(
                    (center_i[0] - center_j[0]) ** 2 + (center_i[1] - center_j[1]) ** 2
                )

                if dist < (w_i + w_j) / 3:
                    group.append(det_j)
                    group_idxs.append(j)
                    processed.add(j)

            processed.add(i)
            grouped.append(group)

        # Compute weighted average for each group
        fused_detections = []
        for group in grouped:
            if not group:
                continue

            weights = [
                0.6 if conf > 0.7 else 0.3 for _, _, _, _, conf in group
            ]  # HOG/Motion weights
            if self.use_skin:
                weights = [
                    w if conf > 0.7 else 0.1
                    for w, (_, _, _, _, conf) in zip(weights, group)
                ]  # Skin weight

            total_weight = sum(weights)
            if total_weight == 0:
                continue

            avg_x = sum(det[0] * w for det, w in zip(group, weights)) / total_weight
            avg_y = sum(det[1] * w for det, w in zip(group, weights)) / total_weight
            avg_w = sum(det[2] * w for det, w in zip(group, weights)) / total_weight
            avg_h = sum(det[3] * w for det, w in zip(group, weights)) / total_weight
            avg_conf = sum(det[4] * w for det, w in zip(group, weights)) / total_weight

            fused_detections.append(
                (int(avg_x), int(avg_y), int(avg_w), int(avg_h), float(avg_conf))
            )

        return fused_detections

    def _apply_nms(self, detections, overlap_threshold=0.5):
        """Apply non-maximum suppression."""
        if not detections:
            return []

        boxes = np.array(
            [[x, y, x + w, y + h, conf] for x, y, w, h, conf in detections]
        )
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        scores = boxes[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= overlap_threshold)[0]
            order = order[inds + 1]

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
        """Enhanced temporal filtering with weighted averaging."""
        self.detection_history.append(detections)
        if len(self.detection_history) < 3:
            return detections

        all_recent = []
        for i, frame_dets in enumerate(self.detection_history):
            weight = 1.0 - (
                i / len(self.detection_history)
            )  # Recent frames have higher weight
            for det in frame_dets:
                all_recent.append(det + (weight,))

        merged_detections = []
        processed = set()

        for i, (x_i, y_i, w_i, h_i, conf_i, weight_i) in enumerate(all_recent):
            if i in processed:
                continue

            center_i = (x_i + w_i // 2, y_i + h_i // 2)
            similar = [(x_i, y_i, w_i, h_i, conf_i, weight_i)]
            similar_idxs = [i]

            for j, (x_j, y_j, w_j, h_j, conf_j, weight_j) in enumerate(
                all_recent[i + 1 :], start=i + 1
            ):
                if j in processed:
                    continue

                center_j = (x_j + w_j // 2, y_j + h_j // 2)
                dist = np.sqrt(
                    (center_i[0] - center_j[0]) ** 2 + (center_i[1] - center_j[1]) ** 2
                )

                if dist < (w_i + w_j) / 3:
                    similar.append((x_j, y_j, w_j, h_j, conf_j, weight_j))
                    similar_idxs.append(j)
                    processed.add(j)

            if similar:
                total_weight = sum(w for _, _, _, _, _, w in similar)
                avg_x = sum(x * w for x, _, _, _, _, w in similar) / total_weight
                avg_y = sum(y * w for _, y, _, _, _, w in similar) / total_weight
                avg_w = sum(w * w for _, _, w, _, _, w in similar) / total_weight
                avg_h = sum(h * w for _, _, _, h, _, w in similar) / total_weight
                avg_conf = (
                    sum(conf * w for _, _, _, _, conf, w in similar) / total_weight
                )

                merged_detections.append(
                    (int(avg_x), int(avg_y), int(avg_w), int(avg_h), float(avg_conf))
                )

        return merged_detections

    def get_performance_metrics(self):
        """Return performance metrics."""
        avg_fps = (
            sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        )
        return {"fps": avg_fps}

    def visualize_detections(self, frame, detections):
        """Draw detection boxes on the frame."""
        result = frame.copy()
        for x, y, w, h, confidence in detections:
            color = (
                (0, 255, 0)
                if confidence > 0.7
                else (0, 255, 255) if confidence > 0.5 else (0, 0, 255)
            )
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            text = f"{confidence:.2f}"
            cv2.putText(
                result, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

        metrics = self.get_performance_metrics()
        fps_text = f"FPS: {metrics['fps']:.1f}"
        cv2.putText(
            result, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

        return result

    def cleanup(self):
        """Clean up threads."""
        self.stop_event.set()
        if self.use_hog and hasattr(self, "hog_thread"):
            self.hog_thread.join()
        if self.use_motion and hasattr(self, "motion_thread"):
            self.motion_thread.join()


def main():
    parser = argparse.ArgumentParser(
        description="Robust Human Detection using Image Processing"
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
        "--skin", action="store_true", help="Enable skin color refinement"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.6, help="Detection confidence threshold"
    )

    args = parser.parse_args()

    detector = HumanDetector(
        use_hog=not args.no_hog,
        use_motion=not args.no_motion,
        use_skin=args.skin,
        confidence_threshold=args.threshold,
    )
    detector.start_workers()

    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not cap.isOpened():
        logging.error(f"Could not open video source {args.source}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (frame_width, frame_height))

    logging.info("Human Detector started")
    logging.info(f"HOG detection: {'Enabled' if not args.no_hog else 'Disabled'}")
    logging.info(f"Motion detection: {'Enabled' if not args.no_motion else 'Disabled'}")
    logging.info(f"Skin refinement: {'Enabled' if args.skin else 'Disabled'}")
    logging.info(f"Confidence threshold: {args.threshold}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.info("End of video stream")
                break

            detections = detector.detect_humans(frame)
            result = detector.visualize_detections(frame, detections)

            if args.display:
                cv2.imshow("Human Detection", result)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if writer:
                writer.write(result)

            if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 30 == 0:
                metrics = detector.get_performance_metrics()
                logging.info(f"FPS: {metrics['fps']:.1f}")

    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        detector.cleanup()
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        logging.info("Human Detector stopped")


if __name__ == "__main__":
    main()
