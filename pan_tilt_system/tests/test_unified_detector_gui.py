"""
Test script for a unified face detector with a GUI to adjust parameters.

This script extends the unified face detector test by adding a tkinter GUI to:
1. Adjust frame size (maintaining aspect ratio).
2. Tune detection parameters (scale factors, min neighbors, min face size).
3. Tune NMS overlap threshold.
4. Visualize results with real-time parameter updates.
"""

import cv2
import numpy as np
import os
import platform
import time
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading

# ===================== CONFIGURATION =====================
CAMERA_ID: int = 0  # Try 0, 1, 2, etc.
DEFAULT_FRAME_WIDTH: int = 640
DEFAULT_FRAME_HEIGHT: int = 480
ASPECT_RATIO: float = DEFAULT_FRAME_WIDTH / DEFAULT_FRAME_HEIGHT

# Haar Cascade files
FRONTAL_FACE_CASCADE_FILENAME: str = "haarcascade_frontalface_default.xml"
PROFILE_FACE_CASCADE_FILENAME: str = "haarcascade_profileface.xml"


# Function to safely get cascade paths
def get_cascade_path(filename: str) -> str:
    """Attempts to find the cascade file, prioritizing cv2.data.haarcascades."""
    cv2_path = os.path.join(cv2.data.haarcascades, filename)
    if os.path.exists(cv2_path):
        print(f"Using Haar Cascade from cv2.data: {cv2_path}")
        return cv2_path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    project_path = os.path.join(project_root, filename)
    if os.path.exists(project_path):
        print(f"Using Haar Cascade from project root: {project_path}")
        return project_path
    current_dir_path = os.path.join(os.getcwd(), filename)
    if os.path.exists(current_dir_path):
        print(f"Using Haar Cascade from current directory: {current_dir_path}")
        return current_dir_path
    raise FileNotFoundError(
        f"Haar Cascade file '{filename}' not found in standard locations."
    )


try:
    FRONTAL_FACE_CASCADE_PATH: str = get_cascade_path(FRONTAL_FACE_CASCADE_FILENAME)
    PROFILE_FACE_CASCADE_PATH: str = get_cascade_path(PROFILE_FACE_CASCADE_FILENAME)
except FileNotFoundError as e:
    print(e)
    exit()


# ===================== CLASSES =====================
class UnifiedDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Unified Face Detector - Parameter Tuning")
        self.running = False
        self.cap = None
        self.frontal_cascade = None
        self.profile_cascade = None

        # Parameters
        self.frame_width = tk.DoubleVar(value=DEFAULT_FRAME_WIDTH)
        self.frontal_scale_factor = tk.DoubleVar(value=1.2)
        self.frontal_min_neighbors = tk.IntVar(value=4)
        self.profile_scale_factor = tk.DoubleVar(value=1.2)
        self.profile_min_neighbors = tk.IntVar(value=4)
        self.min_face_size = tk.IntVar(value=40)
        self.nms_overlap_threshold = tk.DoubleVar(value=0.3)

        # GUI Setup
        self.setup_gui()
        self.initialize_opencv()

    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Parameter controls with discrete steps
        ttk.Label(main_frame, text="Frame Width (px):").grid(
            row=0, column=0, sticky=tk.W
        )
        width_scale = ttk.Scale(
            main_frame,
            from_=320,
            to_=1280,
            orient=tk.HORIZONTAL,
            variable=self.frame_width,
            command=self.update_frame_size,
        )
        width_scale.configure(
            command=lambda x: self.frame_width.set(round(float(x) / 10) * 10)
        )  # Step by 10
        width_scale.grid(row=0, column=1, sticky=(tk.W, tk.E))
        ttk.Label(main_frame, textvariable=self.frame_width).grid(row=0, column=2)

        ttk.Label(main_frame, text="Frontal Scale Factor:").grid(
            row=1, column=0, sticky=tk.W
        )
        frontal_scale = ttk.Scale(
            main_frame,
            from_=1.1,
            to_=1.5,
            orient=tk.HORIZONTAL,
            variable=self.frontal_scale_factor,
        )
        frontal_scale.configure(
            command=lambda x: self.frontal_scale_factor.set(round(float(x) / 0.1) * 0.1)
        )  # Step by 0.1
        frontal_scale.grid(row=1, column=1, sticky=(tk.W, tk.E))
        ttk.Label(main_frame, textvariable=self.frontal_scale_factor).grid(
            row=1, column=2
        )

        ttk.Label(main_frame, text="Frontal Min Neighbors:").grid(
            row=2, column=0, sticky=tk.W
        )
        frontal_neighbors = ttk.Scale(
            main_frame,
            from_=1,
            to_=10,
            orient=tk.HORIZONTAL,
            variable=self.frontal_min_neighbors,
        )
        frontal_neighbors.configure(
            command=lambda x: self.frontal_min_neighbors.set(int(float(x)))
        )  # Step by 1
        frontal_neighbors.grid(row=2, column=1, sticky=(tk.W, tk.E))
        ttk.Label(main_frame, textvariable=self.frontal_min_neighbors).grid(
            row=2, column=2
        )

        ttk.Label(main_frame, text="Profile Scale Factor:").grid(
            row=3, column=0, sticky=tk.W
        )
        profile_scale = ttk.Scale(
            main_frame,
            from_=1.1,
            to_=1.5,
            orient=tk.HORIZONTAL,
            variable=self.profile_scale_factor,
        )
        profile_scale.configure(
            command=lambda x: self.profile_scale_factor.set(round(float(x) / 0.1) * 0.1)
        )  # Step by 0.1
        profile_scale.grid(row=3, column=1, sticky=(tk.W, tk.E))
        ttk.Label(main_frame, textvariable=self.profile_scale_factor).grid(
            row=3, column=2
        )

        ttk.Label(main_frame, text="Profile Min Neighbors:").grid(
            row=4, column=0, sticky=tk.W
        )
        profile_neighbors = ttk.Scale(
            main_frame,
            from_=1,
            to_=10,
            orient=tk.HORIZONTAL,
            variable=self.profile_min_neighbors,
        )
        profile_neighbors.configure(
            command=lambda x: self.profile_min_neighbors.set(int(float(x)))
        )  # Step by 1
        profile_neighbors.grid(row=4, column=1, sticky=(tk.W, tk.E))
        ttk.Label(main_frame, textvariable=self.profile_min_neighbors).grid(
            row=4, column=2
        )

        ttk.Label(main_frame, text="Min Face Size (px):").grid(
            row=5, column=0, sticky=tk.W
        )
        min_size_scale = ttk.Scale(
            main_frame,
            from_=20,
            to_=100,
            orient=tk.HORIZONTAL,
            variable=self.min_face_size,
        )
        min_size_scale.configure(
            command=lambda x: self.min_face_size.set(round(float(x) / 10) * 10)
        )  # Step by 10
        min_size_scale.grid(row=5, column=1, sticky=(tk.W, tk.E))
        ttk.Label(main_frame, textvariable=self.min_face_size).grid(row=5, column=2)

        ttk.Label(main_frame, text="NMS Overlap Threshold:").grid(
            row=6, column=0, sticky=tk.W
        )
        nms_scale = ttk.Scale(
            main_frame,
            from_=0.1,
            to_=0.5,
            orient=tk.HORIZONTAL,
            variable=self.nms_overlap_threshold,
        )
        nms_scale.configure(
            command=lambda x: self.nms_overlap_threshold.set(
                round(float(x) / 0.1) * 0.1
            )
        )  # Step by 0.1
        nms_scale.grid(row=6, column=1, sticky=(tk.W, tk.E))
        ttk.Label(main_frame, textvariable=self.nms_overlap_threshold).grid(
            row=6, column=2
        )

        # Video display
        self.canvas = tk.Canvas(
            main_frame, width=DEFAULT_FRAME_WIDTH, height=DEFAULT_FRAME_HEIGHT
        )
        self.canvas.grid(row=7, column=0, columnspan=3)

        # Start/Stop buttons
        ttk.Button(main_frame, text="Start", command=self.start_detection).grid(
            row=8, column=0
        )
        ttk.Button(main_frame, text="Stop", command=self.stop_detection).grid(
            row=8, column=1
        )

    def update_frame_size(self, *args):
        """Update frame height to maintain aspect ratio."""
        width = int(self.frame_width.get())
        height = int(width / ASPECT_RATIO)
        if self.cap:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.canvas.config(width=width, height=height)

    def initialize_opencv(self):
        """Initialize camera and cascades."""
        try:
            self.cap = cv2.VideoCapture(
                CAMERA_ID, cv2.CAP_DSHOW if platform.system() == "Windows" else None
            )
            if not self.cap.isOpened():
                raise IOError(f"Cannot open webcam ID {CAMERA_ID}.")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_FRAME_HEIGHT)

            self.frontal_cascade = cv2.CascadeClassifier(FRONTAL_FACE_CASCADE_PATH)
            self.profile_cascade = cv2.CascadeClassifier(PROFILE_FACE_CASCADE_PATH)
            if self.frontal_cascade.empty() or self.profile_cascade.empty():
                raise IOError("Failed to load Haar Cascades.")
            print("OpenCV initialized successfully.")
        except Exception as e:
            print(f"OpenCV Initialization Error: {e}")
            self.root.quit()

    def non_max_suppression(
        self, boxes: np.ndarray, scores: np.ndarray, overlap_thresh: float
    ) -> list:
        if len(boxes) == 0:
            return []
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        indices = np.argsort(scores)[::-1]
        keep_indices = []
        while len(indices) > 0:
            i = indices[0]
            keep_indices.append(i)
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
            indices_to_keep_mask = iou <= overlap_thresh
            indices = indices[1:][indices_to_keep_mask]
        return boxes[keep_indices].tolist()

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None
        frame = cv2.resize(
            frame,
            (int(self.frame_width.get()), int(self.frame_width.get() / ASPECT_RATIO)),
        )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_flipped = cv2.flip(gray, 1)
        all_detections_boxes = []
        all_detections_scores = []

        # Frontal detection
        frontal_rects, _, frontal_scores = self.frontal_cascade.detectMultiScale3(
            gray,
            scaleFactor=self.frontal_scale_factor.get(),
            minNeighbors=self.frontal_min_neighbors.get(),
            minSize=(self.min_face_size.get(), self.min_face_size.get()),
            outputRejectLevels=True,
        )
        if len(frontal_rects) > 0:
            all_detections_boxes.extend(frontal_rects)
            all_detections_scores.extend(frontal_scores.flatten())

        # Profile detection (original)
        profile_rects, _, profile_scores = self.profile_cascade.detectMultiScale3(
            gray,
            scaleFactor=self.profile_scale_factor.get(),
            minNeighbors=self.profile_min_neighbors.get(),
            minSize=(self.min_face_size.get(), self.min_face_size.get()),
            outputRejectLevels=True,
        )
        if len(profile_rects) > 0:
            all_detections_boxes.extend(profile_rects)
            all_detections_scores.extend(profile_scores.flatten())

        # Profile detection (flipped)
        profile_flipped_rects, _, profile_flipped_scores = (
            self.profile_cascade.detectMultiScale3(
                gray_flipped,
                scaleFactor=self.profile_scale_factor.get(),
                minNeighbors=self.profile_min_neighbors.get(),
                minSize=(self.min_face_size.get(), self.min_face_size.get()),
                outputRejectLevels=True,
            )
        )
        if len(profile_flipped_rects) > 0:
            for i in range(len(profile_flipped_rects)):
                x, y, w, h = profile_flipped_rects[i]
                profile_flipped_rects[i][0] = int(self.frame_width.get()) - x - w
            all_detections_boxes.extend(profile_flipped_rects)
            all_detections_scores.extend(profile_flipped_scores.flatten())

        # NMS
        final_face_boxes = []
        if len(all_detections_boxes) > 0:
            np_boxes = np.array(all_detections_boxes)
            np_scores = np.array(all_detections_scores)
            final_face_boxes = self.non_max_suppression(
                np_boxes, np_scores, self.nms_overlap_threshold.get()
            )

        # Draw detections
        for x, y, w, h in final_face_boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                "Face",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        # FPS and stats
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.prev_time
        if elapsed > 1.0:
            fps = self.frame_count / elapsed
            self.prev_time = current_time
            self.frame_count = 0
            cv2.putText(
                frame,
                f"FPS: {fps:.2f}",
                (int(self.frame_width.get()) - 100, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )
        cv2.putText(
            frame,
            f"Raw: {len(all_detections_boxes)}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Final: {len(final_face_boxes)}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        return frame

    def update_display(self):
        if not self.running:
            return
        frame = self.process_frame()
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.imgtk = imgtk
        self.root.after(10, self.update_display)

    def start_detection(self):
        if not self.running:
            self.running = True
            self.frame_count = 0
            self.prev_time = time.time()
            self.update_display()

    def stop_detection(self):
        self.running = False

    def cleanup(self):
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


def main():
    root = tk.Tk()
    app = UnifiedDetectorGUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: [app.cleanup(), root.destroy()])
    root.mainloop()


if __name__ == "__main__":
    main()
