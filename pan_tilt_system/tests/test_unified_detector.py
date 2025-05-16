# smart_pan_tilt_system/tests/test_unified_detector.py

"""
Test script for a unified face detector combining frontal and profile Haar cascades.

This script will:
1. Load frontal and profile face Haar cascades.
2. Detect faces using:
    a. Frontal cascade on the original image.
    b. Profile cascade on the original image (catches one side).
    c. Profile cascade on a horizontally flipped image (catches the other side).
3. Combine all detections.
4. Apply Non-Maximum Suppression (NMS) to refine detections.
5. Visualize intermediate and final results.
"""
import cv2
import numpy as np
import os, platform
import time  # For FPS calculation

# ===================== CONFIGURATION =====================
CAMERA_ID: int = 0  # Try 0, 1, 2, etc.
FRAME_WIDTH: int = 640  # Desired width
FRAME_HEIGHT: int = 480  # Desired height

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

    # Fallback: Check if it's in the project root (assuming tests/ is one level down)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    project_path = os.path.join(project_root, filename)
    if os.path.exists(project_path):
        print(f"Using Haar Cascade from project root: {project_path}")
        return project_path

    # Fallback: Check current directory (if script is run from where cascades are)
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
    exit()  # Exit if cascades are not found

# Detection Parameters
# These can be tuned for performance vs. accuracy
FRONTAL_SCALE_FACTOR: float = 1.2  # Increased for potentially faster, less sensitive
FRONTAL_MIN_NEIGHBORS: int = 4
PROFILE_SCALE_FACTOR: float = 1.2
PROFILE_MIN_NEIGHBORS: int = 4  # Profile might need fewer neighbors
MIN_FACE_SIZE: tuple[int, int] = (40, 40)  # Slightly smaller min size

# NMS Parameters
NMS_OVERLAP_THRESHOLD: float = 0.3  # Lower threshold means more merging

# Visualization
DISPLAY_WINDOW_TITLE: str = "Unified Face Detector Test (Frontal + Profile)"
# ===================== INITIALIZATION =====================


def initialize_camera(camera_id: int, width: int, height: int) -> cv2.VideoCapture:
    """Initializes and returns the camera capture object."""
    cap = cv2.VideoCapture(
        camera_id, cv2.CAP_DSHOW if platform.system() == "Windows" else None
    )
    if not cap.isOpened():
        raise IOError(f"Cannot open webcam ID {camera_id}.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera requested: {width}x{height}, Actual: {actual_width}x{actual_height}")

    # Update global frame dimensions if camera provides different ones
    global FRAME_WIDTH, FRAME_HEIGHT
    FRAME_WIDTH = actual_width
    FRAME_HEIGHT = actual_height
    return cap


def load_cascade(cascade_path: str) -> cv2.CascadeClassifier:
    """Loads a Haar Cascade classifier."""
    if not os.path.exists(cascade_path):
        raise IOError(f"Haar Cascade file not found: {cascade_path}")
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise IOError(f"Failed to load Haar Cascade from: {cascade_path}")
    print(f"Successfully loaded {os.path.basename(cascade_path)}")
    return cascade


def non_max_suppression(
    boxes: np.ndarray, scores: np.ndarray, overlap_thresh: float
) -> list:
    """
    Applies Non-Maximum Suppression to filter overlapping bounding boxes.

    Args:
        boxes (np.ndarray): Array of bounding boxes, shape (N, 4) where each row is (x, y, w, h).
        scores (np.ndarray): Array of confidence scores for each box, shape (N,).
        overlap_thresh (float): The IoU threshold for suppressing boxes.

    Returns:
        list: A list of selected bounding boxes after NMS, as (x, y, w, h).
    """
    if len(boxes) == 0:
        return []

    # Convert (x, y, w, h) to (x1, y1, x2, y2)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    indices = np.argsort(scores)[::-1]  # Sort by score in descending order

    keep_indices = []
    while len(indices) > 0:
        i = indices[0]
        keep_indices.append(i)

        # Calculate IoU with remaining boxes
        # xx1, yy1, xx2, yy2 are coordinates of the intersection rectangle
        xx1 = np.maximum(x1[i], x1[indices[1:]])
        yy1 = np.maximum(y1[i], y1[indices[1:]])
        xx2 = np.minimum(x2[i], x2[indices[1:]])
        yy2 = np.minimum(y2[i], y2[indices[1:]])

        w_intersect = np.maximum(0, xx2 - xx1 + 1)
        h_intersect = np.maximum(0, yy2 - yy1 + 1)
        intersection_area = w_intersect * h_intersect

        iou = intersection_area / (areas[i] + areas[indices[1:]] - intersection_area)

        # Keep boxes with IoU less than the threshold
        indices_to_keep_mask = iou <= overlap_thresh
        indices = indices[1:][indices_to_keep_mask]

    return boxes[keep_indices].tolist()


def main_test_loop():
    """Main loop for capturing, processing, and displaying results."""
    try:
        cap = initialize_camera(CAMERA_ID, FRAME_WIDTH, FRAME_HEIGHT)
        frontal_face_cascade = load_cascade(FRONTAL_FACE_CASCADE_PATH)
        profile_face_cascade = load_cascade(PROFILE_FACE_CASCADE_PATH)
    except IOError as e:
        print(f"Initialization Error: {e}")
        return

    print("\nStarting Unified Face Detector Test...")
    print("Press 'q' in the OpenCV window to quit.")

    prev_time = time.time()
    frame_count = 0

    while True:
        ret, frame_original = cap.read()
        if not ret or frame_original is None:
            print("Error: Could not read frame. Exiting.")
            break

        # Ensure frame is of the expected size
        frame_original = cv2.resize(frame_original, (FRAME_WIDTH, FRAME_HEIGHT))

        # --- Image Processing ---
        gray = cv2.cvtColor(frame_original, cv2.COLOR_BGR2GRAY)
        gray_flipped = cv2.flip(gray, 1)  # Flip horizontally

        all_detections_boxes = []
        all_detections_scores = []  # Using levelWeights as scores

        # 1. Frontal Face Detection
        # detectMultiScale3 returns (rects, rejectLevels, levelWeights)
        # levelWeights can be used as confidence scores
        frontal_faces_rects, _, frontal_scores = frontal_face_cascade.detectMultiScale3(
            gray,
            scaleFactor=FRONTAL_SCALE_FACTOR,
            minNeighbors=FRONTAL_MIN_NEIGHBORS,
            minSize=MIN_FACE_SIZE,
            outputRejectLevels=True,  # Important to get scores
        )
        if len(frontal_faces_rects) > 0:
            all_detections_boxes.extend(frontal_faces_rects)
            all_detections_scores.extend(
                frontal_scores.flatten()
            )  # Ensure scores is 1D

        # 2. Profile Face Detection (Original Image - e.g., detects left profiles)
        profile_faces_rects, _, profile_scores = profile_face_cascade.detectMultiScale3(
            gray,
            scaleFactor=PROFILE_SCALE_FACTOR,
            minNeighbors=PROFILE_MIN_NEIGHBORS,
            minSize=MIN_FACE_SIZE,
            outputRejectLevels=True,
        )
        if len(profile_faces_rects) > 0:
            all_detections_boxes.extend(profile_faces_rects)
            all_detections_scores.extend(profile_scores.flatten())

        # 3. Profile Face Detection (Flipped Image - e.g., detects right profiles)
        profile_faces_flipped_rects, _, profile_flipped_scores = (
            profile_face_cascade.detectMultiScale3(
                gray_flipped,
                scaleFactor=PROFILE_SCALE_FACTOR,
                minNeighbors=PROFILE_MIN_NEIGHBORS,
                minSize=MIN_FACE_SIZE,
                outputRejectLevels=True,
            )
        )
        if len(profile_faces_flipped_rects) > 0:
            # Convert coordinates back to original image space
            for i in range(len(profile_faces_flipped_rects)):
                x, y, w, h = profile_faces_flipped_rects[i]
                # Original x_flipped = FRAME_WIDTH - x_original - w
                # So, x_original = FRAME_WIDTH - x_flipped - w
                profile_faces_flipped_rects[i][0] = FRAME_WIDTH - x - w
            all_detections_boxes.extend(profile_faces_flipped_rects)
            all_detections_scores.extend(profile_flipped_scores.flatten())

        # --- Non-Maximum Suppression ---
        final_face_boxes = []
        if len(all_detections_boxes) > 0:
            np_boxes = np.array(all_detections_boxes)
            np_scores = np.array(all_detections_scores)
            final_face_boxes = non_max_suppression(
                np_boxes, np_scores, NMS_OVERLAP_THRESHOLD
            )

        # --- Visualization ---
        # Create a display frame
        display_frame = frame_original.copy()

        # Draw raw detections (before NMS) for comparison - optional
        # for i, (x, y, w, h) in enumerate(all_detections_boxes):
        #     # Different colors for different detectors could be used here
        #     cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 1) # Red for raw
        #     # cv2.putText(display_frame, f"{all_detections_scores[i]:.2f}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255),1)

        # Draw final (NMS) detections
        for x, y, w, h in final_face_boxes:
            cv2.rectangle(
                display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2
            )  # Green for final
            cv2.putText(
                display_frame,
                "Face",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        # Calculate FPS
        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - prev_time
        if elapsed_time > 1.0:  # Update FPS every second
            fps = frame_count / elapsed_time
            prev_time = current_time
            frame_count = 0
            cv2.putText(
                display_frame,
                f"FPS: {fps:.2f}",
                (FRAME_WIDTH - 100, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )
        else:  # Display last known FPS
            cv2.putText(
                display_frame,
                f"FPS: calculating...",
                (FRAME_WIDTH - 150, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )

        cv2.putText(
            display_frame,
            f"Raw Detections: {len(all_detections_boxes)}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            display_frame,
            f"Final Detections (NMS): {len(final_face_boxes)}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        # For more detailed visualization, you could create separate windows or a larger canvas
        # showing intermediate steps (frontal only, profile only, etc.)
        # For now, this shows raw count vs NMS count on the main display.

        cv2.imshow(DISPLAY_WINDOW_TITLE, display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Quit signal received.")
            break

        # time.sleep(0.01) # Optional small delay

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Camera and resources released. Test finished.")


if __name__ == "__main__":
    main_test_loop()
