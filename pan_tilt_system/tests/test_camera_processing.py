# smart_pan_tilt_system/tests/test_camera_processing.py

"""
Test script for camera functionality and visualization of image processing steps
leading to face detection.

This script will display:
1. Original Camera Feed
2. Grayscale Conversion
3. Histogram Equalized Grayscale Image
4. Original Feed with Detected Faces
"""
import cv2
import numpy as np
import os, platform

# ===================== CONFIGURATION =====================
# These can be imported from a central config.py if this test is run within the project context
# For a standalone test, define them here.
CAMERA_ID: int = 0  # Try 0, 1, 2, etc.
FRAME_WIDTH: int = 640
FRAME_HEIGHT: int = 480

# Attempt to locate the Haar Cascade file
# This assumes the test script might be run from the 'tests' directory or the project root
HAAR_CASCADE_FILENAME: str = "haarcascade_frontalface_default.xml"
HAAR_CASCADE_PATH_CV2_DATA: str = os.path.join(
    cv2.data.haarcascades, HAAR_CASCADE_FILENAME
)

# Fallback: if you have the XML file in the project root or a specific 'models' folder
# Adjust this path if necessary. For example, if it's in the project root:
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) # This goes one level up from 'tests'
# HAAR_CASCADE_PATH_PROJECT: str = os.path.join(PROJECT_ROOT, HAAR_CASCADE_FILENAME)

# Select the path for the Haar Cascade
if os.path.exists(HAAR_CASCADE_PATH_CV2_DATA):
    SELECTED_HAAR_CASCADE_PATH = HAAR_CASCADE_PATH_CV2_DATA
    print(f"Using Haar Cascade from cv2.data: {SELECTED_HAAR_CASCADE_PATH}")
# elif os.path.exists(HAAR_CASCADE_PATH_PROJECT):
#     SELECTED_HAAR_CASCADE_PATH = HAAR_CASCADE_PATH_PROJECT
#     print(f"Using Haar Cascade from project root: {SELECTED_HAAR_CASCADE_PATH}")
else:
    print(f"Error: Haar Cascade file '{HAAR_CASCADE_FILENAME}' not found.")
    print(f"Looked in cv2.data.haarcascades: {HAAR_CASCADE_PATH_CV2_DATA}")
    # print(f"Looked in project root: {HAAR_CASCADE_PATH_PROJECT}")
    print("Please ensure the Haar Cascade XML file is available or update the path.")
    SELECTED_HAAR_CASCADE_PATH = (
        None  # Will cause an error later, but good for diagnosis
    )

# Face Detection Parameters
SCALE_FACTOR: float = 1.1
MIN_NEIGHBORS: int = 5
MIN_FACE_SIZE: tuple[int, int] = (50, 50)
# ===================== INITIALIZATION =====================


def initialize_camera(camera_id: int, width: int, height: int) -> cv2.VideoCapture:
    """
    Initializes and returns the camera capture object.

    Args:
        camera_id (int): The ID of the camera.
        width (int): Desired frame width.
        height (int): Desired frame height.

    Returns:
        cv2.VideoCapture: The initialized camera capture object.

    Raises:
        IOError: If the camera cannot be opened.
    """
    cap = cv2.VideoCapture(
        camera_id, cv2.CAP_DSHOW if platform.system() == "Windows" else None
    )
    if not cap.isOpened():
        raise IOError(
            f"Cannot open webcam ID {camera_id}. Please check connection and permissions."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera requested resolution: {width}x{height}")
    print(f"Camera actual resolution: {actual_width}x{actual_height}")

    # Update global frame dimensions if camera provides different ones
    global FRAME_WIDTH, FRAME_HEIGHT
    FRAME_WIDTH = actual_width
    FRAME_HEIGHT = actual_height

    return cap


def load_face_cascade(cascade_path: str) -> cv2.CascadeClassifier:
    """
    Loads the Haar Cascade classifier for face detection.

    Args:
        cascade_path (str): Path to the Haar Cascade XML file.

    Returns:
        cv2.CascadeClassifier: The loaded classifier.

    Raises:
        IOError: If the cascade file cannot be loaded.
    """
    if not cascade_path or not os.path.exists(cascade_path):
        raise IOError(f"Haar Cascade file not found at path: {cascade_path}")
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise IOError(
            f"Failed to load Haar Cascade from: {cascade_path}. File might be corrupt or invalid."
        )
    print(f"Haar Cascade '{os.path.basename(cascade_path)}' loaded successfully.")
    return face_cascade


def main_test_loop():
    """
    Main loop for capturing frames, processing, and displaying results.
    """
    try:
        cap = initialize_camera(CAMERA_ID, FRAME_WIDTH, FRAME_HEIGHT)
        if SELECTED_HAAR_CASCADE_PATH is None:
            return  # Exit if cascade path is not set
        face_cascade = load_face_cascade(SELECTED_HAAR_CASCADE_PATH)
    except IOError as e:
        print(f"Error during initialization: {e}")
        return

    print("\nStarting camera test and image processing visualization...")
    print("Press 'q' in the OpenCV window to quit.")

    while True:
        ret, frame_original = cap.read()
        if not ret or frame_original is None:
            print("Error: Could not read frame from camera. Exiting.")
            break

        # Ensure frame is of the expected size for consistent display
        # This might be needed if the camera doesn't strictly adhere to set dimensions
        frame_original = cv2.resize(frame_original, (FRAME_WIDTH, FRAME_HEIGHT))

        # 1. Grayscale Conversion
        frame_gray = cv2.cvtColor(frame_original, cv2.COLOR_BGR2GRAY)

        # 2. Histogram Equalization (on grayscale)
        frame_equalized = cv2.equalizeHist(frame_gray)

        # 3. Face Detection (on grayscale or equalized grayscale)
        # Using grayscale is standard. Equalized can sometimes help.
        # For consistency with typical usage, we'll detect on the plain grayscale image.
        faces = face_cascade.detectMultiScale(
            frame_gray,  # Input image for detection
            scaleFactor=SCALE_FACTOR,
            minNeighbors=MIN_NEIGHBORS,
            minSize=MIN_FACE_SIZE,
        )

        # 4. Draw rectangles on a copy of the original frame
        frame_with_detections = frame_original.copy()
        for x, y, w, h in faces:
            cv2.rectangle(frame_with_detections, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame_with_detections,
                "Face",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        # Prepare frames for display (convert single-channel to 3-channel BGR for stacking)
        frame_gray_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        frame_equalized_bgr = cv2.cvtColor(frame_equalized, cv2.COLOR_GRAY2BGR)

        # Add labels to each frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (255, 255, 255)  # White
        bg_color = (0, 0, 0)  # Black background for text
        thickness = 1

        cv2.putText(
            frame_original,
            "1. Original",
            (10, 20),
            font,
            font_scale,
            font_color,
            thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_gray_bgr,
            "2. Grayscale",
            (10, 20),
            font,
            font_scale,
            font_color,
            thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_equalized_bgr,
            "3. Equalized Grayscale",
            (10, 20),
            font,
            font_scale,
            font_color,
            thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_with_detections,
            f"4. Faces Detected: {len(faces)}",
            (10, 20),
            font,
            font_scale,
            font_color,
            thickness,
            cv2.LINE_AA,
        )

        # Combine frames into a single display window
        # Top row: Original, Grayscale
        # Bottom row: Equalized, Detections
        top_row = np.hstack((frame_original, frame_gray_bgr))
        bottom_row = np.hstack((frame_equalized_bgr, frame_with_detections))
        combined_display = np.vstack((top_row, bottom_row))

        # Or, if you prefer a single long horizontal strip:
        # combined_display = np.hstack((frame_original, frame_gray_bgr, frame_equalized_bgr, frame_with_detections))
        # Ensure the window is wide enough if using hstack for all four.

        cv2.imshow("Camera and Processing Test (Press 'q' to quit)", combined_display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Quit signal received.")
            break

        # Small delay to manage CPU usage, adjust as needed
        # time.sleep(0.01)

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Camera and resources released. Test finished.")


if __name__ == "__main__":
    # This check ensures the script can be run directly.
    # If importing as a module, these lines won't execute automatically.
    main_test_loop()
