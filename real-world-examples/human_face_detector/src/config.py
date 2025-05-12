"""Camera configurations"""

import cv2
import platform

# Camera configuration
CAMERA_INDEX = 0  # 0 for default webcam
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
USE_DSHOW = (
    True if platform.system() == "Windows" else False
)  # use cv2.CAP_DSHOW for better Windows compatibility (if applicable)

# Cascade classifier configuration
# for human faces; Other options see: https://github.com/opencv/opencv/tree/4.x/data/haarcascades
# CASCADE_FRONTAL_FACE_PATH = f"{cv2.data.haarcascades}haarcascade_frontalface_alt.xml"
CASCADE_FRONTAL_FACE_PATH = f"./cascades/lbpcascade_frontalface_improved.xml"

# for profile faces (captures both left and right sides of a face)
# CASCADE_PROFILE_FACE_PATH = f"{cv2.data.haarcascades}haarcascade_profileface.xml"  # set to none if you don't require profile detector and comment this code
# CASCADE_PROFILE_FACE_PATH = None
CASCADE_PROFILE_FACE_PATH = f"./cascades/lbpcascade_profileface.xml"

# Detection parameters
# Shared parameters for both frontal and profile detectors
# These may need different tuning for frontal vs. profile for optimal results.
# Consider making them separate if fine-grained control is needed.

# scaleFactor: How much the image size is reduced at each image scale.
DETECTOR_SCALE_FACTOR = 1.1

# minNeighbors: How many neighbors each candidate rectangle should have to retain it.
# Higher values result in fewer false positives but may also lead to missing positives.
DETECTOR_MIN_NEIGHBORS_FRONTAL = 5  # can be tuned
DETECTOR_MIN_NEIGHBORS_PROFILE = (
    6  # profile faces can be trickier, might need more neighbors
)

# minSize: Minimum possible object size. If an object is smaller, it will be ignored.
DETECTOR_MIN_SIZE = (20, 20)  # minimum face size (width, height)

# maxSize: Maximum possible object size. If an object is larger, it will be ignored.
DETECTOR_MAX_SIZE = (300, 300)  # maximum face size

# Drawing configuration
BOX_COLOR = (0, 255, 0)  # Green for frontal
BOX_COLOR_PROFILE = (0, 165, 255)  # Orange for profile
BOX_THICKNESS = 2
CENTROID_COLOR = (0, 0, 255)  # Red
CENTROID_RADIUS = 5
TEXT_COLOR = (255, 255, 255)  # White
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_FONT_SCALE = 0.5
TEXT_THICKNESS = 1

# Histogram equalization
HIST_EQUALIZATION = True  # enable or disable histogram equalization
