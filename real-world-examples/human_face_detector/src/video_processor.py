import cv2
import time
import logging
from . import config  # Ensure this works
from .face_detector import FaceDetector  # Updated class name
from .drawing_utils import draw_detections

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Handles video capture, processing frames for human face detection,
    and displaying the output.
    """

    def __init__(self, detector: FaceDetector):  # Updated type hint
        """
        Initializes the VideoProcessor.

        Args:
            detector: An instance of FaceDetector.
        """
        self.detector = detector
        self.cap = None
        self.frame_count = 0
        self.start_time = time.time()

    def _open_capture(self):
        """Opens the video capture source."""
        capture_source = config.CAMERA_INDEX
        api_preference = (
            cv2.CAP_DSHOW
            if config.USE_DSHOW and isinstance(capture_source, int)
            else cv2.CAP_ANY
        )

        self.cap = cv2.VideoCapture(capture_source, api_preference)
        if not self.cap.isOpened():
            error_msg = f"Cannot open video source: {capture_source}"
            logger.error(error_msg)
            raise IOError(error_msg)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        logger.info(
            f"Video source opened: {capture_source} with DSHOW: {config.USE_DSHOW if isinstance(capture_source, int) else 'N/A'}"
        )
        logger.info(
            f"Attempting to set resolution to {config.FRAME_WIDTH}x{config.FRAME_HEIGHT}"
        )
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        logger.info(f"Actual resolution: {actual_width}x{actual_height}")

    def run(self) -> None:
        """Starts the video processing loop."""
        self._open_capture()
        logger.info("Starting video processing loop. Press ESC to exit.")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning(
                        "Failed to grab frame. End of video or camera error."
                    )
                    if isinstance(
                        config.CAMERA_INDEX, str
                    ):  # If it's a file, break at the end
                        break
                    time.sleep(0.1)  # Avoid busy-waiting if camera momentarily fails
                    continue

                self.frame_count += 1

                detections = self.detector.detect_faces(
                    frame,  # Image first
                    scale_factor=config.DETECTOR_SCALE_FACTOR,
                    min_neighbors_frontal=config.DETECTOR_MIN_NEIGHBORS_FRONTAL,
                    min_neighbors_profile=config.DETECTOR_MIN_NEIGHBORS_PROFILE,
                    min_size=config.DETECTOR_MIN_SIZE,
                    max_size=config.DETECTOR_MAX_SIZE,
                )

                draw_detections(frame, detections)
                self._display_fps(frame)

                cv2.imshow("Human Face Detection", frame)  # Updated window title

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    logger.info("ESC key pressed. Exiting.")
                    break
        finally:
            self._release_capture()

    def _display_fps(self, frame: cv2.typing.MatLike) -> None:
        """Calculates and displays FPS on the frame."""
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            fps = self.frame_count / elapsed_time
            cv2.putText(
                frame,
                f"FPS: {fps:.2f}",
                (10, 30),
                config.TEXT_FONT,
                0.7,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"FPS: {fps:.2f}",
                (10, 30),
                config.TEXT_FONT,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    def _release_capture(self) -> None:
        """Releases the video capture and destroys OpenCV windows."""
        if self.cap:
            self.cap.release()
            logger.info("Video capture released.")
        cv2.destroyAllWindows()
        logger.info("OpenCV windows destroyed.")
