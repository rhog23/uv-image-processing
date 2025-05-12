import logging
from .config import (
    CASCADE_FRONTAL_FACE_PATH,
    CASCADE_PROFILE_FACE_PATH,
)  # Updated config var names
from .face_detector import FaceDetector  # Updated class name
from .video_processor import VideoProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def main():
    """
    Main function to initialize and run the human face detection application.
    """
    try:
        logger.info("Initializing Human Face Detector application...")
        # Pass both cascade paths to the detector
        detector = FaceDetector(
            frontal_cascade_path=CASCADE_FRONTAL_FACE_PATH,
            profile_cascade_path=CASCADE_PROFILE_FACE_PATH,
        )
        processor = VideoProcessor(detector=detector)
        processor.run()
    except IOError as e:
        logger.error(f"Failed to initialize components: {e}")
        logger.error(
            "Please ensure cascade files are correctly placed and readable, and the camera is available."
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        logger.info("Application finished.")


if __name__ == "__main__":
    # To run from project root: python -m src.main
    main()
