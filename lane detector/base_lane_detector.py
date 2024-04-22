import cv2
import numpy as np


class VideoProcessor:
    def __init__(self, source: int | str = 0, model_path: str = "") -> None:
        if isinstance(source, int):
            self.cap = cv2.VideoCapture(source)
        else:
            self.cap = cv2.VideoCapture(source)

        self.model_path = model_path
        self.clahe_model = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def preprocess(self, frame: cv2.typing.MatLike) -> cv2.typing.MatLike:

        frame = cv2.GaussianBlur(frame, (5, 5), 0)  #  Blur the image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        norm_image = self.clahe_model.apply(gray)

        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            norm_image, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        edges = cv2.Canny(norm_image, 50, 150)

        opening = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            np.ones(
                (
                    7,
                    7,
                ),
                np.uint8,
            ),
        )

        combined = cv2.bitwise_and(thresh, edges)

        contours, _ = cv2.findContours(
            combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        frame_copy = frame.copy()
        cv2.drawContours(frame_copy, contours, -1, (0, 255, 0), 2)

        return combined

    def process(self):
        while self.cap.isOpened():
            success, frame = self.cap.read()

            if not success:
                break

            preprocessed_frame = self.preprocess(frame)
            cv2.imshow("frame", preprocessed_frame)

            if cv2.waitKey(1) == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    videoProcessor = VideoProcessor()
    videoProcessor.process()
