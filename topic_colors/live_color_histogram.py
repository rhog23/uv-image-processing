import cv2
import numpy as np


def draw_histogram(img):
    h, w, c = img.shape
    hist_img = np.zeros((300, 512, 3), dtype=np.uint8)
    cv2.COLOR_

    colors = ("b", "g", "r")  # OpenCV uses BGR
    for i, col in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        cv2.normalize(hist, hist, 0, hist_img.shape[0], cv2.NORM_MINMAX)
        hist = hist.flatten()

        for x in range(1, 256):
            cv2.line(
                hist_img,
                (2 * (x - 1), hist_img.shape[0] - int(hist[x - 1])),
                (2 * (x), hist_img.shape[0] - int(hist[x])),
                (
                    255 if col == "b" else 0,
                    255 if col == "g" else 0,
                    255 if col == "r" else 0,
                ),
                thickness=2,
            )

    return hist_img


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    print("Press ESC to exit...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))

        hist_img = draw_histogram(frame)

        cv2.imshow("Webcam Frame", frame)
        cv2.imshow("Live Color Histogram", hist_img)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
