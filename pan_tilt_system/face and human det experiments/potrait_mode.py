import cv2
import numpy as np


def apply_portrait_mode(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for consistency

    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    h, w = img.shape[:2]
    rect = (int(w * 0.2), int(h * 0.2), int(w * 0.6), int(h * 0.6))  # Focus center area
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

    blurred = cv2.GaussianBlur(img, (35, 35), 0)

    portrait = img * mask2[:, :, np.newaxis] + blurred * (1 - mask2[:, :, np.newaxis])

    return cv2.cvtColor(portrait, cv2.COLOR_RGB2BGR)


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

        frame = cv2.resize(frame, (320, 240))

        # Apply portrait mode effect
        portrait_frame = apply_portrait_mode(frame)

        cv2.imshow("Live Portrait Mode (No Deep Learning)", portrait_frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
