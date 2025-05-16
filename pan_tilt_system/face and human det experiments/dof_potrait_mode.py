import cv2
import numpy as np


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

        blurred = cv2.GaussianBlur(frame, (45, 45), 0)

        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2

        y_indices, x_indices = np.indices((h, w))
        distance = np.sqrt((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2)

        max_dist = np.max(distance)
        alpha = np.clip(distance / max_dist, 0, 1)
        alpha = np.expand_dims(alpha, axis=2)

        portrait = frame * (1 - alpha) + blurred * alpha
        portrait = portrait.astype(np.uint8)

        cv2.imshow("Depth of Field Portrait Mode", portrait)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
