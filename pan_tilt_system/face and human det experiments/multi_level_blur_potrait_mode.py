import cv2
import numpy as np


def apply_blur_level(img, mask, ksize):
    blurred = cv2.GaussianBlur(img, (ksize, ksize), 0)
    return np.where(mask[..., None], blurred, img)


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

        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2

        y_indices, x_indices = np.indices((h, w))
        distance = np.sqrt((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2)
        distance = distance / np.max(distance)

        level1 = (distance > 0.3) & (distance <= 0.5)
        level2 = (distance > 0.5) & (distance <= 0.7)
        level3 = distance > 0.7

        result = frame.copy()
        result = apply_blur_level(result, level1, 15)
        result = apply_blur_level(result, level2, 25)
        result = apply_blur_level(result, level3, 45)

        cv2.imshow("Multi-Level Blur Portrait Mode", result)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
