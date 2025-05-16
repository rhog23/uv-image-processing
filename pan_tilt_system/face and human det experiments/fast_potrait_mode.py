import cv2
import numpy as np


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50)

    print("Press ESC to exit...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))

        mask = bg_subtractor.apply(frame)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask_inv = cv2.bitwise_not(mask)

        blurred = cv2.GaussianBlur(frame, (35, 35), 0)

        fg = cv2.bitwise_and(frame, frame, mask=mask)
        bg = cv2.bitwise_and(blurred, blurred, mask=mask_inv)

        portrait = cv2.add(fg, bg)

        cv2.imshow("Fast Portrait Mode", portrait)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
