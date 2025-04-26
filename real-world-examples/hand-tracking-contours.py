import cv2
import numpy as np


def hand_tracking_with_contours():
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
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)

        cr_min, cr_max = 133, 173
        cb_min, cb_max = 77, 127

        skin_mask = cv2.inRange(ycrcb, (0, cr_min, cb_min), (255, cr_max, cb_max))

        # Apply morphology to clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=2)

        contours, _ = cv2.findContours(
            skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Draw all contours
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:  # Filter small contours
                cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)

        cv2.imshow("Hand Tracking (Contours)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


hand_tracking_with_contours()
